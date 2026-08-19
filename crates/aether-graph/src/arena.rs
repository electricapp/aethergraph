//! Two-region slab arena for C-tree nodes with grace-period recycling.
//!
//! The backing buffer is split by kind: 64-byte chunk slots bump upward
//! from the low end, 16-byte interior-node slots bump downward from the
//! high end. Same-kind allocations pack densely (no mixed-kind padding
//! holes, chunk scans touch only chunk lines), and offsets are *slot
//! indices*, so a u32 index with bit 31 reserved as the leaf/interior tag
//! addresses far more than 2^31 bytes: capacity is bounded by the
//! interior-slot index range at 32 GiB.
//!
//! Retired slots (superseded by path copying) are recycled: the writer
//! stages retired indices, stamps each staged batch with a snapshot of the
//! striped reader gate, and once every reader that might have entered
//! before the stamp has exited, threads the slots onto intrusive free
//! lists (the link lives in the freed slot itself — recycling costs no
//! memory and no heap allocation). Allocation pops the free list before
//! bumping a cursor, so steady-state ingest reuses garbage instead of
//! growing until [`compact`](crate::DynamicGraph::compact).
//!
//! Mutation goes through [`ArenaWriter`], obtained once from
//! [`Arena::writer`] — the one `unsafe` point where the caller proves the
//! single-writer invariant; every allocation after that is safe code
//! borrowing the handle. Concurrent reads of previously-allocated nodes
//! are safe provided readers hold a [`ReadGuard`] for the duration of the
//! traversal — the guard is what makes slot reuse sound.

use crate::chunk::Chunk;
use crate::ctree::Interior;
use crate::pad::CachePadded;
use std::alloc::Layout;
use std::cell::{Cell, UnsafeCell};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering, fence};

/// Bytes per chunk slot (low region). Matches `Chunk`'s size/alignment.
pub const CHUNK_SLOT: usize = 64;
/// Bytes per interior-node slot (high region). Matches `Interior`'s size.
pub const INTERIOR_SLOT: usize = 16;

/// Free-list terminator / "no slot" sentinel inside the recycler.
const NO_SLOT: u32 = u32::MAX;

/// Reader-gate stripes. Sixteen padded counters keep concurrent readers
/// from serializing on one cache line while staying cheap to snapshot.
const GATE_STRIPES: usize = 16;

/// Retire-log capacity per class before the writer should flush it to the
/// ring. Also the pre-allocated capacity of ring batch buffers, so a flush
/// at this watermark never grows a vector.
pub(crate) const RETIRE_LOG_CAP: usize = 4096;

/// Pending (grace-awaiting) batches. With reads lasting microseconds the
/// grace period is effectively instant, so a short ring suffices; if every
/// slot is somehow still awaiting grace at flush time, the staged slots
/// are dropped (left as garbage for `compact`) rather than blocking.
const RING_SLOTS: usize = 4;

/// One stripe of the reader gate: monotonic entry/exit counters.
struct GateStripe {
    ingress: AtomicU64,
    egress: AtomicU64,
}

/// Striped ingress/egress reader gate.
///
/// A reader bumps `ingress` (SeqCst) on entry and `egress` (Release) on
/// exit. The writer, after unpublishing nodes, issues a SeqCst fence and
/// snapshots every stripe's `ingress`; once each stripe's `egress` reaches
/// its snapshot, every reader that could have observed the old nodes has
/// finished. The SeqCst entry RMW paired with the writer's fence closes
/// the store-buffer race: a reader whose entry the writer's snapshot
/// missed is ordered after the fence and therefore loads the new root.
struct ReadGate {
    stripes: [CachePadded<GateStripe>; GATE_STRIPES],
}

impl ReadGate {
    fn new() -> Self {
        Self {
            stripes: std::array::from_fn(|_| {
                CachePadded(GateStripe {
                    ingress: AtomicU64::new(0),
                    egress: AtomicU64::new(0),
                })
            }),
        }
    }

    fn snapshot(&self, out: &mut [u64; GATE_STRIPES]) {
        fence(Ordering::SeqCst);
        for (i, s) in self.stripes.iter().enumerate() {
            out[i] = s.0.ingress.load(Ordering::Relaxed);
        }
    }

    fn grace_passed(&self, snap: &[u64; GATE_STRIPES]) -> bool {
        self.stripes
            .iter()
            .zip(snap)
            .all(|(s, &want)| s.0.egress.load(Ordering::Acquire) >= want)
    }
}

/// Per-thread stripe assignment: round-robin at first use, cached in a
/// thread-local so gate entry is one cached load plus one striped RMW.
fn gate_stripe() -> usize {
    thread_local! {
        static STRIPE: Cell<usize> = const { Cell::new(usize::MAX) };
    }
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    STRIPE.with(|s| {
        let mut v = s.get();
        if v == usize::MAX {
            v = NEXT.fetch_add(1, Ordering::Relaxed) % GATE_STRIPES;
            s.set(v);
        }
        v
    })
}

/// RAII gate entry for one traversal. Hold it across every dereference of
/// arena nodes; drop it as soon as the data has been copied out.
pub struct ReadGuard<'a> {
    stripe: &'a GateStripe,
}

impl Drop for ReadGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.stripe.egress.fetch_add(1, Ordering::Release);
    }
}

/// A batch of retired slots awaiting the reader grace period.
struct PendingBatch {
    occupied: bool,
    snap: [u64; GATE_STRIPES],
    chunks: Vec<u32>,
    interiors: Vec<u32>,
}

/// Recycling state, reached only through an [`ArenaWriter`] (whose
/// construction contract makes access exclusive).
struct Recycler {
    /// Stamped batches awaiting grace.
    ring: [PendingBatch; RING_SLOTS],
    /// Intrusive free-list heads. The link (next free index) is stored in
    /// the first 4 bytes of the freed slot itself.
    free_chunk_head: u32,
    free_interior_head: u32,
    free_chunks: usize,
    free_interiors: usize,
    /// Slots dropped because the ring was full (garbage until compact).
    leaked: u64,
}

impl Recycler {
    fn new() -> Self {
        Self {
            ring: std::array::from_fn(|_| PendingBatch {
                occupied: false,
                snap: [0; GATE_STRIPES],
                chunks: Vec::with_capacity(RETIRE_LOG_CAP),
                interiors: Vec::with_capacity(RETIRE_LOG_CAP),
            }),
            free_chunk_head: NO_SLOT,
            free_interior_head: NO_SLOT,
            free_chunks: 0,
            free_interiors: 0,
            leaked: 0,
        }
    }
}

/// Superseded slots reported by tree operations, owned by the writer.
///
/// Tree operations only *record* what they superseded; nothing here is
/// reusable until the writer has published the root stores that made the
/// slots unreachable and then handed the log to
/// [`ArenaWriter::retire`], which stamps it with a reader-gate snapshot.
/// Separating "record" from "stamp" is what keeps the grace reasoning
/// sound: a stamp taken before the unpublishing store could clear a
/// reader that goes on to walk the still-published old tree.
pub(crate) struct RetireLog {
    pub(crate) chunks: Vec<u32>,
    pub(crate) interiors: Vec<u32>,
}

impl RetireLog {
    pub(crate) fn new() -> Self {
        Self {
            chunks: Vec::with_capacity(RETIRE_LOG_CAP),
            interiors: Vec::with_capacity(RETIRE_LOG_CAP),
        }
    }

    /// Should the writer flush after the current operation? Leaves slack
    /// below the buffers' capacity so small inserts never trigger growth.
    #[inline]
    pub(crate) fn wants_flush(&self) -> bool {
        const HEADROOM: usize = 128;
        self.chunks.len() >= RETIRE_LOG_CAP - HEADROOM
            || self.interiors.len() >= RETIRE_LOG_CAP - HEADROOM
    }
}

/// Recycling counters for observability and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecycleStats {
    /// Slots currently on the chunk free list.
    pub free_chunks: usize,
    /// Slots currently on the interior free list.
    pub free_interiors: usize,
    /// Retired slots staged or awaiting grace (not yet reusable).
    pub pending: usize,
    /// Slots dropped because the pending ring was full.
    pub leaked: u64,
}

/// Fixed-capacity two-region slab arena. Pre-allocates all memory upfront.
///
/// The backing buffer is a raw `NonNull<u8>` paired with the `Layout` it was
/// allocated with. `Vec<u8>` is not used because `Vec` would deallocate with
/// `align_of::<u8>() = 1`, mismatching the 64-byte alignment we need for
/// cache-line chunks and producing UB at drop.
pub struct Arena {
    /// 64-byte aligned backing storage. Set once in `new` and never
    /// reassigned — a plain field (not `UnsafeCell`) so the compiler may
    /// hoist the base-pointer load out of node-walk loops.
    ptr: NonNull<u8>,
    /// Layout the buffer was allocated with — needed for matched dealloc.
    layout: Layout,
    /// Total capacity in bytes.
    capacity: usize,
    /// Reader gate for grace-period recycling. Striped; read-mostly from
    /// the writer's perspective.
    gate: ReadGate,
    /// Bump cursors, advanced only through an [`ArenaWriter`]. `low` is
    /// the next free byte in the chunk region (grows up); `high` is one
    /// past the last free byte of the interior region (grows down).
    /// Isolated on their own cache line so writer bumps don't evict the
    /// read-mostly fields above from reader caches.
    cursors: CachePadded<(AtomicUsize, AtomicUsize)>,
    /// Recycling state, reached only through an [`ArenaWriter`].
    recycler: UnsafeCell<Recycler>,
}

// SAFETY: Mutation (cursors, recycler, slot writes) flows exclusively
// through `ArenaWriter` / `RegionWriter`, whose construction contracts
// guarantee a single mutator. Concurrent readers access
// previously-allocated nodes, which are immutable until retired; retired
// slots are rewritten only after the reader gate proves no reader can
// still observe them.
unsafe impl Send for Arena {}
// SAFETY: see Send impl above.
unsafe impl Sync for Arena {}

impl Arena {
    /// Largest legal arena capacity: 32 GiB.
    ///
    /// Offsets are u32 *slot indices* with bit 31 stolen by the C-tree as
    /// a leaf/interior tag, so each region holds at most 2^31 slots. The
    /// tighter bound is the 16-byte interior region: 2^31 slots x 16 B.
    pub const MAX_CAPACITY: usize = (1 << 31) * INTERIOR_SLOT;

    /// Create an arena with `capacity` bytes, 64-byte aligned.
    ///
    /// # Panics
    /// Panics if `capacity == 0` or `capacity > Self::MAX_CAPACITY`.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Arena capacity must be > 0");
        assert!(
            capacity <= Self::MAX_CAPACITY,
            "Arena capacity {capacity} exceeds MAX_CAPACITY {} (slot indices are u32 with a tag bit)",
            Self::MAX_CAPACITY
        );
        let layout = Layout::from_size_align(capacity, 64)
            .expect("Layout: capacity > 0, alignment is power of two");
        // SAFETY: layout has non-zero size (capacity > 0); alloc_zeroed
        // initializes the buffer so reads of un-allocated bytes are
        // well-defined zero.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            std::alloc::handle_alloc_error(layout);
        };
        Self {
            ptr,
            layout,
            capacity,
            gate: ReadGate::new(),
            cursors: CachePadded((AtomicUsize::new(0), AtomicUsize::new(capacity))),
            recycler: UnsafeCell::new(Recycler::new()),
        }
    }

    /// Enter the reader gate for one traversal. Every dereference of arena
    /// nodes by a non-writer thread must happen while a guard is live —
    /// the grace period that makes slot recycling sound is defined by it.
    #[inline]
    pub fn read_guard(&self) -> ReadGuard<'_> {
        let stripe = &self.gate.stripes[gate_stripe()].0;
        stripe.ingress.fetch_add(1, Ordering::SeqCst);
        ReadGuard { stripe }
    }

    /// Obtain the arena's write handle. All allocation, retirement, and
    /// reclamation go through the returned [`ArenaWriter`]; this is the
    /// single point where the exclusivity proof is made, so the handle's
    /// methods are safe.
    ///
    /// # Safety
    /// At most one `ArenaWriter` may be live per arena at any time, and no
    /// [`RegionWriter`] (from [`region`](Self::region)) or
    /// [`commit_regions`](Self::commit_regions) call may overlap its
    /// lifetime. In `aether-graph` the proof is `DynamicGraph::writer()`'s
    /// CAS guard: one `Writer` exists at a time and owns the handle.
    #[inline]
    pub unsafe fn writer(&self) -> ArenaWriter<'_> {
        ArenaWriter { arena: self }
    }

    // -- Access -------------------------------------------------------------

    /// Pointer to chunk slot `idx` (for prefetch hints).
    ///
    /// # Safety
    /// `idx` must be an allocated chunk slot.
    #[inline(always)]
    pub unsafe fn chunk_ptr(&self, idx: u32) -> *const u8 {
        // SAFETY: caller asserts idx is in bounds.
        unsafe { self.ptr.as_ptr().add(idx as usize * CHUNK_SLOT) as *const u8 }
    }

    /// Pointer to interior slot `idx` (for prefetch hints).
    ///
    /// # Safety
    /// `idx` must be an allocated interior slot.
    #[inline(always)]
    pub unsafe fn interior_ptr(&self, idx: u32) -> *const u8 {
        // SAFETY: caller asserts idx is in bounds.
        unsafe { self.ptr.as_ptr().add(self.interior_byte(idx)) as *const u8 }
    }

    /// Reference to the chunk at slot `idx`.
    ///
    /// # Safety
    /// `idx` must be an allocated, initialized chunk slot, and the caller
    /// must hold either the write handle or a [`ReadGuard`] entered before
    /// the slot could have been retired.
    #[inline(always)]
    pub unsafe fn chunk(&self, idx: u32) -> &Chunk {
        // SAFETY: `idx` is an allocated slot, so the offset is in bounds.
        let p = unsafe { self.ptr.as_ptr().add(idx as usize * CHUNK_SLOT) } as *const Chunk;
        // SAFETY: the slot is initialized and, per the caller's guard
        // contract, cannot be rewritten while this reference lives.
        unsafe { &*p }
    }

    /// Reference to the interior node at slot `idx`.
    ///
    /// # Safety
    /// As [`chunk`](Self::chunk), for the interior region.
    #[inline(always)]
    pub unsafe fn interior(&self, idx: u32) -> &Interior {
        // SAFETY: `idx` is an allocated slot, so the offset is in bounds.
        let p = unsafe { self.ptr.as_ptr().add(self.interior_byte(idx)) } as *const Interior;
        // SAFETY: the slot is initialized and, per the caller's guard
        // contract, cannot be rewritten while this reference lives.
        unsafe { &*p }
    }

    /// Byte offset of interior slot `idx` (slots count down from the top).
    #[inline(always)]
    fn interior_byte(&self, idx: u32) -> usize {
        self.capacity - (idx as usize + 1) * INTERIOR_SLOT
    }

    /// Read an intrusive free-list link stored at `byte`.
    ///
    /// # Safety
    /// `byte` must be the in-bounds start of a slot on a free list.
    #[inline]
    unsafe fn read_link(&self, byte: usize) -> u32 {
        // SAFETY: `byte` is in bounds per the caller contract.
        let p = unsafe { self.ptr.as_ptr().add(byte) } as *const u32;
        // SAFETY: the location holds a link written by `write_link`.
        unsafe { std::ptr::read(p) }
    }

    /// Write an intrusive free-list link at `byte`.
    ///
    /// # Safety
    /// `byte` must be in-bounds and the slot unobservable by readers.
    #[inline]
    unsafe fn write_link(&self, byte: usize, next: u32) {
        // SAFETY: `byte` is in bounds per the caller contract.
        let p = unsafe { self.ptr.as_ptr().add(byte) }.cast::<u32>();
        // SAFETY: the slot is unobservable by readers per the caller
        // contract, so the write cannot race a reference.
        unsafe { std::ptr::write(p, next) };
    }

    /// Thread every grace-cleared pending batch onto the intrusive free
    /// lists. Writing the link into a freed slot is sound precisely
    /// because grace has passed: no reader can still hold a reference.
    fn reclaim_ready(&self, rec: &mut Recycler) {
        for batch in &mut rec.ring {
            if !batch.occupied || !self.gate.grace_passed(&batch.snap) {
                continue;
            }
            for idx in batch.chunks.drain(..) {
                // SAFETY: grace passed — the slot is unobservable; the
                // link write targets in-bounds memory owned by the sole
                // mutator (`rec` is reached only through `ArenaWriter`).
                unsafe { self.write_link(idx as usize * CHUNK_SLOT, rec.free_chunk_head) };
                rec.free_chunk_head = idx;
                rec.free_chunks += 1;
            }
            for idx in batch.interiors.drain(..) {
                // SAFETY: as above, for the interior region.
                unsafe { self.write_link(self.interior_byte(idx), rec.free_interior_head) };
                rec.free_interior_head = idx;
                rec.free_interiors += 1;
            }
            batch.occupied = false;
        }
    }

    // -- Stats --------------------------------------------------------------

    /// Gross bytes consumed by the two bump regions (recycled slots still
    /// count — they sit inside the regions).
    pub fn used(&self) -> usize {
        let low = self.cursors.0.0.load(Ordering::Relaxed);
        let high = self.cursors.0.1.load(Ordering::Relaxed);
        low + (self.capacity - high)
    }

    /// Total capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    // -- Parallel compaction support ----------------------------------------

    /// Carve a private, disjoint slot region for one compaction thread.
    ///
    /// # Safety
    /// The caller must guarantee (a) exclusive access to the arena's
    /// regions being written (fresh arena, cursors untouched, no live
    /// [`ArenaWriter`]), (b) that the handed-out
    /// `[chunk_start, chunk_start + chunk_count)` and
    /// `[interior_start, interior_start + interior_count)` ranges are
    /// disjoint across threads and within the eventual cursor commit from
    /// [`commit_regions`](Self::commit_regions).
    pub unsafe fn region(
        &self,
        chunk_start: u32,
        chunk_count: u32,
        interior_start: u32,
        interior_count: u32,
    ) -> RegionWriter<'_> {
        RegionWriter {
            arena: self,
            next_chunk: chunk_start,
            chunk_end: chunk_start + chunk_count,
            next_interior: interior_start,
            interior_end: interior_start + interior_count,
        }
    }

    /// Publish the cursor positions after a parallel region build:
    /// `chunk_slots` low-region slots and `interior_slots` high-region
    /// slots are now allocated.
    ///
    /// # Safety
    /// All region writers must be finished, their ranges must exactly
    /// tile `[0, chunk_slots)` and `[0, interior_slots)`, and no reader
    /// may observe the arena until the caller publishes roots.
    pub unsafe fn commit_regions(&self, chunk_slots: usize, interior_slots: usize) {
        self.cursors
            .0
            .0
            .store(chunk_slots * CHUNK_SLOT, Ordering::Relaxed);
        self.cursors.0.1.store(
            self.capacity - interior_slots * INTERIOR_SLOT,
            Ordering::Relaxed,
        );
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was returned by `alloc_zeroed(self.layout)` in
        // `new`, we own it exclusively here, and the layout matches.
        unsafe {
            std::alloc::dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// The arena's write handle: allocation, retirement, and reclamation.
///
/// Constructed by the one `unsafe` call [`Arena::writer`], whose contract
/// (at most one live handle, no overlapping region builds) is what makes
/// every method here safe — the type carries the single-writer proof so
/// call sites don't re-assert it. Derefs to [`Arena`] for the read-side
/// API.
pub struct ArenaWriter<'a> {
    arena: &'a Arena,
}

impl std::ops::Deref for ArenaWriter<'_> {
    type Target = Arena;
    #[inline(always)]
    fn deref(&self) -> &Arena {
        self.arena
    }
}

impl<'a> ArenaWriter<'a> {
    /// The underlying arena, at the handle's full lifetime — so node
    /// references taken for reading coexist with `&mut self` allocation
    /// calls (all arena mutation is interior; shared references to live
    /// nodes are never invalidated by allocating elsewhere).
    #[inline(always)]
    pub fn arena(&self) -> &'a Arena {
        self.arena
    }
}

impl ArenaWriter<'_> {
    /// The recycler, exclusive by the construction contract.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    fn recycler(&self) -> &mut Recycler {
        // SAFETY: `Arena::writer`'s contract makes this handle the only
        // path to the recycler, and the handle is not Sync.
        unsafe { &mut *self.arena.recycler.get() }
    }

    /// Allocate one 64-byte chunk slot. Returns the slot index, or `None`
    /// when the arena is full (free list empty, cursors met, and no
    /// pending batch has cleared its grace period).
    #[inline]
    pub fn alloc_chunk(&mut self) -> Option<u32> {
        let rec = self.recycler();
        if let Some(idx) = self.pop_free_chunk(rec) {
            return Some(idx);
        }
        if let Some(idx) = self.bump_chunk() {
            return Some(idx);
        }
        // Cursors met: reclaim any grace-cleared pending batches and retry
        // the free list before reporting full.
        self.arena.reclaim_ready(rec);
        self.pop_free_chunk(rec)
    }

    /// Allocate one 16-byte interior slot. See [`alloc_chunk`](Self::alloc_chunk).
    #[inline]
    pub fn alloc_interior(&mut self) -> Option<u32> {
        let rec = self.recycler();
        if let Some(idx) = self.pop_free_interior(rec) {
            return Some(idx);
        }
        if let Some(idx) = self.bump_interior() {
            return Some(idx);
        }
        self.arena.reclaim_ready(rec);
        self.pop_free_interior(rec)
    }

    #[inline]
    fn pop_free_chunk(&self, rec: &mut Recycler) -> Option<u32> {
        if rec.free_chunk_head == NO_SLOT {
            return None;
        }
        let idx = rec.free_chunk_head;
        // SAFETY: a free chunk slot stores the next free index in its
        // first 4 bytes; the slot is in bounds by construction.
        rec.free_chunk_head = unsafe { self.arena.read_link(idx as usize * CHUNK_SLOT) };
        rec.free_chunks -= 1;
        Some(idx)
    }

    #[inline]
    fn pop_free_interior(&self, rec: &mut Recycler) -> Option<u32> {
        if rec.free_interior_head == NO_SLOT {
            return None;
        }
        let idx = rec.free_interior_head;
        // SAFETY: a free interior slot stores the next free index in its
        // first 4 bytes; the slot is in bounds by construction.
        rec.free_interior_head = unsafe { self.arena.read_link(self.arena.interior_byte(idx)) };
        rec.free_interiors -= 1;
        Some(idx)
    }

    #[inline]
    fn bump_chunk(&self) -> Option<u32> {
        let low = self.arena.cursors.0.0.load(Ordering::Relaxed);
        let high = self.arena.cursors.0.1.load(Ordering::Relaxed);
        let new_low = low + CHUNK_SLOT;
        if new_low > high {
            return None;
        }
        self.arena.cursors.0.0.store(new_low, Ordering::Relaxed);
        Some((low / CHUNK_SLOT) as u32)
    }

    #[inline]
    fn bump_interior(&self) -> Option<u32> {
        let low = self.arena.cursors.0.0.load(Ordering::Relaxed);
        let high = self.arena.cursors.0.1.load(Ordering::Relaxed);
        if high < low + INTERIOR_SLOT {
            return None;
        }
        let new_high = high - INTERIOR_SLOT;
        let idx = ((self.arena.capacity - new_high) / INTERIOR_SLOT - 1) as u32;
        // Reserve index 0x7FFF_FFFF: tagged it would collide with the
        // NULL sentinel (u32::MAX).
        if idx >= (1 << 31) - 1 {
            return None;
        }
        self.arena.cursors.0.1.store(new_high, Ordering::Relaxed);
        Some(idx)
    }

    /// Allocate a chunk slot and write `val` into it.
    #[inline]
    pub fn alloc_write_chunk(&mut self, val: Chunk) -> Option<u32> {
        let idx = self.alloc_chunk()?;
        // SAFETY: `idx` is a fresh slot, so the offset is in bounds.
        let p = unsafe { self.arena.ptr.as_ptr().add(idx as usize * CHUNK_SLOT) }.cast::<Chunk>();
        // SAFETY: the slot is fresh (unobservable), 64-byte aligned, and
        // `Chunk` is exactly one slot (const-asserted in ctree.rs).
        unsafe { std::ptr::write(p, val) };
        Some(idx)
    }

    /// Allocate an interior slot and write `val` into it.
    #[inline]
    pub fn alloc_write_interior(&mut self, val: Interior) -> Option<u32> {
        let idx = self.alloc_interior()?;
        // SAFETY: `idx` is a fresh slot, so the offset is in bounds.
        let p = unsafe { self.arena.ptr.as_ptr().add(self.arena.interior_byte(idx)) }
            .cast::<Interior>();
        // SAFETY: the slot is fresh (unobservable), 16-byte aligned, and
        // `Interior` is exactly one slot (const-asserted in ctree.rs).
        unsafe { std::ptr::write(p, val) };
        Some(idx)
    }

    /// Stamp the retire log with a gate snapshot and queue it for
    /// reclamation, then fold any batches whose grace period has already
    /// passed onto the free lists.
    ///
    /// The log's buffers are swapped with pre-sized ring buffers, so a
    /// flush at the [`RetireLog::wants_flush`] watermark never allocates.
    /// If every ring slot is still awaiting grace, the logged slots are
    /// dropped (garbage until compact) — recycling is an optimization and
    /// must never block the writer.
    ///
    /// # Safety
    /// Every logged slot must already be unreachable from any published
    /// root (the root stores superseding it have happened), and no slot
    /// may be logged twice. The snapshot's meaning is "any reader that
    /// entered after this point cannot observe the logged slots"; stamping
    /// a still-published slot would let a later reader walk memory that
    /// gets rewritten under it.
    pub(crate) unsafe fn retire(&mut self, log: &mut RetireLog) {
        if log.chunks.is_empty() && log.interiors.is_empty() {
            return;
        }
        let rec = self.recycler();
        self.arena.reclaim_ready(rec);
        let Some(slot) = rec.ring.iter_mut().find(|b| !b.occupied) else {
            rec.leaked += (log.chunks.len() + log.interiors.len()) as u64;
            log.chunks.clear();
            log.interiors.clear();
            return;
        };
        std::mem::swap(&mut slot.chunks, &mut log.chunks);
        std::mem::swap(&mut slot.interiors, &mut log.interiors);
        log.chunks.clear();
        log.interiors.clear();
        self.arena.gate.snapshot(&mut slot.snap);
        slot.occupied = true;
    }

    /// Fold grace-cleared pending batches onto the free lists without
    /// stamping anything new. Useful at commit points when the log is
    /// empty but earlier batches may have cleared.
    pub(crate) fn reclaim(&mut self) {
        let rec = self.recycler();
        self.arena.reclaim_ready(rec);
    }

    /// Recycling counters. Slots recorded in a not-yet-stamped
    /// [`RetireLog`] count as neither free nor pending.
    pub(crate) fn recycle_stats(&self) -> RecycleStats {
        let rec = self.recycler();
        let pending = rec
            .ring
            .iter()
            .filter(|b| b.occupied)
            .map(|b| b.chunks.len() + b.interiors.len())
            .sum::<usize>();
        RecycleStats {
            free_chunks: rec.free_chunks,
            free_interiors: rec.free_interiors,
            pending,
            leaked: rec.leaked,
        }
    }
}

/// Private slot range handed to one compaction thread. Writes are plain
/// (no atomics): the range is disjoint from every other thread's by the
/// [`Arena::region`] contract.
pub struct RegionWriter<'a> {
    arena: &'a Arena,
    next_chunk: u32,
    chunk_end: u32,
    next_interior: u32,
    interior_end: u32,
}

impl RegionWriter<'_> {
    /// Write a chunk into the next reserved slot. Panics (debug) past the
    /// reservation — the compaction prepass sizes ranges exactly.
    #[inline]
    pub fn write_chunk(&mut self, val: Chunk) -> u32 {
        debug_assert!(self.next_chunk < self.chunk_end, "chunk region overrun");
        let idx = self.next_chunk;
        self.next_chunk += 1;
        // SAFETY: `idx` is inside this writer's reservation, in bounds.
        let p = unsafe { self.arena.ptr.as_ptr().add(idx as usize * CHUNK_SLOT) }.cast::<Chunk>();
        // SAFETY: the reservation is exclusive to this writer, so the
        // write cannot race another thread.
        unsafe { std::ptr::write(p, val) };
        idx
    }

    /// Write an interior node into the next reserved slot.
    #[inline]
    pub fn write_interior(&mut self, val: Interior) -> u32 {
        debug_assert!(
            self.next_interior < self.interior_end,
            "interior region overrun"
        );
        let idx = self.next_interior;
        self.next_interior += 1;
        // SAFETY: `idx` is inside this writer's reservation, in bounds.
        let p = unsafe { self.arena.ptr.as_ptr().add(self.arena.interior_byte(idx)) }
            .cast::<Interior>();
        // SAFETY: the reservation is exclusive to this writer, so the
        // write cannot race another thread.
        unsafe { std::ptr::write(p, val) };
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;

    #[test]
    fn basic_alloc_two_regions() {
        let arena = Arena::new(4096);
        assert_eq!(arena.used(), 0);
        // SAFETY: sole handle in a single-threaded test.
        let mut aw = unsafe { arena.writer() };

        assert_eq!(aw.alloc_chunk().unwrap(), 0);
        assert_eq!(aw.alloc_chunk().unwrap(), 1);
        assert_eq!(aw.alloc_interior().unwrap(), 0);
        assert_eq!(arena.used(), 2 * CHUNK_SLOT + INTERIOR_SLOT);
    }

    #[test]
    fn alloc_write_chunk_roundtrip() {
        let arena = Arena::new(4096);
        // SAFETY: sole handle in a single-threaded test.
        let mut aw = unsafe { arena.writer() };
        let chunk = Chunk::from_sorted(&[10, 20, 30]);

        let idx = aw.alloc_write_chunk(chunk).unwrap();
        // SAFETY: idx was just written as a Chunk; the writer holds the
        // sole handle, so the slot cannot have been retired.
        let read = unsafe { arena.chunk(idx) };
        assert_eq!(read.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn arena_full_returns_none() {
        let arena = Arena::new(128);
        // SAFETY: sole handle in a single-threaded test.
        let mut aw = unsafe { arena.writer() };
        assert!(aw.alloc_chunk().is_some());
        assert!(aw.alloc_chunk().is_some());
        assert!(aw.alloc_chunk().is_none());
    }

    #[test]
    fn regions_meet_in_the_middle() {
        let arena = Arena::new(2 * CHUNK_SLOT + 2 * INTERIOR_SLOT);
        // SAFETY: sole handle in a single-threaded test.
        let mut aw = unsafe { arena.writer() };
        assert!(aw.alloc_chunk().is_some());
        assert!(aw.alloc_interior().is_some());
        assert!(aw.alloc_interior().is_some());
        // One chunk slot's worth of bytes remains, and it is free.
        assert!(aw.alloc_chunk().is_some());
        assert!(aw.alloc_interior().is_none());
        assert!(aw.alloc_chunk().is_none());
    }

    #[test]
    fn recycling_reuses_slots_after_grace() {
        let arena = Arena::new(4096);
        let mut log = RetireLog::new();
        // SAFETY: sole handle in a single-threaded test.
        let mut aw = unsafe { arena.writer() };
        let a = aw.alloc_chunk().unwrap();
        let b = aw.alloc_chunk().unwrap();
        log.chunks.push(a);
        log.chunks.push(b);
        // No readers in flight: grace passes at the next reclaim.
        // SAFETY: the logged slots are test-local and never published.
        unsafe { aw.retire(&mut log) };
        aw.reclaim();
        let stats = aw.recycle_stats();
        assert_eq!(stats.free_chunks, 2);
        // LIFO within a batch: `b` was threaded last.
        assert_eq!(aw.alloc_chunk().unwrap(), b);
        assert_eq!(aw.alloc_chunk().unwrap(), a);
        assert_eq!(aw.recycle_stats().free_chunks, 0);
    }

    #[test]
    fn active_reader_blocks_reclaim() {
        let arena = Arena::new(4096);
        let mut log = RetireLog::new();
        // SAFETY: sole handle in a single-threaded test (the guard plays
        // the reader).
        let mut aw = unsafe { arena.writer() };
        let a = aw.alloc_chunk().unwrap();
        let guard = arena.read_guard();
        log.chunks.push(a);
        // SAFETY: the logged slot is test-local and never published.
        unsafe { aw.retire(&mut log) };
        aw.reclaim();
        assert_eq!(
            aw.recycle_stats().free_chunks,
            0,
            "slot must stay pending while the reader is inside the gate"
        );
        assert_eq!(aw.recycle_stats().pending, 1);
        drop(guard);
        aw.reclaim();
        assert_eq!(aw.recycle_stats().free_chunks, 1);
    }

    #[test]
    fn reader_entered_after_stamp_does_not_block() {
        let arena = Arena::new(4096);
        let mut log = RetireLog::new();
        // SAFETY: sole handle in a single-threaded test.
        let mut aw = unsafe { arena.writer() };
        let a = aw.alloc_chunk().unwrap();
        log.chunks.push(a);
        // SAFETY: the logged slot is test-local and never published.
        unsafe { aw.retire(&mut log) };
        // This reader entered after the snapshot: it can only see the
        // post-retirement roots, so it must not block reuse.
        let _guard = arena.read_guard();
        aw.reclaim();
        assert_eq!(aw.recycle_stats().free_chunks, 1);
    }

    #[test]
    fn alloc_exhaustion_reclaims_pending() {
        // Room for exactly two chunk slots.
        let arena = Arena::new(128);
        let mut log = RetireLog::new();
        // SAFETY: sole handle in a single-threaded test.
        let mut aw = unsafe { arena.writer() };
        let a = aw.alloc_chunk().unwrap();
        let _b = aw.alloc_chunk().unwrap();
        log.chunks.push(a);
        // SAFETY: the logged slot is test-local and never published.
        unsafe { aw.retire(&mut log) };
        // The stamped batch is reclaimed by the exhausted alloc itself.
        let c = aw.alloc_chunk().unwrap();
        assert_eq!(c, a);
    }

    #[test]
    fn interior_recycling_roundtrip() {
        let arena = Arena::new(4096);
        let mut log = RetireLog::new();
        // SAFETY: sole handle in a single-threaded test.
        let mut aw = unsafe { arena.writer() };
        let i0 = aw.alloc_interior().unwrap();
        let i1 = aw.alloc_interior().unwrap();
        log.interiors.push(i0);
        // SAFETY: the logged slot is test-local and never published.
        unsafe { aw.retire(&mut log) };
        aw.reclaim();
        assert_eq!(aw.recycle_stats().free_interiors, 1);
        assert_eq!(aw.alloc_interior().unwrap(), i0);
        // The bump cursor was untouched by the recycled alloc.
        assert_eq!(aw.alloc_interior().unwrap(), i1 + 1);
    }

    #[test]
    fn interior_slots_count_from_the_top() {
        let arena = Arena::new(4096);
        // SAFETY: sole handle in a single-threaded test.
        let mut aw = unsafe { arena.writer() };
        let i0 = aw.alloc_interior().unwrap();
        let i1 = aw.alloc_interior().unwrap();
        assert_eq!((i0, i1), (0, 1));
        // SAFETY: slot 0 was just allocated.
        let p0 = unsafe { arena.interior_ptr(0) };
        // SAFETY: slot 1 was just allocated.
        let p1 = unsafe { arena.interior_ptr(1) };
        assert!(p0 > p1);
        assert_eq!(p0.addr() % INTERIOR_SLOT, 0);
    }

    #[test]
    fn region_writer_places_slots_exactly() {
        let arena = Arena::new(4096);
        // SAFETY: fresh arena, single region, cursors committed below.
        let mut region = unsafe { arena.region(0, 2, 0, 1) };
        let c0 = region.write_chunk(Chunk::from_sorted(&[1, 2]));
        let c1 = region.write_chunk(Chunk::from_sorted(&[3]));
        assert_eq!((c0, c1), (0, 1));
        // SAFETY: the sole region writer is finished; its range tiles the
        // committed extents exactly.
        unsafe { arena.commit_regions(2, 1) };
        assert_eq!(arena.used(), 2 * CHUNK_SLOT + INTERIOR_SLOT);
        // SAFETY: slot 0 was written above and never retired.
        let read = unsafe { arena.chunk(0) };
        assert_eq!(read.as_slice(), &[1, 2]);
    }

    #[test]
    #[should_panic(expected = "Arena capacity must be > 0")]
    fn zero_capacity_panics() {
        let _ = Arena::new(0);
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_CAPACITY")]
    fn over_max_capacity_panics() {
        // The assert fires before any allocation happens, so this test
        // does not actually try to reserve >32 GiB.
        let _ = Arena::new(Arena::MAX_CAPACITY + 1);
    }
}
