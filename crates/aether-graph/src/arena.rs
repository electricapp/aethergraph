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
//! The arena is NOT thread-safe for allocation (single writer). It IS safe
//! for concurrent reads of previously-allocated nodes, provided readers
//! hold a [`ReadGuard`] for the duration of the traversal — the guard is
//! what makes slot reuse sound.

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

/// Pads a field group onto its own cache line. 128 bytes covers the
/// adjacent-line prefetcher on x86_64 and Apple Silicon's 128-byte granules.
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), repr(align(128)))]
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    repr(align(64))
)]
struct CachePadded<T>(T);

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

/// Writer-only recycling state (behind `UnsafeCell`, same single-writer
/// contract as the bump cursors).
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
/// [`Arena::retire_log`], which stamps it with a reader-gate snapshot.
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
    /// Bump cursors, writer-only. `low` is the next free byte in the chunk
    /// region (grows up); `high` is one past the last free byte of the
    /// interior region (grows down). Isolated on their own cache line so
    /// writer bumps don't evict the read-mostly fields above from reader
    /// caches.
    cursors: CachePadded<(AtomicUsize, AtomicUsize)>,
    /// Recycling state, writer-only.
    recycler: UnsafeCell<Recycler>,
}

// SAFETY: The arena is single-writer (only the graph writer thread allocates,
// retires, and reclaims — `recycler` and the cursors are never touched
// concurrently). Concurrent readers access previously-allocated nodes, which
// are immutable until retired; retired slots are rewritten only after the
// reader gate proves no reader can still observe them.
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

    // -- Allocation (single-writer) -----------------------------------------

    /// Allocate one 64-byte chunk slot. Returns the slot index, or `None`
    /// when the arena is full (free list empty, cursors met, and no
    /// pending batch has cleared its grace period).
    ///
    /// # Safety
    /// Single-writer invariant: only one thread may allocate, retire, or
    /// reclaim concurrently. Callers in `aether-graph` go through
    /// `DynamicGraph::writer()`, which enforces this at runtime.
    #[inline]
    pub unsafe fn alloc_chunk(&self) -> Option<u32> {
        // SAFETY: single-writer contract makes the recycler ours.
        let rec = unsafe { &mut *self.recycler.get() };
        if rec.free_chunk_head != NO_SLOT {
            let idx = rec.free_chunk_head;
            // SAFETY: a free chunk slot stores the next free index in its
            // first 4 bytes; the slot is in bounds by construction.
            rec.free_chunk_head = unsafe { self.read_link(idx as usize * CHUNK_SLOT) };
            rec.free_chunks -= 1;
            return Some(idx);
        }
        if let Some(idx) = self.bump_chunk() {
            return Some(idx);
        }
        // Cursors met: reclaim any grace-cleared pending batches and retry
        // the free list before reporting full.
        // SAFETY: single-writer contract.
        unsafe { self.reclaim_ready(rec) };
        if rec.free_chunk_head != NO_SLOT {
            let idx = rec.free_chunk_head;
            // SAFETY: as above.
            rec.free_chunk_head = unsafe { self.read_link(idx as usize * CHUNK_SLOT) };
            rec.free_chunks -= 1;
            return Some(idx);
        }
        None
    }

    /// Allocate one 16-byte interior slot. See [`alloc_chunk`](Self::alloc_chunk).
    ///
    /// # Safety
    /// Single-writer invariant (see [`alloc_chunk`](Self::alloc_chunk)).
    #[inline]
    pub unsafe fn alloc_interior(&self) -> Option<u32> {
        // SAFETY: single-writer contract makes the recycler ours.
        let rec = unsafe { &mut *self.recycler.get() };
        if rec.free_interior_head != NO_SLOT {
            let idx = rec.free_interior_head;
            // SAFETY: a free interior slot stores the next free index in
            // its first 4 bytes; the slot is in bounds by construction.
            rec.free_interior_head = unsafe { self.read_link(self.interior_byte(idx)) };
            rec.free_interiors -= 1;
            return Some(idx);
        }
        if let Some(idx) = self.bump_interior() {
            return Some(idx);
        }
        // SAFETY: single-writer contract.
        unsafe { self.reclaim_ready(rec) };
        if rec.free_interior_head != NO_SLOT {
            let idx = rec.free_interior_head;
            // SAFETY: as above.
            rec.free_interior_head = unsafe { self.read_link(self.interior_byte(idx)) };
            rec.free_interiors -= 1;
            return Some(idx);
        }
        None
    }

    #[inline]
    fn bump_chunk(&self) -> Option<u32> {
        let low = self.cursors.0.0.load(Ordering::Relaxed);
        let high = self.cursors.0.1.load(Ordering::Relaxed);
        let new_low = low + CHUNK_SLOT;
        if new_low > high {
            return None;
        }
        self.cursors.0.0.store(new_low, Ordering::Relaxed);
        Some((low / CHUNK_SLOT) as u32)
    }

    #[inline]
    fn bump_interior(&self) -> Option<u32> {
        let low = self.cursors.0.0.load(Ordering::Relaxed);
        let high = self.cursors.0.1.load(Ordering::Relaxed);
        if high < low + INTERIOR_SLOT {
            return None;
        }
        let new_high = high - INTERIOR_SLOT;
        let idx = ((self.capacity - new_high) / INTERIOR_SLOT - 1) as u32;
        // Reserve index 0x7FFF_FFFF: tagged it would collide with the
        // NULL sentinel (u32::MAX).
        if idx >= (1 << 31) - 1 {
            return None;
        }
        self.cursors.0.1.store(new_high, Ordering::Relaxed);
        Some(idx)
    }

    /// Byte offset of interior slot `idx` (slots count down from the top).
    #[inline(always)]
    fn interior_byte(&self, idx: u32) -> usize {
        self.capacity - (idx as usize + 1) * INTERIOR_SLOT
    }

    /// Allocate a chunk slot and write `val` into it.
    ///
    /// # Safety
    /// Single-writer invariant; `T` must be exactly [`CHUNK_SLOT`] bytes
    /// with alignment ≤ 64 (the chunk region is 64-byte aligned).
    #[inline]
    pub unsafe fn alloc_write_chunk<T: Copy>(&self, val: T) -> Option<u32> {
        debug_assert_eq!(std::mem::size_of::<T>(), CHUNK_SLOT);
        debug_assert!(std::mem::align_of::<T>() <= 64);
        // SAFETY: caller upholds the single-writer invariant.
        let idx = unsafe { self.alloc_chunk()? };
        // SAFETY: idx is a fresh in-bounds chunk slot, 64-byte aligned.
        unsafe {
            std::ptr::write(
                self.ptr.as_ptr().add(idx as usize * CHUNK_SLOT) as *mut T,
                val,
            );
        }
        Some(idx)
    }

    /// Allocate an interior slot and write `val` into it.
    ///
    /// # Safety
    /// Single-writer invariant; `T` must be exactly [`INTERIOR_SLOT`] bytes
    /// with alignment ≤ 16 (interior slots are 16-byte aligned: the region
    /// top is the 64-aligned capacity and slots are 16 bytes).
    #[inline]
    pub unsafe fn alloc_write_interior<T: Copy>(&self, val: T) -> Option<u32> {
        debug_assert_eq!(std::mem::size_of::<T>(), INTERIOR_SLOT);
        debug_assert!(std::mem::align_of::<T>() <= 16);
        // SAFETY: caller upholds the single-writer invariant.
        let idx = unsafe { self.alloc_interior()? };
        // SAFETY: idx is a fresh in-bounds interior slot, 16-byte aligned.
        unsafe {
            std::ptr::write(
                self.ptr.as_ptr().add(self.interior_byte(idx)) as *mut T,
                val,
            );
        }
        Some(idx)
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

    /// Typed reference to the chunk at slot `idx`.
    ///
    /// # Safety
    /// `idx` must be an allocated, initialized chunk slot, and the caller
    /// must hold either the writer guard or a [`ReadGuard`] entered before
    /// the slot could have been retired.
    #[inline(always)]
    pub unsafe fn chunk<T>(&self, idx: u32) -> &T {
        debug_assert_eq!(std::mem::size_of::<T>(), CHUNK_SLOT);
        // SAFETY: caller-asserted invariants make the pointer dereferenceable.
        unsafe { &*(self.ptr.as_ptr().add(idx as usize * CHUNK_SLOT) as *const T) }
    }

    /// Typed reference to the interior node at slot `idx`.
    ///
    /// # Safety
    /// As [`chunk`](Self::chunk), for the interior region.
    #[inline(always)]
    pub unsafe fn interior<T>(&self, idx: u32) -> &T {
        debug_assert_eq!(std::mem::size_of::<T>(), INTERIOR_SLOT);
        // SAFETY: caller-asserted invariants make the pointer dereferenceable.
        unsafe { &*(self.ptr.as_ptr().add(self.interior_byte(idx)) as *const T) }
    }

    // -- Retirement + reclamation (single-writer) ---------------------------

    /// Stamp the writer's retire log with a gate snapshot and queue it for
    /// reclamation, then fold any batches whose grace period has already
    /// passed onto the free lists.
    ///
    /// Must be called only after the root stores that made every logged
    /// slot unreachable — the snapshot's meaning is "any reader that
    /// entered after this point cannot observe the logged slots". The
    /// log's buffers are swapped with pre-sized ring buffers, so a flush
    /// at the [`RetireLog::wants_flush`] watermark never allocates. If
    /// every ring slot is still awaiting grace, the logged slots are
    /// dropped (garbage until compact) — recycling is an optimization and
    /// must never block the writer.
    ///
    /// # Safety
    /// Single-writer invariant, every logged slot unpublished, and no slot
    /// logged twice.
    pub(crate) unsafe fn retire_log(&self, log: &mut RetireLog) {
        if log.chunks.is_empty() && log.interiors.is_empty() {
            return;
        }
        // SAFETY: single-writer contract makes the recycler ours.
        let rec = unsafe { &mut *self.recycler.get() };
        // SAFETY: single-writer contract.
        unsafe { self.reclaim_ready(rec) };
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
        self.gate.snapshot(&mut slot.snap);
        slot.occupied = true;
    }

    /// Fold grace-cleared pending batches onto the free lists without
    /// stamping anything new. Useful at commit points when the log is
    /// empty but earlier batches may have cleared.
    ///
    /// # Safety
    /// Single-writer invariant.
    pub(crate) unsafe fn reclaim(&self) {
        // SAFETY: single-writer contract makes the recycler ours.
        let rec = unsafe { &mut *self.recycler.get() };
        // SAFETY: single-writer contract.
        unsafe { self.reclaim_ready(rec) };
    }

    /// Thread every grace-cleared pending batch onto the intrusive free
    /// lists. Writing the link into a freed slot is sound precisely
    /// because grace has passed: no reader can still hold a reference.
    ///
    /// # Safety
    /// Single-writer invariant; `rec` must be the arena's recycler.
    unsafe fn reclaim_ready(&self, rec: &mut Recycler) {
        for batch in &mut rec.ring {
            if !batch.occupied || !self.gate.grace_passed(&batch.snap) {
                continue;
            }
            for idx in batch.chunks.drain(..) {
                // SAFETY: grace passed — the slot is unobservable; the
                // link write targets in-bounds, writer-owned memory.
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

    /// Read an intrusive free-list link stored at `byte`.
    ///
    /// # Safety
    /// `byte` must be the in-bounds start of a slot on a free list.
    #[inline]
    unsafe fn read_link(&self, byte: usize) -> u32 {
        // SAFETY: caller asserts the location holds a link we wrote.
        unsafe { std::ptr::read(self.ptr.as_ptr().add(byte) as *const u32) }
    }

    /// Write an intrusive free-list link at `byte`.
    ///
    /// # Safety
    /// `byte` must be in-bounds and the slot unobservable by readers.
    #[inline]
    unsafe fn write_link(&self, byte: usize, next: u32) {
        // SAFETY: caller asserts exclusivity over the slot.
        unsafe { std::ptr::write(self.ptr.as_ptr().add(byte) as *mut u32, next) };
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

    /// Recycling counters. Writer-thread only (reads the recycler).
    ///
    /// # Safety
    /// Single-writer invariant.
    pub unsafe fn recycle_stats(&self) -> RecycleStats {
        // SAFETY: single-writer contract makes the recycler ours.
        let rec = unsafe { &*self.recycler.get() };
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

    // -- Parallel compaction support ----------------------------------------

    /// Carve a private, disjoint slot region for one compaction thread.
    ///
    /// # Safety
    /// The caller must guarantee (a) exclusive access to the arena's
    /// regions being written (fresh arena, cursors untouched), (b) that
    /// the handed-out `[chunk_start, chunk_start + chunk_count)` and
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
    pub fn write_chunk<T: Copy>(&mut self, val: T) -> u32 {
        debug_assert_eq!(std::mem::size_of::<T>(), CHUNK_SLOT);
        debug_assert!(self.next_chunk < self.chunk_end, "chunk region overrun");
        let idx = self.next_chunk;
        self.next_chunk += 1;
        // SAFETY: idx is inside this writer's exclusive reservation.
        unsafe {
            std::ptr::write(
                self.arena.ptr.as_ptr().add(idx as usize * CHUNK_SLOT) as *mut T,
                val,
            );
        }
        idx
    }

    /// Write an interior node into the next reserved slot.
    #[inline]
    pub fn write_interior<T: Copy>(&mut self, val: T) -> u32 {
        debug_assert_eq!(std::mem::size_of::<T>(), INTERIOR_SLOT);
        debug_assert!(
            self.next_interior < self.interior_end,
            "interior region overrun"
        );
        let idx = self.next_interior;
        self.next_interior += 1;
        // SAFETY: idx is inside this writer's exclusive reservation.
        unsafe {
            std::ptr::write(
                self.arena.ptr.as_ptr().add(self.arena.interior_byte(idx)) as *mut T,
                val,
            );
        }
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

        // SAFETY: test is single-threaded.
        let c0 = unsafe { arena.alloc_chunk() }.unwrap();
        assert_eq!(c0, 0);
        // SAFETY: test is single-threaded.
        let c1 = unsafe { arena.alloc_chunk() }.unwrap();
        assert_eq!(c1, 1);
        // SAFETY: test is single-threaded.
        let i0 = unsafe { arena.alloc_interior() }.unwrap();
        assert_eq!(i0, 0);
        assert_eq!(arena.used(), 2 * CHUNK_SLOT + INTERIOR_SLOT);
    }

    #[test]
    fn alloc_write_chunk_roundtrip() {
        let arena = Arena::new(4096);
        let chunk = Chunk::from_sorted(&[10, 20, 30]);

        // SAFETY: test is single-threaded.
        let idx = unsafe { arena.alloc_write_chunk(chunk) }.unwrap();
        // SAFETY: idx was just written as a Chunk.
        let read: &Chunk = unsafe { arena.chunk(idx) };
        assert_eq!(read.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn arena_full_returns_none() {
        let arena = Arena::new(128);
        // SAFETY: test is single-threaded.
        unsafe {
            assert!(arena.alloc_chunk().is_some());
            assert!(arena.alloc_chunk().is_some());
            assert!(arena.alloc_chunk().is_none());
        }
    }

    #[test]
    fn regions_meet_in_the_middle() {
        let arena = Arena::new(2 * CHUNK_SLOT + 2 * INTERIOR_SLOT);
        // SAFETY: test is single-threaded.
        unsafe {
            assert!(arena.alloc_chunk().is_some());
            assert!(arena.alloc_interior().is_some());
            assert!(arena.alloc_interior().is_some());
            // One chunk slot's worth of bytes remains, and it is free.
            assert!(arena.alloc_chunk().is_some());
            assert!(arena.alloc_interior().is_none());
            assert!(arena.alloc_chunk().is_none());
        }
    }

    #[test]
    fn recycling_reuses_slots_after_grace() {
        let arena = Arena::new(4096);
        let mut log = RetireLog::new();
        // SAFETY: test is single-threaded.
        unsafe {
            let a = arena.alloc_chunk().unwrap();
            let b = arena.alloc_chunk().unwrap();
            log.chunks.push(a);
            log.chunks.push(b);
            // No readers in flight: grace passes at the next reclaim.
            arena.retire_log(&mut log);
            arena.reclaim();
            let stats = arena.recycle_stats();
            assert_eq!(stats.free_chunks, 2);
            // LIFO within a batch: `b` was threaded last.
            assert_eq!(arena.alloc_chunk().unwrap(), b);
            assert_eq!(arena.alloc_chunk().unwrap(), a);
            assert_eq!(arena.recycle_stats().free_chunks, 0);
        }
    }

    #[test]
    fn active_reader_blocks_reclaim() {
        let arena = Arena::new(4096);
        let mut log = RetireLog::new();
        // SAFETY: test is single-threaded (the guard plays the reader).
        unsafe {
            let a = arena.alloc_chunk().unwrap();
            let guard = arena.read_guard();
            log.chunks.push(a);
            arena.retire_log(&mut log);
            arena.reclaim();
            assert_eq!(
                arena.recycle_stats().free_chunks,
                0,
                "slot must stay pending while the reader is inside the gate"
            );
            assert_eq!(arena.recycle_stats().pending, 1);
            drop(guard);
            arena.reclaim();
            assert_eq!(arena.recycle_stats().free_chunks, 1);
        }
    }

    #[test]
    fn reader_entered_after_stamp_does_not_block() {
        let arena = Arena::new(4096);
        let mut log = RetireLog::new();
        // SAFETY: test is single-threaded.
        unsafe {
            let a = arena.alloc_chunk().unwrap();
            log.chunks.push(a);
            arena.retire_log(&mut log);
            // This reader entered after the snapshot: it can only see the
            // post-retirement roots, so it must not block reuse.
            let _guard = arena.read_guard();
            arena.reclaim();
            assert_eq!(arena.recycle_stats().free_chunks, 1);
        }
    }

    #[test]
    fn alloc_exhaustion_reclaims_pending() {
        // Room for exactly two chunk slots.
        let arena = Arena::new(128);
        let mut log = RetireLog::new();
        // SAFETY: test is single-threaded.
        unsafe {
            let a = arena.alloc_chunk().unwrap();
            let _b = arena.alloc_chunk().unwrap();
            log.chunks.push(a);
            arena.retire_log(&mut log);
            // The stamped batch is reclaimed by the exhausted alloc itself.
            let c = arena.alloc_chunk().unwrap();
            assert_eq!(c, a);
        }
    }

    #[test]
    fn interior_recycling_roundtrip() {
        let arena = Arena::new(4096);
        let mut log = RetireLog::new();
        // SAFETY: test is single-threaded.
        unsafe {
            let i0 = arena.alloc_interior().unwrap();
            let i1 = arena.alloc_interior().unwrap();
            log.interiors.push(i0);
            arena.retire_log(&mut log);
            arena.reclaim();
            assert_eq!(arena.recycle_stats().free_interiors, 1);
            assert_eq!(arena.alloc_interior().unwrap(), i0);
            // The bump cursor was untouched by the recycled alloc.
            assert_eq!(arena.alloc_interior().unwrap(), i1 + 1);
        }
    }

    #[test]
    fn interior_slots_count_from_the_top() {
        let arena = Arena::new(4096);
        // SAFETY: test is single-threaded.
        unsafe {
            let i0 = arena.alloc_interior().unwrap();
            let i1 = arena.alloc_interior().unwrap();
            assert_eq!((i0, i1), (0, 1));
            assert!(arena.interior_ptr(0) > arena.interior_ptr(1));
            assert_eq!(arena.interior_ptr(0).addr() % INTERIOR_SLOT, 0);
        }
    }

    #[test]
    fn region_writer_places_slots_exactly() {
        let arena = Arena::new(4096);
        // SAFETY: fresh arena, single region, cursors committed below.
        unsafe {
            let mut region = arena.region(0, 2, 0, 1);
            let c0 = region.write_chunk(Chunk::from_sorted(&[1, 2]));
            let c1 = region.write_chunk(Chunk::from_sorted(&[3]));
            assert_eq!((c0, c1), (0, 1));
            arena.commit_regions(2, 1);
            assert_eq!(arena.used(), 2 * CHUNK_SLOT + INTERIOR_SLOT);
            let read: &Chunk = arena.chunk(0);
            assert_eq!(read.as_slice(), &[1, 2]);
        }
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
