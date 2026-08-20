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
//! Retired slots are recycled: each staged batch is stamped with a reader-
//! gate snapshot and its guard's commit epoch, and is freed once the gate
//! drains past the stamp and no pinned [`Snapshot`](crate::Snapshot) older
//! than it remains. Free lists are intrusive (link in the freed slot) and
//! allocation pops them before bumping, so steady-state ingest reuses
//! garbage instead of growing until [`compact`](crate::DynamicGraph::compact).
//!
//! Mutation goes through [`ArenaWriter`], obtained once from
//! [`Arena::writer`] — the one `unsafe` point proving the single-writer
//! invariant. Concurrent reads are sound while holding a [`ReadGuard`] or
//! reading via a pinned [`Snapshot`](crate::Snapshot).

use crate::chunk::Chunk;
use crate::ctree::Interior;
use crate::pad::CachePadded;
use std::alloc::Layout;
use std::cell::{Cell, UnsafeCell};
use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering, fence};
use std::sync::{Arc, Mutex};

/// Bytes per chunk slot (low region). Matches `Chunk`'s size/alignment.
pub const CHUNK_SLOT: usize = 64;
/// Bytes per interior-node slot (high region). Matches `Interior`'s size.
pub const INTERIOR_SLOT: usize = 16;

/// Free-list terminator / "no slot" sentinel inside the recycler.
const NO_SLOT: u32 = u32::MAX;

/// Reader-gate stripes. Sixteen padded counters keep concurrent readers
/// from serializing on one cache line while staying cheap to snapshot.
const GATE_STRIPES: usize = 16;

/// Retire-log capacity per class before the writer should flush. Also the
/// batch buffers' pre-allocated capacity, so a watermark flush never
/// grows a vector.
pub(crate) const RETIRE_LOG_CAP: usize = 4096;

/// Cleared batches pooled for reuse — keeps unpinned-steady-state
/// retirement allocation-free.
const SPARE_BATCHES: usize = 4;

/// Backstop cap on batches awaiting grace/pin release; past it, staged
/// slots are dropped (garbage until compact) — recycling never blocks
/// the writer.
const MAX_PENDING_BATCHES: usize = 4096;

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

/// A batch of retired slots awaiting reader grace and pin release.
struct PendingBatch {
    /// Epoch the retiring guard commits. A batch is held while any pinned
    /// snapshot's epoch is below it.
    stamp_epoch: u64,
    snap: [u64; GATE_STRIPES],
    chunks: Vec<u32>,
    interiors: Vec<u32>,
}

impl PendingBatch {
    fn new() -> Self {
        Self {
            stamp_epoch: 0,
            snap: [0; GATE_STRIPES],
            chunks: Vec::with_capacity(RETIRE_LOG_CAP),
            interiors: Vec::with_capacity(RETIRE_LOG_CAP),
        }
    }
}

/// Epochs pinned by live [`Snapshot`](crate::Snapshot)s. Registrations
/// only ever happen at the latest committed epoch (which is itself always
/// pinned by the graph), so the cached minimum is monotone nondecreasing
/// and a stale read in the reclaimer is conservative.
pub(crate) struct PinRegistry {
    /// (epoch, refcount), ascending. One entry per distinct pinned epoch.
    pins: Mutex<Vec<(u64, u32)>>,
    /// Cached minimum pinned epoch; `u64::MAX` when nothing is pinned.
    min: AtomicU64,
}

impl PinRegistry {
    fn new() -> Self {
        Self {
            pins: Mutex::new(Vec::new()),
            min: AtomicU64::new(u64::MAX),
        }
    }

    fn register(&self, epoch: u64) {
        let mut pins = self.pins.lock().unwrap();
        match pins.binary_search_by_key(&epoch, |&(e, _)| e) {
            Ok(i) => pins[i].1 += 1,
            Err(i) => pins.insert(i, (epoch, 1)),
        }
        // Release pairs with the reclaimer's Acquire: a pin's reads
        // happen-before any slot rewrite the new minimum permits.
        self.min.store(
            pins.first().map_or(u64::MAX, |&(e, _)| e),
            Ordering::Release,
        );
    }

    fn release(&self, epoch: u64) {
        let mut pins = self.pins.lock().unwrap();
        let i = pins
            .binary_search_by_key(&epoch, |&(e, _)| e)
            .expect("released epoch not pinned");
        pins[i].1 -= 1;
        if pins[i].1 == 0 {
            pins.remove(i);
        }
        self.min.store(
            pins.first().map_or(u64::MAX, |&(e, _)| e),
            Ordering::Release,
        );
    }

    pub(crate) fn min_pinned(&self) -> u64 {
        self.min.load(Ordering::Acquire)
    }

    /// Total live pins (sum of refcounts).
    pub(crate) fn pin_count(&self) -> u64 {
        self.pins
            .lock()
            .unwrap()
            .iter()
            .map(|&(_, c)| u64::from(c))
            .sum()
    }
}

/// RAII pin of one epoch in a [`PinRegistry`]. Clone re-registers.
pub(crate) struct PinTicket {
    reg: Arc<PinRegistry>,
    epoch: u64,
}

impl PinTicket {
    pub(crate) fn new(reg: Arc<PinRegistry>, epoch: u64) -> Self {
        reg.register(epoch);
        Self { reg, epoch }
    }

    pub(crate) fn registry(&self) -> &Arc<PinRegistry> {
        &self.reg
    }
}

impl Clone for PinTicket {
    fn clone(&self) -> Self {
        Self::new(Arc::clone(&self.reg), self.epoch)
    }
}

impl Drop for PinTicket {
    fn drop(&mut self) {
        self.reg.release(self.epoch);
    }
}

/// Recycling state, reached only through an [`ArenaWriter`] (whose
/// construction contract makes access exclusive).
struct Recycler {
    /// Young batches: slots allocated and retired by the same guard, so
    /// unreachable from every committed snapshot — gate grace suffices.
    pending_young: VecDeque<PendingBatch>,
    /// Old batches: reachable from pre-guard snapshots; wait for gate
    /// grace AND all pins below the stamp. Stamps are nondecreasing, so
    /// both queues reclaim from the front (conditions are prefix-closed).
    pending_old: VecDeque<PendingBatch>,
    /// Cleared batches pooled for reuse.
    spare: Vec<PendingBatch>,
    /// Intrusive free-list heads; the link lives in the freed slot.
    free_chunk_head: u32,
    free_interior_head: u32,
    free_chunks: usize,
    free_interiors: usize,
    /// Slots dropped because `pending` hit its cap (garbage until compact).
    leaked: u64,
}

impl Recycler {
    fn new() -> Self {
        Self {
            // Sized for the unpinned steady state (a couple in flight);
            // growth under a long pin allocates on the retire path.
            pending_young: VecDeque::with_capacity(SPARE_BATCHES),
            pending_old: VecDeque::with_capacity(SPARE_BATCHES),
            spare: (0..SPARE_BATCHES).map(|_| PendingBatch::new()).collect(),
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
    /// Slots dropped because the pending queue hit its cap.
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
    /// Epochs pinned by live snapshots; shared with their tickets.
    pins: Arc<PinRegistry>,
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
            pins: Arc::new(PinRegistry::new()),
            cursors: CachePadded((AtomicUsize::new(0), AtomicUsize::new(capacity))),
            recycler: UnsafeCell::new(Recycler::new()),
        }
    }

    /// The pin registry snapshots register their epochs in.
    pub(crate) fn pins(&self) -> &Arc<PinRegistry> {
        &self.pins
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
    /// `RegionWriter` (from [`region`](Self::region)) or
    /// [`commit_regions`](Self::commit_regions) call may overlap its
    /// lifetime. In `aether-graph` the proof is `DynamicGraph::writer()`'s
    /// CAS guard: one `Writer` exists at a time and owns the handle.
    #[inline]
    pub unsafe fn writer(&self) -> ArenaWriter<'_> {
        // Bump cursors at guard start: a slot index at-or-past them was
        // allocated by this guard ("young"), so it was never reachable
        // from any committed snapshot. Recycled indices sit below them
        // and conservatively classify as old.
        let low = self.cursors.0.0.load(Ordering::Relaxed);
        let high = self.cursors.0.1.load(Ordering::Relaxed);
        ArenaWriter {
            arena: self,
            young_chunk_start: (low / CHUNK_SLOT) as u32,
            young_interior_start: ((self.capacity - high) / INTERIOR_SLOT) as u32,
        }
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

    /// Free every ready pending batch: young ones on gate grace, old ones
    /// on grace plus release of all pins below their stamp. Conditions
    /// are prefix-closed over each stamp-ordered queue, so each drains
    /// from the front to its first blocked batch.
    fn reclaim_ready(&self, rec: &mut Recycler) {
        let min_pinned = self.pins.min_pinned();
        loop {
            let ready_young = rec
                .pending_young
                .front()
                .is_some_and(|b| self.gate.grace_passed(&b.snap));
            let queue = if ready_young {
                &mut rec.pending_young
            } else {
                let ready_old = rec.pending_old.front().is_some_and(|b| {
                    b.stamp_epoch <= min_pinned && self.gate.grace_passed(&b.snap)
                });
                if !ready_old {
                    return;
                }
                &mut rec.pending_old
            };
            let mut batch = queue.pop_front().expect("front checked");
            for idx in batch.chunks.drain(..) {
                // SAFETY: the batch's conditions passed — the slot is
                // unobservable; `rec` is exclusive via `ArenaWriter`.
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
            if rec.spare.len() < SPARE_BATCHES {
                rec.spare.push(batch);
            }
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
    /// First bump slot index of each class allocated by this guard.
    young_chunk_start: u32,
    young_interior_start: u32,
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

    /// Stamp the retire log with a gate snapshot and `stamp_epoch` (the
    /// epoch the current guard commits), partition it into young/old
    /// batches, queue them, and fold any ready batches onto the free
    /// lists. Buffers come from a pool, so steady-state flushes never
    /// allocate; at the pending cap the slots are dropped instead
    /// (recycling never blocks the writer).
    ///
    /// # Safety
    /// Every logged slot must already be unreachable from every published
    /// root, and no slot may be logged twice.
    pub(crate) unsafe fn retire(&mut self, log: &mut RetireLog, stamp_epoch: u64) {
        if log.chunks.is_empty() && log.interiors.is_empty() {
            return;
        }
        let rec = self.recycler();
        self.arena.reclaim_ready(rec);
        if rec.pending_young.len() + rec.pending_old.len() >= MAX_PENDING_BATCHES {
            rec.leaked += (log.chunks.len() + log.interiors.len()) as u64;
            log.chunks.clear();
            log.interiors.clear();
            return;
        }
        // Split by guard-start cursor: young slots (allocated this guard)
        // free on gate grace alone; old ones also wait out older pins.
        let mut young = rec.spare.pop().unwrap_or_else(PendingBatch::new);
        let mut old = rec.spare.pop().unwrap_or_else(PendingBatch::new);
        for &idx in &log.chunks {
            if idx >= self.young_chunk_start {
                young.chunks.push(idx);
            } else {
                old.chunks.push(idx);
            }
        }
        for &idx in &log.interiors {
            if idx >= self.young_interior_start {
                young.interiors.push(idx);
            } else {
                old.interiors.push(idx);
            }
        }
        log.chunks.clear();
        log.interiors.clear();
        let mut snap = [0u64; GATE_STRIPES];
        self.arena.gate.snapshot(&mut snap);
        for (mut batch, queue) in [(young, &mut rec.pending_young), (old, &mut rec.pending_old)] {
            if batch.chunks.is_empty() && batch.interiors.is_empty() {
                rec.spare.push(batch);
            } else {
                batch.stamp_epoch = stamp_epoch;
                batch.snap = snap;
                queue.push_back(batch);
            }
        }
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
            .pending_young
            .iter()
            .chain(rec.pending_old.iter())
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
        unsafe { aw.retire(&mut log, 1) };
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
        unsafe { aw.retire(&mut log, 1) };
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
        unsafe { aw.retire(&mut log, 1) };
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
        unsafe { aw.retire(&mut log, 1) };
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
        unsafe { aw.retire(&mut log, 1) };
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
    fn pin_blocks_old_batches_until_release() {
        let arena = Arena::new(4096);
        // SAFETY: dropped before the second handle exists.
        let a = unsafe { arena.writer() }.alloc_chunk().unwrap();

        // `a` predates this guard, so it partitions as old.
        // SAFETY: the first handle is gone.
        let mut aw = unsafe { arena.writer() };
        let ticket = PinTicket::new(Arc::clone(arena.pins()), 5);
        let mut log = RetireLog::new();
        log.chunks.push(a);
        // SAFETY: the logged slot is test-local and never published.
        unsafe { aw.retire(&mut log, 10) };
        aw.reclaim();
        assert_eq!(
            aw.recycle_stats().free_chunks,
            0,
            "pin 5 must block stamp 10"
        );
        assert_eq!(aw.recycle_stats().pending, 1);
        drop(ticket);
        aw.reclaim();
        assert_eq!(aw.recycle_stats().free_chunks, 1);
    }

    #[test]
    fn pin_at_or_past_stamp_does_not_block() {
        let arena = Arena::new(4096);
        // SAFETY: dropped before the second handle exists.
        let a = unsafe { arena.writer() }.alloc_chunk().unwrap();
        // SAFETY: the first handle is gone.
        let mut aw = unsafe { arena.writer() };
        let _ticket = PinTicket::new(Arc::clone(arena.pins()), 10);
        let mut log = RetireLog::new();
        log.chunks.push(a);
        // SAFETY: the logged slot is test-local and never published.
        unsafe { aw.retire(&mut log, 10) };
        aw.reclaim();
        assert_eq!(aw.recycle_stats().free_chunks, 1);
    }

    #[test]
    fn young_slots_recycle_under_any_pin() {
        let arena = Arena::new(4096);
        // SAFETY: sole handle in a single-threaded test.
        let mut aw = unsafe { arena.writer() };
        let _ticket = PinTicket::new(Arc::clone(arena.pins()), 0);
        // Allocated by this guard: young, gate grace alone frees it.
        let b = aw.alloc_chunk().unwrap();
        let mut log = RetireLog::new();
        log.chunks.push(b);
        // SAFETY: the logged slot is test-local and never published.
        unsafe { aw.retire(&mut log, 10) };
        aw.reclaim();
        assert_eq!(aw.recycle_stats().free_chunks, 1);
    }

    #[test]
    fn min_pinned_tracks_registrations() {
        let reg = Arc::new(PinRegistry::new());
        assert_eq!(reg.min_pinned(), u64::MAX);
        let t5 = PinTicket::new(Arc::clone(&reg), 5);
        let t5b = t5.clone();
        let t7 = PinTicket::new(Arc::clone(&reg), 7);
        assert_eq!(reg.min_pinned(), 5);
        assert_eq!(reg.pin_count(), 3);
        drop(t5);
        assert_eq!(reg.min_pinned(), 5, "refcount holds the epoch");
        drop(t5b);
        assert_eq!(reg.min_pinned(), 7);
        drop(t7);
        assert_eq!(reg.min_pinned(), u64::MAX);
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
