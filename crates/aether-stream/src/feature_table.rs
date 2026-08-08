//! Seqlock-protected SoA feature table in HugePage RAM.
//!
//! Each node gets a compact slot with head and tail version counters
//! wrapping the feature vector. Writers bump head to odd, write features,
//! then set tail and head to the next even version.
//!
//! This head/tail layout is RDMA-safe under a two-snapshot protocol: a
//! remote reader takes two complete one-sided reads of the slot and accepts
//! a row only when both snapshots carry the same even version and identical
//! payload bytes (see the RDMA reader contract on [`FeatureTable::read_node`]).
//!
//! This table is backed by a separate `SharedMemoryRing` (not the UMEM).
//! When RDMA is enabled, the memory is registered with the HCA so GPU nodes
//! can do one-sided reads at <5μs without waking the CPU.

use aether_mem::hooks::MlockHook;
use aether_mem::{MemoryHook, SharedMemoryRing};
use std::sync::atomic::{AtomicU64, Ordering};

/// Feature offset within a slot (immediately after head_version u64).
const FEATURE_OFFSET: usize = 8;

/// Per-node slot layout:
/// ```text
/// [0..8]         head_version: AtomicU64  (odd=writing, even=ready, 0=uninit)
/// [8..8+N]       features: [f32; feature_dim]   (N = feature_dim * 4)
/// [8+N+P..16+N+P] tail_version: AtomicU64       (P = padding to 8-byte align)
/// ```
///
/// Logical slot size = 16 + N + P. There is no inter-field cache-line padding:
/// the head and tail counters sit directly around the feature payload so a
/// single one-sided RDMA READ of the slot fetches both version stamps. Note the
/// stored stride is NOT this compact size — `SharedMemoryRing` rounds each
/// slot up to a page boundary, so the per-slot stride (`schema.slot_size`) is
/// page-aligned; the gather reads only the live prefix of each slot (through
/// `tail_offset_in_slot + 8`).
pub struct FeatureTable {
    ring: SharedMemoryRing,
    node_count: usize,
    feature_dim: usize,
    /// Byte offset from slot start to the tail_version field.
    tail_offset: usize,
}

/// Schema metadata for RDMA advertisement.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeatureSchema {
    pub node_count: usize,
    pub feature_dim: usize,
    pub slot_size: usize,
    pub feature_offset_in_slot: usize,
    pub tail_offset_in_slot: usize,
}

/// RAII guard armed during the odd-head window of a seqlock write. If the
/// writer panics between `head→odd` and `head→even`, this guard's `Drop`
/// runs as the stack unwinds and forces head to an even target value,
/// releasing readers that would otherwise spin forever.
///
/// `disarmed = true` is set explicitly on the success path so the guard's
/// Drop is a no-op (the writer already wrote the final `target` value).
struct SeqlockWriteGuard<'a> {
    head: &'a AtomicU64,
    tail: &'a AtomicU64,
    target: u64,
    disarmed: bool,
}

impl Drop for SeqlockWriteGuard<'_> {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }
        // We're unwinding from a panic between the head→odd RMW and
        // head→even store. Force tail then head to `target` (even) — the
        // feature payload is logically abandoned (readers will see a
        // valid-version copy of whatever was in the slot last). Without
        // this, the odd head would trap every future reader in the
        // spin-on-odd loop.
        self.tail.store(self.target, Ordering::Release);
        self.head.store(self.target, Ordering::Release);
    }
}

/// Compute tail version offset for a given feature_dim.
///
/// tail_offset = 8 (head) + feature_dim * 4, rounded up to 8-byte alignment.
#[inline]
fn compute_tail_offset(feature_dim: usize) -> usize {
    let after_features = FEATURE_OFFSET + feature_dim * std::mem::size_of::<f32>();
    // Round up to 8-byte alignment for AtomicU64
    (after_features + 7) & !7
}

/// Compute raw slot size (before page-rounding by SharedMemoryRing).
#[inline]
fn compute_slot_size(feature_dim: usize) -> usize {
    compute_tail_offset(feature_dim) + 8 // tail_version is 8 bytes
}

/// Element-wise volatile copy of `len` f32s.
///
/// The seqlock payload is read while a writer may be storing into it (and
/// vice versa); volatile accesses make each element a real load/store the
/// compiler cannot elide, fuse, or invent under an exclusive-access
/// assumption, so the racing copy stays well-defined for race-checking
/// tools while the version checks arbitrate validity.
///
/// # Safety
/// `src` must be valid for `len` reads and `dst` for `len` writes.
#[inline]
unsafe fn volatile_copy_f32s(src: *const f32, dst: *mut f32, len: usize) {
    for i in 0..len {
        // SAFETY: `i < len`; the caller guarantees `src` covers `len` reads.
        let s = unsafe { src.add(i) };
        // SAFETY: `i < len`; the caller guarantees `dst` covers `len` writes.
        let d = unsafe { dst.add(i) };
        // SAFETY: `s` is in-bounds per above.
        let v = unsafe { s.read_volatile() };
        // SAFETY: `d` is in-bounds per above.
        unsafe { d.write_volatile(v) };
    }
}

impl FeatureTable {
    /// Allocate a new feature table.
    ///
    /// `node_count` is rounded up to the next power of two.
    /// `feature_dim` is the number of f32 features per node.
    ///
    /// Returns `None` if `node_count == 0`, `feature_dim == 0`, or the
    /// underlying allocation fails.
    pub fn new(
        node_count: usize,
        feature_dim: usize,
        extra_hooks: Vec<Box<dyn MemoryHook>>,
    ) -> Option<Self> {
        if node_count == 0 || feature_dim == 0 {
            return None;
        }

        let slot_count = node_count.next_power_of_two();
        let tail_offset = compute_tail_offset(feature_dim);
        let slot_size = compute_slot_size(feature_dim);

        let mut hooks: Vec<Box<dyn MemoryHook>> = vec![Box::new(MlockHook::new())];
        hooks.extend(extra_hooks);

        let (ring, hook_failures) = SharedMemoryRing::new(slot_count, slot_size, hooks).ok()?;
        for failure in &hook_failures {
            tracing::warn!(error = %failure, "feature table memory hook failed");
        }

        // We access slots positionally by node ID, not via the free list;
        // the ring is used for its allocation + hooks only.
        Some(Self {
            ring,
            node_count,
            feature_dim,
            tail_offset,
        })
    }

    /// Write features for a node. Head/tail seqlock protocol:
    ///
    /// 1. head → odd (signals "writer in progress")
    /// 2. copy features (non-atomic)
    /// 3. tail → next even
    /// 4. head → next even (matches tail)
    ///
    /// # Panic safety
    /// If the user-supplied `features` slice access — or any code path inside
    /// this function — panics between steps 1 and 4, an internal
    /// [`SeqlockWriteGuard`] still runs and forces head to an even value
    /// (`prev + 2`, abandoning the in-progress generation). Without this,
    /// any reader that observed the odd head would spin forever waiting for
    /// the write to complete.
    ///
    /// # Concurrency
    /// Writers to the *same* node serialize on the head CAS below: a second
    /// writer spins until the first restores an even head. Different nodes
    /// can be written concurrently without issue.
    pub fn write_node(&self, node: usize, features: &[f32]) {
        assert!(
            node < self.node_count,
            "node {node} out of range (node_count {})",
            self.node_count
        );
        assert_eq!(
            features.len(),
            self.feature_dim,
            "features slice length {} != feature_dim {}",
            features.len(),
            self.feature_dim
        );

        let base = self.slot_ptr(node);

        let head_ptr = base as *const AtomicU64;
        // SAFETY: tail_offset lies within the slot; `base` is bounds-checked.
        let tail_ptr = unsafe { base.add(self.tail_offset) } as *const AtomicU64;
        // SAFETY: `head_ptr` references the head AtomicU64 inside the slot.
        let head = unsafe { &*head_ptr };
        // SAFETY: `tail_ptr` references the tail AtomicU64 inside the slot.
        let tail = unsafe { &*tail_ptr };

        // Step 1: head even→odd via CAS. Same-node writers serialize here:
        // while another writer holds the head odd, spin until it releases.
        // AcqRel on success: Release orders prior writes before head goes
        // odd; Acquire prevents the feature copy below from being reordered
        // before this RMW.
        let mut prev = head.load(Ordering::Relaxed);
        let prev = loop {
            if prev & 1 != 0 {
                std::hint::spin_loop();
                prev = head.load(Ordering::Relaxed);
                continue;
            }
            match head.compare_exchange_weak(prev, prev + 1, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(p) => break p,
                Err(actual) => prev = actual,
            }
        };
        let target = prev + 2;

        // Arm the panic-recovery guard. From here through the explicit
        // disarm below, ANY panic restores head to `target` (even),
        // releasing waiting readers.
        let guard = SeqlockWriteGuard {
            head,
            tail,
            target,
            disarmed: false,
        };

        // Step 2: copy features (volatile, element-wise). Readers and the
        // remote HCA race this copy by design — the seqlock versions
        // arbitrate validity — so each element is a volatile store the
        // compiler cannot elide or widen under an exclusive-access
        // assumption.
        // SAFETY: FEATURE_OFFSET is within the slot.
        let feat_ptr = unsafe { base.add(FEATURE_OFFSET) } as *mut f32;
        // SAFETY: `feat_ptr` points to `feature_dim` f32 slots inside the slot;
        // `features` has the same length (asserted above).
        unsafe {
            volatile_copy_f32s(features.as_ptr(), feat_ptr, self.feature_dim);
        }

        // Step 3: tail → target.
        tail.store(target, Ordering::Release);

        // Step 4: head → target. Disarm the guard so its Drop is a
        // no-op on the success path.
        head.store(target, Ordering::Release);
        std::mem::forget(guard);
    }

    /// Read features for a node into `out`. Returns `true` if the node has been
    /// written at least once, `false` if uninitialized.
    ///
    /// Local readers MUST detect torn writes that complete during the feature
    /// copy. Naively just `head == tail` is insufficient — the writer's plain
    /// `Release` stores can sit in its CPU's store buffer, so the reader can
    /// observe both the old `tail` AND the old `head` even though the writer's
    /// `RMW` (head→odd) has already taken effect. The standard fix (used by
    /// the Linux kernel seqlock) is to RE-LOAD HEAD after the data copy:
    /// because the writer's first action is a LOCK-prefixed RMW that drains
    /// its store buffer and is globally visible immediately, any writer that
    /// touched the slot during our copy is guaranteed to leave a head value
    /// different from the snapshot we started with.
    ///
    /// We keep the `head == tail` check too — the RDMA path relies on the
    /// same version comparison within each of its slot snapshots, and it
    /// provides a cheap early-out here.
    ///
    /// # RDMA reader contract (two-snapshot validation)
    /// A remote HCA cannot re-load head after its payload copy the way the
    /// local loop below does, and it can assume nothing about the order in
    /// which the bytes of one READ complete: a single slot READ may be split
    /// into multiple PCIe transactions whose completions land in any order,
    /// so within one snapshot a stale version pair can accompany fresher
    /// payload bytes (or vice versa). No single-snapshot version comparison
    /// is sound under that model. Remote readers therefore take TWO complete
    /// snapshots of the slot — the second issued only after the first has
    /// fully completed — and accept a row iff:
    ///   - `snap1.head == snap1.tail == snap2.head == snap2.tail`,
    ///   - the common version is even and nonzero, and
    ///   - the payload bytes of the two snapshots are identical.
    ///
    /// Soundness rests on three properties: (1) the writer's head→odd
    /// transition is a LOCK-prefixed RMW, globally ordered across the
    /// coherence fabric (which the HCA's DMA reads participate in), so it is
    /// visible to every coherent observer before any of the writer's payload
    /// stores; (2) per-location visibility is monotone — once a snapshot has
    /// observed a value at a location, a later snapshot observes that value
    /// or a newer one; (3) snapshot 2 begins strictly after snapshot 1 ends.
    /// A writer whose payload stores land during either snapshot leaves
    /// either mismatched versions across the snapshots or a payload byte
    /// that differs between them; a writer stalled mid-payload across both
    /// snapshots leaves an odd version in snapshot 2. No assumption about
    /// intra-READ completion ordering is required.
    pub fn read_node(&self, node: usize, out: &mut [f32]) -> bool {
        assert!(
            node < self.node_count,
            "node {node} out of range (node_count {})",
            self.node_count
        );
        assert!(
            out.len() >= self.feature_dim,
            "out slice length {} < feature_dim {}",
            out.len(),
            self.feature_dim
        );

        let base = self.slot_ptr(node);

        // SAFETY: `base` points to head AtomicU64 at offset 0 of the slot.
        let head = unsafe { &*(base as *const AtomicU64) };
        // SAFETY: `tail_offset` lies within the slot.
        let tail_ptr = unsafe { base.add(self.tail_offset) } as *const AtomicU64;
        // SAFETY: `tail_ptr` references the tail AtomicU64 inside the slot.
        let tail = unsafe { &*tail_ptr };

        loop {
            let h1 = head.load(Ordering::Acquire);
            if h1 == 0 {
                return false; // never written
            }
            if h1 & 1 != 0 {
                std::hint::spin_loop();
                continue; // writer in progress
            }

            // Read features (volatile, element-wise) — a writer may be
            // storing into the payload concurrently; the version checks
            // below arbitrate validity.
            // SAFETY: FEATURE_OFFSET is within the slot.
            let feat_ptr = unsafe { base.add(FEATURE_OFFSET) } as *const f32;
            // SAFETY: `feat_ptr` covers `feature_dim` f32s; `out` has
            // `>= feature_dim` slots (asserted above).
            unsafe {
                volatile_copy_f32s(feat_ptr, out.as_mut_ptr(), self.feature_dim);
            }

            // Ensure all feature reads complete before we read tail/head.
            // Without this fence, ARM/RISC-V can reorder the non-atomic
            // feature reads past the version loads.
            std::sync::atomic::fence(Ordering::Acquire);

            // Two checks:
            //   1. h1 == t  — RDMA-compatible, catches torn writes whose
            //      tail.store has propagated.
            //   2. h1 == h2 — local-only, catches writers that started a
            //      new generation during our copy. Since the writer's
            //      first op is a LOCK-RMW that's immediately globally
            //      visible, h2 cannot equal the old even h1 if a writer
            //      ran concurrently.
            let t = tail.load(Ordering::Acquire);
            let h2 = head.load(Ordering::Acquire);
            if h1 == t && h1 == h2 {
                return true;
            }
            std::hint::spin_loop();
        }
    }

    /// Feature dimension.
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Number of nodes this table can hold (original, before rounding to power-of-two).
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Schema for RDMA advertisement.
    pub fn schema(&self) -> FeatureSchema {
        FeatureSchema {
            node_count: self.node_count,
            feature_dim: self.feature_dim,
            slot_size: self.ring.slot_size(),
            feature_offset_in_slot: FEATURE_OFFSET,
            tail_offset_in_slot: self.tail_offset,
        }
    }

    /// Base address of the table (for RDMA registration).
    pub fn base_addr(&self) -> u64 {
        self.ring.base_addr() as u64
    }

    /// Total allocated size.
    pub fn total_size(&self) -> usize {
        self.ring.total_size()
    }

    /// Raw pointer to a node's slot.
    #[inline]
    fn slot_ptr(&self, node: usize) -> *mut u8 {
        self.ring.slot_ptr_for_ffi(node)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read() {
        let table = FeatureTable::new(4, 8, vec![]).unwrap();
        let features = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        table.write_node(0, &features);

        let mut out = vec![0.0f32; 8];
        assert!(table.read_node(0, &mut out));
        assert_eq!(out, features);
    }

    #[test]
    fn seqlock_recovers_from_panic_mid_write() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        use std::sync::Arc;

        // Wrap the table in Arc so we can share it between the panicking
        // closure and the post-panic reader.
        let table = Arc::new(FeatureTable::new(4, 4, vec![]).unwrap());
        let baseline = vec![1.0_f32, 2.0, 3.0, 4.0];
        table.write_node(0, &baseline);

        // Simulate a panic mid-write by handing `write_node` a feature
        // slice from a struct whose Drop panics during the copy. We can't
        // inject a panic *inside* the copy directly without exposing
        // internals, so instead we use the simpler model: take a
        // SeqlockWriteGuard manually, then drop it without disarming.
        let table_for_panic = Arc::clone(&table);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let base = table_for_panic.slot_ptr(0);
            // SAFETY: same slot layout as write_node uses.
            let head = unsafe { &*(base as *const AtomicU64) };
            // SAFETY: tail_offset is within the slot.
            let tail_ptr = unsafe { base.add(table_for_panic.tail_offset) } as *const AtomicU64;
            // SAFETY: `tail_ptr` refs the tail AtomicU64.
            let tail = unsafe { &*tail_ptr };
            {
                let prev = head.fetch_add(1, Ordering::AcqRel);
                let target = prev + 2;
                // Arm the guard but don't disarm — `panic!` below triggers
                // the guard's Drop, which must restore head to `target`.
                let _guard = SeqlockWriteGuard {
                    head,
                    tail,
                    target,
                    disarmed: false,
                };
                panic!("simulated mid-write panic");
            }
        }));
        assert!(result.is_err(), "the simulated panic should propagate");

        // After unwinding, the guard's Drop should have left head at the
        // post-write even value. A reader must NOT spin forever.
        let mut out = vec![0.0_f32; 4];
        let got = table.read_node(0, &mut out);
        assert!(got, "reader must complete (not spin) after writer panic");
    }

    #[test]
    fn uninitialized_returns_false() {
        let table = FeatureTable::new(4, 8, vec![]).unwrap();
        let mut out = vec![0.0f32; 8];
        assert!(!table.read_node(0, &mut out));
    }

    #[test]
    fn multiple_nodes() {
        let table = FeatureTable::new(4, 4, vec![]).unwrap();

        let f0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let f1 = vec![5.0f32, 6.0, 7.0, 8.0];
        table.write_node(0, &f0);
        table.write_node(1, &f1);

        let mut out = vec![0.0f32; 4];
        assert!(table.read_node(0, &mut out));
        assert_eq!(out, f0);

        assert!(table.read_node(1, &mut out));
        assert_eq!(out, f1);
    }

    #[test]
    fn overwrite() {
        let table = FeatureTable::new(4, 4, vec![]).unwrap();

        let f1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let f2 = vec![10.0f32, 20.0, 30.0, 40.0];
        table.write_node(0, &f1);
        table.write_node(0, &f2);

        let mut out = vec![0.0f32; 4];
        assert!(table.read_node(0, &mut out));
        assert_eq!(out, f2);
    }

    #[test]
    fn schema_correct() {
        let table = FeatureTable::new(100, 768, vec![]).unwrap();
        let schema = table.schema();
        assert_eq!(schema.node_count, 100);
        assert_eq!(schema.feature_dim, 768);
        assert_eq!(schema.feature_offset_in_slot, 8);
        // 768 * 4 = 3072, 8 + 3072 = 3080, aligned to 8 = 3080
        assert_eq!(schema.tail_offset_in_slot, 3080);
    }

    #[test]
    fn tail_offset_alignment() {
        // Even feature_dim (common for GNN): N is divisible by 8, no padding
        assert_eq!(compute_tail_offset(128), 8 + 128 * 4); // 520
        assert_eq!(compute_tail_offset(256), 8 + 256 * 4); // 1032
        assert_eq!(compute_tail_offset(512), 8 + 512 * 4); // 2056
        assert_eq!(compute_tail_offset(768), 8 + 768 * 4); // 3080

        // Odd feature_dim: needs padding to 8-byte align
        // feature_dim=3: after_features = 8 + 12 = 20, round to 24
        assert_eq!(compute_tail_offset(3), 24);
        // feature_dim=5: after_features = 8 + 20 = 28, round to 32
        assert_eq!(compute_tail_offset(5), 32);
    }

    #[test]
    fn slot_size_computation() {
        // feature_dim=4: tail_offset = 8 + 16 = 24, slot_size = 24 + 8 = 32
        assert_eq!(compute_slot_size(4), 32);
        // feature_dim=768: tail_offset = 3080, slot_size = 3088
        assert_eq!(compute_slot_size(768), 3088);
    }

    #[test]
    fn head_tail_versions_match_after_write() {
        let table = FeatureTable::new(4, 4, vec![]).unwrap();
        let features = vec![1.0f32, 2.0, 3.0, 4.0];

        table.write_node(0, &features);

        // Verify head and tail are both 2 (first even version after write)
        let base = table.slot_ptr(0);
        // SAFETY: `base` points to head AtomicU64 at offset 0.
        let head = unsafe { &*(base as *const AtomicU64) };
        // SAFETY: tail_offset is within the slot.
        let tail_ptr = unsafe { base.add(table.tail_offset) } as *const AtomicU64;
        // SAFETY: `tail_ptr` refs the tail AtomicU64.
        let tail = unsafe { &*tail_ptr };
        assert_eq!(head.load(Ordering::Relaxed), 2);
        assert_eq!(tail.load(Ordering::Relaxed), 2);

        // Write again — should be 4
        table.write_node(0, &features);
        assert_eq!(head.load(Ordering::Relaxed), 4);
        assert_eq!(tail.load(Ordering::Relaxed), 4);
    }
}
