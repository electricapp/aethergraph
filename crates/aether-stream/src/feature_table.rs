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
//! This table is backed by its own [`SlotRegion`] (not the UMEM), addressed by
//! node ID.
//! When RDMA is enabled, the memory is registered with the HCA so GPU nodes
//! can do one-sided reads at <5μs without waking the CPU.

use aether_mem::hooks::{MlockHook, NumaInterleaveHook};
use aether_mem::{MemoryHook, SlotRegion};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};

/// Feature offset within a slot: `u64` head plus 8 bytes of pad. Owned by
/// `aethergraph-core` so host, wire, and device readers share one geometry.
const FEATURE_OFFSET: usize = aethergraph_core::FEATURE_SLOT_HEAD_BYTES;

/// Per-node slot layout:
/// ```text
/// [0..8]           head_version: AtomicU64 (odd=writing, even=ready, 0=uninit)
/// [8..16]          pad
/// [16..16+N]       features: [f32; feature_dim]   (N = feature_dim * 4)
/// [16+N+P..24+N+P] tail_version: AtomicU64        (P = padding to 8-byte align)
/// ```
///
/// Logical slot size = 24 + N + P. The pad after the head keeps the payload
/// 16-byte aligned under the 64-byte stride below, which is what the device
/// gather's `ld.global.cs.v4.f32` requires; the stride rounding absorbs it for
/// every dim not already a multiple of 64, so it costs no DRAM.
///
/// There is no inter-field cache-line padding: the head and tail counters sit
/// directly around the feature payload so a single one-sided RDMA READ of the
/// slot fetches both version stamps. The stored stride (`schema.slot_size`)
/// rounds the compact size up to a 64-byte cache line — the table registers
/// one MR and addresses slots by offset, so page-strided slots would only
/// waste DRAM on dead padding. The gather reads only the live prefix of each
/// slot (through `tail_offset_in_slot + 8`).
pub struct FeatureTable {
    region: SlotRegion,
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
/// poisons the slot to version 0 (even, uninitialized), releasing readers
/// that would otherwise spin forever without publishing torn payload as a
/// successful generation.
///
/// `disarmed = true` is set explicitly on the success path so the guard's
/// Drop is a no-op (the writer already wrote the final even version).
struct SeqlockWriteGuard<'a> {
    head: &'a AtomicU64,
    tail: &'a AtomicU64,
    disarmed: bool,
}

impl Drop for SeqlockWriteGuard<'_> {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }
        // We're unwinding from a panic between the head→odd RMW and
        // head→even store. Poison the slot to version 0 (uninitialized):
        // an even head releases readers from the spin-on-odd loop, but we
        // must not publish the intended next version — that would mark a
        // torn payload as a successful new generation. Version 0 makes
        // `read_node` return false until a later successful write.
        self.tail.store(0, Ordering::Release);
        self.head.store(0, Ordering::Release);
    }
}

/// Compute tail version offset for a given feature_dim.
///
/// tail_offset = `FEATURE_OFFSET` + feature_dim * 4, rounded up to 8-byte
/// alignment for the `AtomicU64`.
#[inline]
fn compute_tail_offset(feature_dim: usize) -> usize {
    aethergraph_core::feature_slot_tail_offset(feature_dim)
}

/// Compute raw slot size (before stride-rounding by the region).
#[inline]
fn compute_slot_size(feature_dim: usize) -> usize {
    aethergraph_core::feature_slot_size(feature_dim)
}

// The u64-packed volatile copies below assemble two adjacent f32s into one
// word assuming the first element occupies the low half — little-endian
// layout, like every other wire structure in this crate.
#[cfg(target_endian = "big")]
compile_error!("feature_table's packed seqlock copies assume a little-endian target");

/// Volatile store of an owned f32 slice into a slot payload.
///
/// Only the slot side of the copy races (readers and the remote HCA pull
/// it mid-write by design; the version checks arbitrate validity), so only
/// the slot access is volatile — the source slice is exclusively owned.
/// Pairs move as single 8-byte volatile stores: `FEATURE_OFFSET` is a
/// multiple of 8 inside an aligned slot, so `dst` is 8-aligned, and
/// halving the volatile-op count roughly halves the copy cost the
/// element-wise version paid (volatile forbids the compiler from widening
/// it).
///
/// # Safety
/// `dst` must be valid for `src.len()` f32 writes and 8-byte aligned.
#[inline]
unsafe fn volatile_store_payload(src: &[f32], dst: *mut f32) {
    let pairs = src.len() / 2;
    let dst64 = dst as *mut u64;
    for i in 0..pairs {
        let lo = src[2 * i].to_bits() as u64;
        let hi = src[2 * i + 1].to_bits() as u64;
        // SAFETY: `i < pairs`, so the word lies within the payload.
        let word = unsafe { dst64.add(i) };
        // SAFETY: `dst` is valid and 8-aligned per the contract.
        unsafe { word.write_volatile(lo | (hi << 32)) };
    }
    if src.len() % 2 == 1 {
        let last = src.len() - 1;
        // SAFETY: `last < src.len()`, within the payload.
        let tail = unsafe { dst.add(last) } as *mut u32;
        // SAFETY: a 4-byte access needs only 4-byte alignment, which every
        // f32 lane of an 8-aligned payload has.
        unsafe { tail.write_volatile(src[last].to_bits()) };
    }
}

/// Volatile load of a slot payload into an owned f32 slice. Mirror of
/// [`volatile_store_payload`]: volatile on the racy slot side only, 8 bytes
/// per operation.
///
/// # Safety
/// `src` must be valid for `dst.len()` f32 reads and 8-byte aligned.
#[inline]
unsafe fn volatile_load_payload(src: *const f32, dst: &mut [f32]) {
    let src64 = src as *const u64;
    // `as_chunks_mut` carries the pair length in the type, so the two lane
    // stores need no bounds check and the tail is what remains.
    for (i, lanes) in dst.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        // SAFETY: the chunk iterator yields one pair per 8-byte word of the
        // payload, so `i` stays within it.
        let p = unsafe { src64.add(i) };
        // SAFETY: `src` is valid and 8-aligned per the contract.
        let word = unsafe { p.read_volatile() };
        lanes[0] = f32::from_bits(word as u32);
        lanes[1] = f32::from_bits((word >> 32) as u32);
    }
    if dst.len() % 2 == 1 {
        let last = dst.len() - 1;
        // SAFETY: `last < dst.len()`, within the payload.
        let tail = unsafe { src.add(last) } as *const u32;
        // SAFETY: a 4-byte access needs only 4-byte alignment, which every
        // f32 lane of an 8-aligned payload has.
        let bits = unsafe { tail.read_volatile() };
        dst[last] = f32::from_bits(bits);
    }
}

impl FeatureTable {
    /// Allocate a new feature table of exactly `node_count` slots.
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

        let tail_offset = compute_tail_offset(feature_dim);
        let slot_size = compute_slot_size(feature_dim);

        // Interleave first, then lock: the table is read by every worker,
        // so spreading pages across memory controllers beats piling them
        // on the allocating thread's node (single-node machines no-op).
        // `extra_hooks` run last, so a caller-supplied placement hook
        // (e.g. `NumaBindHook` on the NIC's node for a DMA-served table)
        // overrides the interleave.
        let mut hooks: Vec<Box<dyn MemoryHook>> = vec![
            Box::new(NumaInterleaveHook::all_nodes()),
            Box::new(MlockHook::new()),
        ];
        hooks.extend(extra_hooks);

        // Slots pack at cache-line stride, not page stride: the table
        // registers one MR and addresses slots by offset, so per-slot page
        // alignment would only round a 3096-byte dim-768 slot up to 4096 —
        // ~25% of table DRAM spent on dead padding.
        let (region, hook_failures) = SlotRegion::new(
            node_count,
            slot_size,
            NonZeroUsize::new(aethergraph_core::FEATURE_SLOT_STRIDE_ALIGN),
            hooks,
        )
        .ok()?;
        for failure in &hook_failures {
            tracing::warn!(error = %failure, "feature table memory hook failed");
        }

        Some(Self {
            region,
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
    /// `SeqlockWriteGuard` poisons the slot to version 0 (even, uninitialized).
    /// That releases any reader spinning on the odd head without publishing
    /// a torn payload as a successful new generation. Without the guard,
    /// readers that observed the odd head would spin forever.
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
        // odd; Acquire orders this writer after the previous one's release.
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
        // A release RMW orders only what precedes it: nothing keeps the
        // payload stores below from becoming visible before the odd head.
        // This fence does, for CPUs (pairing with the reader's acquire
        // fence) and for DMA readers alike.
        aether_mem::dma_release_fence();

        // Arm the panic-recovery guard. From here through the explicit
        // disarm below, ANY panic poisons the slot to version 0 (even),
        // releasing waiting readers without publishing torn data.
        let guard = SeqlockWriteGuard {
            head,
            tail,
            disarmed: false,
        };

        // Step 2: copy features (volatile on the slot side, 8 bytes per
        // op). Readers and the remote HCA race this copy by design — the
        // seqlock versions arbitrate validity.
        // SAFETY: FEATURE_OFFSET is within the slot.
        let feat_ptr = unsafe { base.add(FEATURE_OFFSET) } as *mut f32;
        // SAFETY: `feat_ptr` points to `feature_dim` f32 slots inside the
        // slot and is 8-aligned (FEATURE_OFFSET is a multiple of 8 inside
        // an aligned slot); `features` has the same length (asserted above).
        unsafe {
            volatile_store_payload(features, feat_ptr);
        }

        // Step 3: tail → target. The fence extends the Release ordering of
        // payload-before-version to DMA readers.
        aether_mem::dma_release_fence();
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
    /// copy. Naively just `head == tail` is insufficient — a writer that
    /// starts after the reader's version loads leaves both versions
    /// unchanged while it rewrites the payload under the copy. The standard
    /// fix (used by the Linux kernel seqlock) is to RE-LOAD HEAD after the
    /// data copy: the writer fences between its head→odd RMW and its
    /// payload stores, and the reader fences between its payload loads and
    /// the re-load, so a copy that read any byte of a concurrent write sees
    /// that write's odd head (or a later version) on the re-load.
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
    /// Soundness rests on three properties: (1) the writer fences with
    /// [`aether_mem::dma_release_fence`] between its head→odd RMW and its
    /// payload stores, and again between the payload and the version
    /// stores, so every coherent observer — the HCA's DMA reads included —
    /// sees the odd head before any of the write's payload bytes and the
    /// payload before the new even versions; (2) per-location visibility is monotone — once a snapshot has
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

            // Read features (volatile on the slot side, 8 bytes per op) —
            // a writer may be storing into the payload concurrently; the
            // version checks below arbitrate validity.
            // SAFETY: FEATURE_OFFSET is within the slot.
            let feat_ptr = unsafe { base.add(FEATURE_OFFSET) } as *const f32;
            // SAFETY: `feat_ptr` covers `feature_dim` f32s and is 8-aligned
            // (FEATURE_OFFSET is a multiple of 8 inside an aligned slot);
            // `out` has `>= feature_dim` slots (asserted above).
            unsafe {
                volatile_load_payload(feat_ptr, &mut out[..self.feature_dim]);
            }

            // Ensure all feature reads complete before we read tail/head.
            // Without this fence, ARM/RISC-V can reorder the non-atomic
            // feature reads past the version loads.
            std::sync::atomic::fence(Ordering::Acquire);

            // Two checks:
            //   1. h1 == t  — RDMA-compatible, catches torn writes whose
            //      tail.store has propagated.
            //   2. h1 == h2 — local-only, catches writers that started a
            //      new generation during our copy: if the copy read any of
            //      their payload, the acquire fence above synchronizes
            //      with the writer's fence after its head→odd RMW, so h2
            //      cannot be the old even h1.
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

    /// Number of nodes this table holds.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Schema for RDMA advertisement.
    pub fn schema(&self) -> FeatureSchema {
        FeatureSchema {
            node_count: self.node_count,
            feature_dim: self.feature_dim,
            slot_size: self.region.slot_size(),
            feature_offset_in_slot: FEATURE_OFFSET,
            tail_offset_in_slot: self.tail_offset,
        }
    }

    /// Base address of the table (for RDMA registration).
    pub fn base_addr(&self) -> u64 {
        self.region.base_addr() as u64
    }

    /// Total allocated size.
    pub fn total_size(&self) -> usize {
        self.region.total_size()
    }

    /// Raw pointer to a node's slot.
    #[inline]
    fn slot_ptr(&self, node: usize) -> *mut u8 {
        self.region.slot_ptr(node)
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
                let _prev = head.fetch_add(1, Ordering::AcqRel);
                // Arm the guard but don't disarm — `panic!` below triggers
                // the guard's Drop, which must poison the slot to version 0.
                let _guard = SeqlockWriteGuard {
                    head,
                    tail,
                    disarmed: false,
                };
                panic!("simulated mid-write panic");
            }
        }));
        assert!(result.is_err(), "the simulated panic should propagate");

        // After unwinding, the guard's Drop poisons the slot. A reader must
        // NOT spin forever, and must NOT treat the torn payload as valid.
        let mut out = vec![0.0_f32; 4];
        let got = table.read_node(0, &mut out);
        assert!(
            !got,
            "poisoned slot must read as uninitialized, not torn-valid"
        );
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
        assert_eq!(schema.feature_offset_in_slot, 16);
        // 768 * 4 = 3072, 16 + 3072 = 3088, aligned to 8 = 3088
        assert_eq!(schema.tail_offset_in_slot, 3088);
    }

    #[test]
    fn tail_offset_alignment() {
        // Even feature_dim (common for GNN): N is divisible by 8, no padding
        assert_eq!(compute_tail_offset(128), 16 + 128 * 4); // 528
        assert_eq!(compute_tail_offset(256), 16 + 256 * 4); // 1040
        assert_eq!(compute_tail_offset(512), 16 + 512 * 4); // 2064
        assert_eq!(compute_tail_offset(768), 16 + 768 * 4); // 3088

        // Odd feature_dim: needs padding to 8-byte align
        // feature_dim=3: after_features = 16 + 12 = 28, round to 32
        assert_eq!(compute_tail_offset(3), 32);
        // feature_dim=5: after_features = 16 + 20 = 36, round to 40
        assert_eq!(compute_tail_offset(5), 40);
    }

    #[test]
    fn common_cuh_matches_core_layout() {
        // Nothing but this test connects the device constants to the host
        // ones. Ungated on purpose: drift fails the default Linux job, not
        // only a run on a box with a GPU.
        let cuh = include_str!("gpu/kernels/common.cuh");
        let define = |name: &str| -> usize {
            let needle = format!("#define {name} ");
            cuh.lines()
                .find_map(|l| l.trim().strip_prefix(&needle))
                .unwrap_or_else(|| panic!("common.cuh must define {name}"))
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} is an integer"))
        };
        assert_eq!(
            define("AETHER_FEATURE_OFFSET"),
            FEATURE_OFFSET,
            "drifted from aethergraph_core::FEATURE_SLOT_HEAD_BYTES"
        );
        assert_eq!(
            define("AETHER_QUANT_BLOCK"),
            aethergraph_core::BlockScaledI8::BLOCK,
            "drifted from aethergraph_core::BlockScaledI8::BLOCK"
        );
    }

    #[test]
    fn payload_base_is_sixteen_byte_aligned() {
        // What licenses `ld.global.cs.v4.f32`: a 64-byte stride plus a
        // 16-byte head keeps every payload base 16-aligned, at every dim.
        for dim in [1usize, 3, 4, 5, 128, 768] {
            let table = FeatureTable::new(8, dim, vec![]).unwrap();
            let schema = table.schema();
            assert_eq!(schema.slot_size % 16, 0, "stride for dim {dim}");
            assert_eq!(
                schema.feature_offset_in_slot % 16,
                0,
                "offset for dim {dim}"
            );
            assert_eq!(table.base_addr() % 16, 0, "base for dim {dim}");
        }
    }

    #[test]
    fn slot_size_computation() {
        // feature_dim=4: tail_offset = 16 + 16 = 32, slot_size = 32 + 8 = 40
        assert_eq!(compute_slot_size(4), 40);
        // feature_dim=768: tail_offset = 3088, slot_size = 3096
        assert_eq!(compute_slot_size(768), 3096);
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
