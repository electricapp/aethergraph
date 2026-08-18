//! The sampler's per-call output buffers must not climb a reallocation ladder.
//!
//! `sample_neighbors` hands its filled node and edge buffers to the caller and
//! swaps empty ones back in, so every call starts those arrays from scratch.
//! If the replacements come back at a fixed small capacity, each call regrows
//! them by doubling — and a `realloc` that cannot extend in place copies the
//! whole array. At three hops that is hundreds of thousands of elements copied
//! several times over, across five parallel edge arrays.
//!
//! Wall-clock timing of that effect is drowned out by machine noise on a busy
//! host, so this pins the property directly: with the sampler reused across
//! same-shaped batches, a steady-state call performs no output-buffer regrowth
//! at all.

use aethergraph_core::{Graph, NeighborSampler, NodeId, SamplingConfig};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static REALLOCS: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);

/// Passes everything through to the system allocator, counting `realloc`
/// calls while armed. `Vec::with_capacity` goes to `alloc`, and only growth
/// beyond a vector's capacity reaches `realloc`, so this counter isolates the
/// regrowth ladder from ordinary allocation.
struct CountingAlloc;

// SAFETY: every method forwards its arguments unchanged to `System`, which
// upholds the `GlobalAlloc` contract; the counter is a `Relaxed` side effect
// that touches no allocator state.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is forwarded unchanged from the caller.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` are forwarded unchanged from the caller.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            REALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: all three arguments are forwarded unchanged from the caller.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

const NUM_NODES: usize = 200_000;
const AVG_DEGREE: usize = 12;

fn build_graph() -> Graph {
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut src: Vec<NodeId> = Vec::new();
    let mut dst: Vec<NodeId> = Vec::new();
    for u in 0..NUM_NODES as NodeId {
        let degree = (next() as usize % AVG_DEGREE) + 1;
        for _ in 0..degree {
            src.push(u);
            dst.push((next() % NUM_NODES as u64) as NodeId);
        }
    }
    Graph::from_src_dst(NUM_NODES, &src, &dst, None).expect("csr build")
}

/// A sampler reused across same-shaped batches reaches a steady state where no
/// output buffer needs to grow at all. Sizing each replacement at an exact fit
/// would not achieve that: sampling is randomized, so a call that produces one
/// more element than the last would pay a full-array copy, on roughly half of
/// all calls. The reserved headroom is what drives this to zero.
#[test]
fn steady_state_sampling_does_not_regrow_output_buffers() {
    let graph = build_graph();
    let config = SamplingConfig {
        fanout: vec![15, 10],
        replace: false,
        seed: Some(42),
        ..Default::default()
    };
    let mut sampler = NeighborSampler::new(&graph, config);

    let seeds: Vec<NodeId> = (0..1024).map(|i| (i * 97) % NUM_NODES as NodeId).collect();

    // Warm up outside the measured window. The output buffers are sized from
    // the previous call and settle after one, but the internal scratch the
    // sampler keeps across calls (the accumulating frontier) only reaches its
    // high-water mark after a few batches have varied around it.
    let warm = sampler.sample_neighbors(&seeds);
    assert!(
        warm.num_edges() > 50_000,
        "batch too small to exercise regrowth: {} edges",
        warm.num_edges(),
    );
    for _ in 0..8 {
        let _ = sampler.sample_neighbors(&seeds);
    }

    COUNTING.store(true, Ordering::Relaxed);
    REALLOCS.store(0, Ordering::Relaxed);
    let subgraph = sampler.sample_neighbors(&seeds);
    COUNTING.store(false, Ordering::Relaxed);

    let reallocs = REALLOCS.load(Ordering::Relaxed);
    assert!(
        subgraph.num_edges() > 50_000,
        "steady-state batch shrank unexpectedly: {} edges",
        subgraph.num_edges(),
    );
    assert_eq!(
        reallocs, 0,
        "steady-state sampling regrew output buffers {reallocs} times; the \
         per-call buffers are not being sized from the previous call plus \
         headroom",
    );
}
