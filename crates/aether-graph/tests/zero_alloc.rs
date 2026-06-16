//! Allocation-regression guards for the dynamic-graph hot paths.
//!
//! The C-tree promises a zero-allocation read path and arena-only (heap-free)
//! edge inserts. Those are claims, so they are tests: a counting global
//! allocator records every heap allocation made while a measured region is
//! "armed", and each region asserts the count is exactly zero. A future change
//! that reintroduces a hidden `Vec`/`Box`/`format!` on these paths fails CI
//! instead of silently eroding the property.
//!
//! Both the "armed" flag and the counter are thread-local, so the tests in
//! this binary stay independent under parallel execution, and the allocator
//! lives in this test binary only — production builds and the library's own
//! unit tests are untouched.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use aether_graph::DynamicGraph;

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static HEAP_OPS: Cell<usize> = const { Cell::new(0) };
}

/// Count one heap operation against the current thread when it is armed.
/// Const-initialized thread-locals never allocate on first access, so this is
/// safe to call from inside the global allocator without re-entry.
fn record() {
    if ARMED.with(Cell::get) {
        HEAP_OPS.with(|c| c.set(c.get() + 1));
    }
}

struct CountingAllocator;

// SAFETY: every method forwards verbatim to the system allocator with the
// same arguments, so all `GlobalAlloc` invariants are preserved; the only
// added behavior is a thread-local counter bump, which has no bearing on
// allocation soundness.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        // SAFETY: `layout` is forwarded unchanged from the caller.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr`/`layout` come straight from the caller, which obtained
        // `ptr` from this allocator with the same `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record();
        // SAFETY: arguments are forwarded unchanged from the caller.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Run `f` with allocation counting armed and assert it allocated nothing.
fn assert_no_alloc<R>(what: &str, f: impl FnOnce() -> R) -> R {
    let start = HEAP_OPS.with(Cell::get);
    ARMED.with(|a| a.set(true));
    let out = f();
    ARMED.with(|a| a.set(false));
    let observed = HEAP_OPS.with(Cell::get) - start;
    assert_eq!(
        observed, 0,
        "{what} performed {observed} heap allocation(s)"
    );
    out
}

/// The guard itself must observe allocations, otherwise the other tests are
/// vacuously green.
#[test]
fn guard_detects_allocations() {
    let start = HEAP_OPS.with(Cell::get);
    ARMED.with(|a| a.set(true));
    let v: Vec<u32> = (0..32).collect();
    ARMED.with(|a| a.set(false));
    assert!(HEAP_OPS.with(Cell::get) > start);
    assert_eq!(v.len(), 32);
}

/// `neighbors_into` reuses the caller's buffer and walks the tree on the
/// stack — repeated reads must not touch the heap.
#[test]
fn neighbor_reads_are_allocation_free() {
    let g = DynamicGraph::new(256, 1 << 20);
    {
        let mut w = g.writer().unwrap();
        for dst in 0..200u32 {
            w.insert_edge(5, dst).unwrap();
        }
    }

    let mut buf = Vec::new();
    // Warm the buffer to its steady-state capacity (this read is allowed to
    // allocate; the measured reads below are not).
    g.neighbors_into(5, &mut buf);
    assert_eq!(buf.len(), 200);

    assert_no_alloc("neighbors_into", || {
        for _ in 0..1_000 {
            g.neighbors_into(5, &mut buf);
        }
    });
}

/// Steady-state inserts path-copy into the bump arena and reuse the writer's
/// rebalance scratch — after warm-up they must not allocate on the heap.
#[test]
fn steady_state_inserts_touch_only_the_arena() {
    let g = DynamicGraph::new(2048, 64 << 20);
    let mut w = g.writer().unwrap();

    // Warm-up: drive one vertex to a high degree so the scapegoat rebuilds
    // grow the writer's rebalance-scratch buffer to its high-water mark.
    for dst in 0..1_500u32 {
        w.insert_edge(0, dst).unwrap();
    }

    // Measured window: inserts spread across fresh, low-degree vertices. Their
    // subtrees stay far below the warmed scratch size and the arena is
    // pre-sized, so a correct implementation never reaches for the heap.
    assert_no_alloc("Writer::insert_edge", || {
        for vertex in 1..64u32 {
            for dst in 0..64u32 {
                w.insert_edge(vertex, dst).unwrap();
            }
        }
    });
}
