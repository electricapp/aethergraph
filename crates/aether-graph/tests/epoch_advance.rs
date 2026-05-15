//! Cross-subsystem epoch-advance invariants for [`DynamicGraph`].

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use aether_epoch::EpochClock;
use aether_graph::DynamicGraph;

#[test]
fn clean_writer_drop_advances_epoch_once() {
    let g = DynamicGraph::new(64, 1 << 20);
    let before = g.current_epoch();

    {
        let mut w = g.writer().unwrap();
        w.insert_edge(0, 1).unwrap();
        w.insert_edge(0, 2).unwrap();
    }

    let after = g.current_epoch();
    assert_eq!(after.as_u64(), before.as_u64() + 1);
}

#[test]
fn many_writers_advance_monotonically() {
    let g = DynamicGraph::new(64, 1 << 20);
    let start = g.current_epoch().as_u64();

    for i in 0..16u32 {
        let mut w = g.writer().unwrap();
        w.insert_edge(i, (i + 1) % 64).unwrap();
    }

    assert_eq!(g.current_epoch().as_u64(), start + 16);
}

#[test]
fn shared_clock_observed_from_outside() {
    let clock = Arc::new(EpochClock::new());
    let g = DynamicGraph::new_with_epoch(64, 1 << 20, Arc::clone(&clock));
    assert_eq!(clock.current().as_u64(), 0);

    {
        let mut w = g.writer().unwrap();
        w.insert_edge(0, 1).unwrap();
    }

    // The external clone of the clock sees the new epoch — the property
    // FeatureStore et al. rely on for cross-subsystem read pinning.
    assert_eq!(clock.current().as_u64(), 1);
    assert_eq!(g.current_epoch().as_u64(), 1);
}

#[test]
fn panic_poisoned_drop_does_not_advance_epoch() {
    let g = Arc::new(DynamicGraph::new(64, 1 << 20));
    let before = g.current_epoch().as_u64();

    let g2 = Arc::clone(&g);
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _w = g2.writer().unwrap();
        panic!("simulated mid-insert failure");
    }));
    assert!(result.is_err());

    // Poisoned: the epoch must NOT have advanced, because the in-flight
    // edits may be inconsistent and no reader should pin a version that
    // claims to include them.
    assert_eq!(g.current_epoch().as_u64(), before);
}
