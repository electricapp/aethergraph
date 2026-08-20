//! Model-based serializability check for snapshots: every snapshot must
//! equal the reference model at its commit, no matter how many later
//! commits churn (and recycle) the arena underneath it.

use std::collections::BTreeSet;

use aether_graph::{DynamicGraph, Snapshot};
use proptest::prelude::*;

const NV: u32 = 64;

type Model = Vec<BTreeSet<u32>>;

fn assert_matches(g: &DynamicGraph, snap: &Snapshot, model: &Model) {
    let mut buf = Vec::new();
    let mut total = 0u64;
    for v in 0..NV {
        snap.neighbors_into(g, v, &mut buf);
        let want: Vec<u32> = model[v as usize].iter().copied().collect();
        assert_eq!(buf, want, "vertex {v} diverged from model");
        assert_eq!(snap.degree(g, v), want.len());
        total += want.len() as u64;
    }
    assert_eq!(snap.num_edges(), total);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Random batches, one commit each; every held snapshot stays equal
    /// to the model state captured at its commit.
    #[test]
    fn snapshots_are_serializable(
        batches in prop::collection::vec(
            prop::collection::vec((0..NV, 0..NV), 1..40),
            1..25,
        )
    ) {
        let g = DynamicGraph::new(NV as usize, 8 << 20);
        let mut model: Model = vec![BTreeSet::new(); NV as usize];
        let mut held: Vec<(Snapshot, Model)> = vec![(g.acquire(), model.clone())];

        for batch in &batches {
            {
                let mut w = g.writer().unwrap();
                for &(s, d) in batch {
                    w.insert_edge(s, d).unwrap();
                    model[s as usize].insert(d);
                }
            }
            held.push((g.acquire(), model.clone()));
            // Re-verify a prefix mid-run so pinned state is checked while
            // later commits are still churning.
            let (old_snap, old_model) = &held[held.len() / 2];
            assert_matches(&g, old_snap, old_model);
        }

        for (snap, m) in &held {
            assert_matches(&g, snap, m);
        }
    }
}
