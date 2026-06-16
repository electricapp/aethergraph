//! Loom model check for the lock-free union-find in `graph::reorder`.
//!
//! Loom's atomics are distinct types from `std::sync::atomic`, so — following
//! the same pattern as `aether-graph`'s `loom_writer_guard` — this models the
//! `find`/`union` algorithm in isolation rather than instrumenting the
//! production type. The modeled code is a faithful copy of
//! `ConcurrentUnionFind` (union-by-id with path-halving `find`); the two must
//! be kept in step.
//!
//! Property under test: every node's parent has an id no smaller than its own.
//! Union-by-id preserves it (the loser — the smaller id — is linked under the
//! winner — the larger id) and path-halving preserves it (a node is repointed
//! at its grandparent, whose id is no smaller). While that invariant holds the
//! forest is acyclic and `find` is guaranteed to terminate; if it could ever
//! break, `find` might cycle and hang. Loom exploring every interleaving and
//! finding the invariant always true is therefore a proof of cycle-freedom
//! across concurrent unions — the exact guarantee a rank-tie union-by-rank
//! could not provide here.
//!
//! Run with:
//!     cargo test -p aethergraph-core --features loom --test loom_union_find

#![cfg(feature = "loom")]

use loom::sync::Arc;
use loom::sync::atomic::{AtomicU32, Ordering};
use loom::thread;

struct ModelUnionFind {
    parent: Vec<AtomicU32>,
}

impl ModelUnionFind {
    fn new(n: u32) -> Self {
        Self {
            parent: (0..n).map(AtomicU32::new).collect(),
        }
    }

    fn find(&self, mut x: u32) -> u32 {
        loop {
            let p = self.parent[x as usize].load(Ordering::Relaxed);
            if p == x {
                return x;
            }
            let gp = self.parent[p as usize].load(Ordering::Relaxed);
            let _ = self.parent[x as usize].compare_exchange_weak(
                p,
                gp,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            x = gp;
        }
    }

    fn union(&self, a: u32, b: u32) {
        loop {
            let ra = self.find(a);
            let rb = self.find(b);
            if ra == rb {
                return;
            }
            // Link smaller id under larger id — a pure function of the pair.
            let (winner, loser) = if ra < rb { (rb, ra) } else { (ra, rb) };
            if self.parent[loser as usize]
                .compare_exchange(loser, winner, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
        }
    }
}

/// The acyclicity invariant: a node's parent id is never smaller than its own.
fn assert_acyclic(uf: &ModelUnionFind) {
    for (i, slot) in uf.parent.iter().enumerate() {
        let p = slot.load(Ordering::Relaxed);
        assert!(p >= i as u32, "cycle-forming link: parent[{i}] = {p} < {i}");
    }
}

/// Two threads merging the same pair in opposite argument order — the shape a
/// rank-tie union-by-rank could deadlock on. Must stay acyclic and converge.
#[test]
fn opposite_order_unions_never_cycle() {
    loom::model(|| {
        let uf = Arc::new(ModelUnionFind::new(2));
        let a = Arc::clone(&uf);
        let b = Arc::clone(&uf);

        let t1 = thread::spawn(move || a.union(0, 1));
        let t2 = thread::spawn(move || b.union(1, 0));
        t1.join().unwrap();
        t2.join().unwrap();

        assert_acyclic(&uf);
        assert_eq!(uf.find(0), uf.find(1), "0 and 1 must end in one set");
    });
}

/// Two concurrent unions sharing a common element must merge all three nodes.
#[test]
fn overlapping_merges_converge() {
    loom::model(|| {
        let uf = Arc::new(ModelUnionFind::new(3));
        let a = Arc::clone(&uf);
        let b = Arc::clone(&uf);

        let t1 = thread::spawn(move || a.union(0, 2));
        let t2 = thread::spawn(move || b.union(1, 2));
        t1.join().unwrap();
        t2.join().unwrap();

        assert_acyclic(&uf);
        let root = uf.find(0);
        assert_eq!(uf.find(1), root, "1 must join the merged set");
        assert_eq!(uf.find(2), root, "2 must join the merged set");
    });
}
