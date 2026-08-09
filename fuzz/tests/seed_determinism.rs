//! Regression test: the fuzz corpus seeder MUST be byte-deterministic and
//! the committed corpus MUST match what the seeder emits today. Any drift
//! — a wall-clock dependency, an unseeded RNG, a HashMap iteration order
//! leak, or someone editing the seed list without re-running the seeder —
//! fails this test.
//!
//! The two checks live in one `#[test]` because they both mutate
//! `fuzz/corpus/`, so running them in parallel would race. Cargo runs
//! `#[test]`s within one file in parallel by default; keeping it as a
//! single function is simpler than coordinating a global mutex.

use std::collections::BTreeMap;
use std::fs;
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::process::Command;

const TARGETS: &[&str] = &["ctree_insert_sequences", "csr_loader_bytes"];

fn snapshot_corpus(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut snap = BTreeMap::new();
    for target in TARGETS {
        let dir = root.join("corpus").join(target);
        for entry in fs::read_dir(&dir).expect("read corpus dir") {
            let entry = entry.expect("dir entry");
            let name = format!("{target}/{}", entry.file_name().to_string_lossy());
            snap.insert(name, fs::read(entry.path()).expect("read seed"));
        }
    }
    snap
}

fn hash_snapshot(snap: &BTreeMap<String, Vec<u8>>) -> u64 {
    let mut h = twox_hash::XxHash64::with_seed(0);
    for (k, v) in snap {
        h.write(k.as_bytes());
        h.write(v);
    }
    h.finish()
}

fn run_seeder(manifest: &Path) {
    let status = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--bin", "seed_corpus"])
        .current_dir(manifest)
        .status()
        .expect("spawn cargo");
    assert!(status.success(), "seed_corpus binary returned non-zero");
}

#[test]
fn seed_corpus_is_reproducible_and_matches_committed_floor() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Step 1: snapshot the corpus as committed to git BEFORE we run anything.
    let committed = snapshot_corpus(&manifest);

    // Step 2: run the seeder, snapshot, hash.
    run_seeder(&manifest);
    let first = snapshot_corpus(&manifest);
    let first_hash = hash_snapshot(&first);

    // Step 3: run the seeder a second time. Byte-equal output ⇒ deterministic.
    run_seeder(&manifest);
    let second = snapshot_corpus(&manifest);
    let second_hash = hash_snapshot(&second);
    assert_eq!(
        first_hash, second_hash,
        "seed_corpus is NOT deterministic — two runs produced different output",
    );

    // Step 4: the seeder output must match the committed corpus. If a
    // contributor edits the seed list and forgets to commit the regenerated
    // corpus, this catches it.
    let committed_hash = hash_snapshot(&committed);
    assert_eq!(
        committed_hash, first_hash,
        "committed corpus drifts from seed_corpus output. Run \
         `cargo run -p aethergraph-fuzz --bin seed_corpus` and commit the diff."
    );
}
