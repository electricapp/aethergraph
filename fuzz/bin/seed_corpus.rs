//! Reproducible fuzz-corpus seeder.
//!
//! Writes a fixed set of seed inputs to `fuzz/corpus/<target>/`. Every seed
//! is fully determined by this source file:
//!
//!   - Hand-crafted boundary cases (empty, single, exact CHUNK_CAP, etc.)
//!     live in `static SEEDS_*` arrays below.
//!   - Random seeds are generated with [`rand::rngs::StdRng`] seeded from a
//!     fixed `SEED: u64 = 0x_AE7E_AE7E_AE7E_AE7E`. The same seed produces
//!     bit-identical corpus on every run, on every machine.
//!   - File names are the lowercase hex of `xxhash3_64` of the bytes — also
//!     deterministic; libfuzzer simply uses the file content, the name is
//!     informational.
//!
//! Run:
//!     cargo run -p aethergraph-fuzz --bin seed_corpus
//!
//! CI regenerates the corpus from this binary before invoking the fuzzer so
//! the floor is auditable in git rather than implicit in shell history.

use std::fs;
use std::hash::Hasher;
use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use twox_hash::XxHash64;

/// Deterministic seed for the in-target random corpus. Bumping this value
/// changes the corpus; doing so should be a deliberate, reviewable commit.
const SEED: u64 = 0xAE7E_AE7E_AE7E_AE7E;

// ─── ctree_insert_sequences ───────────────────────────────────────────────
//
// The fuzz target takes a `Vec<u32>` via libfuzzer's `Arbitrary` impl, which
// reads u32s from the raw byte slice. So a seed file is simply
// little-endian u32s concatenated.
//
// Seeds cover: empty, single, sorted, reversed, duplicates, exact CHUNK_CAP,
// split trigger, and three random distributions.

const CTREE_BOUNDARY_SEEDS: &[&[u32]] = &[
    // empty input
    &[],
    // single
    &[42],
    // exactly CHUNK_CAP — fills a leaf without splitting
    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    // CHUNK_CAP + 1 — forces a split
    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    // reverse-sorted — exercises the splay path
    &[15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    // duplicates dominate
    &[1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
    // boundary values
    &[0, u32::MAX, 1, u32::MAX - 1, 0],
    // dense block
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ],
];

const CTREE_RANDOM_SEED_COUNT: usize = 8;
const CTREE_RANDOM_SEED_LEN: usize = 64;
const CTREE_RANDOM_SEED_MAX: u32 = 10_000;

// ─── csr_loader_bytes ─────────────────────────────────────────────────────
//
// The fuzz target writes the byte slice to a temp file and asks the CSR
// loader to parse it. Seeds: empty, truncated header, valid-magic + valid-
// version + nonsense body, random.

const CSR_MAGIC: u32 = 0x4145_5448; // "AETH"
const CSR_VERSION: u32 = 1;
const CSR_HEADER_BYTES: usize = 32;

fn build_csr_header(num_nodes: u64, num_edges: u64, has_weights: bool) -> [u8; CSR_HEADER_BYTES] {
    let mut hdr = [0u8; CSR_HEADER_BYTES];
    hdr[0..4].copy_from_slice(&CSR_MAGIC.to_le_bytes());
    hdr[4..8].copy_from_slice(&CSR_VERSION.to_le_bytes());
    hdr[8..16].copy_from_slice(&num_nodes.to_le_bytes());
    hdr[16..24].copy_from_slice(&num_edges.to_le_bytes());
    hdr[24..28].copy_from_slice(&u32::from(has_weights).to_le_bytes());
    // checksum field at 28..32 left zero (= absent / legacy).
    hdr
}

// ─── compressed_loader_bytes ──────────────────────────────────────────────
//
// A version-2 file is the 32-byte header followed by an Elias-Fano offsets
// record, a StreamVByte edges record, and raw f32 weights when flagged. The
// seeds are built through the codecs' own serializers so they parse; the
// checksum field is zero (absent), letting mutations reach the decoders.

const COMPRESSED_VERSION: u32 = 2;

fn build_compressed(offsets: &[u64], edges: &[u32], weights: Option<&[f32]>) -> Vec<u8> {
    let num_nodes = offsets.len().saturating_sub(1) as u64;
    let mut out = build_csr_header(num_nodes, edges.len() as u64, weights.is_some()).to_vec();
    out[4..8].copy_from_slice(&COMPRESSED_VERSION.to_le_bytes());
    aethergraph_core::EliasFano::encode(offsets).write_into(&mut out);
    aethergraph_core::StreamVByte::encode_deltas(edges).write_into(&mut out);
    for w in weights.unwrap_or(&[]) {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

// ─── helpers ──────────────────────────────────────────────────────────────

fn seed_filename(bytes: &[u8]) -> String {
    let mut h = XxHash64::with_seed(0);
    h.write(bytes);
    format!("seed_{:016x}", h.finish())
}

fn write_seed(dir: &Path, bytes: &[u8]) -> std::io::Result<PathBuf> {
    let name = seed_filename(bytes);
    let path = dir.join(name);
    fs::write(&path, bytes)?;
    Ok(path)
}

fn reset_dir(p: &Path) -> std::io::Result<()> {
    if p.exists() {
        for entry in fs::read_dir(p)? {
            let entry = entry?;
            // Skip libfuzzer-discovered crashes / regression artifacts —
            // keep the user-curated corpus only. cargo-fuzz writes those to
            // sibling dirs (`artifacts/`), not into `corpus/`, so anything
            // inside `corpus/<target>/` is fair game to drop.
            let _ = fs::remove_file(entry.path());
        }
    } else {
        fs::create_dir_all(p)?;
    }
    Ok(())
}

fn corpus_root() -> PathBuf {
    // Run from anywhere; locate the fuzz crate root via Cargo manifest dir.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// ─── corpus builders ──────────────────────────────────────────────────────

fn build_ctree_corpus() -> std::io::Result<usize> {
    let dir = corpus_root().join("corpus/ctree_insert_sequences");
    reset_dir(&dir)?;

    let mut count = 0;
    for seq in CTREE_BOUNDARY_SEEDS {
        let mut bytes = Vec::with_capacity(seq.len() * 4);
        for v in *seq {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        write_seed(&dir, &bytes)?;
        count += 1;
    }

    let mut rng = StdRng::seed_from_u64(SEED);
    for _ in 0..CTREE_RANDOM_SEED_COUNT {
        let mut bytes = Vec::with_capacity(CTREE_RANDOM_SEED_LEN * 4);
        for _ in 0..CTREE_RANDOM_SEED_LEN {
            let v = rng.random_range(0..CTREE_RANDOM_SEED_MAX);
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        write_seed(&dir, &bytes)?;
        count += 1;
    }

    Ok(count)
}

fn build_csr_corpus() -> std::io::Result<usize> {
    let dir = corpus_root().join("corpus/csr_loader_bytes");
    reset_dir(&dir)?;

    let mut count = 0;

    // Empty input
    write_seed(&dir, &[])?;
    count += 1;

    // Truncated header (first 4 bytes only — magic without the rest).
    let trunc = CSR_MAGIC.to_le_bytes();
    write_seed(&dir, &trunc)?;
    count += 1;

    // Valid magic + version, zero nodes/edges, no weights — a minimal
    // well-formed graph.
    let hdr = build_csr_header(0, 0, false);
    write_seed(&dir, &hdr)?;
    count += 1;

    // Valid magic + version, claims 4 nodes / 6 edges but body is empty —
    // exercises bounds-checking on offsets parsing.
    write_seed(&dir, &build_csr_header(4, 6, false))?;
    count += 1;

    // Bad magic but right size — exercises the rejection path.
    let mut bad_magic = vec![0u8; CSR_HEADER_BYTES];
    bad_magic[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    write_seed(&dir, &bad_magic)?;
    count += 1;

    // Random bytes, fixed-seed, varying lengths.
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x_CB07_CB07_CB07_CB07);
    for len in [16usize, 64, 256, 1024] {
        let mut bytes = vec![0u8; len];
        rng.fill_bytes(&mut bytes);
        write_seed(&dir, &bytes)?;
        count += 1;
    }

    Ok(count)
}

fn build_compressed_corpus() -> std::io::Result<usize> {
    let dir = corpus_root().join("corpus/compressed_loader_bytes");
    reset_dir(&dir)?;

    let mut count = 0;

    // Empty graph.
    write_seed(&dir, &build_compressed(&[0], &[], None))?;
    count += 1;

    // Two triangles joined by one edge, sorted neighbor lists.
    let offsets = [0u64, 2, 4, 7, 9, 11, 13];
    let edges = [1u32, 2, 0, 2, 0, 1, 3, 2, 4, 3, 5, 3, 4];
    write_seed(&dir, &build_compressed(&offsets, &edges, None))?;
    count += 1;

    // The same graph with weights.
    let weights: Vec<f32> = (0..edges.len()).map(|i| i as f32 * 0.25).collect();
    write_seed(&dir, &build_compressed(&offsets, &edges, Some(&weights)))?;
    count += 1;

    // A hub with unsorted, repeated neighbors (wrapping deltas).
    write_seed(
        &dir,
        &build_compressed(&[0, 6, 6, 6], &[2, 0, 2, 1, 0, 1], None),
    )?;
    count += 1;

    // Header only: counts promise records that are missing.
    let mut header_only = build_csr_header(4, 6, false).to_vec();
    header_only[4..8].copy_from_slice(&COMPRESSED_VERSION.to_le_bytes());
    write_seed(&dir, &header_only)?;
    count += 1;

    Ok(count)
}

fn main() -> std::io::Result<()> {
    let ctree = build_ctree_corpus()?;
    let csr = build_csr_corpus()?;
    let compressed = build_compressed_corpus()?;
    println!("seeded ctree_insert_sequences: {ctree} files");
    println!("seeded csr_loader_bytes:        {csr} files");
    println!("seeded compressed_loader_bytes: {compressed} files");
    println!("RNG seed: 0x{SEED:016x} (change deliberately and review the diff)");
    Ok(())
}
