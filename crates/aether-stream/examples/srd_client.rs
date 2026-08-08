//! SRD client — fetches a feature advertisement over TCP, posts RDMA READs
//! over SRD to gather a random sample, and byte-for-byte verifies the
//! payload against the known per-node signature written by `srd_server`.
//!
//! Usage (on a peer node in the same subnet):
//!
//!   cargo run --features efa --release -p aether-stream --example srd_client \
//!       -- --server 172.31.36.137:7100 --iters 1000 --batch 64

#[cfg(not(all(target_os = "linux", feature = "efa")))]
fn main() {}

#[cfg(all(target_os = "linux", feature = "efa"))]
use aether_stream::rdma::srd::{SrdContext, SrdFeatureClient, SrdShardedFeatureClient};
#[cfg(all(target_os = "linux", feature = "efa"))]
use std::time::Instant;

#[cfg(all(target_os = "linux", feature = "efa"))]
const EFA_GID_INDEX: u8 = 0;

#[cfg(all(target_os = "linux", feature = "efa"))]
fn arg(flag: &str, default: Option<&str>) -> Option<String> {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == flag {
            return it.next();
        }
    }
    default.map(str::to_owned)
}

#[cfg(all(target_os = "linux", feature = "efa"))]
fn main() {
    let server = arg("--server", None).expect("--server <ip:port> required");
    let iters: usize = arg("--iters", Some("500")).unwrap().parse().unwrap();
    let batch: usize = arg("--batch", Some("64")).unwrap().parse().unwrap();
    let warmup: usize = arg("--warmup", Some("16")).unwrap().parse().unwrap();
    let shards: usize = arg("--shards", Some("1")).unwrap().parse().unwrap();

    if SrdContext::open(16, EFA_GID_INDEX).is_err() {
        eprintln!("no EFA device (open failed); aborting");
        std::process::exit(2);
    }

    if shards > 1 {
        run_sharded(&server, iters, batch, warmup, shards);
    } else {
        run_single(&server, iters, batch, warmup);
    }
}

#[cfg(all(target_os = "linux", feature = "efa"))]
fn run_single(server: &str, iters: usize, batch: usize, warmup: usize) {
    let client =
        SrdFeatureClient::connect(server, EFA_GID_INDEX, batch).expect("SrdFeatureClient::connect");
    let schema = client.schema().clone();
    eprintln!(
        "srd_client: 1 shard, slot_size={}B feature_dim={}",
        schema.slot_size, schema.feature_dim
    );

    // Byte-for-byte verification sweep.
    let nodes: Vec<usize> = (0..batch).collect();
    client.gather(&nodes).expect("gather verify");
    verify_slice(client.dst_slice(), &nodes, &schema);
    eprintln!("srd_client: verified {batch} nodes byte-for-byte");

    // Random-gather bench.
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let max_node = schema.node_count;
    for _ in 0..warmup {
        let s = sample(&mut rng, batch, max_node);
        client.gather(&s).expect("warmup gather");
    }
    let start = Instant::now();
    for _ in 0..iters {
        let s = sample(&mut rng, batch, max_node);
        client.gather(&s).expect("gather");
    }
    report("single", iters, batch, schema.slot_size, start.elapsed());
}

#[cfg(all(target_os = "linux", feature = "efa"))]
fn run_sharded(server: &str, iters: usize, batch: usize, warmup: usize, shards: usize) {
    assert!(batch.is_multiple_of(shards), "--batch must divide --shards");
    let per_shard = batch / shards;
    let client = SrdShardedFeatureClient::connect(server, EFA_GID_INDEX, shards, per_shard)
        .expect("SrdShardedFeatureClient::connect");
    let schema = client.schema().clone();
    eprintln!(
        "srd_client: {shards} shards × {per_shard} slots/shard, slot_size={}B feature_dim={}",
        schema.slot_size, schema.feature_dim
    );

    // Byte-for-byte verification sweep — each shard gets a distinct node
    // range so the per-shard dst slices can be verified independently.
    let nodes: Vec<usize> = (0..batch).collect();
    client.gather(&nodes).expect("gather verify");
    for sidx in 0..shards {
        let slice = &nodes[sidx * per_shard..(sidx + 1) * per_shard];
        verify_slice(client.shard_dst(sidx), slice, &schema);
    }
    eprintln!("srd_client: verified {batch} nodes byte-for-byte across {shards} shards");

    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let max_node = schema.node_count;
    for _ in 0..warmup {
        let s = sample(&mut rng, batch, max_node);
        client.gather(&s).expect("warmup gather");
    }
    let start = Instant::now();
    for _ in 0..iters {
        let s = sample(&mut rng, batch, max_node);
        client.gather(&s).expect("gather");
    }
    report(
        &format!("{shards}-shard"),
        iters,
        batch,
        schema.slot_size,
        start.elapsed(),
    );
}

#[cfg(all(target_os = "linux", feature = "efa"))]
fn sample(rng: &mut u64, n: usize, max_node: usize) -> Vec<usize> {
    (0..n)
        .map(|_| {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            (*rng as usize) % max_node
        })
        .collect()
}

#[cfg(all(target_os = "linux", feature = "efa"))]
fn verify_slice(dst: &[u8], nodes: &[usize], schema: &aether_stream::feature_table::FeatureSchema) {
    let feat_dim = schema.feature_dim;
    for (i, &node) in nodes.iter().enumerate() {
        let slot_bytes = &dst[i * schema.slot_size..(i + 1) * schema.slot_size];
        let head = u64::from_le_bytes(slot_bytes[0..8].try_into().unwrap());
        let tail = u64::from_le_bytes(
            slot_bytes[schema.tail_offset_in_slot..schema.tail_offset_in_slot + 8]
                .try_into()
                .unwrap(),
        );
        assert_eq!(head, tail, "node {node}: torn (h={head} t={tail})");
        assert_eq!(head, 2, "node {node}: head should be 2 after one write");
        let feat_bytes = &slot_bytes
            [schema.feature_offset_in_slot..schema.feature_offset_in_slot + feat_dim * 4];
        let got: Vec<f32> = feat_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let want: Vec<f32> = (0..feat_dim)
            .map(|d| (node * 1000 + d) as f32 + 0.5)
            .collect();
        assert_eq!(got, want, "node {node}: payload mismatch");
    }
}

#[cfg(all(target_os = "linux", feature = "efa"))]
fn report(label: &str, iters: usize, batch: usize, slot: usize, elapsed: std::time::Duration) {
    let per_iter_us = elapsed.as_secs_f64() * 1e6 / iters as f64;
    let bytes_total = (iters * batch * slot) as f64;
    let mb_per_sec = bytes_total / elapsed.as_secs_f64() / 1e6;
    eprintln!(
        "srd_client [{label}]: {iters} batches of {batch} in {elapsed:?} ({per_iter_us:.1} µs/batch, {mb_per_sec:.0} MB/s, {:.2} µs/read)",
        per_iter_us / batch as f64,
    );
}
