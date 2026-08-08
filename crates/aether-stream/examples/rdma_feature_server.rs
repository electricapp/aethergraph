//! RDMA feature server (RC / SoftRoCE).
//!
//! Serves an in-memory `FeatureTable` over RDMA RC so a Python
//! `NeighborLoader(feature_source="rdma://...")` can gather against it.
//! The SRD variant is `srd_server.rs`.
//!
//! Usage:
//!
//!   cargo run --features rdma --release -p aether-stream \
//!     --example rdma_feature_server -- \
//!     --bind 127.0.0.1:9999 --nodes 4096 --dim 128 [--live-rate 10000]
//!
//! On bind success the server prints `READY port=<port>` on stderr (parseable
//! by harness scripts), then blocks in the control-plane accept loop. With
//! `--live-rate N` a writer thread overwrites every node N times per second
//! so consumers see continuously updating features.

#[cfg(not(all(target_os = "linux", feature = "rdma")))]
fn main() {
    eprintln!("rdma_feature_server: build with --features rdma on Linux");
}

#[cfg(all(target_os = "linux", feature = "rdma"))]
use aether_stream::feature_table::FeatureTable;
#[cfg(all(target_os = "linux", feature = "rdma"))]
use aether_stream::rdma::context::RdmaContext;
#[cfg(all(target_os = "linux", feature = "rdma"))]
use aether_stream::rdma::control::{RdmaAdvertisement, serve_control_plane_with_qp};
#[cfg(all(target_os = "linux", feature = "rdma"))]
use aether_stream::rdma::ffi::{IBV_ACCESS_LOCAL_WRITE, IBV_ACCESS_REMOTE_READ};

#[cfg(all(target_os = "linux", feature = "rdma"))]
const ROCE_V2_GID_INDEX: u8 = 1;

#[cfg(all(target_os = "linux", feature = "rdma"))]
fn arg(flag: &str, default: Option<&str>) -> Option<String> {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == flag {
            return it.next();
        }
    }
    default.map(str::to_owned)
}

#[cfg(all(target_os = "linux", feature = "rdma"))]
fn main() {
    use std::sync::Arc;

    let bind = arg("--bind", Some("127.0.0.1:9999")).unwrap();
    let node_count: usize = arg("--nodes", Some("4096")).unwrap().parse().unwrap();
    let feature_dim: usize = arg("--dim", Some("128")).unwrap().parse().unwrap();
    // Writes/sec across the whole table. 0 disables the writer (table is
    // initialized once and left alone).
    let live_rate: u64 = arg("--live-rate", Some("0")).unwrap().parse().unwrap();

    let table = Arc::new(FeatureTable::new(node_count, feature_dim, vec![]).expect("table alloc"));
    for n in 0..node_count {
        let v: Vec<f32> = (0..feature_dim)
            .map(|i| (n * 1000 + i) as f32 + 0.5)
            .collect();
        table.write_node(n, &v);
    }
    let schema = table.schema();
    eprintln!(
        "rdma_feature_server: table {} nodes x {} dim, slot_size={}B, total={} MiB, live_rate={}/s",
        node_count,
        feature_dim,
        schema.slot_size,
        table.total_size() / (1024 * 1024),
        live_rate,
    );

    // Background writer thread (optional). `stop` is unused once the
    // process is structured around `serve_control_plane_with_qp` blocking
    // forever — the writer dies on process exit. We hold a reference here
    // to keep the JoinHandle alive for the lifetime of the server.
    let _writer = if live_rate > 0 {
        let table = Arc::clone(&table);
        Some(std::thread::spawn(move || {
            let mut generation: u64 = 1;
            let inter_batch = std::time::Duration::from_micros(
                1_000_000u64.checked_div(live_rate).unwrap_or(0).max(1),
            );
            loop {
                for n in 0..node_count {
                    let v: Vec<f32> = (0..feature_dim)
                        .map(|i| (n * 1000 + i) as f32 + 0.5 + generation as f32 * 0.001)
                        .collect();
                    table.write_node(n, &v);
                }
                generation = generation.wrapping_add(1);
                std::thread::sleep(inter_batch);
            }
        }))
    } else {
        None
    };

    let ctx = RdmaContext::open(256, ROCE_V2_GID_INDEX).expect("RdmaContext::open");
    // SAFETY: `table` is Arc-held to the end of main (see the final `let _`),
    // so the registered range outlives the MR and all remote reads.
    let mr = unsafe {
        ctx.reg_mr(
            table.base_addr() as *mut u8,
            table.total_size(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .expect("reg_mr");

    let adv = RdmaAdvertisement {
        base_addr: table.base_addr(),
        rkey: mr.rkey(),
        schema: schema.clone(),
    };

    // Mark the server as live so test harnesses can wait deterministically
    // instead of polling for connect(). `port=` is the part the test parses.
    let port = bind.rsplit_once(':').map(|(_, p)| p).unwrap_or("?");
    eprintln!("rdma_feature_server: READY port={port}");
    eprintln!("rdma_feature_server: listening on {bind}");

    if let Err(e) = serve_control_plane_with_qp(&bind, &adv, &ctx) {
        eprintln!("serve_control_plane_with_qp: {e}");
        std::process::exit(1);
    }

    let _ = (mr, ctx, table);
}
