//! SRD server — half of the cross-node EFA integration test.
//!
//! Run on the "feature table owner" node. Allocates + registers a
//! `FeatureTable`, brings up an SRD QP, and serves an `SrdAdvertisement`
//! over TCP forever.
//!
//! Usage (on the server node):
//!
//!   cargo run --features efa --release -p aether-stream --example srd_server \
//!       -- --bind 0.0.0.0:7100 --nodes 4096 --dim 128
//!
//! Pair with `cargo run … --example srd_client` on another box in the same
//! subnet (same security group so EFA traffic is allowed).

#![cfg(all(target_os = "linux", feature = "efa"))]

use aether_stream::feature_table::FeatureTable;
use aether_stream::rdma::ffi::{IBV_ACCESS_LOCAL_WRITE, IBV_ACCESS_REMOTE_READ};
use aether_stream::rdma::srd::{
    serve_srd_control_plane, SrdAdvertisement, SrdContext, SrdQp, DEFAULT_SRD_QP_CAP,
};

const EFA_GID_INDEX: u8 = 0;

fn arg(flag: &str, default: Option<&str>) -> Option<String> {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == flag {
            return it.next();
        }
    }
    default.map(str::to_owned)
}

fn main() {
    let bind = arg("--bind", Some("0.0.0.0:7100")).unwrap();
    let node_count: usize = arg("--nodes", Some("4096")).unwrap().parse().unwrap();
    let feature_dim: usize = arg("--dim", Some("128")).unwrap().parse().unwrap();

    // Build the feature table with a recognizable per-node signature.
    let table = FeatureTable::new(node_count, feature_dim, vec![]).expect("table alloc");
    for n in 0..node_count {
        let v: Vec<f32> = (0..feature_dim).map(|i| (n * 1000 + i) as f32 + 0.5).collect();
        table.write_node(n, &v);
    }
    let schema = table.schema();
    eprintln!(
        "srd_server: table {} nodes x {} dim, slot_size={}B, total={} MiB",
        node_count,
        feature_dim,
        schema.slot_size,
        table.total_size() / (1024 * 1024),
    );

    let ctx = SrdContext::open(256, EFA_GID_INDEX).expect("SrdContext::open");
    let mr = ctx
        .reg_mr(
            table.base_addr() as *mut u8,
            table.total_size(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
        .expect("reg_mr");
    let qp = SrdQp::create(&ctx, &DEFAULT_SRD_QP_CAP).expect("qp create");
    qp.bring_up().expect("bring_up");
    let ep = qp.endpoint(&ctx);
    eprintln!(
        "srd_server: qpn={} qkey=0x{:x} gid={:02x?}",
        ep.qpn, ep.qkey, ep.gid
    );

    let adv = SrdAdvertisement {
        base_addr: table.base_addr(),
        rkey: mr.rkey(),
        schema: schema.clone(),
        endpoint: ep,
    };
    eprintln!("srd_server: listening on {bind}");
    if let Err(e) = serve_srd_control_plane(&bind, &adv, &ctx) {
        eprintln!("serve_srd_control_plane: {e}");
        std::process::exit(1);
    }

    // Keep the MR + QP + context alive while the listener serves (serve
    // blocks forever — this line is unreachable in practice).
    let _ = (mr, qp, ctx, table);
}
