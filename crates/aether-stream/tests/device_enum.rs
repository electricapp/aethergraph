//! Device enumeration + open-by-index coverage.
//!
//! Guards `enumerate_devices()` against sysfs shape regressions and
//! confirms `RdmaContext::open_on_device` honors the enumerated index.
//! NUMA node is `None` on SoftRoCE (no `device/` sysfs entry under
//! `/sys/class/infiniband/rxe0/`) and `Some(_)` on physical HCAs — the
//! test accepts either.

#![cfg(all(target_os = "linux", feature = "rdma"))]

use aether_stream::rdma::context::{RdmaContext, enumerate_devices};

const ROCE_V2_GID_INDEX: u8 = 1;

fn rdma_available() -> bool {
    if std::env::var("AETHER_SKIP_RDMA").is_ok() {
        return false;
    }
    RdmaContext::open(16, ROCE_V2_GID_INDEX).is_ok()
}

macro_rules! skip_if_no_rdma {
    () => {
        if !rdma_available() {
            eprintln!("skipping: no RDMA device");
            return;
        }
    };
}

#[test]
fn enumerate_lists_at_least_one_device() {
    skip_if_no_rdma!();
    let devs = enumerate_devices().expect("enumerate");
    assert!(!devs.is_empty(), "at least one RDMA device expected");
    for d in &devs {
        assert!(!d.name.is_empty(), "device name must be populated");
        // NUMA node is Some(i32) or None — both valid. Reject garbage.
        if let Some(n) = d.numa_node {
            assert!(n >= -1, "numa_node must be >= -1, got {n}");
        }
        eprintln!("  [{}] {} numa={:?}", d.index, d.name, d.numa_node);
    }
    // Indices must be 0..len.
    for (i, d) in devs.iter().enumerate() {
        assert_eq!(
            d.index, i,
            "enumerate_devices must return contiguous indices starting at 0"
        );
    }
}

#[test]
fn open_on_device_honors_index() {
    skip_if_no_rdma!();
    let devs = enumerate_devices().expect("enumerate");
    for d in &devs {
        let ctx = RdmaContext::open_on_device(16, d.index, ROCE_V2_GID_INDEX)
            .unwrap_or_else(|e| panic!("open_on_device({}, {}): {e}", d.index, d.name));
        assert_eq!(ctx.gid_index, ROCE_V2_GID_INDEX);
        drop(ctx);
    }
}

#[test]
fn open_on_device_out_of_range_rejects() {
    skip_if_no_rdma!();
    let devs = enumerate_devices().expect("enumerate");
    let bad = devs.len() + 100;
    let err = match RdmaContext::open_on_device(16, bad, ROCE_V2_GID_INDEX) {
        Ok(_) => panic!("open_on_device must reject out-of-range index"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}
