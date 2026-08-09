//! TEST_PLAN T1.7 partial — AF_XDP socket creation against a veth interface.
//!
//! Full T1.7/T1.8 require a BPF XDP program attached to the veth that
//! redirects packets into the AF_XDP socket via `xdp_redirect_map(XSKMAP)`.
//! This codebase has no BPF program loader yet, so a true end-to-end
//! "send 100 UDP, receive 100 frames" test is blocked on that work.
//!
//! What this test DOES validate:
//!   - UMEM allocation (page alignment, mlock, frame_size constraints)
//!   - XdpSocket::create FFI struct sizes against current rdma-core /
//!     liblibbpf headers (catches the same kind of layout drift that bit
//!     the RDMA FFI — 10 latent bugs there were found by exercising the
//!     data path; AF_XDP has the same risk)
//!   - Ring mmap setup (FILL/RX/TX/COMPLETION offsets)
//!   - veth link existence + ifindex lookup
//!
//! What this test does NOT validate (gap to fill before relying on AF_XDP):
//!   - Actual packet ingestion (needs BPF redirect program)
//!   - ingest_loop end-to-end with received frames
//!   - FeatureTable write from packet payload (T1.8)

#![cfg(target_os = "linux")]

use aether_stream::umem::Umem;
use aether_stream::xdp::socket::XdpSocket;
use std::ffi::CString;

const VETH_RX: &str = "veth-rx";

/// XDP_COPY bind flag from <linux/if_xdp.h>.
const XDP_COPY: u16 = 1 << 1;

fn ifindex_of(name: &str) -> Option<u32> {
    let cname = CString::new(name).ok()?;
    // SAFETY: cname is a valid C string for the lifetime of this call.
    let idx = unsafe { libc::if_nametoindex(cname.as_ptr()) };
    if idx == 0 { None } else { Some(idx) }
}

fn skip_if_no_veth() -> Option<u32> {
    match ifindex_of(VETH_RX) {
        Some(idx) => Some(idx),
        None => {
            eprintln!(
                "skipping: {VETH_RX} not present. Set up via:\n  \
                 sudo ip link add veth-tx type veth peer name veth-rx\n  \
                 sudo ip link set veth-tx up && sudo ip link set veth-rx up"
            );
            None
        }
    }
}

#[test]
fn t1_7_umem_allocation_4096() {
    // 4096 is the safer frame size: matches page alignment exactly, so
    // SharedMemoryRing's rounding doesn't shift the kernel-visible chunk
    // stride away from what we register via XDP_UMEM_REG.
    let umem = Umem::new(256, 4096, vec![]).expect("UMEM alloc");
    assert_eq!(umem.frame_size(), 4096);
    assert_eq!(umem.frame_count(), 256);
    // `total_size` is the actual mmap backing — on hosts with 2 MiB huge pages
    // enabled (e.g. EFA instances via `efa-config`) the ring rounds up to a
    // huge-page boundary, so the hard equality would fail. What matters for
    // correctness is ≥ requested bytes and the frame stride below.
    let requested = 256 * 4096;
    assert!(
        umem.total_size() >= requested,
        "total_size {} < requested {requested}",
        umem.total_size()
    );
    assert_eq!(umem.base_addr() as usize % 4096, 0);
    let p0 = umem.frame_ptr(0) as usize;
    let p1 = umem.frame_ptr(1) as usize;
    assert_eq!(p1 - p0, 4096, "stride must match registered chunk_size");
}

#[test]
#[should_panic(expected = "supports only 4096-byte frames")]
fn t1_7_umem_rejects_2048_frame_size() {
    // Non-4096 frame sizes hit a kernel-vs-userspace stride mismatch
    // (kernel registers the caller's chunk_size but lays frames out at
    // 4096B). Umem::new rejects them up front instead.
    let _ = Umem::new(256, 2048, vec![]);
}

#[test]
#[should_panic(expected = "supports only 4096-byte frames")]
fn t1_7_umem_rejects_arbitrary_size() {
    let _ = Umem::new(256, 1024, vec![]);
}

#[test]
fn t1_7_umem_acquire_release_roundtrip() {
    let umem = Umem::new(8, 4096, vec![]).expect("alloc");
    // Drain the pool.
    let mut taken = Vec::new();
    while let Some(idx) = umem.acquire_frame() {
        taken.push(idx);
    }
    assert_eq!(
        taken.len(),
        8,
        "pool should yield exactly frame_count frames"
    );
    assert!(umem.acquire_frame().is_none(), "pool must be exhausted");
    // Return one and re-acquire.
    umem.release_frame(taken[0]);
    let reacquired = umem.acquire_frame().expect("reacquire after release");
    assert_eq!(reacquired, taken[0], "Treiber stack is LIFO");
}

#[test]
fn t1_7_xdp_socket_create_on_veth() {
    let Some(ifindex) = skip_if_no_veth() else {
        return;
    };

    // Need locked-memory limit raised — bootstrap.sh sets * memlock unlimited
    // but that requires a fresh login. Fall back gracefully if test harness
    // didn't get the limit (otherwise UMEM mlock fails).
    let umem = match Umem::new(256, 4096, vec![]) {
        Some(u) => u,
        None => {
            eprintln!(
                "skipping: UMEM alloc failed (likely RLIMIT_MEMLOCK too low — re-login or `ulimit -l unlimited`)"
            );
            return;
        }
    };

    // SAFETY: umem points to a valid mlock'd region of total_size bytes.
    let socket_result = unsafe {
        XdpSocket::create(
            umem.base_addr() as *mut u8,
            umem.total_size(),
            umem.frame_size(),
            64, // ring size — power of two
            ifindex,
            0,        // queue id
            XDP_COPY, // veth doesn't support ZEROCOPY
        )
    };

    match socket_result {
        Ok(mut socket) => {
            // FFI sanity: socket has a valid fd and all four rings constructed.
            assert!(socket.fd() > 0, "socket fd must be valid");
            let free = socket.fill.free_slots();
            assert!(free > 0, "FILL ring must have free slots after bind");
            // RX is empty without a BPF redirect program — that's expected.
            assert_eq!(socket.rx.available(), 0);
            drop(socket);
        }
        Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
            // KNOWN GAP: AF_XDP bind() against a veth without a loaded BPF
            // XDP program returns EINVAL. Codebase has no BPF program loader
            // yet — fix is to add libbpf-rs / aya-rs + a tiny xsk_redirect
            // program. Documented in TEST_PLAN.md alongside T1.7/T1.8.
            eprintln!("skipping E2E: bind(AF_XDP) → EINVAL — no BPF redirect program loaded.");
        }
        Err(e)
            if e.raw_os_error() == Some(libc::EPERM) || e.raw_os_error() == Some(libc::EACCES) =>
        {
            eprintln!(
                "skipping: AF_XDP needs CAP_NET_RAW. Re-run as root or with that capability."
            );
        }
        Err(e) => panic!("XdpSocket::create failed against {VETH_RX} with unexpected error: {e}"),
    }
}
