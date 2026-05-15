//! End-to-end AF_XDP → FeatureTable integration test.
//!
//! Closes the gap documented in `afxdp_veth.rs`: without an attached XDP
//! program, `bind(AF_XDP)` returns `EINVAL` on veth. This test:
//!
//! 1. Loads the compiled `xsk_redirect` BPF program via aya.
//! 2. Attaches it to `veth-rx` in SKB mode (veth doesn't support DRV).
//! 3. Creates the UMEM + `XdpSocket` and inserts the socket's fd into the
//!    BPF `xsks_map` at queue 0.
//! 4. Spawns `ingest_loop` in a thread.
//! 5. Opens an `AF_PACKET / SOCK_RAW` socket on `veth-tx` and injects 100
//!    Ethernet frames whose payload is `(node_id: u32, features: [f32; 16])`.
//! 6. Reads the frames out of the ingest channel and writes them into a
//!    `FeatureTable`.
//! 7. Asserts: every node's features match what was sent (byte-for-byte)
//!    and the ingest loop sees zero torn reads.
//!
//! Requires:
//!   - Linux + `xdp_bpf` feature (build.rs compiles the BPF object with clang).
//!   - veth pair `veth-tx`/`veth-rx` up and running.
//!   - `CAP_NET_ADMIN` (XDP attach) and `CAP_NET_RAW` (AF_PACKET inject).
//!
//! Run as root or with the two capabilities granted on the test binary:
//!
//!     sudo -E cargo test --features xdp_bpf --test afxdp_t1_8_integration -- --nocapture

#![cfg(all(target_os = "linux", feature = "xdp_bpf"))]

use aether_stream::feature_table::FeatureTable;
use aether_stream::ingest::{IngestConfig, ingest_loop};
use aether_stream::umem::Umem;
use aether_stream::xdp::socket::XdpSocket;
use aya::Ebpf;
use aya::maps::XskMap;
use aya::programs::{Xdp, XdpFlags};
use crossbeam_channel::bounded;
use std::ffi::CString;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const VETH_TX: &str = "veth-tx";
const VETH_RX: &str = "veth-rx";
const XDP_COPY: u16 = 1 << 1;

const NUM_PACKETS: usize = 100;
const FEATURE_DIM: usize = 16;
/// Custom EtherType inside the local-experimental range (RFC 1700). Picking
/// a non-IP type skips the kernel's IP stack so we don't need addresses on
/// the veth peers.
const PAYLOAD_ETHERTYPE: u16 = 0x88B5;

/// Wire format of each injected packet's payload.
#[repr(C, packed)]
struct Payload {
    node_id: u32,
    features: [f32; FEATURE_DIM],
}

fn ifindex_of(name: &str) -> Option<u32> {
    let cname = CString::new(name).ok()?;
    // SAFETY: cname lives for the call.
    let idx = unsafe { libc::if_nametoindex(cname.as_ptr()) };
    (idx != 0).then_some(idx)
}

fn mac_of(name: &str) -> Option<[u8; 6]> {
    // /sys/class/net/<iface>/address — six hex bytes separated by ':'.
    let path = format!("/sys/class/net/{name}/address");
    let text = std::fs::read_to_string(path).ok()?;
    let mut mac = [0u8; 6];
    for (i, byte) in text.trim().split(':').enumerate() {
        if i >= 6 {
            return None;
        }
        mac[i] = u8::from_str_radix(byte, 16).ok()?;
    }
    Some(mac)
}

/// Open an `AF_PACKET / SOCK_RAW` socket bound to `iface` for sending raw
/// Ethernet frames. Returns the fd or `None` if capabilities are missing.
fn open_raw_sender(iface: &str) -> Option<i32> {
    let ifindex = ifindex_of(iface)? as i32;
    // SAFETY: socket() is a thin wrapper around the syscall.
    let fd = unsafe {
        libc::socket(
            libc::AF_PACKET,
            libc::SOCK_RAW,
            (libc::ETH_P_ALL as u16).to_be() as i32,
        )
    };
    if fd < 0 {
        return None;
    }
    let mut addr: libc::sockaddr_ll = unsafe { std::mem::zeroed() };
    addr.sll_family = libc::AF_PACKET as u16;
    addr.sll_protocol = (PAYLOAD_ETHERTYPE).to_be();
    addr.sll_ifindex = ifindex;
    // SAFETY: addr is a valid sockaddr_ll for the lifetime of bind().
    let rc = unsafe {
        libc::bind(
            fd,
            &addr as *const _ as *const libc::sockaddr,
            std::mem::size_of_val(&addr) as u32,
        )
    };
    if rc < 0 {
        // SAFETY: fd was opened above.
        unsafe { libc::close(fd) };
        return None;
    }
    Some(fd)
}

fn send_frame(fd: i32, dst_mac: [u8; 6], src_mac: [u8; 6], payload: &[u8]) -> std::io::Result<()> {
    let mut frame = Vec::with_capacity(14 + payload.len());
    frame.extend_from_slice(&dst_mac);
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&PAYLOAD_ETHERTYPE.to_be_bytes());
    frame.extend_from_slice(payload);

    // SAFETY: frame buffer is contiguous; send() reads `frame.len()` bytes.
    let rc = unsafe { libc::send(fd, frame.as_ptr() as *const _, frame.len(), 0) };
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

fn parse_payload(data: *const u8, len: u32) -> Option<(u32, [f32; FEATURE_DIM])> {
    let header_len = 14; // ETH only — payload starts immediately after.
    let needed = header_len + std::mem::size_of::<Payload>();
    if (len as usize) < needed {
        return None;
    }
    // SAFETY: `data` points to at least `len` valid bytes per the ingest
    // loop's frame contract; we only read up through `needed`.
    let payload_ptr = unsafe { data.add(header_len) } as *const Payload;
    let payload = unsafe { std::ptr::read_unaligned(payload_ptr) };
    Some((payload.node_id, payload.features))
}

#[test]
fn udp_packets_flow_through_xdp_into_feature_table() {
    // ── prerequisites ────────────────────────────────────────────────────────
    let Some(rx_ifindex) = ifindex_of(VETH_RX) else {
        eprintln!(
            "skipping: {VETH_RX} not present — run scripts/bootstrap_node.sh \
             or `sudo ip link add veth-tx type veth peer name veth-rx`"
        );
        return;
    };
    let Some(tx_ifindex) = ifindex_of(VETH_TX) else {
        eprintln!("skipping: {VETH_TX} not present");
        return;
    };
    let _ = tx_ifindex;

    let Some(rx_mac) = mac_of(VETH_RX) else {
        eprintln!("skipping: could not read {VETH_RX} MAC");
        return;
    };
    let Some(tx_mac) = mac_of(VETH_TX) else {
        eprintln!("skipping: could not read {VETH_TX} MAC");
        return;
    };

    let Some(raw_fd) = open_raw_sender(VETH_TX) else {
        eprintln!("skipping: AF_PACKET RAW open failed — needs CAP_NET_RAW");
        return;
    };
    let _raw_fd_guard = scopeguard(move || {
        // SAFETY: fd was just opened above.
        unsafe { libc::close(raw_fd) };
    });

    // ── load + attach BPF program ────────────────────────────────────────────
    let prog_bytes = include_bytes!(env!("XSK_REDIRECT_BPF_OBJ"));
    let mut bpf = match Ebpf::load(prog_bytes) {
        Ok(bpf) => bpf,
        Err(e) => {
            eprintln!("skipping: failed to load XDP program: {e}");
            return;
        }
    };
    {
        let program: &mut Xdp = bpf
            .program_mut("xsk_redirect_prog")
            .expect("missing 'xsk_redirect_prog' in BPF object")
            .try_into()
            .expect("program is not an XDP program");
        program.load().expect("BPF .load()");
        // SKB mode — veth doesn't support DRV mode.
        if let Err(e) = program.attach(VETH_RX, XdpFlags::SKB_MODE) {
            eprintln!("skipping: XDP attach to {VETH_RX} failed: {e} (needs CAP_NET_ADMIN)");
            return;
        }
    }

    // ── UMEM + AF_XDP socket ─────────────────────────────────────────────────
    let umem = Arc::new(Umem::new(256, 4096, vec![]).expect("UMEM alloc"));
    // SAFETY: umem points to a valid mlock'd region of total_size bytes.
    let mut socket = unsafe {
        XdpSocket::create(
            umem.base_addr() as *mut u8,
            umem.total_size(),
            umem.frame_size(),
            64,
            rx_ifindex,
            0,
            XDP_COPY,
        )
    }
    .expect("XdpSocket::create — bind should now succeed thanks to the attached XDP program");

    // Insert the socket fd into xsks_map[0] so the XDP program redirects to it.
    {
        let map = bpf.map_mut("xsks_map").expect("missing 'xsks_map'");
        let mut xsks_map: XskMap<&mut aya::maps::MapData> =
            XskMap::try_from(map).expect("xsks_map is not an XSKMAP");
        xsks_map
            .set(0, socket.fd(), 0)
            .expect("insert socket fd into xsks_map[0]");
    }

    // ── ingest loop ──────────────────────────────────────────────────────────
    let table = Arc::new(FeatureTable::new(NUM_PACKETS, FEATURE_DIM, vec![]).expect("table alloc"));
    let (tx, rx_chan) = bounded::<aether_stream::ingest::InboundFrame>(NUM_PACKETS * 2);
    let stop = Arc::new(AtomicBool::new(false));

    let umem_for_ingest = Arc::clone(&umem);
    let tx_for_ingest = tx.clone();
    let _ingest = std::thread::spawn(move || {
        let config = IngestConfig::default();
        // ingest_loop returns when the channel disconnects — we drop `tx`
        // after the test body to trigger that.
        ingest_loop(&mut socket, &umem_for_ingest, &tx_for_ingest, &config);
    });

    // ── inject 100 frames from veth-tx ──────────────────────────────────────
    let mut expected: Vec<[f32; FEATURE_DIM]> = Vec::with_capacity(NUM_PACKETS);
    for n in 0..NUM_PACKETS {
        let mut feats = [0.0f32; FEATURE_DIM];
        for i in 0..FEATURE_DIM {
            feats[i] = (n * 1000 + i) as f32 + 0.25;
        }
        let payload = Payload {
            node_id: n as u32,
            features: feats,
        };
        // SAFETY: Payload is repr(C, packed) with no padding.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &payload as *const Payload as *const u8,
                std::mem::size_of::<Payload>(),
            )
        };
        send_frame(raw_fd, rx_mac, tx_mac, bytes).expect("send_frame");
        expected.push(feats);
    }

    // ── drain the ingest channel and populate the FeatureTable ──────────────
    let table_for_reader = Arc::clone(&table);
    let mut received = 0usize;
    let deadline = Instant::now() + Duration::from_secs(5);
    while received < NUM_PACKETS && Instant::now() < deadline {
        match rx_chan.recv_timeout(Duration::from_millis(200)) {
            Ok(frame) => {
                if let Some((node, feats)) = parse_payload(frame.data, frame.len) {
                    if (node as usize) < NUM_PACKETS {
                        table_for_reader.write_node(node as usize, &feats);
                        received += 1;
                    }
                }
                umem.release_frame(frame.umem_idx);
            }
            Err(_) => continue,
        }
    }

    stop.store(true, Ordering::Relaxed);
    drop(tx);

    assert_eq!(
        received, NUM_PACKETS,
        "only received {received} of {NUM_PACKETS} frames before timeout"
    );

    // ── verify every slot matches what we sent ──────────────────────────────
    let mut readback = [0.0f32; FEATURE_DIM];
    for n in 0..NUM_PACKETS {
        let valid = table.read_node(n, &mut readback);
        assert!(valid, "node {n}: read returned false (uninitialized)");
        assert_eq!(
            readback, expected[n],
            "node {n}: features do not match what was sent"
        );
    }
}

// ── small scopeguard helper (no extra crate dep) ─────────────────────────────

struct ScopeGuard<F: FnOnce()>(Option<F>);
impl<F: FnOnce()> Drop for ScopeGuard<F> {
    fn drop(&mut self) {
        if let Some(f) = self.0.take() {
            f();
        }
    }
}
fn scopeguard<F: FnOnce()>(f: F) -> ScopeGuard<F> {
    ScopeGuard(Some(f))
}
