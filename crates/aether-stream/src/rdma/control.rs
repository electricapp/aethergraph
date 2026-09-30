//! TCP control plane for RDMA advertisement and QP endpoint exchange.
//!
//! GPU nodes connect via TCP, exchange QP endpoints for bidirectional
//! RDMA connection, then perform one-sided RDMA reads directly into
//! GPU memory to fetch node features at <5μs without waking the CPU.
//!
//! The advertisement is parsed into a [`RemoteTable`] the moment it
//! arrives; nothing past [`connect_with_qp`] sees the raw schema.

use crate::feature_table::FeatureSchema;
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

/// Maximum control plane message size (64 KiB). Prevents DoS from
/// malicious length prefixes causing multi-GB allocations.
const MAX_MSG_SIZE: usize = 64 * 1024;

/// Wall-clock bound on a whole handshake, however the peer paces its
/// bytes. Keeps a slow or stalled client from holding the single-threaded
/// accept loop.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

#[cfg(feature = "rdma")]
use super::context::RdmaContext;
#[cfg(feature = "rdma")]
use super::layout::RemoteTable;
#[cfg(feature = "rdma")]
use super::qp::{DEFAULT_QP_CAP, QpEndpoint, RESPONDER_QP_CAP, RdmaQp};

/// Advertisement sent to GPU nodes over the TCP control channel.
///
/// The region exposed for one-sided reads is bounded by `schema.node_count`
/// slots of `schema.slot_size` bytes starting at `base_addr`. Clients parse
/// it into a [`RemoteTable`], which proves that span and checks every node
/// id against it. No separate region-length field is carried because
/// `schema.node_count` already pins the logical slot count and the server
/// registers at least `node_count * slot_size` bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdmaAdvertisement {
    /// Base virtual address of the feature table (for computing RDMA offsets).
    pub base_addr: u64,
    /// Remote key from `ibv_reg_mr` — needed for RDMA reads.
    pub rkey: u32,
    /// Feature table layout so the reader knows slot sizes and offsets.
    pub schema: FeatureSchema,
}

/// Server-side response: advertisement + server's QP endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ServerHello {
    advertisement: RdmaAdvertisement,
    #[cfg(feature = "rdma")]
    server_endpoint: QpEndpoint,
}

/// Client-side response: client's QP endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClientHello {
    #[cfg(feature = "rdma")]
    client_endpoint: QpEndpoint,
}

/// Server's last word: whether its QP reached RTR against the client's
/// endpoint. A client posts nothing until it has read `Ok`.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum ServerAck {
    Ok,
    Err(String),
}

/// Point `conn`'s socket timeout at what is left of `deadline`.
fn arm_timeout(conn: &TcpStream, deadline: Instant, write: bool) -> io::Result<()> {
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "control-plane handshake deadline passed",
        ));
    }
    if write {
        conn.set_write_timeout(Some(left))
    } else {
        conn.set_read_timeout(Some(left))
    }
}

/// `read_exact`, bounded by `deadline` across all the reads it takes rather
/// than per syscall.
fn read_exact_by(conn: &mut TcpStream, buf: &mut [u8], deadline: Instant) -> io::Result<()> {
    let mut filled = 0;
    while filled < buf.len() {
        arm_timeout(conn, deadline, false)?;
        match conn.read(&mut buf[filled..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "peer closed mid-message",
                ));
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// `write_all`, bounded by `deadline` across all the writes it takes.
fn write_all_by(conn: &mut TcpStream, mut buf: &[u8], deadline: Instant) -> io::Result<()> {
    while !buf.is_empty() {
        arm_timeout(conn, deadline, true)?;
        match conn.write(buf) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "peer stopped accepting bytes",
                ));
            }
            Ok(n) => buf = &buf[n..],
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Send a length-prefixed JSON message over a TCP stream.
pub(crate) fn send_msg(conn: &mut TcpStream, payload: &[u8], deadline: Instant) -> io::Result<()> {
    // The wire prefix is a u32; a payload ≥ 4 GiB cannot be framed. Error
    // rather than truncate the length (which would desync the stream and let
    // the peer read a short, attacker-influenced body).
    let len = u32::try_from(payload.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "control message too large to frame: {} bytes",
                payload.len()
            ),
        )
    })?;
    write_all_by(conn, &len.to_le_bytes(), deadline)?;
    write_all_by(conn, payload, deadline)
}

/// Receive a length-prefixed JSON message from a TCP stream.
pub(crate) fn recv_msg(conn: &mut TcpStream, deadline: Instant) -> io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    read_exact_by(conn, &mut len_buf, deadline)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_MSG_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("message too large: {len} bytes (max {MAX_MSG_SIZE})"),
        ));
    }
    let mut buf = vec![0u8; len];
    read_exact_by(conn, &mut buf, deadline)?;
    Ok(buf)
}

fn to_json<T: Serialize>(v: &T) -> io::Result<Vec<u8>> {
    serde_json::to_vec(v).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn from_json<'a, T: Deserialize<'a>>(buf: &'a [u8]) -> io::Result<T> {
    serde_json::from_slice(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// A fresh deadline for one handshake.
pub(crate) fn handshake_deadline() -> Instant {
    Instant::now() + HANDSHAKE_TIMEOUT
}

/// Serve RDMA advertisements to connecting GPU nodes.
///
/// Blocks forever, accepting connections and sending the advertisement.
/// Run this in a dedicated thread.
///
/// Without the `rdma` feature, this sends the advertisement only (no QP exchange).
///
/// # Trust model
/// This handshake is UNAUTHENTICATED. The advertisement carries the MR `rkey`
/// and `base_addr`, so any TCP peer that connects can obtain everything needed
/// to issue one-sided RDMA reads against the registered region — i.e. the
/// region is effectively readable by every host that can reach this port. The
/// region is registered `IBV_ACCESS_REMOTE_READ` only (no remote write/atomic),
/// so exposure is limited to reading feature bytes. Deploy this only on a
/// trusted fabric / private network, or front it with an authenticated channel.
pub fn serve_control_plane(bind_addr: &str, advertisement: &RdmaAdvertisement) -> io::Result<()> {
    let listener = TcpListener::bind(bind_addr)?;
    tracing::info!(addr = bind_addr, "RDMA control plane listening");

    let payload = to_json(advertisement)?;

    for stream in listener.incoming() {
        match stream {
            Ok(mut conn) => {
                let peer = conn.peer_addr().ok();
                tracing::info!(?peer, "GPU node connected to control plane");
                if let Err(e) = send_msg(&mut conn, &payload, handshake_deadline()) {
                    tracing::warn!(?peer, error = %e, "Failed to send advertisement");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to accept connection");
            }
        }
    }

    Ok(())
}

/// Serve RDMA advertisements with bidirectional QP endpoint exchange.
///
/// For each connecting client:
/// 1. Creates a server-side responder QP
/// 2. Sends advertisement + server QP endpoint
/// 3. Receives client QP endpoint
/// 4. Connects the server QP to the client and acknowledges, so the
///    client never posts against a QP not yet ready to receive
///
/// Blocks forever. Run in a dedicated thread.
///
/// # Concurrency and lifetime
/// The accept loop is single-threaded: each handshake is processed to
/// completion before the next connection is accepted, so a stalled client
/// blocks others for at most `HANDSHAKE_TIMEOUT`. Each server QP must outlive
/// the RDMA connection it backs, so QPs are retained in `active_qps` for the
/// server's entire lifetime; the table grows monotonically with the number of
/// clients ever served and is freed only when this function returns (server
/// shutdown). Server QPs post nothing, so they share the context's CQ.
///
/// # Trust model
/// Like [`serve_control_plane`], the handshake is UNAUTHENTICATED — any peer
/// that connects receives the MR `rkey` + `base_addr` and can read the
/// registered (REMOTE_READ-only) region. Deploy on a trusted fabric only.
#[cfg(feature = "rdma")]
pub fn serve_control_plane_with_qp(
    bind_addr: &str,
    advertisement: &RdmaAdvertisement,
    ctx: &RdmaContext,
) -> io::Result<()> {
    let listener = TcpListener::bind(bind_addr)?;
    tracing::info!(
        addr = bind_addr,
        "RDMA control plane (QP exchange) listening"
    );

    // Hold connected QPs alive for the server's lifetime.
    let mut active_qps: Vec<RdmaQp> = Vec::new();

    for stream in listener.incoming() {
        match stream {
            Ok(mut conn) => {
                let peer = conn.peer_addr().ok();
                tracing::info!(?peer, "GPU node connected for QP exchange");
                match serve_one(&mut conn, advertisement, ctx) {
                    Ok(qp) => {
                        tracing::info!(?peer, "QP exchange complete — RDMA reads enabled");
                        active_qps.push(qp);
                    }
                    Err(e) => tracing::warn!(?peer, error = %e, "QP exchange failed"),
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to accept connection");
            }
        }
    }

    Ok(())
}

#[cfg(feature = "rdma")]
fn serve_one(
    conn: &mut TcpStream,
    advertisement: &RdmaAdvertisement,
    ctx: &RdmaContext,
) -> io::Result<RdmaQp> {
    let deadline = handshake_deadline();
    let server_qp = RdmaQp::create(ctx, &RESPONDER_QP_CAP)?;

    let hello = ServerHello {
        advertisement: advertisement.clone(),
        server_endpoint: server_qp.endpoint(ctx),
    };
    send_msg(conn, &to_json(&hello)?, deadline)?;

    let client_hello: ClientHello = from_json(&recv_msg(conn, deadline)?)?;
    let connected = server_qp.connect(ctx, &client_hello.client_endpoint);
    let ack = match &connected {
        Ok(()) => ServerAck::Ok,
        Err(e) => ServerAck::Err(e.to_string()),
    };
    // Tell the client either way; a send failure only matters on success.
    let sent = send_msg(conn, &to_json(&ack)?, deadline);
    connected?;
    sent?;
    Ok(server_qp)
}

/// Fetch the RDMA advertisement from the control plane.
///
/// Used by GPU nodes to discover the feature table's memory layout.
pub fn fetch_advertisement(addr: &str) -> io::Result<RdmaAdvertisement> {
    let mut conn = TcpStream::connect(addr)?;
    let buf = recv_msg(&mut conn, handshake_deadline())?;
    from_json(&buf)
}

/// Fetch the advertisement, parse it, and exchange QP endpoints with the
/// server.
///
/// Returns the validated remote table and the connected client QP, once the
/// server has confirmed its side reached RTR — the QP is ready to post.
#[cfg(feature = "rdma")]
pub fn connect_with_qp(addr: &str, ctx: &RdmaContext) -> io::Result<(RemoteTable, RdmaQp)> {
    let mut conn = TcpStream::connect(addr)?;
    let deadline = handshake_deadline();

    let server_hello: ServerHello = from_json(&recv_msg(&mut conn, deadline)?)?;
    let adv = &server_hello.advertisement;
    let table = RemoteTable::parse(adv.base_addr, adv.rkey, &adv.schema)?;

    let client_qp = RdmaQp::create(ctx, &DEFAULT_QP_CAP)?;
    let client_hello = ClientHello {
        client_endpoint: client_qp.endpoint(ctx),
    };
    send_msg(&mut conn, &to_json(&client_hello)?, deadline)?;
    client_qp.connect(ctx, &server_hello.server_endpoint)?;

    match from_json::<ServerAck>(&recv_msg(&mut conn, deadline)?)? {
        ServerAck::Ok => Ok((table, client_qp)),
        ServerAck::Err(e) => Err(io::Error::other(format!(
            "server could not connect its QP: {e}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// A peer that trickles one byte per poll never trips a per-syscall
    /// timeout; the whole-message deadline still fires.
    #[test]
    fn trickling_peer_hits_the_message_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let writer = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            // Announce 8 bytes, then dribble them slower than the deadline.
            let _ = s.write_all(&8u32.to_le_bytes());
            for b in 0..8u8 {
                std::thread::sleep(Duration::from_millis(60));
                if s.write_all(&[b]).is_err() {
                    return;
                }
            }
        });
        let mut conn = TcpStream::connect(addr).unwrap();
        let deadline = Instant::now() + Duration::from_millis(200);
        let err = recv_msg(&mut conn, deadline).unwrap_err();
        assert!(
            matches!(
                err.kind(),
                io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
            ),
            "{err:?}"
        );
        drop(conn);
        writer.join().unwrap();
    }

    #[test]
    fn framed_messages_round_trip() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let echo = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let got = recv_msg(&mut s, handshake_deadline()).unwrap();
            send_msg(&mut s, &got, handshake_deadline()).unwrap();
        });
        let mut conn = TcpStream::connect(addr).unwrap();
        send_msg(&mut conn, b"hello", handshake_deadline()).unwrap();
        assert_eq!(recv_msg(&mut conn, handshake_deadline()).unwrap(), b"hello");
        echo.join().unwrap();
    }

    #[test]
    fn oversized_length_prefix_is_rejected_before_allocating() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let writer = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let _ = s.write_all(&u32::MAX.to_le_bytes());
        });
        let mut conn = TcpStream::connect(addr).unwrap();
        let err = recv_msg(&mut conn, handshake_deadline()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        writer.join().unwrap();
    }
}
