//! TCP control plane for RDMA advertisement and QP endpoint exchange.
//!
//! GPU nodes connect via TCP, exchange QP endpoints for bidirectional
//! RDMA connection, then perform one-sided RDMA reads directly into
//! GPU memory to fetch node features at <5μs without waking the CPU.

use crate::feature_table::FeatureSchema;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

/// Maximum control plane message size (64 KiB). Prevents DoS from
/// malicious length prefixes causing multi-GB allocations.
const MAX_MSG_SIZE: usize = 64 * 1024;

/// TCP read/write timeout for control plane connections.
/// Prevents a dead client from blocking the single-threaded accept loop.
const TCP_TIMEOUT: Duration = Duration::from_secs(10);

#[cfg(feature = "rdma")]
use super::context::RdmaContext;
#[cfg(feature = "rdma")]
use super::qp::{QpEndpoint, RdmaQp, DEFAULT_QP_CAP};

/// Advertisement sent to GPU nodes over the TCP control channel.
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

/// Send a length-prefixed JSON message over a TCP stream.
fn send_msg(conn: &mut TcpStream, payload: &[u8]) -> std::io::Result<()> {
    let len = (payload.len() as u32).to_le_bytes();
    conn.write_all(&len)?;
    conn.write_all(payload)?;
    Ok(())
}

/// Receive a length-prefixed JSON message from a TCP stream.
fn recv_msg(conn: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    conn.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_MSG_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("message too large: {len} bytes (max {MAX_MSG_SIZE})"),
        ));
    }
    let mut buf = vec![0u8; len];
    conn.read_exact(&mut buf)?;
    Ok(buf)
}

/// Set read/write timeouts on a TCP connection.
fn set_timeouts(conn: &TcpStream) -> std::io::Result<()> {
    conn.set_read_timeout(Some(TCP_TIMEOUT))?;
    conn.set_write_timeout(Some(TCP_TIMEOUT))?;
    Ok(())
}

/// Serve RDMA advertisements to connecting GPU nodes.
///
/// Blocks forever, accepting connections and sending the advertisement.
/// Run this in a dedicated thread.
///
/// Without the `rdma` feature, this sends the advertisement only (no QP exchange).
pub fn serve_control_plane(
    bind_addr: &str,
    advertisement: &RdmaAdvertisement,
) -> std::io::Result<()> {
    let listener = TcpListener::bind(bind_addr)?;
    tracing::info!(addr = bind_addr, "RDMA control plane listening");

    let payload = serde_json::to_vec(advertisement)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    for stream in listener.incoming() {
        match stream {
            Ok(mut conn) => {
                let peer = conn.peer_addr().ok();
                if let Err(e) = set_timeouts(&conn) {
                    tracing::warn!(?peer, error = %e, "Failed to set TCP timeouts");
                    continue;
                }
                tracing::info!(?peer, "GPU node connected to control plane");

                // Length-prefixed JSON
                if send_msg(&mut conn, &payload).is_err() {
                    tracing::warn!(?peer, "Failed to send advertisement");
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
/// 1. Creates a server-side QP
/// 2. Sends advertisement + server QP endpoint
/// 3. Receives client QP endpoint
/// 4. Connects the server QP to the client
///
/// Blocks forever. Run in a dedicated thread.
#[cfg(feature = "rdma")]
pub fn serve_control_plane_with_qp(
    bind_addr: &str,
    advertisement: &RdmaAdvertisement,
    ctx: &RdmaContext,
) -> std::io::Result<()> {
    let listener = TcpListener::bind(bind_addr)?;
    tracing::info!(addr = bind_addr, "RDMA control plane (QP exchange) listening");

    // Hold connected QPs alive for the server's lifetime.
    let mut active_qps: Vec<RdmaQp> = Vec::new();

    for stream in listener.incoming() {
        match stream {
            Ok(mut conn) => {
                let peer = conn.peer_addr().ok();
                if let Err(e) = set_timeouts(&conn) {
                    tracing::warn!(?peer, error = %e, "Failed to set TCP timeouts");
                    continue;
                }
                tracing::info!(?peer, "GPU node connected for QP exchange");

                // Create a server-side QP for this client
                let server_qp = match RdmaQp::create(ctx, &DEFAULT_QP_CAP) {
                    Ok(qp) => qp,
                    Err(e) => {
                        tracing::warn!(?peer, error = %e, "Failed to create server QP");
                        continue;
                    }
                };

                // Send advertisement + server endpoint
                let hello = ServerHello {
                    advertisement: advertisement.clone(),
                    server_endpoint: server_qp.endpoint(ctx),
                };
                let payload = serde_json::to_vec(&hello)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

                if send_msg(&mut conn, &payload).is_err() {
                    tracing::warn!(?peer, "Failed to send server hello");
                    continue;
                }

                // Receive client endpoint
                let client_buf = match recv_msg(&mut conn) {
                    Ok(buf) => buf,
                    Err(e) => {
                        tracing::warn!(?peer, error = %e, "Failed to receive client hello");
                        continue;
                    }
                };

                let client_hello: ClientHello = match serde_json::from_slice(&client_buf) {
                    Ok(h) => h,
                    Err(e) => {
                        tracing::warn!(?peer, error = %e, "Invalid client hello");
                        continue;
                    }
                };

                // Connect server QP to client
                if let Err(e) = server_qp.connect(ctx, &client_hello.client_endpoint) {
                    tracing::warn!(?peer, error = %e, "Failed to connect server QP");
                    continue;
                }

                tracing::info!(?peer, "QP exchange complete — RDMA reads enabled");

                // QP must outlive the RDMA connection. Store it rather than
                // leaking via mem::forget (which exhausts HCA QP table under
                // client churn). The Vec grows monotonically — QPs are cleaned
                // up when this function returns (server shutdown).
                active_qps.push(server_qp);
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to accept connection");
            }
        }
    }

    Ok(())
}

/// Fetch the RDMA advertisement from the control plane.
///
/// Used by GPU nodes to discover the feature table's memory layout.
pub fn fetch_advertisement(addr: &str) -> std::io::Result<RdmaAdvertisement> {
    let mut conn = TcpStream::connect(addr)?;
    set_timeouts(&conn)?;
    let buf = recv_msg(&mut conn)?;
    serde_json::from_slice(&buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Fetch advertisement and exchange QP endpoints with the server.
///
/// Returns the advertisement, the connected client QP, and the remote endpoint.
#[cfg(feature = "rdma")]
pub fn connect_with_qp(
    addr: &str,
    ctx: &RdmaContext,
) -> std::io::Result<(RdmaAdvertisement, RdmaQp)> {
    let mut conn = TcpStream::connect(addr)?;
    set_timeouts(&conn)?;

    // Receive server hello
    let buf = recv_msg(&mut conn)?;
    let server_hello: ServerHello = serde_json::from_slice(&buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    // Create client QP
    let client_qp = RdmaQp::create(ctx, &DEFAULT_QP_CAP)?;

    // Send client hello
    let client_hello = ClientHello {
        client_endpoint: client_qp.endpoint(ctx),
    };
    let payload = serde_json::to_vec(&client_hello)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    send_msg(&mut conn, &payload)?;

    // Connect client QP to server
    client_qp.connect(ctx, &server_hello.server_endpoint)?;

    Ok((server_hello.advertisement, client_qp))
}
