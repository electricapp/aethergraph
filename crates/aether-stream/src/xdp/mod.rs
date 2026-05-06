//! AF_XDP socket subsystem — kernel-bypass packet I/O.
//!
//! Provides raw access to NIC queues via AF_XDP sockets, bypassing the
//! kernel network stack entirely for minimum-latency ingestion.

pub mod constants;
pub mod rings;
pub mod socket;

pub use rings::XdpRing;
pub use socket::XdpSocket;
