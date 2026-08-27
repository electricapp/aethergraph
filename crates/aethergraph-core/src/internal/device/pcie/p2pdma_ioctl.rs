//! K3.1 userspace client for `/dev/aether_p2pdma`.
//!
//! Talks to the out-of-tree module in `modules/aether_p2pdma/`. Pure layout
//! tests run everywhere; the ioctl path is Linux-only and returns
//! [`std::io::ErrorKind::Unsupported`] when the device node is missing.

use super::{P2pdmaPath, P2pdmaPolicy};
use std::ffi::CString;
use std::fs::OpenOptions;
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;

const AETHER_P2PDMA_BDF_LEN: usize = 32;
const AETHER_P2PDMA_IOCTL_MAGIC: u8 = 0xAE;
const AETHER_P2PDMA_VALIDATE_NR: u8 = 1;

const AETHER_P2PDMA_OK: u32 = 0;
const AETHER_P2PDMA_TOO_FAR: u32 = 1;
const AETHER_P2PDMA_NO_IOMMU: u32 = 2;
const AETHER_P2PDMA_ACS_REDIRECTED: u32 = 3;
const AETHER_P2PDMA_UNSUPPORTED: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy)]
struct AetherP2pdmaReq {
    producer_bdf: [u8; AETHER_P2PDMA_BDF_LEN],
    consumer_bdf: [u8; AETHER_P2PDMA_BDF_LEN],
    dmabuf_fd: i32,
    maximum_distance: u32,
    require_iommu: u8,
    _pad: [u8; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AetherP2pdmaResp {
    status: u32,
    distance: u32,
    peer_bus_addr: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AetherP2pdmaIoctl {
    req: AetherP2pdmaReq,
    resp: AetherP2pdmaResp,
}

fn encode_bdf(bdf: &str) -> io::Result<[u8; AETHER_P2PDMA_BDF_LEN]> {
    let c = CString::new(bdf).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let bytes = c.as_bytes_with_nul();
    if bytes.len() > AETHER_P2PDMA_BDF_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "PCI BDF string too long",
        ));
    }
    let mut out = [0u8; AETHER_P2PDMA_BDF_LEN];
    out[..bytes.len()].copy_from_slice(bytes);
    Ok(out)
}

fn classify_resp(resp: AetherP2pdmaResp, policy: P2pdmaPolicy) -> (P2pdmaPath, Option<u64>) {
    let path = match resp.status {
        AETHER_P2PDMA_OK => P2pdmaPath::Ok {
            distance: resp.distance,
        },
        AETHER_P2PDMA_TOO_FAR => P2pdmaPath::TooFar {
            distance: resp.distance,
            maximum: policy.maximum_distance,
        },
        AETHER_P2PDMA_NO_IOMMU => P2pdmaPath::NoIommu,
        AETHER_P2PDMA_ACS_REDIRECTED => P2pdmaPath::AcsRedirected,
        _ => P2pdmaPath::Unsupported,
    };
    let bus = matches!(path, P2pdmaPath::Ok { .. }).then_some(resp.peer_bus_addr);
    (path, bus)
}

/// Result of a live `/dev/aether_p2pdma` validate ioctl.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct P2pdmaValidateResult {
    pub path: P2pdmaPath,
    /// Peer bus address when [`P2pdmaPath::Ok`]; otherwise `None`.
    pub peer_bus_addr: Option<u64>,
}

/// Open the misc device and validate a producer→consumer peer-DMA path.
///
/// `dmabuf_fd` must remain open across the ioctl. `producer_bdf` /
/// `consumer_bdf` accept `BB:DD.F` or `DDDD:BB:DD.F`.
pub fn validate_p2pdma_path(
    device: &Path,
    producer_bdf: &str,
    consumer_bdf: &str,
    dmabuf_fd: i32,
    policy: P2pdmaPolicy,
) -> io::Result<P2pdmaValidateResult> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (device, producer_bdf, consumer_bdf, dmabuf_fd, policy);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "aether_p2pdma ioctl is Linux-only",
        ))
    }

    #[cfg(target_os = "linux")]
    {
        if !device.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "{} missing — load modules/aether_p2pdma.ko",
                    device.display()
                ),
            ));
        }
        let file = OpenOptions::new().read(true).write(true).open(device)?;
        let mut msg = AetherP2pdmaIoctl {
            req: AetherP2pdmaReq {
                producer_bdf: encode_bdf(producer_bdf)?,
                consumer_bdf: encode_bdf(consumer_bdf)?,
                dmabuf_fd,
                maximum_distance: policy.maximum_distance,
                require_iommu: u8::from(policy.require_iommu),
                _pad: [0; 3],
            },
            resp: AetherP2pdmaResp::default(),
        };

        // _IOWR(0xAE, 1, struct aether_p2pdma_ioctl)
        let dir: u64 = 3; // _IOC_READ|_IOC_WRITE
        let typ = u64::from(AETHER_P2PDMA_IOCTL_MAGIC);
        let nr = u64::from(AETHER_P2PDMA_VALIDATE_NR);
        let size = std::mem::size_of::<AetherP2pdmaIoctl>() as u64;
        let request = (dir << 30) | (typ << 8) | nr | (size << 16);

        // SAFETY: msg is a plain POD matching the kernel UAPI; fd is open.
        let rc = unsafe { libc::ioctl(file.as_raw_fd(), request as _, &mut msg as *mut _) };
        if rc < 0 {
            return Err(io::Error::last_os_error());
        }
        let (path, peer_bus_addr) = classify_resp(msg.resp, policy);
        Ok(P2pdmaValidateResult {
            path,
            peer_bus_addr,
        })
    }
}

/// Convenience: validate against `/dev/aether_p2pdma`.
pub fn validate_p2pdma_path_default(
    producer_bdf: &str,
    consumer_bdf: &str,
    dmabuf_fd: i32,
    policy: P2pdmaPolicy,
) -> io::Result<P2pdmaValidateResult> {
    validate_p2pdma_path(
        Path::new("/dev/aether_p2pdma"),
        producer_bdf,
        consumer_bdf,
        dmabuf_fd,
        policy,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_maps_kernel_status_codes() {
        let policy = P2pdmaPolicy {
            maximum_distance: 2,
            require_iommu: true,
        };
        let (path, bus) = classify_resp(
            AetherP2pdmaResp {
                status: AETHER_P2PDMA_OK,
                distance: 1,
                peer_bus_addr: 0xdead_0000,
            },
            policy,
        );
        assert_eq!(path, P2pdmaPath::Ok { distance: 1 });
        assert_eq!(bus, Some(0xdead_0000));
        assert_eq!(
            classify_resp(
                AetherP2pdmaResp {
                    status: AETHER_P2PDMA_UNSUPPORTED,
                    ..Default::default()
                },
                policy
            )
            .0,
            P2pdmaPath::Unsupported
        );
    }

    #[test]
    fn bdf_encoder_rejects_interior_nul() {
        assert!(encode_bdf("00:01.0").is_ok());
        assert!(encode_bdf("00:01.0\0x").is_err());
    }
}
