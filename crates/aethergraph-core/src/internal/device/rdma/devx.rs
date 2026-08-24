//! K2.2 DEVX / GPU-terminated Ethernet session builder.
//!
//! Turns [`super::GpuRingPlacement`] + [`super::FlowSteeringRule`] into a
//! session description a ConnectX DEVX (or DOCA GPUNetIO) adapter can apply.
//! The apply step that talks to `libmlx5` / DEVX ioctls is behind
//! [`DevxGpuEthBackend`].

use super::{FlowSteeringRule, GpuRingPlacement};

/// QP + CQ rings that must live in GPU memory before steering is attached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevxGpuEthPlan {
    pub qp_ring: GpuRingPlacement,
    pub cq_ring: GpuRingPlacement,
    pub doorbell_record_gpu_va: u64,
    pub steering: FlowSteeringRule,
}

impl DevxGpuEthPlan {
    /// Validate placements and a non-null doorbell VA.
    pub fn new(
        qp_ring: GpuRingPlacement,
        cq_ring: GpuRingPlacement,
        doorbell_record_gpu_va: u64,
        steering: FlowSteeringRule,
    ) -> Option<Self> {
        (doorbell_record_gpu_va != 0).then_some(Self {
            qp_ring,
            cq_ring,
            doorbell_record_gpu_va,
            steering,
        })
    }
}

/// Opaque handle returned by a backend after DEVX object creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevxSessionId(pub u64);

/// Backend that materializes a [`DevxGpuEthPlan`] on a live ConnectX device.
pub trait DevxGpuEthBackend {
    type Error;

    /// Create GPU-backed QP/CQ, install flow steering, return a session id.
    fn create_session(&mut self, plan: DevxGpuEthPlan) -> Result<DevxSessionId, Self::Error>;

    /// Tear down DEVX objects for `id`.
    fn destroy_session(&mut self, id: DevxSessionId) -> Result<(), Self::Error>;
}

/// In-process backend for tests: records plans, assigns monotonic ids.
#[derive(Debug, Default)]
pub struct MockDevxBackend {
    pub created: Vec<DevxGpuEthPlan>,
    next_id: u64,
}

impl DevxGpuEthBackend for MockDevxBackend {
    type Error = &'static str;

    fn create_session(&mut self, plan: DevxGpuEthPlan) -> Result<DevxSessionId, Self::Error> {
        self.created.push(plan);
        self.next_id += 1;
        Ok(DevxSessionId(self.next_id))
    }

    fn destroy_session(&mut self, id: DevxSessionId) -> Result<(), Self::Error> {
        if id.0 == 0 || id.0 > self.next_id {
            Err("unknown session")
        } else {
            Ok(())
        }
    }
}

/// Live mlx5 DEVX backend placeholder — opens when linked on the ConnectX rig.
///
/// Methods return [`std::io::ErrorKind::Unsupported`] until `libmlx5` DEVX
/// symbols are linked (feature `mlx5dv` on aether-stream). The plan validation
/// still runs so call sites can share one code path.
#[derive(Debug, Default)]
pub struct Mlx5DevxBackend {
    pub device_name: Option<String>,
}

impl DevxGpuEthBackend for Mlx5DevxBackend {
    type Error = std::io::Error;

    fn create_session(&mut self, plan: DevxGpuEthPlan) -> Result<DevxSessionId, Self::Error> {
        let _ = plan;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Mlx5DevxBackend: link mlx5 DEVX on the ConnectX rig \
             (cuda-registered QP/CQ rings + flow table)",
        ))
    }

    fn destroy_session(&mut self, _id: DevxSessionId) -> Result<(), Self::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Mlx5DevxBackend: no live session",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::device::rdma::gpu_eth::FlowMatch;

    #[test]
    fn mock_backend_round_trips_session() {
        let qp = GpuRingPlacement::new(0x1000, 4096, 64).unwrap();
        let cq = GpuRingPlacement::new(0x2000, 4096, 16).unwrap();
        let rule = FlowSteeringRule::new(
            1,
            FlowMatch {
                ethernet_type: Some(0x0800),
                ip_protocol: Some(17),
                udp_dst_port: Some(9000),
            },
            42,
        )
        .unwrap();
        let plan = DevxGpuEthPlan::new(qp, cq, 0x3000, rule).unwrap();
        let mut backend = MockDevxBackend::default();
        let id = backend.create_session(plan).unwrap();
        assert_eq!(id, DevxSessionId(1));
        assert_eq!(backend.created.len(), 1);
        backend.destroy_session(id).unwrap();
    }
}
