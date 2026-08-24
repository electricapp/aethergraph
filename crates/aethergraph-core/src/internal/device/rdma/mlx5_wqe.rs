//! K2.1: mlx5 RDMA READ WQE segments from `mlx5_ifc`.
//!
//! This 48-byte builder is ordered as control (bytes 0..16), data (16..32),
//! and remote-address (32..48), the segment order consumed by the IBGDA path.
//! `qpn_ds` stores the QP number in bits 31:8 and the 16-byte segment count in
//! bits 5:0. The runtime doorbell record and BlueFlame write remain hardware
//! integration concerns.

/// mlx5 transport opcode for RDMA READ.
pub const MLX5_OPCODE_RDMA_READ: u8 = 0x10;
const WQE_SEGMENTS: u32 = 3;

/// A packed mlx5 RDMA READ WQE: ctrl, data, then remote-address segment.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mlx5RdmaReadWqe {
    /// `opmod_idx_opcode`, big-endian: opcode occupies bits 7:0.
    pub ctrl_opmod_idx_opcode: [u8; 4],
    /// `qpn_ds`, big-endian: QPN bits 31:8, DS bits 5:0.
    pub ctrl_qpn_ds: [u8; 4],
    pub ctrl_signature: u8,
    pub ctrl_reserved: [u8; 3],
    pub ctrl_imm: [u8; 4],
    /// Data segment: byte count, local key, local address.
    pub byte_count: [u8; 4],
    pub lkey: [u8; 4],
    pub local_address: [u8; 8],
    /// Remote-address segment: remote address, remote key, reserved.
    pub remote_address: [u8; 8],
    pub rkey: [u8; 4],
    pub remote_reserved: [u8; 4],
}

const _: () = assert!(core::mem::size_of::<Mlx5RdmaReadWqe>() == 48);

impl Mlx5RdmaReadWqe {
    /// Build one-signature-less RDMA READ WQE.
    ///
    /// `wqe_counter` is the producer index assigned by the QP; `qpn` must fit
    /// the 24-bit field prescribed by the mlx5 control segment.
    pub fn new(
        qpn: u32,
        wqe_counter: u16,
        local_address: u64,
        lkey: u32,
        byte_count: u32,
        remote_address: u64,
        rkey: u32,
    ) -> Self {
        assert!(qpn <= 0x00ff_ffff, "mlx5 QPN exceeds 24 bits");
        let opmod_idx_opcode = (u32::from(wqe_counter) << 8) | u32::from(MLX5_OPCODE_RDMA_READ);
        let qpn_ds = (qpn << 8) | WQE_SEGMENTS;
        Self {
            ctrl_opmod_idx_opcode: opmod_idx_opcode.to_be_bytes(),
            ctrl_qpn_ds: qpn_ds.to_be_bytes(),
            ctrl_signature: 0,
            ctrl_reserved: [0; 3],
            ctrl_imm: [0; 4],
            byte_count: byte_count.to_be_bytes(),
            lkey: lkey.to_be_bytes(),
            local_address: local_address.to_be_bytes(),
            remote_address: remote_address.to_be_bytes(),
            rkey: rkey.to_be_bytes(),
            remote_reserved: [0; 4],
        }
    }

    /// The encoded 24-bit QP number.
    pub fn qpn(self) -> u32 {
        u32::from_be_bytes(self.ctrl_qpn_ds) >> 8
    }

    /// Number of 16-byte segments consumed by this WQE.
    pub fn segment_count(self) -> u8 {
        (u32::from_be_bytes(self.ctrl_qpn_ds) & 0x3f) as u8
    }
}

// IBGDA path: [`super::ibgda::IbgdaQueue`] + aether-stream GPU WQE kernel.
// TODO(HARDWARE): On the bare-metal ConnectX rig, submit this WQE from GPU
// memory through an IBGDA doorbell and compare completion/data semantics with
// an equivalent NVSHMEM RDMA READ on the same QP.

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    #[test]
    fn rdma_read_wqe_has_mlx5_segment_offsets() {
        assert_eq!(size_of::<Mlx5RdmaReadWqe>(), 48);
        assert_eq!(offset_of!(Mlx5RdmaReadWqe, ctrl_opmod_idx_opcode), 0);
        assert_eq!(offset_of!(Mlx5RdmaReadWqe, byte_count), 16);
        assert_eq!(offset_of!(Mlx5RdmaReadWqe, remote_address), 32);
    }

    #[test]
    fn builder_packs_control_data_and_remote_segments() {
        let wqe = Mlx5RdmaReadWqe::new(
            0x12_3456,
            0x42,
            0x1000_2000,
            0x1111_2222,
            4096,
            0x9000_a000,
            0x3333_4444,
        );
        assert_eq!(wqe.qpn(), 0x12_3456);
        assert_eq!(wqe.segment_count(), 3);
        assert_eq!(u32::from_be_bytes(wqe.ctrl_opmod_idx_opcode), 0x0000_4210);
        assert_eq!(u32::from_be_bytes(wqe.byte_count), 4096);
        assert_eq!(u64::from_be_bytes(wqe.remote_address), 0x9000_a000);
    }
}
