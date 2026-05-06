//! GPUDirect Storage feature loading -- NVMe to GPU with no CPU bounce.
//!
//! Uses NVIDIA cuFile batch API to DMA features directly from NVMe to VRAM.
//! Requires: Linux + CUDA 11.4+ + nvidia-gds driver package.
//!
//! The caller owns the CUDA context and pre-allocates the GPU buffer;
//! this module only needs raw device pointers and the cuFile FFI.
//!
//! # Performance
//!
//! The batch API (`cuFileBatchIOSubmit`) submits all reads in a single
//! kernel crossing. For a batch of 3000 nodes this is one syscall instead
//! of 3000 individual `cuFileRead` calls. Nodes are sorted by ID before
//! submission so file offsets are monotonically increasing, maximizing
//! NVMe sequential prefetch.

use crate::features::header::{parse_feature_header, FeatureDtype};
use crate::graph::NodeId;
use anyhow::{Context, Result, ensure};
use std::ffi::c_void;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// cuFile FFI bindings
// ---------------------------------------------------------------------------

#[allow(unsafe_code)]
mod ffi {
    use std::ffi::c_void;
    use std::os::raw::c_int;

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct CUfileError {
        pub err: c_int,    // CUfileOpError
        pub cu_err: c_int, // CUresult
    }

    /// Opaque handle returned by `cuFileHandleRegister`.
    pub type CUfileHandle = *mut c_void;

    #[repr(C)]
    pub struct CUfileDescr {
        pub handle_type: c_int, // CU_FILE_HANDLE_TYPE_OPAQUE_FD = 0
        pub handle: CUfileDescrUnion,
        pub fs_ops: *const c_void, // null for default
    }

    #[repr(C)]
    pub union CUfileDescrUnion {
        pub fd: c_int,
    }

    // -- Batch I/O types --

    /// Operation type for batch I/O entries.
    pub const CUFILE_READ: c_int = 0;

    /// Completion status for a batch I/O entry.
    pub const CUFILE_BATCH_NOT_STARTED: c_int = 0;
    pub const CUFILE_BATCH_COMPLETE: c_int = 2;

    /// Per-operation descriptor for batch I/O.
    ///
    /// This matches the `CUfileIOParams_t` struct from cufile.h.
    /// Each entry describes one read or write operation.
    #[repr(C)]
    pub struct CUfileIOParams {
        pub mode: c_int,             // CUFILE_READ or CUFILE_WRITE
        pub pad: [u8; 4],           // padding for alignment
        pub fh: CUfileHandle,       // registered file handle
        pub opcode: c_int,          // internal, set to 0
        pub status: c_int,          // completion status (output)
        pub cookie: *mut c_void,    // user cookie (unused)
        pub dev_ptr_base: *mut c_void, // registered GPU buffer base
        pub file_offset: i64,       // offset in file
        pub dev_ptr_offset: i64,    // offset within GPU buffer
        pub size: usize,            // bytes to transfer
        pub bytes_done: isize,      // bytes actually transferred (output)
    }

    /// Opaque batch handle.
    pub type CUfileBatchHandle = *mut c_void;

    #[link(name = "cufile")]
    extern "C" {
        pub fn cuFileDriverOpen() -> CUfileError;
        pub fn cuFileDriverClose() -> CUfileError;

        pub fn cuFileHandleRegister(
            fh: *mut CUfileHandle,
            descr: *mut CUfileDescr,
        ) -> CUfileError;
        pub fn cuFileHandleDeregister(fh: CUfileHandle);

        pub fn cuFileBufRegister(
            dev_ptr: *const c_void,
            size: usize,
            flags: c_int,
        ) -> CUfileError;
        pub fn cuFileBufDeregister(dev_ptr: *const c_void) -> CUfileError;

        // Single-read fallback (kept for small batches where overhead matters less)
        pub fn cuFileRead(
            fh: CUfileHandle,
            dev_ptr: *mut c_void,
            size: usize,
            file_offset: i64,
            dev_offset: i64,
        ) -> isize;

        // Batch I/O API
        pub fn cuFileBatchIOSetUp(
            batch_handle: *mut CUfileBatchHandle,
            num_entries: u32,
        ) -> CUfileError;

        pub fn cuFileBatchIOSubmit(
            batch_handle: CUfileBatchHandle,
            num_entries: u32,
            io_params: *mut CUfileIOParams,
            flags: u32, // 0 for default
        ) -> CUfileError;

        pub fn cuFileBatchIOGetStatus(
            batch_handle: CUfileBatchHandle,
            num_entries: u32,
            io_params: *mut CUfileIOParams,
            num_complete: *mut u32,
        ) -> CUfileError;

        pub fn cuFileBatchIODestroy(batch_handle: CUfileBatchHandle);
    }

    pub const CU_FILE_SUCCESS: c_int = 0;
    pub const CU_FILE_HANDLE_TYPE_OPAQUE_FD: c_int = 0;
}

// ---------------------------------------------------------------------------
// Driver init / shutdown
// ---------------------------------------------------------------------------

/// Initialize the cuFile driver. Call once at startup before creating any
/// `GdsFeatureStore` instances.
#[allow(unsafe_code)]
pub fn gds_driver_open() -> Result<()> {
    let err = unsafe { ffi::cuFileDriverOpen() };
    ensure!(
        err.err == ffi::CU_FILE_SUCCESS,
        "cuFileDriverOpen failed: err={}, cu_err={}",
        err.err,
        err.cu_err,
    );
    debug!("cuFile driver initialized");
    Ok(())
}

/// Shut down the cuFile driver. Call once at process exit.
#[allow(unsafe_code)]
pub fn gds_driver_close() {
    let err = unsafe { ffi::cuFileDriverClose() };
    if err.err != ffi::CU_FILE_SUCCESS {
        warn!(
            "cuFileDriverClose returned err={}, cu_err={}",
            err.err, err.cu_err
        );
    }
}

// ---------------------------------------------------------------------------
// GdsReadResult
// ---------------------------------------------------------------------------

/// Metadata returned by a successful batch read.
///
/// The feature data lives in the pre-registered GPU buffer at `device_ptr`.
/// No host-side Vec is produced -- the caller feeds the pointer directly
/// to a CUDA kernel or cuBLAS.
pub struct GdsReadResult {
    /// Device pointer to the start of the feature data in VRAM.
    pub device_ptr: u64,
    /// Number of nodes actually loaded.
    pub num_nodes: usize,
    /// Feature dimension per node.
    pub feature_dim: usize,
    /// Data type of features in GPU memory.
    pub dtype: FeatureDtype,
}

/// Threshold: batches smaller than this use single cuFileRead calls
/// (lower per-call overhead). Larger batches use the batch API.
const BATCH_API_THRESHOLD: usize = 8;

// ---------------------------------------------------------------------------
// GdsFeatureStore
// ---------------------------------------------------------------------------

/// Feature store that loads features directly from NVMe to GPU via cuFile.
///
/// The GPU buffer is pre-allocated by the caller and registered with cuFile
/// for DMA. Each `get_batch` / `get_batch_into` call submits reads via the
/// cuFile batch API -- one kernel crossing for the entire batch instead of
/// one per node.
pub struct GdsFeatureStore {
    /// cuFile handle for the feature file.
    file_handle: ffi::CUfileHandle,
    /// Keep the standard `File` alive so the fd remains valid.
    _file: File,
    /// Number of nodes in the feature file.
    num_nodes: usize,
    /// Features per node.
    feature_dim: usize,
    /// Byte offset where feature payload starts in the file.
    features_start_offset: u64,
    /// Element data type (F32 or F16).
    dtype: FeatureDtype,
    /// Pre-allocated GPU buffer base pointer (owned by caller).
    gpu_buffer: *mut c_void,
    /// GPU buffer size in bytes.
    gpu_buffer_size: usize,
    /// Maximum batch size this store supports.
    max_batch_size: usize,
}

// SAFETY: The cuFile handle and GPU pointer are valid for the store's lifetime.
// cuFile operations are thread-safe (each cuFileRead is independent).
unsafe impl Send for GdsFeatureStore {}
// SAFETY: cuFileRead/cuFileBatchIO are safe to call concurrently from
// multiple threads on different buffer offsets.
unsafe impl Sync for GdsFeatureStore {}

impl GdsFeatureStore {
    /// Open a feature file and register a GPU buffer for DMA reads.
    ///
    /// # Arguments
    /// * `path` -- path to an AETHFEAT feature file
    /// * `gpu_device_ptr` -- raw CUDA device pointer to the pre-allocated buffer
    /// * `gpu_buffer_size` -- size of the GPU buffer in bytes
    /// * `max_batch_size` -- maximum number of nodes per batch
    ///
    /// # Safety contract
    /// The caller must ensure:
    /// - `gpu_device_ptr` is a valid CUDA device pointer
    /// - The region `[gpu_device_ptr, gpu_device_ptr + gpu_buffer_size)` is allocated
    /// - The allocation outlives this `GdsFeatureStore`
    /// - The cuFile driver has been initialized via `gds_driver_open()`
    #[allow(unsafe_code)]
    pub fn open(
        path: &Path,
        gpu_device_ptr: u64,
        gpu_buffer_size: usize,
        max_batch_size: usize,
    ) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open feature file: {}", path.display()))?;
        let header = parse_feature_header(&file)?;

        let feature_size = header.feature_dim * header.dtype.element_size();
        let required_buffer = max_batch_size
            .checked_mul(feature_size)
            .ok_or_else(|| anyhow::anyhow!("GPU buffer size overflow"))?;
        ensure!(
            gpu_buffer_size >= required_buffer,
            "GPU buffer too small: need {} bytes for {} nodes * {} feature bytes, have {}",
            required_buffer,
            max_batch_size,
            feature_size,
            gpu_buffer_size,
        );

        // Register the file with cuFile.
        let mut file_handle: ffi::CUfileHandle = std::ptr::null_mut();
        let mut descr = ffi::CUfileDescr {
            handle_type: ffi::CU_FILE_HANDLE_TYPE_OPAQUE_FD,
            handle: ffi::CUfileDescrUnion {
                fd: file.as_raw_fd(),
            },
            fs_ops: std::ptr::null(),
        };

        let err = unsafe {
            ffi::cuFileHandleRegister(&raw mut file_handle, &raw mut descr)
        };
        ensure!(
            err.err == ffi::CU_FILE_SUCCESS,
            "cuFileHandleRegister failed: err={}, cu_err={}",
            err.err,
            err.cu_err,
        );

        // Register the GPU buffer for DMA.
        let gpu_ptr = gpu_device_ptr as *mut c_void;
        let err = unsafe { ffi::cuFileBufRegister(gpu_ptr, gpu_buffer_size, 0) };
        if err.err != ffi::CU_FILE_SUCCESS {
            unsafe { ffi::cuFileHandleDeregister(file_handle) };
            anyhow::bail!(
                "cuFileBufRegister failed: err={}, cu_err={}",
                err.err,
                err.cu_err,
            );
        }

        debug!(
            num_nodes = header.num_nodes,
            feature_dim = header.feature_dim,
            dtype = ?header.dtype,
            gpu_buffer_size,
            max_batch_size,
            "GDS feature store opened",
        );

        Ok(Self {
            file_handle,
            _file: file,
            num_nodes: header.num_nodes,
            feature_dim: header.feature_dim,
            features_start_offset: header.features_start_offset,
            dtype: header.dtype,
            gpu_buffer: gpu_ptr,
            gpu_buffer_size,
            max_batch_size,
        })
    }

    /// Feature dimension per node.
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Number of nodes in the feature file.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Element data type.
    pub fn dtype(&self) -> FeatureDtype {
        self.dtype
    }

    /// Maximum batch size.
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Bytes per feature row (feature_dim * element_size).
    fn feature_size(&self) -> usize {
        self.feature_dim * self.dtype.element_size()
    }

    /// Read features for `nodes` directly into the start of the GPU buffer.
    ///
    /// Returns a `GdsReadResult` with the device pointer and metadata.
    /// The features are packed contiguously in the caller's original node
    /// order: node\[0\]'s features at offset 0, node\[1\]'s at `feature_size`, etc.
    pub fn get_batch(&self, nodes: &[NodeId]) -> Result<GdsReadResult> {
        ensure!(
            nodes.len() <= self.max_batch_size,
            "batch size {} exceeds max {}",
            nodes.len(),
            self.max_batch_size,
        );

        self.read_nodes_into(nodes, 0)?;

        Ok(GdsReadResult {
            device_ptr: self.gpu_buffer as u64,
            num_nodes: nodes.len(),
            feature_dim: self.feature_dim,
            dtype: self.dtype,
        })
    }

    /// Read features for `nodes` into the GPU buffer at `dev_offset` bytes.
    ///
    /// Returns the total number of bytes written to the GPU buffer.
    pub fn get_batch_into(&self, nodes: &[NodeId], dev_offset: usize) -> Result<usize> {
        let feature_size = self.feature_size();
        let total_bytes = nodes.len() * feature_size;
        ensure!(
            dev_offset + total_bytes <= self.gpu_buffer_size,
            "batch of {} nodes at offset {} exceeds GPU buffer ({} bytes)",
            nodes.len(),
            dev_offset,
            self.gpu_buffer_size,
        );

        self.read_nodes_into(nodes, dev_offset)
    }

    /// Submit all reads via the cuFile batch API.
    ///
    /// Sorts nodes by ID for sequential NVMe access, builds one
    /// `CUfileIOParams` per node, submits in a single `cuFileBatchIOSubmit`,
    /// then polls `cuFileBatchIOGetStatus` until all complete.
    ///
    /// For very small batches (< 8 nodes) falls back to individual
    /// `cuFileRead` calls to avoid batch setup overhead.
    #[allow(unsafe_code)]
    fn read_nodes_into(&self, nodes: &[NodeId], base_dev_offset: usize) -> Result<usize> {
        if nodes.is_empty() {
            return Ok(0);
        }

        let feature_size = self.feature_size();

        // Validate all nodes upfront.
        for &node in nodes {
            ensure!(
                (node as usize) < self.num_nodes,
                "node {} out of bounds (max {})",
                node,
                self.num_nodes,
            );
        }

        // Sort by node ID for sequential file access. Track original index
        // so output lands in the caller's expected order.
        let mut sorted: Vec<(NodeId, usize)> =
            nodes.iter().copied().enumerate().map(|(i, n)| (n, i)).collect();
        sorted.sort_unstable_by_key(|&(n, _)| n);

        // Small batches: skip batch API overhead.
        if sorted.len() < BATCH_API_THRESHOLD {
            return self.read_nodes_sequential(&sorted, feature_size, base_dev_offset);
        }

        // Set up batch handle.
        let num_entries = sorted.len() as u32;
        let mut batch_handle: ffi::CUfileBatchHandle = std::ptr::null_mut();
        let err = unsafe {
            ffi::cuFileBatchIOSetUp(&raw mut batch_handle, num_entries)
        };
        ensure!(
            err.err == ffi::CU_FILE_SUCCESS,
            "cuFileBatchIOSetUp failed: err={}, cu_err={}",
            err.err,
            err.cu_err,
        );

        // Build IO params array.
        let mut io_params: Vec<ffi::CUfileIOParams> = sorted
            .iter()
            .map(|&(node, orig_idx)| {
                let file_offset = self.features_start_offset as i64
                    + (node as i64) * (feature_size as i64);
                let dev_offset =
                    (base_dev_offset + orig_idx * feature_size) as i64;

                ffi::CUfileIOParams {
                    mode: ffi::CUFILE_READ,
                    pad: [0; 4],
                    fh: self.file_handle,
                    opcode: 0,
                    status: ffi::CUFILE_BATCH_NOT_STARTED,
                    cookie: std::ptr::null_mut(),
                    dev_ptr_base: self.gpu_buffer,
                    file_offset,
                    dev_ptr_offset: dev_offset,
                    size: feature_size,
                    bytes_done: 0,
                }
            })
            .collect();

        // Submit all reads in one kernel crossing.
        let err = unsafe {
            ffi::cuFileBatchIOSubmit(
                batch_handle,
                num_entries,
                io_params.as_mut_ptr(),
                0,
            )
        };
        if err.err != ffi::CU_FILE_SUCCESS {
            unsafe { ffi::cuFileBatchIODestroy(batch_handle) };
            anyhow::bail!(
                "cuFileBatchIOSubmit failed: err={}, cu_err={}",
                err.err,
                err.cu_err,
            );
        }

        // Poll until all entries complete.
        let mut num_complete: u32 = 0;
        loop {
            let err = unsafe {
                ffi::cuFileBatchIOGetStatus(
                    batch_handle,
                    num_entries,
                    io_params.as_mut_ptr(),
                    &raw mut num_complete,
                )
            };
            if err.err != ffi::CU_FILE_SUCCESS {
                unsafe { ffi::cuFileBatchIODestroy(batch_handle) };
                anyhow::bail!(
                    "cuFileBatchIOGetStatus failed: err={}, cu_err={}",
                    err.err,
                    err.cu_err,
                );
            }
            if num_complete >= num_entries {
                break;
            }
            // Yield to avoid busy-spinning. GDS typically completes in <100us
            // for a batch of 3000 nodes, so one yield is usually enough.
            std::thread::yield_now();
        }

        // Validate all entries completed successfully.
        let mut total_bytes: usize = 0;
        for (i, param) in io_params.iter().enumerate() {
            ensure!(
                param.status == ffi::CUFILE_BATCH_COMPLETE,
                "GDS batch entry {} not complete (status={})",
                i,
                param.status,
            );
            ensure!(
                param.bytes_done >= 0 && param.bytes_done as usize == feature_size,
                "GDS batch entry {} short read: expected {} bytes, got {}",
                i,
                feature_size,
                param.bytes_done,
            );
            total_bytes += param.bytes_done as usize;
        }

        unsafe { ffi::cuFileBatchIODestroy(batch_handle) };
        Ok(total_bytes)
    }

    /// Fallback for small batches: individual cuFileRead calls.
    #[allow(unsafe_code)]
    fn read_nodes_sequential(
        &self,
        sorted: &[(NodeId, usize)],
        feature_size: usize,
        base_dev_offset: usize,
    ) -> Result<usize> {
        let mut total_bytes: usize = 0;

        for &(node, orig_idx) in sorted {
            let file_offset = self.features_start_offset as i64
                + (node as i64) * (feature_size as i64);
            let dev_offset =
                (base_dev_offset + orig_idx * feature_size) as i64;

            let n = unsafe {
                ffi::cuFileRead(
                    self.file_handle,
                    self.gpu_buffer,
                    feature_size,
                    file_offset,
                    dev_offset,
                )
            };

            if n < 0 {
                anyhow::bail!(
                    "cuFileRead failed for node {}: returned {}",
                    node,
                    n,
                );
            }
            let n = n as usize;
            if n != feature_size {
                anyhow::bail!(
                    "cuFileRead short read for node {}: expected {} bytes, got {}",
                    node,
                    feature_size,
                    n,
                );
            }

            total_bytes += n;
        }

        Ok(total_bytes)
    }
}

#[allow(unsafe_code)]
impl Drop for GdsFeatureStore {
    fn drop(&mut self) {
        // Deregister the GPU buffer first (while the file handle is still valid).
        let err = unsafe { ffi::cuFileBufDeregister(self.gpu_buffer) };
        if err.err != ffi::CU_FILE_SUCCESS {
            warn!(
                "cuFileBufDeregister failed: err={}, cu_err={}",
                err.err, err.cu_err
            );
        }

        // Deregister the file handle.
        unsafe { ffi::cuFileHandleDeregister(self.file_handle) };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gds_driver_open_fails_without_gpu() {
        // On machines without nvidia-gds, cuFileDriverOpen should return an
        // error rather than panic. Validates FFI linkage and error handling.
        let result = gds_driver_open();
        if let Err(e) = &result {
            assert!(
                format!("{e}").contains("cuFileDriverOpen failed"),
                "unexpected error: {e}",
            );
        }
    }

    #[test]
    fn gds_feature_store_buffer_arithmetic() {
        let feature_dim: usize = 128;
        let max_batch: usize = 1024;
        let elem_size: usize = 4; // f32
        let required = max_batch * feature_dim * elem_size;
        assert_eq!(required, 524_288);
        // f16 halves it
        let required_f16 = max_batch * feature_dim * 2;
        assert_eq!(required_f16, 262_144);
    }
}
