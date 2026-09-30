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

use crate::features::header::{FeatureDtype, parse_feature_header};
use crate::features::set_direct_io;
use crate::graph::NodeId;
use anyhow::{Context, Result, ensure};
use std::ffi::c_void;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::atomic::Ordering;
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

    /// `CUfileDescr_t`.
    #[repr(C)]
    pub struct CUfileDescr {
        pub handle_type: c_int, // CUfileFileHandleType
        pub handle: CUfileDescrUnion,
        pub fs_ops: *const c_void, // null for default
    }

    /// The `handle` union of `CUfileDescr_t`. The pointer member sets its
    /// size and alignment to 8, which places `fs_ops` at offset 16.
    #[repr(C)]
    pub union CUfileDescrUnion {
        pub fd: c_int,           // Linux
        pub handle: *mut c_void, // Windows
    }

    // -- Batch I/O types --

    /// `CUfileOpcode_t`: operation selector for a batch entry.
    pub const CUFILE_READ: c_int = 0;

    /// `CUfileBatchMode_t`: the only defined mode value.
    pub const CUFILE_BATCH: c_uint = 1;

    /// `CUfileStatus_t` completion values reported in `CUfileIOEvents`.
    pub const CUFILE_COMPLETE: c_uint = 0x10;

    use std::os::raw::c_uint;

    /// The `u.batch` member of `CUfileIOParams_t`. The union has exactly
    /// one defined member, so it is modeled as a struct wrapped in a
    /// single-member union to keep the C layout.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct CUfileBatchOp {
        pub dev_ptr_base: *mut c_void, // registered GPU buffer base
        pub file_offset: i64,          // off_t: offset in file
        pub dev_ptr_offset: i64,       // off_t: offset within GPU buffer
        pub size: usize,               // bytes to transfer
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub union CUfileIOParamsUnion {
        pub batch: CUfileBatchOp,
    }

    /// Per-operation descriptor for batch I/O (`CUfileIOParams_t`).
    ///
    /// Field order matches cufile.h exactly: `mode`, the transfer union,
    /// `fh`, `opcode`, `cookie`. Completion status is NOT part of this
    /// struct — it is reported through a separate [`CUfileIOEvents`]
    /// array by `cuFileBatchIOGetStatus`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct CUfileIOParams {
        pub mode: c_uint, // CUfileBatchMode_t: CUFILE_BATCH
        pub u: CUfileIOParamsUnion,
        pub fh: CUfileHandle,    // registered file handle
        pub opcode: c_int,       // CUfileOpcode_t: CUFILE_READ / CUFILE_WRITE
        pub cookie: *mut c_void, // returned in the matching event
    }

    /// Completion record (`CUfileIOEvents_t`) filled by
    /// `cuFileBatchIOGetStatus`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct CUfileIOEvents {
        pub cookie: *mut c_void,
        pub status: c_uint, // CUfileStatus_t
        pub ret: usize,     // bytes transferred (or negative errno bit-cast)
    }

    /// Opaque batch handle.
    pub type CUfileBatchHandle = *mut c_void;

    #[link(name = "cufile")]
    unsafe extern "C" {
        pub fn cuFileDriverOpen() -> CUfileError;
        pub fn cuFileDriverClose() -> CUfileError;

        pub fn cuFileHandleRegister(fh: *mut CUfileHandle, descr: *mut CUfileDescr) -> CUfileError;
        pub fn cuFileHandleDeregister(fh: CUfileHandle);

        pub fn cuFileBufRegister(dev_ptr: *const c_void, size: usize, flags: c_int) -> CUfileError;
        pub fn cuFileBufDeregister(dev_ptr: *const c_void) -> CUfileError;

        // Single-read fallback (kept for small batches where overhead matters less)
        pub fn cuFileRead(
            fh: CUfileHandle,
            dev_ptr: *mut c_void,
            size: usize,
            file_offset: i64,
            dev_offset: i64,
        ) -> isize;

        // Batch I/O API (signatures from cufile.h)
        pub fn cuFileBatchIOSetUp(
            batch_handle: *mut CUfileBatchHandle,
            num_entries: c_uint,
        ) -> CUfileError;

        pub fn cuFileBatchIOSubmit(
            batch_handle: CUfileBatchHandle,
            num_entries: c_uint,
            io_params: *mut CUfileIOParams,
            flags: c_uint, // 0 for default
        ) -> CUfileError;

        /// `nr` is in/out: in = capacity of `events`, out = events filled.
        /// Blocks until at least `min_nr` operations complete or `timeout`
        /// elapses (null = no timeout).
        pub fn cuFileBatchIOGetStatus(
            batch_handle: CUfileBatchHandle,
            min_nr: c_uint,
            nr: *mut c_uint,
            events: *mut CUfileIOEvents,
            timeout: *mut libc::timespec,
        ) -> CUfileError;

        pub fn cuFileBatchIOCancel(batch_handle: CUfileBatchHandle) -> CUfileError;

        pub fn cuFileBatchIODestroy(batch_handle: CUfileBatchHandle);
    }

    pub const CU_FILE_SUCCESS: c_int = 0;
    /// `CUfileFileHandleType`: 1 = Linux fd (2 = Win32, 3 = userspace FS).
    pub const CU_FILE_HANDLE_TYPE_OPAQUE_FD: c_int = 1;

    // Layouts pinned to cufile.h on LP64, the only ABI libcufile ships for.
    #[cfg(target_pointer_width = "64")]
    const _: () = {
        use std::mem::{offset_of, size_of};
        assert!(size_of::<CUfileDescr>() == 24);
        assert!(offset_of!(CUfileDescr, handle) == 8);
        assert!(offset_of!(CUfileDescr, fs_ops) == 16);
        assert!(size_of::<CUfileIOParams>() == 64);
        assert!(offset_of!(CUfileIOParams, u) == 8);
        assert!(offset_of!(CUfileIOParams, fh) == 40);
        assert!(offset_of!(CUfileIOParams, opcode) == 48);
        assert!(offset_of!(CUfileIOParams, cookie) == 56);
        assert!(size_of::<CUfileIOEvents>() == 24);
        assert!(offset_of!(CUfileIOEvents, status) == 8);
        assert!(offset_of!(CUfileIOEvents, ret) == 16);
    };
}

// ---------------------------------------------------------------------------
// Driver init / shutdown
// ---------------------------------------------------------------------------

/// Initialize the cuFile driver. Call once at startup before creating any
/// `GdsFeatureStore` instances.
#[allow(unsafe_code)]
pub fn gds_driver_open() -> Result<()> {
    // SAFETY: cuFileDriverOpen is a no-arg FFI call provided by libcufile;
    // it is idempotent and safe to invoke from any thread.
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
    // SAFETY: cuFileDriverClose is a no-arg FFI call; calling it on a
    // not-yet-opened driver is documented to return an error, not crash.
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

/// Most operations one cuFile batch may carry. `cuFileBatchIOSetUp`
/// rejects anything above cufile.json's `io_batchsize`, whose default is
/// 128; a smaller configured limit is found by halving on rejection.
const MAX_BATCH_ENTRIES: usize = 128;

/// Batches submitted ahead of the one being reaped, so the device queue
/// stays fed across batch boundaries.
const MAX_INFLIGHT_BATCHES: usize = 8;

/// Upper bound on one batch's completion wait.
const BATCH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// A set-up cuFile batch. Dropping it cancels whatever it still has in
/// flight — so the driver quiesces DMA before the handle goes away — and
/// destroys the handle, which makes every early return in the gather
/// clean up without a hand-written teardown.
struct BatchHandle {
    raw: ffi::CUfileBatchHandle,
    /// Operations submitted on this handle.
    len: usize,
    /// Every operation has reported; nothing is left to cancel.
    drained: bool,
}

impl Drop for BatchHandle {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if !self.drained {
            // SAFETY: `raw` came from a successful cuFileBatchIOSetUp and is
            // destroyed only below.
            let _ = unsafe { ffi::cuFileBatchIOCancel(self.raw) };
        }
        // SAFETY: same handle; cancel above quiesced any live operation.
        unsafe { ffi::cuFileBatchIODestroy(self.raw) };
    }
}

/// Why a batch could not be submitted.
enum SubmitError {
    /// `cuFileBatchIOSetUp` refused the batch size.
    SetupRejected(ffi::CUfileError),
    Other(anyhow::Error),
}

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
    /// Operations per cuFile batch, lowered if the driver's configured
    /// `io_batchsize` rejects [`MAX_BATCH_ENTRIES`].
    batch_entries: std::sync::atomic::AtomicUsize,
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
    /// # Safety
    /// cuFile DMAs into whatever `gpu_device_ptr` names, so the caller must
    /// ensure:
    /// - `gpu_device_ptr` is a valid CUDA device pointer
    /// - The region `[gpu_device_ptr, gpu_device_ptr + gpu_buffer_size)` is allocated
    /// - The allocation outlives this `GdsFeatureStore`
    /// - The cuFile driver has been initialized via `gds_driver_open()`
    #[allow(unsafe_code)]
    pub unsafe fn open(
        path: &Path,
        gpu_device_ptr: u64,
        gpu_buffer_size: usize,
        max_batch_size: usize,
    ) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open feature file: {}", path.display()))?;
        let header = parse_feature_header(&file)?;
        // Without O_DIRECT cuFile serves reads through its POSIX
        // compatibility path, bouncing every row through host memory.
        // Setting it on the descriptor just validated keeps the geometry
        // and the reads on one inode.
        if let Err(e) = set_direct_io(&file, true) {
            warn!(
                "O_DIRECT unavailable for {} ({e}); cuFile will bounce reads through host memory",
                path.display()
            );
        }

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

        // Register the file with cuFile. The union starts zeroed through its
        // pointer member so the bytes past `fd` are defined.
        let mut file_handle: ffi::CUfileHandle = std::ptr::null_mut();
        let mut handle = ffi::CUfileDescrUnion {
            handle: std::ptr::null_mut(),
        };
        handle.fd = file.as_raw_fd();
        let mut descr = ffi::CUfileDescr {
            handle_type: ffi::CU_FILE_HANDLE_TYPE_OPAQUE_FD,
            handle,
            fs_ops: std::ptr::null(),
        };

        // SAFETY: `file_handle` and `descr` are stack-allocated locals we
        // pass exclusive mutable pointers to; `descr.handle.fd` came from a
        // `File` we still own, so the fd remains valid for the call.
        let err = unsafe { ffi::cuFileHandleRegister(&raw mut file_handle, &raw mut descr) };
        ensure!(
            err.err == ffi::CU_FILE_SUCCESS,
            "cuFileHandleRegister failed: err={}, cu_err={}",
            err.err,
            err.cu_err,
        );

        // Register the GPU buffer for DMA.
        let gpu_ptr = gpu_device_ptr as *mut c_void;
        // SAFETY: the caller has documented (in this function's `Safety
        // contract`) that `gpu_ptr` points at an allocation of at least
        // `gpu_buffer_size` bytes and outlives this store.
        let err = unsafe { ffi::cuFileBufRegister(gpu_ptr, gpu_buffer_size, 0) };
        if err.err != ffi::CU_FILE_SUCCESS {
            // SAFETY: `file_handle` was just successfully registered above;
            // deregistering it on the error path is the documented cleanup.
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
            batch_entries: std::sync::atomic::AtomicUsize::new(MAX_BATCH_ENTRIES),
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

    /// File byte offset of `node`'s feature row, with overflow checking.
    fn file_offset(&self, node: NodeId) -> Result<i64> {
        let off = (node as u64)
            .checked_mul(self.feature_size() as u64)
            .and_then(|o| o.checked_add(self.features_start_offset))
            .ok_or_else(|| anyhow::anyhow!("file offset overflow for node {}", node))?;
        i64::try_from(off).map_err(|_| anyhow::anyhow!("file offset exceeds i64 for node {}", node))
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
        let total_bytes = nodes
            .len()
            .checked_mul(feature_size)
            .ok_or_else(|| anyhow::anyhow!("batch byte size overflow"))?;
        let end = dev_offset
            .checked_add(total_bytes)
            .ok_or_else(|| anyhow::anyhow!("GPU buffer end offset overflow"))?;
        ensure!(
            end <= self.gpu_buffer_size,
            "batch of {} nodes at offset {} exceeds GPU buffer ({} bytes)",
            nodes.len(),
            dev_offset,
            self.gpu_buffer_size,
        );

        self.read_nodes_into(nodes, dev_offset)
    }

    /// Submit all reads via the cuFile batch API.
    ///
    /// Sorts nodes by ID for sequential NVMe access and builds one
    /// `CUfileIOParams` per node. The driver caps a batch at its
    /// `io_batchsize`, so the reads go out as a pipeline of batches of at
    /// most that many, up to [`MAX_INFLIGHT_BATCHES`] submitted ahead of
    /// the one being reaped.
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

        // Sort by node ID for sequential file access, tracking the original
        // index so output lands in the caller's expected order. Node and
        // index pack into one u64 (node in the high half) — half the
        // footprint of a (u32, usize) tuple and a plain integer sort.
        let mut sorted: Vec<u64> = nodes
            .iter()
            .enumerate()
            .map(|(i, &n)| ((n as u64) << 32) | i as u64)
            .collect();
        sorted.sort_unstable();
        let sorted: Vec<(NodeId, usize)> = sorted
            .into_iter()
            .map(|packed| ((packed >> 32) as NodeId, packed as u32 as usize))
            .collect();

        // Small batches: skip batch API overhead.
        if sorted.len() < BATCH_API_THRESHOLD {
            return self.read_nodes_sequential(&sorted, feature_size, base_dev_offset);
        }

        // Build the IO params array first, so an offset-overflow error comes
        // before any handle exists. Cookies carry the entry index, which a
        // failed completion reports.
        let mut io_params: Vec<ffi::CUfileIOParams> = sorted
            .iter()
            .enumerate()
            .map(|(entry_idx, &(node, orig_idx))| {
                let file_offset = self.file_offset(node)?;
                let dev_offset = orig_idx
                    .checked_mul(feature_size)
                    .and_then(|o| o.checked_add(base_dev_offset))
                    .and_then(|o| i64::try_from(o).ok())
                    .ok_or_else(|| anyhow::anyhow!("GPU buffer device offset overflow"))?;

                Ok(ffi::CUfileIOParams {
                    mode: ffi::CUFILE_BATCH,
                    u: ffi::CUfileIOParamsUnion {
                        batch: ffi::CUfileBatchOp {
                            dev_ptr_base: self.gpu_buffer,
                            file_offset,
                            dev_ptr_offset: dev_offset,
                            size: feature_size,
                        },
                    },
                    fh: self.file_handle,
                    opcode: ffi::CUFILE_READ,
                    cookie: entry_idx as *mut c_void,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Pipeline: keep up to MAX_INFLIGHT_BATCHES batches submitted and
        // reap them oldest-first. `io_params` outlives every handle, and a
        // handle dropped on any early return cancels and destroys itself.
        let mut inflight: std::collections::VecDeque<BatchHandle> =
            std::collections::VecDeque::with_capacity(MAX_INFLIGHT_BATCHES);
        let mut events = vec![
            ffi::CUfileIOEvents {
                cookie: std::ptr::null_mut(),
                status: 0,
                ret: 0,
            };
            MAX_BATCH_ENTRIES
        ];
        let mut next = 0usize;
        let mut total_bytes = 0usize;
        while next < io_params.len() || !inflight.is_empty() {
            while next < io_params.len() && inflight.len() < MAX_INFLIGHT_BATCHES {
                let limit = self.batch_entries.load(Ordering::Relaxed);
                let end = (next + limit).min(io_params.len());
                match self.submit_batch(&mut io_params[next..end]) {
                    Ok(handle) => {
                        inflight.push_back(handle);
                        next = end;
                    }
                    // The configured io_batchsize is below this limit:
                    // halve and retry, remembering the size that works.
                    Err(SubmitError::SetupRejected(_)) if limit > 1 => {
                        let _ = self.batch_entries.compare_exchange(
                            limit,
                            limit / 2,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        );
                    }
                    Err(SubmitError::SetupRejected(err)) => anyhow::bail!(
                        "cuFileBatchIOSetUp failed: err={}, cu_err={}",
                        err.err,
                        err.cu_err,
                    ),
                    Err(SubmitError::Other(e)) => return Err(e),
                }
            }
            if let Some(mut handle) = inflight.pop_front() {
                total_bytes += Self::reap_batch(&mut handle, &mut events, feature_size)?;
            }
        }
        Ok(total_bytes)
    }

    /// Set up a batch handle for `params` and submit every operation.
    #[allow(unsafe_code)]
    fn submit_batch(
        &self,
        params: &mut [ffi::CUfileIOParams],
    ) -> std::result::Result<BatchHandle, SubmitError> {
        debug_assert!(params.len() <= MAX_BATCH_ENTRIES);
        let n = params.len() as std::os::raw::c_uint;
        let mut raw: ffi::CUfileBatchHandle = std::ptr::null_mut();
        // SAFETY: `raw` is a stack local with exclusive mutable access;
        // cuFile writes a fresh opaque pointer into it.
        let err = unsafe { ffi::cuFileBatchIOSetUp(&raw mut raw, n) };
        if err.err != ffi::CU_FILE_SUCCESS {
            return Err(SubmitError::SetupRejected(err));
        }
        // Owned from here on: a failed submit still destroys the handle.
        let mut handle = BatchHandle {
            raw,
            len: params.len(),
            drained: true,
        };
        // SAFETY: `raw` was just set up for `n` entries and `params` holds
        // exactly `n` initialized descriptors that outlive the batch.
        let err = unsafe { ffi::cuFileBatchIOSubmit(raw, n, params.as_mut_ptr(), 0) };
        if err.err != ffi::CU_FILE_SUCCESS {
            return Err(SubmitError::Other(anyhow::anyhow!(
                "cuFileBatchIOSubmit failed: err={}, cu_err={}",
                err.err,
                err.cu_err,
            )));
        }
        handle.drained = false;
        Ok(handle)
    }

    /// Wait for every operation on `handle`, checking each completed in
    /// full, and return the bytes transferred. A deadline bounds the wait
    /// so a wedged device surfaces as an error instead of a hang.
    #[allow(unsafe_code)]
    fn reap_batch(
        handle: &mut BatchHandle,
        events: &mut [ffi::CUfileIOEvents],
        feature_size: usize,
    ) -> Result<usize> {
        let deadline = std::time::Instant::now() + BATCH_DEADLINE;
        let mut completed = 0usize;
        let mut bytes = 0usize;
        while completed < handle.len {
            let remaining = (handle.len - completed) as std::os::raw::c_uint;
            let mut nr = remaining;
            let mut timeout = libc::timespec {
                tv_sec: 0,
                tv_nsec: 10_000_000, // 10ms per wait slice
            };
            // min_nr = remaining: block once for the rest of the batch
            // rather than one driver round-trip per completion; the
            // timeout bounds each slice so the deadline check runs.
            // SAFETY: `handle.raw` is live; `events` holds at least
            // `remaining` entries; `nr` and `timeout` are stack locals.
            let err = unsafe {
                ffi::cuFileBatchIOGetStatus(
                    handle.raw,
                    remaining,
                    &raw mut nr,
                    events.as_mut_ptr(),
                    &raw mut timeout,
                )
            };
            ensure!(
                err.err == ffi::CU_FILE_SUCCESS,
                "cuFileBatchIOGetStatus failed: err={}, cu_err={}",
                err.err,
                err.cu_err,
            );
            let got = (nr as usize).min(remaining as usize);
            for event in &events[..got] {
                // The cookie is the entry's index in the sorted batch.
                let entry_idx = event.cookie as usize;
                ensure!(
                    event.status == ffi::CUFILE_COMPLETE,
                    "GDS batch entry {} failed (status={:#x})",
                    entry_idx,
                    event.status,
                );
                ensure!(
                    event.ret == feature_size,
                    "GDS batch entry {} short read: expected {} bytes, got {}",
                    entry_idx,
                    feature_size,
                    event.ret,
                );
                bytes += event.ret;
            }
            completed += got;
            ensure!(
                completed >= handle.len || std::time::Instant::now() < deadline,
                "GDS batch timed out: {} of {} reads completed within {:?}",
                completed,
                handle.len,
                BATCH_DEADLINE,
            );
        }
        handle.drained = true;
        Ok(bytes)
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
            let file_offset = self.file_offset(node)?;
            let dev_offset = orig_idx
                .checked_mul(feature_size)
                .and_then(|o| o.checked_add(base_dev_offset))
                .and_then(|o| i64::try_from(o).ok())
                .ok_or_else(|| anyhow::anyhow!("GPU buffer device offset overflow"))?;

            // SAFETY: `self.file_handle` and `self.gpu_buffer` were
            // registered with cuFile in `open()` and remain valid for the
            // store's lifetime; `feature_size` fits within the buffer per
            // the bounds check in `open()`.
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
                anyhow::bail!("cuFileRead failed for node {}: returned {}", node, n,);
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
        // SAFETY: `self.gpu_buffer` was registered in `open()` and we are
        // the unique owner being dropped; no further cuFile ops will run.
        let err = unsafe { ffi::cuFileBufDeregister(self.gpu_buffer) };
        if err.err != ffi::CU_FILE_SUCCESS {
            warn!(
                "cuFileBufDeregister failed: err={}, cu_err={}",
                err.err, err.cu_err
            );
        }

        // Deregister the file handle.
        // SAFETY: `self.file_handle` was registered in `open()` and is
        // about to go out of scope; cuFile reclaims its internal state.
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
