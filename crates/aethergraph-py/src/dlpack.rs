//! DLPack tensor export for zero-copy CUDA → PyTorch transfer.
//!
//! Creates PyCapsule objects wrapping DLPack v1.0 `DLManagedTensorVersioned`
//! structs that PyTorch can consume via `torch.from_dlpack()`. This is the
//! standard zero-copy path for sharing GPU tensors across frameworks.

use aether_stream::rdma::gather::OwnedGpuFeatures;
use pyo3::ffi as pyffi;
use pyo3::prelude::*;
use std::os::raw::c_void;

// DLPack ABI constants (dlpack.h v1.0): versioned managed tensors under the
// "dltensor_versioned" capsule name.
const KDLCUDA: i32 = 2; // kDLCUDA
const KDLFLOAT: u8 = 2; // kDLFloat
const DLPACK_MAJOR_VERSION: u32 = 1;
const DLPACK_MINOR_VERSION: u32 = 0;

/// DLPack ABI version stamp, checked by consumers before reading the rest
/// of the struct.
#[repr(C)]
struct DLPackVersion {
    major: u32,
    minor: u32,
}

/// DLPack device descriptor.
#[repr(C)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

/// DLPack data type descriptor.
#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

/// DLPack tensor descriptor.
#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

/// DLPack v1.0 managed tensor with version stamp, flags, and destructor.
///
/// Field order is ABI: version first (so consumers can check compatibility
/// before touching anything else), `dl_tensor` last.
#[repr(C)]
struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    /// Bitmask; bit 0 is DLPACK_FLAG_BITMASK_READ_ONLY. Our tensors are
    /// writable views, so 0.
    flags: u64,
    dl_tensor: DLTensor,
}

/// Context passed to the DLPack deleter.
struct DlpackContext {
    shape: Vec<i64>,
    strides: Vec<i64>,
    /// Keeps the CUDA buffer alive for the tensor's lifetime. `Some` for
    /// gather results (the tensor owns its VRAM via the inner `Arc`);
    /// `None` when the caller owns the buffer (test-only raw-pointer path).
    owner: Option<OwnedGpuFeatures>,
}

/// DLPack deleter callback — runs when PyTorch releases the tensor.
///
/// Dropping the context drops its `owner` (when present), releasing the
/// tensor's reference on the CUDA buffer — the VRAM is freed once the last
/// `Arc` clone goes away. With `owner: None` the caller keeps ownership of
/// the CUDA memory and only the context/tensor structs are freed.
unsafe extern "C" fn dlpack_deleter(managed: *mut DLManagedTensorVersioned) {
    if managed.is_null() {
        return;
    }
    // SAFETY: `managed` was non-null per the check above; pointers were valid
    // when handed off to PyCapsule_New, and torch only calls the deleter once.
    let ctx = unsafe { (*managed).manager_ctx } as *mut DlpackContext;
    if !ctx.is_null() {
        // SAFETY: `ctx` originated from `Box::into_raw` in `build_managed_tensor`.
        drop(unsafe { Box::from_raw(ctx) });
    }
    // SAFETY: `managed` originated from `Box::into_raw` in `build_managed_tensor`.
    drop(unsafe { Box::from_raw(managed) });
}

/// PyCapsule destructor — frees the managed tensor if the capsule was never
/// consumed.
///
/// `torch.from_dlpack()` takes ownership by renaming the capsule from
/// "dltensor_versioned" to "used_dltensor_versioned" and then drives
/// `deleter` itself. If the capsule is dropped while still named
/// "dltensor_versioned" (never consumed), no one else will ever call
/// `deleter`, so we do it here exactly once. A consumed capsule fails the
/// validity check and we leave it alone.
unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut pyffi::PyObject) {
    // SAFETY: `capsule` is the PyCapsule being finalized; the GIL is held
    // during capsule destruction. `c"dltensor_versioned"` outlives the call.
    let valid = unsafe { pyffi::PyCapsule_IsValid(capsule, c"dltensor_versioned".as_ptr()) };
    if valid == 0 {
        return;
    }
    // SAFETY: the validity check above confirms the capsule still carries a
    // pointer under the "dltensor_versioned" name, so this returns it
    // without error.
    let ptr = unsafe { pyffi::PyCapsule_GetPointer(capsule, c"dltensor_versioned".as_ptr()) };
    if ptr.is_null() {
        return;
    }
    // SAFETY: the stored pointer is the `*mut DLManagedTensorVersioned`
    // produced by `build_managed_tensor`; the unconsumed capsule is its sole
    // owner, so the deleter runs exactly once here, freeing the managed
    // tensor and context.
    unsafe { dlpack_deleter(ptr as *mut DLManagedTensorVersioned) };
}

/// Build the raw `DLManagedTensorVersioned` Box for a CUDA f32
/// `(num_nodes, feature_dim)` view backed by `ptr`. Pure-Rust so we can
/// assert on the struct fields without going through the Python GIL — see
/// `tests` module below.
///
/// Layout: row-major `f32` in CUDA memory. Strides are
/// `[feature_dim, 1]` — i.e. row stride = `feature_dim` elements, column stride
/// = 1 element. The DLPack spec defines strides in **elements** (not bytes),
/// matching numpy's `arr.strides // dtype.itemsize`. Consumers that interpret
/// them in bytes (or that flip row/column major) will silently corrupt reads.
///
/// Ownership: caller takes the returned `*mut DLManagedTensorVersioned` and
/// is responsible for either (a) handing it to `PyCapsule_New` whose
/// consumer will eventually call `dlpack_deleter`, or (b) calling
/// `dlpack_deleter` themselves. Leaking is a memory bug.
///
/// # Safety
/// `ptr` must reference valid CUDA memory for the lifetime implied by
/// whoever eventually consumes the tensor. Passing the buffer's owner as
/// `owner` satisfies that for gather results; with `owner: None` the caller
/// must keep the allocation alive themselves.
fn build_managed_tensor(
    ptr: u64,
    num_nodes: usize,
    feature_dim: usize,
    gpu_id: i32,
    owner: Option<OwnedGpuFeatures>,
) -> *mut DLManagedTensorVersioned {
    let mut ctx = Box::new(DlpackContext {
        shape: vec![num_nodes as i64, feature_dim as i64],
        // Row-major: stride[0]=feature_dim elements, stride[1]=1 element.
        // DLPack measures strides in elements, not bytes — see comment above.
        strides: vec![feature_dim as i64, 1],
        owner,
    });

    let tensor = DLTensor {
        data: ptr as *mut c_void,
        device: DLDevice {
            device_type: KDLCUDA,
            device_id: gpu_id,
        },
        ndim: 2,
        dtype: DLDataType {
            code: KDLFLOAT,
            bits: 32,
            lanes: 1,
        },
        shape: ctx.shape.as_mut_ptr(),
        strides: ctx.strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let managed = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion {
            major: DLPACK_MAJOR_VERSION,
            minor: DLPACK_MINOR_VERSION,
        },
        manager_ctx: Box::into_raw(ctx) as *mut c_void,
        // PyTorch rejects DLPack capsules whose `deleter` is None — even when
        // the capsule itself has its own destructor. We always set this Some
        // so `torch.from_dlpack()` accepts the capsule.
        deleter: Some(dlpack_deleter),
        flags: 0,
        dl_tensor: tensor,
    });

    Box::into_raw(managed)
}

/// Test-only Python entry point. Lets pytest hand in an existing CUDA buffer
/// (e.g. `torch.empty(..., device='cuda').data_ptr()`) and round-trip it
/// through the capsule path + `torch.from_dlpack`. The capsule's deleter does
/// not free VRAM on this path (no owner is attached), so the original tensor
/// stays the owner.
///
/// # Safety
/// Caller is responsible for `ptr` being a valid CUDA allocation with at least
/// `num_nodes * feature_dim * sizeof(f32)` bytes that outlives the resulting
/// numpy/torch view. Exposed only so tests can exercise the capsule path
/// without spinning up a full RDMA server.
#[pyfunction]
#[pyo3(name = "_dlpack_capsule_from_cuda_ptr")]
pub fn dlpack_capsule_from_cuda_ptr_py(
    py: Python<'_>,
    ptr: u64,
    num_nodes: usize,
    feature_dim: usize,
    gpu_id: i32,
) -> PyResult<Py<PyAny>> {
    capsule_from_parts(py, ptr, num_nodes, feature_dim, gpu_id, None)
}

/// Create a PyCapsule wrapping a DLPack managed tensor for an owned CUDA
/// feature buffer.
///
/// The capsule can be consumed by `torch.from_dlpack()` for zero-copy access.
/// `features` moves into the capsule's manager context, so the VRAM lives
/// exactly as long as the consuming tensor (or the unconsumed capsule).
pub fn create_dlpack_capsule(py: Python<'_>, features: OwnedGpuFeatures) -> PyResult<Py<PyAny>> {
    let ptr = features.device_ptr();
    let num_nodes = features.num_nodes;
    let feature_dim = features.feature_dim;
    let gpu_id = features.gpu_id;
    capsule_from_parts(py, ptr, num_nodes, feature_dim, gpu_id, Some(features))
}

/// Shared capsule construction for the owned and raw-pointer entry points.
///
/// # Safety
/// `ptr` must be a valid CUDA device pointer with at least
/// `num_nodes * feature_dim * sizeof(f32)` bytes allocated, kept alive by
/// `owner` (or by the caller when `owner` is `None`) for the lifetime of the
/// returned tensor.
fn capsule_from_parts(
    py: Python<'_>,
    ptr: u64,
    num_nodes: usize,
    feature_dim: usize,
    gpu_id: i32,
    owner: Option<OwnedGpuFeatures>,
) -> PyResult<Py<PyAny>> {
    let managed_ptr = build_managed_tensor(ptr, num_nodes, feature_dim, gpu_id, owner);

    // Create PyCapsule with the DLPack v1.0 name "dltensor_versioned".
    // PyCapsule_New stores the name *pointer*, not a copy, and consumers
    // strcmp it for the capsule's whole lifetime — the name must be 'static.
    // SAFETY: PyCapsule_New requires the GIL, which `Python<'_>` proves.
    // The destructor frees the managed tensor if the capsule is dropped
    // without being consumed; a consumer (torch.from_dlpack) renames the
    // capsule and takes over the DLPack `deleter` itself.
    let raw = unsafe {
        pyffi::PyCapsule_New(
            managed_ptr as *mut c_void,
            c"dltensor_versioned".as_ptr(),
            Some(dlpack_capsule_destructor),
        )
    };
    if raw.is_null() {
        // SAFETY: `managed_ptr` was just returned by `build_managed_tensor`.
        unsafe { dlpack_deleter(managed_ptr) };
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Failed to create DLPack PyCapsule",
        ));
    }
    // SAFETY: `raw` is a non-null PyCapsule we own; GIL is held.
    let capsule = unsafe { Bound::from_owned_ptr(py, raw) }.unbind();

    Ok(capsule)
}

#[cfg(test)]
mod tests {
    //! DLPack capsule ABI.
    //!
    //! The live `torch.from_dlpack()` consumer path is exercised by the full
    //! PyG training loop integration test via `next_with_gpu_features`. What
    //! we cover locally is the struct encoding — catching the failure modes
    //! that make PyTorch silently reject: wrong `device_type`, wrong
    //! `dtype.code`/`bits`, missing deleter, bad shape/strides/ndim.
    //!
    //! Targets the pure-Rust `build_managed_tensor` helper rather than going
    //! through `PyCapsule_New`. The capsule step on top is a thin adapter —
    //! if the underlying struct is right, `torch.from_dlpack` accepts it.
    use super::*;

    #[test]
    fn managed_tensor_struct_matches_dlpack_spec() {
        const PTR: u64 = 0xDEAD_BEEF_CAFE_0000;
        const NUM_NODES: usize = 7;
        const FEATURE_DIM: usize = 128;
        const GPU_ID: i32 = 3;

        let managed_ptr = build_managed_tensor(PTR, NUM_NODES, FEATURE_DIM, GPU_ID, None);
        assert!(!managed_ptr.is_null());

        // SAFETY: `managed_ptr` is a valid heap allocation from build_managed_tensor.
        let managed = unsafe { &*managed_ptr };

        assert_eq!(managed.version.major, DLPACK_MAJOR_VERSION);
        assert_eq!(managed.version.minor, DLPACK_MINOR_VERSION);
        assert_eq!(managed.flags, 0, "writable tensor: no READ_ONLY flag");

        let t = &managed.dl_tensor;

        assert_eq!(t.data as u64, PTR, "tensor data pointer");
        assert_eq!(t.device.device_type, KDLCUDA, "device_type must be kDLCUDA");
        assert_eq!(t.device.device_id, GPU_ID, "device_id");
        assert_eq!(t.ndim, 2, "2D tensor (num_nodes, feature_dim)");
        assert_eq!(t.dtype.code, KDLFLOAT, "dtype code must be kDLFloat");
        assert_eq!(t.dtype.bits, 32, "f32 → 32 bits");
        assert_eq!(t.dtype.lanes, 1, "scalar lanes");
        assert_eq!(t.byte_offset, 0);

        // SAFETY: shape/strides point into the boxed DlpackContext owned by `managed`.
        let shape = unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) };
        // SAFETY: same.
        let strides = unsafe { std::slice::from_raw_parts(t.strides, t.ndim as usize) };
        assert_eq!(shape, &[NUM_NODES as i64, FEATURE_DIM as i64]);
        assert_eq!(
            strides,
            &[FEATURE_DIM as i64, 1],
            "row-major: stride[0]=feature_dim, stride[1]=1"
        );

        assert!(
            managed.deleter.is_some(),
            "PyTorch rejects capsules without a deleter"
        );
        assert!(
            !managed.manager_ctx.is_null(),
            "deleter needs the DlpackContext to free shape/strides"
        );

        // SAFETY: `managed_ptr` came from `build_managed_tensor`; we drop the
        // borrow `managed` here and hand back ownership.
        unsafe { dlpack_deleter(managed_ptr) };
    }
}
