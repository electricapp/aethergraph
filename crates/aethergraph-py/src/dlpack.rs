//! DLPack tensor export for zero-copy CUDA → PyTorch transfer.
//!
//! Creates PyCapsule objects wrapping DLManagedTensor structs that PyTorch
//! can consume via `torch.from_dlpack()`. This is the standard zero-copy
//! path for sharing GPU tensors across frameworks.

use pyo3::ffi as pyffi;
use pyo3::prelude::*;
use std::os::raw::c_void;

// DLPack ABI constants (dlpack.h v0.8)
// TODO: DLPack v1.0 changed the capsule name and added versioned managed
// tensors (DLManagedTensorVersioned). If PyTorch adopts v1.0, these structs
// and the capsule creation logic will need updating.
const KDLCUDA: i32 = 2; // kDLCUDA
const KDLFLOAT: u8 = 2; // kDLFloat

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

/// DLPack managed tensor with destructor.
#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// Context passed to the DLPack deleter.
struct DlpackContext {
    shape: Vec<i64>,
    strides: Vec<i64>,
}

/// DLPack deleter callback — frees the context when PyTorch releases the tensor.
///
/// Note: we do NOT free the CUDA memory here. The VRAM is owned by
/// `SeqlockValidator.output` and reused across gather calls. The Python
/// tensor is a view into that buffer.
unsafe extern "C" fn dlpack_deleter(managed: *mut DLManagedTensor) {
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
/// "dltensor" to "used_dltensor" and then drives `deleter` itself. If the
/// capsule is dropped while still named "dltensor" (never consumed), no one
/// else will ever call `deleter`, so we do it here exactly once. A consumed
/// capsule fails the "dltensor" validity check and we leave it alone.
unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut pyffi::PyObject) {
    // SAFETY: `capsule` is the PyCapsule being finalized; the GIL is held
    // during capsule destruction. `c"dltensor"` outlives the call.
    let valid = unsafe { pyffi::PyCapsule_IsValid(capsule, c"dltensor".as_ptr()) };
    if valid == 0 {
        return;
    }
    // SAFETY: the validity check above confirms the capsule still carries a
    // pointer under the "dltensor" name, so this returns it without error.
    let ptr = unsafe { pyffi::PyCapsule_GetPointer(capsule, c"dltensor".as_ptr()) };
    if ptr.is_null() {
        return;
    }
    // SAFETY: the stored pointer is the `*mut DLManagedTensor` produced by
    // `build_managed_tensor`; the unconsumed capsule is its sole owner, so the
    // deleter runs exactly once here, freeing the managed tensor and context.
    unsafe { dlpack_deleter(ptr as *mut DLManagedTensor) };
}

/// Build the raw `DLManagedTensor` Box for a CUDA f32 `(num_nodes, feature_dim)`
/// view backed by `ptr`. Pure-Rust so we can assert on the struct fields
/// without going through the Python GIL — see `tests` module below.
///
/// Layout: row-major `f32` in CUDA memory. Strides are
/// `[feature_dim, 1]` — i.e. row stride = `feature_dim` elements, column stride
/// = 1 element. The DLPack spec defines strides in **elements** (not bytes),
/// matching numpy's `arr.strides // dtype.itemsize`. Consumers that interpret
/// them in bytes (or that flip row/column major) will silently corrupt reads.
///
/// Ownership: caller takes the returned `*mut DLManagedTensor` and is
/// responsible for either (a) handing it to `PyCapsule_New` whose consumer
/// will eventually call `dlpack_deleter`, or (b) calling `dlpack_deleter`
/// themselves. Leaking is a memory bug.
///
/// # Safety
/// `ptr` must reference valid CUDA memory for the lifetime implied by
/// whoever eventually consumes the tensor.
fn build_managed_tensor(
    ptr: u64,
    num_nodes: usize,
    feature_dim: usize,
    gpu_id: i32,
) -> *mut DLManagedTensor {
    let mut ctx = Box::new(DlpackContext {
        shape: vec![num_nodes as i64, feature_dim as i64],
        // Row-major: stride[0]=feature_dim elements, stride[1]=1 element.
        // DLPack measures strides in elements, not bytes — see comment above.
        strides: vec![feature_dim as i64, 1],
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

    let managed = Box::new(DLManagedTensor {
        dl_tensor: tensor,
        manager_ctx: Box::into_raw(ctx) as *mut c_void,
        // PyTorch rejects DLPack capsules whose `deleter` is None — even when
        // the capsule itself has its own destructor. We always set this Some
        // so `torch.from_dlpack()` accepts the capsule.
        deleter: Some(dlpack_deleter),
    });

    Box::into_raw(managed)
}

/// Test-only Python entry point. Lets pytest hand in an existing CUDA buffer
/// (e.g. `torch.empty(..., device='cuda').data_ptr()`) and round-trip it
/// through `create_dlpack_capsule` + `torch.from_dlpack`. The capsule's
/// deleter does not free VRAM, so the original tensor stays the owner.
///
/// # Safety
/// Caller is responsible for `ptr` being a valid CUDA allocation with at least
/// `num_nodes * feature_dim * sizeof(f32)` bytes that outlives the resulting
/// numpy/torch view. Exposed only so T2.2 can exercise the capsule path
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
    create_dlpack_capsule(py, ptr, num_nodes, feature_dim, gpu_id)
}

/// Create a PyCapsule wrapping a DLPack managed tensor for a CUDA f32 buffer.
///
/// The capsule can be consumed by `torch.from_dlpack()` for zero-copy access.
///
/// # Safety
/// `ptr` must be a valid CUDA device pointer with at least
/// `num_nodes * feature_dim * sizeof(f32)` bytes allocated.
/// The memory must remain valid for the lifetime of the returned tensor.
pub fn create_dlpack_capsule(
    py: Python<'_>,
    ptr: u64,
    num_nodes: usize,
    feature_dim: usize,
    gpu_id: i32,
) -> PyResult<Py<PyAny>> {
    let managed_ptr = build_managed_tensor(ptr, num_nodes, feature_dim, gpu_id);

    // Create PyCapsule with name "dltensor" (required by DLPack spec).
    // PyCapsule_New stores the name *pointer*, not a copy, and consumers
    // strcmp it for the capsule's whole lifetime — the name must be 'static.
    // SAFETY: PyCapsule_New requires the GIL, which `Python<'_>` proves.
    // The destructor frees the managed tensor if the capsule is dropped
    // without being consumed; a consumer (torch.from_dlpack) renames the
    // capsule and takes over the DLPack `deleter` itself.
    let raw = unsafe {
        pyffi::PyCapsule_New(
            managed_ptr as *mut c_void,
            c"dltensor".as_ptr(),
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
    //! TEST_PLAN T2.2 — DLPack capsule ABI.
    //!
    //! The live `torch.from_dlpack()` consumer path is already exercised in
    //! T2.5 (full PyG training loop) via `next_with_gpu_features`. What we
    //! cover locally is the struct encoding — catching the failure modes that
    //! make PyTorch silently reject: wrong `device_type`, wrong
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

        let managed_ptr = build_managed_tensor(PTR, NUM_NODES, FEATURE_DIM, GPU_ID);
        assert!(!managed_ptr.is_null());

        // SAFETY: `managed_ptr` is a valid heap allocation from build_managed_tensor.
        let managed = unsafe { &*managed_ptr };
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
