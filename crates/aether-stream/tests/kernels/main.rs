//! KERNELS.md Tier A integration tests (require `--features gpudirect`).
//!
//! Skip cleanly when no CUDA device is present.

#![cfg(all(target_os = "linux", feature = "gpudirect"))]

mod decompress;
mod persistent;
mod quant;
mod sampler;
mod tma;
mod validate;

use aether_stream::gpu::kernels::harness::cuda_or_skip;
use aether_stream::gpu::kernels::{NVRTC_ARCH_FLOOR, nvrtc_units};
use cudarc::nvrtc::{CompileOptions, Ptx};
use std::process::Command;

#[test]
fn cuda_device_probe() {
    match cuda_or_skip() {
        Some(_) => eprintln!("CUDA device 0 available"),
        None => eprintln!("skipping: no CUDA device"),
    }
}

/// Compile one unit at [`NVRTC_ARCH_FLOOR`]; `None` when this box has no
/// NVRTC to compile with.
fn compile_at_floor(name: &str, src: &str) -> Option<Ptx> {
    let opts = CompileOptions {
        options: vec![format!("--gpu-architecture={NVRTC_ARCH_FLOOR}")],
        ..Default::default()
    };
    match cudarc::nvrtc::compile_ptx_with_opts(src, opts) {
        Ok(ptx) => Some(ptx),
        Err(e) => {
            let text = e.to_string();
            // No toolkit on this box: skip rather than fail, the same way
            // the device tests do.
            if text.contains("NVRTC_ERROR_BUILTIN_OPERATION_FAILURE")
                || text.contains("library not found")
                || text.contains("cannot open shared object")
            {
                eprintln!("skipping {name}: no nvrtc ({text})");
                return None;
            }
            panic!("nvrtc rejected {name} at {NVRTC_ARCH_FLOOR}: {text}");
        }
    }
}

/// Compile every `.cu` unit. NVRTC needs no device, so this catches a broken
/// kernel source on any box with the toolkit rather than on the next box
/// with a GPU.
#[test]
fn nvrtc_compiles_every_unit() {
    for (name, src) in nvrtc_units() {
        if compile_at_floor(name, src).is_none() {
            return;
        }
        eprintln!("nvrtc ok: {name}");
    }
}

/// Assemble every unit's PTX with `ptxas`.
///
/// NVRTC copies inline `asm` bodies through verbatim and cudarc only asks it
/// for PTX, so a typo'd mnemonic clears both the clang parse and the NVRTC
/// compile and surfaces as `CUDA_ERROR_INVALID_PTX` at driver JIT. `ptxas` is
/// that assembler, and needs no device.
#[test]
fn ptxas_assembles_every_unit() {
    let arch = NVRTC_ARCH_FLOOR.replace("compute_", "sm_");
    if Command::new("ptxas").arg("--version").output().is_err() {
        eprintln!("skipping: ptxas not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join("aethergraph-ptxas");
    std::fs::create_dir_all(&dir).expect("ptxas scratch dir");
    for (name, src) in nvrtc_units() {
        let Some(ptx) = compile_at_floor(name, src) else {
            return;
        };
        let path = dir.join(format!("{name}.ptx"));
        std::fs::write(&path, ptx.to_src()).expect("write ptx");
        let out = Command::new("ptxas")
            .arg(format!("--gpu-name={arch}"))
            .arg("--output-file")
            .arg(dir.join(format!("{name}.cubin")))
            .arg(&path)
            .output()
            .expect("run ptxas");
        assert!(
            out.status.success(),
            "ptxas rejected {name} at {arch}:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        eprintln!("ptxas ok: {name}");
    }
}
