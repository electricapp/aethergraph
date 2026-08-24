// Build the libibverbs shim so the data-path static-inline functions
// (ibv_post_send / ibv_poll_cq) become real ABI symbols Rust can link against.
//
// With the `efa` feature we additionally compile the SRD (EFA) helpers behind
// AETHER_EFA_SHIM and link libefa. The C compiler will pull the efadv headers
// transitively if needed; the link step is what activates the provider.

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let rdma = std::env::var("CARGO_FEATURE_RDMA").is_ok();
    let efa = std::env::var("CARGO_FEATURE_EFA").is_ok();
    let mlx5dv = std::env::var("CARGO_FEATURE_MLX5DV").is_ok();
    let gdrcopy = std::env::var("CARGO_FEATURE_GDRCOPY").is_ok();
    let xdp_bpf = std::env::var("CARGO_FEATURE_XDP_BPF").is_ok();

    // GDRCopy binds NVIDIA's userspace libgdrapi. The link is only needed
    // when the feature is on; `cargo check` records it without linking, so
    // the compile-gate CI job type-checks the FFI without the library
    // present.
    if target_os == "linux" && gdrcopy {
        println!("cargo:rustc-link-lib=gdrapi");
    }

    if target_os == "linux" && rdma {
        let mut build = cc::Build::new();
        build.file("csrc/ibv_shim.c").flag("-O2");
        if efa {
            build.define("AETHER_EFA_SHIM", None);
        }
        if mlx5dv {
            build.file("csrc/mlx5dv_shim.c");
            println!("cargo:rustc-link-lib=mlx5");
            println!("cargo:rerun-if-changed=csrc/mlx5dv_shim.c");
        }
        build.compile("aether_ibv_shim");
        println!("cargo:rerun-if-changed=csrc/ibv_shim.c");

        // Cross-check our #[repr(C)] ibverbs mirrors against the real headers.
        // The file is only `_Static_assert`s, so it compiles to an empty object
        // but fails the build on any ABI drift. Separate `cc::Build` so it
        // cannot perturb the shim's compile flags.
        cc::Build::new()
            .file("csrc/abi_assert.c")
            .compile("aether_abi_assert");
        println!("cargo:rerun-if-changed=csrc/abi_assert.c");

        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rustc-link-lib=ibverbs");
        if efa {
            println!("cargo:rustc-link-lib=efa");
        }
    }

    if target_os == "linux" && xdp_bpf {
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
        let obj = format!("{out_dir}/xsk_redirect.bpf.o");
        let src = "bpf/src/xsk_redirect.c";

        // Resolve `bpf/bpf_helpers.h` across distros. Order of preference:
        //   1. `pkg-config --cflags libbpf`   (Debian/Ubuntu with libbpf-dev,
        //      Fedora with libbpf-devel, AL2023 with libbpf-devel)
        //   2. fall back to scanning a few well-known locations directly so
        //      the build still works when pkg-config isn't installed
        let mut includes: Vec<String> = Vec::new();
        if let Ok(out) = std::process::Command::new("pkg-config")
            .args(["--cflags", "libbpf"])
            .output()
            && out.status.success()
        {
            let flags = String::from_utf8_lossy(&out.stdout);
            for tok in flags.split_whitespace() {
                if let Some(path) = tok.strip_prefix("-I") {
                    includes.push(format!("-I{path}"));
                }
            }
        }
        if includes.is_empty() {
            for candidate in [
                "/usr/include/bpf",
                "/usr/include/x86_64-linux-gnu",
                "/usr/include/aarch64-linux-gnu",
                "/usr/include",
                "/usr/local/include",
            ] {
                if std::path::Path::new(candidate).is_dir() {
                    includes.push(format!("-I{candidate}"));
                }
            }
        }

        let mut cmd = std::process::Command::new("clang");
        cmd.args(["--target=bpf", "-O2", "-g", "-c", src, "-o", &obj]);
        cmd.args(&includes);
        let status = cmd.status();
        match status {
            Ok(s) if s.success() => {
                println!("cargo:rustc-env=XSK_REDIRECT_BPF_OBJ={obj}");
                println!("cargo:rerun-if-changed={src}");
            }
            Ok(s) => panic!(
                "clang failed to compile {src} (exit {s}); install clang + \
                 libbpf-dev (or libbpf-devel), or drop the `xdp_bpf` feature"
            ),
            Err(e) => panic!(
                "could not invoke clang to compile {src}: {e}; install clang + \
                 libbpf-dev (or libbpf-devel), or drop the `xdp_bpf` feature"
            ),
        }
    }

    // K4.1: track the sched_ext skeleton so edits rebuild; optional compile
    // when `sched_ext_bpf` is enabled (soft-fail if clang/bpf headers missing).
    println!("cargo:rerun-if-changed=bpf/src/sched_ext_aether.c");
    if cfg!(all(target_os = "linux", feature = "sched_ext_bpf")) {
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
        let obj = format!("{out_dir}/sched_ext_aether.bpf.o");
        let src = "bpf/src/sched_ext_aether.c";
        let status = std::process::Command::new("clang")
            .args(["-O2", "-g", "-target", "bpf", "-c", src, "-o", &obj])
            .status();
        match status {
            Ok(s) if s.success() => {
                println!("cargo:rustc-env=AETHER_SCHED_EXT_BPF_OBJ={obj}");
            }
            Ok(_) | Err(_) => {
                println!(
                    "cargo:warning=sched_ext_bpf: clang --target=bpf compile skipped/failed; \
                     skeleton left as source-only"
                );
            }
        }
    }
}
