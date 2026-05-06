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

    if target_os == "linux" && rdma {
        let mut build = cc::Build::new();
        build.file("csrc/ibv_shim.c").flag("-O2");
        if efa {
            build.define("AETHER_EFA_SHIM", None);
        }
        build.compile("aether_ibv_shim");
        println!("cargo:rerun-if-changed=csrc/ibv_shim.c");
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rustc-link-lib=ibverbs");
        if efa {
            println!("cargo:rustc-link-lib=efa");
        }
    }
}
