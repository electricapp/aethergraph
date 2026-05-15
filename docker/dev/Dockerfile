# Linux dev image. Lets macOS / Windows contributors run the
# `cfg(target_os = "linux")` test surface (~41 tests) without needing a
# physical Linux machine. Two tiers:
#
#   `dev`     — Ubuntu + Rust + Python toolchain. Runs everything EXCEPT
#               the SoftRoCE / SRD tests that need a real RDMA fabric or
#               the `rdma_rxe` kernel module. ~95% of the gated tests.
#
#   `dev-rdma`— `dev` + ibverbs userland. The `rdma_rxe` module itself
#               lives in the host kernel, so this tier still needs the
#               host to have `rdma_rxe` loaded (Linux only). Use the
#               compose file's `rdma` profile for that.
#
# Build:  docker compose -f docker/dev/compose.yaml build dev
# Run:    docker compose -f docker/dev/compose.yaml run --rm dev

FROM ubuntu:24.04 AS dev

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_BACKTRACE=1

# Build essentials + Python + the libs aether-stream's Linux-only deps
# reach for (libc/libpthread are implicit; libibverbs lives in the rdma
# stage).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        git \
        libc6-dev \
        libssl-dev \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Rust stable + nightly (nightly is for miri / cargo-fuzz). Same default
# components as `rust-toolchain.toml` so the image picks up clippy + rustfmt.
ARG RUST_VERSION=1.95.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain ${RUST_VERSION} \
                   --component clippy,rustfmt \
 && rustup toolchain install nightly --component miri \
 && cargo install --locked cargo-deny cargo-audit cargo-fuzz \
 && rustup --version && cargo --version

# uv for the Python side — matches what `python/pyproject.toml [dependency-groups]`
# expects. Installing this in /root means `uv` is on PATH for the default
# (root) container user.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && ln -s /root/.local/bin/uv /usr/local/bin/uv \
 && uv --version

WORKDIR /workspace

# --- rdma tier ------------------------------------------------------------
#
# Adds ibverbs userland for the `rdma` feature build. Loading the
# `rdma_rxe` kernel module is the HOST's responsibility (the container
# can't insmod). Compose `rdma` profile launches with the right
# `--cap-add=NET_ADMIN` and shared `/dev/infiniband` mount.
FROM dev AS dev-rdma
RUN apt-get update && apt-get install -y --no-install-recommends \
        ibverbs-utils \
        libibverbs-dev \
        librdmacm-dev \
        rdma-core \
        perftest \
    && rm -rf /var/lib/apt/lists/*
