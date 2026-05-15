FROM ubuntu:24.04 AS dev

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_BACKTRACE=1

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

# nightly is needed for miri / cargo-fuzz.
ARG RUST_VERSION=1.95.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain ${RUST_VERSION} \
                   --component clippy,rustfmt \
 && rustup toolchain install nightly --component miri \
 && cargo install --locked cargo-deny cargo-audit cargo-fuzz \
 && rustup --version && cargo --version

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && ln -s /root/.local/bin/uv /usr/local/bin/uv \
 && uv --version

WORKDIR /workspace

# Loading rdma_rxe is the host's responsibility; the container only adds
# the ibverbs userland.
FROM dev AS dev-rdma
RUN apt-get update && apt-get install -y --no-install-recommends \
        ibverbs-utils \
        libibverbs-dev \
        librdmacm-dev \
        rdma-core \
        perftest \
    && rm -rf /var/lib/apt/lists/*
