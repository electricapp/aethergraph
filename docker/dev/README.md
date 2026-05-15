# Linux dev container

Run the `cfg(target_os = "linux")` test surface from any host (macOS, Windows,
Linux).

## Quick start

```sh
# Default tier — runs every cross-platform + every Linux-gated test that
# doesn't need physical RDMA hardware. Covers ~95% of the surface.
docker compose -f docker/dev/compose.yaml run --rm dev

# Inside the container:
cargo test --workspace                          # rust suite
cargo test --workspace --features rdma          # +RDMA tests (most still #[ignore]'d)
uv run --project python maturin develop         # build the python wheel
uv run --project python pytest python/tests     # python integration
```

## RDMA tier (Linux host only)

The `dev-rdma` tier adds `libibverbs` + `librdmacm`. To run the SoftRoCE / SRD
tests against an actual RDMA fabric, your host kernel must have `rdma_rxe`
loaded:

```sh
# On a Linux host:
sudo modprobe rdma_rxe
sudo rdma link add rxe0 type rxe netdev lo   # or any netdev

# Then:
docker compose -f docker/dev/compose.yaml --profile rdma run --rm dev-rdma
# inside:
cargo test -p aether-stream --features rdma --tests -- --ignored
```

Docker Desktop on macOS / Windows does **not** expose an RDMA fabric to
containers — Mac users can build with `--features rdma` (verifies the link
graph) but cannot run the gated tests. CI's `rdma-hardware` job uses a
self-hosted Linux runner for that.
