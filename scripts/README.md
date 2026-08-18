# scripts/

Local development gates and operational scripts for spinning up a fresh AWS
instance.

## `check-linux.sh`

Compiles the `cfg(target_os = "linux")` surface — io_uring gather, NVMe
passthrough, userfaultfd pager, memfd shared store, perf counters, NUMA — from
a non-Linux machine, using zig as the cross toolchain. `cargo check` on macOS
compiles none of it, so this is what catches a Linux-only break before CI does.

```bash
scripts/check-linux.sh                    # type-check
scripts/check-linux.sh --clippy --tests   # what CI runs
```

Needs `brew install zig` and `rustup target add x86_64-unknown-linux-gnu`. Run
it yourself before pushing; it is not a pre-commit or pre-push hook.
`zigcc` is the compiler shim it drives.

## `bootstrap_node.sh`

Installs build deps + ibverbs + (optionally) EFA + Rust on a fresh Ubuntu 24.04
instance. Idempotent — rerunning after an AMI refresh is safe.

```bash
# Default: toolchain + ibverbs + Rust 1.94.1
ssh ubuntu@<ip> bash ~/aether-scripts/bootstrap_node.sh

# EFA-equipped instance (c6gn.16xlarge, c7gn.16xlarge, p4d.24xlarge, etc.)
ssh ubuntu@<ip> bash ~/aether-scripts/bootstrap_node.sh --efa

# SoftRoCE on a non-EFA box (i4i.large etc.)
ssh ubuntu@<ip> bash ~/aether-scripts/bootstrap_node.sh --rxe

# GPU box with instance-store NVMe (g4dn.xlarge): CUDA + rxe + /nvme mount
ssh ubuntu@<ip> bash ~/aether-scripts/bootstrap_node.sh --rxe --cuda --nvme
```

`--efa`, `--rxe`, `--cuda`, `--nvme` are orthogonal; combine as hardware
supports. `--cuda` requires a reboot afterwards for the NVIDIA kernel module to
load; re-run with `--rxe --nvme` after reboot to restore SoftRoCE + mount (both
are lost across reboot since rxe is dynamic and `/etc/fstab` stays out of it —
the xfs survives but needs remount).

## AWS EFA security-group requirements

For cross-node EFA traffic (c6gn.16xlarge, c7gn.16xlarge, p4d.24xlarge, hpc7g
etc.), the nodes' shared security group needs **both an ingress AND an egress
rule that reference the SG itself** (all protocols, from/to `sg-…`). A generic
egress `0.0.0.0/0` rule is NOT sufficient — EFA traffic gets silently dropped.
AWS's own `fi_pingpong` exhibits the same failure. Symptom: RDMA READ completes
with `vendor_err=14` (`REMOTE_ERROR_UNKNOWN_PEER`) even though TCP / ICMP work
between the instances.

CLI setup for a fresh SG:

```bash
aws ec2 authorize-security-group-ingress --group-id "$SG" --source-group "$SG" --protocol -1
aws ec2 authorize-security-group-egress  --group-id "$SG" --source-group "$SG" --protocol -1
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr $(curl -s -4 ifconfig.me)/32
```
