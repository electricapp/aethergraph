# K5.3 — herd7 litmus for the PTX seqlock reader

Prerequisites: herd7 with an NVIDIA/PTX model (see KERNELS.md Verification).

```sh
herd7 -model nvidia seqlock_publish_acquire.litmus
herd7 -model nvidia seqlock_odd_head.litmus
herd7 -model nvidia seqlock_payload_recheck.litmus
```

The forbidden `exists` clauses encode the memory-model claims the device reader
in `seqlock_reader.cu` relies on: published versions are observed in order, an
odd head is never taken as published, and a payload word from a concurrent write
forces the post-copy head re-load to see that write. Clear `TODO(HARDWARE)` in
that file once all three report no allowed forbidden outcome on the model you
trust for sys-scoped PTX.
