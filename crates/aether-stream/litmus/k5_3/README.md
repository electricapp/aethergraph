# K5.3 README — herd7 litmus for PTX seqlock acquire/release claims.

#

# Prerequisites: herd7 with an NVIDIA/PTX model (see KERNELS.md Verification).

#

# herd7 -model nvidia seqlock_publish_acquire.litmus

# herd7 -model nvidia seqlock_odd_head.litmus

#

# The forbidden `exists` clauses encode the memory-model claims the device

# reader in seqlock_reader.cu relies on. Clear TODO(HARDWARE) in that file

# once both litmus files report no allowed forbidden outcomes on the model

# you trust for sys-scoped PTX.
