# Kernel Roadmap

The Metal backend already contains broad kernel coverage under the legacy
`ThunderMittens/` tree. The roadmap is to make that coverage explicit and
comparable across QuixiCore backends.

Priorities:

1. Apply the semantic family taxonomy to new work.
2. Expand `.quixicore/kernels.yaml` from family-level status to operation-level
   status.
3. Bring Metal and CUDA operation coverage into parity.
4. Keep MLX and PyTorch MPS bindings aligned over the same kernel semantics.
