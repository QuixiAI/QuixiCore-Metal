# Kernel Roadmap

The Metal backend already contains broad kernel coverage under the canonical
`kernels/<family>/<operation>/` tree. The roadmap is to make that coverage
explicit and comparable across QuixiCore backends.

Priorities:

1. Expand `.quixicore/kernels.yaml` from family-level status to operation-level
   status.
2. Bring Metal and CUDA operation coverage into parity.
3. Keep MLX and PyTorch MPS bindings aligned over the same kernel semantics.
