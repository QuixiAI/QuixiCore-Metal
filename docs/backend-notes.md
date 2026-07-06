# Backend Notes

QuixiCore Metal is the Apple implementation of the QuixiCore contract. Metal
source may use MSL, simdgroups, Objective-C++, Xcode targets, MLX integration,
and PyTorch MPS integration when those choices remain behind the shared
operation semantics.

Active kernel sources now live under `kernels/<family>/<operation>/`. Framework
bindings live under `bindings/`, and shared Metal substrate headers live under
`include/metal/`.
