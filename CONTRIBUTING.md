# Contributing To QuixiCore Metal

QuixiCore Metal is one native backend in the QuixiCore family. Contributions
should preserve the shared QuixiCore contract while using Metal-native
implementation techniques.

## Backend Boundary

- Implementation code belongs in this repository, not in the QuixiCore umbrella
  repository and not in another backend.
- Shared semantics belong in QuixiAI/QuixiCore.
- Metal-specific tuning, MSL, Objective-C++, Xcode project state, MLX glue, and
  PyTorch MPS glue belong in this repository.

## Adding Or Changing A Kernel

1. Put source under `kernels/<family>/<operation>/`.
2. Keep framework glue in `bindings/`; do not hide MLX or MPS bindings inside
   kernel directories.
3. Add or update correctness coverage under `tests/correctness/<family>/<operation>/`.
4. Add or update benchmark coverage under `perf/`.
5. Update `.quixicore/kernels.yaml`.
6. Update `.quixicore/quant-formats.yaml` when quant layouts, packing, or
   supported formats change.

## Required Checks

Use the common entrypoints when possible:

```bash
scripts/configure
scripts/build
scripts/test
scripts/bench
scripts/coverage-report
```

For Xcode work, commit shared project and scheme changes only. Do not commit
user-specific `xcuserdata/`. Mention the Apple Silicon model, Xcode version,
and integration path tested: Xcode, MLX, PyTorch MPS, or a combination.

## Pull Request Checklist

- Kernel semantics match the QuixiCore contract or document a Metal-only
  extension.
- Correctness tests cover the changed behavior.
- Benchmarks cover the relevant shapes or explain why benchmark coverage is not
  applicable yet.
- `.quixicore/kernels.yaml` reflects implementation status.
- `.quixicore/quant-formats.yaml` reflects quant format support.
- New source follows `docs/repository-structure.md`.
