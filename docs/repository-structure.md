# Repository Structure

QuixiCore-Metal should share the same contract-facing structure as the other
QuixiCore backends while preserving a native Apple developer workflow through
Xcode, Metal Shading Language, Objective-C++, PyTorch MPS, and MLX.

The rule is: public taxonomy is common; Xcode and framework integration are
metadata and bindings over that taxonomy.

## Target Layout

```text
QuixiCore-Metal/
  .quixicore/
    backend.yaml
    kernels.yaml
    quant-formats.yaml

  docs/
    repository-structure.md
    development.md
    kernel-roadmap.md
    performance.md
    backend-notes.md

  include/
    metal/
      MetalSingle.hpp
      tk.metal
      ops/
    quixicore/metal/
      backend.hpp
      runtime.hpp
      ops.hpp

  src/
    runtime/
    dispatch/
    errors/

  kernels/
    common/
    norms/
    activations/
    attention/
    linear_attention/
    ssm/
    matmul/
    quantization/
    vision/
    moe/
    sampling/
    serving/
    optimizers/
    collectives/
    utils/

  bindings/
    c/
    python/
    pytorch_mps/
    mlx/
    swift/

  tests/
    correctness/
    integration/
    smoke/
    testdata/

  perf/
    harness/
    configs/
    results/
    baselines/

  examples/
  scripts/
  tools/
  assets/

  Configs/
    Debug.xcconfig
    Release.xcconfig

  QuixiCoreMetal.xcodeproj/
```

The previous monolithic source tree has been moved into this canonical layout.
New work should land directly under the top-level taxonomy below.

## Xcode Structure

Xcode is the normal developer entrypoint for Metal, but the filesystem remains
canonical. The `.xcodeproj` should reference source files in the canonical
directories instead of creating a separate conceptual tree.

Recommended Xcode groups:

```text
QuixiCoreMetal
  Public Headers      -> include/
  Host Runtime        -> src/
  Kernels             -> kernels/
    norms/
    activations/
    attention/
    linear_attention/
    ssm/
    matmul/
    quantization/
    vision/
    moe/
    sampling/
    serving/
    optimizers/
    collectives/
    utils/
  Bindings
    PyTorch MPS       -> bindings/pytorch_mps/
    MLX               -> bindings/mlx/
    Python            -> bindings/python/
    Swift             -> bindings/swift/
  Tests               -> tests/
  Benchmarks          -> perf/
  Examples            -> examples/
  Docs                -> docs/
```

Recommended Xcode targets:

- `QuixiCoreMetal`: host runtime plus compiled Metal libraries.
- `QuixiCoreMetalTests`: correctness and integration tests.
- `QuixiCoreMetalBenchmarks`: benchmark and profiling entrypoints.
- `QuixiCoreMetalPyTorch`: optional PyTorch MPS extension.
- `QuixiCoreMetalMLX`: optional MLX integration.
- `QuixiCoreMetalSwift`: optional Swift-facing API.

Xcode rules:

- Commit shared schemes and project settings needed by every developer.
- Do not commit user-specific `xcuserdata/`.
- Keep `.metal`, `.mm`, `.hpp`, and test sources in canonical directories; the
  project should only reference them.
- Use `Configs/*.xcconfig` for shared SDK, deployment target, Metal language
  version, warning, and optimization settings.
- Keep generated build products in derived data or ignored build directories.

## Manifests

`.quixicore/backend.yaml` identifies this repository as the Metal backend and
declares supported Apple GPU families and contract compatibility.

`.quixicore/kernels.yaml` should be the machine-readable parity source for
implemented operations:

```yaml
operations:
  paged_attention:
    family: attention
    status: implemented
    path: kernels/attention/paged_attention
    bindings:
      pytorch_mps: bindings/pytorch_mps/paged_attention.mm
      mlx: bindings/mlx/paged_attention.mm
    tests:
      correctness: tests/correctness/attention/paged_attention
    benchmarks:
      default: perf/configs/attention_paged.yaml
    variants:
      - name: metal_apple_gpu_family_8
        status: optimized
```

`.quixicore/quant-formats.yaml` should list supported quant formats, packing
layouts, and any Metal-only layout constraints.

## Kernel Families

The top-level directories under `kernels/` are semantic families, not Xcode
groups or framework buckets:

- `norms/`: RMSNorm, LayerNorm, add-norm, norm-to-quant, QK norm.
- `activations/`: GELU, GLU, SiLU/SwiGLU helpers, standalone softmax.
- `attention/`: flash attention, causal/non-causal/varlen attention, backward,
  paged attention, MLA, rotary, quantized-KV attention, functional cache decode,
  state merging.
- `linear_attention/`: Based, Hedgehog, linear attention, causal/decay linear
  attention, GDN, complex linear attention primitives.
- `ssm/`: Mamba, SSD, selective scan, FFT convolution.
- `matmul/`: dense/staged/complex GEMM, Flux, decode epilogues, and packed
  decode SwiGLU.
- `quantization/`: act quant, runtime quant, qgemm, qgemv, quantized LM head,
  packed embedding lookup/reduction, sparse candidate projection,
  fp8/int8/fp4 packing, TurboQuant.
- `vision/`: window-attention support operations, patch merging,
  space-to-depth/norm/projection, and fixed-shape projection heads.
- `moe/`: routing, expert alignment, gather/scatter, grouped GEMM, quantized
  MoE GEMM, LoRA alignment, finalize.
- `sampling/`: sampling, logit transforms, penalties, rejection sampling, beam
  search, speculative decode and EAGLE helpers.
- `serving/`: KV cache mutation, block/page tables, indexers, MInference, cache
  copy/gather helpers.
- `optimizers/`: AdamW and other training optimizer kernels.
- `collectives/`: multi-device operations only where meaningful on Apple
  platforms; these are capability-gated extensions.
- `utils/`: bit packing, column permutation, Hadamard/FWHT, small reusable
  user-visible utilities.

## Operation Layout

Use one directory per contract operation. The filesystem rule is semantic
first, target variant second: do not create branch-only, architecture-first, or
framework-first kernel trees.

```text
kernels/<family>/<operation>/
  README.md
  common/
    include/
    src/
  variants/
    metal_apple_gpu_family_7/
      include/
      src/
      tests/
      bench/
    metal_apple_gpu_family_8/
      include/
      src/
      tests/
      bench/
    metal_simdgroup/
      include/
      src/
      tests/
      bench/
  tests/
  bench/
```

`common/` is for target-independent operation code only. Anything that depends
on an Apple GPU family, SIMD-group feature, threadgroup memory assumption,
pipeline specialization, framework-specific launch path, or shape-specific
tuning belongs under a variant directory.

For very small operations, direct `.metal` or `.mm` files under the operation
directory are acceptable only when they are genuinely target-independent. As
soon as an operation has more than one architecture- or framework-specific
implementation, move all target-specific source into `variants/`.

Framework glue belongs under `bindings/`, not inside kernel operation
directories.

Branches are not the architecture boundary. A backend repository should keep
all supported target variants in `main` side by side, with Xcode targets and
scripts selecting variants by requested or detected target.

## Target-Specific Internals

Public backend headers remain under `include/quixicore/metal/`, and shared
native Metal support can remain under `include/metal/`. Low-level implementation
headers that differ by Apple GPU family or framework integration should use
explicit internal target directories, for example:

```text
include/internal/metal/
  common/
  apple_gpu_family_7/
  apple_gpu_family_8/
  simdgroup/
```

Operation variants may include these internal headers, but contract-facing
headers should not expose target-specific implementation layouts.

## Tests And Benchmarks

Correctness and performance assets should mirror the kernel taxonomy:

```text
tests/correctness/<family>/<operation>/
perf/configs/<family>_<operation>.yaml
perf/baselines/<family>/<operation>/
```

Common developer entrypoints should exist:

```text
scripts/configure
scripts/build
scripts/test
scripts/bench
scripts/coverage-report
scripts/clean
```

For Metal these scripts should wrap `xcodebuild`, MLX/PyTorch extension builds,
or direct command-line Metal tooling as appropriate.

## Completed Migration Map

| Former area | Current area |
| --- | --- |
| `ThunderMittens/kernels/*` | `kernels/<family>/<operation>/` |
| `ThunderMittens/kernels/tk_torch` | `bindings/pytorch_mps/` |
| `ThunderMittens/mlx` | `bindings/mlx/` |
| `ThunderMittens/include` | `include/metal/` and `kernels/common/` |
| `ThunderMittens/tests` | `tests/` |
| `ThunderMittens/kernels/tests_parity` | `tests/correctness/` |
| `ThunderMittens.xcodeproj` | `QuixiCoreMetal.xcodeproj/` |

Rename APIs only when synchronizing the CUDA, Metal, ROCm, XPU, and Gaudi
bindings deliberately.

## Rules For New Work

- Add new kernels under semantic family directories.
- Keep PyTorch MPS, MLX, Python, Swift, and Objective-C++ glue in `bindings/`.
- Keep Xcode project state as references to canonical source files.
- Put Apple GPU family and simdgroup-specific tuning under operation variants.
- Keep all supported Metal target variants in the repo; do not depend on one
  branch per architecture or framework target.
- Build, test, benchmark, and Xcode entrypoints must filter variants by the
  selected Metal target and must not compile unrelated target variants.
- If an operation has no meaningful Metal implementation, mark it unsupported in
  metadata rather than adding a stub kernel.
