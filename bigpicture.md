# ThunderMittens Big Picture Strategy

ThunderMittens should become an Apple Metal / MLX home for the useful parts of the ThunderKittens ecosystem first, then a curated Apple Silicon kernel lab for strong ideas from related projects.

The order matters:

1. Port ThunderKittens to ThunderMittens until the core abstraction and kernel coverage are credible.
2. Use that stable ThunderMittens substrate to port selected kernels from HipKittens, MLX, vLLM Metal, Alloy, llama.cpp, llm.metal, and other references.
3. Keep every imported idea shaped like ThunderMittens, with consistent types, layouts, tests, bindings, and benchmarks.

## Long-Term Goal And Current Status (read this first)

**Goal: full ThunderKittens parity.** Eventually port all ~58 ThunderKittens kernels (see
`discrepencies.md`) to Apple Metal, each with a correctness oracle and a benchmark. We are doing this
**Foundation First**: make the build/validate loop real, lock down the existing kernels, then port the
next tier — but the destination is "all of it." Track progress in `docs/porting/thunderkittens.md`.

**The substrate is the asset.** `ThunderMittens/include/` is already a ~90% complete Metal port of TK's
primitive layer (register/shared/global types; `simdgroup_matrix` MMA wrappers `mma_AB/ABt/AtB/AtBt`;
global↔shared↔register load/store; row/col reductions; maps; conversions; swizzle). Most kernel ports
need **no new primitives** — they compose what is already there. Missing-primitive tracking lives in
`docs/porting/primitives.md`.

**Porting rule: port the algorithm, not the H100 code.** TK's fast kernels are deeply coupled to
NVIDIA hardware (TMA, WGMMA, warpgroups, `cp.async`, mbarriers). Do not translate that machinery.
Re-express the *algorithm* on the TM substrate using `simdgroup_matrix` + simd shuffles + threadgroup
memory. Drop async double-buffering for v1 (Metal has no `cp.async`); add staging only when a kernel
is correct and needs it. Difficulty order (easiest→hardest): layernorm (0 H100 features) → rotary →
softmax / flux → bf16 GEMM parity → causal / multi-warp attention → linear/based/hedgehog/mamba2/
fftconv → quantized GEMM (fp8/int8/mxfp8/nvfp4, may need emulation) → distributed/parallel.

**Build + validation environment (established — reuse it, do not re-derive):**
- Python **3.10 venv** at `/Users/eric/ThunderMittens/.venv` (homebrew Python is PEP-668 externally
  managed; the system default 3.14 is too new for this MLX snapshot). Call its `python`/`pip` by
  absolute path — the shell is fish and state does not persist between commands.
- Vendored **MLX 0.21.0** is built editable into that venv. The kernel `.cpp` files use MLX 0.21.0
  internal APIs (`allocator::malloc_or_wait`, `metal::device`, `get_command_encoder`,
  `set_input_array`/`set_bytes`/`dispatch_threadgroups`), so a modern PyPI wheel would NOT work — keep
  the vendored build. Two gotchas were required to build it:
  1. **Metal Toolchain** must be installed: `xcodebuild -downloadComponent MetalToolchain` (recent
     Xcode ships `metal`/`metallib` as a separate ~700 MB component; `xcrun --find metal` resolving a
     path is not enough).
  2. **Pin Python for CMake** or it links against the wrong interpreter:
     `CMAKE_ARGS="-DPython_EXECUTABLE=$VENV/bin/python -DPython3_EXECUTABLE=$VENV/bin/python"`.
     Both MLX's and the kernels' `CMakeBuild` append `$CMAKE_ARGS`.
- Build the kernels: `cd ThunderMittens/kernels && CMAKE_ARGS="-DPython_EXECUTABLE=$VENV/bin/python" \
  CMAKE_BUILD_PARALLEL_LEVEL=8 $VENV/bin/python setup.py build_ext -j8 --inplace`. `mlx.extension`
  auto-sets `MLX_DIR`, so `find_package(MLX)` resolves from the editable install. `kernels/setup.cfg`
  pins `build_base` to repo-root `/build` so CMake artifacts land **outside** the Xcode-synchronized
  `ThunderMittens/` tree (see below).
- Validate against MLX builtins: `mx.fast.layer_norm`, `mx.fast.rms_norm`, `mx.fast.rope`,
  `mx.fast.scaled_dot_product_attention` are ready-made oracles. Tests live in
  `kernels/<name>/correctness/`. `time_<name>.py` scripts benchmark vs the MLX builtin. All 4
  kernels pass: `cd kernels && $VENV/bin/python -m pytest */correctness/`.
- On-device primitive tests (`tests/unit/`) build & run via Xcode (90/90 pass):
  `xcodebuild -scheme ThunderMittens -configuration Debug build CODE_SIGNING_ALLOWED=NO` then run the
  built `ThunderMittens` binary. Two Xcode-16 synchronized-folder gotchas were fixed and must stay
  fixed: CMake `build/` must live outside `ThunderMittens/` (the `setup.cfg` relocation — its `.cpp.o.d`
  files are otherwise mis-compiled as DTrace → "Multiple commands produce"), and standalone-`main`
  files (`kernels/attn_fwd/correctness/c_attn.m`) plus the kernel `.cpp` must be in the target's
  `membershipExceptions` in `project.pbxproj`. Details in `docs/porting/primitives.md`.

**Worked example: `kernels/layernorm/`.** The canonical "anatomy of a port" — one simdgroup per row,
`rv_fl<D>` register vector, vec `sum`/`sub`/`mul`/`add` + inline `metal::rsqrt`, templated on `D`,
instantiated per-width via `[[host_name("layernorm_<D>")]]`, dispatched from `layernorm.cpp`, bound in
`bindings.cpp`, exported in `tk/__init__.py`, validated vs `mx.fast.layer_norm`. Copy this shape for
the next kernel.

## Goals

- Build a Metal Shading Language analogue of the ThunderKittens programming model.
- Expose useful kernels through MLX Python bindings where that improves testing, benchmarking, or usability.
- Maintain a clear split between reusable primitives and one-off benchmark kernels.
- Prefer correctness and inspectable structure first, then optimize.
- Make each port reproducible: source reference, shape/dtype constraints, validation method, benchmark method, and known gaps.

## Non-Goals

- Do not clone every reference repository into ThunderMittens.
- Do not preserve CUDA/HIP structure when it fights Apple GPU architecture.
- Do not add a kernel without a test or at least a clearly documented validation plan.
- Do not make ThunderMittens depend on `.reference/`; references are local source material, not runtime dependencies.

## Phase 0: Organize The Porting Map

`discrepencies.md` is the first inventory. Keep extending it, but split long-term tracking into smaller documents as needed:

- `docs/porting/thunderkittens.md`: ThunderKittens parity checklist.
- `docs/porting/external-kernels.md`: curated kernels from other repos.
- `docs/porting/primitives.md`: missing MSL primitive coverage.
- `docs/porting/benchmarks.md`: benchmark targets and methodology.

For each candidate kernel, track:

- Reference path and commit.
- Kernel family: attention, GEMM, norm, rotary, convolution, sequence/state-space, distributed, quantization, sampling, utility.
- Required primitives: global layout, register tile, shared tile, MMA, reductions, async/prefetch equivalent, swizzle/layout transforms.
- Supported dtype and shapes.
- Correctness oracle.
- Benchmark shape set.
- Port status: not started, primitives blocked, compiling, correct, benchmarked, optimized.

## Phase 1: ThunderKittens Parity First

ThunderKittens is the primary source of truth for what ThunderMittens should cover.

### 1. Stabilize The Core MSL Substrate

Before porting many kernels, make the primitive layer dependable:

- Register tiles and vectors.
- Shared tiles and vectors.
- Global layouts.
- Load/store paths between global, shared, and registers.
- Row/column reductions.
- Elementwise maps.
- MMA wrappers around Apple `simdgroup_matrix`.
- Layout conversion utilities.
- Unit tests for each primitive family.

The existing `ThunderMittens/include` tree is the right place for this. Resist adding kernel-local copies of primitives unless they are temporary experiments.

### 2. Bring The Test Harness Online

The unit test tree already exists but is gated by compile flags. Make it practical to run small suites:

- Start with low-intensity warp/register and memory tests.
- Add focused tests for any primitive needed by the next kernel.
- Keep Xcode tests for native MSL development.
- Add MLX Python correctness tests for public kernels.

Every port should answer: does the primitive work, does the kernel work, and does the MLX binding call the intended Metal kernel?

### 3. Port ThunderKittens Kernels In Dependency Order

Suggested order:

1. GEMM baseline: finish and validate `matmul_custom`, then map it against ThunderKittens BF16 GEMM variants.
2. Attention forward: finish and validate `attn_fwd`, then expand toward causal/non-causal MHA variants.
3. Small high-value transformer ops: layernorm, rotary, softmax, flux GELU/gate.
4. Sequence kernels: based/linear attention, hedgehog, mamba2, fftconv.
5. Quantized and newer GEMM variants: FP8, INT8, MXFP8, NVFP4 where Apple hardware and MSL support make sense.
6. Distributed/parallel kernels only after single-device kernels are solid.

For each ThunderKittens kernel:

- Read the reference kernel and test/benchmark harness.
- Identify missing ThunderMittens primitives.
- Port or improve primitives first.
- Implement a minimal Metal kernel.
- Add MLX binding only once the kernel has a stable interface.
- Validate numerically against MLX/PyTorch/reference CPU.
- Benchmark against MLX built-ins where possible.
- Document unsupported cases explicitly.

### 4. Do Not Chase Perfect CUDA Parity

ThunderKittens targets NVIDIA. Apple GPUs differ materially:

- SIMD group is 32 lanes, but memory hierarchy and threadgroup costs differ.
- Apple exposes `simdgroup_matrix` rather than NVIDIA tensor cores.
- Some CUDA scheduling strategies may not transfer.
- Some dtype targets, especially FP8/NVFP4/MXFP8, may need emulation or a different value proposition.

Port the idea, not the incidental CUDA shape.

## Phase 2: Curated External Kernel Ports

After ThunderKittens parity is useful, start selecting kernels from the other references. Selection should be deliberate.

### HipKittens

Use HipKittens as an algorithmic sibling, especially for newer kernel structure:

- GQA forward/backward attention.
- BF16/FP8/MXFP8 GEMM scheduling ideas.
- LayerNorm, rotary, softmax.
- AMD scheduling analyses as contrast material.

HipKittens is not a direct Metal implementation source, but it may contain better decompositions than older ThunderKittens kernels.

### MLX

Use MLX as the production Apple Metal reference:

- Extension build patterns.
- Runtime dispatch patterns.
- Steel GEMM and attention kernels.
- LayerNorm/RMSNorm/RoPE/softmax implementations.
- Quantized kernel conventions.

MLX is the closest reference for integration quality.

### vLLM Metal

Use vLLM Metal for inference-specific Apple kernels:

- Paged attention.
- Tiled paged attention using `simdgroup_matrix`.
- MLA.
- GDN linear attention and recurrent decode.
- KV cache update/reshape/copy patterns.
- TurboQuant ideas.

This repo is especially relevant after ThunderMittens has stable attention primitives.

### Alloy

Use Alloy as a compiler and abstraction reference:

- Tile IR design.
- MSL MMA emitter.
- Tunable GEMM layouts.
- Attention and decode kernels.
- Quantized matmul families.
- MoE and sampling kernels.

Do not blindly import Alloy's DSL. Instead, mine it for lowering strategies, shape planning, and MSL emission patterns that can simplify ThunderMittens primitives.

### llama.cpp

Use llama.cpp as a broad production kernel catalogue:

- Quantized matvec/matmul formats.
- RoPE variants.
- RMSNorm/norm/softmax.
- Flash attention implementations.
- SSM/RWKV/GDN-style kernels.
- GGUF-oriented kernel constraints.

It is less aligned with ThunderMittens structure, so prefer borrowing algorithms and edge-case handling over code shape.

### llm.metal

Use llm.metal for small, readable kernels and host-dispatch examples:

- GPT-2 forward-pass validation ideas.
- Simple layernorm, softmax, GELU, residual, cross entropy kernels.
- Minimal Objective-C Metal host code.

It is useful for onboarding and sanity checks, not the main performance target.

### Metal-Puzzles And metal-benchmarks

Use these for education and tuning:

- Metal-Puzzles: small MLX custom-kernel examples and debugging workflow.
- metal-benchmarks: Apple GPU architecture assumptions, instruction throughput, memory behavior.

They should inform development, not become product code.

## Kernel Acceptance Criteria

A kernel is not "ported" until it has:

- A reference source path and commit.
- A ThunderMittens Metal implementation.
- Shape and dtype constraints documented.
- Correctness tests.
- A benchmark script or benchmark entry.
- A known baseline comparison.
- MLX binding if it is user-facing.
- Clear status for CPU fallback, autodiff, and vmap support.

For research-only kernels, it is acceptable to skip CPU fallback and autodiff if the limitation is explicit.

## Repository Shape

Keep the repo organized around reusable structure:

- `ThunderMittens/include`: primitives, types, layouts, reusable ops.
- `ThunderMittens/kernels`: public kernels with MLX bindings.
- `ThunderMittens/tests`: primitive and native Metal tests.
- `benchmarks` or `ThunderMittens/kernels/*/bench*.py`: performance tests.
- `docs/porting`: tracking and design notes.

Avoid adding reference snapshots outside `.reference`.

## Practical Porting Loop

For each new kernel:

1. Pick a narrow shape/dtype target.
2. Write the correctness oracle first.
3. Identify missing primitives.
4. Add primitive tests.
5. Port the kernel in the simplest correct form.
6. Add MLX binding if appropriate.
7. Benchmark against a meaningful baseline.
8. Optimize only after correctness is stable.
9. Record what changed in the porting checklist.

## Near-Term Priorities

1. Clean up the current ThunderMittens kernel state:
   - Verify `add_rt` intent or rename/remove it.
   - Fix dtype/assert issues in `matmul_custom` and `attn_fwd`.
   - Add correctness tests for current exposed kernels.
2. Get a minimal repeatable build/test path.
3. Finish BF16 GEMM.
4. Finish dense BF16 attention forward.
5. Add LayerNorm, RoPE, and softmax as smaller confidence-building ports.
6. Revisit `discrepencies.md` and convert it into a maintained ThunderKittens parity checklist.

The guiding principle: ThunderMittens should become a coherent Apple Silicon kernel framework, not a pile of translated kernels. ThunderKittens parity gives the framework a backbone; the other references provide a menu of valuable next targets once that backbone is strong.
