# QuixiCore Metal Performance Handbook

This document is the operating guide for optimizing every kernel under
`kernels/` for Apple Metal. The goal is not to collect tricks. The
goal is to run a disciplined loop: find references, form a bottleneck hypothesis,
measure a clean baseline, run controlled experiments, keep only verified wins,
and record enough detail that the next pass can start from evidence instead of
memory.

The running notebook for the effort is `perf/optimization_status.md`. Baseline
snapshots live in `perf/baseline_status.md`.

## Principles

Optimization starts from correctness and measurement. A change is not a win until
it passes the kernel's correctness tests, improves the target metric on realistic
shapes, and does not introduce a material regression on supported edge shapes.

Prefer experiments that attack a specific bottleneck:

- Memory-bound: reduce bytes moved, improve coalescing, improve cache reuse,
  avoid extra global-memory passes, or use narrower formats.
- Compute-bound: increase arithmetic intensity, use `simdgroup_matrix` more
  effectively, reduce scalar side work, or fuse epilogues.
- Latency-bound: increase resident work, improve launch geometry, reduce serial
  loops, and avoid per-lane divergence.
- Synchronization-bound: remove unnecessary `threadgroup_barrier` calls, reduce
  threadgroup-memory traffic, or switch to simdgroup-local reductions when
  cross-simdgroup sharing does not pay.
- Launch-bound: fuse tiny kernels, batch work into fewer dispatches, or route
  small shapes to an existing MLX primitive when launch overhead dominates.

The Apple-specific baseline assumption is important: do not blindly port H100
machinery. `docs/porting/primitives.md` records that Metal has no direct
`cp.async`/TMA equivalent. Apple also documents that optimized GEMM on Apple
silicon often does not need explicit threadgroup-memory staging, and that enough
occupancy can naturally overlap memory and compute. Our existing
`gemm_staged`/`attn_multiwarp` measurements agree: staging can be correct and
competitive without being faster.

## Repo Facts To Preserve

Core kernel sources are in `kernels/<family>/<operation>/`:

- Elementwise and row kernels: `add_rt`, `gelu`, `layernorm`, `rms_norm`,
  `rotary`, `softmax`, `glu`, `hadamard`, `add_norm` (fused add+norm(+fp8)).
- Dense GEMM/fusion: `matmul_custom`, `gemm_staged`, `flux`,
  `cmplx_matmul`, `fftconv`.
- Attention: `attn_fwd`, `attn_causal`, `attn_multiwarp`, `attn_bwd`,
  `attn_q`.
- Linear/state-space attention family: `linear_attn`, `lin_attn_causal`,
  `lin_attn_decay`, `hedgehog`, `based`, `mamba2`.
- Quantized matmul/decode/fusion: `qgemm`, `qgemv`, `qflux`,
  `qgemm_int`, `qgemv_int`, `quant_rt` (runtime per-tensor/per-token quant).
- Serving/decode: `paged_attn_v2` (paged decode v1/staged/v2, fp8/alibi/
  block-sparse/xcache variants), `mla` (DeepSeek MLA insert/decode incl. fp8),
  `kv_cache`, `rope_kv`, `moe` (route/permute/grouped-GEMM/finalize),
  `sampling`.

Device context for all current numbers: Apple M4 Max, 40-core GPU, 128 GB,
theoretical DRAM bandwidth ~546 GB/s. Memory-bound kernels should be judged as a
fraction of that roofline; note that repeatedly re-read buffers (batched timing)
can sit in the SLC and report above-DRAM bandwidth — A/B comparisons remain
valid, absolute GB/s near or above peak means "cache-resident", not "magic".

The MLX extension builds all active kernels from
`bindings/python/CMakeLists.txt`. Python dispatch lives in
`bindings/python/tk/__init__.py`, with MLX bindings in
`bindings/python/bindings.cpp` and PyTorch MPS support under
`bindings/pytorch_mps/`.

Correctness tests mirror the kernel taxonomy:

```bash
scripts/test correctness
scripts/test mps
scripts/test parity
```

Legacy timing entry points (`perf/harness/time_perf.py`, `time_gemm.py`,
`time_attn.py`, `time_layernorm.py`) predate the harness and time one call per
sync; prefer `perf/bench_kernels.py` for anything that will be recorded.

The shared benchmark harness is `perf/bench_kernels.py`. Run it from the repo
root with the repo virtualenv or another environment that has MLX/NumPy
installed:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset smoke --kernel all
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel all
.venv/bin/python perf/bench_kernels.py --backend torch --preset smoke --kernel all
```

Useful narrowing examples:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset quick --kernel qgemv,qgemm
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qgemv --formats q8_0,q4_0,q4_K,bitnet
```

Packed quantized weights are cached under `perf/results/.wcache/` (the Python
encoders are slow for large N×K); delete the cache after changing an encoder.

Each run writes:

```text
perf/results/YYYY-MM-DD/<run-id>/run.json
perf/results/YYYY-MM-DD/<run-id>/results.jsonl
perf/results/YYYY-MM-DD/<run-id>/summary.md
```

`perf/results/` is git-ignored. Copy only summary snippets into
`perf/optimization_status.md` unless a raw result file is explicitly requested.

## Reference Search Protocol

For each kernel, start by finding any reference implementations and benchmark
scripts under `.reference/`. Record the exact files used in
`perf/optimization_status.md`.

Reference roots currently present in `.reference/` (2026-07-01): `vllm`,
`flashinfer`, `TensorRT-LLM`, `onnxruntime`. The mirrors below were used in
earlier phases and have been cleaned from disk — re-clone into `.reference/`
(git-ignored) if a kernel needs them:

- `ThunderKittens` — original CUDA algorithms and benchmarks.
- `llama.cpp` (`ggml/src/ggml-metal/`) — the most mature Metal kernels:
  quantized matmul/GEMV geometry (multiple rows per simdgroup, register-cached
  activations), attention, reductions.
- `vllm-metal` — Apple-focused serving kernels and benchmark tooling.
- `mlx` — MLX's own Metal kernels are the framework baseline's source of truth
  (`mlx/backend/metal/kernels/`).
- `BitNet`, `Metal-Puzzles`, `llm.metal`, `metal-benchmarks`, `HipKittens`.

Use targeted searches, for example:

```bash
rg -n "kernel_name|algorithm_name|qgemv|mul_mv|softmax|rope" .reference kernels
find .reference -path '*metal*' -o -path '*kernels*' | sort
```

When reading a reference, separate ideas into three buckets:

- Portable algorithm idea: worth considering.
- Hardware-specific CUDA/ROCm mechanism: translate only if Metal has a real
  analogue.
- Benchmark or shape idea: usually worth adopting even if the kernel code is not.

## Measurement Harness Requirements

Every benchmark result must include:

- Git commit or working-tree label.
- Kernel name and exact public entry point.
- Backend: MLX, PyTorch MPS, or both.
- Device name, macOS version, Python version, MLX version, PyTorch version if
  used, and power mode if known.
- Shape, dtype, quant format, causal flag, and any route flag such as
  `multiwarp=True`.
- Baseline implementation and target implementation.
- Warmup count, timed iteration count, repeat count, median, p20/p80 or
  min/max, and coefficient of variation when useful.
- Correctness tolerance and observed max absolute/relative error.
- Derived throughput: GB/s, GFLOP/s, tokens/s, or elements/s as appropriate.
- Raw output location.

`perf/bench_kernels.py` records these fields in JSONL using schema version 1.
Important columns include `backend`, `preset`, `kernel`, `variant`, `shape`,
`dtype`, `format`, target and baseline median times, p20/p80 times, GB/s,
weight-only GB/s for packed quantized weights, GFLOP/s, speedup, max absolute
and relative error, status, and skip reason.

Use warmups and explicit synchronization. MLX timings should force work with
`mx.eval(...)`; PyTorch MPS timings should synchronize with
`torch.mps.synchronize()`. Do not time only Python dispatch unless that is the
metric being studied.

Beware the sync-latency floor: one submit+sync round trip costs ~0.15-0.25 ms on
this stack, so timing one small kernel per sync measures dispatch latency, not
the kernel. `time_thunk` in the harness therefore batches several calls per
sample (adaptively, targeting >=2 ms per sample) and divides — a
throughput-style per-call latency. The measured 2026-07-01 effect: layernorm
(4096x1024) "improved" from 0.26 ms to 0.084 ms purely by fixing the
methodology. Per-call-sync latency is still worth a separate look for
launch-bound routing decisions (tiny shapes routed to MLX).

For low-level timing, Apple exposes command-buffer
`gpuStartTime`/`gpuEndTime` after completion, and Xcode/Instruments should be
used for GPU captures and counters when a timing A/B cannot explain the result.

Benchmark against at least three baselines:

- Framework baseline: `mx.matmul`, `mx.fast.scaled_dot_product_attention`,
  `mx.fast.layer_norm`, `mx.softmax`, `torch` equivalent, or a known MLX op.
- Naive decomposed baseline: for fused or quant kernels, materialize the obvious
  intermediate and call a framework primitive, for example
  `dequantize(wq) @ x`.
- Current QuixiCore Metal baseline: current `tk.<kernel>` implementation before
  the experiment.

For quantized decode, the decisive metric is effective weight bandwidth:

```text
effective_GBps = bytes_of_packed_weight_read / time_seconds / 1e9
```

For GEMM-like kernels:

```text
GEMM FLOPs = 2 * M * N * K
arithmetic_intensity ~= FLOPs / bytes_moved
```

For attention forward:

```text
approx FLOPs = 4 * B * H * N * N * D
```

For elementwise/normalization kernels, compute a conservative bytes-moved number
from required reads and writes. Be explicit when the estimate ignores cache reuse
or repeated passes.

## Shape Strategy

Do not optimize only square toy shapes. Each kernel needs:

- Small edge shapes: the lowest supported sizes and non-power-of-two dimensions
  if the API claims to support them.
- Tile-aligned shapes: expected fast path.
- Tile-ragged shapes: padding/slicing or boundary checks can dominate.
- Real model shapes: dimensions from Llama-style MLP/attention and BitNet where
  relevant.
- Stress shapes: long context, large K, large N, and batch sweeps.

Useful starter shapes:

- GEMM: `(M,N,K)` around `256`, `512`, `1024`, `2048`, plus rectangular LLM
  shapes such as `K=4096`, `N=11008`, `N=14336`, and smaller `M` prefill values.
- Attention: `(B,H,N,D)` with `D in {64,128}` and `N in {512,1024,2048,4096}`.
- Norm/softmax/GELU/rotary: rows in `{4096,16384,65536}` and hidden sizes in
  `{64,128,256,512,768,1024,2048,4096}` where supported.
- Quant GEMV/GEMM: BitNet and LLM projection shapes such as
  `N=3840,K=2560`, `N=13824,K=2560`, `N=2560,K=6912`, plus batch sweep
  `M in {1,2,4,8,16,32,64,128}`.

Record skipped shapes with the reason: unsupported API contract, out of memory,
known correctness gap, or not relevant to the kernel.

## Per-Kernel Optimization Loop

Run this loop for each active kernel directory.

1. Inventory the kernel.
   Identify public Python APIs, C++ launch code, Metal kernels, supported dtypes,
   supported shapes, current correctness tests, and existing benchmark coverage.

2. Find references.
   Search `.reference/` and docs. Write down reference files, benchmark shapes,
   and any ideas worth testing.

3. Establish baseline.
   Run correctness tests. Measure the current kernel against framework and naive
   baselines on the agreed shape set. Save raw output under `perf/results/` or a
   similar dated path once the harness exists. Summarize the table in the
   notebook.

4. Classify the bottleneck.
   Compute bytes, FLOPs, arithmetic intensity, achieved GB/s, achieved GFLOP/s,
   occupancy clues, and variance. Use Xcode/Instruments when the bottleneck is
   not clear from timing.

5. Define experiments.
   Write hypotheses before editing code. Each experiment should change one
   meaningful factor: tile shape, launch geometry, memory layout, fusion,
   barrier placement, dequant strategy, branch elimination, vectorization, or
   routing threshold.

6. Execute experiments.
   Keep experiments small and reversible. Rebuild, run the focused correctness
   test first, benchmark the same shape matrix, then run broader tests for any
   candidate winner.

7. Decide.
   Apply changes only if they improve the target shapes enough to justify any
   added complexity and do not regress important shapes. Reject noisy wins,
   wins isolated to toy shapes, and changes that degrade correctness tolerance.

8. Record and publish.
   Update `perf/optimization_status.md` with final numbers, rationale, and
   rejected alternatives. Commit and push only when the user explicitly asks for
   publication, using a normal descriptive commit message with no AI attribution.

## Experiment Catalogue

Use these as templates. Not every kernel needs every experiment.

### Launch Geometry

- Vary threads per threadgroup while keeping a multiple of the pipeline
  `threadExecutionWidth`.
- Sweep simdgroups per threadgroup: one simdgroup, two, four, and any established
  local pattern.
- Change rows/items per threadgroup for row kernels.
- For GEMM/attention, test output tile sizes that trade occupancy against reuse.
- For quantized GEMV, test one simdgroup per output row versus multiple rows per
  threadgroup and split-K reductions.

Watch for low occupancy, long tail effects when the number of threadgroups is
not divisible by GPU cores, and wasted work on partial edge tiles.

### Memory Layout And Coalescing

- Verify adjacent lanes read adjacent addresses on hot global-memory paths.
- Compare row-major shared/threadgroup layout with padded or swizzled layouts.
- For dequantized tiles, test llama.cpp-style blocked layouts versus natural
  row-major writes when `simdgroup_load` follows.
- Replace scalar loads/stores with vectorized loads/stores where alignment and
  correctness allow.
- Test read-only metadata/scales in `constant` address space or predecoded
  lookup tables when practical.

If a layout change helps, record whether the suspected cause is global-memory
coalescing, cache locality, threadgroup-memory bank behavior, or fewer address
calculations.

### Tiling And Reuse

- Sweep `BM`, `BN`, `BK`, sequence block size, and rows per block.
- Compare device-memory direct loads against explicit threadgroup staging.
- For K loops, test barrier frequency. Apple's MPP guide recommends tuning this
  balance because barriers can reduce available independent work even if they
  help cache working-set control.
- Test static/full-tile fast paths separate from dynamic/edge-tile paths.
- For GEMM, test threadgroup walk order, including linear order and a locality
  preserving order such as Morton order.

Keep the Apple lesson in view: staging is only useful if reuse beats added
barriers, extra stores/loads, and occupancy loss.

### Fusion

- Fuse epilogues such as bias, residual, scale, GELU, gate, or normalization
  when the intermediate would otherwise round-trip through device memory.
- Fuse dequantization with matmul or attention if dequantized weights/KV are used
  once.
- Split a fused kernel if register pressure, branching, or lower occupancy
  dominates the saved memory traffic.

Candidate fused kernels in this repo already include `flux`, `qflux`, attention
softmax/value accumulation, and quantized-KV attention. New fusion must beat the
decomposed MLX baseline, not just look elegant.

### Branches And Scalar Side Work

- Hoist format-dependent decisions out of inner loops through templates/function
  constants.
- Replace per-element branches in dequantization with table lookups, bitwise
  formulas, or block-level predecode where measurable.
- Specialize common dimensions such as `D=64`, `D=128`, `K%tile==0`, and
  supported quant block sizes.
- Remove repeated address arithmetic by precomputing base offsets and using
  simple increments in hot loops.

This is especially important for `qgemv`, `qgemm`, `attn_q`, and the integer
quant kernels because a small amount of scalar decode work can erase the benefit
of reduced weight bytes.

### Reductions And Numerics

- Use simdgroup reductions before threadgroup reductions when possible.
- Compare tree reductions, pairwise reductions, and accumulator precision.
- Keep fp32 accumulation for numerically sensitive softmax, norm, attention, and
  long K reductions unless a measured lower-precision variant passes tolerance.
- Test compensated or reordered accumulation only when correctness or variance
  points to a real issue.

Behavior must remain identical under the documented tolerances. For exact
integer kernels, exactness is part of the contract.

### Routing And Shape Specialization

- Find qgemv-to-qgemm crossover by sweeping `M`.
- Route tiny elementwise shapes to MLX if launch overhead dominates custom code.
- Add fast paths for aligned shapes only when edge handling remains correct.
- Keep generic padding/slicing outside hot kernels when the host overhead is
  smaller than in-kernel predicates.

Routing changes need API-level tests because they can silently affect both MLX
and PyTorch MPS backends.

## Kernel-Specific Starting Hypotheses

Use these as first-pass ideas. Replace them with measured facts as the project
progresses.

### `qgemv`

Highest priority for decode. Measure against `dequantize(wq) @ x` and fp16
`mx.matmul`. Effective packed-weight GB/s should scale with bits per weight. If
4-bit formats are not faster than 8-bit formats, investigate dequant branch cost,
uncoalesced packed loads, low occupancy from one simdgroup per output row, and
shared/register layout.

Experiments: format sweep, output rows per threadgroup, split-K, vectorized
packed loads, branchless dequant variants, scale/min predecode, blocked
dequantized layout, and decode-only microkernel.

### `qgemm` And `qflux`

Prefill may be compute-bound for larger `M`; do not assume quantized GEMM beats
`mx.matmul` on throughput. It can still be valuable for memory footprint. Compare
staged dequant, dequant-direct-to-fragment, `qgemm_direct`, and fused epilogues.
Revisit barrier placement in any staged path.

Experiments: `M` sweep, `BK` sweep, static aligned fast path, direct versus
threadgroup staging, epilogue fusion, act-order gather overhead, and 2D scale
layout for `fp8_block2d`.

### `qgemv_int` And `qgemm_int`

These buy exact integer-dot numerics, not guaranteed speed, because Apple GPUs do
not expose an int8 matrix unit through this substrate. Measure exact paths
against dequant-to-half paths and only optimize exact paths for workloads that
actually require exactness.

Experiments: `idot4` packing/alignment, multiple output rows per threadgroup,
activation layout, scale application placement, and split-K.

### `attn_fwd`, `attn_causal`, `attn_multiwarp`, `attn_bwd`

Attention performance depends on on-chip softmax state, Q/K/V memory traffic,
and launch geometry. Existing notes say multiwarp staging has not beaten the
single-simdgroup forward. Treat that as a baseline finding, then re-test on long
contexts and backward-specific shapes.

Experiments: sequence block size, D-specific specialization, causal branch
placement, K/V staging versus direct loads, multiwarp count, logsumexp storage
format, dQ versus dKV split geometry, and recompute-versus-store tradeoffs.

### `attn_q`

Quantized-KV attention should save bandwidth only if K/V dequantization does not
dominate. Compare against dequantized-KV SDPA and unquantized attention. Check
whether K and V want different dequant/layout strategies.

Experiments: K dequant-to-shared layout, V dequant-to-register layout, causal
and non-causal separate specialization, `multiwarp` routing, format sweep, and
dequant-only microkernel.

### `matmul_custom` And `gemm_staged`

The current evidence says simple single-simdgroup GEMM is near optimal for tested
shapes and staged multi-simdgroup GEMM can lose from occupancy/staging overhead.
Use these kernels as calibration points for the harness and for Apple-specific
tiling lessons.

Experiments: tile size sweep, threadgroup walk order, aligned fast path,
rectangular LLM shapes, edge-tile cost, and MPP/cooperative tensor exploration
only if the deployment target supports it.

### `flux`

Fused GEMM epilogues should be judged against `mx.matmul` plus separate MLX
epilogue ops. A win should come from avoiding intermediate writes.

Experiments: bias/gate/residual load coalescing, epilogue in register fragments,
tile sizes shared with GEMM, and activation approximation cost.

### `layernorm`, `rms_norm`, `softmax`

These are row reductions plus pointwise writes. They are usually memory- and
reduction-bound. Compare against MLX fast paths and report GB/s.

Experiments: rows per threadgroup, vectorized loads, simdgroup-only reductions
for small hidden sizes, two-pass versus one-pass variance, reciprocal/sqrt
placement, and specializing hidden sizes.

### `gelu`, `rotary`, `add_rt`

These are bandwidth-sensitive pointwise transforms unless transcendental math
dominates. Framework baselines may already be strong.

Experiments: vectorized contiguous loads/stores, fusing with neighboring ops,
shape routing for small tensors, and approximations only when the oracle allows
the same tolerance.

### `linear_attn`, `lin_attn_causal`, `lin_attn_decay`, `hedgehog`, `based`, `mamba2`

These kernels combine MMA-style work, scans/causal structure, and exp/decay
state. The likely bottleneck varies by sequence length and feature dimension.

Experiments: chunk size, state layout, intra-chunk causal mask placement,
feature-map materialization versus recompute, exp/exp2 choice where numerically
valid, and scan staging.

### `cmplx_matmul` And `fftconv`

Complex kernels stress register pressure, layout transforms, and intermediate
traffic. Measure against `torch.fft`/MLX equivalents where possible but keep in
mind that decomposition differs.

Experiments: complex tile layout, transpose strategy, radix/Monarch block size,
pointwise complex multiply fusion, and reducing global writes between stages.

### Serving kernels (`paged_attn_v2`, `mla`, `moe`, `kv_cache`, `rope_kv`, `quant_rt`, `sampling`)

Existing findings (keep unless re-measured otherwise): `paged_attention_v2`
partition/reduce beats v1 ~2.2x at long context (occupancy from the partition
grid axis); GQA KV-reuse staging is ~1.7x slower on MLX (grid collapse +
per-token barriers) and stays non-default. These decode kernels are KV-bandwidth
and occupancy bound.

Experiments: `partition_size` sweep per context length, fp8 dequant-on-read cost
vs bf16 cache, block_size sensitivity, MLA latent-cache read layout, MoE
grouped-GEMM vs per-expert dispatch crossover at small E, and scatter/gather
vectorization in `kv_cache`/`rope_kv`.

## Decision Rules

A change is a candidate winner when:

- It passes the focused correctness test for the kernel.
- It improves median performance on priority realistic shapes by at least 3% for
  low-risk local changes, or at least 8-10% for higher-complexity changes.
- It does not regress any required correctness shape.
- It does not regress secondary performance shapes by more than the agreed
  tolerance unless the routing intentionally narrows the target.
- It has a plausible explanation backed by bytes/FLOPs/counters or a clean A/B.

Reject or defer a change when:

- The win is inside measurement noise.
- The win appears only on tiny or artificial shapes.
- It adds substantial complexity without a durable real-shape win.
- It helps MLX but breaks or significantly regresses PyTorch MPS routing.
- It depends on a hardware/API feature unavailable to the supported deployment
  target.

## Recording Format

Each kernel section in `perf/optimization_status.md` should contain:

- Status: not started, baselining, experimenting, candidate, landed, deferred.
- Current best implementation and current public route.
- References inspected.
- Correctness command and last result.
- Baseline table.
- Experiment table.
- Decision log.
- Open questions.

Raw results should be stored in a stable location once a harness exists, for
example:

```text
perf/results/YYYY-MM-DD/<kernel>/<run-id>.json
perf/results/YYYY-MM-DD/<kernel>/<run-id>.txt
```

Do not commit enormous profiler traces unless explicitly requested. Record their
path, device, and summary instead.

## Final Verification Before Landing A Win

Before applying an optimization permanently:

```bash
python -m pytest tests/correctness/<family>/<operation> -q
scripts/test parity -q
python <relevant benchmark script or harness>
```

For shared substrate changes under `include/metal/`, also run the
broader kernel correctness suite and the Xcode primitive tests described in
`docs/porting/primitives.md`.

When publishing a verified improvement, include the performance table in the PR
or commit notes. Commit messages must be normal descriptive messages with no AI
co-author or generated-by trailer.

## External References

- Apple, [Metal Performance Primitives Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).
  Useful for current Apple silicon GEMM guidance, threadgroup/simdgroup tiling,
  barriers, threadgroup walk order, static extents, cooperative tensors, and
  postfix fusion.
- Apple Metal documentation, [Creating threads and threadgroups](https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups).
  Useful for SIMD-group execution behavior, divergence, `threadExecutionWidth`,
  and threadgroup sizing.
- Apple Metal documentation for
  [`MTLCommandBuffer.gpuStartTime`](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)
  and [`gpuEndTime`](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpuendtime).
  Useful for low-level command-buffer timing when MLX/PyTorch timing is not
  enough.
