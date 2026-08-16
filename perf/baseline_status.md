# QuixiCore Metal Baseline Status

Method and measurement policy live in `perf/perf.md`. The running experiment
log is `perf/optimization_status.md` (append-only, oldest-first). Distilled
established truths are in `perf/findings.md`; the active idea queue is
`perf/backlog.md`. Raw benchmark output lives under `perf/results/`
(gitignored) — only summaries and conclusions are copied into these files.

## Environment

Dated 2026-07-24, from the most recent recorded runs
(`perf/results/2026-07-24/basert-*`): MacBook Pro Mac17,6, Apple M5 Max,
128 GB; macOS 26.5.2 (25F84); Xcode 26.6 (17F113); Apple Metal 32023.883 /
Metal toolchain 17.6.109.0; Python 3.12.13; MLX 0.21.1; PyTorch 2.13.0 MPS;
working-tree label `bc968fc-dirty`.

Hardware-era note: every recorded number dated 2026-07-14 or earlier was
measured on the previous machine — MacBook Pro Mac16,5, Apple M4 Max
(40-core GPU, 128 GB, ~546 GB/s theoretical DRAM), macOS 26.5.1/26.5.2,
Python 3.12.9 — and has not been re-measured on the M5 Max.

## Build + gate (the standing invariant)

The gate before any landed kernel/routing/benchmark change or performance
claim: build both integrations (`scripts/build kernels`,
`scripts/build pytorch_mps`), pass `scripts/test correctness`,
`scripts/test parity`, `scripts/test mps`, and the Xcode build-for-testing,
and complete at least one focused benchmark run recorded in
`perf/optimization_status.md`. Last recorded full gate (2026-07-24, BaseRT
completion tranche): correctness 2,434 passed, cross-backend parity 464,
direct MPS 604, Python package tests 44, Xcode build-for-testing succeeded,
kernel registry parsed, `git diff --check` clean.

Each real baseline entry should record:

- Apple Silicon model and memory configuration.
- macOS version and Xcode/Metal toolchain version.
- Integration path: MLX, PyTorch MPS, native Metal harness, or Xcode test.
- Git commit or working-tree label.
- Command line and benchmark script path.
- Warmups, measured iterations, median, variance, and raw result path.
- Correctness tolerance and observed error.

The shared harness (`perf/bench_kernels.py`, schema v1) performs a one-second
GPU clock ramp, at least 50 ms of per-thunk warmup, adaptive batching to at
least 2 ms per synchronized sample, and reports per-call median, p20/p80, and
CV. Conclusions drawn from the pre-2026-07-01 per-call-sync harness did not
all survive (see `perf/findings.md`, patterns).

### Current Harness Index

| Area | Source | Notes |
|---|---|---|
| Attention timing | `perf/harness/time_attn.py` | Focused timing for attention kernels |
| GEMM timing | `perf/harness/time_gemm.py` | Focused timing for GEMM kernels |
| LayerNorm timing | `perf/harness/time_layernorm.py` | Focused timing for row-reduction kernels |
| General timing helpers | `perf/harness/time_perf.py` | Shared harness utilities |
| Shared kernel harness | `perf/bench_kernels.py` | Schema-v1 correctness and timing cases for every active family, including specialized composed operations |
| Kernel notebook | `perf/optimization_status.md` | Detailed historical optimization entries |

Note: the `perf/harness/time_*.py` entry points predate the shared harness
and time one call per sync; prefer `perf/bench_kernels.py` for anything that
will be recorded (`perf/perf.md`).

## Kernel roofline snapshot (dated)

Dated 2026-07-01/02, Apple M4 Max (~546 GB/s theoretical DRAM). Measured
anchor points from the notebook:

- hadamard D=512: 554 GB/s ≈ roofline (2026-07-01, Baseline classification).
- glu vec4 at 16384×4096: 362 GB/s (2026-07-01).
- qgemv packed-weight bandwidth after E1/E2/E3: 245–466 W-GB/s across
  formats; q4_K at vocab scale (32000×4096): 307 W-GB/s, 2.5× vs fp16
  matmul (2026-07-01, comprehensive validation).
- gelu_bwd vec4: fp32 ~352–415 GB/s, bf16 ~250–344 GB/s (2026-07-03,
  Wave-7). dropout vec4: ~246 GB/s (2026-07-03, Wave-8).
- cascade_attention: 212–255 GB/s (2026-07-02, Wave-6 perf pass).
- Caveat from `perf/perf.md`: batched-timing re-reads can sit in the SLC and
  report above-DRAM bandwidth — A/B comparisons stay valid; absolute GB/s at
  or above peak means cache-resident.

M5 Max roofline snapshot: TBD (record on next Apple Silicon session).

## Per-family status

Synthesized from recorded notebook statements only; dates name the entry
that supports each line. M4-era lines are unverified on M5 Max.

| family | status | evidence |
|---|---|---|
| Elementwise/row (layernorm, rms_norm, softmax, gelu, add_norm) | ahead of MLX fast ops 1.5–2.6×; untouched | 2026-07-01 |
| rotary / glu / add_rt / hadamard | LANDED flat-vectorized geometry; hadamard at roofline | 2026-07-01 |
| Dense attention (fwd/causal/bwd) | 1.24× / 3.8–5.8× / 2.1–2.5× ahead of sdpa; q16 tile for D=128; multiwarp stays non-default | 2026-07-01 |
| attn_q (quantized KV) | multiwarp auto route landed (q8_0 0.447 ms); residual ~2.5× structural gap vs pre-dequantized KV — Beam #2 | 2026-07-01 / 2026-07-14 |
| Linear/state-space attention | chunked L=64 pipeline landed (2.4–10×); non-causal routes to framework | 2026-07-01 |
| Dense GEMM (matmul_custom/gemm_staged/flux) | calibration kernels at parity; flux epilogues 1.08–1.16×; no routing changes | 2026-07-01 |
| qgemv (float formats) | every format beats fp16 GEMV 2–3.1×, 245–466 W-GB/s; moderate-N BitNet shapes accepted limitation — Beam #5 | 2026-07-01 |
| qgemm / prefill | parity with fp16 matmul (footprint win); k-quants route dequant+GEMM at M≥64 | 2026-07-01 |
| Serving paged decode / MLA | v2 default, partition_size 256; MLA partitioned 1.7–6.3×; fp8 read penalty 1.25×; staged-vs-v1 regime question open — Beam #4 | 2026-07-01 |
| Quantized grouped MoE | rect within 5–8% of dense at 4–8.5× fewer bytes; SwiGLU two-warp landed (MXFP8/FP8); prefill gap open — Beam #1 | 2026-07-05 / 2026-07-14 |
| Sampling / spec-decode / beam | beam advance 2.5–6.5×; samplers bandwidth- or overhead-bound at floor; rejection/EAGLE ports overhead-bound | 2026-07-02 / 2026-07-05 |
| Training (CE, norm/gelu/dropout bwd, AdamW, BitNet) | fused-linear CE 9.5–13.8×; fused norm bwd ≈ mx.fast; BitNet quant kernels 1.9–3.7× | 2026-07-02 / 2026-07-03 / 2026-07-07 |
| Embedding / multimodal glue | lookup 2.3× vs framework gather; packed lookup/bag 2.25–4.96× | 2026-07-02 / 2026-07-13 |
| Quant-format hot paths (NVFP4/MXFP4/MXFP8/FP8) | whole-block/fragment decoders, format specialization, bit encoders landed per format; wire layouts unchanged | 2026-07-13 / 2026-07-14 |
| Fused serving ops (mean_pool, qgemv_fused, rms_norm_residual_next, qk_norm_rope_kv_f16, attn_fwd_sg_d256) | kept, 1.1–3.2× (D=256 attention 1.7–2.5×) | 2026-07-22 |
| BaseQN (Q2–Q8 canonical planes) | GEMV 7.67× vs composition; M>1 = dequant+framework GEMM; fused SwiGLU/QKV(K≤1024); MoE split-K SwiGLU only; LM-head composed | 2026-07-23 |
| Extended RoPE / Q8_0 KV / GDN io / LoRA / BERT | kept with measured routes (RoPE 1.5–4.9×; Q8_0 direct paged read 1.46–2.91×; LoRA direct only M≤4,R≤16) | 2026-07-23 |
| BaseRT vision/audio | direct kernels kept where they win (up to 25×); permutation-only ops routed to framework | 2026-07-24 |

## Deferred (bigger projects, flagged not faked)

- selective_scan Blelloch/chunked time-scan rewrite (2026-07-05, Wave-9 K5).
- GDN chunked-WY parallel prefill — Beam #3 (2026-07-05 / 2026-07-23).
- TurboQuant attention integration: rotated-domain V accumulate (2026-07-05).
- MInference prefill block-sparse consumer + CSR merge (2026-07-05).
- DSA indexer fused sparse-decode consumer — blocked on a reusable contract
  (2026-07-05 / 2026-07-14).
- MLA fp8 656-byte-layout cp_gather_upconvert_fp8_mla (2026-07-05).
- LM-head non-quant T>1 weight-amortized tile rewrite (2026-07-03).
- attn_q 32-row KV tiles or op-level dequant-to-scratch route — Beam #2
  (2026-07-01).
- copy_and_expand_eagle_inputs full padded-batch builder (2026-07-05).
- Norm→quant: standalone non-add variant, scale_ub, block sizes ≠128
  (2026-07-05).
- mean_pool_rms_l2 multi-simdgroup large-M variant (2026-07-22).
- NVFP4/MXFP4 QGEMM: a genuinely different matrix execution strategy
  (2026-07-13).
- M5 Max re-baseline of all M4-era thresholds and the roofline snapshot —
  TBD (record on next Apple Silicon session).

## Decision log

- 2026-08-15: file restructured into the fleet-canonical shape; the
  2026-07-01 optimization queue is superseded by `perf/backlog.md`;
  distilled truths split out to `perf/findings.md`. No numbers changed.
- 2026-07-23/24: BaseQN + BaseRT tranche routes recorded (direct where
  measured wins; framework for permutation-only and long-memory cases).
- 2026-07-22: hardware switch to Apple M5 Max (Mac17,6) with the new fused
  serving ops; no cross-era re-measurement performed.
- 2026-07-13/14: quant-format hot-path passes (NVFP4, MXFP4, MXFP8, FP8)
  landed whole-block/fragment/specialization wins; wire layouts unchanged;
  split-plane MXFP8 layout removed.
- 2026-07-07: BitNet ports kept; qgemm_bwd/gemm_v3 explicitly not exposed
  as speedups.
- 2026-07-03: Wave-7 reversals landed (fused norm backward, gelu_bwd vec4).
- 2026-07-02: partition_size default 512→256; single-call decode A/B keeps
  v2 default; Wave-4 audit found no shippable win in the new families.
- 2026-07-01: harness rewritten (schema v1); two timing-bias fixes (clock
  ramp, time-based warmup) — several queue items shown to be artifacts;
  qgemv E1/E2/E3 (commit a38e8a8) and flat-vectorized elementwise
  (commit f16b9d5) landed; linear_attn/hedgehog routed to framework.

## Superseded (historical)

The most recent (2026-07-24) references from the index below are promoted
into Environment and Build + gate above. The pre-restructure header's
"Most recent run date: 2026-07-23" was stale (2026-07-24 runs were already
indexed below) and is corrected in Environment. Everything below is retained
verbatim as history; nothing was deleted.

### Baseline snapshots (historical run index)

#### 2026-07-13 Specialized Operation Baseline

The focused MLX baseline and final optimized runs for packed embeddings, decode
epilogues/SwiGLU, masked and candidate output projection, spatial projection,
and functional cache attention are indexed below. Exact per-case p20/p80, CV,
correctness error, and framework baseline statistics are in the JSONL files;
durable decisions are in `perf/optimization_status.md`.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| Initial geometry | `294f8bd-dirty` | MLX / quick | 5 / 20 | `perf/results/2026-07-13/new-kernels-baseline-quick/` |
| Candidate geometry | `294f8bd-dirty` | MLX / quick | 5 / 20 | `perf/results/2026-07-13/new-kernels-candidate-quick/` |
| Final edge shapes | `294f8bd-dirty` | MLX / smoke | 3 / 5 | `perf/results/2026-07-13/new-kernels-final-smoke/` |
| Final priority shapes | `294f8bd-dirty` | MLX / quick | 10 / 30 | `perf/results/2026-07-13/new-kernels-final-quick/` |
| Second-pass baseline | `294f8bd-dirty` | MLX / quick | 10 / 30 | `perf/results/2026-07-13/new-kernels-second-pass-baseline-quick/` |
| Second-pass final | `294f8bd-dirty` | MLX / quick | 15 / 50 | `perf/results/2026-07-13/new-kernels-second-pass-final-quick/` |
| Cache edge | `294f8bd-dirty` | MLX / smoke | 10 / 40 | `perf/results/2026-07-13/new-kernels-second-pass-cache-smoke/` |
| Cache comprehensive | `294f8bd-dirty` | MLX / comprehensive | 10 / 30 | `perf/results/2026-07-13/new-kernels-second-pass-cache-comprehensive/` |
| Cache MPS edge | `294f8bd-dirty` | PyTorch MPS / smoke | 10 / 40 | `perf/results/2026-07-13/new-kernels-second-pass-cache-mps-smoke/` |
| Cache MPS priority | `294f8bd-dirty` | PyTorch MPS / quick | 10 / 30 | `perf/results/2026-07-13/new-kernels-second-pass-cache-mps-quick/` |
| Cache MPS comprehensive | `294f8bd-dirty` | PyTorch MPS / comprehensive | 10 / 30 | `perf/results/2026-07-13/new-kernels-second-pass-cache-mps-comprehensive/` |
| Cross-kernel follow-up baseline | `bc90717` | MLX / quick | 10 / 30 | `perf/results/2026-07-13/cross-kernel-followups-baseline/` |
| Cross-kernel follow-up final | `bc90717-dirty` | MLX / quick | 10 / 30 | `perf/results/2026-07-13/cross-kernel-final-mlx/` |
| Cross-kernel follow-up MPS | `bc90717-dirty` | PyTorch MPS / quick | 10 / 30 | `perf/results/2026-07-13/cross-kernel-final-mps/` |
| NVFP4 inference baseline | `c880769-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-13/nvfp4-experiments-baseline/` |
| NVFP4 inference final | `c880769-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-13/nvfp4-experiments-final/` |
| MXFP4 inference baseline | `3cab797-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-13/mxfp4-inference-baseline/` |
| MXFP4 generic coverage control | `3cab797-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-13/mxfp4-coverage-generic/` |
| MXFP4 inference final | `3cab797-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-13/mxfp4-inference-final/` |

#### 2026-07-14 MXFP8 Coverage Baseline

The compatibility run covers packed embedding lookup/bag, decode epilogues and
SwiGLU, LM-head sampling/sparse projection/beam advance, rectangular and
SwiGLU quantized MoE, and single-/multi-warp quantized-KV attention. Per-case
p20/p80, CV, correctness fields, and equivalent controls are in the JSONL;
durable decisions are in `perf/optimization_status.md`.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| MXFP8 inference brainstorm baseline | `3cab797-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-14/mxfp8-inference-baseline/` |
| MXFP8 QGEMV repeat | `3cab797-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-14/mxfp8-qgemv-baseline-repeat/` |
| MXFP8 inference comprehensive | `3cab797-dirty` | MLX / comprehensive | 10 / 30 | `perf/results/2026-07-14/mxfp8-inference-comprehensive/` |
| MXFP8 coverage control | `3cab797-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-14/mxfp8-coverage-generic/` |

#### 2026-07-14 MXFP8 Hot-Path Optimization Index

These runs use the MLX integration on MacBook Pro Mac16,5 (Apple M4 Max,
40-core GPU, 128 GB), macOS 26.5.1 (25F80), Xcode 26.6 (17F113), Apple Metal
32023.883 / toolchain 17.6.109.0, Python 3.12.9, and MLX 0.21.1. Exact
p20/p80, CV, correctness fields, and controls are in each JSONL; all durable
decisions are in `perf/optimization_status.md`.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| Core baseline | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/mxfp8-experiments-baseline-core/` |
| Fused baseline | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/mxfp8-experiments-baseline-fused/` |
| QGEMV whole/span interleaved | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/mxfp8-exp-qgemv-whole-vs-span-interleaved/` |
| Masked LUT narrow route | `455463c-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-14/mxfp8-exp-masked-lut-narrow/` |
| MoE SwiGLU scale broadcast | `455463c-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-14/mxfp8-exp-moe-scale-shuffle-swiglu-only/` |
| MoE SwiGLU two warp | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/mxfp8-exp-moe-swiglu-2warp/` |
| Beam matrix route (MLX) | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/mxfp8-exp-beam-matrix-all/` |
| Beam row control (MLX) | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/mxfp8-exp-beam-row-control/` |
| Beam matrix route (MPS) | `455463c-dirty` | PyTorch MPS / quick | 15 / 60 | `perf/results/2026-07-14/mxfp8-exp-beam-matrix-mps/` |
| Beam row control (MPS) | `455463c-dirty` | PyTorch MPS / quick | 15 / 60 | `perf/results/2026-07-14/mxfp8-exp-beam-row-control-mps/` |
| QGEMM 2x32 comprehensive | `455463c-dirty` | MLX / comprehensive | 10 / 40 | `perf/results/2026-07-14/mxfp8-exp-qgemm-2x32-comprehensive/` |
| Attention staging/warp sweep | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/mxfp8-exp-attn-mw4-stage4-repeat/` |
| Split-plane repeat | `455463c-dirty` | MLX / quick | 30 / 200 | `perf/results/2026-07-14/mxfp8-exp-split-plane-repeat/` |
| Final retained state | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/mxfp8-experiments-final-quick/` |

#### 2026-07-14 FP8 Hot-Path Optimization Index

These runs use the MLX integration on MacBook Pro Mac16,5 (Apple M4 Max,
40-core GPU, 128 GB), macOS 26.5.1 (25F80), Xcode 26.6 (17F113), Apple Metal
32023.883 / toolchain 17.6.109.0, Python 3.12.9, and MLX 0.21.1. Each JSONL
contains the exact shape, dtype/format, median, p20/p80, CV, framework control
where available, and throughput fields. Keep/reject decisions and correctness
commands are recorded in `perf/optimization_status.md`.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| Core inference baseline | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-experiments-baseline-core/` |
| Serving baseline | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-experiments-baseline-serving/` |
| Added-coverage baseline | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-experiments-baseline-added/` |
| Paged scale-hoist control | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-paged-scale-hoist-baseline-d64-d128/` |
| Paged scale-hoist candidate | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-paged-scale-hoist-candidate-d64-d128/` |
| Paged format-specialization control | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-paged-format-specialization-baseline/` |
| Paged format-specialization candidate | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-paged-format-specialization-candidate/` |
| E4M3 bit-encoder control repeat | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-bit-encoder-baseline-repeat/` |
| E4M3 bit-encoder candidate repeat | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-bit-encoder-candidate-repeat/` |
| E5M2 bit-encoder control | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-e5m2-encoder-baseline/` |
| E5M2 bit-encoder candidate | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-e5m2-encoder-candidate/` |
| FP8 SwiGLU two-warp candidate | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-moe-swiglu-two-warp-candidate/` |
| QGEMM shared-weight candidate | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-qgemm-shared-weight-64-candidate/` |
| Blockscale grouped-scale candidate | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-blockscale-grouped-scale-candidate/` |
| Quantized-attention topology candidates | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-attn-q-d128-q16-candidate/` |
| QGEMV whole-block candidate | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-qgemv-whole-block-candidate/` |
| Final retained state | `455463c-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/fp8-experiments-final-quick/` |

#### 2026-07-14 Cross-Kernel FP8 Transfer Index

These MLX runs use MacBook Pro Mac16,5 (Apple M4 Max, 40-core GPU,
128 GB), macOS 26.5.2 (25F84), Xcode 26.6 (17F113), Apple Metal
32023.883 / toolchain 17.6.109.0, Python 3.12.9, and MLX 0.21.1.
Exact per-case p20/p80, CV, formats, and shapes are in each JSONL. The
keep/reject analysis is in `perf/optimization_status.md`.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| Expanded baseline repeat | `376e5e4-dirty` | MLX / quick | 15 / 60 | `perf/results/2026-07-14/cross-kernel-specialization-baseline-repeat/` |
| Atomic-zero corrected baseline | `376e5e4-dirty` | MLX / quick | 30 / 120 | `perf/results/2026-07-14/atomic-zero-sentinel-corrected-baseline/` |
| Group-mode runtime A/B control | `376e5e4-dirty` | MLX / quick | 50 / 240 | `perf/results/2026-07-14/act-quant-group-runtime-ab-control/` |
| Group-mode specialized A/B candidate | `376e5e4-dirty` | MLX / quick | 50 / 240 | `perf/results/2026-07-14/act-quant-group-specialized-ab-candidate/` |
| Production-only retained state | `376e5e4-dirty` | MLX / quick | 50 / 240 | `perf/results/2026-07-14/cross-kernel-transfer-final-retained/` |
| Attention softcap repeat | `376e5e4-dirty` | MLX / quick | 40 / 180 | `perf/results/2026-07-14/attention-softcap-specialization-candidate-repeat/` |
| MXFP8 decode SwiGLU two-warp | `376e5e4-dirty` | MLX / quick | 50 / 240 | `perf/results/2026-07-14/decode-swiglu-mxfp8-two-warp-candidate/` |

#### 2026-07-23 Canonical BaseQN Baseline

These MLX runs use MacBook Pro Mac17,6 (Apple M5 Max, 128 GB), macOS
26.5.2 (25F84), Xcode 26.6 (17F113), Apple Metal 32023.883 / toolchain
17.6.109.0, Python 3.12.13, and MLX 0.21.1. They cover the canonical
separate-plane BaseQN operation contract with F16 scale/bias storage and F16
activations. Exact correctness error, median, p20/p80, CV, and composition
baseline fields are in each JSONL; decisions are in
`perf/optimization_status.md`.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| GEMV one-value control | `bc968fc-dirty` | MLX / smoke, Q4 | 5 / 30 | `perf/results/2026-07-23/basert-baseq-before/` |
| GEMV eight-value candidate | `bc968fc-dirty` | MLX / smoke, Q4 | 5 / 30 | `perf/results/2026-07-23/basert-baseq-after/` |
| Direct GEMM rejection run | `bc968fc-dirty` | MLX / quick, Q4 | 5 / 30 | `perf/results/2026-07-23/basert-baseq-gemm-direct/` |
| Final Q3/Q4/Q6/Q8 route | `bc968fc-dirty` | MLX / quick | 5 / 30 | `perf/results/2026-07-23/basert-baseq-final/` |

#### 2026-07-23 BaseQN Fused Consumer Baseline

Same Apple M5 Max/macOS/Xcode/Metal/Python/MLX environment as the canonical
BaseQN baseline above. Correctness error, median, p20/p80, CV, and composition
timings are in the JSONL; the keep/reject record is in
`perf/optimization_status.md`.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| QKV all-grid control | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/153250-mlx-quick/` |
| Decoded-value threshold rejection | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/basert-baseq-fused-threshold-final/` |
| Retained K-bracketed route | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/basert-baseq-fused-k-route-final/` |

#### 2026-07-23 BaseQN LM-Head Routing

Same Apple M5 Max environment and measurement method as the BaseQN sections
above. The optimization notebook records why both dedicated reductions were
removed in favor of columnwise direct GEMV composition.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| Eight-simdgroup reduction control | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/basert-baseq-lm-head-strong-baseline/` |
| Serial-simdgroup reduction candidate | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/basert-baseq-lm-head-serial-candidate/` |
| Retained columnwise GEMV route | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/basert-baseq-lm-head-retained/` |
| Retained QuixiCore-argmax route | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/basert-baseq-lm-head-retained-argmax-sampling/` |

#### 2026-07-23 BaseQN Grouped Expert Baseline

Same Apple M5 Max environment as the BaseQN sections above. Expert stacks use
the canonical separate code/scale/bias planes and the existing 32-row padded
MoE schedule. The optimization notebook records the operation-specific
one-simdgroup versus four-way split-K decision.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| One-simdgroup control | `bc968fc-dirty` | MLX / smoke, Q4 | 5 / 20 | `perf/results/2026-07-23/basert-baseq-moe-one-warp/` |
| Four-way split-K candidate | `bc968fc-dirty` | MLX / smoke, Q4 | 5 / 20 | `perf/results/2026-07-23/basert-baseq-moe-four-warp/` |
| Retained geometry quick repeat | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/basert-baseq-moe-final-repeat/` |
| Retained Q3/Q4/Q6/Q8 sweep | `bc968fc-dirty` | MLX / smoke | 20 / 50 | `perf/results/2026-07-23/basert-baseq-moe-format-sweep/` |

#### 2026-07-23 Extended RoPE Baseline

Same Apple M5 Max/macOS/Xcode/Metal/Python/MLX environment as the BaseQN
sections above. These runs compare explicit-position partial and three-axis
M-RoPE kernels, plus fused Q/K RMSNorm variants, against framework table
gather/rotation/concatenation compositions. Exact error, median, p20/p80, CV,
and decisions are recorded in the raw JSONL and optimization notebook.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| Generic positioned/M-RoPE smoke | `bc968fc-dirty` | MLX / smoke | 3 / 20 | `perf/results/2026-07-23/basert-rope-generic-smoke/` |
| Generic positioned/M-RoPE retained quick | `bc968fc-dirty` | MLX / quick | 3 / 20 | `perf/results/2026-07-23/basert-rope-generic-quick/` |
| Fused Q/K positioned/M-RoPE smoke | `bc968fc-dirty` | MLX / smoke | 3 / 20 | `perf/results/2026-07-23/basert-qk-rope-extended-smoke/` |
| Fused Q/K positioned/M-RoPE retained quick | `bc968fc-dirty` | MLX / quick | 3 / 20 | `perf/results/2026-07-23/basert-qk-rope-extended-quick/` |

#### 2026-07-23 Q8_0 KV Baseline

Same Apple M5 Max/macOS/Xcode/Metal/Python/MLX environment as the BaseQN
sections above. The codec runs compare Q8_0 encode/decode with equivalent BF16
cache copies. The attention runs compare direct Q8_0 paged reads with both
direct BF16 paged reads and Q8_0 gather followed by framework SDPA. The quick
runs request 10 warmups and 40 synchronized samples in addition to the
harness's 50 ms clock warmup and adaptive batching. Exact median, p20/p80, CV,
shape, and control fields are in the JSONL.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| Codec smoke | `bc968fc-dirty` | MLX / smoke | 3 / 10 | `perf/results/2026-07-23/basert-q8-kv-codec-smoke/` |
| Paged-read smoke | `bc968fc-dirty` | MLX / smoke | 3 / 10 | `perf/results/2026-07-23/basert-q8-kv-attn-smoke/` |
| Codec quick | `bc968fc-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-23/basert-q8-kv-codec-quick/` |
| Paged-read retained quick | `bc968fc-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-23/basert-q8-kv-attn-quick/` |

#### 2026-07-23 Gated DeltaNet I/O Baseline

Same Apple M5 Max/macOS/Xcode/Metal/Python/MLX environment as the BaseQN
sections above. This run compares fused GDN short-convolution SiLU, QKV
split/normalization, decay/beta, gated RMSNorm, and the reusable sigmoid output
gate with equivalent unfused or framework compositions. Exact median,
p20/p80, CV, shape, and speedup fields are in the JSONL; decisions are in
`perf/optimization_status.md`.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| Preparation/output smoke | `bc968fc-dirty` | MLX / smoke | 2 / 3 | `perf/results/2026-07-23/basert-gdn-io-smoke/` |
| Preparation/output retained quick | `bc968fc-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-23/basert-gdn-io-quick/` |

#### 2026-07-23 BaseRT Completion Kernel Baselines

These runs use the Apple M5 Max/macOS/Xcode/Metal/Python/MLX environment
recorded above. They cover calibration/output transforms, fused LoRA routing,
and BERT embedding/pooling tensor primitives. Exact medians, p20/p80, CV,
shapes, and framework controls are in each JSONL; keep/reject decisions are in
`perf/optimization_status.md`.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| Calibration absmax and logit softcap | `bc968fc-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-23/basert-aux-optimized-quick/` |
| Fused LoRA route sweep | `bc968fc-dirty` | MLX / comprehensive | 10 / 60 | `perf/results/2026-07-23/basert-lora-fused-routing/` |
| BERT token/type embedding and masked pooling | `bc968fc-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-23/basert-embedding-quick/` |

#### 2026-07-24 BaseRT Vision and Audio Kernel Baselines

Same Apple M5 Max environment. Vision compares general patch extraction,
position interpolation, and pooling with framework compositions. Audio
compares general/depthwise convolution and short/long-memory cross-attention
routes. The notebook records which direct candidates were retained or rejected.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| Vision patch/position/pooling route | `bc968fc-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-24/basert-vision-quick/` |
| Audio convolution and cross-attention route | `bc968fc-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-24/basert-audio-cross-quick/` |
| Strict BaseRT vision/audio contract audit | `bc968fc-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-24/basert-contract-quick/` |
| Qwen temporal/spatial patch route | `bc968fc-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-24/basert-qwen3d-routing/` |
| Gemma scalar value-clip route | `bc968fc-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-24/basert-value-clip-routing/` |
| Qwen global-split vision RoPE | `bc968fc-dirty` | MLX / quick | 10 / 40 | `perf/results/2026-07-24/basert-qwen-vision-rope/` |

### Migration Tasks

- Promote stable benchmark runs into compact per-kernel baseline tables.
- Keep large profiler traces out of git; record trace paths and summaries only.
- Store normalized raw output under `perf/results/YYYY-MM-DD/<kernel>/<run-id>/`.
