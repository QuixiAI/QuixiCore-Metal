# QuixiCore Metal Baseline Status

Method and measurement policy are described in `perf/perf.md`. Raw benchmark
output should live under `perf/results/`; stable conclusions should be copied
into `perf/optimization_status.md`.

## Environment

Most recent run date: 2026-07-23.

Each real baseline entry should record:

- Apple Silicon model and memory configuration.
- macOS version and Xcode/Metal toolchain version.
- Integration path: MLX, PyTorch MPS, native Metal harness, or Xcode test.
- Git commit or working-tree label.
- Command line and benchmark script path.
- Warmups, measured iterations, median, variance, and raw result path.
- Correctness tolerance and observed error.

## Current Harness Index

| Area | Source | Notes |
|---|---|---|
| Attention timing | `perf/harness/time_attn.py` | Focused timing for attention kernels |
| GEMM timing | `perf/harness/time_gemm.py` | Focused timing for GEMM kernels |
| LayerNorm timing | `perf/harness/time_layernorm.py` | Focused timing for row-reduction kernels |
| General timing helpers | `perf/harness/time_perf.py` | Shared harness utilities |
| Shared kernel harness | `perf/bench_kernels.py` | Schema-v1 correctness and timing cases for every active family, including specialized composed operations |
| Kernel notebook | `perf/optimization_status.md` | Detailed historical optimization entries |

## 2026-07-13 Specialized Operation Baseline

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

## 2026-07-14 MXFP8 Coverage Baseline

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

## 2026-07-14 MXFP8 Hot-Path Optimization Index

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

## 2026-07-14 FP8 Hot-Path Optimization Index

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

## 2026-07-14 Cross-Kernel FP8 Transfer Index

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

## 2026-07-23 Canonical BaseQN Baseline

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

## 2026-07-23 BaseQN Fused Consumer Baseline

Same Apple M5 Max/macOS/Xcode/Metal/Python/MLX environment as the canonical
BaseQN baseline above. Correctness error, median, p20/p80, CV, and composition
timings are in the JSONL; the keep/reject record is in
`perf/optimization_status.md`.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| QKV all-grid control | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/153250-mlx-quick/` |
| Decoded-value threshold rejection | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/basert-baseq-fused-threshold-final/` |
| Retained K-bracketed route | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/basert-baseq-fused-k-route-final/` |

## 2026-07-23 BaseQN LM-Head Routing

Same Apple M5 Max environment and measurement method as the BaseQN sections
above. The optimization notebook records why both dedicated reductions were
removed in favor of columnwise direct GEMV composition.

| Run | Working tree | Backend / preset | Warmups / iterations | Raw results |
| --- | --- | --- | ---: | --- |
| Eight-simdgroup reduction control | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/basert-baseq-lm-head-strong-baseline/` |
| Serial-simdgroup reduction candidate | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/basert-baseq-lm-head-serial-candidate/` |
| Retained columnwise GEMV route | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/basert-baseq-lm-head-retained/` |
| Retained QuixiCore-argmax route | `bc968fc-dirty` | MLX / quick, Q4 | 20 / 50 | `perf/results/2026-07-23/basert-baseq-lm-head-retained-argmax-sampling/` |

## 2026-07-23 BaseQN Grouped Expert Baseline

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

## 2026-07-23 Extended RoPE Baseline

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

## 2026-07-23 Q8_0 KV Baseline

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

## 2026-07-23 Gated DeltaNet I/O Baseline

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

## 2026-07-23 BaseRT Completion Kernel Baselines

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

## 2026-07-24 BaseRT Vision and Audio Kernel Baselines

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

## Migration Tasks

- Promote stable benchmark runs into compact per-kernel baseline tables.
- Keep large profiler traces out of git; record trace paths and summaries only.
- Store normalized raw output under `perf/results/YYYY-MM-DD/<kernel>/<run-id>/`.
