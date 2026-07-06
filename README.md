# QuixiCore Metal

QuixiCore Metal is the Apple Metal implementation of the QuixiCore kernel library.

It is a standalone native implementation for Apple Silicon, with integrations for MLX and PyTorch MPS. It shares no implementation code with the other QuixiCore backends.

It implements the contract defined by [QuixiAI/QuixiCore](https://github.com/QuixiAI/QuixiCore): the same operation names, quant formats, correctness expectations, benchmark methodology, and public library identity as the other QuixiCore backends.

**Native implementations. Shared contract. No shared code.**

## QuixiCore Standard Files

- Contract metadata: [`.quixicore/backend.yaml`](.quixicore/backend.yaml)
- Kernel coverage manifest: [`.quixicore/kernels.yaml`](.quixicore/kernels.yaml)
- Quant format manifest: [`.quixicore/quant-formats.yaml`](.quixicore/quant-formats.yaml)
- Repository structure: [`docs/repository-structure.md`](docs/repository-structure.md)
- Contribution workflow: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security policy: [`SECURITY.md`](SECURITY.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)

Common developer entrypoints:

```bash
scripts/configure
scripts/build
scripts/test
scripts/bench
scripts/coverage-report
scripts/clean
```

These scripts keep the QuixiCore workflow consistent while wrapping Xcode,
Metal, MLX, and PyTorch MPS tooling.

<div align="center" >
    <img src="assets/mittens.jpeg" height=350 alt="QuixiCore Metal logo" style="margin-bottom:px"/> 
</div>

<br>
<br>

## Origin And Focus

This repository was renamed from ThunderMittens, [QuixiAI](https://github.com/QuixiAI)'s Apple Metal Shading Language (MSL) fork of
[ThunderKittens](https://github.com/HazyResearch/ThunderKittens/) — the tile-based GPU-kernel framework
from HazyResearch. This backend brings the substrate to Apple Silicon and extends it with a large set of
serving, training, and model-architecture kernels running on **two integrations** (MLX + PyTorch/MPS) from
one metallib.

## Performance highlights

Hand-tuned tile kernels that **beat Apple's own optimized primitives** — and, where the algorithm
allows, change the complexity class entirely. All numbers are measured median per-call latency on an
**Apple M4 Max** (40-core GPU, ~546 GB/s), MLX backend, `speedup = baseline_ms / tk_ms`, reproducible
via `perf/bench_kernels.py`. ~50 kernel families ship on **two backends** (MLX + PyTorch/MPS) from one
metallib, cross-checked by 2,277 correctness + parity + MPS tests.

### Faster than the framework's own tuned kernels

| kernel | shape | vs | speedup |
|---|---|---|--:|
| **cross-entropy** (fused-linear) | T2048 × V128256 | framework CE composition | **13.6×** |
| **layernorm** | 16384 × 768 | `mx.fast.layer_norm` | **3.1×** |
| **rms_norm** | 16384 × 768 | `mx.fast.rms_norm` | **3.1×** |
| **gelu** (tanh) | 16384 × 256 | `mx.nn.gelu_approx` | **3.1×** |
| **softmax** | 16384 × 768 | `mx.softmax` | **2.9×** |
| **causal attention** | 1×8×2048×128 | `scaled_dot_product_attention` | **3.9×** |
| **qk_norm_rope** (fused per-head) | 4096 × 32 × 128 | `mx.fast` rms_norm + rope | **2.5×** |
| **fused add + rms_norm** | 4096 × 1024 | `add` + `mx.fast.rms_norm` | **1.9×** |
| **rms_norm backward** (fused 1-pass) | 65536 × 512 | `mx.fast.rms_norm` VJP | **1.5×** |
| **layernorm backward** (fused 1-pass) | 65536 × 512 | `mx.fast.layer_norm` VJP | **1.6×** |
| **matmul** | 4096×4096×32 (thin-K) | `mx.matmul` | **1.8×** |

### Algorithmic & serving wins

| kernel | shape | vs | speedup |
|---|---|---|--:|
| **linear attention** (chunked, causal) | 2×8×4096×64 | masked O(N²) reference | **6.4×** |
| **Mamba-2 / SSD backward** (linear-time chunked) | 1×8×8192×64 | O(N²) quadratic backward | **15.9×** |
| **MLA decode** (paged, partitioned) | 8×16, ctx 8192 | single-partition v1 | **6.3×** |
| **attention backward** (FlashAttention-2) | 1×8×2048×64 | naive VJP | **5.4×** |
| **beam-search advance** (single-pass top-M) | 16 beams × V128256 | framework top-k | **5.2×** |
| **paged attention v2** (partitioned decode) | 8×32, ctx 2048 | v1 decode | **4.6×** |
| **Mamba-2 / SSD forward** (chunked) | 2×16×4096×64 | O(N²) quadratic | **4.7×** |
| **Hadamard transform** | 16384 × 512 | matmul-with-H | **9.4×** |
| **weight-only quantized GEMV** | 32000 × 4096 (BitNet) | fp16 matmul | **3.8×** |
| **grouped MoE** (schedule + gather) | E8, H2048 | per-expert loop | **1.5×** |

Weight-only quant covers **~29 formats** (GGUF k-/i-quants, fp8, NVFP4, AWQ, GPTQ act-order, BitNet
ternary, …), and **every format now beats the fp16 GEMV baseline**. Sliding-window, ALiBi, and
block-sparse masks compose on the paged-decode path; the Mamba-2/linear-attention families auto-route
between an O(N²) kernel (small N) and a linear-time chunked pipeline (D∈{64,128}) at measured
crossovers.

### Serving & training kernels (Waves 5–8)

A full serving + training surface, all dual-backend (MLX + PyTorch-MPS from one metallib) and
cross-backend parity-tested:

- **Sampling / decoding** — top-k, top-p, min-p, **typical-p**, categorical, argmax (fused Gumbel-max,
  seed-reproducible on host); grammar **allow-bitmask** + **bad/stop-word** masking; repetition /
  presence / frequency penalties; fused quantized LM-head sampling (argmax / categorical / top-k /
  **exact top-p nucleus** — true full-vocab normalizer, no `(T,V)` logits materialization).
- **Speculative decoding** — linear (vLLM) **and dynamic-tree** rejection verification (exact for any
  sibling count, first-generation `tree_valid` fallback), a device-resident **tree-pointer builder**,
  plus `spec_compact` / `spec_update_kv_meta` post-processing (no host readback between verify and KV append).
- **Beam search** — single-pass advance, on-device KV reorder (copy-path **and** zero-copy
  block-table remap), device copy-pair builder.
- **Attention** — cascade / shared-prefix decode, **N-level** (multi shared prefix) with an optional
  **fp8 prefix**, over the paged-v2 partition/reduce; **fully device-resident varlen prefill** (device
  Q pad/gather + output re-gather, no host pad/transpose loop).
- **Multimodal / embeddings** — token embedding lookup **+ backward** (atomic scatter-add *and* an
  atomic-free sorted-segment variant), on-device multimodal `src` builder + span merge.
- **Training** — RMSNorm / LayerNorm / GELU **backward** (single-pass fused, rstd/mean computed
  in-kernel), fused-add-RMSNorm backward, GLU-family backward (all 6 modes), **dropout** (mask-free,
  seed-reproducible), and a fused **AdamW** step — all reachable through **first-order autograd** on
  both backends (MLX `mx.custom_function`, PyTorch `autograd.Function`).

Measured perf wins (MLX, M4 Max), across the optimization passes — see `perf/optimization_status.md`
for the full sweep + the honest rejects:

- **Fused single-pass norm backward** — one kernel computes rstd/mean in-kernel, writes dX, and
  accumulates dweight/dbias via `atomic_add_float`: **~2.3–2.5× over the old 3-pass hybrid**, on par
  with (up to ~1.6× over) `mx.fast`'s fused VJP.
- **gelu backward** vec4 **2.1–3.5×**, **dropout** fwd/bwd vec4 **~1.9×**, **typical-p sampling
  1.8×** (trimmed the surprise-bisection's redundant full-vocab re-scans).
- **embedding_lookup 8.0×** and **merge_multimodal_spans 11.9×** (threadgroup-per-token + vec4;
  embedding is now 2.3× *faster* than the framework gather), **beam KV reorder ~1.45×**.

The passes are honest about rejects: fused-write cascade, copy-pair compaction, and `adamw` vec4 were
all built or prototyped, measured to not win, and reverted/documented.

### New model architectures (Wave 9) — ported from AlpinDale's metal-forge

Wave 9 is a gap-closing wave that ports the kernels modern open models need to actually run. The
algorithms come from **[metal-forge](https://github.com/AlpinDale)**, a vLLM-style Metal
inference-kernel library by **[AlpinDale](https://github.com/AlpinDale)**
([@AlpinDale](https://x.com/AlpinDale)) — **huge thanks** for the reference implementations. Each
kernel below is re-expressed on the ThunderMittens tile substrate (not copied), then run through the
per-kernel loop: naive-correct → validated against a numpy/HF oracle → cross-backend parity-tested →
benched → optimized keep-if-win. Details in `perf/optimization_status.md`.

Every kernel here is credited to AlpinDale / metal-forge as the source of the algorithm:

| model target | ThunderMittens kernel | ported from metal-forge | measured |
|---|---|---|---|
| **gpt-oss** | quantized grouped expert GEMMs (`moe_grouped_gemm_rect_q` / `swiglu_q`, dequant-in-register + `mma_ABt`, MXFP4/NVFP4/fp8/int + swiglu_oai + expert bias) | `moe/moe.metal` | ~dense-bf16 speed at prefill, **reads 4–8× fewer weight bytes**; wins at decode |
| **gpt-oss / Gemma-2/3** | attention **sinks + logit softcap** (fwd / causal / window / varlen / paged) | `attention.metal` | flagless path regression-free |
| **gpt-oss** | fused **act→quant epilogues** (`silu_mul_quant_fp8`/`int8`/`fp8_group`) | act-quant fusion | **~1.4×** the unfused swiglu→quantize chain |
| **DeepSeek-V3 / Kimi-K2** | grouped **`noaux_tc` routing** (softmax/sigmoid/sqrt-softplus, two-level group top-k) | `moe/moe.metal` | ids-exact vs HF; within noise of plain top-k |
| **Qwen3-Next / Kimi-Linear** | **GatedDeltaNet linear attention** (`gdn_recur`, varlen + paged state pool + GQA) | `gdn_linear_attention` | novel mixer (no MLX baseline) |
| **Qwen3** | fused per-head **QK-RMSNorm + RoPE** (`qk_norm_rope`, NeoX + GPT-J) | `qk_norm_rope` | **2.5×** the `mx.fast` rms_norm+rope composition |
| **Mamba-1 hybrids** | **selective scan** (S6) dense / varlen / **APC** paged-state checkpointing | `sequence/selective_scan.metal` | reference-faithful recurrence |
| **W8A8 serving** | per-group-128 + asymmetric-int8 (**azp**) activation quant + azp-corrected W8A8 GEMM | `quantization` | codes bit-exact vs numpy |
| **sub-4-bit KV** | **TurboQuant KV codec** (K asymmetric-uniform + V random-sign-FWHT / Lloyd-Max, 2–8 bit) | `quantization/turboquant.metal`, `hadamard.metal` | K codes bit-exact vs fp16 oracle |
| **long-context** | **MInference** decode block-mask builder + per-head block-sparse paged attention | `attention.metal` | exact-int oracle |
| **sampling** | the modern **sampler zoo** (top-nσ / top-A / ε- / η-cutoff / quadratic / skew / XTC / no-repeat-ngram / DRY / top-k & top-p renorm) | `sampling/sample_top_p.metal` | bandwidth-bound, one simdgroup/row |
| **layout / utils** | `tau_tail`, `packbits` / `segment_packbits`, `permute_cols` | `cache/tau.metal`, `sampling/bitpack.metal`, `layout/layout.metal` | exact vs `np.packbits` / gather |

The Wave-9 optimization pass then vectorized the two bf16-bound kernels that measured a win —
`gdn_recur` (vec4 k/q loads: prefill ~7%, decode ~5%) and `act_quant` (vec4 amax+encode:
int8 4096×2880 **~27%**) — and honestly reverted the f32 sampler-zoo vec4, which regressed at scale
(f32 strided loads are already coalesced). See `perf/optimization_status.md`.

## Prerequisites (all paths)

- Apple Silicon Mac.
- **Xcode with the Metal Toolchain installed.** Recent Xcode ships the Metal compiler as a
  separate component — `xcrun --find metal` resolving a path is not enough. Install it once with:
  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```
  Without this, both the MLX build and the PyTorch metallib build fail with
  `cannot execute tool 'metal' due to missing Metal Toolchain`.

## Project Structure

The same framework-agnostic `.metal` kernels (under `ThunderMittens/include` + `ThunderMittens/kernels`)
power three use cases:

### 1. MSL Kernel Development

For writing Metal Shading Language (MSL) kernels:
- Open the project in Xcode
- Xcode will handle all build processes
- Primitive unit tests live in `ThunderMittens/tests/unit` (gated by `ENABLE_TESTS` in
  `tests/unit/testing_commons/testing_flags.hpp`); build/run the `ThunderMittens` scheme.

### 2. MLX Kernel Integration with Python

For using QuixiCore Metal kernels within MLX in Python:

#### Prerequisites
- Python 3.8+
- CMake
- Xcode Command Line Tools

#### Installation Steps

1. Navigate to ThunderMittens/mlx directory
2. Install MLX with parallel build:
   ```bash
   CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e ".[dev]"
   ```

3. Navigate to ThunderMittens/kernels directory
4. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

5. Build kernels and bindings:
   ```bash
   python setup.py build_ext -j8 --inplace
   ```

> Build artifacts are written to a repo-root `build/` dir (via `kernels/setup.cfg`) so they stay
> out of the Xcode-synchronized source tree. Validate with `python -m pytest */correctness/`.

### 3. PyTorch (Apple MPS) Integration with Python

For using QuixiCore Metal kernels from PyTorch on the `mps` device. This path is **independent of
MLX** — it needs only PyTorch and the Metal toolchain (no CMake/nanobind, no MLX build). The same
`.metal` kernels are compiled into a standalone metallib with `xcrun metal` and dispatched onto
PyTorch's MPS stream.

#### Prerequisites
- Python 3.9+
- PyTorch with MPS (`torch>=2.1`; nightlies work). MPS custom-kernel support uses
  `torch::mps::get_command_buffer()`.
- The Metal toolchain (see [Prerequisites](#prerequisites-all-paths)).

#### Installation Steps

1. Navigate to the PyTorch backend and install it (pulls in PyTorch, leaves an existing install
   such as a nightly untouched):
   ```bash
   pip install -e ThunderMittens/kernels/tk_torch
   ```
2. Use it — the metallib and the ObjC++ extension build automatically on first import:
   ```python
   import torch
   import tk_torch

   x = torch.randn(2, 128, 1024, dtype=torch.bfloat16, device="mps")
   w = torch.randn(1024, dtype=torch.bfloat16, device="mps")
   b = torch.randn(1024, dtype=torch.bfloat16, device="mps")
   y = tk_torch.layernorm(x, w, b)   # matches torch.nn.functional.layer_norm
   ```

## Testing

Run from `ThunderMittens/kernels/`:

```bash
# MLX correctness (each kernel vs an MLX oracle, e.g. mx.fast.layer_norm)
python -m pytest */correctness/

# PyTorch MPS correctness (each kernel vs a torch reference, e.g. F.layer_norm)
python -m pytest tk_torch/tests/

# Cross-backend parity: the SAME metallib kernel on MLX vs MPS must agree for
# identical inputs (catches host-ABI drift between the two backends)
python -m pytest tests_parity/

# ...or everything at once (each suite skips cleanly if its framework is absent)
python -m pytest */correctness/ tk_torch/tests/ tests_parity/
```

Primitive-level MSL unit tests (register/shared tile ops) build and run through Xcode — see
[1. MSL Kernel Development](#1-msl-kernel-development).

## References & credits

QuixiCore Metal was originally ThunderMittens, a [QuixiAI](https://github.com/QuixiAI) fork of
[ThunderKittens](https://github.com/HazyResearch/ThunderKittens/) by **HazyResearch** — the original,
upstream project. See HazyResearch's [blog post](https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx)
and [paper](https://arxiv.org/abs/2410.20399) to learn more about the ThunderKittens framework this backend
builds on.

The Wave-9 model-architecture kernels (quantized grouped expert GEMMs, attention sinks + softcap,
DeepSeek `noaux_tc` routing, GatedDeltaNet linear attention, `qk_norm_rope`, Mamba-1 selective scan,
azp/per-group activation quant, the TurboQuant KV codec, MInference block masks, the sampler zoo, and
the `tau_tail` / `packbits` / `permute_cols` utilities) are ported from
**[metal-forge](https://github.com/AlpinDale)** by **[AlpinDale](https://github.com/AlpinDale)**
([@AlpinDale](https://x.com/AlpinDale)) — thank you for the reference implementations that made these
kernels possible.
