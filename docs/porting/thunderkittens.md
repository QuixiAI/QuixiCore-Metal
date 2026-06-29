# ThunderKittens → ThunderMittens Parity Checklist

Tracks the port of each ThunderKittens (CUDA) kernel to ThunderMittens (Apple Metal).
Source inventory: `discrepencies.md`. Strategy: `bigpicture.md`. Substrate gaps: `primitives.md`.

**Status legend:** ☐ not started · ◐ compiling · ✅ correct (validated vs oracle) · 🏎️ benchmarked · 🚫 blocked on a primitive.

**Porting rule:** port the *algorithm* on the TM substrate, not the H100 machinery (TMA/WGMMA/warpgroups).
Drop async double-buffering for v1. Validate every kernel against an MLX/NumPy oracle.

## Done / in repo

| Kernel | Status | Dtype / shape | Oracle | Notes |
|---|---|---|---|---|
| `add_rt` | ✅ | f32/f16/bf16, 8×8-multiple | `x + y` | Elementwise-add smoke test (was a broken stub). `kernels/add_rt/` |
| `matmul_custom` | ✅ | f32/bf16, N%32,M%32,K%16 | `x @ y` | Naive blocked GEMM, fixed `<4,2,4>` tiling. Generalizing shapes is future work. `kernels/matmul_custom/` |
| `attn_fwd` | ✅ | bf16, D∈{64,128}, non-causal | `mx.fast.scaled_dot_product_attention` (scale=1/√D) | Warp-level flash-attn forward. `kernels/attn_fwd/` |
| `layernorm` | ✅ | bf16, D∈{256,512,768,1024} | `mx.fast.layer_norm` | Worked-example port. fp32 compute, inline `metal::rsqrt`. `kernels/layernorm/` |
| `rms_norm` | ✅ | bf16, D∈{256,512,768,1024} | `mx.fast.rms_norm` | layernorm minus mean/bias. `kernels/rms_norm/` |
| `softmax` | ✅ | bf16, D∈{256,512,768,1024} | `mx.softmax` | Standalone row-softmax (attn_fwd's inline softmax extracted). `kernels/softmax/` |
| `rotary` | ✅ | bf16, D∈{64,128} | `mx.fast.rope(traditional=False)` | Split-half RoPE; precomputed cos/sin inputs. `kernels/rotary/` |
| `gelu` | ✅ | bf16, D∈{256,512,768,1024} | `mx.nn.gelu_approx` | Tanh-approx GELU activation. Added `tanh` base_op (via `exp`). `kernels/gelu/` |
| `matmul_custom` (arbitrary shapes) | `gemm/bf16_h100` | ✅ | `mx.matmul` | Any N/K/M via host zero-pad-to-tile + slice (`tk.matmul_custom`). f32/bf16. |
| `attn_causal` | `attention/mha_h100` | ✅ | masked SDPA (additive causal) | Causal flash-attn fwd; `make_causal` on the diagonal block. `kernels/attn_causal/` |
| `flux_gelu` / `flux_gate` | `flux/flux_gelu.cu`, `flux_gate.cu` | ✅ | gelu(x@w+b) / (x@w+b)*g+r | Fused GEMM epilogue (register `add_col`/`mul_col`/`gelu` + tile add). `kernels/flux/` |
| `gemm_staged` | `gemm/bf16_h100` | ✅ 🏎️ | `mx.matmul` | Multi-simdgroup, threadgroup-staged GEMM (2 warps share the A block via shared mem). Competitive with `matmul_custom` and `mx.matmul`. `kernels/gemm_staged/` |
| `attn_multiwarp` | `attention/mha_h100` | ✅ 🏎️ | SDPA (scale 1/√D) | Multi-warp flash-attn fwd (4 simdgroups share each K/V block via shared mem). Correct; not yet faster than `attn_fwd` at tested shapes (staging overhead) — perf tuning is future work. `kernels/attn_multiwarp/` |
| `linear_attn` | `linear_attention`, `based/linear_attn` | ✅ | `Q @ (Kᵀ @ V)` | Non-causal linear attention (identity feature map), D=64; `mma_AtB` then `mma_AB` with D×D register state. `kernels/linear_attn/` |
| `hedgehog` | `hedgehog` | ✅ | `phi(Q)@(phi(K)ᵀ@V)` | Feature-map linear attention, φ(x)=exp(x−rowmax(x)) (col-layout feature map), D=64. `kernels/hedgehog/` |
| `lin_attn_causal` | `based/linear_attn` | ✅ | `tril(Q@Kᵀ)@V` | Causal linear attention via chunked running-KV scan + intra-chunk `make_causal`, D=64. `kernels/lin_attn_causal/` |
| `mamba2` | `mamba2` | ✅ | `((C@Bᵀ)⊙exp(Δcumlog)⊙tril)@X` | Selective SSD forward (materialized chunked form); decay tile via `add_row`/`sub_col`/`exp` from a host-precomputed `cumlog=cumsum(log a)`, D=64. `kernels/mamba2/` |
| `cmplx_matmul` | `fftconv` (building block) | ✅ | complex `A@B` | Complex GEMM exercising the **complex-multiply MMA** (`complex_mma_AB`); operands carry a leading size-2 (real,imag) axis. f32/bf16. `kernels/cmplx_matmul/` |
| `fftconv` | `fftconv` | ✅ | `torch.fft` circular conv (exact) | Monarch FFT convolution, N=S² (S∈{16,32}); complex matmuls (`complex_mm_AB`) + transposes + pointwise complex mul. **rel=0.00000 vs torch.fft.** `kernels/fftconv/` |

All kernels ship on **both** backends (MLX + PyTorch MPS) via `tk_launch.h`. Run all:
`cd ThunderMittens/kernels && python -m pytest */correctness/ tk_torch/tests/ tests_parity/ -q`
(210 passing). Primitive unit tests: Xcode `ThunderMittens` scheme (126 passing).
Benchmark the perf kernels: `python time_perf.py`.

**Complex-multiply MMA** (`include/ops/warp/register/tile/mma.metal`): `complex_mma_AB`/`_ABt`/`_AtB`/
`_AtBt` + `complex_mm_AB` operate on the `crt` complex tiles as four real MMAs on the `.real`/`.imag`
components (`Dr = Ar·Br − Ai·Bi`, `Di = Ar·Bi + Ai·Br`; the `−Ai·Bi` is folded by negating `Ai` once).
Validated by `cmplx_matmul` and used to build `fftconv` — **every algorithmically-distinct,
Apple-feasible TK kernel is now ported.**

## Completion map — the full 58-file TK inventory on Apple

"Absolute completion" on Apple means covering every *algorithmically-distinct, Apple-feasible* kernel
and honestly accounting for the rest. The 58 TK files break down as:

**Done (algorithms ported, dual-backend, validated)** — covers the bulk of the inventory because most
TK files are hardware-specific *variants* of one algorithm:
- Attention: `attention/{mha_h100, mha_h100_lcf, bf16_b300_mha_causal, bf16_b300_mha_noncausal}` are
  all flash-attention forward → ported as `attn_fwd` (non-causal), `attn_causal`, `attn_multiwarp`.
- GEMM: `gemm/{bf16_h100, bf16_b200}` (+ the `educational_h100`/`educational_b200` level_01..09 = GEMM
  tutorials) → `matmul_custom` (+ arbitrary shapes) and `gemm_staged`.
- Norm/rotary/activation/fusion: `layernorm`, `rotary`, `flux/{flux_gelu,flux_gate}` → ported; plus
  `rms_norm`, `softmax`, `gelu` (TK has these inline/fused).
- Sequence / state-space (the whole family): `linear_attention` / `based/linear_attn` / `hedgehog` /
  `mamba2` → ported as `linear_attn` (non-causal), `lin_attn_causal` (causal scan), `hedgehog`
  (feature-map), and `mamba2` (selective SSD with the decay-tile). A Taylor feature map for `based` is
  a small variant of `hedgehog`/`linear_attn`.
- FFT / complex: `fftconv` → ported as the Monarch FFT convolution (`kernels/fftconv/`) on the new
  complex-multiply MMA. Validated exact (rel=0.00000) vs `torch.fft`.

**Perf tuning (investigated — finding below):**
All distinct algorithmic kernels are ported. The multi-simdgroup shared-staging kernels
(`gemm_staged`, `attn_multiwarp`) are correct and *competitive* but do **not** beat the
single-simdgroup kernels on Apple GPUs, and tuning confirmed this is structural:
- A bigger 4-simdgroup `BM=128` GEMM tile benchmarked **−20…26%** at 1024/2048 (occupancy) — reverted.
- 2 vs 4 simdgroups for `attn_multiwarp` were equivalent (~5% behind `attn_fwd`).
- Root cause: Metal has no async global→shared copy (`cp.async`/TMA) to overlap staging with compute
  the way the H100 kernels do, and these shapes are compute/cache-bound — so reducing global traffic
  via sharing doesn't pay. The simpler single-simdgroup kernels are near-optimal; `matmul_custom`/
  `gemm_staged` sit within ~5% of `mx.matmul`. (Benchmark: `python time_perf.py`.)

**Done (was substrate-blocked):**
- `fftconv` — ✅ ported (`kernels/fftconv/`). The former blocker (complex-multiply MMA) is implemented
  (`complex_mma_*` in `mma.metal`); the Monarch FFT-conv kernel is built on `crt` complex tiles +
  `complex_mma_*` + transposes + pointwise complex mul, and matches `torch.fft` exactly. **No
  distinct, Apple-feasible TK kernel remains.**

**Quantized GEMM/GEMV (Marlin's method) — COMPLETE (was wrongly parked as N/A):**
- The dequant-in-register approach makes the whole quantized family feasible on Apple — dequant
  packed weights → `half` → standard `simdgroup_matrix` MMA (GEMM) or simd-reduction (GEMV). See
  `marlin-quant.md` for the plan; references: Marlin `dequant.h`, vLLM-Metal, llama.cpp `kernel_mul_mm`.
- ✅ Done: `kernels/qgemm/` (prefill/batched) + `kernels/qgemv/` (batch-1 decode) + `kernels/qflux/`
  `qflux_gelu` (fused gelu(dequant(Wq)@X+bias)); dequant primitive in `include/.../tile/dequant.metal`
  (MMA `BK=32` decoupled from `block_k`). `tk.qgemm` auto-routes M==1 → `qgemv`. **29 weight formats**,
  dual-backend, validated vs `dequantize(Wq)@x` — integer `q8_0/q4_0/q4_1/q5_0/q5_1/q4_K/q5_K/q6_K/
  q2_K/q3_K/kU4B8/kU4/hqq`, codebook+lattice `iq4_nl/iq4_xs/iq2_xxs/iq2_xs/iq3_xxs/iq1_s`, float
  `fp8_e4m3/e5m2/fp8_block/fp4_e2m1/mxfp8/mxfp4/mxfp6_e3m2/mxfp6_e2m3/nvfp4`, ternary `bitnet`.
- ✅ Activation quant: W·A8 parity via `tk.qmm(wq, x, w_format, act="int8"|"fp8")` (dequant-to-half),
  AND a true integer-dot decode path (`idot4` primitive → `qgemv_w8a8` SmoothQuant / `qgemv_w2a8`
  BitNet; int8×int8→int32, validated vs the integer oracle, faster than dequant-to-half on large shapes).
- ✅ **Dequant-direct-to-fragment** (Marlin zero-shuffle) — the default `qgemm` AND `qflux_gelu` path
  (`dequant_into_register`); bit-identical to staged, **~40% faster**. The one multi-simdgroup
  optimization that wins on Apple (quantized GEMM is weight-bandwidth-bound).
- ✅ Layout/indexing: GPTQ act-order (`tk.qgemm_actorder`, g_idx reorder layer), HQQ (int4+zp g64).

**Quant — deferrals & flags (deliberate; correctness is complete, these are scope/quality choices):**
- ☐ **Quantized-KV attention** — not built. The dequant primitive would drop into the attention K/V
  `load`s exactly as it did for `qgemm`/`qflux_gelu`; only optional future work remaining in this family.
- ⚠️ **Integer *prefill* is decode-only by design.** The exact-int32 `idot` path is `qgemv_w8a8`/
  `_w2a8` (batch-1) only — Apple has no int8 `simdgroup_matrix`, so a tiled int prefill would lose to
  the half tensor cores. W8A8/W2A8 *prefill* (M>1) parity is via `tk.qmm(act="int8")`, which is
  dequant-both-to-half (fp16 accumulate) — NOT bit-exact int32. No fused (`qflux`) integer variant.
- ⚠️ **Host encoders are naive (kernel decoders are faithful).** `quantize_*` for the k-quants/i-quants
  use nearest-grid / simple scale selection, not ggml's optimized encoders, so round-trip-vs-original
  error is poor for some (q3_K ~20%, mxfp6 ~38% — the latter the same e8m0-floor artifact as mxfp4/8).
  The kernel *decoders* match the GGUF byte layouts (real GGUF weights decode correctly); kernel-vs-
  oracle testing isolates this. Production-grade encoders deferred.
- ⚠️ **`fp8_block`** stores the 128×128 tile scale **replicated per row** (correct block scale, but
  storage-redundant), and falls back to smaller row-tiles when N % 128 ≠ 0. Storage-optimal 2D-scale
  layout deferred.
- ⚠️ **GPTQ act-order** is a host-side activation gather (`tk.qgemm_actorder`), not a fused in-kernel
  `g_idx` gather (this is the "reordering layer at load" that was specified).
- ℹ️ `nvfp8` intentionally not added (not a real format); `fp8_e4m3` + `mxfp8` cover 8-bit float.

**Not applicable on Apple:**
- `parallel/*` (`ag_gemm`, `all_reduce`, `all_gather`, `ring_attn`, `ulysses_attn`, `gemm_rs`, …) —
  multi-GPU collectives; a single Apple GPU has no NVLink/multi-device fabric. N/A for this target.
- `gemm/baselines/*` (cuBLAS reference impls) — reference baselines, not TK kernels.
- **Correction — NOT N/A:** the low-precision GEMM family (`gemm/{fp8_*, int8_*, int4, mxfp8_*,
  nvfp4_*}`) was *previously* parked here as "no Apple low-precision tensor cores." That was wrong —
  Marlin's dequant-to-half-then-standard-MMA makes all of it feasible, and it is now fully implemented
  (29 formats, above). The only genuinely-N/A items are the multi-GPU collectives and reference baselines.

See `discrepencies.md` for the raw file listing.
