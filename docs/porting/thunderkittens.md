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

**Quantized GEMM/GEMV (Marlin's method) — IN PROGRESS (was wrongly parked as N/A):**
- The dequant-in-register approach makes the whole quantized family feasible on Apple — dequant
  packed weights → `half` → standard `simdgroup_matrix` MMA (GEMM) or simd-reduction (GEMV). See
  `marlin-quant.md` for the plan; references: Marlin `dequant.h`, vLLM-Metal, llama.cpp `kernel_mul_mm`.
- ✅ Done: `kernels/qgemm/` (dequant-to-shared → MMA, prefill/batched) and `kernels/qgemv/` (batch-1
  decode, simd-reduction); dequant primitive in `include/.../tile/dequant.metal` (MMA `BK=32`
  decoupled from `block_k`). `tk.qgemm` auto-routes M==1 → `qgemv`. **All 9 formats**, dual-backend,
  validated vs `dequantize(Wq)@x`: integer `q8_0, q4_0, q4_K, kU4B8, kU4`; float `fp8_e4m3, fp4_e2m1,
  mxfp8, nvfp4`.
- ✅ Activation quant (W·A8 parity): `tk.qmm(wq, x, w_format, act="int8"|"fp8")` → fp8 W8A8, int8
  W8A8, int8 W4A8 (snap activations to the grid + dequant-to-half GEMM; parity, not speed).
- ✅ Quantized fused kernel: `kernels/qflux/` `qflux_gelu` = gelu(dequant(Wq)@X + bias), all formats.
- ✅ **Dequant-direct-to-fragment** (Marlin zero-shuffle) — now the default `qgemm` path
  (`dequant_into_register`); bit-identical to staged, **~40% faster**. The one multi-simdgroup
  optimization that wins on Apple (quantized GEMM is weight-bandwidth-bound).
- ☐ Optional: same zero-shuffle for `qflux`; attention quantized-KV.

**Not applicable on Apple:**
- `parallel/*` (`ag_gemm`, `all_reduce`, `all_gather`, `ring_attn`, `ulysses_attn`, `gemm_rs`, …) —
  multi-GPU collectives; a single Apple GPU has no NVLink/multi-device fabric. N/A for this target.
- `gemm/baselines/*` (cuBLAS reference impls) — reference baselines, not TK kernels.

See `discrepencies.md` for the raw file listing.
