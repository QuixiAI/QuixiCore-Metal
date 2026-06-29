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
| `lin_attn_decay` | `linear_attention` | ✅ | `((QKᵀ⊙Λ)@V)`, Λ=exp(−slope·(i−j)) | RetNet/Lightning-Attn-2: causal LA with per-head exp decay; mamba2 decay-tile mechanic on a −slope·pos ramp, D=64. `kernels/lin_attn_decay/` |
| `based` | `based/linear_attn` | ✅ | `((1+x+x²/2)⊙tril)@V`, x=QKᵀ/√Dqk | Based 2nd-order Taylor feature-map LA; materialized form, D_QK=16/D_VO=64. `kernels/based/` |
| `attn_fwd_l` / `attn_bwd` | `attention/mha_h100` (bwd) | ✅ | PyTorch autograd on SDPA (dQ/dK/dV) | FlashAttention-2 backward; forward emits log2-logsumexp L, then prep/dQ/dKV (one simdgroup/block, `swap_layout`→`mma_AtB`), non-causal+causal, D∈{64,128}. `kernels/attn_bwd/` |
| `qgemm_fp8_scaled` | `gemm/fp8_h100_scaled` | ✅ | `(dequant·dequant)·w_scale·a_scale` | fp8-both rank-1 scaled GEMM (per-channel × per-token); the fp8 analog of W8A8. `kernels/qgemm/` |

All kernels ship on **both** backends (MLX + PyTorch MPS) via `tk_launch.h`. Run all:
`cd ThunderMittens/kernels && python -m pytest */correctness/ tk_torch/tests/ tests_parity/ -q`
(210 passing). Primitive unit tests: Xcode `ThunderMittens` scheme (126 passing).
Benchmark the perf kernels: `python time_perf.py`.

**Complex-multiply MMA** (`include/ops/warp/register/tile/mma.metal`): `complex_mma_AB`/`_ABt`/`_AtB`/
`_AtBt` + `complex_mm_AB` operate on the `crt` complex tiles as four real MMAs on the `.real`/`.imag`
components (`Dr = Ar·Br − Ai·Bi`, `Di = Ar·Bi + Ai·Br`; the `−Ai·Bi` is folded by negating `Ai` once).
Validated by `cmplx_matmul` and used to build `fftconv`.

## Full-parity close-out — four genuine gaps closed

A reconciliation pass against the full 58-file TK inventory (see the refreshed `discrepencies.md`) found
four algorithms the earlier "everything distinct is ported" claim had missed (the rest are confirmed
hardware/scheduling/pedagogical variants or N/A multi-GPU). All four are now ported, dual-backend, validated:

| Kernel | Reference | Oracle |
|---|---|---|
| `qgemm_fp8_scaled` | `gemm/fp8_h100_scaled` | both operands fp8, rank-1 per-token×per-channel scaling vs `dequant·dequant·scales` |
| `lin_attn_decay` | `linear_attention/linear_attention` | RetNet/Lightning-Attn-2 per-head exp decay vs `((QKᵀ⊙Λ)@V)` |
| `based` | `based/linear_attn` | 2nd-order Taylor map (φ=[1,x,x²/√2]) vs `((1+x+x²/2)⊙tril)@V` |
| `attn_bwd` (+ `attn_fwd_l`) | `attention/mha_h100` bwd | dQ/dK/dV vs PyTorch autograd on SDPA (non-causal+causal, D∈{64,128}) |

`attn_bwd` is the FlashAttention-2 backward (the distinct algorithm vs the forward): `attn_fwd_l` emits
the log2-domain logsumexp `L`, then `attn_bwd_prep`/`attn_bwd_dq`/`attn_bwd_dkv` (one simdgroup per
block, no atomics; `swap_layout` feeds the col-layout operand to `mma_AtB`). With these, **every
algorithmically-distinct, Apple-feasible TK kernel is ported** (897 Python + 126 Xcode tests).

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
  (feature-map), `mamba2` (selective SSD with the decay-tile), plus — closed in the parity pass —
  `based` (2nd-order Taylor feature map) and `lin_attn_decay` (RetNet/retention per-head exp decay).
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

**Quant — former deferrals, now all PAID DOWN:**
- ✅ **Quantized-KV attention** — `kernels/attn_q/` `attn_q<FMT,D[,CAUSAL]>` + `attn_q_mw` (multiwarp):
  K dequant→shared→col-load, V dequant→register; non-causal + causal + multiwarp, q8_0/q4_0/fp8_e4m3
  × D{64,128}, dual-backend (`tk.attn_q(q,kq,vq,format,causal,multiwarp)`, host `quantize_kv`).
- ✅ **Integer prefill (exact int32)** — `kernels/qgemm_int/` `qgemm_w8a8`/`qgemm_w2a8` (M>1, tiled
  int8 MAC + simd_sum on `idot4`). Validated vs the integer oracle. **Honest perf: ~7–10× slower than
  the dequant-to-half MMA** (Apple has no int8 matrix unit) — it buys bit-exact int32 numerics, not
  speed; `qmm(act="int8")` (dequant-to-half) stays the recommended prefill default.
- ✅ **Production-grade encoders** — `quantize_*` now port ggml's optimizers (`make_qkx2`/`make_qx`/
  `make_q3`, the iq4 scale sweep, lattice scale+parity refit, best-of-floor/ceil e8m0 for MX). Round-
  trip-vs-W is now ggml-grade (mxfp8 0.19→0.027, q2_K 0.33→0.30 [2-bit floor], iq2_xxs 0.71→0.34, …);
  decoders unchanged. Locked in by `tk/tests/test_encoders.py`.
- ✅ **fp8_block2d** — codes-only fp8 (`fp8_raw`) + a separate `(N/128,K/128)` tile scale buffer
  (`qgemm_blockscale` / `tk.qgemm_fp8_block2d`); removes the 128× per-row scale replication.
- ✅ **GPTQ act-order in-kernel g_idx** — `qgemm_actorder<FMT>` gathers the X K-rows by `perm` during
  the fragment fill (`tk.qgemm_actorder(..., fused=True)`); no materialized permuted-X copy.
- ℹ️ `nvfp8` intentionally not added (not a real format); `fp8_e4m3` + `mxfp8` cover 8-bit float.

**Not applicable on Apple:**
- `parallel/*` (16 kernels: `ag_gemm`(+b200,+fp8), `all_reduce`(+educational), `all_gather`,
  `all_to_all`, `reduce_scatter`, `ring_attn`, `ulysses_attn`, `gemm_rs`(+b200,+fp8), `gemm_ar`(+lcsc),
  `moe_dispatch_gemm`) — multi-GPU collectives. A single Apple GPU per SoC has no NVLink/NVSwitch
  `multimem` multicast or GPU-initiated P2P. At N=1 each degenerates to a local kernel we already have
  (`matmul_custom`/`attn_*`), so a "port" adds nothing. The nearest-neighbor patterns (`ring_attn`,
  `all_to_all`, `ulysses_attn`, `ag_gemm`, `gemm_rs`, `moe_dispatch_gemm`) are network-mappable only as
  a full host-driven rewrite (MLX-distributed/MPI over Thunderbolt) — a separate future project; the
  NVLS-multicast families (`all_reduce`, `reduce_scatter`, `all_gather`, `gemm_ar`) have no Apple analog.
- `gemm/baselines/*` (cuBLAS reference impls) — reference baselines, not TK kernels.
- **Correction — NOT N/A:** the low-precision GEMM family (`gemm/{fp8_*, int8_*, int4, mxfp8_*,
  nvfp4_*}`) was *previously* parked here as "no Apple low-precision tensor cores." That was wrong —
  Marlin's dequant-to-half-then-standard-MMA makes all of it feasible, and it is now fully implemented
  (29 formats, above). The only genuinely-N/A items are the multi-GPU collectives and reference baselines.

See `discrepencies.md` for the raw file listing.
