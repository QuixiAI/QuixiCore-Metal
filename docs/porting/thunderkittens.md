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

All kernels ship on **both** backends (MLX + PyTorch MPS) via `tk_launch.h`. Run all:
`cd ThunderMittens/kernels && python -m pytest */correctness/ tk_torch/tests/ tests_parity/ -q`
(184 passing). Primitive unit tests: Xcode `ThunderMittens` scheme (126 passing).
Benchmark the perf kernels: `python time_perf.py`.

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
- Sequence (entry point): `linear_attention` / `based/linear_attn` → `linear_attn` (non-causal,
  identity feature map).

**Portable — remaining distinct kernels (open work, feasible on Apple):**
- `mamba2` — selective SSD (chunked scan / segsum); large, distinct algorithm.
- Perf tuning of `gemm_staged` (double-buffered, larger tiles) and `attn_multiwarp`.
- (Done: `hedgehog` feature-map linear attention; `lin_attn_causal` causal linear attention.
  A Taylor feature map for `based` is a small variant of `hedgehog`/`linear_attn`.)

**Substrate-blocked:**
- `fftconv` — needs **complex MMA** wrappers (the `crt`/`crv` complex types exist but have no
  complex-multiply MMA yet; see `primitives.md`).

**Not applicable / emulation-only on Apple (documented, not "ported"):**
- `parallel/*` (`ag_gemm`, `all_reduce`, `all_gather`, `ring_attn`, `ulysses_attn`, `gemm_rs`, …) —
  multi-GPU collectives; a single Apple GPU has no NVLink/multi-device fabric. N/A for this target.
- Quantized tensor-core GEMM `gemm/{fp8_*, mxfp8_*, nvfp4_*}` — Apple `simdgroup_matrix` has no
  fp8/mxfp8/nvfp4 path; only **dequant-to-bf16 emulation** is possible (the MLX-style quantized
  matmul), which is a different value proposition than the TK tensor-core kernels. `int8` is the one
  plausible emulation port.
- `gemm/baselines/*` (cuBLAS reference impls) — reference baselines, not TK kernels.

See `discrepencies.md` for the raw file listing.
