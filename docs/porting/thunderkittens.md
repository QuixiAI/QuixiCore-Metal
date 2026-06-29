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

All kernels ship on **both** backends (MLX + PyTorch MPS) via `tk_launch.h`. Run all:
`cd ThunderMittens/kernels && python -m pytest */correctness/ tk_torch/tests/ tests_parity/ -q`
(119 passing). Primitive unit tests: Xcode `ThunderMittens` scheme (126 passing).

## Next tier (remaining work / deferred)

| Item | Status | Notes |
|---|---|---|
| flux **fused** gelu+matmul / gate+matmul+residual | ☐ | The GELU *activation* is done (`kernels/gelu/`); the fused flux kernels (matmul+gelu, matmul+gate+residual) still need the GEMM fusion. |
| GEMM **performance** (shared-tile staging, real general kernel) | ☐ | Arbitrary-shape *correctness* done via padding; a staged/tuned kernel (threadgroup tiles) is the perf follow-up. |
| Attention **multi-warp / long-context tiling** | ☐ | Causal+non-causal correctness done (warp-level, 8-row query tiles); multi-simdgroup tiling for throughput is the perf follow-up. |

## Later (heavier / hardware-coupled)

- Sequence/state-space: `based/linear_attn`, `linear_attention`, `hedgehog`, `mamba2`, `fftconv`.
- Quantized GEMM: `fp8`, `int8`, `mxfp8`, `nvfp4` (Apple support varies; may need emulation).
- Distributed/parallel: `ag_gemm`, `all_reduce`, `ring_attn`, `ulysses_attn`, etc. (single-device first).

See `discrepencies.md` for the full 58-kernel inventory.
