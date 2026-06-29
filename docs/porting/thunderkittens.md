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

Run all: `cd ThunderMittens/kernels && python -m pytest */correctness/ -v`.

## Next tier (difficulty order)

| Kernel | TK reference | Status | Oracle | Notes |
|---|---|---|---|---|
| rotary (RoPE) | `kernels/rotary/rotary.cu` | ☐ | `mx.fast.rope` | Mostly tile arithmetic; load sin/cos, rotate. |
| softmax | (within attention) | ☐ | `mx.softmax` | Row-softmax as a standalone kernel. |
| rms_norm | — (TK has fused LN) | ☐ | `mx.fast.rms_norm` | Trivial variant of layernorm; good `rsqrt`-vector follow-up. |
| flux gelu / gate | `kernels/flux/flux_gelu.cu`, `flux_gate.cu` | ☐ | NumPy GELU | Elementwise + gating. |
| GEMM bf16 parity | `kernels/gemm/bf16_h100/bf16_h100_gemm.cu` | ☐ | `mx.matmul` | Generalize `matmul_custom` to arbitrary shapes + shared-tile staging. |
| attention (causal / multi-warp) | `kernels/attention/mha_h100/` | ☐ | masked SDPA | Extend `attn_fwd`: causal mask, multi-simdgroup tiling. |

## Later (heavier / hardware-coupled)

- Sequence/state-space: `based/linear_attn`, `linear_attention`, `hedgehog`, `mamba2`, `fftconv`.
- Quantized GEMM: `fp8`, `int8`, `mxfp8`, `nvfp4` (Apple support varies; may need emulation).
- Distributed/parallel: `ag_gemm`, `all_reduce`, `ring_attn`, `ulysses_attn`, etc. (single-device first).

See `discrepencies.md` for the full 58-kernel inventory.
