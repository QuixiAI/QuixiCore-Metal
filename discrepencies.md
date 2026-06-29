# ThunderKittens → ThunderMittens Kernel Reconciliation

Accurate per-file status of every non-baseline ThunderKittens (CUDA) kernel under
`.reference/ThunderKittens/kernels` against the ThunderMittens (Apple Metal) port. Maintained checklist
(supersedes the original from-scratch inventory). Detailed parity tracking: `docs/porting/thunderkittens.md`.

**Porting rule (from `bigpicture.md`):** port the *algorithm* on the TM substrate, not the H100/B200
machinery (TMA, WGMMA, tcgen05, warpgroups, cp.async, NVLink). Most TK files are hardware/scheduling/
pedagogical *variants* of one algorithm, so they collapse onto a single Apple kernel.

## Headline counts (58 non-baseline TK kernels)
- **Algorithmically covered: 42** — attention fwd/bwd, the GEMM suite + 19 educational levels, fftconv,
  flux, norm/rotary, the sequence/SSM family, and the low-precision GEMMs (via the `qgemm` quant family).
- **N/A on single-GPU Apple: 16** — the `parallel/*` distributed collectives (no NVLink/NVSwitch/P2P).
- Net: every Apple-feasible TK algorithm is ported. Four genuine gaps found during reconciliation were
  closed (marked **NEW** below): Based Taylor map, decay/retention LA, flash-attn backward, fp8 scaled GEMM.

## Status legend
`ported→X` = covered by ThunderMittens kernel X · `variant→X` = hardware/scheduling/pedagogical variant
of an algorithm already ported as X · `N/A` = no single-GPU Apple analog.

## Attention (4)
| TK file | Status |
|---|---|
| `attention/mha_h100/mha_h100.cu` (fwd) | ported → `attn_fwd` / `attn_causal` / `attn_multiwarp` |
| `attention/mha_h100/mha_h100.cu` (bwd: `bwd_attend_prep`+`bwd_attend`) | **NEW** ported → `attn_bwd` (dQ/dK/dV, non-causal+causal) |
| `attention/mha_h100_lcf/mha_h100_lcf.cu` | variant → `attn_fwd` (load-compute-finish scheduling, fwd-only) |
| `attention/bf16_b300_mha_causal/…` | variant → `attn_causal` (Blackwell causal fwd) |
| `attention/bf16_b300_mha_noncausal/…` | variant → `attn_fwd` (Blackwell non-causal fwd) |

## Sequence / linear attention / SSM (4)
| TK file | Status |
|---|---|
| `linear_attention/linear_attention.cu` | **NEW** ported → `lin_attn_decay` (RetNet/Lightning-Attn-2, per-head exp decay) |
| `based/linear_attn.cu` | **NEW** ported → `based` (2nd-order Taylor feature map φ=[1,x,x²/√2]) |
| `hedgehog/hedgehog.cu` | ported → `hedgehog` (φ(x)=exp(x−rowmax)); plus `linear_attn` (identity), `lin_attn_causal` |
| `mamba2/mamba2.cu` | ported → `mamba2` (selective SSD, decay-tile materialized form) |

## GEMM (9) + educational (19)
| TK file | Status |
|---|---|
| `gemm/bf16_h100`, `gemm/bf16_b200` | ported → `matmul_custom` (+ arbitrary shapes) / `gemm_staged` |
| `gemm/educational_h100/{launch,level_01..08}` (9) | variant → one matmul, WGMMA/TMA optimization ladder (no Apple analog) → `matmul_custom`/`gemm_staged` |
| `gemm/educational_b200/{launch,level_01..09}` (10) | variant → one matmul, tcgen05/cluster optimization ladder → `matmul_custom`/`gemm_staged` |
| `gemm/fp8_h100`, `gemm/fp8_b200` | ported → `qgemm` format `fp8_e4m3` |
| `gemm/fp8_h100_scaled` | **NEW** ported → `qgemm_fp8_scaled` (per-token × per-channel rank-1 fp8 scaling) |
| `gemm/int8_h100`, `gemm/int8_b200` | ported → `qgemm_w8a8` (int8×int8→int32) |
| `gemm/mxfp8_b200` | ported → `qgemm` format `mxfp8` (1×32 e8m0 block scale) |
| `gemm/nvfp4_b200` | ported → `qgemm` format `nvfp4` (1×16 e4m3 block + per-tensor scale) |

## Conv / norm / rotary / fusion (6)
| TK file | Status |
|---|---|
| `fftconv/fftconv_non_pc.cu` | ported → `fftconv` (Monarch FFT convolution + complex-multiply MMA) |
| `fftconv/fftconv_pc.cu` | variant → `fftconv` (persistent producer-consumer scheduling) |
| `flux/flux_gelu.cu`, `flux/flux_gate.cu` | ported → `flux_gelu` / `flux_gate` |
| `layernorm/layernorm.cu` | ported → `layernorm` (+ `rms_norm`, `softmax`, `gelu` are TK-inline ops) |
| `rotary/rotary.cu` | ported → `rotary` (split-half RoPE) |

## Parallel / distributed (16) — N/A on single-GPU Apple
`parallel/{all_reduce, all_reduce_educational, reduce_scatter, all_gather, all_to_all, ring_attn,
ulysses_attn, ag_gemm(+b200,+fp8), gemm_rs(+b200,+fp8), gemm_ar(+lcsc), moe_dispatch_gemm}` — all need
NVLink/NVSwitch `multimem` multicast or GPU-initiated P2P, which Apple Silicon (one GPU per SoC) has no
analog for. At N=1 each degenerates to a local kernel ThunderMittens already has (`matmul_custom`/
`attn_*`), so porting them adds nothing. The nearest-neighbor patterns (`ring_attn`, `all_to_all`,
`ulysses_attn`, `ag_gemm`, `gemm_rs`, `moe_dispatch_gemm`) are network-mappable only as a full
host-driven rewrite (MLX-distributed / MPI over Thunderbolt/Ethernet) — a separate future project, not a
kernel port. The NVLS-multicast families (`all_reduce`, `reduce_scatter`, `all_gather`, `gemm_ar`) have
no Apple equivalent at all.

## ThunderMittens-only kernels (no TK counterpart)
The full quant family beyond the TK GEMM subset — `qgemm`/`qgemv`/`qflux` over 29 weight formats
(llama.cpp k-quants/i-quants, BitNet, MX, fp8/fp4 variants), `qgemv_w8a8`/`qgemv_w2a8`/`qgemm_w2a8`
integer decode/prefill, `qgemm_blockscale` (fp8_block2d), `qgemm_actorder` (GPTQ in-kernel gather),
`attn_q` (quantized-KV attention) — plus `add_rt`, `cmplx_matmul`. See `docs/porting/marlin-quant.md`.
