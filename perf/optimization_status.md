# ThunderMittens — performance status

Running notebook for the per-kernel optimization loop described in `perf/perf.md`.
Numbers are throughput-style median per-call ms from `perf/bench_kernels.py`
(adaptive batched timing, ≥2 ms per timed sample), Apple M4 Max 40-core
(~546 GB/s DRAM), MLX backend unless noted. Baseline run:
`perf/results/2026-07-01/172040-mlx-quick/` at `d902519`.

**Timing-methodology note (2026-07-01):** the harness was rewritten. Earlier
numbers in this file used one submit+sync per call, which adds a ~0.15–0.25 ms
latency floor and swamped small kernels; conclusions drawn only from per-call-sync
timing (notably "staged paged attention is slower") did NOT survive the fix —
see the serving section.

## Baseline classification (2026-07-01, quick preset)

Speedup = best-baseline ms / tk ms (>1 means tk wins).

### Already ahead of the framework — protect, don't churn
| kernel | shape | tk ms | vs | speedup |
|---|---|---:|---|---:|
| layernorm | 4096×1024 | 0.066 | mx.fast.layer_norm | 1.97 |
| rms_norm | 4096×1024 | 0.035 | mx.fast.rms_norm | 1.93 |
| softmax | 4096×1024 | 0.046 | mx.softmax | 1.57 |
| gelu | 16384×1024 | 0.161 | mx.nn.gelu_approx | 2.62 |
| add_norm (fused) | 4096×1024 | 0.093 | add + mx.fast.rms_norm | 1.97 |
| attn_causal | 1×8×2048×128 | 0.795 | sdpa+mask | 3.87 |
| attn_fwd D=128 | 1×8×2048×128 | 1.499 | sdpa | 1.11 |
| attn_bwd | 1×8×1024×64 | 0.741 | mx.vjp naive | 2.46 |
| lin_attn_causal | 2×8×4096×64 | 2.050 | masked naive | 6.36 |
| matmul_custom | 2048³ bf16 | 1.233 | mx.matmul | 1.01 |
| flux gate | 2048³ | 1.227 | matmul+epilogue | 1.08 |
| cmplx_matmul | 1024³ | 0.663 | 4×mx.matmul | 1.22 |
| moe grouped | E8 H2048 | 1.401 | per-expert loop | 1.45 |
| paged_attention_v2 | 8×32 ctx2048 | 0.382 | v1 | 4.58 |

### Gaps — the optimization queue (worst first, weighted by real-model impact)
| # | kernel | shape | tk ms | vs | speedup | first hypothesis |
|---|---|---|---:|---|---:|---|
| 1 | qgemm (staged route) | q4_0 M=512 | 11.88 | tk.qgemm_direct | 0.11 | routing bug: staged path collapses at large M; direct is 9× faster |
| 2 | qgemv generic fmts | q4_K 4096² | 0.199 | fp16 matmul | 0.24 | per-element div/mod + branchy dequant in the generic template (fast paths exist only for q8_0/q4_0); W-GB/s 44–127 vs 200–430 for fast paths |
| 3 | attn_q | q4_0 1×8×1024×128 | 1.225 | attn_fwd on dequant K/V | 0.32 | in-kernel dequant dominates; multiwarp already 1.2–1.65× better |
| 4 | linear_attn (non-causal) | 2×8×4096×64 | 2.197 | Q@(KᵀV) via mx.matmul | 0.06 | grid is B·H simdgroups — no sequence parallelism; hedgehog same (0.38) |
| 5 | rotary | 1×32×2048×128 | 0.187 | mx.fast.rope | 0.44 | mx computes trig in-kernel (no cos/sin table reads) + vectorized; ours reads tables scalar |
| 6 | v2_fp8 paged decode | 8×32 ctx2048 | 0.859 | v2 bf16 cache | 0.44 | dequant-on-read costs 2.3× despite half the bytes |
| 7 | glu (all modes) | 16384×4096 | 3.925 | silu(x)*gate composed | 0.60 | scalar loads; 103 GB/s vs 546 peak |
| 8 | add_rt | 4096×1024 | 0.155 | mx add | 0.34 | 8×8 register-tile machinery for a pure elementwise op |
| 9 | hadamard D≤128 | 16384×128 | 0.150 | matmul vs H | 0.35 | geometry: too little work per threadgroup at small D (D=512 wins 2.1×) |
| 10 | qgemv q8_0/q4_0 | 4096×4096 | 0.087 | mlx_q4/q8 gemv | 0.63 | N=4096 → 4096 single-simdgroup TGs; occupancy. At N=11008 q4_0 BEATS mlx 1.73× |
| 11 | qgemv_int w8a8 | 4096² | 0.147 | tk.qgemv q8_0 | 0.26 | same small-N geometry issue |
| 12 | attn_fwd D=64 | 1×8×1024×64 | 0.238 | sdpa | 0.77 | D=64 tile geometry |
| 13 | quantize_per_tensor_fp8 | 16384×1024 | 1.529 | (per_token: 0.274) | — | 33 GB/s; global atomic-max pass dominates |
| 14 | flux gelu @1024³ | 1024³ | 0.348 | matmul+gelu | 0.65 | small-shape only (2048³ is 1.09) — low priority |

### Serving decode re-measurement (supersedes the 2026-06 table)
With pipelined timing at 8×32×2048×128: v1 1.837 ms, **staged 0.981 ms (1.80×
FASTER than v1)**, v2(p256) 0.382 ms. The earlier "staged 1.7× slower" was an
artifact of per-call-sync timing. v2 remains the default and is still the right
choice. TODO: sweep partition_size per context; re-check staged under real
decode loops (one call per step, no pipelining) before changing any default.

## Per-kernel log

### qgemv — status: experimenting (highest priority)
- References: llama.cpp `ggml-metal` GEMV geometry (N rows/simdgroup, register-
  cached activations, block-major iteration) — mirror not on disk, working from
  structure knowledge + measurement.
- Fast paths (hand-written q8_0/q4_0) hit 204–430 W-GB/s; generic template
  (26 formats) does per-element `k/block_k`, `k%block_k`, re-reads the block
  scale per element, branches per nibble, scalar X loads → 44–127 W-GB/s.
- MLX quantized_matmul comparison: tk q4_0 wins 1.73× at N=11008 but loses
  0.64× at N=4096 → geometry (one simdgroup per row, 32-thread TGs) starves the
  GPU at moderate N; bigger TGs (multiple simdgroups) should fix small-N.
- Plan: (a) restructure the generic template block-major (each lane owns
  contiguous cols within a block; scale loaded once per block; no div/mod),
  (b) 2–4 simdgroups per threadgroup, (c) vectorized X loads, then re-sweep all
  formats; correctness gate `qgemv/correctness/`.

### qgemm — status: experimenting
- M=512 q4_0 via the default (staged) path is 9.2× slower than qgemm_direct.
  Fix routing (M threshold or make direct the only path), then re-sweep BK.
- vs fp16 matmul: parity at M≥128 (compute-bound), as expected; value is memory
  footprint, not speed.

### Elementwise/row family — status: baselining done
- layernorm/rms_norm/softmax/gelu/add_norm all beat MLX fast ops — record and
  keep; re-run after any substrate change.
- glu/add_rt/rotary/hadamard need vectorization/geometry passes (queue #5/7/8/9).

### Attention — status: baselining done
- fwd D=128 & causal & bwd all ahead; fwd D=64 slightly behind (queue #12);
  multiwarp confirmed ≈0.93× of single-warp fwd (unchanged conclusion).
- attn_q needs its dequant restructured (queue #3) — likely shares the fix with
  the qgemv generic template.

### Linear-attention family — status: baselining done
- Causal/scan kernels (lin_attn_causal, mamba2, based, lin_attn_decay) healthy.
- Non-causal linear_attn and hedgehog lose massively to 2 composed matmuls
  (queue #4): consider routing non-causal shapes to composition (φ in-kernel or
  via mx ops), or a sequence-parallel two-phase kernel.
- lin_attn_decay public wrapper rebuilds the decay ramp in numpy per call —
  move to device ops (host-overhead fix independent of the kernel).

### Complex — status: healthy
- cmplx_matmul ≥ 4-matmul composition; fftconv ≈ mx.fft path. Low priority.

### Serving — status: re-baselined (see table above)
- v2_fp8 dequant-on-read (queue #6) and quantize_per_tensor_fp8 (queue #13) are
  the actionable items; also re-examine staged-vs-v1 defaults under non-pipelined
  decode conditions.

## Decision log
- 2026-07-01: harness rewritten (schema v1, all families, batched timing);
  perf.md updated to match reality (reference mirrors, device context, timing
  methodology, serving section).
