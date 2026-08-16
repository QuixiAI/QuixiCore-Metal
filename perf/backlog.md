# Metal Optimization Backlog

The beam: 3-5 active idea families, best first. Pick from the top. Update
after every concluded experiment. Kill criteria are binding — when one fires,
record the kill in `perf/findings.md` and remove the family.

Where measurable, each family carries a quantitative target derived from
recorded data — a percentage of the measured roofline, or beating a named
baseline by a stated margin — set from `perf/findings.md` or
`perf/baseline_status.md`, never invented. The backend's aggregate score
lives in `perf/scoreboard.md`.

## Beam

### 1. Quantized grouped-MoE prefill gap
- Parent result: moe_grouped_gemm_swiglu_q is 1.48–1.52× the dense bf16
  wall clock at 512 rows despite reading ~8× fewer bytes (2026-07-05,
  "Wave-9 — metal-forge gap port, kernel 1" and its optimization pass);
  MXFP8 MoE coverage rows512 was 0.41–0.88× of resident bf16 before the
  two-warp SwiGLU fix (2026-07-14).
- Hypothesis: the residual gap is dequant-fill cost per K-step; K-step 64,
  vectorized A loads, or folding moe_gather into the A load closes it. A
  shape-routed prefill-only E8M0 bit decoder may add ~8% (2026-07-13 MXFP4
  MoE result) without the rows32 regression.
- Evidence so far: two-warp SwiGLU topology won −29.5/−51.8% (MXFP8) and
  −19/−52.5% (FP8); span decoders fixed the e8m0 exp2 bottleneck; per-warp
  shared tiles and 1-warp dequant-to-shared already rejected. The E=4
  decode tile fits in SLC and masks the bandwidth advantage — the recorded
  blocker is an E=32 decode-shape sweep.
- Next action: run the recorded E=32 decode-shape sweep
  (`perf/bench_kernels.py --backend mlx --preset quick --kernel moe_q` with
  an E=32 case) on the current M5 Max, then a K-step-64 A/B on
  moe_grouped_gemm_swiglu_q at the 2880×2880 rows32/rows512 shapes.
- Kill criteria: if K-step 64 + vectorized A loads cannot bring swiglu_q
  within ~1.2× of the dense bf16 wall clock at rows512 without regressing
  rows32 decode, record the family as a capacity-only win in
  `perf/findings.md` and kill.

### 2. Quantized-KV attention structural rework (attn_q)
- Parent result: attn_q's residual ~2.5× gap vs attending on pre-dequantized
  KV is the 8-row-KV-tile shared round-trip (tiny tiles, 2 barriers per 8
  rows), NOT dequant ALU (2026-07-01, "Baseline classification", attention
  section); MXFP8 pass confirmed "the remaining attention gap is dominated
  by the attention schedule rather than scale decode" (2026-07-14). Seeded
  from queue row #3 of the 2026-07-01 Gaps table — six weeks stale.
- Hypothesis: either 32-row KV tiles (rectangular causal masking) or an
  op-level dequant-to-scratch route (~0.45 ms estimated at (1,8,1024,128),
  at 2× memory) beats the current multiwarp route.
- Evidence so far: multiwarp 4-tile staging landed (q8_0 0.98→0.447 ms);
  D128 Q16 row tile rejected on the quantized path (+19%, 2026-07-14);
  eight warps flat.
- Next action: re-verify the baseline first — re-run the attn_q quick cases
  on the current M5 Max before building either variant; the 2026-07-01
  numbers are M4 Max.
- Kill criteria: if a 32-row-tile or dequant-to-scratch prototype cannot
  beat the multiwarp route by ≥8–10% at (1,8,1024,128)-class shapes (the
  perf.md threshold for higher-complexity changes), record the structural
  limit in `perf/findings.md` and kill.

### 3. Linear-recurrence chunked prefill (GDN chunked-WY, selective_scan)
- Parent result: gdn_recur prefill 2×2048 tokens (Hv=8, Dk=Dv=128) measures
  1.50 ms after the vec4 win; selective_scan N=128 sits at ~5.1 ms,
  sequential-scan bound with a per-timestep barrier (2026-07-05, Wave-9
  kernels 5/6 and the gap-port optimization pass).
- Hypothesis: the chunked-WY parallel prefill (recorded high-risk/high-value
  for GDN) and a Blelloch/chunked time-scan (recorded for selective_scan)
  convert the serial recurrence into chunk-parallel work, as the L=64
  chunked pipeline did for lin_attn_causal/mamba2 (3.1×/2.4–4.7×,
  2026-07-01 Pass 2).
- Evidence so far: the chunked causal linear-attention pipeline is the
  proven parent pattern; the GDN and selective_scan baselines are measured
  and recorded as the required comparison points.
- Next action: prototype the GDN chunked-WY prefill behind a flag and A/B
  it against the measured 1.50 ms 2×2048 recurrent baseline (fp32 state
  parity gate first).
- Kill criteria: if the chunked prototype fails state/output parity or wins
  <1.3× at the 2×2048 prefill shape, kill (it was flagged high-risk; the
  recurrent kernels remain the correctness baseline).

### 4. Serving decode-loop regime verification (staged vs v1, defaults on M5)
- Parent result: staged paged decode is 1.80× FASTER than v1 under
  pipelined timing but ~6% SLOWER single-call; v2 wins both regimes
  3.6–4.2× (2026-07-01, "Baseline classification" serving re-measurement
  and Pass 2 serving sweeps). The standing TODO: re-check staged under real
  decode loops before changing any default. Seeded from the 2026-07-01
  queue/serving section — six weeks stale and pre-M5.
- Hypothesis: a real decode loop (one call per step, no pipelining) sits
  between the two measured regimes; defaults (v2, partition_size 256) are
  expected to survive on M5 Max but are unverified there.
- Evidence so far: partition sweep 256 ≈ 128 > 512 > 1024 (M4 Max);
  v2_fp8 penalty 1.25×; fp8/sparse MLA partition upgrades all measured
  single-call.
- Next action: re-verify the baseline first — re-run the paged_attn quick
  preset plus the single-call A/B on the current M5 Max.
- Kill criteria: if the M5 re-run confirms v2 + partition 256 defaults and
  staged non-default within noise in both regimes, record the confirmation
  in `perf/findings.md` and kill (the question is closed, not the kernels).

### 5. Moderate-N float GEMV third geometry
- Parent result: qgemv float paths at moderate-N BitNet shapes (2560–3840
  rows) trail the SLC-fed fp16 matmul; 2-row and two-simdgroup split-K
  geometries both rejected — split-K won 3–4× at small BitNet shapes but
  lost 2–3× at K=4096 because the ~22 MB working set sits at the SLC
  boundary (2026-07-01, Pass 2 and Pass 3; queue row #10 of the stale
  2026-07-01 Gaps table).
- Hypothesis: the M4 Max SLC-boundary reasoning may not hold on M5 Max; if
  the boundary moved, a split-K or row-group geometry may now win both
  shape classes, or the gap may have closed by itself.
- Evidence so far: formally recorded as an accepted limitation "needs a
  different geometry idea"; integer w8a8 got its win from 2-row, float
  paths resist both tried geometries.
- Next action: re-verify the baseline first — re-run the qgemv float-path
  BitNet and K=4096 shapes on the current M5 Max before proposing a third
  geometry.
- Kill criteria: if a third distinct geometry also loses at the K=4096 LLM
  shapes, make the accepted-limitation finding permanent in
  `perf/findings.md` and kill the family.

## Parked (not on the beam)

- fftconv batch scaling: 2.2× behind mx.fft at (8,32,32) (2026-07-01,
  Pass 2 comprehensive validation notes).
- attn_fwd D=64 sequence-block tuning, judged ≤10% (2026-07-01, Baseline
  classification, attention section).
- LM-head non-quant T>1 weight-amortized tile rewrite — 6-kernel family
  rewrite, deferred (2026-07-03, Wave-8).
- typical_p one-pass mass-histogram + local refine — could roughly halve
  again (2026-07-03, Wave-8).
- TurboQuant attention integration: rotated-domain V accumulate + one
  deferred inverse FWHT per head (2026-07-05, Wave-9 K11).
- MInference prefill block-sparse consumer + serial CSR merge port
  (2026-07-05, Wave-9 K10).
- DSA indexer fused sparse-decode consumer — blocked on a reusable
  query/scoring/top-k contract (2026-07-05 Wave-10 K3; 2026-07-14
  Cross-kernel FP8 transfer).
- Per-tensor (vs per-kv_head) fp8 KV gather variant; MLA 656-byte-layout
  cp_gather_upconvert_fp8_mla (2026-07-05, Wave-10 K2).
- Norm→quant follow-ups: standalone non-add variant, scale_ub clamp, block
  sizes ≠128 (2026-07-05, Wave-10 K1).
- copy_and_expand_eagle_inputs full padded-batch builder (2026-07-05,
  Wave-10 K5).
- selective_scan full logical-chunk vLLM scheduler metadata exercise
  (2026-07-05, Wave-9 varlen_apc follow-up).
- GDN ssd_decode-style row-owned geometry at Dk=64 (2026-07-05, Wave-9 K6).
- tau_tail float2 _d64 variant — bench-gated (2026-07-05, Wave-9 K12).
- adamw_masked compacted-index variant if segment sparsity is high enough
  (2026-07-07, BitNet training kernel port, open questions).
- Dense-teacher KD end-to-end baseline when a training harness exposes the
  loss path (2026-07-07, BitNet training kernel port).
- q4_0/q6_K decode-epilogue crossover: revisit only with isolated long runs
  (2026-07-13, Packed embedding … pass, open questions).
- NVFP4/MXFP4 QGEMM matrix path: needs a genuinely different execution
  strategy, not more fragment temporaries; profile instruction mix/register
  pressure of complete-block decode at larger hidden sizes (2026-07-13,
  NVFP4 + MXFP4 passes, open questions).
- NVFP4 T8 LM-head: a design that shares packed weights across rows without
  materializing logits (2026-07-13, NVFP4 pass, open questions).
- attn_decode_bh long-context MLX gap: kernel is 0.36–0.46× of MLX SDPA at
  T512–2048 (framework routed there today; MPS already Metal) (2026-07-13,
  Cross-kernel follow-ups).
- Spatial block-2 priority shape still loses to framework (0.39×) after the
  8×32 tile; framework crossover retained (2026-07-13).
- mean_pool_rms_l2 multi-simdgroup row-parallel variant for large M
  (2026-07-22).
- LoRA broader monotonic crossover beyond the conservative M≤4 && rank≤16
  region (isolated wins at M8 excluded) (2026-07-23).
- GDN chunk-parallel prefill also listed in Beam #3; the sequential
  gdn_recur stays the correctness baseline meanwhile (2026-07-23, Gated
  DeltaNet entry).
- Swin window attention direct kernel (non-default, framework 2.3–3.4×
  faster) — revisit only with a fundamentally different schedule
  (2026-07-13, Fused and specialized kernel integration pass).
- moe_lora_align — deliberate scope cut; revisit if a multi-LoRA MoE
  serving path lands (2026-07-05, Wave-9 K12).

## Migrated sources

- The 14-row "Gaps — the optimization queue" table in the 2026-07-01
  "Baseline classification" entry of `perf/optimization_status.md` — now
  marked superseded there. Disposition: rows 1/12/14 were harness
  artifacts; rows 2/4/5/6/7/8/9/11/13 were resolved by the 2026-07-01
  passes (see `perf/findings.md` Wins); row 3 seeded Beam #2; row 10 seeded
  Beam #5; the serving TODO under the same entry seeded Beam #4.
- "Open questions" / follow-up / deferred lines from later entries:
  Wave 3/4 and Wave-8 (LM-head T>1, typical_p), Wave-9 kernels 1/5/6/10/11/12
  and the varlen_apc follow-up, Wave-10 K1/K2/K3/K5, BitNet port entries
  (2026-07-07), the 2026-07-13 integration/second-pass/cross-kernel and
  NVFP4/MXFP4 entries, the 2026-07-14 MXFP8/FP8/transfer entries, the
  2026-07-22 mean_pool_rms_l2 entry, and the 2026-07-23 LoRA and Gated
  DeltaNet entries. Everything not grouped into a beam family is Parked
  above with its source.
