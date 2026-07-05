# ThunderMittens — performance status

## Wave-10 — metal-forge serving-glue, K1: norm->quant matrix completion (2026-07-05)

Completed the fused-add-norm quant matrix in kernels/add_norm/ (metal-forge
normalization/layer_norm_quant.metal): layernorm_add_int8_dyn (the one-off int8 LayerNorm gap)
plus per-128-block fp8/int8 for BOTH rms_add and layernorm_add. The per-block variant emits
(rows, D/128) group scales directly, so its codes feed the block-quant expert GEMMs
(moe_grouped_gemm_*_q) with no separate quantize_per_group pass. Novel piece: the per-128-block
amax in the rv_fl<D> register layout — with G=128 (compile-time, canonical) each lane's w-th
element lives in block w/4 independent of lane, so per-block absmax is a simd_max over WPB=4
consecutive w (register-resident, no threadgroup scratch, unlike the reference's group_max[256]).
Extended the AddNormFp8 primitive with group_size_/ue8m0_; fp8 gets the ue8m0 power-of-two option.

- Tests: int8 codes off-by-one vs a numpy twin (fp32 rsqrt/weight chain flips borderline codes;
  res_out bit-exact); fp8 half-ulp reconstruction + power-of-two/coverage; 41 add_norm green +
  parity (codes atol=1 across the two metallibs, scales 1e-4).
- Perf: the fusion IS the optimization (register-resident single-simdgroup). Fused per-block int8
  = 1.6x the unfused rms_norm_add -> quantize_per_group chain (16384x1024: 0.287 vs 0.456 ms;
  65536x768: 0.839 vs 1.396 ms) — eliminates the (N,D) bf16 round-trip. No further opt needed.
- Deferred (documented): standalone non-add norm-quant, scale_ub clamp, block sizes != 128.

## Wave-9 — optimization pass over the gap-port kernels (2026-07-05)

Measure-first sweep over the 12 new kernel families. The clean finding: the **bf16-I/O**
kernels win from manual vec4 (scalar bf16 loads waste bandwidth — the same lesson as
gelu_bwd/dropout in Wave-8), while the **f32** kernels do NOT (their scalar strided loads are
already coalesced and compiler-vectorized, so manual vec4 only adds overhead at scale).

WINS (kept):
- **gdn_recur** vec4 k/q loads (lanes already own contiguous Dk slices; k now read once/step
  instead of twice) — prefill 2x2048 1.56 -> 1.50 ms (~7%), decode R64 0.45 -> 0.43 ms (~5%).
- **act_quant** silu_mul_quant_{fp8,int8,fp8_group} vec4 (quant_rt float4-chunk pattern on
  both the amax and encode passes) — int8 T4096xD2880 0.30 -> 0.22 ms (~27%), T512 0.038 ->
  0.027 ms (~30%). Confirms scalar bf16 load, not the silu exp, was the ceiling.

REJECTS (measured, reverted):
- **quadratic_transform** (and by extension the f32 sampler zoo) vec4: WON at T256 (0.31 ->
  0.16) but REGRESSED at T1024 (0.70 -> 0.80, repeatable 3x) — the throughput-bound regime
  that matters more. f32 strided loads are already optimal; reverted. Applies to the whole
  logit-transform family (all f32) -> left scalar.

LEFT AS-IS (assessed, at/near floor):
- **selective_scan** N128 5.09 ms — sequential Mamba scan with a per-timestep threadgroup
  reduce barrier; B/C are strided by total_tokens (not vec4-able) and the recurrence is
  serial. Only the chunked/Blelloch rewrite (recorded, high-risk) would move it — not an
  opt-pass change.
- **moe_grouped_gemm_swiglu_q** swiglu_oai 512-row 1.93 ms vs 1.27 dense (1.52x) — already
  5-variant-tuned in the port; the two-pass reduce fits the 28 KB threadgroup budget and the
  gap is dequant + epilogue. Closing it needs the documented deep candidates (K-step 64, fold
  moe_gather into the A load) that risk correctness for a kernel already reading ~8x fewer
  bytes than dense. Deferred with spec.
- **turboquant** (fp16 chain kept verbatim for bit-exactness), **qk_norm_rope** (2.6x already,
  substrate rv_fl vector loads), **quant_rt** (already float4), tiny utilities
  (tau_tail/permute_cols/packbits/minference/moe_route, bandwidth/latency-bound) — no change.

## Wave-9 — follow-up: selective_scan varlen_apc (2026-07-05)

Completes D1.1 (selective scan): the varlen + automatic-prefix-caching (APC) variant.
Same S6 recurrence as varlen but the running state is checkpointed into PAGED state blocks
at chunk boundaries (last chunk -> block_idx_last_scheduled_token) and the initial state is
read from a possibly-cached prefix block (initial_state_idx). Buffer table transcribed 1:1
from metal-forge's selective_scan_fwd_varlen_apc_state_float32_typed (block_idx_first/last
scheduled token, initial_state_idx, cu_chunk_seqlen, last_chunk_indices, block_size,
cache_indices_stride, use_chunk_metadata). New SelectiveScanApc primitive (dedicated, not
overloaded onto SelectiveScan) with the functional pool-clone prepass; use_chunk_metadata=0
falls back to fixed block_size chunking. Gated behind its own test per the plan (highest-risk
chunk metadata).

- Tests: 4 fp64-oracle cases (uniform-chunk f32/bf16, prefix-cache-initial-state from a
  non-zero block, multi-chunk intermediate checkpoints, untouched-slot preservation) +
  fp32 parity; 16 selective_scan tests green total. The chunk-metadata (cu_chunk_seqlen)
  path is implemented and exercised via use_chunk_metadata=False fallback in these tests;
  the full logical-chunk scheduler metadata is a vLLM-runtime input (recorded).

## Wave-9 — gap port, kernel 12: marginal layout/bit utilities (2026-07-05)

New kernels/marginal/ (one dir, four ops via a small kind-dispatched primitive):
- tau_tail: scale the Q and V slices of a packed (T, 3*q_dim) QKV by tanh(tok_qv_lin)+
  tau_pos_table[pos, head] (K slice passes through); functional via the shared byte-clone
  prepass. Flat-grid elementwise v1 (any head_dim); the float2 _d64 variant is a bench-gated
  follow-up. int32 positions (TM convention; ref int64).
- packbits / segment_packbits: bool/uint8 -> bits, big/little order (np.packbits). Segment
  variant binary-searches output_indptr (host cumsum of ceil(len/8)); total_output_bytes is
  a caller-provided int (MLX's lazy graph can't read output_indptr[-1] at build time).
- permute_cols: dtype-agnostic 16-bit column gather x[:, perm] (Marlin act-order reperm).

- Tests: packbits/segment_packbits exact vs np.packbits (both bit orders, ragged rows);
  permute_cols exact vs x[:, perm] (uint16 + bf16); tau_tail vs numpy transcription with
  K-slice-untouched check; 6 green + parity atol=0 (ints/codes) / 1e-5 (tau_tail).
- Trivial bandwidth-bound ops; no bench entries.

DESIGNATED CUT (per plan): moe_lora_align — vLLM's LoRA-alignment metadata format has no
ThunderMittens consumer (moe_grouped_gemm* take the existing route/permute/pad output), so
porting it would ship dead code. Recorded here as the plan's explicit scope-tightening cut,
not an oversight; revisit if/when a multi-LoRA MoE serving path lands.

## Wave-9 — gap port, kernel 11: TurboQuant KV codec (2026-07-05)

New kernels/turboquant/ (arXiv 2502): tq_encode + tq_decode. K = asymmetric-uniform
per-32-element fp16 scale+zp (2-8 bits, signed q8_0 or unsigned sub-8-bit); V = random-sign
FWHT rotation -> per-32 fp16 RMS scale -> Lloyd-Max nearest-centroid (searchsorted against
midpoint boundaries, 2/3/4/8 bits, sub-8-bit byte-packed). The fp16 arithmetic chain is
transcribed VERBATIM from metal-forge so the numpy oracle reproduces K codes bit-for-bit.
One threadgroup per (token, kv_head), HEAD_SIZE threads, one simdgroup == one 32-elem scale
group (min/max/RMS are simd_* reductions); FWHT stages 0-4 are register shuffles, 5+ go
through threadgroup memory. head_size in {64,128,256}. TM divergences from the reference:
signs (tq_signs) + Lloyd-Max centroids (lloyd_max_centroids) are BUFFERS not baked tables;
k_bits/k_signed/v_bits are runtime scalars; slot_mapping int32. Functional 5-cache-array
return via a byte-clone prepass (untouched slots preserved). Attention integration (rotated-
domain V accumulate + one deferred inverse FWHT per head, exploiting FWHT linearity) is
spec'd in the reference and DEFERRED — this cache format already supports it.

- Tests: K codes/scale/zp bit-exact vs the fp16 oracle (8-bit signed + 4-bit unsigned +
  sub-8-bit byte-straddle); V codes >= 95% exact with off-by-one only at fp16-borderline
  boundaries; round-trip SNR floors (K 8-bit > 30 dB, V 4-bit > 18 dB); decode-vs-oracle;
  functional untouched-slot preservation (slot -1 skip). Parity: V codes + all scales
  atol=0, K codes off-by-one (borderline fp16 rint across separately compiled metallibs).
- No standalone bench entry (codec throughput dominated by the paged scatter it replaces;
  the win is the sub-4-bit cache footprint — recorded, revisit with the deferred attention
  integration).

## Wave-9 — gap port, kernel 10: MInference block-mask builder (2026-07-05)

New kernels/minference/: minference_build_block_mask converts per-head vertical column
indexes + slash diagonal offsets ((B, H, nnz) i32, -1 pad) into the per-head KV block mask
(B, H, max_blocks) that paged_attention_block_sparse now consumes directly — the consumer
gained a mask_heads scalar (buffer 16; 1 = legacy per-batch (B, max_blocks) unchanged,
H = per-head) so MInference's per-head selectivity is preserved instead of unioned away.
vertical_topk/slash_topk caps give the _mergehead budget without a second kernel;
last_n_blocks keeps the local window. The reference's serial two-pointer CSR merge is
deliberately skipped (deferred until a prefill block-sparse consumer exists) — for decode
the block mask IS the consumer format.

- Tests: exact int equality vs the numpy marking rule (+topk-cap case), end-to-end per-head
  mask -> block-sparse attention vs dense numpy restricted to kept blocks, legacy 2-D mask
  regression, full kv_cache suite (152 green), torch paged subset (24), parity atol=0.
- Trivial-cost builder (one 32-lane simdgroup per (head, batch)); no bench entry — the win
  is the KV blocks the consumer skips.

## Wave-9 — gap port, kernel 9: the sampler zoo (2026-07-05)

New kernels/sampling/sampling_transforms.metal (+ transforms.cpp/.h): 11 transforms on the
one-simdgroup-per-row substrate — quadratic/smoothing, top-nsigma, top-A (log-space exact),
epsilon-cutoff, eta-cutoff (typical_p's S1 entropy trick, one fewer pass than the reference),
XTC (on-device coin at a non-token RNG counter; e-domain comparisons, no division), skew
(index-order CDF pow via simd_prefix_exclusive_sum — verified metal-forge contract, no sort;
exllamav2 sorted-CDF variant deferred), top-k renorm (masked_topk, ties -> smaller id),
top-p renorm (32-iter bisection, deliberately tighter than the reference's 5), no-repeat-
ngram (per-lane history starts, benign -inf scatters), and DRY (faithful reference loop with
the O(max_ngram) inner unwind parallelized via first-violation simd_min; shared breakers
list + launch-uniform scalars per TM convention).

- Tests: exact-SET oracles on margin-safe 1/64-grid logits + property tests; DRY/ngram vs
  direct python transcriptions of the reference loops; 11 new + 87-test sampling regression
  green. Parity: 1e-4 on O(10) logit values (last-ulp fast-math between the two metallib
  compilers; set flips would read ~1e30), probs-domain 1e-6.
- Perf: bandwidth-bound as expected — quadratic/nsigma ~0.32 ms at (256, 32000) (~206 GB/s,
  2 passes), top_a/eta ~0.60 (3 passes), xtc 0.75 (4 passes), DRY 0.18 (copy + uniform scan).

## Wave-9 — gap port, kernel 8: fused act->quant epilogues (2026-07-05)

New kernels/act_quant/ + two substrate additions: tk_e2m1_encode (fp4 nearest-of-16, tie
behavior matching the host packer's argmin — unblocks device fp4 epilogues later) and the
glu_eval lift into include/common/glu_eval.metal (one activation definition shared by
kernels/glu and act_quant; glu suite re-run green). Kernels: silu_mul_quant_{fp8,int8}
(per-token dynamic, feeding qgemm_fp8_scaled / qgemm_w8a8), silu_mul_quant_fp8_group
(per-group-128 + ue8m0, feeding block-quant GEMMs), each with mode 0 swiglu / 1 gpt-oss
swiglu_oai via the shared glu_eval; plus rms_norm_add_int8_dyn (int8 sibling of the fp8
residual-stream epilogue). Two-pass amax+encode, activation recomputed (memory-bound).

- Tests: reconstruction-bound oracles (exp() between input and code makes bit-exact-vs-numpy
  the wrong contract), power-of-two/coverage checks for ue8m0, fused-vs-unfused composition
  (>= 95% identical codes, rest off-by-one from bf16-vs-fp32 activation rounding), 91 tests
  green incl. full glu/add_norm regression. Parity codes off-by-one max (separately compiled
  metallibs round exp differently at borderlines — same rationale as the qgemm tolerance).
- Perf: int8 epilogue T=4096 D=2880: 0.251 ms vs 0.321 ms for swiglu -> quantize_per_token
  (1.28x, the eliminated bf16 round-trip); T=512: 0.034 vs 0.036 ms.

## Wave-9 — gap port, kernel 7: per-group + asymmetric activation quant (2026-07-05)

quant_rt extensions: quantize_per_group_fp8/int8 (group-wise dynamic quant along the row,
canonical G=128 — the activation side of block-quantized GEMMs; scale (rows, D/G) f32;
ue8m0 flag rounds fp8 scales up to powers of two, MX convention) and
quantize_per_token_int8_azp (vLLM asymmetric: scale=(max-min)/255, azp=rint(-128-min/s),
constant-row fallback documented). Plus qgemm_w8a8_azp in qgemm_int: the zero-point
correction epilogue acc - azp[m]*w_rowsum[n] (w_rowsum host-precomputed) — validates the
azp layout with a real consumer. int8 codes/scales/azp BIT-EXACT vs numpy twins
(no transcendentals); fp8 verified by power-of-two + coverage + half-ulp reconstruction
bounds; azp GEMM int-exact vs int64 numpy. 40 correctness + parity-atol-0 green.

## Wave-9 — gap port, kernel 6: GDN / GatedDeltaNet linear attention (2026-07-05)

New kernels/gdn/: the Qwen3-Next / Kimi-Linear hybrid-mixer recurrence — per-timestep
delta rule S = g*S + k*beta*(v - k.S), y = q.S — one simdgroup per (request, hv, dv),
32 lanes partitioning Dk (Dk in {64,128} compile-time), fp32 state promoted from the
reference's io dtype. Varlen packed sequences (cu_seqlens) + persistent per-request
fp32 state pool via slot_mapping (race-free in-place row updates; functional via the
sscan_pool_clone prepass on both backends). GQA hk = hv/(Hv/Hk); load_initial switches
decode-continuation vs fresh-prefill.

- Tests: 8 fp64-oracle cases (3 dtypes x GQA shapes, decode step R=16, fresh-prefill
  ignores pool, untouched-slot preservation) + fp32 cross-backend parity (y and pool).
- Perf: prefill 2x2048 tokens (Hv=8, Dk=Dv=128) 1.55 ms; decode R=64 single-step
  0.42 ms (launch-overhead-dominated at tiny work). Recorded bench-gated follow-ups:
  vec4 k/q loads, ssd_decode-style row-owned geometry at Dk=64, and the chunked-WY
  parallel prefill (high-risk/high-value, only against this measured baseline).

## Wave-9 — gap port, kernel 5: Mamba-1 (S6) selective scan, dense + varlen (2026-07-05)

New kernels/selective_scan/: sequential-in-time / parallel-over-state (threadgroup per
(batch, dim), one thread per state index, dstate <= 256, fp32 state, io f32/f16/bf16).
Reference-faithful port of metal-forge/vLLM mamba semantics (softplus discretization,
exp(delta*A), D*u skip, silu(z) gate; channel-major layouts) with tk-native bf16 instead
of uint16 bit-twiddling. Varlen: flattened token axis + query_start_loc, per-request paged
state pool via cache_indices (null_block_id skip) and has_initial_state; the MLX path is
functional via an sscan_pool_clone prepass (untouched slots preserved), torch clones too.
Unlocks Mamba-1 hybrids (Jamba, Falcon-Mamba). varlen_apc (chunked state checkpoints for
prefix caching) is the recorded next step for this family.

- Tests: 12 fp64-oracle cases (3 dtypes x 3 shapes incl. dstate=160 multi-simd reduce,
  no-optionals, varlen-vs-dense with scattered pool slots + untouched-slot preservation,
  null-block skip) + fp32 cross-backend parity (out and state, atol 1e-5).
- Perf: B2/d2048/L512 0.93 ms (N=16), 5.16 ms (N=128) — sequential-scan bound, no
  framework baseline exists (per-step composition is pathological to trace). Optimization
  candidates recorded: vec4 B/C loads, one-simdgroup geometry for dstate<=32, Blelloch
  time-scan (major rewrite).

## Wave-9 — gap port, kernel 4: fused per-head QK-RMSNorm + RoPE (2026-07-05)

New kernels/qk_norm_rope/: one warp per (token, head) over packed QKV (T, (Hq+Hk+Hv)*D) —
per-head RMSNorm (gemma (1+w) flag) + RoPE (NeoX split-half via the rope_kv rv_fl<D/2>
half-vector idiom; GPT-J interleaved via the mla contiguous-lane idiom, pairs lane-local),
V heads vec-copied through. Functional out-of-place, one dispatch (the Qwen3/gpt-oss
attention-prep pattern). D in {64,128,256}, full rotary.

- Tests: 9 fp64-oracle cases (both rope styles x gemma, V-region bit-identity,
  composition cross-check) + cross-backend parity (atol 1e-2 bf16).
- Perf: Qwen3-8B shape (T=4096, 32/8/8, D=128) 0.388 ms vs 1.005 ms for the
  mx.fast.rms_norm + fast.rope + concat composition — 2.6x (target was >= 1.5x).
  T=512: 0.055 vs 0.111 ms. No optimization pass needed at v1.

## Wave-9 — gap port, kernel 3: DeepSeek grouped MoE routing (2026-07-05)

moe_route_grouped: HF DeepSeek-V3 "noaux_tc" semantics — sigmoid / softmax / sqrt-softplus
scoring, e_score_correction_bias for SELECTION only, per-group top-2-sum ranking, top
`topk_group` groups, expert top-k among survivors, weights from UNBIASED scores
(renormalize + routed_scaling_factor). One simdgroup/token over threadgroup-staged
scored/biased (E <= 512, n_group <= 32), both selection levels via the existing masked_topk
butterfly — barrier-free vs the reference's 256-thread tree-reduce design. Output contract
== moe_route_topk, so it drops into moe_permute/moe_mlp unchanged.

- Tests: 9 oracle tests (DeepSeek-V3 256/8/4/8, Kimi-K2 384/1/8, all scorings, ids-exact
  with explicit-tie and bias-flips-selection-not-weights fixtures, group-mask exclusion);
  cross-backend parity ids atol=0.
- Perf: T=4096/E=256 grouped 0.1442 ms vs plain moe_route_topk 0.1430 ms — within noise;
  the reference's E=256 bitonic fast path is confirmed unnecessary on this geometry (skipped
  as planned).

## Wave-9 — gap port, kernel 2: attention softcap + sinks (2026-07-05)

Gemma-2/3 logit soft-capping (runtime scalar, <=0 off) and gpt-oss attention sinks
(per-head denominator-only logit) across attn_fwd (+q16), attn_causal, attn_window,
attn_varlen_prefill, and paged v2 (softcap in the partitions incl. fp8; the sink merged
EXACTLY ONCE in the shared reduce — which also makes cascade/MLA compositions sink-capable
for free). Correctness traps pinned in-kernel comments: no log2(e) fold into Q when capped;
cap BEFORE masks; sink seeds the running max. sinks bound as always-present placeholder
buffers (q / tmp_out) gated by has_sink — no dummy allocations, no host_name doubling.

- Tests: fp64 oracles per kernel incl. the Gemma-2 (window+softcap) and gpt-oss
  (window+sink) layer configs; 231 attention-family correctness + 79 parity green.
- Perf: flagless-path regression guard measured clean — attn fwd 0.454 ms (SDPA 0.504),
  causal 0.226/0.878 ms, paged v2 0.433 ms (dense-loop base 1.78) at the quick shapes;
  the flags cost a uniform branch + 3 bound args when off.

## Wave-9 — metal-forge gap port, kernel 1: quantized grouped expert GEMMs (2026-07-05)

New kernels `moe_grouped_gemm_rect_q<FMT>` / `moe_grouped_gemm_swiglu_q<FMT>` (formats mxfp4,
kU4, fp8_e4m3, q8_0, nvfp4, q4_K; swiglu has act_mode 0/1 = swiglu / gpt-oss swiglu_oai with
pre-activation expert bias). Experts packed (N_out, K) so quant groups run along the
contraction axis; contraction is `mma_ABt` fed by a new col-layout register fill
(`dequant_into_register_col`, mirrors the col-layout `load` lane map — the two thread
elements are vertically adjacent). Bench: `@register("moe_q")`, baseline = dense bf16
grouped GEMM on the same schedule.

### Variant matrix (mlx quick, 2880×2880 gpt-oss tile, M4 Max; 512-row prefill numbers —
### the 32-row decode cases were session-noise-limited, ratios ~parity with dense)

| variant | rect mxfp4 | rect q8_0 | swiglu_oai mxfp4 |
|---|---|---|---|
| dense bf16 baseline | 0.625 | 0.624 | 1.259 |
| v1 naive frag fill, 1 warp | 0.716 | 0.731 | 2.333 |
| dequant-to-shared, 1 warp | 0.767 | 0.729 | 2.138 |
| 4-warp split-K, frag fill | 0.692 | 0.689 | 2.553 |
| 4-warp split-K, per-warp shared tiles | 0.918 | 0.670 | 2.947 |
| **4-warp split-K + cols4 span fill (KEPT)** | **0.674** | **0.658** | **1.863** |

### Kept
- **4-warp intra-threadgroup split-K** (each warp owns every 4th K-step, private fp32
  accumulator, one staged reduce by warp 0) — fixes the 32-thread-per-tile occupancy starvation
  of the naive port.
- **`tk_dequant_cols4_s8<FMT>` span decoder** (dequant.metal): decodes the col-fill's 4
  stride-8 columns with ONE scale unpack (specialized q8_0/fp8_e4m3/mxfp8/mxfp4/kU4; the mxfp4
  case reads just 2 bytes + 1 exp2 for 4 weights). The e8m0 formats were paying an `exp2` per
  ELEMENT in the naive fill — that, not bandwidth, was the bottleneck (mxfp4 regressed under
  plain split-K while q8_0 improved; the span decoder fixed exactly the mxfp4 side).

### Rejected (measured)
- Dequant-to-shared at 1 warp: barrier cost eats the span-decode win for the single-tile rect
  kernel (0.767 vs 0.716); only helped the 2-tile swiglu.
- Per-warp shared tiles under split-K: 20-28 KB threadgroup memory tanks occupancy
  (rect mxfp4 0.918 — worst variant).

### Status / follow-ups
- rect_q within 5-8% of the dense bf16 kernel's wall clock while reading 4-8.5× fewer weight
  bytes — the capacity win (gpt-oss-120B-class MoEs fitting at all) ships; swiglu_q still
  1.48× dense (two dequant fills/step), candidates: K-step 64, 2-warp split with per-warp
  gate/up split, vectorized A loads. Decode-shape bench needs an E=32 sweep (the E=4 decode
  tile fits in SLC, masking the bandwidth advantage) — revisit when the routing kernel lands.
- Correctness: 18 MLX tests (6 formats × bias, swiglu×act modes, q8_0-exact-vs-dense,
  end-to-end quant moe_mlp) + 8 cross-backend parity cases, all green; full qgemm suite
  re-run green after the dequant.metal change (219 + 66 tests).

## Wave-8 — optimization pass over the Wave-6/7 kernels (2026-07-03)

A measure-first pass over every new kernel. First registered the last unmeasured families in
`perf/bench_kernels.py` (typical_p, apply_bad_words, dropout, adamw, rms_norm_add, embedding_backward,
spec_verify_tree, spec_compact, build_dynamic_tree), then benched the whole surface (MLX comprehensive
+ the new quick cases). Finding: the surface is **already near-optimal** (Waves 6/7 did the heavy
lifting), with two genuine outliers, both fixed.

**Wins (measured, kept):**
- **typical_p_sample 1.8×** (10.6 → 5.87 ms, T256×V32000). The kernel's cost is the surprise-threshold
  bisection, and each step re-scans the full vocab with a per-element `exp` (bandwidth-bound at
  ~100 GB/s). 32 steps resolve tau to `smax/2³²` — pure overkill; **16 steps** give `smax/65536`,
  far below the V-token surprise spacing, so the kept set is unchanged. (A one-pass mass-histogram +
  local refine could roughly halve it again — noted as a follow-on; not done, the 1.8× is the safe win.)
- **dropout fwd+bwd ~1.9×** (2.14 → 1.09 ms, 16384×4096; ~128 → 246 GB/s). Both kernels were scalar
  one-thread-per-element — the same bf16 scalar-access bottleneck `gelu_bwd` had. `packed_four` vec4
  with a scalar tail; the keep-mask stays keyed by the element index so it's byte-identical and the
  backward still recomputes it from the seed.

**Rejects (measured, reverted/kept-as-is):**
- **adamw vec4 — reject.** Neutral (0.29→0.29, 1.11→1.13 ms). AdamW is dominated by its **four f32
  moment arrays** (m_in/v_in/m_out/v_out); f32 scalar access is already aligned/efficient, so the
  bf16-access penalty that made dropout/gelu_bwd vec4 win simply doesn't exist here. Reverted to the
  simpler scalar kernel.
- **lm_head (non-quant) at T>1 — documented tradeoff, not a bug.** argmax_T8 is 0.38× vs matmul+argmax
  because the fused partials re-read W once per token (T× the weight bandwidth) while the dense matmul
  reads W once. This is the fused path's off-purpose regime: it exists for T=1 decode and for quant
  weights (lm_head_q topk_q4_0_T8 is ~0.95×, near parity, since on-the-fly dequant offsets the
  re-reads). Non-quant T>1 should use matmul+sample. Batching T tokens per tile to amortize W would fix
  it but is a 6-kernel rewrite (argcat/topk/topp × quant/non-quant) of a hot, well-tested family for a
  narrower regime — deferred.

**Confirmed already near-optimal (no change):** rms_norm_add ~305 GB/s (register-tile fused norm),
apply_bad_words / min_p / apply_token_bitmask (bandwidth-bound over (T,V), effective BW ~230 GB/s once
the multi-pass factor is counted), embedding_backward (the **atomic** default is *faster* than the
sorted path even at V=256 heavy-dup — 0.11 vs 0.57 ms — Apple's native atomic_float wins over the
argsort+segment overhead; sorted stays available but is rarely the pick), spec_verify_tree/compact and
build_dynamic_tree (overhead-bound at sub-30 µs), the Wave-7 fused norm backwards (already on par with
mx.fast), glu/embedding_lookup/gelu_bwd (already 2–3× ahead). The tiny GB/s numbers for
varlen_build_worklist / beam_build_copy_pairs / beam advance are latency/overhead-bound, not bandwidth
(already measured + rejected in Wave 7).

---

## Wave-7 close-out (2026-07-03)

Wave 7 closed the full 20-item tail of Wave-6 deferrals (robustness, test gaps, feature
completeness, one refactor, four measure-first perf items, first-order autograd, bench/docs). All
dual-backend + parity-tested; full 3-suite regression **2133 passed** and `xcodebuild` SUCCEEDED.

**Perf items (measure-first, keep-if-win) — two prior-wave rejects REVERSED:**
- **#1 fused norm backward — WIN, reverses the Wave-6 reject below.** New `rms_norm_bwd_fused` /
  `layernorm_bwd_fused`: one simdgroup per row computes rstd (and mean) in-kernel, writes dX, and
  accumulates dweight (+ dbias for LN) via `atomic_add_float` in a **single pass**. The dweight-atomic
  contention the prior wave feared does **not** bite — Apple's native `atomic_float` handles it and
  the one-pass memory saving dominates. Measured **~2.3–2.5× faster than the old 3-pass hybrid and on
  par with `mx.fast`'s fused VJP (0.97–1.00×)** across rows 4k–16k, D 2k–8k. Both routers wired to the
  fused path; the dx-only kernels stay available.
- **#7 gelu_bwd vec4 — WIN, reverses the Wave-6 "left as-is".** `packed_four` vec4 loads/stores
  (scalar tail). Measured fp32 ~166–184 → ~352–415 GB/s (2.1–2.3×), bf16 ~88–99 → ~250–344 GB/s
  (2.8–3.5×). The earlier "tanh-bound, vec4 won't help" call was wrong: the scalar **bf16** element
  access, not the tanh, was the bottleneck.
- **#6 beam_build_copy_pairs compaction — REJECT (measured).** The fixed-slot emit is overhead-bound
  (~130 µs flat from 2k to 262k slots), so a scan + atomic-cursor compaction can't beat the launch/eval
  floor and would only add contention + nondeterministic ordering. Kept the atomic-free kernel.
- **#5 cascade single-dispatch fusion — REJECT (measured).** The 3 host concatenates are 5–23% of
  cascade time (23% only at B=1). A fused write requires decoupling the output stride from the dispatch
  count + a write-offset in the SHARED paged-attention partition kernels — a regression surface
  (paged_attention_v2 / cascade / fp8 / multi all route through them) disproportionate to a 5–23%
  single-path win. Kept the concatenate cascade (already 212–255 GB/s).

**Robustness / correctness (P1):**
- **#4 spec_compact** now uses the chunked single-threadgroup scan → **any B** (removed the B≤256 cap).
- **#9 spec_verify_tree** dropped the 64-sibling cap: the residual re-walks `last`'s child chain
  (exact for any sibling count) + a `tree_valid` first-generation fallback.
- **#19** Family-A callers validate `K ≤ #candidates` at the host boundary (lm_head `k≤V`,
  beam `V≥2·beam_width`) so the `masked_topk` `-1`-degenerate round is unreachable.
- **#11 glu geglu_erf** backward now differentiates the A&S erf approximation itself
  (`glu_erf_approx_deriv`) → bit-consistent forward/backward pair (tight 3e-5 analytic test).

**Feature completeness (P3):**
- **#8 exact quant top-p** — new `lm_head_topp_partials_q` emits a per-tile logsumexp so the reduce
  uses the **true full-vocab normalizer** (not the pool-only approximation); nucleus is exact whenever
  it fits in the over-selected pool.
- **#2 build_dynamic_tree** — device-resident draft-tree pointer builder (cap-free, scratch-free)
  replacing the host serial `spec_build_tree_pointers`.
- **#3 embedding_backward `method="sorted"`** — atomic-free segment-reduce (argsort + one threadgroup
  per id) alongside the default atomic scatter; wins under heavy id duplication.

**Refactor (#20):** `masked_topk_local` in `include/ops/group/topk.metal` — the Family-B K-round merge
shared by the three lm_head partials + beam_topk (emit functor per site).

**Autograd (#10):** `tk.autograd` — first-order differentiable `gelu/glu/rms_norm/layernorm/
embedding_lookup/dropout` on both backends (MLX `mx.custom_function` vjp → the tk backward; torch
`autograd.Function`). First order only (the kernels have no CPU path). Gotcha handled:
`mx.custom_function` passes a mis-shaped `primals` when the forward casts dtype, so each vjp closes
over the original inputs.

**#16 Xcode unit tests — N/A (confirmed).** The `tests/unit/` harness only tests substrate
register/shared tile+vec primitives (`warp::tests`/`group::tests`, gated by `TEST_*` leaf flags in
`testing_flags.hpp`); there is no registration point for kernel-level ops, so sampling/spec/embedding/
lm_head are inherently Python-tested by design. Nothing to build.

**Test-gap closures (P2):** swiglu_oai clamp-branch gradient (analytic ref, not finite-diff),
spec_verify_tree residual-distribution histogram (30k rows), cascade_attention_fp8 standalone MPS
oracle. **Bench (P7):** cascade `full-paged [prefix++suffix]` baseline (#17); numpy `ref=` oracles on
beam_build_copy_pairs + varlen_build_worklist (#18); torch comprehensive sweep recorded (#15).

---

## Wave-6 close-out (2026-07-02/03)

New serving/training families landed dual-backend + parity-tested (see the README "Serving &
training kernels" table): typical-p / bad-words / grammar-bitmask masking, quant LM-head top-p,
linear + **tree** speculative verification (`spec_verify_tree`), `spec_compact` / `spec_update_kv_meta`,
zero-copy beam `beam_remap_block_table`, **N-level** cascade attention, on-device `build_multimodal_src`,
`embedding_backward` (atomic scatter-add), GLU/GELU/RMSNorm/LayerNorm/fused-add-RMSNorm backward,
`dropout`, and fused `AdamW`. The comprehensive bench sweep now runs **0 skips across 46 families**
(`perf/results/2026-07-02/235051-mlx-comprehensive/`); an end-to-end serving+training integration
test (`tk/tests/test_integration.py`) chains them on both backends. Perf wins + rejects are in the
"Wave-6 perf pass" section below.

**Follow-ons — now DONE:**
- **Fully device-resident varlen** ✅ (`attn_varlen_prefill_device`): device `varlen_q_pad_gather` +
  `varlen_o_regather` replace the host pad/transpose loop; the whole path (worklist → pad/gather →
  attention → re-gather) runs on-device from a DEVICE `cu_seqlens` with a single scalar readback
  (`total_padded`). `varlen_build_worklist` now handles **any B** — a single-threadgroup CHUNKED scan
  (each thread owns a contiguous batch chunk: local totals → threadgroup exclusive scan → re-walk with
  the base offset) lifted the old B≤256 cap. Validated == host worklist for B up to 1000.
- **fp8 cascade prefix** ✅ (`cascade_attention_fp8` / `cascade_prefix_partition_fp8`): uint8 fp8
  (e4m3/e5m2) shared prefix, per-kv-head dequant on read, mirroring `paged_attention_v2_fp8`;
  validated == full attention over [dequant(prefix) ++ suffix].

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

### qgemv — status: LANDED (2026-07-01, three stacked wins)
Three changes, all format-generic, validated by the full regression
(862 MLX + 601 parity/MPS green):

1. **E1 — branchless float-code decoders** (`dequant.metal`): `tk_e4m3/e5m2/
   e2m1/e3m2/e2m3_decode` now shift the code into the fp16 field positions and
   rescale by a power-of-two constant instead of branch + `metal::exp2`.
   Verified exact over every finite code in numpy before landing. Subnormal
   codes (e==0) take a select computed in normal-half arithmetic — REQUIRED
   because tk_torch's offline `xcrun metal -O2` build flushes subnormal
   arithmetic (fast-math FTZ) while MLX's metallib does not; the pure-bit-trick
   version silently broke ONLY torch-side nvfp4 (caught by parity tests).
2. **E2 — block-major qgemv walk** (`qgemv.metal` template): each lane owns an
   8-col contiguous span inside a block (no per-element div/mod, `half4` X
   loads, scale reads CSE across the span).
3. **E3 — `tk_dequant8<FMT>` span decoders** (`dequant.metal`): per-format
   specializations unpack the block/sub-block scales ONCE per 8-col span
   (q4_K/q5_K/q6_K/q2_K/q3_K/iq4_xs/iq4_nl/kU4B8/kU4/hqq/nvfp4/mxfp4/q4_1/
   q5_0/q5_1). Every 8-span is provably inside one sub-block/nibble-half for
   all these layouts.

Results (ms, N=11008 K=4096 M=1; baseline = fp16 `mx.matmul` on the same run):
| format | before | after | vs fp16 matmul | W-GB/s |
|---|---:|---:|---:|---:|
| q4_K | 0.391 | 0.090 | 2.4× faster | ~281 |
| q5_K | 0.568* | 0.115 | 2.1× | ~271 |
| q6_K | 0.411 | 0.116 | 1.9× | ~250 |
| iq4_xs | 0.267* | 0.083 | 2.6× | ~281 |
| kU4B8 | 0.279* | 0.071 | 3.1× | ~322 |
| hqq | 0.220 | 0.070 | 3.1× | ~364 |
| nvfp4 | 0.321 | 0.100 | 2.3× | ~254 |
| mxfp4 | 0.288 | 0.098 | 2.5× | ~245 |
| fp8_e4m3 | 0.262 | 0.103 | 2.0× | ~466 |
| bitnet | 0.188 | 0.113 | 2.1× | ~125 |
(*= measured mid-way, post-E2 pre-E3.) Every format now beats the fp16 GEMV
2–3×; before, most LOST to it. At N=4096² all formats sit at 0.029–0.055 ms.

- Hand-written q8_0/q4_0 fast paths: still ~15–25% ahead of the improved
  template (measured by routing q8_0/q4_0 through the template) — KEPT.
- Rejected/deferred: multi-row-per-simdgroup geometry (llama.cpp N_R0-style X
  register reuse) — remaining headroom looks ≤10–30% and formats are now at
  245–466 W-GB/s; revisit if decode becomes the bottleneck again.
- Side effect of E1 on serving: **paged_attention_v2_fp8 0.859 → 0.487 ms**
  (dequant-on-read penalty 2.3× → 1.25× vs the bf16 cache) — queue #6 done.
- attn_q did NOT move (its cost is the per-element dequant_into_shared/register
  structure, not decode ALU) — see attention section.
- Also fixed: tk_torch metallib staleness check ignored `include/` substrate
  changes (stale-metallib false-greens); now walks include/*.metal mtimes.

### qgemm — status: baselined (no action needed)
- The "M=512 q4_0 is 9× slower than qgemm_direct" queue item was a MEASUREMENT
  ARTIFACT: `tk.qgemm` and `tk.qgemm_direct` construct the identical primitive
  (both direct_=true) and measure at parity once the harness warms the GPU
  clocks properly. Two harness fixes landed: a 1 s pre-run clock ramp, and
  time-based (≥50 ms) per-thunk warmup — before these, whichever thunk was
  timed first in a case read 1.5–5× slow.
- Clean numbers: q4_0/q8_0 at parity with fp16 `mx.matmul` for M∈{32,128,512}
  (compute-bound; the win is memory footprint). fp8_e4m3 M≥128 ~13% behind
  q8_0 even after E1 — remaining gap is in the fragment-path decode; candidate:
  span-decode (tk_dequant8) inside dequant_into_register/shared, shared with
  the attn_q work.

### Elementwise/row family — status: LANDED (2026-07-01)
- layernorm/rms_norm/softmax/gelu/add_norm already beat MLX fast ops (1.5–2.6×)
  — untouched.
- **rotary**: one-simdgroup-per-row + scalar substrate loads → flat one thread
  per 4 rotation pairs (bf16_4 vectors, fp32 math; ABI +M at buffer 5).
  0.187→0.078 ms at (1,32,2048,128); 0.44× → ~0.95× of mx.fast.rope (the
  remaining few % is the cos/sin table reads mx avoids by computing trig
  in-kernel — judged not worth matching).
- **glu** (6 modes): scalar → vec4 + scalar tail. 3.93→1.11 ms at 16384×4096
  (362 GB/s); 0.60× → 2.3× vs composed silu·gate.
- **add_rt**: 8×8-register-tile demo layout → flat 8 elems/thread (16-byte
  transactions). 0.34× → ~1.05× of mx add (bf16), parity f32. The rt smoke-test
  role moved to the Xcode primitive tests.
- **hadamard**: D-thread threadgroup with log2(D) barriers → one simdgroup/row,
  in-register + simd_shuffle_xor butterflies, zero barriers, zero threadgroup
  memory. D=128: 0.150→0.037 ms (1.4× vs matmul-H); D=512: 0.298→0.061 ms
  (554 GB/s ≈ roofline, 10× vs matmul-H).
- **quantize_per_tensor_fp8** (quant_rt): absmax pass now 16 elems/thread
  (16× fewer contended atomics) + vec4 encode. 1.53→0.35 ms at 16384×1024
  (4.3×); per_token 0.27→0.21 (vec4 main loops). Encoder function untouched —
  codes stay bit-identical across backends.
- torch-MPS note: the same kernels win on MPS too (glu 1.38× vs torch silu·mul,
  hadamard D=512 10×), but SMALL kernels carry ~0.05–0.1 ms extra per-call
  overhead from the tk_torch dispatch glue (add_rt bf16 0.42× of torch's add
  there despite parity on MLX) — host-glue follow-up, not a kernel issue.

### Attention — status: measured & recorded (2026-07-01)
- With clean (clock-warmed) timing: fwd D=128 **1.24× ahead** of sdpa, causal
  **3.8–5.8× ahead**, bwd 2.1–2.5× ahead of the mx.vjp naive. fwd D=64 is 0.91×
  (the earlier 0.77× was first-timed-thunk bias); remaining hypothesis —
  D=64-specific sequence-block tuning — deferred as ≤10%.
- multiwarp stays ≈0.8–0.93× of single-warp fwd → keep non-default (standing
  conclusion re-confirmed).
- **attn_q**: V now staged through shared memory like K (span dequant), K/V
  staging uses the span-decoding dequant_into_shared. fp8 single-warp 1.61→1.11
  ms, multiwarp 0.98→0.82–0.94 at (1,8,1024,128). Structural finding: the
  remaining ~2.5× gap vs attend-on-dequantized-KV is the 8-row-KV-tile shared
  round-trip (tiny tiles, 2 barriers per 8 rows), NOT dequant ALU. Options
  deferred: 32-row KV tiles (rectangular causal masking complexity) or an
  op-level dequant-to-scratch route (~0.45 ms estimated, at 2× memory).
  Recommendation recorded: prefer multiwarp=True for attn_q today.

### Linear-attention family — status: LANDED (routing) (2026-07-01)
- Causal/scan kernels (lin_attn_causal, mamba2, based, lin_attn_decay) healthy;
  lin_attn_causal beats the masked-naive baseline 5–6×.
- **Non-causal linear_attn and hedgehog now ROUTE to framework composition by
  default** (`use_kernel=True` keeps the ported kernels; parity tests pin it).
  Rationale: the kernels run one simdgroup per (batch,head) — 16 simdgroups
  total at (2,8,·,·) — while the composition uses the whole GPU per GEMM.
  linear_attn 2.20→0.142 ms (15.5×), hedgehog 0.64→0.25 ms (2.5×). A
  sequence-parallel split-KV kernel was considered and rejected: it cannot beat
  two mx.matmul calls and needs cross-threadgroup reduction scratch.
- lin_attn_decay wrapper now builds the decay ramp on-device (was numpy per
  call); neutral in pipelined benchmarks, removes the host stall in real use.

### Complex — status: healthy (no action)
- cmplx_matmul ≥ 4-matmul composition (1.0–1.22×); fftconv ≈ mx.fft path.

### GEMM (matmul_custom / gemm_staged / flux) — status: recorded (no action)
- With clean timing, flux gelu/gate beat the composed baseline at ALL measured
  shapes (1.08–1.16×) — the earlier 0.65× at 1024³ was first-thunk bias.
- matmul_custom @1024³ is 0.58× of mx.matmul while gemm_staged @1024³ is at
  parity (0.177 vs 0.176 ms); both ≈ parity at ≥2048³. Finding recorded:
  2-simdgroup staged wins at the smaller size. Not routed — these are TK-parity
  /calibration kernels; mx.matmul is the practical dense GEMM.

### Serving — status: LANDED items (2026-07-01)
- v2_fp8 dequant-on-read: fixed by the branchless decoders — 0.859→0.487 ms
  (penalty vs bf16 cache 2.3× → 1.25×, with half the cache bytes).
- quantize_per_tensor_fp8: 4.3× (see elementwise section).
- Still open: partition_size sweep per context; staged-vs-v1 default re-check
  under non-pipelined single-call decode (the pipelined harness reverses the
  old conclusion; a real decode loop sits between the two regimes).

## Pass 2 (2026-07-01, commits e00a76d..): structural rewrites

### Chunked linear-time causal linear attention family — LANDED
The three scan/decay kernels had structurally bad parallelism: lin_attn_causal
ran ONE simdgroup per (batch,head) (a serial O(N·D²) scan on B·H simdgroups);
mamba2/lin_attn_decay were parallel but QUADRATIC (every 8-row query tile
rescanned all earlier keys). New shared 3-kernel chunked pipeline (L=64):
per-chunk KV states (parallel) → exclusive chunk prefix (decay factors
telescope exactly through chunk reference points) → per-chunk output
(intra-chunk bounded to L keys + one state MMA). Both backends; small/ragged
N (<128 or N%64≠0) keeps the old kernels.
| kernel | shape | before | after | × |
|---|---|---:|---:|---:|
| lin_attn_causal | 2×8×4096×64 | 2.05 | 0.66 ms | 3.1 |
| mamba2 | 1×8×2048×64 | 0.40 | 0.17 ms | 2.4 |
| mamba2 | 2×16×4096×64 | ~6.4 (est) | 1.36 ms | ~4.7 |
| lin_attn_decay | 2×16×8192×64 | — | 1.28 ms | ~10 (est) |

### attn_q — LANDED (multiwarp staging + auto route)
Multiwarp now stages 4 KV tiles per barrier pair; single-warp keeps 1 tile
(STAGE_T=4 there collapsed occupancy: 16KB threadgroup memory on a 32-thread
TG → 6.09 ms, rejected). Added the missing q4_0 multiwarp instantiation;
tk.attn_q defaults to multiwarp="auto" (4-warp whenever non-causal, N%32==0).
q8_0 0.98 → **0.447 ms (1.14× of attending pre-dequantized KV — target hit)**;
fp8 1.61 → 0.50; q4_0 1.23 → 0.61 at (1,8,1024,128).

### mla_decode — LANDED (v2-style partitioned decode, 1.7–6.3×)
New mla_decode_partition: sequence-partition grid axis (grid (H,B,P), one
simdgroup per (head,partition)), paged-v2-style partials + the existing reduce
instantiated at D=512. (8,32,2048): 0.61→0.36; (16,32,4096): 2.27→1.31;
(8,16,8192): 4.38→**0.69 ms (6.3×)**. REJECTED with measurement: a
4-heads-per-TG token-staging variant (MQA reuse is num_heads-wide) was 1.5–1.6×
slower at 32 heads — the cache already serves cross-head reuse and the barriers
cost more than the reads they save; third confirmation of the anti-staging
lesson (gemm_staged, gqa_staged, now MLA).

### qgemv_int — LANDED
w2a8 block-major (8 codes/lane from one 2-byte read, group scale once per
span): 2× (0.176→0.094 ms at 11008×4096). w8a8 uint4 loads + TWO rows per
simdgroup: 1.9× at 4096² (0.109→0.056). The same 2-row geometry on the float
dequant q8_0/q4_0 fast paths measured 1.6–2.8× WORSE (register pressure) —
tried and reverted. Note: the int path's value remains exactness; the sped-up
dequant path is still faster at most shapes.

### Serving sweeps — measured
- partition_size sweep (ctx 2048/4096/8192): 256 ≈ 128 > 512 > 1024 everywhere;
  **default changed 512 → 256** (worth 2–4%).
- Single-call decode A/B (one call per sync, models a real decode loop):
  staged is ~6% SLOWER than v1 (its pipelined 1.8× advantage exists only when
  independent calls overlap); v2 wins 3.6–4.2× in BOTH regimes. Defaults stand.
- First-time latencies: paged alibi/block_sparse ≈ v1 base (bias/mask cost ~0);
  mla_decode_fp8 0.665 ms at (8,16,4096) on the old geometry — the partition
  upgrade for the fp8/sparse MLA variants is the identified follow-up.

### Comprehensive validation (2026-07-01, runs 192040 + 193424-mlx-comprehensive)
230 non-quant + 110 quant cases across every family and edge shape: **0 skips,
0 correctness failures** (all under tolerance). Remaining >18%-behind cases are
known/accepted: multiwarp variants (non-default), v2_fp8 (1.25× cost for half
the cache bytes), attn_q residual dequant cost (grows with N), matmul/staged
small-shape calibration kernels, int-path exactness kernels vs the (now much
faster) dequant path. Headline quant validation at vocab scale (32000×4096):
q4_K 0.240 ms (2.5× vs fp16 matmul, 307 W-GB/s), fp8 1.8×, bitnet 2.3×,
q4_0/q8_0 ≥ parity with MLX's own quantized matmul.
NEW notes catalogued for a future pass:
- hadamard D=64 at 65536 rows: 2.4× behind matmul-H (E=2/lane starves the
  simdgroup — wants 2 rows/simdgroup at D=64).
- fftconv (8,32,32): 2.2× behind mx.fft (mx scales better with batch).
- attn fwd (2,16,4096,128): 0.79× of sdpa (largest shape only — sequence-block
  tuning candidate).
- q4_K (256-superblock) PREFILL via the fragment path is 2–2.3× behind fp16
  matmul at all M (2-element gathered dequant of the branchy superblock format;
  q4_0/q8_0/fp8 prefill are at parity — use those for prefill, or apply a
  span-decode staged path for k-quants).
- qgemv float paths at moderate-N BitNet shapes (2560–3840 rows) trail the
  SLC-fed fp16 matmul; the 2-row fix that worked for w8a8 hurt the float paths
  (register pressure) — needs a different geometry idea.
Also fixed in the harness/encoders during validation: the runner now consumes
case builders lazily and clears the MLX buffer cache between cases (the
comprehensive quant sweep OOM'd twice); tk/quant.py `_nearest`/`_nearest_index`
now chunk the nearest-code search (the naive broadcast built an elements×256
float array — 134 GB for a 32000×4096 pack; results bit-identical, encoder
tests unchanged).

### torch-MPS dispatch overhead — investigated, NO ISSUE
Controlled A/B: the per-op cost of the tk_torch encode path (new encoder +
dispatch_sync per op) is ~1.5 µs over torch's own add (8×8: 0.0105 vs
0.0091 ms); at 4096×1024 tk.add_rt is 1.2× of torch add. The earlier 0.42×
harness reading did not reproduce; no fix needed.

## Pass 3 (2026-07-01, mop-up): the five catalogued gaps

### mla_decode_fp8 / fp8_sparse — LANDED (partition upgrade, 1.7–3.8× single-call)
Same v2-style partition as the bf16 decode (dense partitions the token range;
sparse partitions the top-k INDEX LIST; both feed paged_attention_reduce<bf16,512>).
The win lives in the regime that matters — sequential single-call decode:
(8,16,4096): 2.75→0.85 ms (3.2×); (8,16,8192): 5.47→1.44 ms (3.8×);
(8,32,2048): 1.46→0.84 ms (1.7×); sparse topk=2048: 0.57 ms flat regardless of
ctx. The fully-pipelined synthetic regime pays a small partial+reduce tax at
moderate ctx (overlap already saturated the old kernel there) — accepted, same
trade as the bf16 upgrade. Key measurement lesson re-confirmed: the old
geometry's pipelined numbers (0.6–1.2 ms) completely masked a 2.7–5.5 ms
single-call reality.

### k-quant prefill routing — LANDED (2–2.3×)
New `qdequant_fp16<FMT>` kernel (flat, one thread per 8-col span via
tk_dequant8) + route in qgemm (both backends): 256-superblock formats at M≥64
dequantize the whole weight and use the framework GEMM. q4_K (11008,4096)
M=512: 8.43→3.65 ms (within 11% of a pre-dequantized fp16 matmul — the residual
is the dequant pass itself). M=32 measured a wash (fixed dequant cost dominates)
→ threshold M≥64; below it the fragment path stays.

### hadamard — LANDED (lanes-per-row parameterization)
Kernel generalized to LPR lanes per row (32/LPR rows per simdgroup, xor-shuffles
confined to the row's lane group). D=64 → LPR=8 (16-byte loads, 4 rows/sg):
65536×64 0.209→0.139 ms, from 0.42× BEHIND the matmul-H baseline to 1.38×
ahead. D=128 → LPR=16: 2.2× ahead (was 1.4×). First attempt (2 whole rows per
lane at LPR=32) was only marginal — the fix was load WIDTH, not just work per
simdgroup.

### attn_fwd 16-row Q tile (D=128) — LANDED (1.3–1.6×)
attn_fwd templated on TNQ; a 16-row-Q variant (halves the passes over K/V)
routed in for D=128 when N%16==0. (2,16,4096,128): 36.8→23.6 ms — from 0.79× of
sdpa to **1.48× ahead**; (1,8,2048,128) 1.29×; (1,8,1024,128) 1.47×. D=64 q16
TRIED AND REJECTED: K/V stream is half the bytes there and the doubled register
footprint made it ~1.4× slower.

### Moderate-N float GEMV geometry — TRIED AND REJECTED (2nd idea)
Two-simdgroup split-K on the q4_0 fast path (both half-split and interleaved
strips): 3–4× faster at the small BitNet shapes (3840×2560, 2560×6912) but
2–3× SLOWER at the K=4096 LLM shapes (11008×4096, 4096×11008) — the ~22 MB
working set sits at the SLC boundary and the extra concurrency thrashes it.
Reverted; both rejected geometries documented at the launch site. The moderate-N
float GEMV gap is now formally an accepted limitation (integer w8a8 got its win
from 2-row; float paths resist both geometries).

## Wave 3/4 — serving + training families (2026-07-02)

Seven new families (MoE schedule/gather, varlen prefill, fused LM-head, beam
advance, sliding-window paged decode, Mamba2 backward + D=128, fused
cross-entropy) plus a Wave-4 close-out pass. Capability + honest perf tradeoffs:

### LM-head sampling — DEFAULT ROUTES THROUGH MATMUL (fused GEMV is a fallback)
The fused per-lane `lm_head_argcat/topk_partials` GEMV reads W column-major and
re-reads it per token; on Apple that loses to `mx.matmul(h, Wᵀ)` + the fast
sampler by 2.7–3.9× at (T∈{1,8}, V∈{32000,128256}). So `tk.lm_head_sample`
**defaults to matmul+sampler** (dense and quant: `qgemv/qgemm` then the sampler).
The fused path is kept (now vec4-coalesced + a `tk_dequant8<FMT>` quant reader for
q8_0/q4_0) for `fused=True` callers and cross-backend parity, but it is not the
latency winner — the coalesced/tiled rewrite closed the gap toward the matmul
bandwidth floor without beating it. Correctness re-covered by fused-vs-logits
oracle tests (dense + quant).

### beam advance — single-pass register top-M (2.5–6.5×, now beats framework)
Replaced the multi-pass selection with a single-pass register top-M merge over
the (beam×vocab) candidate rows (parent-id penalty history threaded in). Exact,
and 2.5–6.5× faster than the framework top-k baseline. `beam_reorder_kv` (a
host helper over `kv_cache_copy_blocks`) and `beam_length_penalty`
(FasterTransformer `((5+len)/6)^α` normalization) round out the beam API — both
host-side policy, no kernel.

### fused cross-entropy — ~11× in the fused-linear regime; +softcap capability
`fused_linear_cross_entropy` stays the big win (avoids materializing the (T,V)
logits). Added the Gemma-2 final-logit **softcap** (`z' = c·tanh(z/c)`, gradient
through tanh) threaded fwd+bwd; vec4 vocab-scan loads; and a 4-simdgroup-per-row
variant routed for the latency-bound small-T/large-V case (2–3× there over the
1-simdgroup scan). All measured; the 4-simdgroup path is kept only where it wins.

### sliding-window paged decode — threaded through fp8 + v2 partition paths
`window` now clamps the key loop on `paged_attention_fp8`,
`paged_attention_partition`, and `_partition_fp8` (raises the lower bound only;
out-of-window whole partitions write l==0 and are dropped by the reduce). So
long-context windowed decode uses the partitioned path instead of falling back.
`window=0` / `window≥ctx` are bit-identical to the pre-existing outputs;
`window+alibi` and `window+block_sparse` compose.

### Mamba2 chunked linear-time backward — LANDED (D=64), parity with quadratic
The quadratic backward rescans every earlier key tile per query tile (O(N²D)).
The chunked backward mirrors the forward pipeline (intra-chunk bounded quadratic
+ inter-chunk D×D states P/Q/Qᵀ staged through device scratch), giving O(N(L+D)D)
and matching the O(N²) route to <2% on dC/dB/dX/dcl. Routed for D=64, N%64==0,
N≥128 (else the quadratic fallback, unchanged for D=128 / small N), exactly like
the forward. Comprehensive sweep: bwd (1,8,2048,64) 0.369 ms, (2,16,4096,64)
2.76 ms — total work grows 8× between those points (4× batch·head, 2× N) yet
time grows only 7.5×, i.e. near-linear in N (the quadratic route would grow with
N²). Backward is ~3–4.5× the matching forward (dC/dB/dX + P + Q + Qᵀ states).

### Wave-4 perf-pass audit (2026-07-02) — the new families are already at their optima
Went kernel-by-kernel over the Wave-3/4 code looking for wins; the honest result
is that they were built perf-first and there is no shippable win left:
- **beam advance** already beats the framework top-k 1.2–5.1× (single-pass
  register top-M); **fused cross-entropy** is 9.5–13.8× ahead (bandwidth-floor
  logit scan, lane-coalesced; vec4 was correctly *skipped* — the exp/tanh is
  compute-bound, and vec4 would only muddy coalescing); **sliding-window** is a
  key-loop clamp that strictly *reduces* work; **lm-head fused** is vec4'd and
  its W-reuse rewrite was already tried and rejected (lost parallelism) — the
  default routes T>1 through matmul+sampler, so the fused path is a T=1
  specialization that already wins there. No change to any of these.
- **REJECTED — mamba2 backward Qᵀ-via-transpose.** Qt_c = Q_c^T *exactly* (the
  reverse scan's decay is element-independent; verified bit-exact in numpy), so
  the second `qkv` MMA + reverse-scan pass that builds Qt is provably redundant
  and could be replaced by transposing the scanned Q. Implemented on both
  backends and MEASURED against the two-pass in the same session: the general
  (…,D,D) transpose is scatter-bound and **regressed the common seq-2048 shape
  ~30%** (0.37→0.49 ms) for only ~4.5% at seq-4096 (2.67→2.55 ms). A
  fused-transpose-in-scan hits the same scatter-bound 64×64 write. Reverted; the
  two-pass (cheap qkv MMA + coalesced scan) is kept and the tradeoff is
  documented at the launch site.

## Decision log
- 2026-07-02: Wave-4 perf-pass audit — the new families are already perf-first;
  no shippable win. Rejected the mamba2-backward Qᵀ-via-transpose (bit-exact but
  scatter-bound; −30% on seq-2048). Re-confirmed beam/CE/lm-head/window optima.
- 2026-07-02: Wave-4 close-out — sliding-window on fp8/v2 decode; beam
  reorder-kv + length-penalty; CE softcap + vec4/4-simdgroup; LM-head coalesced
  GEMV + quant reader (default still matmul+sampler); Mamba2 chunked linear-time
  backward (parity-checked vs the quadratic route). All dual-backend + parity.
- 2026-07-01: harness rewritten (schema v1, all families, batched timing);
  perf.md updated to match reality (reference mirrors, device context, timing
  methodology, serving section).
- 2026-07-01: TWO timing-bias fixes (1 s clock pre-ramp; ≥50 ms time-based
  per-thunk warmup). Queue items #1 (qgemm M=512 "routing bug"), most of #12
  (attn D=64) and #14 (flux gelu small) were artifacts of the biased harness —
  always re-verify a gap on the fixed harness before optimizing.
- 2026-07-01: qgemv E1/E2/E3 landed (branchless decoders + block-major walk +
  span dequant) — every quant format now beats the fp16 GEMV; commit a38e8a8.
- 2026-07-01: rotary/glu/add_rt/hadamard flat-vectorized geometry; attn_q V
  staging; commit f16b9d5.
- 2026-07-01: linear_attn/hedgehog routed to framework composition
  (use_kernel escape hatch); quant_rt vectorized + atomic-thinning; lin_attn_decay
  device-side ramp.
- Rejected this pass: multi-row qgemv geometry (≤10–30% left, formats at
  245–466 W-GB/s), attn_q 32-row KV tiles (complexity vs niche kernel),
  matmul_custom small-shape routing (calibration kernel), fp16 LUT decoders
  (bit-tricks already exact + cheap).

## Wave-6 perf pass (2026-07-02, comprehensive sweep, 46 families / 536 cases / 0 skips)

Full run: `perf/results/2026-07-02/235051-mlx-comprehensive/`. First measurement of the
Wave-5 serving/training kernels (they shipped correctness-first, un-profiled).

### Wins landed (measure-and-keep)
- **embedding_lookup 8.0×** (1.817 → 0.227 ms, T4096·D4096·V128256) — now **2.3× faster than
  the framework gather** (was 3.5× slower). **merge_multimodal_spans 11.9×** (7.478 → 0.626 ms).
  Cause: the flat one-thread-per-element kernels did two integer divides per element (t=gid/D,
  d=gid%D) and scalar loads. Fix: one threadgroup per token (row base hoisted, zero per-element
  division) + `packed_four` vec4 row load/store when D%4==0, scalar tail otherwise.
- **beam_reorder_kv ~1.45×** (6.11 → 4.12 ms; 12.27 → 8.47 ms) — vec4'd `kv_cache_clone` (the
  full-cache copy that dominates the reorder) and `kv_cache_copy_blocks`. The clone is the driver;
  the block copy alone moved ~2%. The direct kv-copy path also benefits.

### Rejected / documented tradeoffs (this pass)
- **rms_norm / layernorm backward: 5–6× slower than `mx.fast.*_norm` vjp** (e.g. rms_norm_bwd
  N65536·D1024 8.9 ms vs 1.6 ms). NOT a kernel-speed bug — the router computes rstd/dweight/dbias
  as *separate framework reductions* and only the per-row dX as the tk kernel (multi-pass, ~3× the
  traffic), while `mx.fast` fuses the whole VJP in one pass. Matching it needs a fully-fused
  backward kernel (rstd+dX in one pass, dweight via `atomic_add_float`/second reduction). Deferred:
  large effort against a hand-tuned builtin on the less-latency-critical training path; the current
  path is correct (matches torch autograd ~1e-2..1e-7) and reuses the dX kernel.
- **gelu_bwd**: compute-bound (per-element `precise::tanh` + cubic), no per-element division; already
  76–131 GB/s. vec4 wouldn't move the tanh-dominated cost — left as-is.
- **cascade_attention** already 212–255 GB/s; **beam_build_copy_pairs / varlen_build_worklist**
  already sub-20 µs (fixed-slot / single-threadgroup scan) — not hotspots.
- **min_p / apply_token_bitmask / spec_verify_linear** run at 46–756 GB/s over the (T,V) logits —
  bandwidth-bound and already near the floor for a full-vocab pass.
