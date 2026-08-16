# Metal Established Findings — Do Not Re-Derive

Distilled from `perf/optimization_status.md` through 2026-07-24. Treat as
current truth until re-measured; every entry names its date and notebook
entry so it can be challenged with new data.

## Environment anchor

Two hardware eras. Every number dated 2026-07-01 through 2026-07-14 was
measured on MacBook Pro Mac16,5, Apple M4 Max (40-core GPU, 128 GB,
~546 GB/s theoretical DRAM), macOS 26.5.1 (25F80) → 26.5.2 (25F84), Xcode
26.6 (17F113), Apple Metal 32023.883 / toolchain 17.6.109.0, Python 3.12.9,
MLX 0.21.1, PyTorch 2.12.1 MPS. Every number dated 2026-07-22 onward was
measured on MacBook Pro Mac17,6, Apple M5 Max (128 GB), macOS 26.5.2
(25F84), same Xcode/Metal toolchain, Python 3.12.13, MLX 0.21.1, PyTorch
2.13.0 MPS. No cross-era re-measurement exists; treat pre-07-22 absolute
times as M4 Max facts. Timing method throughout: `perf/bench_kernels.py`
(1 s clock ramp, ≥50 ms per-thunk warmup, adaptive batching to ≥2 ms per
synchronized sample, per-call median with p20/p80 and CV).

## Wins

| finding | effect | date | notebook entry |
|---|---|---|---|
| qgemv E1/E2/E3: branchless float-code decoders + block-major 8-col span walk + `tk_dequant8<FMT>` span decoders | every quant format beats fp16 GEMV 2–3.1× at N11008 K4096 M1 (q4_K 0.391→0.090 ms, hqq/kU4B8 3.1×, 245–466 W-GB/s); before, most lost to it | 2026-07-01 | Baseline classification (2026-07-01, quick preset) |
| paged_attention_v2_fp8 branchless-decoder side effect | 0.859→0.487 ms; dequant-on-read penalty 2.3×→1.25× vs bf16 cache at half the bytes | 2026-07-01 | Baseline classification (2026-07-01, quick preset) |
| rotary: flat one-thread-per-4-pairs, bf16_4 vectors | 0.187→0.078 ms at (1,32,2048,128); 0.44×→~0.95× of mx.fast.rope | 2026-07-01 | Baseline classification (2026-07-01, quick preset) |
| glu (6 modes): scalar→vec4 + scalar tail | 3.93→1.11 ms at 16384×4096 (362 GB/s); 0.60×→2.3× vs composed silu·gate | 2026-07-01 | Baseline classification (2026-07-01, quick preset) |
| add_rt: 8×8 register tile→flat 8 elems/thread | 0.34×→~1.05× of mx add (bf16) | 2026-07-01 | Baseline classification (2026-07-01, quick preset) |
| hadamard: one simdgroup/row, simd_shuffle_xor butterflies, zero barriers | D=128 0.150→0.037 ms; D=512 0.298→0.061 ms (554 GB/s ≈ roofline, 10× vs matmul-H) | 2026-07-01 | Baseline classification (2026-07-01, quick preset) |
| quantize_per_tensor_fp8: 16 elems/thread absmax (16× fewer atomics) + vec4 encode | 1.53→0.35 ms at 16384×1024 (4.3×); per_token 0.27→0.21 | 2026-07-01 | Baseline classification (2026-07-01, quick preset) |
| non-causal linear_attn / hedgehog routed to framework composition by default | linear_attn 2.20→0.142 ms (15.5×), hedgehog 0.64→0.25 ms (2.5×) | 2026-07-01 | Baseline classification (2026-07-01, quick preset) |
| k-quant prefill routing: `qdequant_fp16<FMT>` + framework GEMM at M≥64 | q4_K (11008,4096) M=512 8.43→3.65 ms (within 11% of pre-dequantized fp16 matmul); M=32 a wash → threshold M≥64 | 2026-07-01 | Pass 3 (2026-07-01, mop-up): the five catalogued gaps |
| hadamard lanes-per-row (LPR) parameterization | D=64 65536 rows 0.209→0.139 ms (0.42×→1.38× vs matmul-H); D=128 1.4×→2.2× | 2026-07-01 | Pass 3 (2026-07-01, mop-up): the five catalogued gaps |
| attn_fwd 16-row Q tile for D=128 (N%16==0) | (2,16,4096,128) 36.8→23.6 ms — 0.79×→1.48× vs sdpa; 1.29–1.47× at smaller shapes | 2026-07-01 | Pass 3 (2026-07-01, mop-up): the five catalogued gaps |
| mla_decode_fp8 / fp8_sparse v2-style partition upgrade | single-call (8,16,4096) 2.75→0.85 ms (3.2×); (8,16,8192) 5.47→1.44 (3.8×); sparse topk=2048 0.57 ms flat | 2026-07-01 | Pass 3 (2026-07-01, mop-up): the five catalogued gaps |
| chunked (L=64) 3-kernel causal linear-attention pipeline | lin_attn_causal 2.05→0.66 ms (3.1×); mamba2 0.40→0.17 (2.4×) and ~4.7× at 2×16×4096; lin_attn_decay ~10× (est) | 2026-07-01 | Pass 2 (2026-07-01, commits e00a76d..): structural rewrites |
| attn_q multiwarp 4-KV-tile staging + multiwarp="auto" route | q8_0 0.98→0.447 ms (1.14× of attending pre-dequantized KV — target hit); fp8 1.61→0.50; q4_0 1.23→0.61 | 2026-07-01 | Pass 2 (2026-07-01, commits e00a76d..): structural rewrites |
| mla_decode v2-style partitioned decode | 1.7–6.3×; (8,16,8192) 4.38→0.69 ms | 2026-07-01 | Pass 2 (2026-07-01, commits e00a76d..): structural rewrites |
| qgemv_int: w2a8 block-major; w8a8 uint4 loads + 2 rows/simdgroup | w2a8 2× (0.176→0.094 ms at 11008×4096); w8a8 1.9× at 4096² (0.109→0.056) | 2026-07-01 | Pass 2 (2026-07-01, commits e00a76d..): structural rewrites |
| paged-attention partition_size default 512→256 | 256 ≈ 128 > 512 > 1024 at ctx 2048/4096/8192; worth 2–4% | 2026-07-01 | Pass 2 (2026-07-01, commits e00a76d..): structural rewrites |
| beam advance: single-pass register top-M merge | exact, 2.5–6.5× faster than framework top-k | 2026-07-02 | Wave 3/4 — serving + training families |
| fused_linear_cross_entropy (+softcap, vec4 scan, 4-simdgroup small-T route) | 9.5–13.8× ahead of composed baseline; 4-simdgroup path 2–3× in the small-T/large-V case | 2026-07-02 | Wave 3/4 — serving + training families |
| lm_head_sample default = matmul + fast sampler (fused GEMV kept as fallback) | matmul route 2.7–3.9× faster at T∈{1,8}, V∈{32000,128256} | 2026-07-02 | Wave 3/4 — serving + training families |
| Mamba2 chunked linear-time backward (D=64, N%64==0, N≥128) | matches quadratic route <2%; near-linear in N (8× work → 7.5× time) | 2026-07-02 | Wave 3/4 — serving + training families |
| embedding_lookup / merge_multimodal_spans: one TG/token + vec4 (no per-element div) | 8.0× (1.817→0.227 ms) and 11.9× (7.478→0.626 ms); lookup now 2.3× faster than framework gather | 2026-07-02 | Wave-6 perf pass (2026-07-02, comprehensive sweep, 46 families / 536 cases / 0 skips) |
| beam_reorder_kv: vec4 kv_cache_clone + copy_blocks | ~1.45× (6.11→4.12 ms; 12.27→8.47 ms) | 2026-07-02 | Wave-6 perf pass (2026-07-02, comprehensive sweep, 46 families / 536 cases / 0 skips) |
| fused one-pass norm backward with atomic_float dweight (REVERSES the Wave-6 defer) | ~2.3–2.5× vs the 3-pass hybrid; 0.97–1.00× of mx.fast fused VJP across rows 4k–16k, D 2k–8k | 2026-07-03 | Wave-7 close-out |
| gelu_bwd vec4 (REVERSES the Wave-6 "tanh-bound, left as-is" call) | fp32 ~166–184→~352–415 GB/s (2.1–2.3×); bf16 ~88–99→~250–344 GB/s (2.8–3.5×) | 2026-07-03 | Wave-7 close-out |
| typical_p_sample: 32→16 bisection steps | 1.8× (10.6→5.87 ms, T256×V32000); kept set unchanged | 2026-07-03 | Wave-8 — optimization pass over the Wave-6/7 kernels |
| dropout fwd+bwd vec4 | ~1.9× (2.14→1.09 ms at 16384×4096; ~128→246 GB/s); mask byte-identical | 2026-07-03 | Wave-8 — optimization pass over the Wave-6/7 kernels |
| quant grouped expert GEMM: 4-warp intra-TG split-K + `tk_dequant_cols4_s8` span fill | rect_q within 5–8% of dense bf16 wall clock reading 4–8.5× fewer weight bytes; the span decoder fixed the e8m0 per-element exp2 bottleneck | 2026-07-05 | Wave-9 — metal-forge gap port, kernel 1: quantized grouped expert GEMMs |
| qk_norm_rope fused per-head QK-RMSNorm + RoPE | Qwen3-8B shape 0.388 vs 1.005 ms composition (2.6×) | 2026-07-05 | Wave-9 — gap port, kernel 4: fused per-head QK-RMSNorm + RoPE |
| fused act→quant epilogues (silu_mul_quant int8) | 1.28× vs swiglu→quantize chain at T4096 D2880 (0.251 vs 0.321 ms) | 2026-07-05 | Wave-9 — gap port, kernel 8: fused act->quant epilogues |
| gdn_recur vec4 k/q loads; act_quant vec4 amax+encode | gdn ~5–7% (prefill 1.56→1.50 ms); act_quant ~27–30% (0.30→0.22 ms int8 T4096) | 2026-07-05 | Wave-9 — optimization pass over the gap-port kernels |
| fused per-128-block norm→quant (register-resident amax) | 1.6× vs unfused rms_norm_add→quantize_per_group (16384×1024: 0.287 vs 0.456 ms) | 2026-07-05 | Wave-10 — metal-forge serving-glue, K1: norm->quant matrix completion |
| BitNet fake_quant / weight_quant_ternary ports | fake_quant 1.86–2.38×, weight_quant_ternary 2.99–3.71× vs decomposed framework paths | 2026-07-07 | BitNet training kernel port |
| Fused specialized kernels kept as defaults (decode add+LN, patch merge+LN, decode linear+erf-GELU, dequant gather, constrained LM head) | 1.20–3.91× vs framework/decomposed baselines on MLX and MPS | 2026-07-13 | Fused and specialized kernel integration pass |
| packed embedding lookup/bag, sparse LM-head projections, packed decode epilogues | embedding/bag 2.25–4.96×; lm_head_masked/candidates 4.53–10.28×; decode_swiglu q4_0/q8_0 2.46–2.66× (second-pass finals) | 2026-07-13 | New-kernel second optimization pass |
| decode_cache_attention: 32 SIMD groups (hardware-limit 1024-thread TG) | T512/1024/2048 0.321/0.692/1.439→0.135/0.251/0.496 ms; Metal is the public route for all measured 64–4096 slots (1.42–2.02× MLX, up to ~4× MPS) | 2026-07-13 | New-kernel second optimization pass |
| LM-head q4_0 whole-32-value-block decode in sequential sampler dots | 10.85× / 4.58× (T1/T8) over generic 8-value helper; 20.62× vs dense top-k at T1 V32000 K4096 | 2026-07-13 | Cross-kernel follow-ups and optimization pass |
| pairwise edge MLP factorized project/combine | 9.63× over the old direct kernel; 1.38× over the materialized framework composition | 2026-07-13 | Cross-kernel follow-ups and optimization pass |
| quantized beam advance: staged exact fusion B≤4, packed-GEMM route above | q4_0 B1 1.31× vs resident fp16; matrix route 2.41×/3.66× over row-wise fusion at B4 | 2026-07-13 | Cross-kernel follow-ups and optimization pass |
| spatial 8×32 compile-time matrix tile | 1.73×/2.42× (S2/S4) over the prior direct kernel | 2026-07-13 | Cross-kernel follow-ups and optimization pass |
| attn_decode_bh context partitions (8 groups T512–2047, 32 from T2048) | 8.21–14.91× over the serial kernel; MPS auto-routes Metal at all shapes (3.37–4.70× vs framework) | 2026-07-13 | Cross-kernel follow-ups and optimization pass |
| NVFP4 row/column fragment decoders + whole-block fused decoders | QGEMM −8.1 to −13.1%; MoE rows512 −11.9%; decode epilogue −70.3%, SwiGLU −84.6%, masked −61.4%, candidates −48.6%, LM-head T1 −26% | 2026-07-13 | NVFP4 inference decode and output-projection pass |
| MXFP4 whole-block QGEMV + complete-block sequential decode + E8M0 bit reconstruction (fp32 span/sequential paths only) | QGEMV −18.7/−31.4% (444 W-GB/s at N11008); LM-head T1 −46.4%; masked T1 −62.4%; beam B1 −35.9%; embedding T1 −17.9% | 2026-07-13 | MXFP4 inference coverage and hot-path pass |
| MXFP8 hot-path set: whole-block QGEMV, masked-only E4M3 LUT, arithmetic whole-block sparse projections, beam matrix route, MoE SwiGLU two-warp, QGEMM 2×32 staged (M%64==0) | QGEMV −3.7/−4.4%; masked −33%; candidates −8/−11%; beam B1 −10.4% (MPS −11.9%); MoE SwiGLU −29.5%/−51.8%; QGEMM staged 1–3% | 2026-07-14 | MXFP8 inference hot-path experiments |
| FP8 paged/cascade scale hoist + compile-time E4M3/E5M2 format specialization | hoist −2.1 to −9.1%; specialization −1.4 to −20.7% (E5M2 v2 D128 0.489→0.388 ms) | 2026-07-14 | FP8 inference hot-path experiments |
| E4M3/E5M2 normal-value bit encoders (IEEE-754 exponent/mantissa derivation) | per-token quant −67.6%, per-tensor −33.2%, fused act quant −15.9/−22.8%, KV scatter −4.1 to −30.1% | 2026-07-14 | FP8 inference hot-path experiments |
| FP8 E4M3 SwiGLU MoE two-warp gate/up | rows32 −19.0%, rows512 −52.5% (2.92→1.39 ms) | 2026-07-14 | FP8 inference hot-path experiments |
| grouped act-quant compile-time SwiGLU/SwiGLU-OAI symbols; atomic-zero `orderable_uint_to_float(0)` fix | −3.2 to −9.7% on grouped shapes; all-zero inputs now yield zero codes/scale (correctness, no speed claim) | 2026-07-14 | Cross-kernel FP8 transfer experiments |
| mean_pool_rms_l2 fused pooling (small M) | 1.35× vs torch composition at M128 D768; large-M regression inherent to single-simdgroup design | 2026-07-22 | mean_pool_rms_l2 — new embedding-pooling serving kernel |
| qgemv_fused packed-Q4_0 decode GEMVs (up+gate+GELU / up+gate / QKV) | up_gate_gelu 1.43–3.16×; qkv 1.12–1.69×; up_gate 1.03–1.30× vs composing standalone qgemv calls | 2026-07-22 | qgemv_fused — fused packed-Q4_0 decode GEMVs (up+gate+GELU / up+gate / QKV) |
| rms_norm_residual_next fused residual seam | 1.15–1.83× vs two rms_norms + add | 2026-07-22 | rms_norm_residual_next — fused residual-stream seam (two RMSNorms + add) |
| qk_norm_rope_kv_f16 fused f16 KV split-store | 1.34–2.26× vs qk_norm_rope + un-pack/f16 cast | 2026-07-22 | qk_norm_rope_kv_f16 — qk_norm_rope with a fused f16 KV split-store |
| attn_fwd_sg_d256 simdgroup_matrix flash attention (D=256, GQA, f16 KV) | 1.71–2.54× vs torch SDPA at D=256 shapes the TK attn_fwd path does not cover | 2026-07-22 | attn_fwd_sg_d256 — simdgroup_matrix flash attention (D=256, GQA, f16 KV) |
| BaseQN eight-value-per-scale-load GEMV; M>1 routed to dequant + framework GEMM | 1.87× kernel improvement (0.2088→0.1115 ms) and 7.67× over decode-then-matmul; quick sweep 0.108–0.283 ms vs 0.833–2.566 ms composition across Q3/Q4/Q6/Q8 | 2026-07-23 | canonical BaseQN dequant, GEMV, and GEMM routing |
| BaseQN fused SwiGLU and short-K QKV fusion (K≤1024) | SwiGLU 1.35–1.69×; QKV 1.13–1.23× at K≤1024, direct-GEMV composition above | 2026-07-23 | BaseQN QKV and SwiGLU consumer fusion |
| BaseQN grouped expert decode: one simdgroup rect, four-way split-K SwiGLU only | SwiGLU split-K 2.31×; retained sweep 1.09–2.45× vs dequant+dense; format sweep rect 1.05–1.12×, SwiGLU 2.16–2.23× | 2026-07-23 | BaseQN grouped expert projection and SwiGLU |
| BaseQN LM-head via columnwise direct-GEMV + argmax composition | equals the strongest baseline; 5.70×/2.91× vs the M>1 materialize-weight GEMM route at batch 2/4 | 2026-07-23 | BaseQN greedy LM-head routing |
| positioned/partial/M-RoPE + fused positioned qk_norm_rope | standalone 1.50–3.19×; fused 4.07–4.85× vs decomposed framework compositions | 2026-07-23 | Positioned, partial, and multimodal RoPE |
| Q8_0 KV direct dequant-on-read paged attention (separate-plane ABI, safe-FP exact encoder) | 1.46–2.91× vs BF16 paged reads; 2.68–3.90× vs Q8 gather+SDPA; cache is 53.125% of BF16 bytes | 2026-07-23 | QuixiCore Q8_0 KV codec and direct paged read |
| GDN preparation/output kernels + sigmoid attention gate | 1.12–6.18× across fused short-conv, QKV prep, decay/beta, gated RMSNorm, sigmoid gate | 2026-07-23 | Gated DeltaNet preparation/output and sigmoid attention gate |
| calibration channel-absmax reduction and final-logit softcap | calibration 2.87–3.68×; softcap 3.34–3.64× | 2026-07-23 | calibration reduction and final-logit softcap |
| fused LoRA apply, conservative route M≤4 && rank≤16 | 2.08× at M1/R8; framework outside the contiguous measured region | 2026-07-23 | fused low-rank adapter application and routing |
| BERT token/type embedding + masked normalized pooling | embedding 2.57–2.73×; masked pool 2.82–6.52× | 2026-07-23 | BERT token/type embedding and masked normalized pooling |
| vision position interpolation and token avg-pool direct kernels | interpolation 22.21–24.96×; avg pool 2.79–3.18× (canonical patchify stays framework) | 2026-07-24 | reusable vision patch, position, and pooling operations |
| audio depthwise conv1d + fused SiLU; short-memory (Tk≤128) online cross-attention | depthwise 6.90×; short cross-attention 1.80× (general conv and long memory stay framework) | 2026-07-24 | audio convolution and cross-attention routes |
| factorized position, two-axis vision RoPE, causal depthwise, blocked relative attention | 2.34×, 10.11×, 6.13×, 5.31× vs framework compositions | 2026-07-24 | strict BaseRT vision/audio contract audit |
| value_clip scalar-bounds kernel; Qwen global-split vision RoPE mode | clip 1.33–1.96×; Qwen 2-D RoPE 5.46× | 2026-07-24 | Qwen temporal patch and Gemma value-clip closure; explicit Qwen vision RoPE layout |

## Rejected — with the reason, so they are not retried

**Threadgroup staging / decoded-weight sharing — REJECTED repeatedly
(2026-07-01 through 2026-07-14).** Every attempt to stage KV tokens or share
decoded weight tiles across warps lost: gemm_staged and gqa_staged
(pre-notebook standing findings), the MLA 4-heads-per-TG token-staging variant
(1.5–1.6× slower at 32 heads — Pass 2, mla_decode), single-warp attn_q
STAGE_T=4 (16 KB threadgroup memory on a 32-thread TG → 6.09 ms), Wave-9
per-warp shared tiles under split-K (rect mxfp4 0.918 ms, worst variant),
NVFP4 two-/four-warp decoded-weight reuse (+2.3%/flat), and FP8 two-warp
shared weights in scaled and block2d GEMM (+19–24%). The cache already serves
cross-thread reuse; barriers and occupancy loss cost more than the reads they
save. Generalized rule: on Apple, staging pays only when measured reuse beats
barriers + occupancy — default to direct loads.

**Multi-row / split-K float qgemv geometry — REJECTED three times
(2026-07-01).** (1) Multi-row-per-simdgroup: remaining headroom ≤10–30% with
formats already at 245–466 W-GB/s (Baseline classification). (2) The 2-row
geometry that won for int w8a8 measured 1.6–2.8× WORSE on the float
q8_0/q4_0 fast paths — register pressure (Pass 2). (3) Two-simdgroup split-K
(half-split and interleaved): 3–4× faster at small BitNet shapes but 2–3×
slower at K=4096 LLM shapes — the ~22 MB working set sits at the SLC boundary
and extra concurrency thrashes it (Pass 3). The moderate-N float GEMV gap is
a formally accepted limitation. Rule: integer-path geometry wins do not
transfer to float dequant paths.

**Sequence-parallel split-KV non-causal linear-attention kernel — REJECTED
(2026-07-01, Baseline classification).** One-simdgroup-per-(batch,head)
kernels present 16 simdgroups of work; the framework composition uses the
whole GPU per GEMM and cannot be beaten by a kernel that also needs
cross-threadgroup reduction scratch. Rule: serial per-(B,H) kernels lose to
whole-GPU composition — route to the framework or partition the sequence.

**fp16 LUT decoders — REJECTED (2026-07-01, Baseline classification decision
log).** The branchless bit-trick decoders are already exact and cheap; a LUT
adds memory traffic for no ALU win.

**matmul_custom small-shape routing — REJECTED (2026-07-01, Baseline
classification decision log).** These are TK-parity/calibration kernels;
mx.matmul is the practical dense GEMM. Do not spend routing complexity on
calibration kernels.

**attn_fwd D=64 16-row Q tile — REJECTED (2026-07-01, Pass 3).** At D=64 the
K/V stream is half the bytes and the doubled register footprint made the q16
variant ~1.4× slower. The q16 tile is a D=128-only win.

**Mamba2 backward Qᵀ-via-transpose — REJECTED (2026-07-02, Wave 3/4).**
Qt_c = Q_c^T exactly (verified bit-exact), so the second MMA + reverse-scan
pass is provably redundant — but the general (…,D,D) transpose is
scatter-bound and regressed seq-2048 ~30% (0.37→0.49 ms) for ~4.5% at
seq-4096. Rule: removing provable redundancy still loses if the replacement
access pattern is scatter-bound.

**LM-head fused GEMV as default — REJECTED (2026-07-02 Wave 3/4; re-measured
2026-07-03 Wave-8).** The fused per-lane GEMV reads W column-major and
re-reads it once per token: 2.7–3.9× slower than matmul+sampler at T∈{1,8},
and non-quant argmax_T8 is 0.38× of matmul+argmax. It is a T=1/quant-weight
specialization only; a W-amortizing tile rewrite is a deferred 6-kernel
project. Rule: never re-read the weight per token when a matmul reads it once.

**beam_build_copy_pairs compaction — REJECTED (2026-07-03, Wave-7).** The
fixed-slot emit is overhead-bound (~130 µs flat from 2k to 262k slots); a
scan + atomic-cursor compaction cannot beat the launch/eval floor and adds
contention plus nondeterministic ordering. Rule: overhead-bound kernels do
not benefit from doing less work.

**Cascade single-dispatch fusion — REJECTED (2026-07-03, Wave-7).** Host
concatenates are only 5–23% of cascade time, and the fused write would touch
the SHARED paged-attention partition kernels (v2/cascade/fp8/multi all route
through them) — a regression surface disproportionate to the win. Rule: do
not destabilize shared kernels for a single-path, sub-25% win.

**adamw vec4 — REJECTED (2026-07-03, Wave-8).** Neutral (0.29→0.29 ms).
AdamW is dominated by four f32 moment arrays whose scalar access is already
aligned/efficient; the bf16-access penalty that made dropout/gelu_bwd vec4
win does not exist for f32.

**f32 sampler-zoo vec4 — REJECTED (2026-07-05, Wave-9 opt pass).**
quadratic_transform vec4 won at T256 (0.31→0.16 ms) but repeatably regressed
at T1024 (0.70→0.80 ms), the regime that matters. Generalized rule (with the
adamw result): manual vec4 pays on bf16 I/O kernels; f32 strided loads are
already compiler-coalesced — leave them scalar.

**embedding_backward sorted segment-reduce as default — REJECTED
(2026-07-03, Wave-8).** The atomic scatter default is faster even at V=256
heavy duplication (0.11 vs 0.57 ms); argsort+segment overhead swamps any
contention saving. Rule: Apple's native atomic_float is fast — do not avoid
atomics on CUDA instincts (same lesson as the Wave-7 fused-dweight win).

**MoE dequant-to-shared at 1 warp — REJECTED (2026-07-05, Wave-9 K1).**
Barrier cost eats the span-decode win for the single-tile rect kernel (0.767
vs 0.716 ms); it only helped the 2-tile swiglu kernel.

**moe_lora_align port — CUT (2026-07-05, Wave-9 K12).** No QuixiCore
consumer for vLLM's LoRA-alignment metadata; porting it would ship dead code.
Deliberate scope cut, revisit only if a multi-LoRA MoE serving path lands.

**BitNet qgemm_bwd / gemm_v3 exposure — REJECTED (2026-07-07).** The source
project's own measurements show qgemm_bwd losing to torch.matmul on every
shape and gemm_v3 at 94–99% of MPS without beating it; ported later for
parity only, with an explicit no-speedup-claim decision.

**Direct kernels that lost to framework composition — REJECTED as defaults
(2026-07-13 and 2026-07-24).** Swin window attention (framework SDPA
3.38×/2.32× faster), head-major GQA decode, dynamic-width LayerNorm, fused
Q6_K LM argmax (vs qgemv+argmax), dense decode SwiGLU (0.64×), the realistic
block-2 spatial priority shape, canonical divisible patchify (0.21×/0.89×)
and 3-D temporal patchify (0.99×/0.86×), general audio conv1d (0.18×/0.23×),
long-memory cross-attention (0.04× at Tk=1500), and dense-grid coordinate
pooling (0.63×). All retain `use_kernel=True` escape hatches. Rule: pure data
permutations and shapes where the framework already saturates the GPU belong
to the framework; keep direct kernels only where a fusion or gather win is
measured.

**Whole-block q4_0 decode in parallel qgemv — REJECTED (2026-07-13,
Cross-kernel follow-ups).** 43% regression (0.0189→0.0271 ms): the LM-head
whole-block win does not generalize to parallel GEMV, where the widened
per-lane register footprint costs occupancy. Companion rejection: q8_0
one-block-per-lane in fused decode lost to its back-to-back generic control
(New-kernel second pass). Rule: whole-block decode wins in sequential
one-lane row dots, not in parallel GEMV.

**Runtime-selected tile width — REJECTED (2026-07-13, Cross-kernel
follow-ups).** Dynamic geometry inhibited Metal specialization (0.1700→
0.2443 ms); the 64-column tile variant was also rejected because S2
regressed. Rule: keep tile geometry compile-time.

**Launch-narrowing candidates from the 2026-07-13 passes — REJECTED
(measured).** 64-thread embedding launches (T256 −16%, bag B128 −32%),
128-thread spatial launch (regression, restored 256), 2-simdgroup cache
attention (~2× regression, restored 4 then superseded by 32), 8 patches per
threadgroup (−19%, lost parallelism), threadgroup input staging for spatial
(≤2.3%, not material), 16 SIMD-group attention partitions (worse than both 8
and 32 at every priority context), the 8→32-slot shared allocation while
launching eight groups (+60%), and the beam four-block vector decode
(0.2340→1.8109 ms). Keep the measured 8/32 specializations.

**NVFP4 16-row serial beam routing — REJECTED (2026-07-13).** Forcing 16
NVFP4 beam rows through serial row fusion regressed 0.9831→2.6393 ms; the
packed-QGEMM route stays above four rows. NVFP4 MoE at one/two warps also
rejected (rows32 +76% at one warp; two warps ≤1.7% and flat).

**MXFP4 row-fragment specialization and E8M0 bit reconstruction in MMA/MoE
decoders — REJECTED (2026-07-13).** Fragment-path scale reads are already
amortized by compiler CSE (M32 regressed; larger shapes under the 3% keep
threshold); bit reconstruction in the four-column MoE decoder traded a 7.9%
rows512 win for a repeatable 31.6% rows32 decode regression. E8M0 bit
reconstruction pays only in fp32 span/sequential paths. Native half exp2
stays in the matrix paths.

**Global E4M3 LUT — REJECTED, narrowed to masked projection (2026-07-14,
MXFP8).** The global LUT regressed QGEMV to 0.0552/0.1576 ms and
candidate/MoE large shapes; only the path-local 256-entry table inside masked
LM-head projection won (−33%). Rule: LUTs must be path-local — a global
table thrashes the caches the hot loops need.

**MXFP8 attention candidates — ALL REJECTED (2026-07-14).** Scale broadcast
(+9%), direct register V decode (flat), two-warp (0.738 vs 0.479 ms), staging
depths 1/2 (flat vs kept depth 4), and the split-plane 32-scale/1024-code
layout (N11008 +6.7%, K%1024 constraint — layout removed). The remaining
quantized-attention gap is the attention schedule, not scale decode.

**MXFP8 sequential complete-block decode everywhere — REJECTED except sparse
projections (2026-07-14).** Decode epilogue 0.0209→0.0221 ms, SwiGLU
0.0334→0.0347, LM top-k T1 0.354→0.372; only masked/candidate projection
improved and only those specializations were retained. The explicit 8-value
half decoder was also rejected (top-k T1 0.354→0.527 ms) — the
compiler-generated generic path is better. Gate/up interleaved decode and
vector E4M3 decode4 also rejected; the four-warp×32-column QGEMM lost to
2×32 on breadth; combined E8M0+E4M3 exponent reconstruction regressed sparse
projection by an order of magnitude.

**FP8 (E4M3) QGEMV whole-block decode — REJECTED (2026-07-14).**
N4096/N11008 0.03041/0.10581→0.03188/0.10952 ms — the whole-block trick that
wins for MXFP4/MXFP8 loses for byte-code FP8 (see Open contradictions).

**FP8 MLA scale candidates — REJECTED (2026-07-14).** Grouped scale loads
flat/noisy (0.674→0.663→0.676 on repeat); lane-0 scale broadcast +36.7%
(0.663→0.906 ms). MLA scale reads are not the bottleneck. The later MLA E8M0
per-value bit decode was also flat/worse (0.6735→0.6805).

**FP8 quantized-attention topology candidates — REJECTED (2026-07-14).**
Eight warps flat (0.5046→0.5032 ms); D128 Q16 row tile +19% — the q16 tile
that wins dense D=128 attention does not transfer to the quantized path.
The FP8 MoE decoder-scale shuffle after the two-warp route regressed rows512
(1.386→1.511 ms); keep the two-warp topology only.

**Bit-built power-of-two producer scales — REJECTED (2026-07-14,
Cross-kernel FP8 transfer).** Scale selection runs once per 64–128 values, so
bit tricks lost (quant group 0.0402→0.0501 ms). Rule: bit-trick
encode/decode pays per-element, never per-group.

**Compile-time symbols for cheap uniform branches — REJECTED (2026-07-14,
Cross-kernel FP8 transfer).** KV compile-time format symbols (mixed, scatter
regressed substantially), per-token activation-mode symbols (large shapes
flat, templating perturbed the standard path), and attention softcap symbols
(uniform tile branch already cheap) all failed. The exception that proves the
rule: the grouped activation path re-evaluates the branch per 64-value group
and its specialization won (−3 to −10%). Rule: specialize inner-loop branches
only. KV scatter one-head-per-warp also rejected (flat to worse).

**MXFP8 decode SwiGLU two-warp — REJECTED (2026-07-14, Cross-kernel FP8
transfer).** 0.0320→0.0339 ms: output-channel parallelism already saturates
the GPU at B1 decode; the two-warp topology only wins where the schedule
starves (grouped MoE SwiGLU).

**FP8 fake-quant bit exponent/step — REJECTED (2026-07-14, Cross-kernel FP8
transfer).** The realistic large shape was unstable/regressed
(0.2348→0.2778 ms on repeat). Related: the first E4M3 bit-encoder rounding
constant produced 23 correctness failures before the half-ULP constant was
corrected (FP8 hot-path entry) — validate encoders bit-exact before timing.

**BaseQN direct decode-GEMM for M>1 — REJECTED (2026-07-23).** 6.20–6.24 ms
at Q4 N4096 K4096 M2/M8 vs 0.89 ms for materialize-then-framework-GEMM.
Route every M>1 through dequant + framework GEMM.

**BaseQN combined QKV grid at long K — REJECTED (2026-07-23).** At
Nq4096/Nkv1024/K4096 the one-grid kernel lost 2.6% to three direct GEMVs;
fusion pays only while fixed dispatch cost dominates decode bandwidth. The
retained crossover is K≤1024.

**BaseQN dedicated LM-head reductions — REJECTED, both, and removed
(2026-07-23).** The 8-simdgroup and serial-simdgroup packed reductions were
0.80–0.81× of the columnwise direct-GEMV + argmax composition at batch 2/4
and only parity at batch 1. The decisive baseline is the strongest
composition, not the weakest GEMM route.

**BaseQN split-K for the rectangular expert projection — REJECTED
(2026-07-23).** 0.90× (0.2489→0.2766 ms); split-K is kept only for the fused
expert SwiGLU where it won 2.31×. Launch geometry is operation-specific.

**Q8_0 KV fast-FP encoder — REJECTED (2026-07-23).** Reciprocal-sensitive
BF16 half ties produced 3–7 one-code mismatches per 8192 values; Metal
safe-FP mode applied to the encoder makes all code/scale planes exact.
Rule: serialized cache determinism outranks an unmeasured arithmetic
shortcut. Also recorded: the codec's bulk scatter/gather is slower than
dense BF16 copies (0.69–0.82×) — the format is kept for footprint and the
direct paged-read win, with no codec speed claim.

## Patterns and generalized rules

- Staging through threadgroup memory loses unless measured reuse beats
  barriers + occupancy loss; the cache already serves cross-thread reuse
  (2026-07-01/02, Pass 2 mla_decode — third confirmation across gemm_staged,
  gqa_staged, MLA).
- Per-call-sync timing has a ~0.15–0.25 ms floor and cold clocks bias the
  first-timed thunk; always re-verify a gap on the fixed harness (1 s clock
  ramp, ≥50 ms per-thunk warmup) before optimizing — queue items #1, #12,
  #14 were artifacts (2026-07-01, Pass 3 decision log).
- Pipelined-throughput timing can completely mask single-call decode reality
  (0.6–1.2 ms pipelined vs 2.7–5.5 ms single-call); measure decode kernels
  in the single-call regime too (2026-07-01, Pass 3 mla_decode_fp8).
- Manual vec4 pays on bf16 I/O kernels (scalar bf16 access is the
  bottleneck), not on f32 kernels (already compiler-coalesced) (2026-07-05,
  Wave-9 — optimization pass over the gap-port kernels; Wave-7 gelu_bwd,
  Wave-8 dropout/adamw).
- Apple's native atomic_float is fast: prefer atomic scatter/fused atomic
  accumulation over sort/multi-pass structures (2026-07-03, Wave-7 close-out
  #1; Wave-8 embedding_backward).
- Whole-block dequant wins in sequential one-lane row dots (LM-head, decode
  epilogues) and loses in parallel GEMV where it costs registers
  (2026-07-13, Cross-kernel follow-ups and optimization pass).
- Unpack the quant scale once per span/block; per-element exp2 on e8m0
  formats is an ALU bottleneck — but only in fp32 span/sequential paths;
  fragment paths are already CSE-amortized (2026-07-05 Wave-9 K1; 2026-07-13
  MXFP4 pass).
- Compile-time specialization pays only for branches inside inner loops;
  uniform per-tile branches are already cheap and extra symbols perturb
  codegen (2026-07-14, FP8 + Cross-kernel FP8 transfer experiments).
- Keep tile geometry compile-time; runtime-selected geometry inhibits Metal
  specialization (2026-07-13, Cross-kernel follow-ups).
- Serial one-simdgroup-per-(B,H) kernels lose to whole-GPU framework
  composition; either route to the framework or partition the
  sequence/context axis (2026-07-01 linear_attn; 2026-07-13 attn_decode_bh).
- Lookup tables must be path-local; a global LUT thrashes cache
  (2026-07-14, MXFP8 inference hot-path experiments).
- Quantized prefill GEMM is parity-at-best with fp16 matmul on this
  hardware — its value is memory footprint; quant wins live in decode GEMV
  (2026-07-01 qgemm; 2026-07-05 Wave-9 K1).
- Fusion wins by removing a device round-trip or a launch; it does not win
  when output-channel parallelism already saturates the GPU (2026-07-22
  qgemv_fused; 2026-07-14 Cross-kernel FP8 transfer, decode SwiGLU 2-warp).
- Sub-0.05 ms cases are queue-scheduling sensitive: decide from p20/p80
  bands and independent repeats, never a single median (2026-07-13, Packed
  embedding … pass; multiple BaseRT entries).
- Separately compiled metallibs (MLX vs torch) diverge on fast-math
  subnormals and borderline rounding: keep FTZ-safe decode selects, safe-FP
  exact encoders, and off-by-one parity tolerances for codes (2026-07-01
  qgemv E1; 2026-07-23 Q8_0 KV codec).
- Load width beats rows-per-simdgroup restructuring for butterfly/row
  kernels — the hadamard fix was 16-byte loads, not more rows per lane
  (2026-07-01, Pass 3 hadamard).
- Overhead-bound kernels (sub-30 µs, flat across sizes) cannot be improved
  by reducing work; the launch/eval floor dominates (2026-07-03, Wave-7
  beam_build_copy_pairs; Wave-8 spec kernels).
- Mask/route first, then touch packed weights: inspecting the bitmask before
  reading weight rows was worth 11–70% (2026-07-13, New-kernel second pass).
- Validate encoders bit-exact against a numpy twin before any timing; a
  wrong rounding constant cost a full re-run (2026-07-14, FP8 inference
  hot-path experiments).

## Open contradictions

- Staged paged-attention decode: pipelined timing says staged is 1.80×
  FASTER than v1 (2026-07-01, Baseline classification, serving
  re-measurement); the single-call A/B says staged is ~6% SLOWER than v1
  (2026-07-01, Pass 2, serving sweeps). v2 wins both regimes 3.6–4.2×, so
  defaults stand, but the staged-vs-v1 ranking is regime-dependent and a
  real decode loop (one call per step, no pipelining) sits between the two
  measured regimes. Resolve with a real decode-loop A/B.
- Whole-block QGEMV decode is kept for MXFP4 (−18.7/−31.4%, 2026-07-13) and
  MXFP8 (−3.7/−4.4%, 2026-07-14) but was rejected for FP8 E4M3 (+3–5%,
  2026-07-14, FP8 inference hot-path experiments). The format asymmetry
  (sub-byte/e8m0-scaled vs byte-code) is unexplained. Resolve with a
  same-session instruction-mix profile or controlled cross-format sweep.
- MoE E8M0 bit reconstruction: −7.9% at SwiGLU rows512 but +31.6% at rows32
  decode (2026-07-13, MXFP4 pass) — a shape-routed prefill-only decoder
  might capture the win; needs a measurement that preserves the small-row
  path.
- QGEMM 2×32 shared-weight staging: kept for MXFP8 at M%64==0 (1–3%,
  2026-07-14, MXFP8 hot-path) but flat/rejected for standard FP8 GEMM
  (2026-07-14, FP8 hot-path). A same-session cross-format A/B would resolve
  whether the win is format-specific or noise.
- Hardware era split: every absolute number through 2026-07-14 is Apple M4
  Max; from 2026-07-22 Apple M5 Max. No kernel has been re-measured across
  the switch, so any M4-era routing threshold (partition_size 256, attn_q
  multiwarp auto, decode crossovers, the SLC-boundary reasoning in the float
  GEMV rejection) is unverified on M5 Max. Resolve by re-running the quick
  preset baseline on the current machine before trusting M4-era thresholds.
