# ThunderMittens — performance status

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
