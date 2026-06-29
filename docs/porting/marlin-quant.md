# Brainstorm: implementing the ThunderKittens kernels with Marlin's methods on Apple

> **Status:** complete — two substrate dequant primitives + **29 weight formats**, each on three
> kernels (`qgemm` dequant-direct-to-fragment prefill, `qgemv` decode reduction, `qflux_gelu` fused),
> dual-backend (MLX + PyTorch MPS), validated kernel-vs-`dequantize(Wq)@x`. Plus an integer-activation
> decode path (`qgemv_w8a8`/`qgemv_w2a8`) and a GPTQ act-order reordering layer. The dequant substrate
> is `include/.../tile/dequant.metal` (per-format structs + `dequant_into_shared` / the zero-shuffle
> `dequant_into_register`); the lattice/codebook tables are auto-generated into `dequant_tables.metal`
> (+ `tk/quant_tables.py`) from ggml-common.h. Host quant + `QUANT_FORMATS` registry in `kernels/tk/quant.py`.
>
> The 29 formats: integer **q8_0, q4_0, q4_1, q5_0, q5_1, q4_K, q5_K, q6_K, q2_K, q3_K, kU4B8 (GPTQ),
> kU4 (AWQ), hqq**; codebook/i-quant **iq4_nl, iq4_xs, iq2_xxs, iq2_xs, iq3_xxs, iq1_s**; float
> **fp8_e4m3, e5m2, fp8_block, fp4_e2m1, mxfp8, mxfp4, mxfp6_e3m2, mxfp6_e2m3, nvfp4**; ternary
> **bitnet**. (Full per-phase detail below.)
> Phase 6 (retrofit) demonstrated: `kernels/qflux/` — `qflux_gelu` = gelu(dequantize(Wq)@X + bias),
> the dequant-direct-to-fragment path (zero-shuffle, like qgemm) + flux bias+GELU epilogue, all formats, dual-backend.
> **Phase 5 DONE and is now the default `qgemm` path:** dequant-direct-to-fragment (Marlin
> zero-shuffle) — `dequant_into_register` writes the dequantized weight straight into the
> `simdgroup_matrix` register slots (using the substrate's lane→(row,col) fragment map), skipping the
> threadgroup tile + barrier. **Bit-identical to the staged path and ~40% faster** (q4_0, 2–8K shapes)
> — the first multi-simdgroup-style optimization that actually wins on Apple, because quantized GEMM
> is weight-bandwidth-bound. No offline weight repack needed (per-lane gathered dequant). `qflux_gelu`
> uses the same zero-shuffle path.

> **Deferrals all paid down** (see `thunderkittens.md` for the per-item detail): production-grade
> ggml encoders (incl. the E8-lattice i-quants); quantized-KV attention (`kernels/attn_q/`, non-causal
> + causal + multiwarp); integer-accumulate prefill (`kernels/qgemm_int/`, exact int32 — ~7–10× slower
> than the half MMA, for numerics not speed); fp8_block2d (separate 2D tile scale); and GPTQ act-order
> in-kernel g_idx gather. 875 Python + 126 Xcode tests.

> **W·A8 (parity).** Beyond weight-only (W·A16), the activation-quantized schemes work via
> `tk.qmm(wq, x, w_format, act=...)`: `act="int8"|"fp8"` snaps activations to the 8-bit grid
> (`tk.quant.quantize_act_int8/_fp8`, per-token) then runs the existing dequant-to-half GEMM —
> reproducing **fp8 W8A8, int8 W8A8, int8 W4A8** fake-quant numerics. Apple has no int8/fp8 matmul,
> so this is parity, not speed. W4A16 / W8A16 are just `act=None`.

> **Fan-out [Phases 3–5, complete].** Phase 3 — GGUF k-quant + legacy: **q2_K, q3_K, q5_K, q6_K**
> (256-superblock; q3_K's 16 6-bit signed scales via kmask packing, q5_K reuses q4_K's
> get_scale_min_k4, q6_K int8 scales) and **q4_1, q5_0, q5_1** (legacy 32-block; `_1` asymmetric
> d*q+m; q5_* 5th bit in a qh uint32). Phase 4 — float sub-formats: **e5m2** (fp8 1-5-2),
> **fp8_block** (128×128 block-scaled e4m3, tile scale replicated per row), **mxfp6 e3m2/e2m3** (e8m0
> scale + 6-bit codes packed 4-per-3-bytes). Phase 5 — layout/indexing: **hqq** (int4+zp, group 64;
> thin kU4 variant) and **GPTQ act-order** (`tk.qgemm_actorder`: gather activations by the g_idx
> permutation, then the standard qgemm — a load-time reordering, not a new format). All on
> qgemm(frag)/qgemv/qflux, dual-backend, validated vs dequantize(Wq)@x. **29 weight formats total.**

> **Codebook / LUT dequant + GGUF i-quant family [Phase 2, complete].** The second new dequant
> style: packed bits index a constant table instead of bit arithmetic. Six formats: **iq4_nl**
> (16-entry non-linear int4 codebook), **iq4_xs** (256-superblock iq4_nl, 6-bit sub-scales), and the
> E8-lattice quants **iq2_xxs / iq2_xs / iq3_xxs / iq1_s** (grid-table lookup + ksigns sign bits +
> per-block scale). The grid/sign tables (iq2xxs 256, iq2xs 512, iq3xxs 256, iq1s_grid_gpu 2048,
> ksigns 128, kmask 8) are auto-generated from ggml-common.h into `dequant_tables.metal` (Metal) and
> `tk/quant_tables.py` (numpy) — not hand-transcribed. Kernel decoders mirror the ggml-metal
> dequantize_* functions exactly; the encoders pick the nearest grid entry per group (kernel-vs-oracle
> isolates quant quality). All on qgemm/qgemv/qflux, dual-backend, validated vs dequantize(Wq)@x.

> **Integer dot path (idot/imma) — true int8 accumulate [Phase 1].** `idot4` (dp4a equivalent: 4
> packed int8 → int32, modeled on BitNet's `__dp4a`/`decode_i2s_to_i8s`) powers a per-lane int8 GEMV
> + `simd_sum`. Two decode kernels in `kernels/qgemv_int/`: **`qgemv_w8a8`** (SmoothQuant — int8
> weight per-channel scale × int8 act per-token scale; int32 accumulate, scale once at the end) and
> **`qgemv_w2a8`** (BitNet ternary 2-bit × int8 act; per-group int32 sums × absmean scale). `q8_0`
> and `bitnet` gained `code()`/`gscale()` int accessors; host `quantize_w8a8` + `quantize_act_int8`.
> **Decision — integer PREFILL stays on the dequant-to-half MMA**: Apple has no int8 matrix unit, so
> a hand-rolled tiled int dot would lose to the half tensor cores, and W8A8 prefill parity already
> exists via `qmm(act="int8")`. The int primitive is the decode (batch-1, memory-bound) regime only.
> **Numerics differ by design:** decode = exact int32 accumulate scaled once; prefill = fp16
> dequant-both-to-half (fp16 accumulation error). Validated against the INTEGER oracle
> `(W_int8 @ x_int8)·w_scale·a_scale`, NOT each other — the decode-vs-prefill gap at M=1 is correct,
> not a bug. **Benchmark (the primitive's justification):** int8-path vs dequant-to-half decode at
> the same (N,K) — int8 is FASTER on the realistic large shapes (+7–20%, incl. K=11008 FFN and
> 32000-vocab projections) and more accurate everywhere; ~10% slower only on the smallest square
> (4096²). Verified, not assumed.

## The correction

I previously parked the quantized-GEMM family (`gemm/{fp8_*, int8_*, mxfp8_*, nvfp4_*}`) as
"N/A on Apple — no fp8/int4 `simdgroup_matrix` path." **That was wrong.** It assumed you need
*native low-precision tensor cores*. Marlin doesn't use those either — its method is:

> store weights quantized + per-group scales → **dequantize to fp16 in registers with an IEEE
> bit-trick** → feed a *standard* fp16 tensor-core MMA → accumulate in fp32 → apply scale.

Apple has no native low-precision matmul, but it has `simdgroup_matrix<half,8,8>` — so
"dequant-to-half then standard MMA" **is** the implementation, not a workaround. And the dequant
bit-tricks are defined by the IEEE-754 fp16/bf16 layouts, which Metal `half`/`float` + `as_type`
honor identically. So the magic constants carry over **verbatim**. Two references confirm it:

- **Marlin** (`.reference/vllm/.../marlin/`) — the method (below).
- **vLLM-Metal** (`.reference/vllm-metal/.../kernels_v2/`) — proves fp8/int/affine/codebook
  (de)quant is already done in **pure MSL** (`float8.metal`, `turboquant.metal`), and that the
  `simdgroup_matrix` fragment↔register bridge needed to wire quant→MMA exists
  (`pagedattention_tiled.metal`). vLLM-Metal just never connected the two — which is the gap we fill.

## Marlin's method, distilled (what's portable)

| Ingredient | Portable? | Apple form |
|---|---|---|
| Weights stored quantized (int4/int8/fp8/fp4) + per-group fp16 scales (± zero-point) | ✅ algorithm | same |
| **In-register dequant via IEEE bit-tricks** (`dequant.h`) | ✅ **verbatim** | `half` is IEEE fp16; `lop3(q,mask,ex)`→`(q&mask)\|ex`; `prmt`→shift/mask or `as_type<uchar4>` |
| Offline weight **repack** so dequant lands straight in MMA fragment slots (zero shuffles) | ✅ technique | re-derive the permutation for Apple's 8×8 simdgroup fragment (or skip via dequant-to-shared) |
| Dequant → fp16 fragments → MMA → fp32 accumulate → ⊙ per-group scale | ✅ structure | `simdgroup_multiply_accumulate` (half×half→float), scale after dequant |
| 4-stage `cp.async` software pipeline + register double-buffer | ⚠️ pattern only | no `cp.async` → cooperative threadgroup multi-buffering (our `gemm_staged` already does this) |
| `ldmatrix`, `mma.sync m16n8k16`, `lop3`, `prmt`, lock-reductions | ❌ NVIDIA | `simdgroup_load`, 8×8 `simdgroup_multiply_accumulate`, plain bit-ops, `atomic_*`+`threadgroup_barrier` |

### The dequant kernels we copy (IEEE constants → valid on Metal `half`)

- **int4→fp16** (`kU4B8`, GPTQ symmetric): `lo=(q&0x000f000f)|0x64006400`; `hi=(q&0x00f000f0)|0x64006400`;
  `frag0 = lo_half2 − 0x64086408`; `frag1 = hi_half2*0x2c002c00 + 0xd480d480`.
  `0x6400`=fp16(1024); OR-ing nibble `v` gives `1024+v`; subtract `1032` → `v−8`. The `−8` (B8 bias)
  is folded into the constant. AWQ asymmetric = same minus a separately-dequantized zero-point.
- **int8→fp16** (`kU8B128`): byte-permute each int8 into the low byte of an fp16 lane with high byte
  `0x64` → `1024+byte`; subtract `0x6480` (1152) → `byte−128`.
- **int8→bf16**: bf16 has only 7 mantissa bits, so go via fp32 base `0x4B000000`(2²³), subtract, take
  high 16 bits. (The one case needing fp32.)
- **fp8 e4m3→fp16**: relocate sign bit, right-shift (exp+mantissa) by `5−4=1`, multiply by
  `2^(15−7)=2^8` to fix the exponent bias. *No subtract — it's a format widening.*
- **fp4 e2m1→fp16**: same shape, shift by `5−2=3`, multiply by `2^14`.
- **mxfp8 / nvfp4 block scales** (`kFE8M0`): the block scale is itself an fp8/e8m0 that gets
  dequantized (`dequant_fp8_scales`) before applying — also pure bit-ops.

vLLM-Metal's `float8.metal` already has e4m3/e5m2 ↔ float in MSL (field-extract + `ldexp` for decode,
RNE for encode), and `turboquant.metal` already has affine int4/int8 and Lloyd-Max codebook dequant in
MSL — so we have reference implementations of every piece on Apple.

## The ThunderMittens design — one new substrate primitive

Add a **quantized-tile dequant op** that mirrors how `complex_mma` and the existing tile loads slot in.
Two flavors, simple-first:

**(1) Dequant-to-shared (recommended first — reuses everything).** Avoids deriving Apple's fragment
permutation. In a `gemm_staged`-style loop:
```
load packed quant bytes (device) ──▶ dequant in registers (IEEE bit-trick) ──▶ write half into a
threadgroup st<half,BK,BM> tile (natural row/col order) ──▶ simdgroup_load → rt<half> ──▶ mma_AB
```
The existing `st` shared tiles + `load(rt, st)` already handle the fragment layout, so weights stay in
a *natural* packed layout (no offline repack). One extra shared round-trip — which `gemm_staged`
already pays anyway. Per-group scale: store scales sized to the simdgroup (32) so `simd_*` reductions
are free, multiply after dequant.

**(2) Dequant-to-fragment (the Marlin optimization, later).** Re-derive the `pack_idx` permutation for
Apple's 8×8 B-fragment lane→element map (it's documented in `pagedattention_tiled.metal:39-58`, and
encoded in our own `make_causal` bitmasks / `transpose` code), repack weights offline, and dequant
straight into `rt_base::thread_elements()` via the `reinterpret_cast<thread vec2&>(...)` bridge — zero
shuffles. This is the speed version.

New pieces, all small and localized:
- `include/.../dequant.metal` — `dequant_u4b8/_u4/_u8b128/_fp8_e4m3/_fp4_e2m1(...)` returning `half`s
  (constants verbatim from Marlin; `lop3→(q&m)|ex`). ~the size of the `complex_mma` addition.
- a packed weight gl/storage convention (uint32 words; group scales as `half`).
- `kernels/qgemm/` — the staged quantized GEMM kernel + dual-backend wiring + tests (validate vs
  `dequant_in_python(weights) @ x` and vs `mx.matmul` of the dequantized reference).

## How it implements the parked kernel families

| TK file(s) | Marlin-method implementation on TM | Validate vs |
|---|---|---|
| `gemm/int8_b200` (int8 weight) | int8→fp16 dequant + staged MMA | dequant-ref @ x |
| `gemm/fp8_h100`, `fp8_h100_scaled`, `fp8_b200` | fp8 e4m3→fp16 dequant + MMA + scale | dequant-ref @ x |
| GPTQ/AWQ int4 (the canonical Marlin) | int4 `kU4B8`/`kU4` dequant + per-group scale + MMA | dequant-ref @ x |
| `gemm/mxfp8_*` | fp8 mantissa + e8m0 block-scale dequant | dequant-ref @ x |
| `gemm/nvfp4_*` | fp4 e2m1 + block scale dequant | dequant-ref @ x |
| **any existing kernel** (`flux`, `attn_*`, `linear_attn`, `mamba2`) with quantized weights | swap the weight `load` for `load_dequant` — the dequant tile drops in wherever an `rt` feeds an MMA | the fp16 version |
| quantized-KV attention | `turboquant`-style int8/affine KV dequant before the `attn_fwd` QK/AV MMAs | SDPA on dequant-ref |

The same `dequant` op is the building block for **all** of them — exactly like `complex_mma` unlocked
`fftconv`. "Implement all TK kernels with Marlin's methods" ≈ *add the dequant-tile primitive, then any
kernel can take quantized inputs.*

## Why this matters on Apple specifically (the real perf story)

My earlier perf-tuning finding was that dense multi-simdgroup staging doesn't beat the simple kernels
because those shapes are compute/cache-bound. **Weight-only quantization is the opposite regime** —
LLM decode GEMMs are *memory-bandwidth-bound on the weights*, and Apple's unified memory bandwidth is
the bottleneck. Shrinking the weight bytes 4–8× (int4/fp8) is a genuine, large win on Apple, *more*
impactful than any dense-tile tuning. So Marlin's method isn't just "unblock the parked kernels" — it's
plausibly the highest-value performance work for real on-device LLM inference.

## Honest re-scoping of "N/A"

- **No longer N/A (this brainstorm):** `fp8`, `int8`, `int4`, `nvfp4`, `mxfp8` GEMM — all implementable
  via dequant-in-register (Marlin's method). The "emulation vs native tensor-core" distinction is moot
  on Apple, which has no native low-precision matmul — dequant-to-half *is* the kernel.
- **Still genuinely hardware-N/A:** `parallel/*` multi-GPU collectives (`all_reduce`, `ag_gemm`,
  `ring_attn`, …) — a single Apple GPU has no multi-device fabric. Unchanged.

## Bonus: vLLM-Metal kernels beyond TK (Apple-proven, worth porting later)

- `gdn_linear_attention` — gated delta-net linear attention (register-only state + `simd_sum`).
- `mla` — multi-head latent attention (paged decode).
- `pagedattention(_tiled)` — KV-cache FlashAttention-2 with 8×8 simdgroup MMA (the dense MMA reference
  for how Apple wants Q/K/V fragments laid out — also the source for deriving our fragment permutation).

## Recommended first step

Per the Marlin spec's own recommendation: **int4 `kU4B8` → fp16, per-group scales, no act-order, no
zero-point**, via the dequant-to-shared path (flavor 1). Concretely: add `dequant_u4b8`, build
`kernels/qgemm/` on the `gemm_staged` skeleton, validate the output equals `(dequant(W)) @ x`
(fp32 reference) and matches `mx.matmul` within bf16/group-scale tolerance. Then add fp8 e4m3, then
int8, then the block-scaled fp4/mxfp8 — each is just another `dequant_*` routine into the same kernel.
