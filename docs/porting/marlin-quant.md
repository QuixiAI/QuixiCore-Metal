# Brainstorm: implementing the ThunderKittens kernels with Marlin's methods on Apple

> **Status:** foundation landed — `include/.../tile/dequant.metal` (dequant primitive),
> `kernels/qgemm/` (dequant-to-shared → simdgroup MMA, prefill) and `kernels/qgemv/` (batch-1 decode
> reduction) on the **q8_0** format, dual-backend, validated vs `dequantize(Wq)@x`. Host quant in
> `kernels/tk/quant.py`. Remaining: fan out formats (q4_0/q4_K/int4-Marlin, fp8/fp4/mxfp8/nvfp4),
> then dequant-direct-to-fragment + retrofitting flux/attention.

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
