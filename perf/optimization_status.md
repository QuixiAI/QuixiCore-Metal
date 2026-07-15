# QuixiCore Metal Optimization Status

This is the running notebook for Metal kernel implementation and optimization.
Raw output belongs under `perf/results/`; stable conclusions belong here.

## Entry Template

Use this structure for every kernel family or optimization pass:

```text
## YYYY-MM-DD: <kernel or pass name>

Status: not started | baselining | experimenting | candidate | landed | deferred.
Current implementation:
Current public route:
References inspected:
Correctness:
Baseline:
Experiments:
Decision:
Open questions:
Raw results:
```

Record enough context to reproduce the run: Apple Silicon model, macOS version,
Xcode/Metal toolchain version, integration path, command, git commit or
working-tree label, dtype, shape, quant format, warmups, iterations, median,
variance, correctness tolerance, and observed error.

## 2026-07-13: MXFP4 inference coverage and hot-path pass

Status: candidate; retained implementations have passed focused benchmarks,
repository-wide correctness/parity validation, both backend builds, and the
Xcode test build.

Current implementation:

- MXFP4 is available in fused decode epilogue/SwiGLU, LM-head
  argmax/categorical/top-k/top-p sampling, packed-mask and CSR-candidate output
  projection, and exact beam advance on MLX and PyTorch MPS.
- QGEMV has an MXFP4 whole-block kernel: one lane consumes the 32 weights behind
  one E8M0 scale instead of invoking four 8-value span decoders per block.
- Decode/SwiGLU and LM-head sequential dots likewise consume complete 32-value
  blocks. Decode and sparse projection keep scale/code products in fp32 where
  their public contract requires one final rounding; the sampler preserves its
  established half-rounded dequant contract.
- `tk_e8m0_decode_f32` reconstructs E8M0 powers of two directly from IEEE-754
  exponent bits, including the code-zero subnormal and code-255 infinity cases.
  The float 8-value decoder used by packed embedding lookup and the retained
  complete-block paths use it instead of a transcendental `exp2`.
- Generic half fragment decode remains unchanged for QGEMM/QFlux. The
  four-column MoE decoder also retains native half `exp2`; controlled variants
  did not improve their full priority shape sets.

Current public route:

- Packed decode and LM-head operations dispatch directly to their MXFP4 Metal
  instantiations. Beam advance uses row-wise no-logits fusion for at most four
  rows and the existing packed QGEMM route above four rows when the vocabulary
  is matrix-tile aligned.
- MXFP4 QGEMV dispatches to the complete-block kernel. QGEMM, QFlux, and
  quantized MoE retain their previous launch geometry and generic decoders.

References inspected: the repository's existing MXFP4
`{E8M0 scale, 16 packed E2M1 bytes}` contract, q4_0 complete-block kernels, and
the preceding NVFP4 inference pass. No external implementation code was
imported.

Environment and method:

- Hardware/toolchain: MacBook Pro Mac16,5, Apple M4 Max, 128 GB; macOS 26.5.1
  (25F80); Xcode 26.6 (17F113); Metal 32023.883 / Metal toolchain 17.6.109.0;
  Python 3.12.9; MLX 0.21.1; PyTorch 2.12.1 MPS.
- Working-tree label: `3cab797-dirty`.
- Performance integration path: MLX Python extension, format `mxfp4`, fp16
  QGEMV/QGEMM/QFlux/MoE and embedding inputs, and fp32 fused decode/LM-head
  inputs. The harness performs its clock ramp and at least 50 ms of per-thunk
  warmup, adaptively batches calls to at least 2 ms per sample, synchronizes
  each sample, and reports per-call median, p20/p80, and CV.
- All focused runs requested 10 warmups and 40 measured samples. The initial
  and final commands were:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset quick \
  --kernel qgemv,qgemm,qflux,moe_q --formats mxfp4 \
  --warmup 10 --iters 40 \
  --out-dir perf/results/2026-07-13/mxfp4-inference-baseline
.venv/bin/python perf/bench_kernels.py --backend mlx --preset quick \
  --kernel qgemv,qgemm,qflux,moe_q,decode_linear_epilogue,decode_swiglu,lm_head_q,lm_head_masked,lm_head_candidates,lm_head_beam,quantized_embedding \
  --formats mxfp4 --warmup 10 --iters 40 \
  --out-dir perf/results/2026-07-13/mxfp4-inference-final
```

Correctness and validation:

- `scripts/build kernels` and `scripts/build pytorch_mps` passed.
- `scripts/test correctness -q`: 2110 passed.
- `scripts/test parity -q`: 420 passed, including MXFP4 decode, sampling,
  top-p, sparse projection, and beam parity.
- `scripts/test mps -q`: 472 passed.
- `scripts/test xcode`: test build succeeded for the shared primitive target.
- The final benchmark observed relative errors of `8.23e-7` and `8.94e-5`
  for QGEMV, `2.32e-7` for decode epilogue, `2.07e-7` for SwiGLU, and zero
  selected-id error for masked/candidate projection. Structured sampling and
  beam outputs are covered by exact-id tests.

Retained QGEMV results compare the original generic 8-value-span kernel with
the final whole-block route. Times are milliseconds; brackets contain p20/p80,
followed by CV.

| Shape | Original | Final | Change | Final packed-weight GB/s |
|---|---:|---:|---:|---:|
| N4096 K4096 | 0.0331 [0.0325/0.0340], .0621 | 0.0269 [0.0261/0.0284], .0777 | -18.7% | 331 |
| N11008 K4096 | 0.0786 [0.0775/0.0857], .0567 | 0.0539 [0.0515/0.0570], .0831 | -31.4% | 444 |

The new fused-operation controls used the simplest correct generic MXFP4
integration before complete-block specialization. Final values include the
retained complete-block and E8M0 bit-reconstruction changes.

| Path / shape | Generic control ms | Final ms [p20/p80], CV | Change | Error / check |
|---|---:|---:|---:|---:|
| Decode epilogue B1 K1536 N4096 | 0.0240 | 0.0191 [0.0187/0.0195], .0736 | -20.4% | 2.32e-7 rel |
| Decode SwiGLU B1 K1536 N4096 | 0.0373 | 0.0292 [0.0279/0.0341], .1261 | -21.9% | 2.07e-7 rel |
| LM-head top-k T1 V32000 K4096 | 0.5141 | 0.2754 [0.2722/0.2807], .0348 | -46.4% | exact selected ids |
| LM-head top-k T8 V32000 K4096 | 2.1984 | 1.2691 [1.2586/1.2952], .0198 | -42.3% | exact selected ids |
| Masked T1 V8192 K1024 legal256 | 0.0834 | 0.0313 [0.0303/0.0340], .0969 | -62.4% | exact ids/log-probs |
| Masked T8 V8192 K1024 legal64 | 0.0564 | 0.0429 [0.0398/0.0462], .0946 | -24.0% | exact ids/log-probs |
| Candidates T1 V8192 K1024 C256 | 0.0574 | 0.0366 [0.0356/0.0379], .0901 | -36.3% | exact ids/log-probs |
| Candidates T8 V8192 K1024 C64 | 0.0185 | 0.0162 [0.0160/0.0169], .1085 | -12.3% | exact ids/log-probs |
| Beam B1 BM4 V32000 K4096 | 1.2039 | 0.7713 [0.7674/0.7852], .0215 | -35.9% | exact token/parent |
| Beam B4 BM4 V32000 K4096 | 0.9835 | 0.9739 [0.9655/0.9906], .0162 | flat | exact token/parent |

Controlled experiments and decisions:

| Factor | Priority control | Candidate / repeat | Decision |
|---|---:|---:|---|
| Packed embedding E8M0 `exp2` -> bit reconstruction, T1/T256 R8192 D1024 | 0.01169 / 0.02205 ms | 0.00959 / 0.02182 ms | Keep; T1 improves 17.9%, T256 is flat, outputs exact. Candidate p20/p80 were 0.00934/0.01032 and 0.02112/0.02263 ms. |
| LM-head generic spans -> whole 32-value block, top-k T1/T8 | 0.5141 / 2.1984 ms | 0.3558 / 1.2619 ms | Keep; 30.8% / 42.6%. |
| E8M0 bit reconstruction after whole-block LM-head, top-k T1/T8 | 0.3558 / 1.2619 ms | 0.2618 / 1.2811 ms | Keep for decode-priority T1 and shared paths; T8 movement is within the central bands. |
| MXFP4 row-fragment specialization, QGEMM M32/M128/M512 | 0.0998 / 0.3611 / 1.3302 ms | first 0.1058 / 0.3524 / 1.3102; repeat 0.1207 / 0.3588 / 1.3193 ms | Reject; M32 regressed and larger shapes were below the 3% keep threshold. Generic compiler CSE already amortizes the scale. |
| Same row fragment, QFlux M128 | 0.3540 ms | 0.3557; repeat 0.3680 ms | Reject and restore generic decoder. |
| Bit reconstruction in generic MMA decoder, QGEMM M32/M128/M512 | 0.0999 / 0.3527 / 1.2935 ms | 0.0976 / 0.3531 / 1.3034 ms | Reject; mixed/noisy and M512 regressed. Restore native half `exp2`. |
| Bit reconstruction in four-column MoE decoder, rect rows32 / SwiGLU rows512 | 0.1089 / 1.9034 ms | repeat 0.1433 / 1.7532 ms | Reject global change: the 7.9% large-SwiGLU win does not justify the repeatable 31.6% decode-shape regression. Restore native half `exp2`. |

Decision: keep the new MXFP4 inference coverage, complete-block QGEMV and
sequential fused decoders, and exact E8M0 reconstruction in fp32 span/complete-
block paths. Reject row-fragment specialization and generic MMA/MoE E8M0 bit
reconstruction. Matrix and MoE paths remain intentionally unchanged.

Open questions: a future matrix-path pass needs a genuinely different MXFP4
execution strategy rather than more fragment temporaries. The large-row MoE
SwiGLU bit-decoder result may justify a separately routed prefill kernel only
if a shape-aware implementation can preserve the small-row decode path.

Raw results:

- Baseline/control/final: `mxfp4-inference-baseline`,
  `mxfp4-coverage-generic`, and `mxfp4-inference-final` under
  `perf/results/2026-07-13/`.
- Retained variants: `mxfp4-coverage-whole-block`,
  `mxfp4-decode-whole-block-repeat`, `mxfp4-coverage-e8m0-bits`,
  `mxfp4-e8m0-bits`, `mxfp4-e8m0-bits-repeat`,
  `mxfp4-embedding-e8m0-bits`, and `mxfp4-embedding-exp2`.
- Rejected variants: `mxfp4-hotpaths-candidate`,
  `mxfp4-row-fragment-repeat`, `mxfp4-pre-e8m0-bits`, and
  `mxfp4-moe-column-exp2-restored`.

## 2026-07-13: NVFP4 inference decode and output-projection pass

Status: landed; retained implementations have passed focused performance and
repository-wide correctness/parity validation.

Current implementation:

- `dequant_into_register` uses an NVFP4-specific row-fragment decoder for the
  `{c,c+1,c+8,c+9,c+16,c+17,c+24,c+25}` register layout. It reads the two E4M3
  scales and four packed bytes needed by the fragment once, preserving the
  established half-rounded tile contract.
- `dequant_into_register_col` uses an NVFP4 two-block decoder for the
  `{c,c+8,c+16,c+24}` column fragment used by rectangular MoE kernels.
- Fused decode epilogue and SwiGLU kernels consume a complete 16-value NVFP4
  block per lane. Their decoder keeps scale/code products in fp32 to match the
  public one-final-rounding contract.
- Quantized LM-head sampling uses a whole-block half-rounded decoder, while
  masked and CSR-candidate projection use a whole-block fp32 decoder. NVFP4 is
  exposed for argmax/categorical/top-k/top-p sampling, masked/candidate output
  projection, and exact beam advance in MLX and PyTorch MPS.
- The MLX CMake target now tracks every `include/metal/*.metal` file as a
  metallib dependency. This prevents header-only decoder changes from leaving
  a stale incremental-build metallib, which was observed during this pass.

Current public route:

- QGEMM/QFlux keep their existing direct launch geometry and use the new
  row-fragment decoder. Rectangular MoE keeps four warps and uses the new
  column-fragment decoder.
- Decode epilogue/SwiGLU and sequential LM-head row dots use complete-block
  NVFP4 decode.
- Beam advance keeps row-wise no-logits fusion for at most four rows and routes
  larger row batches through packed QGEMM, matching the existing q4_0 policy.

References inspected: existing repository q4_0 whole-block decoders and the
local NVFP4 `{E4M3 scale, 8 packed E2M1 bytes}` format contract. No external
implementation code was imported.

Correctness:

- Hardware/toolchain: MacBook Pro (Mac16,5), Apple M4 Max, 128 GB; macOS 26.5.1
  (25F80); Xcode 26.6 (17F113); Metal 32023.883; Python 3.12.9; MLX 0.21.1.
- Working-tree label: `c880769-dirty`.
- QGEMM/QFlux focused suite: 174 passed after the row-fragment change.
- Quantized MoE focused suite: 18 passed for each tested warp topology and the
  retained column decoder.
- `pytest tests/correctness/matmul/decode_linear/test_decode_linear.py -q`:
  47 passed.
- `pytest tests/correctness/quantization/lm_head/test_lm_head.py -q`:
  90 passed before beam coverage was added; subsequent NVFP4 sampling, sparse,
  and beam subsets passed 9, 1, and 4 cases respectively.
- Final builds: `scripts/build kernels` and `scripts/build pytorch_mps` passed.
  Touching `dequant.metal` then rerunning the incremental MLX build emitted
  `Building mlx_ext.metallib`, validating the new header dependency tracking.
- Final suites: `scripts/test correctness -q` passed 2085 tests;
  `scripts/test parity -q` passed 412; `scripts/test mps -q` passed 472.
- Final benchmark reference errors were `2.29e-7` relative for decode epilogue,
  `2.39e-7` for SwiGLU, and zero selected-id error for masked/candidate paths.

Focused commands (MLX integration path, `--warmup 10 --iters 40`):

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset quick \
  --kernel qgemm,qflux,moe_q --formats nvfp4 --warmup 10 --iters 40 \
  --out-dir perf/results/2026-07-13/nvfp4-experiments-baseline
.venv/bin/python perf/bench_kernels.py --backend mlx --preset quick \
  --kernel decode_linear_epilogue,decode_swiglu --formats nvfp4 \
  --warmup 10 --iters 40 --out-dir <variant-directory>
.venv/bin/python perf/bench_kernels.py --backend mlx --preset quick \
  --kernel lm_head_q,lm_head_masked,lm_head_candidates,lm_head_beam \
  --formats nvfp4 --warmup 10 --iters 40 --out-dir <variant-directory>
.venv/bin/python perf/bench_kernels.py --backend mlx --preset quick \
  --kernel qgemm,qflux,moe_q,decode_linear_epilogue,decode_swiglu,lm_head_q,lm_head_masked,lm_head_candidates,lm_head_beam \
  --formats nvfp4 --warmup 10 --iters 40 \
  --out-dir perf/results/2026-07-13/nvfp4-experiments-final
```

Fragment decoder results (median milliseconds; brackets are p20/p80, followed
by coefficient of variation):

| Path / shape | Baseline | Candidate | Change | Decision |
|---|---:|---:|---:|---|
| QGEMM N4096 K4096 M32 | 0.1291 [0.1235/0.1398], CV .1009 | 0.1151 [0.1132/0.1234], CV .0758 | -10.9% | keep row decoder |
| QGEMM N4096 K4096 M128 | 0.4307 [0.4160/0.4566], CV .0495 | 0.3745 [0.3623/0.4050], CV .0577 | -13.1% | keep row decoder |
| QGEMM N4096 K4096 M512 | 1.5207 [1.4695/1.5748], CV .0374 | 1.3976 [1.3577/1.4745], CV .0412 | -8.1% | keep row decoder |
| QFlux N4096 K4096 M128 | 0.4050 [0.3951/0.4333], CV .0600 | 0.3686 [0.3619/0.3950], CV .0545 | -9.0% | keep row decoder |
| MoE E4 K2880 N2880 rows32 | 0.1782 [0.1584/0.2629], CV .2714 | 0.1779 [0.1554/0.2601], CV .3496 | flat | keep for larger shape |
| MoE E4 K2880 N2880 rows512 | 0.8255 [0.7734/0.8970], CV .0680 | 0.7269 [0.7096/0.8144], CV .0672 | -11.9% | keep column decoder |

Whole-block fused results compare the un-specialized generic NVFP4 integration
against the complete-block decoder:

| Path / shape | Generic median ms | Whole-block median ms | Change | Candidate p20/p80, CV | Decision |
|---|---:|---:|---:|---:|---|
| Decode epilogue B1 K1536 N4096 | 0.0752 | 0.0223 | -70.3% | 0.0194/0.0277, .7636 | keep; repeat 0.0170 ms |
| Decode SwiGLU B1 K1536 N4096 | 0.1589 | 0.0244 | -84.6% | 0.0238/0.0257, .2510 | keep; repeat 0.0300 ms |
| LM-head top-k T1 V32000 K4096 | 0.4008 | 0.2967 | -26.0% | 0.2867/0.3211, .0999 | keep; repeat 0.2926 ms |
| LM-head top-k T8 V32000 K4096 | 1.4058 | 1.3460 | -4.2% | 1.3157/1.4519, .0528 | keep; repeat 1.3486 ms |
| Masked T1 V8192 K1024 legal256 | 0.1051 | 0.0406 | -61.4% | 0.0379/0.0512, .2151 | keep; repeat 0.0457 ms |
| Masked T8 V8192 K1024 legal64 | 0.0569 | 0.0425 | -25.4% | 0.0414/0.0498, .2100 | keep; repeat 0.0425 ms |
| Candidates T1 V8192 K1024 C256 | 0.0778 | 0.0400 | -48.6% | 0.0389/0.0443, .0988 | keep; repeat 0.0404 ms |
| Candidates T8 V8192 K1024 C64 | 0.0327 | 0.0169 | -48.5% | 0.0163/0.0213, .2786 | keep; repeat 0.0164 ms |
| Beam B1 BM4 V32000 K4096 | 0.8477 | 0.8152 | -3.8% | 0.8065/0.8289, .0284 | keep shared decoder |
| Beam B4 BM4 V32000 K4096 | 0.9823 | 0.9831 | flat | 0.9714/1.0088, .0221 | keep QGEMM route |

Final retained run medians were 0.1121/0.3582/1.3264 ms for QGEMM M32/M128/M512,
0.3581 ms for QFlux M128, 0.1360/0.6980 ms for MoE rows32/rows512,
0.0187/0.0246 ms for decode epilogue/SwiGLU, 0.2854/1.3310 ms for LM-head
top-k T1/T8, 0.0329/0.0422 ms for masked T1/T8, 0.0346/0.0163 ms for
candidate T1/T8, and 0.8216/0.9794 ms for beam B1/B4. Per-case p20/p80,
CV, bandwidth, and baseline data are in the final JSONL.

Rejected experiments:

- Two-warp, 64-row decoded-weight reuse: M128 was 0.3686 versus 0.3602 ms
  direct (+2.3%); M512 was 1.3667 versus 1.3630 ms (flat). Reject the extra
  barriers and routing.
- Four-warp, 128-row decoded-weight reuse: repeat medians were 0.3589 versus
  0.3643 ms direct at M128 and 1.3566 versus 1.3692 ms at M512 (only 1.5% and
  0.9%). It still trails resident fp16 matmul and does not justify a second
  pipeline or synchronization cost. Reject.
- Rectangular MoE at two warps: rows32 improved only 1.7% (0.1748 versus
  0.1779 ms) while rows512 was flat/slightly worse (0.7281 versus 0.7269 ms).
  One warp regressed rows32 to 0.3136 ms and rows512 to 0.7347 ms. Keep four.
- Forcing 16 NVFP4 beam rows through serial row fusion regressed 0.9831 to
  2.6393 ms. Keep the packed-QGEMM route above four rows.

Decision: keep both fragment decoders, all three whole-block decoder uses, and
the new NVFP4 fused inference coverage. Reject decoded-weight sharing, reduced
MoE warp counts, and the larger-row beam routing change. The main remaining
QGEMM opportunity is a different matrix execution strategy; small launch or
barrier variations did not pay for their complexity.

Open questions: profile instruction mix/register pressure for the complete-block
decode at larger hidden sizes; revisit T8 LM-head sampling only with a design
that shares packed weights across rows without materializing logits.

Raw results:

- Baseline/fragment runs: `nvfp4-experiments-baseline`,
  `nvfp4-experiments-row-fragment`, `nvfp4-experiments-column-fragment`.
- Rejected launch runs: `nvfp4-experiments-reuse-m64-controlled`,
  `nvfp4-experiments-reuse-m128-controlled`,
  `nvfp4-experiments-reuse-m128-repeat`, `nvfp4-experiments-moe-w1`,
  `nvfp4-experiments-moe-w2`, `nvfp4-experiments-lm-head-beam-force-row`.
- Fused decoder runs: `nvfp4-experiments-decode-generic`,
  `nvfp4-experiments-decode-whole-block`,
  `nvfp4-experiments-decode-whole-block-repeat`,
  `nvfp4-experiments-lm-head-generic`,
  `nvfp4-experiments-lm-head-whole-block`,
  `nvfp4-experiments-lm-head-whole-block-repeat`,
  `nvfp4-experiments-lm-head-sparse-generic`,
  `nvfp4-experiments-lm-head-sparse-whole-block`,
  `nvfp4-experiments-lm-head-sparse-whole-block-repeat`,
  `nvfp4-experiments-lm-head-beam-generic`, and
  `nvfp4-experiments-lm-head-beam-whole-block`.
- Final retained run: `perf/results/2026-07-13/nvfp4-experiments-final/`.

## 2026-07-07: BitNet remaining kernel parity port

Status: landed as parity/coverage; no performance claim.

Current implementation: ported the remaining first-party BitNet Metal kernels that were absent
from QuixiCore Metal: `attn_decode`, `fake_quant_fp8`, `kd_kl_dense`, `qgemm_bwd`,
`qgemm_w2a8_fused`, `quantize_tq2_0`, `ternary_stats`, and `gemm_v3`. Also added
`tq2_0` dequant/GEMM/GEMV/MoE format support, `qgemv_w2a8_v2`, dynamic-width `rms_norm`,
and MoE backward helpers.

Current public route: MLX and PyTorch MPS bindings expose the new kernels through `tk` and
`tk_torch`; manifests include the new paths and `tq2_0` format metadata.

References inspected:
- `/Users/eric/BitNet/bitnet_train/metal/tk_torch/torch_kernels.mm`
- `/Users/eric/BitNet/bitnet_train/metal/tk_torch/__init__.py`
- `/Users/eric/BitNet/bitnet_train/metal/kernels/`

Correctness:
- Hardware/toolchain: Apple M4 Max MacBook Pro, macOS 26.5.1 (25F80), Metal 32023.883,
  Python 3.12.9.
- `PYTHON=/Users/eric/QuixiCore/QuixiCore-Metal/.venv/bin/python ./scripts/build python`
  passed.
- `PYTHON=/Users/eric/QuixiCore/QuixiCore-Metal/.venv/bin/python ./scripts/build pytorch_mps`
  passed.
- `PYTHONPATH=/Users/eric/QuixiCore/QuixiCore-Metal/bindings/pytorch_mps
  /Users/eric/QuixiCore/QuixiCore-Metal/.venv/bin/python -c 'import tk_torch; ...'`
  compiled/imported the PyTorch metallib and ObjC++ extension; all checked new symbols were present.
- Targeted correctness command:
  `.venv/bin/python -m pytest -q tests/correctness/quantization/quantize_tq2_0
  tests/correctness/quantization/ternary_stats tests/correctness/utils/kd_kl_dense
  tests/correctness/attention/attn_decode tests/correctness/quantization/qgemv_int/test_qgemv_int.py`
  passed: 14 passed.
- `PYTHON=/Users/eric/QuixiCore/QuixiCore-Metal/.venv/bin/python ./scripts/test mps -q`
  passed: 452 passed.
- `PYTHON=/Users/eric/QuixiCore/QuixiCore-Metal/.venv/bin/python ./scripts/test python -q`
  passed: 44 passed.
- Direct PyTorch MPS smoke checked `quantize_tq2_0`, dense KD-KL fwd/bwd, and `attn_decode`
  against CPU references; passed.

Baseline: not run for this parity port.

Experiments: none. This was a source/API parity port, not a focused optimization pass.

Decision: keep the ported kernels as coverage and interoperability surface. The previous
performance decision for `qgemm_bwd` and `gemm_v3` still stands: do not present them as Metal
speedups without a new focused benchmark showing a win on priority shapes.

Open questions: run a focused benchmark before any future speedup claim or routing preference
change for these kernels.

## 2026-07-07: BitNet training kernel port

Status: landed.

Current implementation: ported the production BitNet training kernels that were missing from
QuixiCore Metal:
`weight_quant_ternary` / `weight_quant_ternary_pt`, `fake_quant_int8`,
`silu_mul_fake_quant_int8`, `kd_kl_topk_fwd` / `kd_kl_topk_bwd`, and `adamw_masked`.
Added MLX primitives, PyTorch MPS wrappers, Metal source registration, manifest paths,
correctness tests, MPS tests, and benchmark cases.

Current public route: `tk.weight_quant_ternary`, `tk.weight_quant_ternary_pt`,
`tk.fake_quant_int8`, `tk.silu_mul_fake_quant_int8`, `tk.kd_kl_topk_fwd`,
`tk.kd_kl_topk_bwd`, and `tk.adamw_masked`; all auto-route to MLX or `tk_torch`
based on tensor type.

References inspected:
- `/Users/eric/BitNet/bitnet_train/metal/kernels/bitnet/weight_quant_ternary.metal`
- `/Users/eric/BitNet/bitnet_train/metal/kernels/bitnet/fake_quant.metal`
- `/Users/eric/BitNet/bitnet_train/metal/kernels/bitnet/kd_kl_topk.metal`
- `/Users/eric/BitNet/bitnet_train/metal/kernels/optimizers/optim/adamw.metal`
- `/Users/eric/BitNet/bitnet_train/metal/perf/bitnet_training_kernels.md`
- Rejected after inspection: `/Users/eric/BitNet/bitnet_train/metal/kernels/bitnet/qgemm_bwd.metal`
  and `/Users/eric/BitNet/bitnet_train/metal/kernels/matmul/gemm_v3/gemm_v3.metal`.
  BitNet's notebook records `qgemm_bwd` losing to `torch.matmul` on every measured shape and
  `gemm_v3` reaching 94-99% of MPS without beating it, so neither is a QuixiCore win to expose.

Correctness:
- `PYTHON=/Users/eric/QuixiCore/QuixiCore-Metal/.venv/bin/python ./scripts/build python`
  passed.
- `PYTHON=/Users/eric/QuixiCore/QuixiCore-Metal/.venv/bin/python ./scripts/test correctness -q
  tests/correctness/quantization/weight_quant_ternary
  tests/correctness/quantization/fake_quant tests/correctness/utils/kd_kl_topk
  tests/correctness/optimizers/optim/test_adamw.py` ran the correctness suite and passed:
  1420 passed, 27 skipped.
- `PYTHON=/Users/eric/BitNet/.venv/bin/python ./scripts/build pytorch_mps` passed.
- `PYTHON=/Users/eric/BitNet/.venv/bin/python ./scripts/test mps -q` passed:
  452 passed.

Focused perf run:
- Integration path: MLX Python extension.
- Hardware/toolchain: Apple M4 Max MacBook Pro, macOS 26.5.1 (25F80), Xcode 26.6
  (17F113), Metal 32023.883, Python 3.12.9, MLX 0.21.1.
- Working-tree label: `e484dc7-dirty`.
- Command: `/Users/eric/QuixiCore/QuixiCore-Metal/.venv/bin/python perf/bench_kernels.py
  --backend mlx --preset quick --kernel fake_quant,weight_quant_ternary,kd_kl_topk,adamw_masked
  --warmup 5 --iters 20 --out-dir perf/results/2026-07-07/bitnet-port-quick`
- Raw results: `perf/results/2026-07-07/bitnet-port-quick/`.

| kernel | dtype/path | shape | median ms | p20/p80 ms | CV | baseline | speedup | rel err | decision |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| fake_quant plain | bf16 MLX | T512 D2880 | 0.0200 | 0.0195/0.0216 | 0.5619 | quantize_then_dequant 0.0438 | 2.19 | 6.45e-03 | keep |
| fake_quant swiglu | bf16 MLX | T512 D2880 | 0.0320 | 0.0304/0.0415 | 0.3738 | swiglu_then_quant_dequant 0.0631 | 1.97 | 3.46e-03 | keep |
| fake_quant plain | bf16 MLX | T4096 D2880 | 0.1383 | 0.1351/0.1953 | 0.5535 | quantize_then_dequant 0.3286 | 2.38 | 5.95e-03 | keep |
| fake_quant swiglu | bf16 MLX | T4096 D2880 | 0.2851 | 0.2766/0.3211 | 0.2518 | swiglu_then_quant_dequant 0.5296 | 1.86 | 3.39e-03 | keep |
| weight_quant_ternary | bf16 MLX | N512 K2880 G32 | 0.0443 | 0.0441/0.0452 | 0.1691 | framework_deq_only 0.1323 | 2.99 | 2.88e-03 | keep |
| weight_quant_ternary | bf16 MLX | N4096 K2880 G32 | 0.2877 | 0.2855/0.2943 | 0.1763 | framework_deq_only 0.9898 | 3.44 | 3.48e-03 | keep |
| weight_quant_ternary_pt | bf16 MLX | E2 N512 K2880 | 0.0805 | 0.0793/0.0886 | 0.2181 | framework_deq_only 0.2986 | 3.71 | 2.30e-03 | keep |
| kd_kl_topk tail0 | f32 MLX | T256 V32000 K32 | 0.5136 | 0.5023/0.5583 | 0.0539 | none | n/a | 1.66e-07 | keep |
| kd_kl_topk tail1 | f32 MLX | T256 V32000 K32 | 0.4775 | 0.4701/0.4949 | 0.0584 | none | n/a | 8.30e-08 | keep |
| kd_kl_topk tail0 | f32 MLX | T1024 V32000 K32 | 0.7561 | 0.7454/0.7716 | 0.2550 | none | n/a | 1.28e-07 | keep |
| kd_kl_topk tail1 | f32 MLX | T1024 V32000 K32 | 0.7493 | 0.7386/0.7636 | 0.2480 | none | n/a | 6.38e-08 | keep |
| adamw_masked mode0 | f32 MLX | numel 4194304 seg256 active 0.650 | 0.3177 | 0.3141/0.3283 | 0.3174 | unmasked_adamw 0.2981 | 0.94 | 5.38e-08 | keep as semantic path |
| adamw_masked mode1 | f32 MLX | numel 4194304 seg256 active 0.650 | 0.3435 | 0.3385/0.3496 | 0.1060 | unmasked_adamw 0.2914 | 0.85 | 5.38e-08 | keep as semantic path |
| adamw_masked mode0 | f32 MLX | numel 16777216 seg256 active 0.653 | 1.2584 | 1.1901/1.6126 | 0.1472 | unmasked_adamw 1.1074 | 0.88 | 5.33e-08 | keep as semantic path |
| adamw_masked mode1 | f32 MLX | numel 16777216 seg256 active 0.653 | 1.0982 | 1.0900/1.1216 | 0.1075 | unmasked_adamw 1.1169 | 1.02 | 5.32e-08 | keep as semantic path |

Decision: keep the production BitNet kernels. `fake_quant*` and `weight_quant_ternary*`
are clear wins over decomposed framework paths. `kd_kl_topk` has no useful decomposed baseline
because its value is avoiding dense teacher materialization, but the sparse fwd+bwd path passed
dense-reference checks and is fast enough for the intended training route. `adamw_masked` is
not a speedup over unmasked AdamW and should not be marketed as one; it is kept for the segment
mask semantics. Reject `qgemm_bwd` and `gemm_v3` for this port because the source project's own
measurements do not show a Metal win.

Open questions: add a future dense-teacher KD baseline if a training harness exposes the exact
end-to-end loss path; revisit `adamw_masked` only if segment sparsity is high enough to justify
a compacted-index variant.

## Wave-10 K5: EAGLE spec-decode input-prep builders (2026-07-05)

Extended kernels/sampling/ (metal-forge sequence/spec_decode.metal; credit AlpinDale) with
EAGLE's draft-input plumbing (zero prior TM coverage): eagle_prepare_inputs_padded (rejected =
num_draft>0 ? num_draft+1-valid : 0; token_indices_to_sample, num_rejected),
eagle_prepare_next_token_padded (next seed = last valid sampled or backup),
eagle_step_slot_mapping_metadata (new_pos = min(pos+1, max_len); block-table -> paged slot;
advance seq_lens; pad beyond the real batch), eagle_expand_int32 (broadcast a per-request scalar
across its ragged token span with a replace substitution). Integer, one thread/request; TM int32;
cu_* (B+1,) leading-0. One EagleMeta primitive (kind selector). Completes TM's spec-decode
surface (verify + rejection + EAGLE prep).

- Tests: each builder vs a direct python transcription (exact int32, incl. padded input_batch
  and exceed-max-len paths); 5 green + parity atol=0. Overhead-bound integer kernels — no bench.
- Deferred (documented): copy_and_expand_eagle_inputs (the full padded-batch layout builder) —
  the four builders above cover the metadata a draft step needs.

Wave-10 COMPLETE: K1 norm->quant matrix, K2 fp8 KV gather+scale-update, K3 DeepSeek indexer,
K4 rejection samplers, K5 EAGLE prep. All credited to AlpinDale / metal-forge.

## Wave-10 K4: vLLM v1 ragged rejection samplers (2026-07-05)

Extended kernels/sampling/ (metal-forge sequence/spec_decode.metal; credit AlpinDale) with the
vLLM v1 ragged rejection pipeline TM's dense spec_verify_linear didn't cover:
rejection_greedy_sample (argmax-match verify, no probs — no prior TM analogue),
rejection_random_sample (stochastic u <= p_target/q_draft, recovered token a precomputed input),
sample_recovered_tokens (argmax of max(0, p_t - q_d) * inv_q via a 32-lane simd reduction with
smaller-id tie-break). Variable drafts/request via cu_num_draft_tokens (B+1,) with a leading 0;
TM int32 ids; external-noise buffers (uniform_probs, inv_q) match the vLLM contract (host-
generated, seed-reproducible). Output (B, max_draft+1) cleared to -1. One RejectionSampler
primitive (kind selector). is_greedy per-request gate; no_draft_probs -> q=1.

- Tests: each kernel vs a direct python transcription of the reference loop (exact int ids),
  incl. the two-kernel pipeline (sample_recovered -> rejection_random); 6 green + parity atol=0.
- Overhead-bound integer kernels (like the existing spec_verify_* / build_dynamic_tree) — no
  bench entry.

## Wave-10 K3: DeepSeek-V3.2 indexer K quant-and-cache (2026-07-05)

New kernels/indexer/ (metal-forge indexer_k_quant_and_cache; credit AlpinDale). Quantizes the
DSA/NSA indexer K per quant_block_size (canonical 128) into a low-precision e4m3 cache the
sparse-attention top-k selector reads cheaply — pairs with the MInference block masks to give
TM a real sparse-attention SELECTION path, not just a mask consumer. TM-native layout: SEPARATE
code cache (uchar, num_slots x head_dim) + fp32 scale cache (num_slots x head_dim/qbs) indexed
directly by slot_mapping (like the TurboQuant codec), not the reference's interleaved paged
single-buffer. One simdgroup per (token, qblock), simd_max absmax (no threadgroup scratch);
use_ue8m0 rounds the fp32 scale to a power of two. indexer_k_gather dequantizes back to bf16 for
a slot list. Functional (clone-then-insert; untouched slots preserved).

- Tests: fp32 scales bit-exact vs numpy (plain) / power-of-two + coverage (ue8m0); e4m3 codes
  reconstruction-bounded (round-to-nearest-even ties differ from a numpy argmin — the repo's
  fp8 contract); round-trip + slot<0 skip + untouched-slot preservation; 19 green + parity
  (codes off-by-one across the two metallibs, scales 1e-4).
- Perf: bandwidth-bound (reads bf16 K, writes u8 codes), 16384x128 0.062 ms; near-optimal
  one-simdgroup/qblock. No further opt.
- Deferred (documented): fused DSA sparse decode over the indexer cache (the consumer).

## Wave-10 K2: fp8 KV gather+upconvert + incremental scale update (2026-07-05)

Extended kernels/kv_cache/ (metal-forge cache/gather_kv_cache.metal + kv_scale_update.metal,
credit AlpinDale) to close the fp8 KV loop. kv_cache_gather_fp8<OUT_T>: the READ path for a
paged fp8 prefix cache — reads e4m3/e5m2 code bytes and dequantizes to bf16 via
code * scale[kv_head] (per-kv_head scales, round-trips exactly with kv_cache_scatter_fp8),
same worklist as kv_cache_gather (one TG/token, cu_seq_lens binary search, block<0 zero-fill),
fmt runtime scalar. kv_cache_scale_update: incremental per-tensor running-max (new = max(old,
absmax/240)) — the streaming-decode analogue of the one-shot kv_cache_scales; single 256-thread
threadgroup, no atomics needed for the scalar (grid-stride reduction seeds from the old value).

- Tests: scatter_fp8 -> gather_fp8 round-trip (== decode(code)*scale exactly, within fp8
  relative precision of the original K, e4m3 + e5m2); scale_update vs numpy running-max
  (only-raises verified); 154 kv_cache green + parity (bf16 rows 2e-2, scales 1e-5).
- Perf: bandwidth-bound read path, reads 1-byte codes vs the bf16 gather's 2-byte (halved
  cache read bandwidth); nb256 gather 0.104 ms. Same near-optimal one-TG/token coalesced
  structure as kv_cache_gather — no further opt.
- Deferred (documented): per-tensor (vs per-kv_head) fp8 gather variant; the MLA-cache
  upconvert-gather (cp_gather_upconvert_fp8_mla — different 656-byte cache layout).

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

## Fused and specialized kernel integration pass (2026-07-13)

Focused hypothesis: the specialized kernels should win when fusion removes an
intermediate tensor or packed GGUF reads avoid expanded weights. Serial
one-simdgroup mappings may lose to framework matmul/SDPA on realistic vision
shapes and must remain non-default when they do.

Environment: Apple M4 Max MacBook Pro (16 CPU cores, 128 GB), macOS 26.5.1
(25F80), Xcode 26.6 (17F113), Apple Metal 32023.883 / Metal toolchain
17.6.109.0, Python 3.12.9, MLX 0.21.1, and PyTorch 2.12.1 MPS. Commands:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset quick \
  --kernel kernel_extensions --warmup 20 --iters 50
.venv/bin/python perf/bench_kernels.py --backend torch --preset quick \
  --kernel kernel_extensions --warmup 20 --iters 50
```

The harness uses a one-second clock ramp, at least 50 ms of per-thunk warmup,
adaptive batching to at least 2 ms/sample, 20 requested warmups, 50 measured
samples, and reports medians. P20/p80 and CV are in the raw JSONL. Target CV
ranged 0.010–0.247 on MLX and 0.008–0.151 on MPS because a few launch-bound
cases had isolated OS/queue outliers; their central p20/p80 bands remained
narrow (for example MLX Q6_K gather 0.0099–0.0106 ms). Runs overlapping an
unrelated compiler workload were discarded. Final raw results:

- `perf/results/2026-07-13/153245-mlx-quick/`
- `perf/results/2026-07-13/153405-torch-quick/`

All 16 cases passed their in-run oracle. Maximum relative error was 3.07e-3
(bfloat16 patch merge); all three dequant-gather formats and token-selection
cases were exact. The Q6_K gather result includes an MPS fast-math halfway-case
regression that pins FP32 operation order and one final FP16 rounding. Focused
MLX tests and the dedicated MPS extension suite cover the same paths and edge shapes.

| Kernel / integration path | dtype / format | priority shape | MLX candidate / baseline ms | MPS candidate / baseline ms | Decision |
|---|---|---|---:|---:|---|
| dynamic LayerNorm | bf16 | R8,D1536 | 0.0115 / 0.0115 | 0.0301 / 0.0193 | Route dynamic widths to framework LayerNorm; retain `use_kernel=True`. |
| decode add + LayerNorm | bf16 | R8,D256 | 0.0126 / 0.0192 | 0.0257 / 0.0764 | Keep default (1.53× / 2.97×). |
| head-major GQA decode | f32 | B4,Hq8,Hkv2,T256,D32 | 0.0335 / 0.0332 | 0.1090 / 0.0975 | Route to framework SDPA; retain `use_kernel=True`. |
| Swin window attention | f32 | BW8,N144,H4,D32 | 0.3754 / 0.1110 | 0.3859 / 0.1661 | Reject as default; framework SDPA is 3.38× / 2.32× faster. Retain `use_kernel=True`. |
| patch merge + LayerNorm | bf16 | B1,96x96,C128 | 0.0356 / 0.0427 | 0.0403 / 0.0787 | Keep default (1.20× / 1.95×). |
| pairwise edge MLP | f32 | B1,L64,H256,C7 | 1.3840 / 0.1687 | 1.3267 / 0.1385 | Reject as default; framework composition is 8.20× / 9.58× faster. Retain `use_kernel=True`. |
| decode linear + erf-GELU | f32 | B1,K256,N256 | 0.0097 / 0.0336 | 0.0125 / 0.0383 | Keep default (3.47× / 3.07×). |
| q8_0 decode linear + epilogue | f32 / q8_0 | B1,K256,N256 | 0.0105 / 0.0332 | 0.0113 / 0.0422 | Keep default (3.16× / 3.74×). |
| Flux erf-GELU | f32 | N128,K64,M128 | 0.0088 / 0.0331 | 0.0127 / 0.0352 | Keep default (3.77× / 2.76×). |
| dequant gather | f16 / q4_0 | R4096,D256,T256 | 0.0091 / 0.0319 | 0.0201 / 0.0786 | Keep default (3.50× / 3.91×). |
| dequant gather | f16 / q8_0 | R4096,D256,T256 | 0.0093 / 0.0268 | 0.0180 / 0.0760 | Keep default (2.87× / 4.23×). |
| dequant gather | f16 / q6_K | R4096,D1536,T256 | 0.0102 / 0.0352 | 0.0265 / 0.0830 | Keep default (3.45× / 3.13×). |
| fp32 qgemv | f32 / q4_0 | N6144,K1536 | 0.0196 / 0.0523 | 0.0353 / 0.0894 | Keep packed path (2.67× / 2.53×). |
| fp32 qgemv | f32 / q6_K | N1536,K1536 | 0.0127 / 0.0168 | 0.0163 / 0.0278 | Keep packed path (1.32× / 1.71×). |
| fused Q6_K LM argmax | f32 / q6_K | T1,V32768,K1536 | 0.2633 / 0.1390 | 0.4367 / 0.1851 | Reject as default versus packed qgemv+argmax; route T=1 through qgemv, retain `fused=True`. |
| constrained LM head | f32 | T8,V357,K256 | 0.0358 / 0.1133 | 0.0699 / 0.2270 | Keep default (3.17× / 3.25×). |

The baseline is the public framework SDPA path for attention, the
framework/decomposed path for other fused operations, an FP32 expanded-table
gather with one final FP16 cast or pre-dequantized matmul for packed formats,
and packed qgemv+argmax for the Q6_K LM-head routing comparison. No speed claim
is made for the five retained non-default kernels. The implementations and
benchmark routing are accepted with the keep/reject decisions above.

## 2026-07-13: Packed embedding, decode, sparse projection, spatial, and cache-attention pass

Status: candidate; correctness and focused performance work complete.

Current implementation: added packed quantized embedding lookup and CSR
embedding-bag reduction; dense/q4_0/q8_0/q6_K decode projection with fused
epilogue and SwiGLU; packed-mask and CSR-candidate output projection; block-2/4
space-to-depth + LayerNorm + projection; and a functional decode-cache attention
step with optional Q/K RMSNorm, split-half RoPE, cache append, and GQA. The
shared fp32 packed decoder now covers all embedding formats without widening a
half-precision intermediate.

Current public route:

- Packed embedding, packed decode, and sparse output projection use their Metal
  kernels directly.
- Dense `decode_linear_epilogue` uses Metal below 1,000,000 multiply terms and
  framework matmul above that measured crossover. Dense `decode_swiglu` uses
  the framework composition by default; packed weights use Metal.
- `space_to_depth_norm_linear` uses Metal below 1,000,000 projection multiply
  terms and the framework composition above it.
- `decode_cache_attention` uses Metal below 512 allocated cache slots and the
  equivalent functional framework composition from 512 slots onward. Explicit
  `use_kernel=True/False` remains available on routed operations.

References inspected: existing QuixiCore Metal dequant, qgemv, LayerNorm,
embedding, and SDPA implementations. No application runtime or application
contract is part of these APIs.

Correctness:

- `scripts/build kernels` passed for the native MLX extension and Metal library.
- `PYTHONPATH=bindings/python .venv/bin/python -m pytest -q
  tests/correctness/quantization/dequant_gather/test_dequant_gather.py
  tests/correctness/matmul/decode_linear/test_decode_linear.py
  tests/correctness/quantization/lm_head/test_lm_head.py
  tests/correctness/vision/patch_merge/test_patch_merge.py
  tests/correctness/attention/attn_decode/test_attn_decode.py` passed: 187.
- `PYTHON=.venv/bin/python scripts/test mps -q -k 'quantized_embedding or
  decode_linear or lm_head_masked or lm_head_candidates or
  space_to_depth_norm_linear or decode_cache_attention'` passed: 9 focused
  cases (463 deselected).
- `PYTHON=.venv/bin/python scripts/test correctness -q` passed: 2025.
- `PYTHON=.venv/bin/python scripts/test mps -q` passed: 472.
- `PYTHON=.venv/bin/python scripts/test parity -q` passed: 401, including
  direct cross-backend parity for every new operation family.
- The final quick benchmark ran 21 oracle-checked cases. Maximum relative error
  was 1.64e-6; embedding lookup and both sparse projection paths were exact.
  Tests cover fp32/fp16/bfloat16 output, every supported packed embedding
  decoder, dense and q4_0/q8_0/q6_K projection, invalid ids, empty/weighted
  bags, odd spatial padding, variable cache lengths, normalization modes, and
  both routed and direct-kernel paths.

Focused performance run:

- Integration path: MLX Python extension, explicit direct-kernel targets versus
  equivalent MLX/decomposed baselines.
- Hardware/toolchain: Apple M4 Max (128 GB), macOS 26.5.1 (25F80), Xcode 26.6
  (17F113), Apple Metal 32023.883 / Metal toolchain 17.6.109.0, Python 3.12.9,
  MLX 0.21.1.
- Working-tree label: `294f8bd-dirty`.
- Initial/candidate command: `.venv/bin/python perf/bench_kernels.py --backend
  mlx --preset quick --kernel quantized_embedding,quantized_embedding_bag,
  decode_linear_epilogue,decode_swiglu,lm_head_masked,lm_head_candidates,
  space_to_depth_norm_linear,decode_cache_attention --warmup 5 --iters 20`.
- Final command: `PYTHONPATH=bindings/python .venv/bin/python
  perf/bench_kernels.py --backend mlx --preset quick --kernel
  quantized_embedding,quantized_embedding_bag,decode_linear_epilogue,
  decode_swiglu,lm_head_masked,lm_head_candidates,
  space_to_depth_norm_linear,decode_cache_attention --warmup 10 --iters 30
  --out-dir perf/results/2026-07-13/new-kernels-final-quick`.
- `time_thunk` warms for at least 50 ms, adaptively batches short calls to at
  least 2 ms per sample, synchronizes every sample, and reports per-call median,
  p20/p80, and CV. Small throughput-style calls showed OS/queue outliers (CV up
  to 0.60), so decisions use central bands and the independent initial,
  candidate, and final runs rather than a single minimum.

Baseline: packed operations compare against a resident pre-dequantized table or
weight plus the equivalent framework operations. Sparse output projection
compares against full/gathered dense logits, masking, top-k, and logsumexp.
Spatial projection compares against materialized space-to-depth, framework
LayerNorm, and matmul. Cache attention's equivalent baseline includes RoPE,
functional cache construction, and SDPA; its much smaller pre-updated,
output-only SDPA number is recorded in raw results but is not used for the
decision.

Experiments (one launch/decode factor changed at a time):

| Family / priority shape | Initial ms | Candidate ms | Decision |
|---|---:|---:|---|
| embedding lookup, q4_0 T256 D1024, 256 -> 128 threads | 0.0234 | 0.0110 | Keep; 2.14x faster. |
| embedding bag, q4_0 B128 L8 D1024, 256 -> 128 threads | 0.0331 | 0.0131 | Keep; 2.53x faster. |
| embedding bag, q4_0 B32 L32 D1024, 256 -> 128 threads | 0.0288 | 0.0155 | Keep; 1.86x faster. |
| q4_0 decode epilogue B1 K1536 N4096, scalar -> paired nibbles | 0.1051 | 0.0488 | Keep; 2.15x faster. |
| q4_0 SwiGLU B1 K1536 N4096, scalar -> paired nibbles | 0.0823 | 0.0279 | Keep; 2.95x faster. |
| masked output projection T1/T8, vocabulary tile 256 -> 128 | 0.0909 / 0.1635 | 0.0338 / 0.1234 | Keep; 2.69x / 1.32x. |
| space-to-depth block2/block4, 256 -> 128 threads | 0.7146 / 0.0894 | 0.7499 / 0.1325 | Reject and restore 256 threads. |
| cache attention T512/T2048, four -> two simdgroups | 0.3203 / 1.4659 | 0.6117 / 2.7687 | Reject and restore four simdgroups. |

The CSR candidate projection does not use the masked-LM vocabulary tile. Its
one-simdgroup CSR scan was retained without attributing run-to-run timing
movement to that unrelated experiment; the final direct-versus-equivalent
measurements below are the decision basis for this kernel.

Final priority-shape results are milliseconds. Each timing cell is `median
[p20,p80], CV`; speedup is equivalent-baseline/target.

| kernel | variant | target median [p20,p80], CV | equivalent baseline median [p20,p80], CV | speedup | rel err |
|---|---|---:|---:|---:|---:|
| quantized_embedding | q4_0 R8192 D1024 T1 | 0.0144 [0.0127,0.0232], 0.527 | 0.0158 [0.0115,0.0241], 0.393 | 1.09x | 0.00e+00 |
| quantized_embedding | q4_0 R8192 D1024 T256 | 0.0143 [0.0128,0.0171], 0.603 | 0.0344 [0.0298,0.0415], 0.277 | 2.40x | 0.00e+00 |
| quantized_embedding_bag | q4_0 R8192 D1024 B128 L8 | 0.0163 [0.0141,0.0191], 0.354 | 0.0674 [0.0636,0.0834], 0.393 | 4.14x | 1.14e-07 |
| quantized_embedding_bag | q4_0 R8192 D1024 B32 L32 | 0.0182 [0.0161,0.0201], 0.199 | 0.0658 [0.0604,0.0753], 0.371 | 3.62x | 1.66e-07 |
| decode_linear_epilogue | dense B1 K1536 N4096 | 0.0431 [0.0402,0.0518], 0.364 | 0.0431 [0.0414,0.0602], 0.281 | 1.00x | 2.29e-07 |
| decode_linear_epilogue | q4_0 B1 K1536 N4096 | 0.0439 [0.0404,0.0495], 0.370 | 0.0433 [0.0413,0.0522], 0.282 | 0.99x | 2.40e-07 |
| decode_linear_epilogue | q8_0 B1 K1536 N4096 | 0.0193 [0.0173,0.0252], 0.364 | 0.0429 [0.0408,0.0539], 0.342 | 2.22x | 2.36e-07 |
| decode_linear_epilogue | q6_K B1 K1536 N4096 | 0.0395 [0.0355,0.0440], 0.123 | 0.0417 [0.0387,0.0490], 0.376 | 1.06x | 2.35e-07 |
| decode_swiglu | dense B1 K1536 N4096 | 0.1244 [0.1137,0.1400], 0.174 | 0.0896 [0.0778,0.1026], 0.336 | 0.72x | 1.48e-07 |
| decode_swiglu | q4_0 B1 K1536 N4096 | 0.0343 [0.0329,0.0423], 0.396 | 0.0733 [0.0679,0.0864], 0.243 | 2.14x | 1.83e-07 |
| decode_swiglu | q8_0 B1 K1536 N4096 | 0.0493 [0.0448,0.0605], 0.235 | 0.1102 [0.1000,0.1227], 0.220 | 2.24x | 2.71e-07 |
| decode_swiglu | q6_K B1 K1536 N4096 | 0.0484 [0.0462,0.0544], 0.309 | 0.0705 [0.0684,0.0781], 0.312 | 1.46x | 1.98e-07 |
| lm_head_masked | q4_0 T1 V8192 K1024 L256 | 0.0495 [0.0446,0.0590], 0.133 | 0.1384 [0.1333,0.1668], 0.293 | 2.80x | 0.00e+00 |
| lm_head_masked | q4_0 T8 V8192 K1024 L64 | 0.1348 [0.1301,0.1518], 0.253 | 0.1712 [0.1586,0.1998], 0.216 | 1.27x | 0.00e+00 |
| lm_head_candidates | q4_0 T1 V8192 K1024 C256 | 0.0371 [0.0324,0.0417], 0.128 | 0.1702 [0.1545,0.1878], 0.122 | 4.59x | 0.00e+00 |
| lm_head_candidates | q4_0 T8 V8192 K1024 C64 | 0.0210 [0.0173,0.0298], 0.326 | 0.1775 [0.1612,0.2106], 0.160 | 8.45x | 0.00e+00 |
| space_to_depth_norm_linear | B1 48x48 C128 O512 S2 | 0.5965 [0.5726,0.6466], 0.112 | 0.0721 [0.0666,0.0821], 0.208 | 0.12x | 3.84e-07 |
| space_to_depth_norm_linear | B1 32x32 C64 O256 S4 | 0.0912 [0.0852,0.0977], 0.113 | 0.0872 [0.0808,0.1085], 0.181 | 0.96x | 3.07e-07 |
| decode_cache_attention | B4 H16/4 T512 D128 | 0.3789 [0.3621,0.4298], 0.109 | 0.3387 [0.2981,0.4148], 0.154 | 0.89x | 1.02e-06 |
| decode_cache_attention | B4 H16/4 T1024 D128 | 0.7808 [0.7538,0.8745], 0.086 | 0.4398 [0.4025,0.5846], 0.175 | 0.56x | 1.30e-06 |
| decode_cache_attention | B4 H16/4 T2048 D128 | 1.5635 [1.5090,1.7283], 0.094 | 0.8210 [0.7767,0.9536], 0.160 | 0.53x | 1.64e-06 |

Decision:

- Keep the 128-thread embedding launch, 128-token masked-LM vocabulary tile,
  and q4_0 paired-nibble decoder. Restore the 256-thread spatial launch and
  four-simdgroup attention launch after measured regressions.
- Keep packed embedding/bag and both sparse output projection kernels as direct
  defaults. They avoid expanded intermediates and win on priority shapes.
- Keep packed decode paths. q8_0 and packed SwiGLU are clear wins; q4_0 decode
  epilogue is parity with a resident expanded-weight baseline in the final run,
  but is substantially faster than its initial decoder and preserves packed
  storage. Do not claim a q4_0 epilogue speedup over resident dense weights.
- Route dense SwiGLU to the framework composition. Auto-route dense decode
  epilogue by size: the direct kernel wins at B1,K256,N256 (0.0108 vs 0.0235
  ms) and is tied at B1,K1536,N4096.
- Auto-route spatial projection: Metal wins the edge shape B1,8x8,C32,O64,S2
  (0.0139 vs 0.0563 ms), while the framework path is 8.27x faster on the
  realistic block-2 priority shape.
- Auto-route functional cache attention: Metal wins B1,H4/2,T64,D64 against
  the equivalent functional composition (0.0809 vs 0.2974 ms); the framework
  composition is 1.12x, 1.78x, and 1.90x faster at T512/1024/2048. The
  exclusive 512-slot allocation threshold keeps the measured sides of the
  crossover.

Open questions: re-evaluate a tiled/multi-query cache-attention algorithm that
parallelizes the context dimension, and a matrix-tiled spatial projection that
reuses each normalized patch across output channels. Revisit the q4_0/q6_K
decode crossover only with isolated long runs because sub-0.05 ms cases remain
sensitive to queue scheduling.

Raw results:

- `perf/results/2026-07-13/new-kernels-baseline-quick/`
- `perf/results/2026-07-13/new-kernels-candidate-quick/`
- `perf/results/2026-07-13/new-kernels-final-smoke/`
- `perf/results/2026-07-13/new-kernels-final-quick/`

## 2026-07-13: New-kernel second optimization pass

Status: complete. This pass supersedes the launch/decode conclusions in the
preceding entry where explicitly noted; the earlier measurements remain above
as the historical starting point.

Hypotheses:

- Embedding had reached a launch/occupancy balance at 128 threads; a smaller
  group might help T=1 but could starve batched lookup and bag reduction.
- The q4_0 decode paths still decoded one packed block across two lanes. A
  whole-block-per-lane mapping could load its scale/address once. The same
  whole-block idea needed an independent q8_0 test rather than being assumed.
- Allowed-only masked projection should inspect the bitmask before touching a
  packed weight row. Masked and CSR-candidate projections could also decode a
  complete q4_0 row block with one scale and 16 packed-byte reads.
- Spatial projection was weight-traffic bound on the block-2 direct path. A
  threadgroup could reuse each projection weight across several patches.
- Cache attention was context-serial within four SIMD groups; increasing
  context partitions could trade a small merge for much more parallelism.

Environment: Apple M4 Max MacBook Pro (16 CPU cores, 128 GB), macOS 26.5.1
(25F80), Xcode 26.6 (17F113), Apple Metal 32023.883 / Metal toolchain
17.6.109.0, Python 3.12.9, MLX 0.21.1, and PyTorch 2.12.1 MPS. Working-tree
label: `294f8bd-dirty`.

Measurement method: `perf/bench_kernels.py` with its one-second clock ramp,
at least 50 ms per-thunk warmup, adaptive batching to at least 2 ms/sample,
and a synchronization per sample. The fresh baseline used 10 requested
warmups and 30 samples; isolated candidates used 10/40; the final unified MLX
run used 15/50. Comprehensive MLX and MPS cache runs used 10/30. Every timing
below is a per-call median; p20/p80 and CV are retained in the raw JSONL.

Commands:

```bash
PYTHONPATH=bindings/python .venv/bin/python perf/bench_kernels.py \
  --backend mlx --preset quick \
  --kernel quantized_embedding,quantized_embedding_bag,decode_linear_epilogue,decode_swiglu,lm_head_masked,lm_head_candidates,space_to_depth_norm_linear,decode_cache_attention \
  --warmup 15 --iters 50 \
  --out-dir perf/results/2026-07-13/new-kernels-second-pass-final-quick
PYTHONPATH=bindings/python .venv/bin/python perf/bench_kernels.py \
  --backend mlx --preset comprehensive --kernel decode_cache_attention \
  --warmup 10 --iters 30 \
  --out-dir perf/results/2026-07-13/new-kernels-second-pass-cache-comprehensive
PYTHONPATH=bindings/python:bindings/pytorch_mps .venv/bin/python perf/bench_kernels.py \
  --backend torch --preset quick --kernel decode_cache_attention \
  --warmup 10 --iters 30 \
  --out-dir perf/results/2026-07-13/new-kernels-second-pass-cache-mps-quick
```

Controlled experiments (one meaningful factor at a time):

| Family / priority shape | Control ms | Candidate ms | Decision |
|---|---:|---:|---|
| q4_0 embedding T1 / T256, 128 -> 64 threads | 0.0104 / 0.0120 | 0.0097 / 0.0140 | Reject: T256 regressed 16%. |
| q4_0 bag B128xL8 / B32xL32, 128 -> 64 threads | 0.0130 / 0.0157 | 0.0172 / 0.0156 | Reject: priority B128 regressed 32%. |
| q4_0 decode epilogue / SwiGLU, paired lanes -> one block per lane | 0.0241 / 0.0285 | 0.0207 / 0.0256 | Keep: 14% / 10% faster. |
| q8_0 decode epilogue / SwiGLU, generic -> one block per lane | 0.0169 / 0.0265 | 0.0187 / 0.0372 | Reject against back-to-back generic control. |
| masked LM T1 L256 / T8 L64, project then mask -> mask first | 0.0332 / 0.1264 | 0.0296 / 0.0376 | Keep: 11% / 70% faster. Full-normalization mode still projects every row. |
| masked LM T1 / T8, generic q4_0 chunks -> whole-block decoder | 0.0296 / 0.0376 | 0.0279 / 0.0330 | Keep: 6% / 12% faster. |
| candidate LM T1 C256 / T8 C64, generic q4_0 chunks -> whole-block decoder | 0.0360 / 0.0164 | 0.0228 / 0.0154 | Keep: 37% / 6% faster. |
| masked LM T1 / T8, fixed tile 128 -> fixed tile 256 | 0.0279 / 0.0330 | 0.0287 / 0.0315 | Mixed; reject the global change. |
| masked LM T1 / T8, fixed tile 128 -> 128 below T4, otherwise 256 | 0.0279 / 0.0330 | 0.0272 / 0.0324 | Keep the routed tile; both branches pass the full LM suite. |
| spatial S2 / S4, reread input -> stage raw input in threadgroup memory | 0.7339 / 0.0848 | 0.7171 / 0.0840 | Reject: <=2.3%, not material or robust. |
| spatial S2, one patch -> four patches per threadgroup | 0.7339 | 0.3211 | Keep for dimension <=1024 and at least 256 patches; 56% faster. |
| spatial S2, four -> eight patches per threadgroup | 0.3211 | 0.3833 | Reject: 19% regression from lost parallelism. |
| cache T512 / T1024 / T2048, 4 -> 8 SIMD groups | 0.3210 / 0.6916 / 1.4389 | 0.2158 / 0.4099 / 0.8237 | Keep and continue sweep. |
| cache T512 / T1024 / T2048, 8 -> 16 SIMD groups | 0.2158 / 0.4099 / 0.8237 | 0.1507 / 0.3188 / 0.7466 | Keep and continue sweep. |
| cache T512 / T1024 / T2048, 16 -> 32 SIMD groups | 0.1507 / 0.3188 / 0.7466 | 0.1365 / 0.2484 / 0.4987 | Keep; hardware-limit 1024-thread group is best at every priority shape. |

Final MLX quick results are milliseconds. The baseline is the equivalent
framework/decomposed operation, not cache attention's separately recorded
pre-updated output-only SDPA. Each timing cell is `median [p20,p80], CV`.

| kernel | variant | target median [p20,p80], CV | equivalent baseline median | speedup | rel err |
|---|---|---:|---:|---:|---:|
| quantized_embedding | q4_0 R8192 D1024 T1 | 0.0105 [0.0102,0.0111], 0.168 | 0.0092 | 0.88x | 0.00e+00 |
| quantized_embedding | q4_0 R8192 D1024 T256 | 0.0123 [0.0116,0.0127], 0.216 | 0.0276 | 2.25x | 0.00e+00 |
| quantized_embedding_bag | q4_0 B128 L8 D1024 | 0.0137 [0.0134,0.0141], 0.138 | 0.0591 | 4.31x | 1.14e-07 |
| quantized_embedding_bag | q4_0 B32 L32 D1024 | 0.0190 [0.0186,0.0196], 0.121 | 0.0944 | 4.96x | 1.66e-07 |
| decode_linear_epilogue | dense B1 K1536 N4096 | 0.0599 [0.0543,0.0724], 0.121 | 0.0389 | 0.65x | 2.29e-07 |
| decode_linear_epilogue | q4_0 B1 K1536 N4096 | 0.0201 [0.0196,0.0211], 0.092 | 0.0440 | 2.19x | 2.40e-07 |
| decode_linear_epilogue | q8_0 B1 K1536 N4096 | 0.0309 [0.0306,0.0452], 0.213 | 0.0567 | 1.84x | 2.36e-07 |
| decode_linear_epilogue | q6_K B1 K1536 N4096 | 0.0407 [0.0374,0.0415], 0.069 | 0.0416 | 1.02x | 2.35e-07 |
| decode_swiglu | dense B1 K1536 N4096 | 0.1035 [0.0976,0.1114], 0.112 | 0.0659 | 0.64x | 1.48e-07 |
| decode_swiglu | q4_0 B1 K1536 N4096 | 0.0260 [0.0256,0.0266], 0.051 | 0.0692 | 2.66x | 2.74e-07 |
| decode_swiglu | q8_0 B1 K1536 N4096 | 0.0268 [0.0262,0.0275], 0.059 | 0.0661 | 2.46x | 2.71e-07 |
| decode_swiglu | q6_K B1 K1536 N4096 | 0.0452 [0.0449,0.0457], 0.019 | 0.0662 | 1.46x | 1.98e-07 |
| lm_head_masked | q4_0 T1 V8192 K1024 L256 | 0.0260 [0.0249,0.0273], 0.090 | 0.1302 | 5.00x | 0.00e+00 |
| lm_head_masked | q4_0 T8 V8192 K1024 L64 | 0.0328 [0.0322,0.0339], 0.081 | 0.1486 | 4.53x | 0.00e+00 |
| lm_head_candidates | q4_0 T1 V8192 K1024 C256 | 0.0231 [0.0227,0.0236], 0.112 | 0.1410 | 6.11x | 0.00e+00 |
| lm_head_candidates | q4_0 T8 V8192 K1024 C64 | 0.0144 [0.0140,0.0150], 0.094 | 0.1480 | 10.28x | 0.00e+00 |
| space_to_depth_norm_linear | B1 48x48 C128 O512 S2 | 0.2913 [0.2904,0.2925], 0.019 | 0.0637 | 0.22x | 3.84e-07 |
| space_to_depth_norm_linear | B1 32x32 C64 O256 S4 | 0.0916 [0.0894,0.0948], 0.037 | 0.0770 | 0.84x | 3.07e-07 |
| decode_cache_attention | B4 H16/4 T512 D128 | 0.1352 [0.1322,0.1380], 0.086 | 0.2727 | 2.02x | 9.52e-07 |
| decode_cache_attention | B4 H16/4 T1024 D128 | 0.2508 [0.2459,0.2609], 0.080 | 0.3673 | 1.46x | 1.25e-06 |
| decode_cache_attention | B4 H16/4 T2048 D128 | 0.4960 [0.4929,0.5022], 0.106 | 0.7048 | 1.42x | 1.68e-06 |

The comprehensive MLX cache case B16,H32/Hkv8,T4096,D128 measured 7.1676
[7.1217,7.2459] ms (CV 0.009) versus 8.7295 ms for the equivalent functional
step, a 1.22x win with 2.68e-6 relative error. MPS measured 0.1592/0.2598/
0.5247 ms at T512/1024/2048 versus equivalent 0.6482/1.1121/2.0271 ms, and
7.1023 ms at the comprehensive T4096 case versus 27.9344 ms. The public
`decode_cache_attention` default therefore returns to the Metal path for all
measured sizes; `use_kernel=False` preserves the framework fallback.
The smoke B1,H4/Hkv2,T64,D64 case measured 0.0449 ms versus 0.2254 ms on MLX
and 0.0963 ms versus 0.5391 ms on MPS, closing the small-context side of that
routing decision.

Correctness and validation:

- `scripts/build kernels` passed with the retained q4_0 decoder, group-of-four
  spatial kernel, and 32-SIMD-group cache kernel.
- The five focused test modules passed 187 tests.
- `scripts/test correctness -q` passed 2025 tests.
- `scripts/test parity -q` passed 401 tests.
- `scripts/test mps -q` passed 472 tests.
- All benchmark cases passed their in-run oracle. The maximum relative error
  was 2.75e-6 across the final quick/comprehensive MLX and MPS runs; embedding
  lookup and both sparse projection families were exact.

Final decisions:

- Retain 128-thread embedding and bag launches; reject 64 threads.
- Keep whole-q4_0-block decode in fused decode and sparse LM-head dots. Keep
  the existing generic q8_0 decoder.
- Keep mask-first allowed normalization and route masked vocabulary tiles to
  128 rows below four tokens, 256 otherwise.
- Keep four-patch spatial weight reuse only for dimensions <=1024 with at
  least 256 patches. The public high-work framework crossover remains because
  the direct priority shapes are still slower than framework composition.
- Keep 32 SIMD groups for functional cache decode and make Metal the automatic
  public route across the measured 64-through-4096-slot range.
- Dense fused decode and untouched q6_K/q8_0 paths retain their existing public
  routing. No speedup is attributed to unchanged or rejected variants.

Raw results:

- Baseline and final: `perf/results/2026-07-13/new-kernels-second-pass-baseline-quick/`,
  `perf/results/2026-07-13/new-kernels-second-pass-final-quick/`.
- Embedding/decode: `new-kernels-second-pass-embedding-64/`,
  `new-kernels-second-pass-decode-q4-full-block/`,
  `new-kernels-second-pass-decode-q8-full-block/`, and
  `new-kernels-second-pass-decode-q8-control-generic/` under the same date.
- Sparse projection: `new-kernels-second-pass-lm-mask-first/`,
  `new-kernels-second-pass-lm-q4-block-decode/`,
  `new-kernels-second-pass-lm-mask-tile256/`, and
  `new-kernels-second-pass-lm-mask-dynamic-tile/`.
- Spatial: `new-kernels-second-pass-space-stage-input/`,
  `new-kernels-second-pass-space-stage-control/`,
  `new-kernels-second-pass-space-group4/`, and
  `new-kernels-second-pass-space-group8/`.
- Cache: `new-kernels-second-pass-cache-4simd-control/`,
  `new-kernels-second-pass-cache-8simd/`,
  `new-kernels-second-pass-cache-16simd/`,
  `new-kernels-second-pass-cache-32simd/`,
  `new-kernels-second-pass-cache-smoke/`,
  `new-kernels-second-pass-cache-comprehensive/`,
  `new-kernels-second-pass-cache-mps-smoke/`,
  `new-kernels-second-pass-cache-mps-quick/`, and
  `new-kernels-second-pass-cache-mps-comprehensive/`.

## 2026-07-13: Cross-kernel follow-ups and optimization pass

Status: complete. This entry implements the follow-ups identified after the
new-kernel passes: whole-block q4_0 LM-head decode, factorized pairwise MLP,
quantized LM-head beam advance, matrix-tiled spatial projection, and
context-partitioned head-major decode attention.

Hypotheses:

- The LM-head sampler's sequential q4_0 dot paid generic dequant helper and
  scale-address overhead four times per block; decoding all 32 weights after
  one scale load should remove most of that cost.
- A pairwise 512->256 first layer is separable into one left and one right
  sequence projection. Projecting each sequence once should replace an
  O(L^2*512*256) stage with O(L*256*256), leaving only the small pairwise GELU
  and 256->7 combine at O(L^2).
- Beam advance needs only exact row log-sum-exp and top-2*beam-width candidates,
  not a resident full-logits tensor. A staged quantized projection/reduce can
  preserve exact beam scores; larger row batches should instead reuse weights
  through the existing packed 32-column GEMM.
- Spatial projection is a matrix multiply hidden inside gather and LayerNorm.
  An 8-patch matrix tile should reuse normalized features and engage SIMD-group
  matrix operations without materializing the merged tensor.
- Generic head-major decode is context-serial. Compile-time 8- and 32-SIMD-group
  partitions can expose long-context parallelism, then merge online-softmax
  state in threadgroup memory.

Environment: MacBook Pro Mac16,5, Apple M4 Max, 128 GB; macOS 26.5.1 (25F80);
Xcode 26.6 (17F113); Apple Metal 32023.883 / Metal toolchain 17.6.109.0;
Python 3.12.9; MLX 0.21.1; PyTorch 2.12.1 MPS. Working-tree label:
`bc90717-dirty`.

Measurement method: `perf/bench_kernels.py` with its clock ramp, adaptive
per-sample batching, and synchronization per sample. The initial baseline and
final unified runs requested 10 warmups and 30 measured samples. Isolated q4_0
LM-head and edge candidates used 10/40; all other controlled variants used
10/30. Tables report per-call median, p20/p80, and coefficient of variation
(CV); raw JSONL retains the complete result records.

Final commands:

```bash
.venv/bin/python perf/bench_kernels.py --backend mlx --preset quick \
  --kernel lm_head_q,lm_head_beam,edge_mlp,space_to_depth_norm_linear,attn_decode_bh \
  --formats q4_0,q8_0 --warmup 10 --iters 30 \
  --out-dir perf/results/2026-07-13/cross-kernel-final-mlx
PYTHONPATH=bindings/python:bindings/pytorch_mps .venv/bin/python \
  perf/bench_kernels.py --backend torch --preset quick \
  --kernel lm_head_q,lm_head_beam,edge_mlp,space_to_depth_norm_linear,attn_decode_bh \
  --formats q4_0,q8_0 --warmup 10 --iters 30 \
  --out-dir perf/results/2026-07-13/cross-kernel-final-mps
scripts/test correctness -q
scripts/test parity -q
scripts/test mps -q
```

Controlled experiments, one meaningful factor at a time:

| Family / priority shape | Control ms | Candidate ms | Decision |
|---|---:|---:|---|
| LM-head q4_0 top-k T1/T8, generic 8-value helper -> whole 32-value block | 2.9924 / 5.5223 | 0.2758 / 1.2047 | Keep; 10.85x / 4.58x. The final unified sample is 0.2786 / 1.2737 ms. |
| qgemv f32 q4_0 N6144 K1536, paired-block mapping -> whole block per lane | 0.0189 | 0.0271 | Reject and revert; 43% regression. The LM-head result does not generalize to parallel qgemv. |
| Pairwise edge MLP B1 L64, direct pair kernel -> factorized project/combine | 1.3031 | 0.1353 | Keep; 9.63x over the old kernel and 1.38x over the materialized framework composition. |
| Quantized beam q8_0 smoke, scalar block decode -> four-block vector attempt | 0.2340 | 1.8109 | Reject and revert. |
| Quantized beam tile rows, one -> four rows per threadgroup | 0.2340 q8 smoke | 0.1168 | Keep in the no-logits partials launch. |
| Quantized beam full shape, row-wise fusion -> routed q4 fusion / packed GEMM | q4/q8 B4 2.4258 / 3.6890 | 1.0053 / 1.0080 | Keep matrix route for rows >4 and regular vocab widths; 2.41x / 3.66x. |
| Spatial S2/S4, group-of-four scalar projection -> 8x32 matrix tile | 0.3152 / 0.0978 | 0.1821 / 0.0404 | Keep; 1.73x / 2.42x over the prior direct kernel. |
| Spatial matrix tile, 32 -> 64 output columns | 0.1700 / 0.0407 | 0.1815 / 0.0328 | Mixed; reject globally because S2 regressed. |
| Spatial tile width selected at runtime instead of compile time | 0.1700 / 0.0407 | 0.2443 / 0.0436 | Reject; dynamic geometry inhibited Metal specialization. |
| Attention T512/T2048, one context SIMD group -> eight | 0.5451 / 5.2582 | 0.0664 / 0.5826 | Keep eight groups for T512-2047; 8.21x / 9.03x over serial. |
| Attention T512/T2048, eight -> four groups | 0.0664 / 0.5826 | 0.1035 / 0.7054 | Reject. |
| Attention T512/T1024/T2048, routed 8/8/32 -> sixteen groups | 0.0665 / 0.1401 / 0.3441 | 0.0718 / 0.2177 / 0.4837 | Reject at every priority context. |
| Attention T2048, eight -> 32 groups | 0.5826 | 0.3526 | Keep from T2048; 1.65x over eight groups and 14.91x over serial. |
| Attention shared allocation, 8-slot -> 32-slot while launching eight groups | 0.0664 | 0.1062 | Reject shared generic state; retain separate compile-time 8/32 specializations. |

Final MLX quick results:

| Kernel / shape | Target median [p20,p80], CV ms | Best baseline ms | Speedup | Error |
|---|---:|---:|---:|---:|
| LM-head q4_0 T1 V32000 K4096 | 0.2786 [0.2725,0.3099], 0.130 | 5.7441 dense top-k | 20.62x | exact ids in tests |
| LM-head q4_0 T8 V32000 K4096 | 1.2737 [1.1930,1.3664], 0.061 | 5.8540 dense top-k | 4.60x | exact ids in tests |
| Beam q4_0 B1 BM4 V32000 K4096 | 0.7464 [0.7305,0.9175], 0.111 | 0.9799 resident fp16 | 1.31x | exact token/parent in tests |
| Beam q8_0 B1 BM4 V32000 K4096 | 1.0183 [0.9681,1.1375], 0.074 | 1.0039 resident fp16 | 0.99x | exact token/parent in tests |
| Beam q4_0 B4 BM4 V32000 K4096 | 1.0053 [0.9675,1.1331], 0.082 | 1.2019 packed GEMM | 1.20x | exact token/parent in tests |
| Beam q8_0 B4 BM4 V32000 K4096 | 1.0080 [0.9727,1.1527], 0.108 | 1.1805 packed GEMM | 1.17x | exact token/parent in tests |
| Edge MLP B1 L64 | 0.1399 [0.1237,0.2271], 0.284 | 0.1877 framework | 1.34x | 3.10e-7 rel |
| Spatial B1 48x48 C128 O512 S2 | 0.1769 [0.1664,0.2273], 0.181 | 0.0688 framework | 0.39x | 3.49e-7 rel |
| Spatial B1 32x32 C64 O256 S4 | 0.0445 [0.0427,0.0496], 0.127 | 0.0833 framework | 1.87x | 3.07e-7 rel |
| Attention B4 H16/4 T512 D128 | 0.0703 [0.0645,0.0747], 0.229 | 0.0326 MLX SDPA | 0.46x | 8.53e-7 rel |
| Attention B4 H16/4 T1024 D128 | 0.1586 [0.1422,0.2022], 0.251 | 0.0567 MLX SDPA | 0.36x | 1.42e-6 rel |
| Attention B4 H16/4 T2048 D128 | 0.3545 [0.3389,0.3933], 0.085 | 0.1326 MLX SDPA | 0.37x | 1.47e-6 rel |

Final MPS quick results:

| Kernel / shape | Target median [p20,p80], CV ms | Framework baseline ms | Speedup | Error |
|---|---:|---:|---:|---:|
| Beam q4_0 B1 BM4 | 0.9125 [0.8907,1.0647], 0.086 | 1.2525 | 1.37x | parity exact/2e-5 score |
| Beam q8_0 B1 BM4 | 1.2240 [1.1630,1.4088], 0.106 | 1.2144 | 0.99x | parity covered through matrix route |
| Beam q4_0 B4 BM4 | 1.2101 [1.1829,1.4594], 0.106 | 1.3945 | 1.15x | parity exact/2e-5 score |
| Beam q8_0 B4 BM4 | 1.2241 [1.1627,1.4572], 0.112 | 1.3466 | 1.10x | parity covered through matrix route |
| Edge MLP B1 L64 | 0.0943 [0.0924,0.1023], 0.299 | 0.1603 | 1.70x | 3.10e-7 rel |
| Spatial B1 48x48 C128 O512 S2 | 0.1961 [0.1895,0.2252], 0.176 | 0.1237 | 0.63x | 3.49e-7 rel |
| Spatial B1 32x32 C64 O256 S4 | 0.0972 [0.0958,0.1056], 0.198 | 0.1451 | 1.49x | 2.79e-7 rel |
| Attention B4 H16/4 T512 D128 | 0.1212 [0.1176,0.1323], 0.276 | 0.4256 | 3.51x | 8.53e-7 rel |
| Attention B4 H16/4 T1024 D128 | 0.2824 [0.2745,0.3609], 0.127 | 0.9509 | 3.37x | 1.42e-6 rel |
| Attention B4 H16/4 T2048 D128 | 0.3799 [0.3687,0.4407], 0.109 | 1.7871 | 4.70x | 1.33e-6 rel |

Correctness and validation:

- Final Python/MLX and PyTorch MPS builds completed with the retained kernels.
- Focused suites passed 81 LM-head, 4 edge-MLP, 35 spatial, and 32 attention
  tests. These include irregular vocabularies, fp32/bf16, tiled block-2/block-4
  spatial shapes, and the 8-/32-partition attention thresholds.
- `scripts/test correctness -q`: 2062 passed.
- `scripts/test parity -q`: 406 passed, including q4_0 beam advance, factorized
  edge MLP, matrix-tiled spatial projection, and both long attention routes.
- `scripts/test mps -q`: 472 passed.
- Every numeric final benchmark case passed its oracle; maximum recorded
  relative error was 1.47e-6. Structured LM-head/beam outputs are validated by
  exact-id tests rather than the benchmark's scalar error field.

Final decisions and routing:

- Keep whole-block q4_0 decode only in sequential LM-head sampler dots. The
  qgemv experiment regressed and was fully reverted; q8_0 decoding is unchanged.
- Keep factorized edge project/combine and make it the public default. It wins
  at both L8 (0.0260 vs 0.1008 ms) and L64; `use_kernel=False` retains the
  materialized framework composition.
- Keep exact staged quantized beam advance. Q4_0 with at most four rows uses the
  no-logits path. Q8_0 and larger row batches use packed 32-column GEMM when the
  vocab permits it; irregular widths retain the exact staged fallback. Do not
  claim a q8_0 B1 win over resident fp16 weights.
- Keep the compile-time 8x32 spatial tile. Auto-route qualifying fp32 block-4
  shapes to Metal and keep the realistic block-2 priority shape on framework;
  small direct workloads retain the existing Metal route.
- Keep barrier-free single-warp attention below T512, eight partitions for
  T512-2047, and 32 from T2048. Auto-route MPS to Metal at all measured shapes;
  on MLX use Metal below T512 and framework SDPA for longer contexts.

Raw results:

- Baseline: `perf/results/2026-07-13/cross-kernel-followups-baseline/`.
- Final: `cross-kernel-final-mlx/`, `cross-kernel-final-mps/`, and
  `cross-kernel-edge-smoke/` under the same date.
- LM-head/qgemv: `cross-kernel-lm-head-q4-whole-block/`,
  `cross-kernel-qgemv-f32-paired-control/`,
  `cross-kernel-qgemv-f32-q4-full-block/`, and
  `cross-kernel-qgemv-q4-full-block/`.
- Beam: `cross-kernel-lm-head-beam-smoke/`,
  `cross-kernel-lm-head-beam-rows4/`,
  `cross-kernel-lm-head-beam-rows4-quick/`,
  `cross-kernel-lm-head-beam-q8-vec4/`, and
  `cross-kernel-lm-head-beam-final-routing/`.
- Spatial: `cross-kernel-spatial-tiled/`,
  `cross-kernel-spatial-tiled64-fixed/`,
  `cross-kernel-spatial-tiled-adaptive/`, and
  `cross-kernel-spatial-tiled-final/`.
- Attention: `cross-kernel-attn-decode-baseline/`,
  `cross-kernel-attn-decode-partition4/`,
  `cross-kernel-attn-decode-partition8/`,
  `cross-kernel-attn-decode-partition16/`,
  `cross-kernel-attn-decode-partition32/`,
  `cross-kernel-attn-decode-shared32-launch8/`, and
  `cross-kernel-attn-decode-final-route-repeat/`.

## 2026-07-14: MXFP8 inference coverage completion

Status: complete for coverage. This pass instantiates the existing MXFP8
decoders across packed embedding lookup/bag, decode linear epilogues and
SwiGLU, LM-head sampling/sparse projection/beam advance, quantized MoE, and
D64/D128 quantized-KV attention. It is a compatibility and baseline pass, not
a speedup claim for every newly covered path.

Hypotheses:

- A shared float32 MXFP8 span decoder can preserve the host dequantization
  oracle for sequential embedding, decode, and sparse LM-head consumers while
  leaving the existing half-accumulation QGEMV fast path unchanged.
- LM-head filtering should benefit most because masked and candidate
  projection avoid materializing the full vocabulary even when MXFP8 decode is
  added to the fused kernel.
- The existing MXFP8 MoE column decoder should make the format immediately
  usable once the Metal template and host layout are instantiated, although
  its current schedule may not beat resident bf16 weights.
- MXFP8 quantized-KV attention should reuse the existing single- and
  multi-warp format abstraction for D64/D128. Its storage saving over the
  existing FP8-E4M3 layout is only one byte per 32 values, so compatibility is
  the primary objective.

Environment: MacBook Pro Mac16,5, Apple M4 Max, 40 GPU cores, 128 GB, Metal 4;
macOS 26.5.1 (25F80); Xcode 26.6 (17F113); Apple Metal 32023.883 / Metal
toolchain 17.6.109.0; Python 3.12.9; MLX 0.21.1; power mode 0. Working-tree
label: `3cab797-dirty`.

Measurement method: `perf/bench_kernels.py` with its clock ramp, adaptive
per-sample batching, and synchronization per sample. The focused MLX quick run
used 10 warmups and 40 measured samples. The table reports per-call median,
p20/p80, and coefficient of variation (CV). Each baseline is the fastest
equivalent resident, predequantized, or existing-kernel control recorded in the
same result row.

Command:

```bash
PYTHONPATH=bindings/python .venv/bin/python perf/bench_kernels.py \
  --backend mlx --preset quick \
  --kernel quantized_embedding,quantized_embedding_bag,decode_linear_epilogue,decode_swiglu,lm_head_q,lm_head_masked,lm_head_candidates,lm_head_beam,moe_q,attn_q \
  --formats mxfp8 --warmup 10 --iters 40 \
  --out-dir perf/results/2026-07-14/mxfp8-coverage-generic
```

Focused coverage results:

| Kernel / priority shape | Target median [p20,p80], CV ms | Best control ms | Relative | Error / oracle |
|---|---:|---:|---:|---:|
| Embedding T1 R8192 D1024 | 0.0118 [0.0107,0.0156], 0.188 | 0.0113 predequantized | 0.96x | exact |
| Embedding T256 R8192 D1024 | 0.0216 [0.0211,0.0228], 0.150 | 0.0500 predequantized | 2.31x | exact |
| Embedding bag B128 L8 D1024 | 0.0233 [0.0229,0.0256], 0.097 | 0.1365 predequantized | 5.85x | 1.13e-7 rel |
| Embedding bag B32 L32 D1024 | 0.0168 [0.0164,0.0214], 0.185 | 0.0894 predequantized | 5.32x | 1.99e-7 rel |
| Decode epilogue B1 K1536 N4096 | 0.0661 [0.0452,0.0712], 0.218 | 0.0465 expanded matmul | 0.70x | 2.37e-7 rel |
| Decode SwiGLU B1 K1536 N4096 | 0.0770 [0.0735,0.1205], 0.246 | 0.0723 expanded matmuls | 0.94x | 1.99e-7 rel |
| LM top-k T1 V32000 K4096 | 0.5073 [0.5008,0.5160], 0.031 | 5.0474 dense top-k | 9.95x | exact ids in tests |
| LM top-k T8 V32000 K4096 | 2.0743 [2.0415,2.1159], 0.019 | 5.1279 dense top-k | 2.47x | exact ids in tests |
| LM masked T1 V8192 K1024 L256 | 0.0783 [0.0740,0.1119], 0.196 | 0.1633 full projection | 2.09x | exact |
| LM masked T8 V8192 K1024 L64 | 0.0548 [0.0534,0.0578], 0.078 | 0.1626 full projection | 2.97x | exact |
| LM candidates T1 V8192 K1024 C256 | 0.0867 [0.0849,0.0917], 0.074 | 0.1392 gathered dense rows | 1.61x | exact |
| LM candidates T8 V8192 K1024 C64 | 0.0158 [0.0155,0.0176], 0.157 | 0.1425 gathered dense rows | 9.00x | exact |
| LM beam B1 BM4 V32000 K4096 | 1.1437 [1.1290,1.1629], 0.016 | 0.9419 resident fp16 | 0.82x | exact token/parent in tests |
| LM beam B4 BM4 V32000 K4096 | 1.0007 [0.9899,1.0200], 0.017 | 1.1820 packed GEMM + beam | 1.18x | exact token/parent in tests |
| MoE rect K/N2880, 32 routed rows | 0.1199 [0.1064,0.1457], 0.275 | 0.0590 resident bf16 | 0.49x | benchmark oracle passed |
| MoE rect K/N2880, 512 routed rows | 0.7135 [0.7051,0.7292], 0.028 | 0.6292 resident bf16 | 0.88x | benchmark oracle passed |
| MoE SwiGLU H/I2880, 32 routed rows | 0.3045 [0.2651,0.3887], 0.279 | 0.1325 resident bf16 | 0.44x | benchmark oracle passed |
| MoE SwiGLU H/I2880, 512 routed rows | 3.1456 [3.0944,3.2304], 0.022 | 1.2875 resident bf16 | 0.41x | benchmark oracle passed |
| Attention B1 H8 N1024 D128 | 0.4811 [0.4743,0.4945], 0.029 | 0.4186 dequantized attention | 0.87x | benchmark oracle passed |
| Attention multi-warp B1 H8 N1024 D128 | 0.4794 [0.4756,0.5012], 0.036 | 0.4770 single-warp | 0.99x | benchmark oracle passed |

Correctness and validation:

- All 22 numeric and structured rows in the focused performance run completed
  with status `ok`; the largest reported scalar relative error was 2.37e-7.
- The affected MLX correctness modules passed 252 tests, including MXFP8
  lookup/bag, both decode epilogues, LM-head routes, rectangular and SwiGLU
  MoE, and single-/multi-warp causal and noncausal attention at D64/D128.
- `scripts/test correctness -q`: 2155 passed.
- `scripts/test parity -q`: 432 passed.
- `scripts/test mps -q`: 472 passed.
- `scripts/test xcode`: test build succeeded.

Decisions and follow-ups:

- Keep every instantiation and host-format addition: the coverage paths are
  correct, public validation is explicit, and format compatibility is now
  complete for the requested surfaces.
- The embedding bag and filtered LM-head paths are already useful wins. Keep
  the existing row-count routing for beam advance: B4 reaches the packed GEMM
  path, while the B1 fused MXFP8 route remains an optimization target.
- Do not claim a speedup for MXFP8 decode, MoE, or quantized-KV attention. The
  generic float32 decoder, current MoE scheduling, and attention decode/layout
  are the next hot-path candidates. Multi-warp MXFP8 attention did not improve
  N1024 D128 and should not be preferred on this evidence alone.

Raw results: `perf/results/2026-07-14/mxfp8-coverage-generic/`.

## 2026-07-14: MXFP8 inference hot-path experiments

Status: complete. Retained variants pass focused and repository-wide
correctness, parity, MPS, Metal, and Xcode validation. Performance claims are
limited to the measured MLX integration path and shapes below.

Current implementation and public route:

- QGEMV assigns one complete 32-value MXFP8 block to each lane. One E8M0
  expansion feeds all 32 E4M3 values instead of four independent 8-value span
  decoders.
- MLX MXFP8 QGEMM keeps the direct-fragment kernel for `M%64 != 0`, including
  the M32 edge, and uses a two-warp staged kernel for `M%64 == 0`. The two
  warps share each decoded 32x32 weight tile and produce 64 output columns.
  PyTorch MPS QGEMM retains its direct-fragment route.
- Masked LM-head projection uses a path-local 256-entry E4M3-to-half constant
  table and one fp32 E8M0 reconstruction per complete block. CSR candidate
  projection uses the same complete-block schedule with arithmetic E4M3
  decode; the table is intentionally not global.
- MXFP8 beam advance uses packed QGEMM plus the established exact beam kernel
  even at four rows. Other four-row formats retain their existing no-logits
  fusion where measured.
- MXFP8 MoE SwiGLU uses a two-warp gate/up split. Lanes covering the same
  column-fragment row pair broadcast two E8M0 scales with SIMD shuffles. The
  rectangular MoE path retains its original decoder and four-warp schedule.
- Quantized-KV attention retains its original generic 8-value shared decoder,
  four-warps, and four-tile staging depth. Every attention-specific candidate
  was neutral or slower.
- The canonical `{E8M0, 32 E4M3 codes}` host layout is unchanged. The
  experimental 32-scale/1024-code split-plane form was removed.

References inspected: repository MXFP8 format and decoder contracts, existing
MXFP4/NVFP4 complete-block kernels, the q4_0 whole-block QGEMV, staged/direct
QGEMM implementations, MoE split-K reduction, and quantized-KV attention
staging code. No external implementation code was imported.

Environment and method:

- MacBook Pro Mac16,5; Apple M4 Max, 40 GPU cores, 128 GB; macOS 26.5.1
  (25F80); Xcode 26.6 (17F113); Apple Metal 32023.883, toolchain
  17.6.109.0; Python 3.12.9; MLX 0.21.1.
- Working-tree label: `455463c-dirty`.
- Integration path: MLX Python extension, packed MXFP8 weights, fp16
  QGEMV/QGEMM/beam inputs, bf16 MoE/attention inputs, and fp32/bf16 LM-head
  correctness coverage.
- The harness performs clock ramp, adaptive per-sample batching, and a device
  synchronization per sample. Baseline, final, and most controlled candidates
  requested 15 warmups and 60 measured samples and report per-call median,
  p20/p80, and CV. Some narrowing runs used 10/40; the split-plane repeat used
  30/200 because the first run was noisy.

Primary commands:

```bash
PYTHONPATH=bindings/python .venv/bin/python perf/bench_kernels.py \
  --backend mlx --preset quick --kernel qgemv,qgemm,qflux \
  --formats mxfp8 --warmup 15 --iters 60 \
  --out-dir perf/results/2026-07-14/mxfp8-experiments-baseline-core
PYTHONPATH=bindings/python .venv/bin/python perf/bench_kernels.py \
  --backend mlx --preset quick \
  --kernel decode_linear_epilogue,decode_swiglu,lm_head_q,lm_head_masked,lm_head_candidates,lm_head_beam,moe_q,attn_q \
  --formats mxfp8 --warmup 15 --iters 60 \
  --out-dir perf/results/2026-07-14/mxfp8-experiments-baseline-fused
PYTHONPATH=bindings/python .venv/bin/python perf/bench_kernels.py \
  --backend mlx --preset quick \
  --kernel qgemv,qgemm,lm_head_masked,lm_head_candidates,lm_head_beam,moe_q,attn_q \
  --formats mxfp8 --warmup 15 --iters 60 \
  --out-dir perf/results/2026-07-14/mxfp8-experiments-final-quick
```

Retained controlled results (milliseconds; brackets are p20/p80, followed by
CV):

| Path / shape | Control | Candidate | Change | Decision |
|---|---:|---:|---:|---|
| QGEMV N4096 K4096 | span 0.03695 [0.03572/0.03754], .0408 | whole block 0.03558 [0.03463/0.03630], .0528 | -3.7% | keep |
| QGEMV N11008 K4096 | span 0.11153 [0.10928/0.11378], .0367 | whole block 0.10659 [0.10388/0.11195], .0469 | -4.4% | keep |
| Masked T1 V8192 K1024 L256 | arithmetic 0.04007 [0.03914/0.04290], .0784 | narrow LUT 0.02656 [0.02564/0.02816], .1924 | -33.7% | keep masked-only LUT |
| Masked T8 V8192 K1024 L64 | arithmetic 0.05262 [0.05012/0.05579], .0709 | narrow LUT 0.03556 [0.03417/0.03905], .0981 | -32.4% | keep masked-only LUT |
| Candidates T1 V8192 K1024 C256 | spans 0.06031 [0.05875/0.06388], .1012 | whole block 0.05534 [0.05405/0.05627], .0587 | -8.2% | keep arithmetic whole block |
| Candidates T8 V8192 K1024 C64 | spans 0.01925 [0.01802/0.02313], .1797 | whole block 0.01718 [0.01662/0.01802], .1396 | -10.7% | keep arithmetic whole block |
| Beam B1 BM4 V32000 K4096 | no-logits 1.13108 [1.12290/1.18208], .0501 | matrix 1.01334 [0.99763/1.06385], .0450 | -10.4% | keep matrix route |
| MoE SwiGLU H/I2880 rows32 | original 0.26927 [0.25432/0.28059], .0620 | two warp 0.18980 [0.18711/0.20329], .0606 | -29.5% | keep |
| MoE SwiGLU H/I2880 rows512 | original 3.10281 [3.08138/3.13913], .0113 | two warp 1.49516 [1.48942/1.50296], .0068 | -51.8% | keep |

The QGEMM 2x32 run compares the retained staged target against
`tk.qgemm_direct` in every row of the same comprehensive run. At N4096, staged
M128/M256/M512 improved 2.2%/2.6%/1.1%; at N11008, M64/M128/M512 improved
3.2%/0.7%/2.9%. N4096 M64 was flat and staged N11008 M256 was 0.5% slower,
within the overlapping central bands. M32 always remains direct.

The equivalent PyTorch MPS beam A/B also favored the shared route: B1/BM4
fell from 1.3988 ms with no-logits fusion to 1.2317 ms through matrix
projection (-11.9%). B4 was already matrix-routed and remained flat at
1.23 ms. Both MPS runs requested 15 warmups and 60 measured samples.

Rejected experiments:

| Factor | Representative result | Decision |
|---|---|---|
| Explicit MXFP8 8-value half decoder | LM top-k T1 rose from 0.3541 to 0.5274 ms; QGEMV and attention did not establish wins | reject; compiler-generated generic path is better |
| Complete-block sequential decode everywhere | Decode epilogue 0.02087 -> 0.02214 ms, SwiGLU 0.03339 -> 0.03468, top-k T1 0.35406 -> 0.37178; sparse projections improved | retain only sparse-projection specializations |
| Gate/up interleaved decode SwiGLU | 0.03339 -> about 0.0360 ms | reject |
| Vector E4M3 `decode4` | QGEMV 0.03170/0.10923 -> 0.0340/0.1106 ms; sparse results mixed | reject |
| Global E4M3 LUT | QGEMV regressed to 0.0552/0.1576 ms and candidate/MoE large shapes regressed, although masked projection improved | narrow to masked projection only |
| Combined E8M0+E4M3 exponent reconstruction | QGEMV rose to 0.1124/0.2800 ms and sparse projection regressed by an order of magnitude | reject |
| MoE scale broadcast on every column decoder | Rectangular rows512 rose from 0.7103 to 0.7605 ms | restrict to SwiGLU |
| Four-warp/two-warp dual-symbol MoE hybrid | Duplicate specialization raised rows32 to 0.253-0.354 ms; all-two-warp remained 0.1898 ms and won the large shape | keep the single two-warp MXFP8 symbol |
| MXFP8 attention scale broadcast | single/multi-warp rose from 0.4772/0.4848 to 0.5189/0.5196 ms | reject |
| Direct register V decode | 0.4745 vs 0.4772 ms at N1024 D128, with no durable comprehensive advantage | reject as flat/complex |
| Two-warp attention | multi-warp 0.7383 vs four-warp repeat 0.4788 ms | reject |
| Attention staging depth 1/2/4 | 0.4786/0.4763/0.4788 ms with overlapping bands | keep established depth 4 |
| QGEMM four warps x 32 columns | improved several N4096 cases but regressed N11008 M128/M256 about 1%; 2x32 was broader | reject 4x32, keep 2x32 |
| Split-plane 32-scale/1024-code layout | N4096 0.03212 -> 0.03103 ms, but N11008 0.10481 -> 0.11183 (+6.7%) and K%1024 required | reject and remove layout |

Correctness and validation:

- Focused final selection: 32 MXFP8 QGEMV/QGEMM/LM-head/MoE/attention tests
  passed. QGEMV final relative errors were 6.38e-6 and 2.22e-5; masked and
  candidate benchmark selected-id errors were zero. Beam token/parent outputs
  are exact in the correctness suite.
- `scripts/build kernels` and `scripts/build pytorch_mps` passed.
- `scripts/test correctness -q`: 2155 passed.
- `scripts/test parity -q`: 432 passed.
- `scripts/test mps -q`: 472 passed.
- `scripts/test xcode`: test build succeeded.

Decision: keep the six scoped improvements above. Do not change the MXFP8
wire layout or quantized-KV attention path based on this pass. The remaining
attention gap is dominated by the attention schedule rather than MXFP8 scale
decode, and the tested dequant/layout changes did not move it safely.

Raw results: baseline and final are
`mxfp8-experiments-baseline-core`, `mxfp8-experiments-baseline-fused`, and
`mxfp8-experiments-final-quick`. Retained controlled runs are
`mxfp8-exp-qgemv-whole-vs-span-interleaved`,
`mxfp8-exp-sequential-whole-block`, `mxfp8-exp-masked-lut-narrow`,
`mxfp8-exp-moe-scale-shuffle-swiglu-only`,
`mxfp8-exp-moe-swiglu-2warp`, `mxfp8-exp-beam-matrix-all`,
`mxfp8-exp-beam-row-control`, `mxfp8-exp-beam-matrix-mps`,
`mxfp8-exp-beam-row-control-mps`, and
`mxfp8-exp-qgemm-2x32-comprehensive`.
Rejected runs are the other `mxfp8-exp-*` directories under
`perf/results/2026-07-14/`, including the attention, global LUT, combined
exponent, 4x32 QGEMM, hybrid MoE, and split-plane variants.

## 2026-07-14: FP8 inference hot-path experiments

Status: candidate; the retained implementation passes the focused kernel
build and 389 affected correctness cases. The changes are measured but not yet
committed.

Current implementation:

- E4M3 and E5M2 normal-value encoders derive the unbiased exponent and rounded
  mantissa directly from IEEE-754 bits. Their existing arithmetic subnormal,
  saturation, NaN, and infinity handling remains in place.
- FP8 paged-attention v1, v2 partition, and cascade-prefix kernels move the K
  scale into the score multiplier and apply the V scale once after online
  softmax normalization. E4M3 and E5M2 decode selection is compile-time in
  distinct pipeline symbols; the public format argument and cache layout are
  unchanged.
- FP8 E4M3 SwiGLU MoE uses the measured two-warp gate/up kernel. Rectangular
  MoE retains its established four-warp kernel.
- The benchmark harness now covers rank-1-scaled FP8 GEMM, 128x128-block-scaled
  FP8 GEMM, D64/D128 E4M3/E5M2 KV scatter and paged attention, a populated FP8
  MLA cache, and fused FP8 activation quantization.

References inspected: the repository's existing E4M3/E5M2 encoder and decoder
contracts, FP8 KV-cache/paged-attention paths, FP8/MXFP8 grouped MoE kernels,
and established QGEMV/QGEMM topology. No external implementation code was
imported.

Hypotheses:

- FP8 attention pays avoidable scale multiplies and a uniform runtime format
  branch for every K/V element.
- FP8 producers are limited by `frexp`/`ldexp` and division on the overwhelmingly
  normal-value encoding path.
- Fused FP8 SwiGLU repeats gate/up dequantization work and may benefit from the
  two-warp topology established for byte-per-value MXFP8.
- Dense/blocked GEMM, QGEMV, MLA, and quantized attention might benefit from
  broader scale sharing or additional warp parallelism, but those changes must
  beat the current priority shapes rather than only reduce source-level work.

Environment and method:

- MacBook Pro Mac16,5; Apple M4 Max, 40-core GPU, 128 GB; macOS 26.5.1
  (25F80); Xcode 26.6 (17F113); Apple Metal 32023.883 / toolchain
  17.6.109.0; Python 3.12.9; MLX 0.21.1; PyTorch 2.12.1.
- Working-tree label `455463c-dirty`; MLX Python-extension integration path;
  BF16/FP16 activations and FP8 E4M3/E5M2 or FP8-block-scaled weights/caches.
- All reported focused runs requested 15 warmups and 60 measured samples. The
  harness also performs its clock ramp and per-thunk warmup, adaptively batches
  calls, synchronizes samples, and records per-call median, p20/p80, and CV.
- Initial baselines:

```bash
PYTHONPATH=bindings/python .venv/bin/python perf/bench_kernels.py \
  --backend mlx --preset quick \
  --kernel qgemv,qgemm,qflux,moe_q,attn_q,quantized_embedding,quantized_embedding_bag \
  --formats fp8_e4m3 --warmup 15 --iters 60 \
  --out-dir perf/results/2026-07-14/fp8-experiments-baseline-core
PYTHONPATH=bindings/python .venv/bin/python perf/bench_kernels.py \
  --backend mlx --preset quick \
  --kernel paged_attn,kv_gather_fp8,quant_rt,act_quant,mla \
  --warmup 15 --iters 60 \
  --out-dir perf/results/2026-07-14/fp8-experiments-baseline-serving
PYTHONPATH=bindings/python .venv/bin/python perf/bench_kernels.py \
  --backend mlx --preset quick \
  --kernel qgemm_fp8_scaled,qgemm_fp8_block2d,kv_scatter_fp8,mla,act_quant \
  --warmup 15 --iters 60 \
  --out-dir perf/results/2026-07-14/fp8-experiments-baseline-added
```

- The final retained configuration used:

```bash
PYTHONPATH=bindings/python .venv/bin/python perf/bench_kernels.py \
  --backend mlx --preset quick \
  --kernel paged_attn,quant_rt,act_quant,kv_scatter_fp8,moe_q \
  --formats fp8_e4m3 --warmup 15 --iters 60 \
  --out-dir perf/results/2026-07-14/fp8-experiments-final-quick
```

Retained controlled results are milliseconds; brackets contain p20/p80 and the
last number is CV.

| Factor / priority shape | Control | Candidate | Change | Decision |
|---|---:|---:|---:|---|
| Scale hoist, paged v1 E4M3 D64 B8 H32 ctx2048 | 0.52622 [0.50750/0.55167], .0542 | 0.51531 [0.49892/0.54383], .0511 | -2.1% | keep |
| Scale hoist, paged v2 E4M3 D64 | 0.35998 [0.35266/0.37442], .0430 | 0.34671 [0.33854/0.36535], .0461 | -3.7% | keep |
| Scale hoist, paged v1 E4M3 D128 | 0.68397 [0.66479/0.70394], .1791 | 0.65818 [0.64594/0.67827], .0348 | -3.8% | keep |
| Scale hoist, paged v2 E4M3 D128 | 0.56668 [0.55123/0.58858], .3186 | 0.51488 [0.50175/0.53604], .0442 | -9.1% | keep |
| Format specialization, v1 E4M3 D64 | 0.37997 [0.37399/0.38903], .0316 | 0.35712 [0.35224/0.36686], .0374 | -6.0% | keep |
| Format specialization, v2 E4M3 D64 | 0.34496 [0.33313/0.35778], .0392 | 0.32421 [0.31828/0.33634], .0391 | -6.0% | keep |
| Format specialization, v1 E5M2 D64 | 0.37951 [0.37382/0.39588], .0398 | 0.31433 [0.31032/0.33115], .0377 | -17.2% | keep |
| Format specialization, v2 E5M2 D64 | 0.33839 [0.33181/0.34820], .0321 | 0.29531 [0.29082/0.30255], .0461 | -12.7% | keep |
| Format specialization, v1 E4M3 D128 | 0.64513 [0.63598/0.66721], .0334 | 0.63585 [0.63075/0.65210], .0257 | -1.4% | keep |
| Format specialization, v2 E4M3 D128 | 0.48786 [0.47897/0.50115], .0293 | 0.45708 [0.44774/0.47129], .0691 | -6.3% | keep |
| Format specialization, v1 E5M2 D128 | 0.65035 [0.63708/0.67058], .0364 | 0.58823 [0.57579/0.60200], .0334 | -9.6% | keep |
| Format specialization, v2 E5M2 D128 | 0.48924 [0.47845/0.50118], .0304 | 0.38811 [0.38146/0.40266], .0422 | -20.7% | keep |
| E4M3 bits, per-tensor N4096 D1024 | 0.14169 [0.12653/0.20402], .2624 | 0.09464 [0.09209/0.10183], .0947 | -33.2% | keep |
| E4M3 bits, per-token N4096 D1024 | 0.10629 [0.08931/0.12525], .1965 | 0.03446 [0.03298/0.03613], .2270 | -67.6% | keep |
| E4M3 bits, per-tensor N16384 D1024 | 0.37754 [0.36697/0.39182], .0937 | 0.37030 [0.35797/0.39467], .0866 | -1.9% | keep; no large-shape regression |
| E4M3 bits, per-token N16384 D1024 | 0.21197 [0.20720/0.22365], .1095 | 0.12752 [0.12422/0.13352], .1911 | -39.8% | keep |
| E4M3 bits, fused act quant T512 D2880 | 0.03940 [0.03863/0.04106], .0860 | 0.03042 [0.02968/0.03188], .1044 | -22.8% | keep |
| E4M3 bits, fused act quant T4096 D2880 | 0.28415 [0.27845/0.29530], .0638 | 0.23904 [0.23318/0.24818], .0726 | -15.9% | keep |
| E4M3 bits, KV scatter T512 H8 D64 | 0.03138 [0.02976/0.03520], .1292 | 0.02195 [0.02117/0.02503], .1302 | -30.1% | keep |
| E4M3 bits, KV scatter T512 H8 D128 | 0.03684 [0.03573/0.03945], .0905 | 0.03258 [0.03063/0.03572], .1480 | -11.6% | keep |
| E5M2 bits, KV scatter T512 H8 D64 | 0.02464 [0.02409/0.02562], .0651 | 0.02364 [0.02343/0.02404], .0879 | -4.1% | keep |
| E5M2 bits, KV scatter T512 H8 D128 | 0.03656 [0.03545/0.03703], .0814 | 0.02971 [0.02859/0.03444], .6165 | -18.7% | keep; final repeat 0.02952 [.02932/.02996], .0764 |
| FP8 SwiGLU MoE H/I2880 rows32 | 0.26140 [0.24922/0.31133], .1360 | 0.21183 [0.18680/0.24543], .1489 | -19.0% | keep two warps |
| FP8 SwiGLU MoE H/I2880 rows512 | 2.91865 [2.86096/2.99012], .0215 | 1.38616 [1.37210/1.43369], .0236 | -52.5% | keep two warps |

The final retained run independently measured E4M3/E5M2 paged-attention v1/v2
at both D64 and D128, producer shapes N4096/N16384 and T512/T4096, both KV
scatter head sizes, and FP8 rectangular/SwiGLU MoE at 32/512 rows. Its stable
large-shape results include E5M2 paged v2 D128 at 0.38491 ms
[0.38100/0.38761], CV .0173, fused FP8 activation quant T4096 at 0.22987 ms
[0.22765/0.23139], CV .0656, and FP8 SwiGLU MoE rows512 at 1.36326 ms
[1.36023/1.37177], CV .0089.

Rejected experiments:

| Factor | Representative result | Decision |
|---|---|---|
| MLA grouped scale loads | FP8 decode 0.67368 -> 0.66291 ms, then 0.67590 ms on repeat | reject as flat/noisy |
| MLA lane-0 scale broadcast | 0.66291 -> 0.90632 ms (+36.7%) | reject |
| Two-warp shared weights in scaled FP8 GEMM | M128 0.42744 -> 0.50924 ms; M512 1.48512 -> 1.72944 | reject |
| Two-warp shared weights in block2d FP8 GEMM | M128 0.37019 -> 0.47786 ms; M512 1.35321 -> 1.67152 | reject |
| Grouped outer scale loop in block2d GEMM | M32/M128/M512 0.11208/0.37019/1.35321 -> 0.12756/0.38513/1.41206 ms | reject |
| Standard FP8 QGEMM 2x32 shared-weight staging | M128/M512 0.36750/1.35244 -> 0.36452/1.34891 ms with overlapping bands | reject as flat/extra complexity |
| FP8 MoE decoder-scale shuffle after two-warp route | rows32 flat at 0.21165 ms; rows512 1.38616 -> 1.51084 ms | reject; keep two-warp topology only |
| Quantized attention eight warps | multi-warp 0.50458 -> 0.50324 ms | reject as flat |
| Quantized attention D128 Q16 row tile | single/multi 0.50613/0.50458 -> 0.60115/0.59942 ms | reject |
| FP8 QGEMV whole-block decode | N4096/N11008 0.03041/0.10581 -> 0.03188/0.10952 ms | reject |
| First E4M3 bit-encoder rounding constant | 23 focused correctness failures | reject immediately; corrected half-ULP constant before timing |

Correctness and validation:

```bash
PYTHONPATH=bindings/python .venv/bin/python -m pytest \
  tests/correctness/quantization/quant_rt/test_quant_rt.py \
  tests/correctness/quantization/act_quant/test_act_quant.py \
  tests/correctness/attention/mla/test_mla.py \
  tests/correctness/serving/kv_cache/test_kv_cache.py \
  tests/correctness/attention/paged_attn_v2/test_paged_attn_v2.py \
  tests/correctness/moe/moe/test_moe_q.py \
  tests/correctness/quantization/qgemm/test_fp8_scaled.py \
  tests/correctness/quantization/qgemm/test_fp8_block2d.py -q
```

- Result: 389 passed. The kernel build also passed after restoring every
  rejected candidate.
- FP8 code/scale reference cases are exact where the contract requires exact
  codes; there were zero code mismatches. Quantized paged-attention cases pass
  their path-specific bounds (up to `atol=3e-2`, `rtol=3e-3`), and MoE passes
  `atol=rtol=6e-2`.
- This was a focused affected-path validation, not a repository-wide
  correctness/parity/MPS run.

Decision: keep the scale algebra and compile-time format specialization in all
FP8 paged/cascade attention variants, the E4M3 and E5M2 normal-value bit
encoders, the E4M3 two-warp SwiGLU MoE route, and the new benchmark coverage.
Restore the MLA, GEMM, QGEMV, and quantized-attention topology candidates.

Raw results: initial baselines are `fp8-experiments-baseline-core`,
`fp8-experiments-baseline-serving`, and `fp8-experiments-baseline-added`.
Retained controlled runs are `fp8-paged-scale-hoist-{baseline,candidate}-d64-d128`,
`fp8-paged-format-specialization-{baseline,candidate}`,
`fp8-bit-encoder-{baseline,candidate}-repeat`,
`fp8-e5m2-encoder-{baseline,candidate}`, and
`fp8-moe-swiglu-two-warp-candidate`. The final retained run is
`fp8-experiments-final-quick`; rejected variants are preserved in the other
`fp8-*candidate` and MLA experiment directories under
`perf/results/2026-07-14/`.

## 2026-07-14: Cross-kernel FP8 transfer experiments

Status: complete; the retained implementation is measured and the focused,
repository-wide, parity, and PyTorch MPS validations pass.

This pass tested whether the successful FP8 encoder, format-specialization,
and two-warp ideas transfer to neighboring kernels. The starting hypothesis
was deliberately broad, but each experiment changed one factor at a time and
all non-wins were restored.

Retained changes:

- Grouped fused activation quantization now has compile-time SwiGLU and
  SwiGLU-OAI symbols. The per-token FP8/INT8 kernels retain their original
  runtime-mode implementation; only the grouped path, which evaluates the
  branch repeatedly for every 64-value group, is specialized.
- `orderable_uint_to_float(0)` treats the raw integer-zero atomic initializer
  as `+0.0`. No finite float maps to orderable integer zero. This fixes NaN
  scales for all-zero per-tensor FP8/INT8 and FP8 fake-quant inputs without
  changing nonzero quantization.
- Benchmark coverage now includes D64/D128 E4M3/E5M2 KV gather/scatter,
  indexer plain/UE8M0 modes, timed MLA FP8 insertion, FP8 fake quantization,
  grouped activation/quantization modes, FP8 UE8M0 norm quantization, and
  capped/non-capped attention.

Environment and method:

- MacBook Pro Mac16,5; Apple M4 Max, 40-core GPU, 128 GB; macOS 26.5.2
  (25F84); Xcode 26.6 (17F113); Apple Metal 32023.883 / toolchain
  17.6.109.0; Python 3.12.9; MLX 0.21.1; PyTorch 2.12.1.
- Working-tree label `376e5e4-dirty`; MLX Python-extension integration path;
  BF16 inputs and FP8 E4M3/E5M2 or UE8M0-scaled formats.
- Initial coverage used 15 warmups and 60 measured samples. Decisions used
  30-50 warmups and 120-240 samples. The harness adaptively batches calls,
  synchronizes samples, and records median, p20/p80, and CV.

Initial baseline command:

```bash
PYTHONPATH=bindings/python .venv/bin/python perf/bench_kernels.py \
  --backend mlx --preset quick \
  --kernel kv_gather_fp8,kv_scatter_fp8,indexer_quant,mla,fake_quant_fp8,act_quant,quant_rt,norm_quant_block,attn,decode_swiglu \
  --formats mxfp8 --warmup 15 --iters 60 \
  --out-dir perf/results/2026-07-14/cross-kernel-specialization-baseline
```

The retained activation result used a same-metallib A/B: both runtime and
specialized group symbols were compiled together and a temporary process-only
selector routed the control. The selector and runtime duplicate were removed
after measurement. Times are milliseconds; brackets are p20/p80 followed by
CV.

| Grouped activation path / shape | Runtime control | Specialized | Change | Decision |
|---|---:|---:|---:|---|
| SwiGLU UE8M0 T512 D2880 G64 | 0.06755 [0.06316/0.08244], .2032 | 0.06100 [0.05967/0.06875], .1096 | -9.7% | keep |
| SwiGLU-OAI UE8M0 T512 D2880 G64 | 0.06696 [0.06457/0.06874], .0456 | 0.06141 [0.06065/0.06304], .0536 | -8.3% | keep |
| SwiGLU UE8M0 T4096 D2880 G64 | 0.40894 [0.40315/0.41974], .0261 | 0.39573 [0.38778/0.41202], .0396 | -3.2% | keep |
| SwiGLU-OAI UE8M0 T4096 D2880 G64 | 0.46063 [0.44983/0.47335], .0425 | 0.42711 [0.41108/0.44406], .0416 | -7.3% | keep |

The atomic-zero fix is a correctness change, not a speedup claim. With the
fix present, the nonzero corrected baseline measured FP8 fake quantization at
0.04306/0.23776 ms for T512/T4096 D2880 and per-tensor FP8 quantization at
0.07856/0.37391 ms for N4096/N16384 D1024. Random FP8 fake-quant output is
bit-exact after output-dtype rounding against Torch E4M3FN, and zero FP8/INT8
per-tensor inputs now return zero codes/data and a zero scale.

Rejected experiments:

| Factor | Representative result | Decision |
|---|---|---|
| KV FP8 compile-time format symbols | Gather control E4/E5 D64 0.1355/0.0960 and D128 0.1407/0.1023 ms vs candidate 0.1603/0.0861 and 0.1268/0.1141; specialized scatter regressed substantially | reject mixed result and symbol growth |
| KV scatter one-head-per-warp | E4/E5 D64 0.0236/0.0215 -> 0.0242/0.0235 ms; D128 0.0317/0.0295 -> 0.0315/0.0326 | reject |
| Bit-built power-of-two producer scales | Quant group 0.0402/0.1499 -> 0.0501/0.1668 ms; activation group mixed; norm/indexer/MLA flat | reject; scale selection is only once per 64-128 values |
| MLA E8M0 bit decode per value | FP8 decode 0.6735 -> 0.6805 and 0.6782 ms on repeat | reject |
| FP8 fake-quant bit exponent/step | Corrected control 0.0431/0.2378 ms; candidate 0.0371/0.2348, then 0.0331/0.2778 | reject; realistic large shape is unstable/regressed |
| Per-token activation mode symbols | Small OAI cases improved, but the large cases were flat and templating perturbed standard-path codegen | restrict specialization to grouped path |
| Attention softcap symbols | Repeat D64 fwd/causal 0.2404/0.1324 -> 0.2428/0.1351; D128 N1024 0.4438/0.2323 -> 0.4556/0.2446; N2048 1.7503/0.9449 -> 1.7795/0.9855 | reject; uniform tile branch is already cheap |
| MXFP8 decode SwiGLU two warps | 0.0320 -> 0.0339 ms at B1 K1536 N4096 | reject; output-channel parallelism already saturates the GPU |

Indexer-consumer fusion was assessed but not implemented. The library exposes
indexer cache quantization/gather and sparse MLA consumption, but it has no
public query projection, scoring definition, or top-k selection contract that
connects them. Adding a fused scorer would invent model/application semantics,
contrary to the pure-kernel scope; it remains deferred until that reusable
contract exists.

Focused correctness completed during the experiment: 294 KV format cases,
154 KV geometry cases, 19 MLA cases, 50 FP8 fake/per-tensor quantization cases,
43 dense/causal attention cases, 55 decode-linear cases, and 8 activation
quantization cases passed for their respective candidates. After restoring all
rejected variants and removing the temporary A/B selector:

- `scripts/build kernels` passed.
- The final affected command passed 58 activation/fake/per-tensor quantization
  cases.
- `scripts/test correctness -q`: 2164 passed.
- `scripts/test parity -q`: 432 passed.
- `scripts/build pytorch_mps` passed.
- `scripts/test mps -q`: 472 passed.
- `scripts/test xcode`: test build succeeded.

The production-only final run (`cross-kernel-transfer-final-retained`, 50
warmups / 240 samples) independently reproduced the retained grouped medians:
0.0587/0.3984 ms for standard T512/T4096 and 0.0617/0.4216 ms for OAI.

Decision: keep grouped activation-mode specialization, the atomic-zero
correctness fix, its FP8 oracle/zero tests, and the expanded benchmark matrix.
Restore all other source variants. Raw controls/candidates are under
`perf/results/2026-07-14/`: `cross-kernel-specialization-baseline-repeat`,
`act-quant-group-{runtime-ab-control,specialized-ab-candidate}`,
`atomic-zero-sentinel-corrected-baseline`, and the `kv-fp8-*`, `pow2-*`,
`mla-e8m0-*`, `fake-quant-fp8-*`, `attention-softcap-*`, and
`decode-swiglu-mxfp8-two-warp-candidate` directories. The production-only
confirmation is `cross-kernel-transfer-final-retained`.
