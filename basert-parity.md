# BaseRT Kernel Parity Plan

Status: inferred operation-level kernel parity implemented and validated; no inference engine in scope
Date: 2026-07-24
QuixiCore backend: Metal
Reference checkout: `.reference/baseRT` at commit `1e9230269db7129dddef604c8d88f468fa55bc40` (Apache-2.0 open tree; proprietary engine excluded)
Scheduling reference: `~/llama.cpp` at commit `2beefef68825aed8de05f0d89981bf5d05266a3c` (MIT)

## Implementation ledger

This document is now both the design record and the parity ledger. Statuses
below are intentionally scoped to observable operation semantics; they do not
claim complete BaseRT engine parity.

| Work item | Status | Implemented boundary / blocker |
| --- | --- | --- |
| BRT-000 provenance | implemented | The user confirmed operation-level parity through QuixiCore's existing MLX/PyTorch contract and authorized clean-room invention from public semantics, `~/llama.cpp`, and existing QuixiCore kernels. Proprietary engine code remains excluded. |
| BRT-010 manifest | implemented within observable boundary | `.quixicore/kernels.yaml` maps every inferred reusable operation to an implementation, measured composition/reuse decision, or explicit runtime exclusion. Exact private pipeline-name/ABI identity remains unknowable and is not claimed. |
| BRT-020 fixtures | implemented for kernel scope | Framework-independent NumPy/framework oracles cover BaseQN, Q8_0 KV, GDN, extended RoPE, LoRA, calibration/output transforms, BERT inputs/pooling, 2-D/3-D vision patch/position/pooling/RoPE, audio convolution/relative attention, clippable projection bounds, and cross-attention. Model-graph fixtures are excluded with the inference engine. |
| BRT-100 descriptor/metadata | implemented | `BaseQDescriptor`, strict dtype/layout/group validation, format metadata, and public binding contract are present. |
| BRT-110 bit-exact unpack | implemented | Q2/Q3/Q4/Q5/Q6/Q8 little-endian code extraction, BF16/F16/E8M0/E4M3 scale decode, and symmetric/asymmetric reconstruction are covered. |
| BRT-120 standalone dequant | implemented | MLX and PyTorch MPS expose F16/BF16/F32 outputs from the shared Metal kernel. |
| BRT-130 decode GEMV | implemented | Direct Metal decode-GEMV is measured and retained; scale/bias loads are amortized across eight values per lane. |
| BRT-140 GEMM | implemented | M=1 routes to GEMV. Measured M>1 execution uses standalone decode plus framework GEMM; the slower direct candidate remains internal for future experiments. |
| BRT-150 BaseQN consumers | implemented | Direct embedding, QKV, gate/up SwiGLU, grouped expert projection/SwiGLU, and greedy LM-head selection are implemented. Down/output, shared-expert, and LM-head selection reuse measured core-kernel compositions where dedicated candidates lost. |
| BRT-200 extended position encoding | implemented | Explicit one-dimensional positions, partial/full split-half or adjacent-pair RoPE, sectioned/interleaved three-axis M-RoPE, D through 512, scaled-table consumption, and fused Q/K norm variants are exposed through MLX and PyTorch MPS. |
| BRT-210 attention schedules | implemented / reuse | Existing causal/bidirectional/varlen/decode/paged attention plus new independent-length GQA cross-attention cover the inferred tensor schedules and D=64/128/256 cross-attention matrix. Model layer schedules remain runtime composition. |
| BRT-300 cache/batching | implemented kernel substrate | The QuixiCore-owned Q8_0 KV ABI, scatter, gather, functional block copy, and direct paged-attention read are implemented in MLX and PyTorch MPS. Existing varlen metadata, last-row, batched LM-head, prefix/cache, and block-table primitives cover the remaining kernel surface; ownership and scheduling are runtime concerns. |
| BRT-400 hybrid/MoE | implemented kernel substrate | GDN short convolution/history, activated QKV split and Q/K normalization, decay/beta, recurrence/state pool, gated RMSNorm, full-attention sigmoid gate, dense/BaseQN projections, and BaseQN expert-stack projection/SwiGLU are exposed as reusable operations. Parallel recurrent prefill remains an optional measured experiment, not a correctness gap. |
| BRT-500 embedding | implemented kernel substrate | Fused token/type gather, LayerNorm with bias, QKV, bidirectional attention, SwiGLU, and mask-aware mean/RMS/L2 pooling are exposed as reusable operations. No BERT engine or graph is included. |
| BRT-600 multimodal | implemented kernel substrate | General 2-D/3-D patch extraction/projection, interpolated and factorized position operations, dense/coordinate pooling, distinct Gemma axis-block and Qwen global-split two-axis vision RoPE plus text M-RoPE, scalar value clipping, causal depthwise convolution, blocked relative audio attention, and independent-length GQA cross-attention compose with existing matmul/norm/attention operations. Media I/O and tower/model orchestration are excluded. |
| BRT-700 adaptation/calibration | implemented | Measured LoRA application, deterministic per-channel calibration absmax/running merge, and final-logit softcap are exposed through MLX and PyTorch MPS. Adapter loading, calibration datasets, and sampling policy remain runtime concerns. |

BaseQN implementation references:

- `kernels/quantization/base_q/README.md`
- `kernels/quantization/base_q/base_q.metal`
- `kernels/common/base_q_descriptor.h`
- `bindings/python/tk/base_q.py`
- `tests/correctness/quantization/base_q`
- `tests/correctness/parity/test_base_q_parity.py`
- `bindings/pytorch_mps/tests/test_base_q_mps.py`
- benchmark families `base_q`, `base_q_fused`, and `base_q_moe` in `perf/bench_kernels.py`
- focused decision in `perf/optimization_status.md`

Q8_0 KV implementation references:

- `kernels/serving/kv_cache/kv_cache.metal`
- `kernels/serving/kv_cache/kv_cache.cpp`
- `tests/correctness/serving/kv_cache/test_kv_cache.py`
- `tests/correctness/parity/test_parity.py`
- `bindings/pytorch_mps/tests/test_mps.py`
- benchmark families `kv_q8_0` and `paged_attn_q8_0` in `perf/bench_kernels.py`
- focused decision in `perf/optimization_status.md`

Q8_0 KV final verification on the recorded Apple M5 Max environment: MLX
extension build, PyTorch-MPS package build, and Xcode build-for-testing passed;
the repository-wide gates passed 2,300 correctness, 453 cross-backend parity,
and 593 direct MPS tests.

Gated DeltaNet implementation references:

- `kernels/linear_attention/gdn/gdn.metal`
- `kernels/linear_attention/gdn/gdn.cpp`
- `include/metal/common/glu_eval.metal`
- `kernels/activations/glu/glu.metal`
- `tests/correctness/linear_attention/gdn/test_gdn.py`
- `tests/correctness/activations/glu/test_glu.py`
- `tests/correctness/parity/test_parity.py`
- `bindings/pytorch_mps/tests/test_mps.py`
- benchmark families `gdn`, `gdn_io`, and `glu` in `perf/bench_kernels.py`
- focused decision in `perf/optimization_status.md`

Adaptation, calibration, and output-transform references:

- `kernels/matmul/lora`
- `kernels/quantization/quant_rt/quant_rt.metal`
- `kernels/sampling/sampling/sampling_transforms.metal`
- `tests/correctness/matmul/lora`
- `tests/correctness/quantization/quant_rt`
- `tests/correctness/sampling/sampling/test_transforms.py`
- benchmark families `lora` and `basert_aux` in `perf/bench_kernels.py`
- focused routing decisions in `perf/optimization_status.md`

BERT input and pooling references:

- `kernels/serving/embedding`
- `kernels/serving/mean_pool_rms_l2`
- `tests/correctness/serving/embedding`
- `tests/correctness/serving/mean_pool_rms_l2`
- benchmark family `basert_embedding` in `perf/bench_kernels.py`

Vision and audio references:

- `kernels/vision/patch_ops`
- `kernels/attention/rotary`
- `kernels/audio/conv1d`
- `kernels/audio/relative_attention`
- `kernels/attention/cross_attn`
- `kernels/sampling/sampling`
- `tests/correctness/vision/patch_ops`
- `tests/correctness/attention/rotary`
- `tests/correctness/audio/conv1d`
- `tests/correctness/audio/relative_attention`
- `tests/correctness/attention/cross_attn`
- benchmark families `basert_vision` and `basert_audio` in
  `perf/bench_kernels.py`
- focused routing decisions in `perf/optimization_status.md`
- public equation checks against Hugging Face's
  [Gemma 4 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)
  and
  [Qwen3-VL implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)

Final repository verification on the recorded Apple M5 Max environment:

- MLX extension and PyTorch MPS package builds passed.
- `scripts/test correctness -q`: 2,434 passed.
- `scripts/test parity -q`: 464 passed.
- `scripts/test mps -q`: 604 passed.
- `scripts/test python -q`: 44 passed.
- `scripts/test xcode`: build-for-testing passed.
- `.quixicore/kernels.yaml` parses successfully and `git diff --check` passes.

This closes the inferred reusable operation inventory. The remaining concepts
visible in BaseRT—model graphs, tower orchestration, shared-prefix ownership,
continuous batching, cache eviction, tokenization, media decoding, timestamp
policy, loading, and serving—are inference/runtime responsibilities and are
not kernel gaps in this repository.

## Executive summary

The goal is to bring the compute capabilities observable in the local BaseRT
reference into QuixiCore Metal without weakening QuixiCore's public contracts,
correctness requirements, licensing rules, or performance process.

The local BaseRT checkout does not contain the engine or its Metal source. It
contains the open model converter, .base format, public C ABI, bindings,
documentation, and benchmarks. The engine and compiled Metal kernels are
distributed separately as a proprietary binary. No reachable local branch,
tag, or historical commit contains .metal, Objective-C++, C++, or other GPU
engine source.

Consequently, an exact source-level inventory of every private BaseRT pipeline
is not possible from this checkout. Per the confirmed direction, QuixiCore uses
the public format/model semantics as an inferred operation manifest and invents
the missing Metal implementations clean-room, using `~/llama.cpp` and existing
QuixiCore kernels only as licensed low-level scheduling references.

This plan is therefore split into:

- A provenance record and inferred operation inventory.
- The complete operation backlog that can be inferred from the open reference.
- A proposed QuixiCore-native design for those operations.
- Integration, correctness, parity, and performance requirements.

The confirmed scope is operation-level kernel parity through QuixiCore's
existing tk and tk_torch MLX/PyTorch contracts. Reproducing BaseRT's complete
engine, .base loader, tokenizer, scheduler, server, grammar implementation, or
C API is out of scope. Those components may consume QuixiCore operations in a
separate runtime project, but they must not shape or duplicate the public
kernel contract here.

## Confirmed design decision

QuixiCore will implement the reusable tensor operations needed for BaseRT
compute parity. It will not reproduce BaseRT model or engine APIs.

This means:

- New public functions must describe framework-neutral tensor semantics.
- MLX and PyTorch MPS must expose the same operation contract.
- Model-specific execution graphs remain compositions of QuixiCore operations.
- BaseRT architecture names must not be used as dispatch substitutes for
  explicit mathematical parameters.
- A future .base loader or serving runtime is a consumer of these operations,
  not part of this parity effort.
- Project completion is measured against the inferred reusable operation
  inventory, not against BaseRT CLI, server, tokenizer, or C-ABI behavior.

## Source and evidence boundaries

### What the reference contains

The local reference provides:

- The public product and architecture overview in
  [.reference/baseRT/README.md](.reference/baseRT/README.md).
- The proprietary-engine boundary and release contents in
  [.reference/baseRT/docs/reference/engine-releases.md](.reference/baseRT/docs/reference/engine-releases.md).
- The public runtime ABI in
  [.reference/baseRT/include/baseRT/baseRT.h](.reference/baseRT/include/baseRT/baseRT.h).
- Model and sampling configuration structures in
  [.reference/baseRT/include/baseRT/types.h](.reference/baseRT/include/baseRT/types.h).
- Architecture mapping and configuration extraction in
  [.reference/baseRT/base-convert/crates/base-arch/src](.reference/baseRT/base-convert/crates/base-arch/src).
- Canonical quantization packers in
  [.reference/baseRT/base-convert/crates/base-quant/src](.reference/baseRT/base-convert/crates/base-quant/src).
- The .base container and layout specification in
  [.reference/baseRT/base-convert/FORMAT.md](.reference/baseRT/base-convert/FORMAT.md).
- The canonical quantization specification in
  [.reference/baseRT/base-convert/CANONICAL_QUANT_SPEC.md](.reference/baseRT/base-convert/CANONICAL_QUANT_SPEC.md).
- Model catalog examples in
  [.reference/baseRT/base-convert/crates/base-hub/catalog.json](.reference/baseRT/base-convert/crates/base-hub/catalog.json).
- Public throughput methodology and historical results in
  [.reference/baseRT/benchmarks](.reference/baseRT/benchmarks).

### What the reference does not contain

It does not provide:

- Metal kernel source.
- Engine C++ or Objective-C++ source.
- An exported kernel manifest.
- Per-kernel buffer indices or layouts.
- Pipeline specialization constants.
- Threadgroup geometry.
- Routing thresholds.
- Supported shape tables.
- Numerical tolerances.
- Per-kernel performance baselines.

Comments in the open converter mention names such as gemv_base_qN,
simd_gemm_qN, and argmax_f16_batched. These are evidence that operation
families exist, but they are not a complete ABI and must not be treated as
source suitable for direct porting.

### Licensing and provenance gate

The open ecosystem repository is Apache-2.0. The engine is described as
proprietary. These are separate provenance domains.

The local `~/llama.cpp` checkout is MIT-licensed at commit
`2beefef68825aed8de05f0d89981bf5d05266a3c`. It was consulted only for public
mathematical/layout semantics and low-level scheduling ideas. The resulting
Metal/C++ code is clean-room QuixiCore code; no source was copied from the
BaseRT engine or imported from llama.cpp.

Before implementation:

- Record the license for every source or specification used.
- Obtain written permission for any engine-derived material.
- Do not extract or reconstruct source from baseRT.metallib or libbaseRT.dylib.
- Do not copy proprietary implementation details into QuixiCore.
- If Apache-2.0 converter code is reused rather than independently
  reimplemented, preserve the required license and NOTICE obligations.
- Prefer implementation from public mathematical specifications and
  independently produced tests.
- Record provenance in the operation README or a central porting ledger.

The repository's instruction not to import reference implementation code
without licensing and provenance review applies to every work package below.

## Scope

### Confirmed in scope

- Metal kernels implementing the observed operations.
- Reusable Metal helpers required by those kernels.
- MLX primitives and Python bindings.
- PyTorch MPS primitives and Python bindings.
- Public QuixiCore operation contracts where an operation is generally useful.
- Internal composition helpers when an operation is model-specific.
- Correctness, MPS, and cross-backend parity tests.
- Benchmark coverage and optimization notebook entries.
- Kernel and quant-format metadata.

### Confirmed out of scope

- Cloning the BaseRT C ABI.
- Loading .base model bundles directly into QuixiCore.
- Tokenizers and chat templates.
- GBNF or JSON-schema grammar parsing.
- HTTP or OpenAI-compatible serving.
- Model download, conversion, signing, and registry behavior.
- Prefix radix trees and cache eviction policy.
- Continuous-batching request scheduling.
- Host-side sampling policy beyond what is needed to expose a kernel.
- Image decoding, WAV decoding, resampling, or other media I/O.

Some out-of-scope components are required for end-to-end BaseRT engine parity.
They belong in a separate runtime layer that consumes QuixiCore operations and
must not be placed under kernels or added to tk/tk_torch as model APIs.

## Observable BaseRT compute surface

### Model families

The open architecture dispatch and public structures establish support or
planned dispatch for the following families:

| Family | Observable compute requirements | Reference |
| --- | --- | --- |
| Llama and Mistral | RMSNorm, RoPE, GQA, causal attention, SwiGLU, dense and quantized projections | base-arch/src/llama.rs |
| Qwen2 and Qwen3 | Llama-style decoder plus Q/K norm variants and model-specific dimensions | base-arch/src/qwen.rs |
| Qwen2/3 MoE | Router, top-k experts, expert GEMMs, optional shared expert | base-arch/src/qwen.rs |
| Qwen3.5 and Qwen3.6 | Hybrid Gated DeltaNet and periodic full attention, partial RoPE, output gate | base-arch/src/qwen.rs and include/baseRT/types.h |
| Qwen3.5/3.6 MoE | Hybrid attention plus routed and shared experts | base-arch/src/qwen.rs |
| Gemma, Gemma 2, Gemma 3 | Gemma RMSNorm convention, local/global attention, attention scaling and softcaps | base-arch/src/gemma.rs |
| Gemma 4 | Per-layer dimensions, local/global head layouts, shared KV layers, PLE, MoE, vision and audio towers | base-arch/src/gemma.rs and include/baseRT/types.h |
| Nomic BERT | Token/type embeddings, LayerNorm with bias, fused QKV, bidirectional attention, SwiGLU and embedding pooling | base-arch/src/bert.rs |
| Whisper | Audio encoder, autoregressive decoder, cross-attention, KV cache and timestamp-aware transcription | include/baseRT/baseRT.h and include/baseRT/types.h |

The .base format also contains generic SSM, SSM-MoE, hybrid, shared-attention,
LoRA, speculator, and compute-graph concepts. Those schema features are not
proof that every corresponding engine kernel is present in the referenced
release. They belong in the authoritative manifest request rather than being
claimed as confirmed kernel coverage.

### Quantization surface

The canonical .base formats are:

| Format | Bits | Canonical group | Packing | Scale/bias |
| --- | ---: | ---: | --- | --- |
| base_q2 | 2 | 32 | lane-strided / 16 lanes per 4 bytes | asymmetric |
| base_q3 | 3 | 32 | bit-spread / 8 lanes per 3 bytes | asymmetric |
| base_q4 | 4 | 64 | two nibbles per byte, low nibble first | asymmetric |
| base_q5 | 5 | 64 | bit-spread / 8 lanes per 5 bytes | asymmetric |
| base_q6 | 6 | 64 | bit-spread / 4 lanes per 3 bytes | asymmetric |
| base_q8 | 8 | 128 | unsigned byte codes | asymmetric |

The reconstruction is conceptually:

    weight = code * scale + bias

The scale storage type is independent of the weight width:

- BF16.
- F16.
- E8M0.
- E4M3 for Q8 where allowed by the canonical specification.

MXFP4, NVFP4, BF16, F16, F32, and GGUF passthrough tensors also appear in the
format. Existing QuixiCore format names must not be assumed byte-compatible;
fixture-level proof is required.

### Serving and generation surface

The public ABI establishes these GPU-relevant paths:

- Contiguous and paged KV caches.
- F16 and Q8_0 KV cache selection.
- Fixed-size KV pages, with page size varying by head dimension.
- Single-sequence prefill and decode.
- Batched decode.
- Packed variable-length mixed prefill and decode.
- Per-sequence block tables.
- Batched output projection.
- Batched argmax.
- Host-readable batched logits.
- GPU temperature and repetition transforms.
- Top-k, top-p, min-p, categorical, and greedy sampling.
- Grammar/token masks.
- Prefix-cache KV sharing.
- KV save/load and rollback.
- N-gram speculation and chained decode.
- LoRA deltas.
- AWQ activation calibration.
- Mean-pooled normalized embeddings.
- Image and audio feature prefill.

Not all of these imply unique kernels. The official manifest must identify
which are separate GPU pipelines, framework compositions, or host-only
operations.

## Current QuixiCore starting point

The current kernel taxonomy is recorded in
[.quixicore/kernels.yaml](.quixicore/kernels.yaml) and the quant surface in
[.quixicore/quant-formats.yaml](.quixicore/quant-formats.yaml).

### Strong reusable coverage

QuixiCore already provides reusable implementations for:

- RMSNorm, LayerNorm, add-norm, residual-next norm, and Q/K norm.
- GELU, GLU, SwiGLU, GeGLU, softmax, and Hadamard transforms.
- Dense GEMM, staged GEMM, decode linear epilogues, and fused decode SwiGLU.
- Causal, non-causal, decode, variable-length, paged, and long-context
  attention.
- Sliding-window attention and attention softcapping.
- Split-half and interleaved RoPE.
- Fused full-dimension Q/K RMSNorm plus RoPE.
- KV scatter, gather, copy, scale calculation, FP8 cache, and paged attention.
- Quantized GEMM/GEMV for GGUF, MX, FP8, FP4, BitNet, TQ2, and related formats.
- Quantized embedding lookup and reduction.
- Quantized LM-head sampling, sparse candidates, masks, and beam advance.
- MoE routing, permutation, scheduling, grouped GEMMs, quantized experts, and
  finalize.
- The core Gated DeltaNet recurrence in
  [kernels/linear_attention/gdn](kernels/linear_attention/gdn).
- Sampling transforms, penalties, speculative helpers, and beam helpers.
- Patch merge, space-to-depth/norm/projection, and a small fixed vision MLP.
- Mean-pool, RMSNorm, and L2 normalization.
- MLX and PyTorch MPS integration.

### Final inferred kernel coverage

| Area | Current status | Required action |
| --- | --- | --- |
| BaseQN weight formats | Implemented | Q2/Q3/Q4/Q5/Q6/Q8 decode, GEMV/GEMM, embedding, QKV, SwiGLU, greedy LM-head selection, and grouped expert projection/SwiGLU are present |
| BaseRT-style Q8_0 KV | Implemented as a distinct KV codec | QuixiCore separate code/scale planes cover exact encode/decode, functional block copy, and direct D64/D128 paged reads without reusing the packed weight ABI |
| Partial and multimodal RoPE | Implemented | `rotary_positioned`, `mrope`, and `qk_norm_rope_positioned` cover explicit positions, partial/full rotary, sectioned/interleaved THW axes, and fused Q/K normalization |
| Llama-3 and linear RoPE scaling | Implemented at the kernel boundary | RoPE kernels consume validated precomputed tables; tests cover uniform linear and Llama-3 piecewise scaled tables without embedding scaling policy in the hot kernel |
| Head dimension 512 | Covered for extended RoPE | Positioned/partial/M-RoPE and fused Q/K norm instantiate D=512; unrelated attention operations add D=512 only when their tensor contract requires it |
| Qwen3.5/3.6 hybrid kernel substrate | Implemented | Dense/BaseQN projections compose with short convolution/history, QKV preparation, decay/beta, `gdn_recur`, gated RMSNorm, output projection, and the full-attention sigmoid gate; layer scheduling remains outside kernel scope |
| Gemma 4 PLE/shared-KV topology | Covered by reusable primitives | Per-layer dimensions are tensor shapes; shared-KV/PLE ownership and layer reuse are model topology, not new arithmetic kernels |
| BaseQN MoE | Implemented kernel substrate | Direct grouped expert GEMM and fused expert SwiGLU use the existing alignment/routing/finalize operations; shared experts compose the measured BaseQN core |
| SigLIP/Gemma vision tower | Implemented kernel substrate | Patch projection, factorized learned positions, coordinate-aware pooling, two-axis vision RoPE, clippable projections, vision blocks, and feature-splice tensor operations are covered; tower orchestration is excluded |
| Qwen VL tower | Implemented kernel substrate | General temporal/spatial 3-D patch extraction/projection, learned-position interpolation, Qwen global-split two-axis vision RoPE, positioned text M-RoPE, attention, spatial merge, and tensor splice operations are covered |
| Gemma Conformer audio | Implemented kernel substrate | Patch/subsample composition, clippable projections/value clamps, causal depthwise convolution, blocked relative-position attention, normalization, GLU, and projections are covered |
| Whisper | Implemented kernel substrate | General audio convolution, self-attention/cache operations, independent-length cross-attention, output projection, and masks are covered; transcription/timestamp orchestration is excluded |
| Nomic BERT | Implemented kernel substrate | Token/type preparation, LayerNorm bias, QKV, bidirectional attention, SwiGLU, and mask-aware normalized pooling are covered |
| LoRA application | Implemented | Fused direct small-batch application plus measured framework fallback, optional base add, explicit scale, and direct ranks through 256 |
| AWQ activation calibration | Implemented | Deterministic per-input-channel FP32 absmax with chunkable running merge and defined NaN/infinity semantics |
| Inference logit softcap | Implemented | Reusable final-logit `cap*tanh(x/cap)` transform composes with existing penalties, masks, and sampling kernels |

## Target architecture

### Contract boundary

QuixiCore should expose operations, not BaseRT model classes.

Public operations should be added only when they describe reusable tensor
semantics. Model-specific orchestration should remain an internal composition
or a narrowly named composed operation. This keeps backend-local optimization
behind the shared QuixiCore contract.

The target layers are:

    Model/runtime orchestration
        -> public tk or tk_torch operation
        -> MLX/PyTorch primitive
        -> shared launch contract
        -> Metal specialization

BaseRT-compatible formats should be data contracts, not a dependency on the
BaseRT runtime.

### Implemented source layout

Prefer extending an existing operation when the semantics are the same.
Create a new directory only for a genuinely new operation.

The clean-room additions landed at reusable operation boundaries:

    kernels/quantization/base_q/
    kernels/quantization/quant_rt/  # extended with calibration_absmax
    kernels/attention/rotary/  # extended in place for positioned/partial/M-RoPE
    kernels/attention/cross_attn/
    kernels/linear_attention/gdn/  # extended in place for preparation/output
    kernels/matmul/lora/
    kernels/vision/patch_ops/
    kernels/audio/conv1d/

Measured reuse and extensions to existing directories:

- kernels/quantization/qgemm for BaseQN matrix execution.
- kernels/quantization/qgemv for BaseQN vector execution.
- kernels/quantization/dequant_gather for BaseQN embeddings.
- kernels/quantization/lm_head for BaseQN output projection.
- kernels/matmul/decode_linear for BaseQN fused epilogues.
- kernels/moe/moe for BaseQN and shared-expert paths.
- kernels/norms/qk_norm_rope for rotary_dim and position-mode extensions.
- kernels/serving/kv_cache for the QuixiCore-owned Q8_0 encode/gather/copy and
  direct paged-attention read path.
- kernels/serving/embedding for token/type gather/add.
- kernels/serving/mean_pool_rms_l2 for mask-aware batched pooling.
- kernels/sampling/sampling for final-logit softcap before existing transforms.

Adding kernels/audio requires updating the canonical family taxonomy in
.quixicore metadata and repository-structure documentation.

### Packed weight contract

BaseQN must not be represented by pretending it is a GGUF block. Codes,
scales, biases, group size, scale dtype, and layout are independent parts of
the contract.

The preferred binding-level shape is:

    base_qgemv(
        codes,
        scales,
        biases,
        x,
        bits,
        group_size,
        scale_dtype,
        layout)

and equivalently for GEMM and embedding operations.

The C++ primitive should normalize these arguments into an internal descriptor:

    BaseQDescriptor
      bits
      group_size
      scale_dtype
      layout
      symmetric
      logical_rows
      logical_columns

Reasons to keep separate buffers:

- It matches the .base format.
- It avoids hidden copies or concatenation.
- It makes scale-dtype validation explicit.
- It supports zero-copy views from a future .base loader.
- It keeps tensor offsets and alignment independent.

A convenience Python container can be considered later, but kernel primitives
must remain expressible as ordinary arrays plus scalar metadata for both MLX
and PyTorch.

### BaseQN Metal design

Build one shared decode substrate with compile-time traits:

    BaseQTraits<bits, scale_type, layout>
      lanes_per_chunk
      bytes_per_chunk
      canonical_group_size
      load_code
      load_scale
      load_bias
      dequantize

Specialization axes:

- Weight width: 2, 3, 4, 5, 6, 8.
- Scale storage: BF16, F16, E8M0, and permitted E4M3.
- Activation dtype: F16, BF16, and F32 only where justified.
- Output dtype.
- GEMV versus GEMM geometry.
- Row-major/lane-strided versus repacked tile layout.

Do not instantiate every Cartesian product blindly. Start with combinations
present in reference profiles, then add others only when metadata and tests
require them. Measure code size, pipeline compile time, and dispatch overhead.

The implementation order should be:

1. Bit-exact CPU/test unpack helpers.
2. Standalone Metal dequantization for diagnostics.
3. Decode GEMV.
4. Prefill GEMM.
5. Small-batch GEMM.
6. Embedding gather.
7. Fused QKV and gate/up paths.
8. LM head.
9. MoE experts.

### Dispatch and routing

Routing belongs in the host primitive and must be shape- and format-aware.

Each operation should define:

- Direct Metal eligibility.
- Framework fallback eligibility.
- Repack requirements.
- Decode versus prefill crossover.
- Small-batch crossover.
- Unsupported edge shapes.
- Required alignment.

Routing thresholds are performance claims. They may be committed only after a
focused run on the affected kernel and shape set.

Kernel names should encode only stable specialization axes, for example:

    base_qgemv_q4_sbf16_bf16
    base_qgemm_q6_sbf16_f16_m32

The final naming scheme should follow current QuixiCore launch conventions and
avoid model names when the operation is generic.

### RoPE and position design

Extend position handling through explicit semantics rather than model flags.

Required modes:

- Split-half.
- Adjacent-pair/interleaved.
- Full rotary.
- Partial rotary through rotary_dim.
- One-dimensional positions.
- Three-axis temporal/height/width positions.
- Sectioned and interleaved M-RoPE.

The proposed fused QK-norm API should accept:

- QKV or separate Q/K/V arrays.
- Q/K norm weights.
- Position IDs.
- Cosine/sine tables or frequency data.
- Rotary dimension.
- Pairing mode.
- M-RoPE sections and interleaving mode when applicable.
- Unit-offset norm behavior as explicit semantics.
- Optional KV output/cache destination.

Frequency-table generation, Llama-3 piecewise scaling, and uniform linear
scaling should remain outside the hot kernel unless measurement proves fusion
valuable. The kernel should consume validated tables so correctness can be
tested independently from table construction.

Do not add a boolean named gemma to new public APIs. Express the mathematical
choice, such as norm_weight_offset, so the operation remains reusable.

### Attention and KV design

Reuse current causal, variable-length, paged, and partition/reduce kernels.
Extend their contracts only where BaseRT requirements differ.

Required dimensions and modes must come from the official manifest. Public
config proves that dimensions can vary by layer and may reach 256 or 512, but
it does not prove that every existing attention kernel must support every
dimension.

The QuixiCore-owned Q8_0 KV tensor ABI is now fixed and intentionally distinct
from packed GGUF or BaseQN weight layouts:

- K and V each have an independent int8 code plane with shape
  `(num_blocks, block_size, num_kv_heads, head_dim)`.
- Each code plane has a parallel F16 scale plane with shape
  `(num_blocks, block_size, num_kv_heads, head_dim / 32)`.
- One scale covers 32 consecutive dimensions of one token/head row. Head
  dimension must therefore be a multiple of 32; the direct paged read is
  specialized for D=64 and D=128.
- Encoding computes `delta = absmax / 127` in FP32, stores delta in F16, uses
  round-half-away-from-zero, clamps codes to `[-127, 127]`, and emits a zero
  scale and zero codes for an all-zero group. Decoding always uses the stored
  F16 scale, making the serialized planes the complete source of truth.
- Logical slots map to `(physical_block, token_offset)` through the existing
  block-major cache convention. K and V do not share codes or scales.
- Paged attention dequantizes while reading the cache, supports MHA/GQA/MQA
  and optional sliding windows, and accumulates scores, online softmax state,
  and values in FP32 before converting to the query dtype.
- The public tensor contract exposes separate planes rather than an unaligned
  packed 34-byte `block_q8_0` struct. One logical K or V value therefore uses
  `1 + 2/32 = 1.0625` persistent bytes, 53.125% of BF16 storage.

This ABI was designed clean-room from observable Q8_0 mathematical semantics.
The llama.cpp `block_q8_0`, reference encoder, and Metal quantization/attention
implementations were consulted for block-size, rounding, safe-FP, and
dequant-on-read ideas; no reference source or packed ABI was imported.

The packed mixed-batch contract should use:

- Packed token rows.
- Per-request cumulative sequence lengths.
- Per-request context lengths.
- Slot mappings.
- Block-table offsets.
- A last-row index for output projection.

This matches existing QuixiCore variable-length conventions and avoids a
BaseRT-specific scheduler contract inside kernels.

### Gated DeltaNet design

The existing `gdn_recur` operation remains the mathematical recurrence
boundary. The Qwen3.5/3.6 tensor substrate is implemented from independently
testable stages:

1. Quantized or dense input projections.
2. Tensor splitting and reshaping.
3. Short causal depthwise convolution with persistent state.
4. Gate and beta transforms.
5. Q/K normalization required by the reference architecture.
6. gdn_recur.
7. Gated normalization/output transform.
8. Output projection.

The retained fused stages are `gdn_short_conv(..., apply_silu=True)`,
`gdn_qkv_prepare`, `gdn_gate_beta`, `gdn_gated_rmsnorm`, and the GLU
`sigmoid` mode exposed as `sigmoid_mul`. Measurements against their unfused
framework compositions are recorded in `perf/optimization_status.md`.

State must be explicit:

- DeltaNet recurrent state pool.
- Short-convolution history pool.
- Per-request slot mapping.
- Fresh-prefill versus continuation flags.
- Functional output semantics for MLX.
- Safe clone/update semantics for PyTorch MPS.

A parallel chunked prefill algorithm is a separate optimization milestone. It
must match sequential recurrence within declared tolerance and show a win on
realistic prompt lengths without degrading decode.

### MoE design

Reuse the current routing, permutation, schedule, grouped GEMM, and finalize
pipeline.

Add:

- BaseQN expert-stack descriptors.
- BaseQN grouped GEMM.
- Fused expert gate+up where the source tensor is fused.
- Optional router score renormalization.
- Optional shared expert.
- Shared-expert scalar gate.
- Per-layer expert width and count.
- Per-layer router scaling or correction.

Keep selection semantics separate from expert execution. Qwen and Gemma differ
in top-k normalization, and this must be an explicit router argument rather
than inferred from a model name.

### Gemma 4 design

Gemma 4 should be represented as reusable primitives plus an integration graph,
not one monolithic kernel.

Required pieces include:

- Per-layer hidden/FFN dimensions.
- Per-layer head dimension and KV-head count.
- Local versus global attention schedule.
- Separate local/global RoPE parameters.
- Partial global RoPE.
- Explicit attention scale.
- Shared KV-layer mapping.
- Post-attention and post-FFN norms.
- Final-logit softcap.
- Per-layer embedding input and projection where present.
- Dense, routed-expert, and shared-expert FFNs.

The integration test must cover a layer schedule with at least one local layer,
one global layer, and differing dimensions.

### Vision design

#### Gemma/SigLIP path

Implement or compose:

- Patch projection.
- Learned/factorized position embedding.
- Position interpolation when input resolution differs.
- Vision norm variants.
- Q/K norm.
- Bidirectional multi-head attention.
- GeGLU or model-declared activation.
- Pre/post residual norms.
- Spatial pooling.
- Projection into the text hidden dimension.
- Feature splice into placeholder token positions.

#### Qwen VL path

Implement or compose:

- Temporal/spatial patch projection.
- Learned position embedding with interpolation.
- Fused QKV projection.
- Vision LayerNorm blocks.
- Two-dimensional or three-axis M-RoPE.
- Bidirectional vision attention.
- MLP activation declared by the architecture.
- Spatial merge, commonly 2 by 2.
- Projection into the text hidden dimension.
- Feature and position splice into the text stream.

Existing patch_merge and space_to_depth_norm_linear should be reused only when
their ordering, padding, norm, and projection semantics match exactly.

### Audio and Whisper design

Create a new audio kernel family rather than mixing audio operations into
vision or generic utilities.

#### Gemma Conformer

Candidate primitive boundaries:

- Feature normalization.
- Two-stage convolutional subsampling.
- Subsampling projection.
- First feed-forward macromodule.
- Relative-position/chunked self-attention.
- LightConv1d with pointwise gate, depthwise convolution, normalization, and
  activation.
- Second feed-forward macromodule.
- Output norm and projection.
- Audio-feature splice.

Media decoding, resampling, and log-mel construction should be explicitly
assigned to CPU/framework or Metal after measurement.

#### Whisper

Candidate primitive boundaries:

- Log-mel or an explicit precomputed-mel input contract.
- Encoder convolution stack.
- Encoder self-attention and MLP.
- Decoder causal self-attention.
- Decoder cross-attention.
- Encoder KV preparation.
- Decoder KV append/read.
- Output projection.
- Timestamp and token masking where GPU execution is justified.

Start with precomputed mel inputs so kernel work is not blocked by media I/O.

### Nomic BERT design

Compose existing primitives where possible:

- Token and token-type embedding gather/add.
- Input LayerNorm with weight and bias.
- Fused QKV projection.
- Bidirectional attention.
- Attention output and residual norm.
- SwiGLU MLP.
- Layer output and residual norm.
- Mean pooling.
- RMSNorm and L2 normalization.

Add BaseQN support to the projection and embedding operations. Preserve BERT's
LayerNorm bias semantics; do not route it through RMSNorm-only model helpers.

### LoRA design

Represent LoRA as a generic post-projection delta:

    intermediate = x times transpose(A)
    delta = intermediate times transpose(B)
    output = output + delta

Requirements:

- F16 A and B.
- Effective rank derived from tensor shape.
- Prefill and decode geometries.
- Fused-QKV and fused-gate/up adapter layouts.
- Quantized base projection plus F16 delta.
- Optional fusion of delta-add when measurement supports it.

The low-rank path belongs under matmul and should not depend on .base file
parsing.

### Calibration design

Add a per-input-channel absolute-maximum reduction:

- Input shape: tokens by channels.
- Output shape: channels.
- Accumulation: F32.
- Multi-chunk merge for long calibration sets.
- Deterministic handling of NaN and infinity.
- Optional running maximum update.

The kernel should emit tensor data only. Mapping results to canonical tensor
names and serializing an AWQ sidecar are runtime/converter responsibilities.

### Binding design

For every public operation:

- Define one framework-neutral semantic contract in bindings/python/tk.
- Add MLX C++ bindings in bindings/python/bindings.cpp.
- Add PyTorch MPS bindings under bindings/pytorch_mps.
- Validate rank, dtype, contiguity, dimensions, metadata, and device.
- Match error behavior across backends.
- Keep internal-only composition helpers out of the top-level API unless they
  have reusable semantics.

Large metadata structs should not be mirrored as opaque byte blobs in Python.
Use explicit scalar arguments or small typed descriptors at the C++ boundary.

## Work breakdown

### Phase 0: freeze the clean-room kernel target

#### BRT-000 — provenance decision

Deliverables:

- Approved license/provenance record.
- Written clean-room rules.
- Decision on whether BaseRT binary output may be used as a black-box oracle.
- Recorded confirmation that the target is operation-level parity through the
  existing MLX/PyTorch contract.

Acceptance:

- Repository maintainers approve the source and test-data policy.
- No implementation starts from proprietary source without authorization.

#### BRT-010 — inferred operation manifest

Maintain a QuixiCore-owned operation manifest containing:

- Operation and pipeline names.
- Buffer ABI.
- Shapes, layouts, alignments, and strides.
- Dtypes and quant formats.
- Apple GPU requirements.
- Dispatch and fallback rules.
- Correctness tolerances.
- Reference model/shape coverage.

Acceptance:

- Every operation inferred from public model/format behavior maps to one row.
- Every row maps to a QuixiCore operation, a planned clean-room work item, or a
  documented unsupported decision.
- Private BaseRT pipeline names and buffer ABIs are explicitly not claimed.

#### BRT-020 — golden fixture pack

Create redistributable fixtures for:

- BaseQN packed groups at each width and scale dtype.
- GEMV and GEMM outputs.
- KV encode/decode.
- RoPE modes.
- DeltaNet state transitions.
- MoE routing and expert aggregation.
- BERT token/type preparation and masked normalized pooling.
- Vision patch extraction, interpolation, and pooling.
- Temporal/spatial 3-D patch extraction and two-axis vision RoPE.
- Factorized learned positions and padded/arbitrary-order coordinate pooling.
- Audio general/causal-depthwise convolution, blocked relative attention, and
  independent-length cross-attention.
- Scalar-bounds value clipping around clippable projections and audio states.
- LoRA, calibration absmax, and final-logit softcap.

Acceptance:

- Fixture provenance is documented.
- Fixtures are small enough to commit.
- Each fixture has a framework-independent oracle description.
- Whole model/layer graph fixtures are not required in this kernel-only
  repository.

### Phase 1: BaseQN substrate

#### BRT-100 — metadata and descriptors

- Add BaseQN entries to .quixicore/quant-formats.yaml.
- Define BaseQDescriptor in the host layer.
- Define scale-dtype and layout enums.
- Add validation and descriptive error messages.

#### BRT-110 — bit-exact unpack

- Implement Q2/Q3/Q4/Q5/Q6/Q8 code extraction.
- Implement BF16/F16/E8M0/E4M3 scale decoding.
- Implement asymmetric bias.
- Test every lane boundary and cross-byte case.

#### BRT-120 — standalone dequant

- Add diagnostic dequant kernels.
- Cover row tails or reject unsupported tails explicitly.
- Verify F16/BF16/F32 outputs.

#### BRT-130 — decode GEMV

- Implement and tune priority decode shapes.
- Compare direct dequant-dot against dequantize-then-framework GEMV.
- Add scale-dtype siblings only when used by profiles.

#### BRT-140 — prefill and small-batch GEMM

- Implement matrix path.
- Decide direct versus repacked tile execution by measurement.
- Cover prompt and continuous-batch row counts.

#### BRT-150 — BaseQN consumers

- Implemented: embedding lookup.
- Implemented: QKV projection with a measured short-K fusion route.
- Implemented: gate/up projection and SwiGLU.
- Implemented by the core linear route: down/output projection.
- Implemented: greedy LM-head selection via the measured columnwise GEMV/argmax composition; two dedicated packed-reduction candidates were rejected.
- Implemented: grouped expert projection and fused expert SwiGLU over the existing padded MoE schedule. Rectangular projection retains one simdgroup; SwiGLU retains measured four-way split-K.

Phase acceptance:

- All canonical widths pass bit-layout tests.
- GEMV/GEMM pass correctness on realistic and edge shapes.
- At least Q4, Q6, and Q8 profile paths work end to end.
- No unmeasured routing threshold is committed.

### Phase 2: dense decoder parity

#### BRT-200 — extended position encoding

Status: implemented as reusable tensor kernels.

- `rotary_positioned`: shared/per-batch explicit positions, partial/full rotary,
  split-half or adjacent-pair layout, and D=64/128/256/512.
- `mrope`: shared/per-batch temporal/height/width positions, contiguous sections
  or Qwen-style `THWTHW...` interleaving, with continuous frequency indices.
- `qk_norm_rope_positioned`: packed QKV preparation with full-head Q/K RMSNorm,
  partial or multimodal rotary, D through 512, and explicit
  `norm_weight_offset` semantics.
- Llama-3 piecewise scaling, uniform linear scaling, and local/global theta are
  represented by the supplied cosine/sine tables. Kernel tests exercise scaled
  tables; no model policy or table generation is hidden in Metal.

#### BRT-210 — attention dimension and schedule coverage

Status: implemented/reused at the kernel boundary.

- Existing causal, variable-length, decode, paged, local/global-window, and
  explicit-scale operations cover decoder schedules.
- `cross_attention` adds independent encoder lengths, GQA mapping, exact FP32
  bias, D=64/128/256, and optional softcap after scale+bias and before softmax.
- The measured route uses direct online Metal through 128 keys and framework
  matmul/softmax for longer encoder memories.

#### BRT-220 — dense fused projections

- Implemented: BaseQN QKV and gate/up SwiGLU.
- Implemented/reused: bias, sigmoid output gate, and batched projection.
- Implemented: final-logit softcap.
- Dedicated candidates are retained only where they beat measured composition.

#### BRT-230 — architecture trace tests

Superseded for this repository by isolated operation, fused-versus-composed,
cross-backend parity, and direct MPS tests. Llama/Mistral, Qwen, and Gemma
layers are model graphs; adding trace runners here would violate the confirmed
kernel-only scope. Heterogeneous dimensions and schedule parameters are tested
directly at each reusable operation boundary.

Phase acceptance: all inferred dense tensor operations have independent
oracles and cross-backend coverage; current supported edge shapes remain in
the repository-wide correctness suite.

### Phase 3: cache and batching parity

#### BRT-300 — Q8_0 KV codec

Implemented under the QuixiCore-owned separate-plane ABI:

- `kv_cache_scatter_q8_0` encodes F32/F16/BF16 K and V, supports arbitrary
  logical slots, and ignores slot `-1`.
- `kv_cache_gather_q8_0` decodes arbitrary slots to F16/BF16/F32 and returns
  zeros for invalid logical slots.
- `kv_cache_copy_blocks_q8_0` functionally clones and remaps all four K/V
  code/scale planes while preserving its inputs.
- `paged_attention_q8_0` reads Q8_0 pages directly for D64/D128 MHA/GQA/MQA,
  explicit attention scale, and optional sliding windows.
- Independent exact code/scale fixtures, MLX/PyTorch parity, BF16 attention
  parity, and persistent-memory accounting are covered.

The codec is a storage-format cost and is not claimed to outperform dense BF16
copy operations. The direct paged read is retained because measurement shows
it avoids both dense cache bandwidth and an intermediate gather.

#### BRT-310 — mixed variable-length batches

Status: reusable kernel substrate already present.

- Packed input rows, fresh-prefill/continuation flags, per-request block
  tables, last-row gather, and batched output/argmax are existing explicit
  tensor operations.
- Constructing a continuously batched request graph is inference-engine work
  and is excluded.

#### BRT-320 — shared-prefix integration

Status: reusable kernel substrate present. Cascade/prefix attention, functional
block copy, and table remap are available. Prefix ownership, zero-copy lifetime,
copy-on-write, eviction, and rollback policy belong to a cache manager and are
explicitly excluded.

Phase acceptance: variable-length and block-table kernels, Q8_0 cache mutation,
functional copies, and batched attention/output primitives pass their isolated
contracts. No scheduler/cache-manager acceptance criterion remains in scope.

### Phase 4: hybrid and MoE parity

#### BRT-400 — Gated DeltaNet preparation

Status: implemented as reusable tensor operations.

- Dense or BaseQN kernels provide projections; `gdn_short_conv` applies the
  varlen depthwise causal convolution and functionally updates explicit FP32
  history slots.
- `gdn_qkv_prepare` splits the already-activated mixed projection and applies
  per-head Q/K RMSNorm with explicit scales (Qwen defaults `1/Dk` and
  `1/sqrt(Dk)`).
- `gdn_gate_beta` emits FP32 recurrence decay and beta values.
- `gdn_recur` consumes those tensors and functionally updates the explicit
  DeltaNet state pool.

#### BRT-410 — Gated DeltaNet output

Status: implemented as reusable tensor operations.

- `gdn_gated_rmsnorm` fuses per-value-head RMSNorm with the SiLU gate product.
- Dense or BaseQN core kernels provide the output projection.
- `sigmoid_mul` implements the full-attention `sigmoid(gate) * attention`
  epilogue, with analytic backward support.
- Both recurrent and short-convolution state pools are explicit, functional,
  slot-indexed inputs/outputs; no cache manager or inference engine is added.

#### BRT-420 — parallel prefill experiment

Optional optimization, not a parity gap. The implemented sequential recurrence
is the correctness baseline. A future chunked/parallel candidate must be kept
only if priority prompt lengths improve without state/tolerance regressions.

#### BRT-430 — BaseQN MoE

- Implemented: quantized expert stacks with runtime tensor dimensions.
- Implemented: fused gate/up expert projection and SwiGLU.
- Reused: per-layer widths through ordinary tensor shapes.
- Reused: shared expert through BaseQN core projection/SwiGLU operations.
- Reused: shared-expert scalar gate through elementwise tensor composition.
- Reused: Qwen/Gemma normalization choices through explicit existing router arguments.

Phase acceptance: isolated dense/MoE/GDN operations and fused-versus-composed
oracles pass; untouched recurrent-state slots remain bit-identical. Model-layer
trace runners are excluded with the inference engine.

### Phase 5: embedding-model parity

#### BRT-500 — Nomic BERT input and layer composition

Status: implemented/reused.

- `embedding_lookup_types` fuses scaled token and token-type gathers with
  independent invalid-id handling.
- Existing LayerNorm with bias, QKV, bidirectional attention, and SwiGLU
  provide the remaining layer operations.

#### BRT-510 — pooling and normalization

Status: implemented. `masked_mean_pool_rms_l2` consumes `(B,T,D)` BF16 states
and a mask, accumulates mean/RMS/L2 terms in FP32, supports
D=256/512/768/1024, and emits exact zero for all-masked rows.

Phase acceptance: independent embedding/pooling oracles, padding/empty cases,
normalization invariants, MLX/PyTorch parity, and direct MPS tests pass.

### Phase 6: multimodal parity

#### BRT-600 — Gemma/SigLIP vision

Status: implemented/reused as tensor operations.

- `extract_patches_2d` plus `vision_patch_projection` cover patch/conv
  projection; canonical patchification uses the measured framework reshape
  route and general padded/overlapping extraction uses direct Metal.
- `factorized_position_2d` gathers and sums independent learned x/y tables
  with explicit invalid-token zeroing. `interpolate_position_2d` remains the
  reusable learned-grid resize operation for models that use one table.
- `pool_tokens_by_position` implements padded or arbitrarily ordered
  coordinate-bucket pooling with Gemma's fixed `1/K^2` weights, `sqrt(D)`
  output scale, and output-validity mask. Dense regular grids retain the
  measured reshape/average or `avg_pool2d_tokens` composition.
- `vision_rope_2d` applies separate x/y split-half rotations within the first
  and second halves of each vision-attention head for D=64/128/256/512.
- `value_clip` composes before and after either dense or BaseQN projections to
  implement scalar-bounds clippable linears without introducing a tower API.
- Gemma's pixel affine `2*(x-0.5)` and post-pool standardization
  `(x-bias)*scale` are ordinary elementwise framework compositions around the
  patch/pool kernels; neither implies a distinct model-specific GPU primitive.
- Existing norms, matmul, attention, GLU, patch merge, projection, and tensor
  indexing cover vision blocks and feature splice arithmetic.

#### BRT-610 — Qwen VL vision

Status: implemented/reused. `extract_patches_3d` and
`vision_patch_projection_3d` cover arbitrary padded/overlapping NTHWC
temporal-spatial extraction and Conv3D-equivalent projection. Canonical
non-overlapping patches use measured reshape/transpose; direct Metal covers
general geometry. `qwen_vision_rope_2d` implements Qwen's global split-half
pairing with x/y frequency sections, distinct from Gemma's local axis blocks.
Learned-position interpolation is `interpolate_position_2d` plus indexed
gather/reordering. The normal merger is LayerNorm-before-
`extract_patches_2d` followed by dense/GELU/dense; the deep-stack merger moves
LayerNorm after extraction. This intentionally does not misuse the existing
post-shuffle fused norm for both variants. Three-axis positioned text M-RoPE,
attention/matmul/norm blocks, and tensor indexing cover the remaining
arithmetic. Position-id construction and tower orchestration are excluded.

#### BRT-620 — Gemma Conformer audio

Status: implemented/reused. Patch extraction plus projection covers 2-D
subsampling. `audio_relative_attention` implements the learned per-dimension
query scale, projected relative keys, block context, relative shift, optional
softcap, length mask, and online FP32 softmax. `audio_causal_depthwise_conv1d`
implements LightConv's asymmetric left-only geometry, while the underlying
depthwise primitive retains general asymmetric padding and optional fused SiLU.
`value_clip`, existing GLU/activation/norm operations, projections, residual
arithmetic, and tensor splice operations cover the remaining block semantics.

#### BRT-630 — Whisper

Status: implemented/reused. `audio_conv1d` covers the convolutional front end;
existing attention/cache/projection operations cover encoder/self-attention;
`cross_attention` covers independent encoder memory, GQA, lengths, bias, and
softcap. Timestamp/transcription policy is runtime behavior and excluded.

Phase acceptance: isolated block operations pass independent oracles for odd
and padded image geometry, variable audio lengths, empty attention memory, and
cross-backend parity. Media I/O and end-to-end tower/model graphs remain
outside kernels.

### Phase 7: adaptation and calibration

#### BRT-700 — LoRA

Status: implemented. `lora_apply_direct` fuses both F16 low-rank projections,
scale, and optional base add with a threadgroup-resident intermediate. The
measured public route uses it for M<=4/rank<=16 and framework F16 matmuls
otherwise; explicit forcing supports ranks through 256. Quantized base output
is supplied as the ordinary optional base tensor, keeping quant format and
adapter math independent.

#### BRT-710 — calibration absmax

Status: implemented. `calibration_absmax` returns FP32 per-channel maxima,
merges an optional prior result exactly for long-input chunking, propagates
NaN, and maps either signed infinity to positive infinity.

#### BRT-720 — output processing completion

- Greedy BaseQN LM head is implemented; stochastic modes compose materialized
  logits with the existing sampling kernels until a separate fusion wins.
- Final softcap is implemented as a measured direct transform.
- Existing batched penalties, bias, masks, top-k/top-p/min-p, categorical,
  rejection, beam, speculative, and greedy operations cover the inferred
  kernel surface. Host sampling policy remains excluded.

Phase acceptance: LoRA's optional-base/disabled composition remains explicit;
adapter replacement semantics and shape validation are tested at the operation
level; calibration and softcap match high-precision oracles across MLX and
PyTorch MPS.

## Correctness strategy

### Test levels

#### Level 0 — format invariants

- Exact code bytes.
- Exact scale/bias bytes.
- Group boundaries.
- Lane and byte ordering.
- Alignment and offsets.

#### Level 1 — isolated operations

- High-precision NumPy or framework oracle.
- Multiple dtypes.
- Minimum, priority, and edge shapes.
- Invalid input tests.

#### Level 2 — fused versus unfused

- Compare every fusion against composition from trusted primitives.
- Use the same rounded input dtype on both sides.
- Verify cache/state outputs as well as visible outputs.

#### Level 3 — architecture blocks

- Dense decoder layer.
- MoE decoder layer.
- Gated DeltaNet layer.
- BERT encoder layer.
- Vision block.
- Conformer block.
- Whisper encoder and decoder blocks.

#### Level 4 — model traces

- Several tokens of prefill and decode.
- KV state after every layer.
- Logits before sampling.
- Recurrent state.
- Multimodal feature splice.

Full-model golden outputs require an approved redistributable model or fixture.

### Required edge cases

- Odd row counts.
- Small and non-priority dimensions.
- Maximum declared head dimensions.
- GQA and MQA.
- Empty or length-one sequences.
- Mixed sequence lengths.
- Sliding-window boundary positions.
- Shared-prefix boundary positions.
- Partial rotary boundaries.
- M-RoPE axis transitions.
- Constant quantization groups.
- Negative-only and positive-only groups.
- Non-finite scale inputs.
- Expert ties and deterministic top-k.
- Untouched cache/state slots.
- Image padding and spatial merge edges.
- Audio padding and chunk boundaries.

### Tolerance policy

Each operation must declare:

- Oracle accumulation dtype.
- Input rounding point.
- Absolute and relative tolerance.
- Whether codes must be bit-exact.
- Whether selected token IDs must be exact.
- Whether cache/state must be exact or tolerance-based.

Quantized pack/unpack layout tests should be bit-exact. Floating GEMM and
attention tolerances must be derived from representative error data, not copied
from unrelated kernels.

## Performance strategy

All implementation and routing work is governed by
[perf/perf.md](perf/perf.md), with durable decisions recorded in
[perf/optimization_status.md](perf/optimization_status.md) and baselines in
[perf/baseline_status.md](perf/baseline_status.md).

### Required focused-run record

Every affected kernel run must record:

- Kernel and integration path.
- Dtype and quant format.
- Shape set.
- Correctness result.
- Baseline and candidate timing.
- Apple Silicon model.
- macOS version.
- Xcode and Metal toolchain.
- Full command line.
- Warmups and iterations.
- Median.
- Variance or min/max.
- Keep or reject decision.

Raw output goes under perf/results. Large profiler traces must not be
committed.

### Baseline hierarchy

Compare against:

1. Existing QuixiCore implementation, if any.
2. Framework composition using MLX or PyTorch MPS.
3. Naive correctness kernel where available.
4. Approved BaseRT black-box timing only if licensing permits and measurement
   methodology is comparable.

Do not claim parity or a speedup from different hardware, model files, prompt
lengths, synchronization boundaries, or batching behavior.

### Provisional priority shapes

Final shapes must come from the manifest and supported model catalog. Initial
measurement buckets should cover:

- Decode GEMV: one row and small continuous batches.
- Prefill GEMM: prompt rows around 128, 512, and 2048.
- Output projection: one row and batch rows through the configured scheduler
  maximum.
- Attention: short, medium, and long contexts; local and global windows.
- Gated DeltaNet: single-token decode and long prefill.
- MoE: low token counts, prompt batches, and realistic expert top-k.
- Vision: canonical image resolution plus one interpolation case.
- Audio: short clip, chunk boundary, and long clip.

These are planning buckets, not committed routing thresholds.

### Optimization rule

Change one meaningful factor at a time:

- Tile shape.
- Launch geometry.
- Memory layout.
- Fusion.
- Barrier placement.
- Dequantization strategy.
- Routing threshold.
- Specialization.

Keep a candidate only if it:

- Passes correctness.
- Improves realistic priority shapes.
- Does not materially regress supported edge shapes.
- Has its decision recorded.

If Apple Silicon and Metal timing are unavailable, restrict the change to
documentation or scaffolding and make no performance claim.

## Metadata and documentation

For each completed operation:

- Add or update .quixicore/kernels.yaml.
- Add operation-level status rather than family-only status.
- Add or update .quixicore/quant-formats.yaml.
- Document supported layouts, scale types, and group sizes.
- Document bindings.
- Link correctness tests.
- Link benchmark identifiers.
- Record optimized and fallback variants.
- Update docs/repository-structure.md if the audio family is added.
- Add operation README files for non-obvious contracts.
- Record provenance.

Do not mark a family implemented merely because one specialization exists.
Operation metadata must distinguish implemented, partial, experimental,
planned, unsupported, and capability-gated variants.

## CI and verification commands

The final verification set for code-bearing work is:

    scripts/build
    scripts/test correctness -q
    scripts/test parity -q
    scripts/test mps -q

Focused benchmarks use:

    .venv/bin/python perf/bench_kernels.py --backend mlx --preset quick --kernel <kernel>
    .venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel <kernel>

Add targeted test selections during development, but do not replace the
repository-wide gates with only targeted tests before handoff.

Documentation-only edits may skip Metal benchmarks and must not claim a
performance improvement.

## Milestones and dependency order

| Milestone | Contents | Depends on |
| --- | --- | --- |
| M0 — authorized target | Provenance, source rules, manifest, fixtures | None |
| M1 — BaseQN core | Descriptor, unpack, dequant, GEMV, GEMM | M0 |
| M2 — dense text parity | Embedding, RoPE, attention, dense projections, LM head | M1 |
| M3 — serving parity | Q8_0 KV, mixed batches, batched output | M0, M1, M2 |
| M4 — hybrid/MoE parity | Full Gated DeltaNet and BaseQN MoE | M1, M2 |
| M5 — embedding parity | Nomic BERT | M1, M2 |
| M6 — multimodal parity | Gemma/Qwen vision, Conformer, Whisper | M1, M2 |
| M7 — adaptation parity | LoRA, calibration, remaining output transforms | M1, M2 |
| M8 — closure | Manifest reconciliation, full regression and performance sweep | M1-M7 |

M0 freezes the clean-room evidence boundary; it is not a blocker on inventing
generic operations from public semantics. Complete private-pipeline identity is
neither knowable nor claimed.

## Definition of done

An individual operation is done when:

- Its semantics are documented.
- Its provenance is recorded.
- Metal source and launch code are present.
- MLX and PyTorch MPS bindings agree.
- Public exposure is intentional.
- Input validation is complete.
- Format/layout tests pass.
- Correctness tests pass.
- Cross-backend parity passes.
- Supported edge shapes pass.
- A focused performance run is recorded.
- Keep/reject decisions are in the optimization notebook.
- Metadata names tests, bindings, benchmarks, and variants.
- No unrelated user changes were reverted.

BaseRT kernel parity as a project is done when:

- Every inferred operation row is implemented or explicitly marked
  unsupported with a reason.
- No inferred capability is presented as confirmed without evidence.
- Dense, MoE, hybrid, embedding, vision, audio, and Whisper integration traces
  pass for the approved target set.
- Quantized prefill and decode cover every approved BaseQN format/profile.
- Cache and recurrent state remain correct across continuation, batching, and
  prefix reuse.
- Performance measurements exist on the supported Apple Silicon target.
- The final parity report identifies any remaining runtime-only gaps.

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Proprietary engine source unavailable | Private pipeline identity is unknowable | Maintain an inferred semantic manifest and implement clean-room operations only |
| Public format exceeds released kernel coverage | False parity scope | Separate confirmed runtime behavior from schema capability |
| Similar format names hide different layouts | Silent corruption | Bit-exact fixtures and explicit descriptors |
| Too many Metal specializations | Build size and pipeline latency | Instantiate profile-backed combinations first |
| Monolithic model-specific fusions | Poor reuse and hard testing | Start from reusable primitive boundaries |
| Framework and Metal rounding differ | Brittle tests | Define input rounding and tolerance per operation |
| Q8_0 KV names could hide incompatible layouts | Silent cache corruption | The QuixiCore separate-plane ABI is frozen in metadata and exact fixtures; packed weight Q8_0 is never accepted by KV operations |
| Head dimensions exceed current tile substrate | Large rewrite | Add only evidence-backed dimensions and benchmark alternatives |
| Sequential GDN prefill is slow | Hybrid prompt performance gap | Separate chunked/parallel optimization milestone |
| Multimodal preprocessing scope expands | Kernel project becomes a runtime project | Use tensor inputs and explicit CPU/framework boundaries |
| Performance claims are incomparable | Misleading results | Enforce the performance notebook record |

## Non-blocking target choices

Implementation proceeds clean-room without engine source or binary-derived
details. The remaining choices affect prioritization, not authorization:

1. Which public BaseRT release/model catalog should freeze the inferred list?
2. Which Apple GPU generations and model families are release priorities?
3. Should media-facing tests start at decoded image tensors and mel features?
4. Which redistributable model-derived fixtures, if any, should supplement the
   framework-independent mathematical oracles?

## Reference index

### BaseRT reference

- [.reference/baseRT/README.md](.reference/baseRT/README.md)
- [.reference/baseRT/LICENSE](.reference/baseRT/LICENSE)
- [.reference/baseRT/NOTICE](.reference/baseRT/NOTICE)
- [.reference/baseRT/docs/reference/engine-releases.md](.reference/baseRT/docs/reference/engine-releases.md)
- [.reference/baseRT/include/baseRT/baseRT.h](.reference/baseRT/include/baseRT/baseRT.h)
- [.reference/baseRT/include/baseRT/types.h](.reference/baseRT/include/baseRT/types.h)
- [.reference/baseRT/base-convert/FORMAT.md](.reference/baseRT/base-convert/FORMAT.md)
- [.reference/baseRT/base-convert/CANONICAL_QUANT_SPEC.md](.reference/baseRT/base-convert/CANONICAL_QUANT_SPEC.md)
- [.reference/baseRT/base-convert/crates/base-arch/src/lib.rs](.reference/baseRT/base-convert/crates/base-arch/src/lib.rs)
- [.reference/baseRT/base-convert/crates/base-arch/src/llama.rs](.reference/baseRT/base-convert/crates/base-arch/src/llama.rs)
- [.reference/baseRT/base-convert/crates/base-arch/src/qwen.rs](.reference/baseRT/base-convert/crates/base-arch/src/qwen.rs)
- [.reference/baseRT/base-convert/crates/base-arch/src/gemma.rs](.reference/baseRT/base-convert/crates/base-arch/src/gemma.rs)
- [.reference/baseRT/base-convert/crates/base-arch/src/bert.rs](.reference/baseRT/base-convert/crates/base-arch/src/bert.rs)
- [.reference/baseRT/base-convert/crates/base-quant/src/base_qn.rs](.reference/baseRT/base-convert/crates/base-quant/src/base_qn.rs)
- [.reference/baseRT/base-convert/crates/base-quant/src/base_q2.rs](.reference/baseRT/base-convert/crates/base-quant/src/base_q2.rs)
- [.reference/baseRT/base-convert/crates/base-quant/src/base_q3.rs](.reference/baseRT/base-convert/crates/base-quant/src/base_q3.rs)
- [.reference/baseRT/base-convert/crates/base-quant/src/base_q4.rs](.reference/baseRT/base-convert/crates/base-quant/src/base_q4.rs)
- [.reference/baseRT/base-convert/crates/base-quant/src/base_q5.rs](.reference/baseRT/base-convert/crates/base-quant/src/base_q5.rs)
- [.reference/baseRT/base-convert/crates/base-quant/src/base_q6.rs](.reference/baseRT/base-convert/crates/base-quant/src/base_q6.rs)
- [.reference/baseRT/base-convert/crates/base-quant/src/base_q8.rs](.reference/baseRT/base-convert/crates/base-quant/src/base_q8.rs)
- [.reference/baseRT/base-convert/crates/base-format/src/header.rs](.reference/baseRT/base-convert/crates/base-format/src/header.rs)
- [.reference/baseRT/base-convert/profiles](.reference/baseRT/base-convert/profiles)
- [.reference/baseRT/base-convert/crates/base-hub/catalog.json](.reference/baseRT/base-convert/crates/base-hub/catalog.json)
- [.reference/baseRT/benchmarks/README.md](.reference/baseRT/benchmarks/README.md)

### QuixiCore Metal

- [README.md](README.md)
- [docs/repository-structure.md](docs/repository-structure.md)
- [.quixicore/kernels.yaml](.quixicore/kernels.yaml)
- [.quixicore/quant-formats.yaml](.quixicore/quant-formats.yaml)
- [perf/perf.md](perf/perf.md)
- [perf/optimization_status.md](perf/optimization_status.md)
- [perf/baseline_status.md](perf/baseline_status.md)
- [kernels/linear_attention/gdn](kernels/linear_attention/gdn)
- [kernels/norms/qk_norm_rope](kernels/norms/qk_norm_rope)
- [kernels/attention/attn_varlen](kernels/attention/attn_varlen)
- [kernels/attention/paged_attn_v2](kernels/attention/paged_attn_v2)
- [kernels/serving/kv_cache](kernels/serving/kv_cache)
- [kernels/quantization/qgemm](kernels/quantization/qgemm)
- [kernels/quantization/qgemv](kernels/quantization/qgemv)
- [kernels/quantization/dequant_gather](kernels/quantization/dequant_gather)
- [kernels/quantization/lm_head](kernels/quantization/lm_head)
- [kernels/moe/moe](kernels/moe/moe)
- [kernels/sampling/sampling](kernels/sampling/sampling)
- [kernels/vision/patch_merge](kernels/vision/patch_merge)
- [kernels/serving/mean_pool_rms_l2](kernels/serving/mean_pool_rms_l2)

### Clean-room implementation references

- `/Users/eric/llama.cpp/ggml/src/ggml-common.h` (`block_q8_0` mathematical
  block definition; packed representation not adopted)
- `/Users/eric/llama.cpp/ggml/src/ggml-quants.c`
  (`quantize_row_q8_0_ref` rounding and zero-block semantics)
- `/Users/eric/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` (safe-FP Q8_0
  encode and direct quantized-attention scheduling ideas)
- [kernels/serving/kv_cache](kernels/serving/kv_cache)
- [kernels/quantization/quant_rt](kernels/quantization/quant_rt)
