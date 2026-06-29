# ThunderMittens / ThunderKittens Kernel Discrepancy Report

Scope: this compares `.reference/ThunderKittens/kernels` against `ThunderMittens/kernels`.

Summary:

- ThunderKittens non-baseline CUDA kernel implementations: 58
- ThunderMittens Metal/MLX kernel implementations: 3
- ThunderKittens baseline/reference implementations not present in ThunderMittens: 11
- ThunderKittens non-baseline support files not present in ThunderMittens: 97
- ThunderMittens-only kernel-package files: 19

ThunderMittens currently has these kernel directories:

- `ThunderMittens/kernels/add_rt`
- `ThunderMittens/kernels/attn_fwd`
- `ThunderMittens/kernels/matmul_custom`

Possible partial conceptual overlaps:

- `ThunderMittens/kernels/attn_fwd` is a partial attention-forward port, but it is not a direct path/name match for any of the ThunderKittens attention kernels below.
- `ThunderMittens/kernels/matmul_custom` is a partial GEMM/matmul port, but it is not a direct path/name match for the ThunderKittens GEMM suite below.
- `ThunderMittens/kernels/add_rt` has no direct ThunderKittens kernel counterpart under `.reference/ThunderKittens/kernels`.

## Missing ThunderKittens Non-Baseline Kernel Implementations

These `.cu` files exist in ThunderKittens and do not have direct ThunderMittens kernel counterparts.

### Attention

- `.reference/ThunderKittens/kernels/attention/bf16_b300_mha_causal/bf16_b300_mha_causal.cu`
- `.reference/ThunderKittens/kernels/attention/bf16_b300_mha_noncausal/bf16_b300_mha_noncausal.cu`
- `.reference/ThunderKittens/kernels/attention/mha_h100/mha_h100.cu`
- `.reference/ThunderKittens/kernels/attention/mha_h100_lcf/mha_h100_lcf.cu`

### Based / Linear Attention

- `.reference/ThunderKittens/kernels/based/linear_attn.cu`
- `.reference/ThunderKittens/kernels/linear_attention/linear_attention.cu`

### FFTConv

- `.reference/ThunderKittens/kernels/fftconv/fftconv_non_pc.cu`
- `.reference/ThunderKittens/kernels/fftconv/fftconv_pc.cu`

### Flux

- `.reference/ThunderKittens/kernels/flux/flux_gate.cu`
- `.reference/ThunderKittens/kernels/flux/flux_gelu.cu`

### GEMM

- `.reference/ThunderKittens/kernels/gemm/bf16_b200/bf16_b200_gemm.cu`
- `.reference/ThunderKittens/kernels/gemm/bf16_h100/bf16_h100_gemm.cu`
- `.reference/ThunderKittens/kernels/gemm/fp8_b200/fp8_b200_gemm.cu`
- `.reference/ThunderKittens/kernels/gemm/fp8_h100/fp8_h100_gemm.cu`
- `.reference/ThunderKittens/kernels/gemm/fp8_h100_scaled/fp8_h100_gemm_scaled.cu`
- `.reference/ThunderKittens/kernels/gemm/int8_b200/int8_b200_gemm.cu`
- `.reference/ThunderKittens/kernels/gemm/int8_h100/int8_h100_gemm.cu`
- `.reference/ThunderKittens/kernels/gemm/mxfp8_b200/mxfp8_b200_gemm.cu`
- `.reference/ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu`

### Educational GEMM

- `.reference/ThunderKittens/kernels/gemm/educational_b200/launch.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_b200/level_01.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_b200/level_02.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_b200/level_03.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_b200/level_04.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_b200/level_05.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_b200/level_06.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_b200/level_07.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_b200/level_08.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_b200/level_09.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_h100/launch.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_h100/level_01.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_h100/level_02.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_h100/level_03.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_h100/level_04.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_h100/level_05.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_h100/level_06.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_h100/level_07.cu`
- `.reference/ThunderKittens/kernels/gemm/educational_h100/level_08.cu`

### Hedgehog

- `.reference/ThunderKittens/kernels/hedgehog/hedgehog.cu`

### LayerNorm

- `.reference/ThunderKittens/kernels/layernorm/layernorm.cu`

### Mamba2

- `.reference/ThunderKittens/kernels/mamba2/mamba2.cu`

### Parallel / Distributed

- `.reference/ThunderKittens/kernels/parallel/ag_gemm/ag_gemm_b200.cu`
- `.reference/ThunderKittens/kernels/parallel/ag_gemm/ag_gemm_h100.cu`
- `.reference/ThunderKittens/kernels/parallel/ag_gemm_fp8/ag_gemm_fp8_b200.cu`
- `.reference/ThunderKittens/kernels/parallel/all_gather/all_gather.cu`
- `.reference/ThunderKittens/kernels/parallel/all_reduce/all_reduce.cu`
- `.reference/ThunderKittens/kernels/parallel/all_reduce_educational/all_reduce_educational.cu`
- `.reference/ThunderKittens/kernels/parallel/all_to_all/all_to_all.cu`
- `.reference/ThunderKittens/kernels/parallel/gemm_ar/gemm_ar_h100.cu`
- `.reference/ThunderKittens/kernels/parallel/gemm_ar/gemm_ar_h100_lcsc.cu`
- `.reference/ThunderKittens/kernels/parallel/gemm_rs/gemm_rs_b200.cu`
- `.reference/ThunderKittens/kernels/parallel/gemm_rs/gemm_rs_h100.cu`
- `.reference/ThunderKittens/kernels/parallel/gemm_rs_fp8/gemm_rs_fp8_b200.cu`
- `.reference/ThunderKittens/kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu`
- `.reference/ThunderKittens/kernels/parallel/reduce_scatter/reduce_scatter.cu`
- `.reference/ThunderKittens/kernels/parallel/ring_attn/ring_attn_h100.cu`
- `.reference/ThunderKittens/kernels/parallel/ulysses_attn/ulysses_attn.cu`

### Rotary

- `.reference/ThunderKittens/kernels/rotary/rotary.cu`

## ThunderKittens Baseline / Reference Implementations Not Present

These live under `baselines/` in ThunderKittens and are not counted as primary kernel ports above.

- `.reference/ThunderKittens/kernels/fftconv/baselines/tk_fftconv.py`
- `.reference/ThunderKittens/kernels/gemm/baselines/bf16_cublas/bf16_cublas_gemm.cu`
- `.reference/ThunderKittens/kernels/gemm/baselines/bf16_cublas_lt/bf16_cublas_lt_gemm.cu`
- `.reference/ThunderKittens/kernels/gemm/baselines/fp8_cublas_lt/fp8_cublas_lt_gemm.cu`
- `.reference/ThunderKittens/kernels/gemm/baselines/int8_cublas_lt/int8_cublas_lt_gemm.cu`
- `.reference/ThunderKittens/kernels/gemm/baselines/mxfp8_cublas_lt/mxfp8_cublas_lt_gemm.cu`
- `.reference/ThunderKittens/kernels/gemm/baselines/nvfp4_cublas_lt/nvfp4_cublas_lt_gemm.cu`
- `.reference/ThunderKittens/kernels/layernorm/baselines/layer_norm_triton.py`
- `.reference/ThunderKittens/kernels/mamba2/baselines/ssd_minimal.py`
- `.reference/ThunderKittens/kernels/rotary/baselines/rotary.py`
- `.reference/ThunderKittens/kernels/rotary/baselines/triton_rotary.py`

## Missing ThunderKittens Support / Test / Benchmark Files

These non-`.cu`, non-baseline files are part of the ThunderKittens kernel tree and do not have direct ThunderMittens counterparts.

- `.reference/ThunderKittens/kernels/attention/bf16_b300_mha_causal/Makefile`
- `.reference/ThunderKittens/kernels/attention/bf16_b300_mha_causal/test.py`
- `.reference/ThunderKittens/kernels/attention/bf16_b300_mha_noncausal/Makefile`
- `.reference/ThunderKittens/kernels/attention/bf16_b300_mha_noncausal/test.py`
- `.reference/ThunderKittens/kernels/attention/mha_h100/Makefile`
- `.reference/ThunderKittens/kernels/attention/mha_h100/benchmark.py`
- `.reference/ThunderKittens/kernels/attention/mha_h100/gentests.py`
- `.reference/ThunderKittens/kernels/attention/mha_h100/harness.impl`
- `.reference/ThunderKittens/kernels/attention/mha_h100/test_correctness.py`
- `.reference/ThunderKittens/kernels/attention/mha_h100_lcf/Makefile`
- `.reference/ThunderKittens/kernels/attention/mha_h100_lcf/gentests.py`
- `.reference/ThunderKittens/kernels/based/Makefile`
- `.reference/ThunderKittens/kernels/based/benchmark.py`
- `.reference/ThunderKittens/kernels/based/gentests.py`
- `.reference/ThunderKittens/kernels/based/harness.impl`
- `.reference/ThunderKittens/kernels/common.mk`
- `.reference/ThunderKittens/kernels/fftconv/Makefile`
- `.reference/ThunderKittens/kernels/fftconv/benchmark.py`
- `.reference/ThunderKittens/kernels/fftconv/gentests.py`
- `.reference/ThunderKittens/kernels/fftconv/gentests_1024.py`
- `.reference/ThunderKittens/kernels/fftconv/harness.impl`
- `.reference/ThunderKittens/kernels/fftconv/pytorch_ref.py`
- `.reference/ThunderKittens/kernels/fftconv/test_correctness.py`
- `.reference/ThunderKittens/kernels/flux/Makefile`
- `.reference/ThunderKittens/kernels/flux/benchmark.py`
- `.reference/ThunderKittens/kernels/gemm/bf16_b200/Makefile`
- `.reference/ThunderKittens/kernels/gemm/bf16_h100/Makefile`
- `.reference/ThunderKittens/kernels/gemm/common.cuh`
- `.reference/ThunderKittens/kernels/gemm/educational_b200/Makefile`
- `.reference/ThunderKittens/kernels/gemm/educational_b200/README.md`
- `.reference/ThunderKittens/kernels/gemm/educational_h100/Makefile`
- `.reference/ThunderKittens/kernels/gemm/educational_h100/README.md`
- `.reference/ThunderKittens/kernels/gemm/fp8_b200/Makefile`
- `.reference/ThunderKittens/kernels/gemm/fp8_h100/Makefile`
- `.reference/ThunderKittens/kernels/gemm/fp8_h100_scaled/Makefile`
- `.reference/ThunderKittens/kernels/gemm/fp8_h100_scaled/visualize.py`
- `.reference/ThunderKittens/kernels/gemm/int8_b200/Makefile`
- `.reference/ThunderKittens/kernels/gemm/int8_h100/Makefile`
- `.reference/ThunderKittens/kernels/gemm/mxfp8_b200/Makefile`
- `.reference/ThunderKittens/kernels/gemm/mxfp8_b200/test_gemm.py`
- `.reference/ThunderKittens/kernels/gemm/mxfp8_b200/test_quantize.py`
- `.reference/ThunderKittens/kernels/gemm/nvfp4_b200/Makefile`
- `.reference/ThunderKittens/kernels/gemm/nvfp4_b200/test_gemm.py`
- `.reference/ThunderKittens/kernels/gemm/nvfp4_b200/test_quantize.py`
- `.reference/ThunderKittens/kernels/hedgehog/Makefile`
- `.reference/ThunderKittens/kernels/hedgehog/benchmark.py`
- `.reference/ThunderKittens/kernels/hedgehog/gentests.py`
- `.reference/ThunderKittens/kernels/hedgehog/harness.impl`
- `.reference/ThunderKittens/kernels/hedgehog/test_correctness.py`
- `.reference/ThunderKittens/kernels/hedgehog/util.py`
- `.reference/ThunderKittens/kernels/layernorm/Makefile`
- `.reference/ThunderKittens/kernels/layernorm/benchmark.py`
- `.reference/ThunderKittens/kernels/layernorm/gentests.py`
- `.reference/ThunderKittens/kernels/layernorm/harness.impl`
- `.reference/ThunderKittens/kernels/layernorm/test_correctness.py`
- `.reference/ThunderKittens/kernels/linear_attention/Makefile`
- `.reference/ThunderKittens/kernels/linear_attention/gentests.py`
- `.reference/ThunderKittens/kernels/mamba2/Makefile`
- `.reference/ThunderKittens/kernels/mamba2/benchmark.py`
- `.reference/ThunderKittens/kernels/mamba2/gentests.py`
- `.reference/ThunderKittens/kernels/mamba2/harness.impl`
- `.reference/ThunderKittens/kernels/mamba2/harness2.impl`
- `.reference/ThunderKittens/kernels/mamba2/harness3.impl`
- `.reference/ThunderKittens/kernels/mamba2/test_correctness.py`
- `.reference/ThunderKittens/kernels/parallel/README.md`
- `.reference/ThunderKittens/kernels/parallel/ag_gemm/Makefile`
- `.reference/ThunderKittens/kernels/parallel/ag_gemm/benchmark.py`
- `.reference/ThunderKittens/kernels/parallel/ag_gemm_fp8/Makefile`
- `.reference/ThunderKittens/kernels/parallel/ag_gemm_fp8/benchmark.py`
- `.reference/ThunderKittens/kernels/parallel/all_gather/Makefile`
- `.reference/ThunderKittens/kernels/parallel/all_gather/benchmark.py`
- `.reference/ThunderKittens/kernels/parallel/all_reduce/Makefile`
- `.reference/ThunderKittens/kernels/parallel/all_reduce/benchmark.py`
- `.reference/ThunderKittens/kernels/parallel/all_reduce_educational/Makefile`
- `.reference/ThunderKittens/kernels/parallel/all_to_all/Makefile`
- `.reference/ThunderKittens/kernels/parallel/all_to_all/benchmark.py`
- `.reference/ThunderKittens/kernels/parallel/common.py`
- `.reference/ThunderKittens/kernels/parallel/gemm_ar/Makefile`
- `.reference/ThunderKittens/kernels/parallel/gemm_ar/benchmark.py`
- `.reference/ThunderKittens/kernels/parallel/gemm_rs/Makefile`
- `.reference/ThunderKittens/kernels/parallel/gemm_rs/benchmark.py`
- `.reference/ThunderKittens/kernels/parallel/gemm_rs_fp8/Makefile`
- `.reference/ThunderKittens/kernels/parallel/gemm_rs_fp8/benchmark.py`
- `.reference/ThunderKittens/kernels/parallel/moe_dispatch_gemm/Makefile`
- `.reference/ThunderKittens/kernels/parallel/moe_dispatch_gemm/benchmark.py`
- `.reference/ThunderKittens/kernels/parallel/reduce_scatter/Makefile`
- `.reference/ThunderKittens/kernels/parallel/reduce_scatter/benchmark.py`
- `.reference/ThunderKittens/kernels/parallel/ring_attn/Makefile`
- `.reference/ThunderKittens/kernels/parallel/ring_attn/benchmark.py`
- `.reference/ThunderKittens/kernels/parallel/ulysses_attn/Makefile`
- `.reference/ThunderKittens/kernels/parallel/ulysses_attn/benchmark.py`
- `.reference/ThunderKittens/kernels/rotary/Makefile`
- `.reference/ThunderKittens/kernels/rotary/benchmark.py`
- `.reference/ThunderKittens/kernels/rotary/gentests.py`
- `.reference/ThunderKittens/kernels/rotary/harness.impl`
- `.reference/ThunderKittens/kernels/rotary/harness2.impl`
- `.reference/ThunderKittens/kernels/rotary/test_correctness.py`

## ThunderMittens-Only Kernel-Package Files

These files exist in ThunderMittens but not in the ThunderKittens kernel tree. They are mostly the MLX extension packaging and current Metal ports.

- `ThunderMittens/kernels/CMakeLists.txt`
- `ThunderMittens/kernels/add_rt/add_rt.cpp`
- `ThunderMittens/kernels/add_rt/add_rt.h`
- `ThunderMittens/kernels/add_rt/add_rt.metal`
- `ThunderMittens/kernels/attn_fwd/attn_fwd.cpp`
- `ThunderMittens/kernels/attn_fwd/attn_fwd.h`
- `ThunderMittens/kernels/attn_fwd/attn_fwd.metal`
- `ThunderMittens/kernels/attn_fwd/correctness/c_attn.m`
- `ThunderMittens/kernels/attn_fwd/correctness/gentests.py`
- `ThunderMittens/kernels/bindings.cpp`
- `ThunderMittens/kernels/matmul_custom/matmul_custom.cpp`
- `ThunderMittens/kernels/matmul_custom/matmul_custom.h`
- `ThunderMittens/kernels/matmul_custom/matmul_custom.metal`
- `ThunderMittens/kernels/pyproject.toml`
- `ThunderMittens/kernels/requirements.txt`
- `ThunderMittens/kernels/setup.py`
- `ThunderMittens/kernels/time_attn.py`
- `ThunderMittens/kernels/time_gemm.py`
- `ThunderMittens/kernels/tk/__init__.py`

