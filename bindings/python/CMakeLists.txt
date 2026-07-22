cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

set(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
set(KERNELS_ROOT "${REPO_ROOT}/kernels")
set(METAL_INCLUDE_ROOT "${REPO_ROOT}/include/metal")

set(KERNEL_FAMILY_INCLUDE_DIRS
    "${KERNELS_ROOT}/activations"
    "${KERNELS_ROOT}/attention"
    "${KERNELS_ROOT}/common"
    "${KERNELS_ROOT}/linear_attention"
    "${KERNELS_ROOT}/matmul"
    "${KERNELS_ROOT}/moe"
    "${KERNELS_ROOT}/norms"
    "${KERNELS_ROOT}/optimizers"
    "${KERNELS_ROOT}/quantization"
    "${KERNELS_ROOT}/sampling"
    "${KERNELS_ROOT}/serving"
    "${KERNELS_ROOT}/ssm"
    "${KERNELS_ROOT}/utils"
    "${KERNELS_ROOT}/vision")

set(KERNEL_CPP_SOURCES
    "${KERNELS_ROOT}/utils/add_rt/add_rt.cpp"
    "${KERNELS_ROOT}/attention/attn_fwd/attn_fwd.cpp"
    "${KERNELS_ROOT}/attention/attn_fwd_sg/attn_fwd_sg.cpp"
    "${KERNELS_ROOT}/matmul/matmul_custom/matmul_custom.cpp"
    "${KERNELS_ROOT}/norms/layernorm/layernorm.cpp"
    "${KERNELS_ROOT}/norms/rms_norm/rms_norm.cpp"
    "${KERNELS_ROOT}/norms/rms_norm_residual_next/rms_norm_residual_next.cpp"
    "${KERNELS_ROOT}/serving/mean_pool_rms_l2/mean_pool_rms_l2.cpp"
    "${KERNELS_ROOT}/norms/add_norm/add_norm.cpp"
    "${KERNELS_ROOT}/activations/softmax/softmax.cpp"
    "${KERNELS_ROOT}/attention/rotary/rotary.cpp"
    "${KERNELS_ROOT}/attention/rope_kv/rope_kv.cpp"
    "${KERNELS_ROOT}/norms/qk_norm_rope/qk_norm_rope.cpp"
    "${KERNELS_ROOT}/norms/qk_norm_rope/qk_norm_rope_kv_f16.cpp"
    "${KERNELS_ROOT}/ssm/selective_scan/selective_scan.cpp"
    "${KERNELS_ROOT}/linear_attention/gdn/gdn.cpp"
    "${KERNELS_ROOT}/quantization/act_quant/act_quant.cpp"
    "${KERNELS_ROOT}/quantization/fake_quant/fake_quant.cpp"
    "${KERNELS_ROOT}/quantization/fake_quant_fp8/fake_quant_fp8.cpp"
    "${KERNELS_ROOT}/quantization/weight_quant_ternary/weight_quant_ternary.cpp"
    "${KERNELS_ROOT}/quantization/quantize_tq2_0/quantize_tq2_0.cpp"
    "${KERNELS_ROOT}/quantization/ternary_stats/ternary_stats.cpp"
    "${KERNELS_ROOT}/serving/minference/minference.cpp"
    "${KERNELS_ROOT}/quantization/turboquant/turboquant.cpp"
    "${KERNELS_ROOT}/utils/marginal/marginal.cpp"
    "${KERNELS_ROOT}/serving/indexer/indexer.cpp"
    "${KERNELS_ROOT}/attention/mla/mla.cpp"
    "${KERNELS_ROOT}/activations/gelu/gelu.cpp"
    "${KERNELS_ROOT}/utils/dropout/dropout.cpp"
    "${KERNELS_ROOT}/optimizers/optim/adamw.cpp"
    "${KERNELS_ROOT}/serving/embedding/embedding.cpp"
    "${KERNELS_ROOT}/activations/glu/glu.cpp"
    "${KERNELS_ROOT}/utils/hadamard/hadamard.cpp"
    "${KERNELS_ROOT}/serving/kv_cache/kv_cache.cpp"
    "${KERNELS_ROOT}/attention/paged_attn_v2/paged_attn_v2.cpp"
    "${KERNELS_ROOT}/quantization/quant_rt/quant_rt.cpp"
    "${KERNELS_ROOT}/sampling/sampling/sampling.cpp"
    "${KERNELS_ROOT}/sampling/sampling/transforms.cpp"
    "${KERNELS_ROOT}/moe/moe/moe.cpp"
    "${KERNELS_ROOT}/attention/attn_causal/attn_causal.cpp"
    "${KERNELS_ROOT}/attention/attn_varlen/attn_varlen.cpp"
    "${KERNELS_ROOT}/quantization/lm_head/lm_head.cpp"
    "${KERNELS_ROOT}/utils/cross_entropy/cross_entropy.cpp"
    "${KERNELS_ROOT}/utils/kd_kl_topk/kd_kl_topk.cpp"
    "${KERNELS_ROOT}/utils/kd_kl_dense/kd_kl_dense.cpp"
    "${KERNELS_ROOT}/matmul/flux/flux.cpp"
    "${KERNELS_ROOT}/matmul/gemm_staged/gemm_staged.cpp"
    "${KERNELS_ROOT}/matmul/gemm_v3/gemm_v3.cpp"
    "${KERNELS_ROOT}/attention/attn_multiwarp/attn_multiwarp.cpp"
    "${KERNELS_ROOT}/linear_attention/linear_attn/linear_attn.cpp"
    "${KERNELS_ROOT}/linear_attention/hedgehog/hedgehog.cpp"
    "${KERNELS_ROOT}/linear_attention/lin_attn_causal/lin_attn_causal.cpp"
    "${KERNELS_ROOT}/ssm/mamba2/mamba2.cpp"
    "${KERNELS_ROOT}/linear_attention/lin_attn_decay/lin_attn_decay.cpp"
    "${KERNELS_ROOT}/linear_attention/based/based.cpp"
    "${KERNELS_ROOT}/attention/attn_bwd/attn_bwd.cpp"
    "${KERNELS_ROOT}/matmul/cmplx_matmul/cmplx_matmul.cpp"
    "${KERNELS_ROOT}/ssm/fftconv/fftconv.cpp"
    "${KERNELS_ROOT}/quantization/qgemm/qgemm.cpp"
    "${KERNELS_ROOT}/quantization/qgemm_bwd/qgemm_bwd.cpp"
    "${KERNELS_ROOT}/quantization/qgemm_fused/qgemm_fused.cpp"
    "${KERNELS_ROOT}/quantization/qgemv/qgemv.cpp"
    "${KERNELS_ROOT}/quantization/qgemv_fused/qgemv_fused.cpp"
    "${KERNELS_ROOT}/quantization/qflux/qflux.cpp"
    "${KERNELS_ROOT}/quantization/qgemv_int/qgemv_int.cpp"
    "${KERNELS_ROOT}/attention/attn_q/attn_q.cpp"
    "${KERNELS_ROOT}/attention/attn_decode/attn_decode.cpp"
    "${KERNELS_ROOT}/quantization/qgemm_int/qgemm_int.cpp"
    "${KERNELS_ROOT}/attention/swin_attn/swin_attn.cpp"
    "${KERNELS_ROOT}/matmul/decode_linear/decode_linear.cpp"
    "${KERNELS_ROOT}/quantization/dequant_gather/dequant_gather.cpp"
    "${KERNELS_ROOT}/vision/edge_mlp/edge_mlp.cpp"
    "${KERNELS_ROOT}/vision/patch_merge/patch_merge.cpp")

set(KERNEL_METAL_SOURCES
    "${KERNELS_ROOT}/utils/add_rt/add_rt.metal"
    "${KERNELS_ROOT}/attention/attn_fwd/attn_fwd.metal"
    "${KERNELS_ROOT}/attention/attn_fwd_sg/attn_fwd_sg.metal"
    "${KERNELS_ROOT}/matmul/matmul_custom/matmul_custom.metal"
    "${KERNELS_ROOT}/norms/layernorm/layernorm.metal"
    "${KERNELS_ROOT}/norms/rms_norm/rms_norm.metal"
    "${KERNELS_ROOT}/norms/rms_norm_residual_next/rms_norm_residual_next.metal"
    "${KERNELS_ROOT}/serving/mean_pool_rms_l2/mean_pool_rms_l2.metal"
    "${KERNELS_ROOT}/norms/add_norm/add_norm.metal"
    "${KERNELS_ROOT}/activations/softmax/softmax.metal"
    "${KERNELS_ROOT}/attention/rotary/rotary.metal"
    "${KERNELS_ROOT}/attention/rope_kv/rope_kv.metal"
    "${KERNELS_ROOT}/norms/qk_norm_rope/qk_norm_rope.metal"
    "${KERNELS_ROOT}/norms/qk_norm_rope/qk_norm_rope_kv_f16.metal"
    "${KERNELS_ROOT}/ssm/selective_scan/selective_scan.metal"
    "${KERNELS_ROOT}/linear_attention/gdn/gdn.metal"
    "${KERNELS_ROOT}/quantization/act_quant/act_quant.metal"
    "${KERNELS_ROOT}/quantization/fake_quant/fake_quant.metal"
    "${KERNELS_ROOT}/quantization/fake_quant_fp8/fake_quant_fp8.metal"
    "${KERNELS_ROOT}/quantization/weight_quant_ternary/weight_quant_ternary.metal"
    "${KERNELS_ROOT}/quantization/quantize_tq2_0/quantize_tq2_0.metal"
    "${KERNELS_ROOT}/quantization/ternary_stats/ternary_stats.metal"
    "${KERNELS_ROOT}/serving/minference/minference.metal"
    "${KERNELS_ROOT}/quantization/turboquant/turboquant.metal"
    "${KERNELS_ROOT}/utils/marginal/marginal.metal"
    "${KERNELS_ROOT}/serving/indexer/indexer.metal"
    "${KERNELS_ROOT}/attention/mla/mla.metal"
    "${KERNELS_ROOT}/activations/gelu/gelu.metal"
    "${KERNELS_ROOT}/utils/dropout/dropout.metal"
    "${KERNELS_ROOT}/optimizers/optim/adamw.metal"
    "${KERNELS_ROOT}/serving/embedding/embedding.metal"
    "${KERNELS_ROOT}/activations/glu/glu.metal"
    "${KERNELS_ROOT}/utils/hadamard/hadamard.metal"
    "${KERNELS_ROOT}/serving/kv_cache/kv_cache.metal"
    "${KERNELS_ROOT}/attention/paged_attn_v2/paged_attn_v2.metal"
    "${KERNELS_ROOT}/quantization/quant_rt/quant_rt.metal"
    "${KERNELS_ROOT}/sampling/sampling/sampling.metal"
    "${KERNELS_ROOT}/sampling/sampling/sampling_transforms.metal"
    "${KERNELS_ROOT}/moe/moe/moe.metal"
    "${KERNELS_ROOT}/attention/attn_causal/attn_causal.metal"
    "${KERNELS_ROOT}/attention/attn_varlen/attn_varlen.metal"
    "${KERNELS_ROOT}/quantization/lm_head/lm_head.metal"
    "${KERNELS_ROOT}/utils/cross_entropy/cross_entropy.metal"
    "${KERNELS_ROOT}/utils/kd_kl_topk/kd_kl_topk.metal"
    "${KERNELS_ROOT}/utils/kd_kl_dense/kd_kl_dense.metal"
    "${KERNELS_ROOT}/matmul/flux/flux.metal"
    "${KERNELS_ROOT}/matmul/gemm_staged/gemm_staged.metal"
    "${KERNELS_ROOT}/matmul/gemm_v3/gemm_v3.metal"
    "${KERNELS_ROOT}/attention/attn_multiwarp/attn_multiwarp.metal"
    "${KERNELS_ROOT}/linear_attention/linear_attn/linear_attn.metal"
    "${KERNELS_ROOT}/linear_attention/hedgehog/hedgehog.metal"
    "${KERNELS_ROOT}/linear_attention/lin_attn_causal/lin_attn_causal.metal"
    "${KERNELS_ROOT}/ssm/mamba2/mamba2.metal"
    "${KERNELS_ROOT}/linear_attention/lin_attn_decay/lin_attn_decay.metal"
    "${KERNELS_ROOT}/linear_attention/based/based.metal"
    "${KERNELS_ROOT}/attention/attn_bwd/attn_bwd.metal"
    "${KERNELS_ROOT}/matmul/cmplx_matmul/cmplx_matmul.metal"
    "${KERNELS_ROOT}/ssm/fftconv/fftconv.metal"
    "${KERNELS_ROOT}/quantization/qgemm/qgemm.metal"
    "${KERNELS_ROOT}/quantization/qgemm_bwd/qgemm_bwd.metal"
    "${KERNELS_ROOT}/quantization/qgemm_fused/qgemm_fused.metal"
    "${KERNELS_ROOT}/quantization/qgemv/qgemv.metal"
    "${KERNELS_ROOT}/quantization/qgemv_fused/qgemv_fused.metal"
    "${KERNELS_ROOT}/quantization/qflux/qflux.metal"
    "${KERNELS_ROOT}/quantization/qgemv_int/qgemv_int.metal"
    "${KERNELS_ROOT}/attention/attn_q/attn_q.metal"
    "${KERNELS_ROOT}/attention/attn_decode/attn_decode.metal"
    "${KERNELS_ROOT}/quantization/qgemm_int/qgemm_int.metal"
    "${KERNELS_ROOT}/attention/swin_attn/swin_attn.metal"
    "${KERNELS_ROOT}/matmul/decode_linear/decode_linear.metal"
    "${KERNELS_ROOT}/quantization/dequant_gather/dequant_gather.metal"
    "${KERNELS_ROOT}/vision/edge_mlp/edge_mlp.metal"
    "${KERNELS_ROOT}/vision/patch_merge/patch_merge.metal")

# mlx_build_metallib only tracks explicitly listed dependencies. Every kernel
# includes tk.metal, so header-only kernel changes must also invalidate the
# generated library; otherwise incremental builds can silently benchmark stale
# Metal code.
file(GLOB_RECURSE METAL_HEADER_DEPS CONFIGURE_DEPENDS
    "${METAL_INCLUDE_ROOT}/*.metal")

find_package(MLX CONFIG REQUIRED)
find_package(
  Python 3.8
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

add_library(mlx_ext)
target_sources(mlx_ext PUBLIC ${KERNEL_CPP_SOURCES})
target_include_directories(
  mlx_ext
  PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}"
    ${KERNEL_FAMILY_INCLUDE_DIRS})
target_link_libraries(mlx_ext PUBLIC mlx)

if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET
    mlx_ext_metallib
    TITLE
    mlx_ext
    SOURCES
    ${KERNEL_METAL_SOURCES}
    INCLUDE_DIRS
    "${METAL_INCLUDE_ROOT}"
    "${KERNELS_ROOT}/common"
    ${MLX_INCLUDE_DIRS}
    DEPS
    ${METAL_HEADER_DEPS}
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(mlx_ext mlx_ext_metallib)
endif()

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  "${CMAKE_CURRENT_LIST_DIR}/bindings.cpp")
target_link_libraries(_ext PRIVATE mlx_ext)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
