cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
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

# ----------------------------- Extensions -----------------------------

# Add library
add_library(mlx_ext)

# Add sources
target_sources(mlx_ext
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/add_rt/add_rt.cpp
    ${CMAKE_CURRENT_LIST_DIR}/attn_fwd/attn_fwd.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matmul_custom/matmul_custom.cpp
    ${CMAKE_CURRENT_LIST_DIR}/layernorm/layernorm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rms_norm/rms_norm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/add_norm/add_norm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/softmax/softmax.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rotary/rotary.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rope_kv/rope_kv.cpp
    ${CMAKE_CURRENT_LIST_DIR}/qk_norm_rope/qk_norm_rope.cpp
    ${CMAKE_CURRENT_LIST_DIR}/selective_scan/selective_scan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mla/mla.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gelu/gelu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dropout/dropout.cpp
    ${CMAKE_CURRENT_LIST_DIR}/optim/adamw.cpp
    ${CMAKE_CURRENT_LIST_DIR}/embedding/embedding.cpp
    ${CMAKE_CURRENT_LIST_DIR}/glu/glu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hadamard/hadamard.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kv_cache/kv_cache.cpp
    ${CMAKE_CURRENT_LIST_DIR}/paged_attn_v2/paged_attn_v2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/quant_rt/quant_rt.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sampling/sampling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/moe/moe.cpp
    ${CMAKE_CURRENT_LIST_DIR}/attn_causal/attn_causal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/attn_varlen/attn_varlen.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lm_head/lm_head.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cross_entropy/cross_entropy.cpp
    ${CMAKE_CURRENT_LIST_DIR}/flux/flux.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gemm_staged/gemm_staged.cpp
    ${CMAKE_CURRENT_LIST_DIR}/attn_multiwarp/attn_multiwarp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/linear_attn/linear_attn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/hedgehog/hedgehog.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lin_attn_causal/lin_attn_causal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mamba2/mamba2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lin_attn_decay/lin_attn_decay.cpp
    ${CMAKE_CURRENT_LIST_DIR}/based/based.cpp
    ${CMAKE_CURRENT_LIST_DIR}/attn_bwd/attn_bwd.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cmplx_matmul/cmplx_matmul.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fftconv/fftconv.cpp
    ${CMAKE_CURRENT_LIST_DIR}/qgemm/qgemm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/qgemv/qgemv.cpp
    ${CMAKE_CURRENT_LIST_DIR}/qflux/qflux.cpp
    ${CMAKE_CURRENT_LIST_DIR}/qgemv_int/qgemv_int.cpp
    ${CMAKE_CURRENT_LIST_DIR}/attn_q/attn_q.cpp
    ${CMAKE_CURRENT_LIST_DIR}/qgemm_int/qgemm_int.cpp
)

# Add include headers
target_include_directories(mlx_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Link to mlx
target_link_libraries(mlx_ext PUBLIC mlx)

# ----------------------------- Metal -----------------------------

set(HEADER_ONLY_LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../include/)

# Build metallib
if(MLX_BUILD_METAL)
    # Build metallib for add_custom.metal
    mlx_build_metallib(
        TARGET
        mlx_ext_metallib
        TITLE
        mlx_ext
        SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/add_rt/add_rt.metal
        ${CMAKE_CURRENT_LIST_DIR}/attn_fwd/attn_fwd.metal
        ${CMAKE_CURRENT_LIST_DIR}/matmul_custom/matmul_custom.metal
        ${CMAKE_CURRENT_LIST_DIR}/layernorm/layernorm.metal
        ${CMAKE_CURRENT_LIST_DIR}/rms_norm/rms_norm.metal
        ${CMAKE_CURRENT_LIST_DIR}/add_norm/add_norm.metal
        ${CMAKE_CURRENT_LIST_DIR}/softmax/softmax.metal
        ${CMAKE_CURRENT_LIST_DIR}/rotary/rotary.metal
        ${CMAKE_CURRENT_LIST_DIR}/rope_kv/rope_kv.metal
        ${CMAKE_CURRENT_LIST_DIR}/qk_norm_rope/qk_norm_rope.metal
        ${CMAKE_CURRENT_LIST_DIR}/selective_scan/selective_scan.metal
        ${CMAKE_CURRENT_LIST_DIR}/mla/mla.metal
        ${CMAKE_CURRENT_LIST_DIR}/gelu/gelu.metal
        ${CMAKE_CURRENT_LIST_DIR}/dropout/dropout.metal
        ${CMAKE_CURRENT_LIST_DIR}/optim/adamw.metal
        ${CMAKE_CURRENT_LIST_DIR}/embedding/embedding.metal
        ${CMAKE_CURRENT_LIST_DIR}/glu/glu.metal
        ${CMAKE_CURRENT_LIST_DIR}/hadamard/hadamard.metal
        ${CMAKE_CURRENT_LIST_DIR}/kv_cache/kv_cache.metal
        ${CMAKE_CURRENT_LIST_DIR}/paged_attn_v2/paged_attn_v2.metal
        ${CMAKE_CURRENT_LIST_DIR}/quant_rt/quant_rt.metal
        ${CMAKE_CURRENT_LIST_DIR}/sampling/sampling.metal
        ${CMAKE_CURRENT_LIST_DIR}/moe/moe.metal
        ${CMAKE_CURRENT_LIST_DIR}/attn_causal/attn_causal.metal
        ${CMAKE_CURRENT_LIST_DIR}/attn_varlen/attn_varlen.metal
        ${CMAKE_CURRENT_LIST_DIR}/lm_head/lm_head.metal
        ${CMAKE_CURRENT_LIST_DIR}/cross_entropy/cross_entropy.metal
        ${CMAKE_CURRENT_LIST_DIR}/flux/flux.metal
        ${CMAKE_CURRENT_LIST_DIR}/gemm_staged/gemm_staged.metal
        ${CMAKE_CURRENT_LIST_DIR}/attn_multiwarp/attn_multiwarp.metal
        ${CMAKE_CURRENT_LIST_DIR}/linear_attn/linear_attn.metal
        ${CMAKE_CURRENT_LIST_DIR}/hedgehog/hedgehog.metal
        ${CMAKE_CURRENT_LIST_DIR}/lin_attn_causal/lin_attn_causal.metal
        ${CMAKE_CURRENT_LIST_DIR}/mamba2/mamba2.metal
        ${CMAKE_CURRENT_LIST_DIR}/lin_attn_decay/lin_attn_decay.metal
        ${CMAKE_CURRENT_LIST_DIR}/based/based.metal
        ${CMAKE_CURRENT_LIST_DIR}/attn_bwd/attn_bwd.metal
        ${CMAKE_CURRENT_LIST_DIR}/cmplx_matmul/cmplx_matmul.metal
        ${CMAKE_CURRENT_LIST_DIR}/fftconv/fftconv.metal
        ${CMAKE_CURRENT_LIST_DIR}/qgemm/qgemm.metal
        ${CMAKE_CURRENT_LIST_DIR}/qgemv/qgemv.metal
        ${CMAKE_CURRENT_LIST_DIR}/qflux/qflux.metal
        ${CMAKE_CURRENT_LIST_DIR}/qgemv_int/qgemv_int.metal
        ${CMAKE_CURRENT_LIST_DIR}/attn_q/attn_q.metal
        ${CMAKE_CURRENT_LIST_DIR}/qgemm_int/qgemm_int.metal
        INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}
        ${MLX_INCLUDE_DIRS}
        ${HEADER_ONLY_LIB_DIR}
        OUTPUT_DIRECTORY
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    
# Add dependencies to mlx_ext
add_dependencies(mlx_ext mlx_ext_metallib)

endif()

# ----------------------------- Python Bindings -----------------------------
nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE mlx_ext)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
