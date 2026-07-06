#include "../../../testing_commons/testing_flags.hpp"
#ifdef TEST_WARP_REGISTER_VEC_MAPS
#include "tk.metal"

namespace mittens {

#define gts_kern_PARAMS(T) \
    device T* _input [[buffer(0)]], \
    device T* _output [[buffer(1)]], \
    uint3         id              [[thread_position_in_grid]], \
    uint3         thread_group_id [[threadgroup_position_in_grid]], \
    uint          simd_group_id   [[simdgroup_index_in_threadgroup]], \
    uint          simd_lane_id    [[thread_index_in_simdgroup]]

template<typename T, unsigned W, typename L>
kernel void rv_unary(gts_kern_PARAMS(T))
{
    using reg_vec1 = rv<T, W * 8, L>;
    using reg_vec2 = rv<T, W * 8, L>;
    using GL = gl<T, 1, 1, 1, W * 8>;
    reg_vec1 vec_a;
    reg_vec2 vec_b;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(vec_a, input, {0,0,0,0}, simd_lane_id);
    exp(vec_b, vec_a);
    store(output, vec_b, {0,0,0,0}, simd_lane_id);
}

template<typename T, unsigned W, typename L>
kernel void rv_rv_binary(gts_kern_PARAMS(T))
{
    using reg_vec1 = rv<T, W * 8, L>;
    using reg_vec2 = rv<T, W * 8, L>;
    using GL = gl<T, 1, 1, 1, W * 8>;
    reg_vec1 vec_a;
    reg_vec2 vec_b;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(vec_a, input, {0,0,0,0}, simd_lane_id);
    load(vec_b, input, {0,0,0,0}, simd_lane_id);
    add(vec_b, vec_b, vec_a);
    store(output, vec_b, {0,0,0,0}, simd_lane_id);
}
    
template<typename T, unsigned W, typename L>
kernel void rv_t_binary(gts_kern_PARAMS(T))
{
    using reg_vec1 = rv<T, W * 8, L>;
    using reg_vec2 = rv<T, W * 8, L>;
    using GL = gl<T, 1, 1, 1, W * 8>;
    reg_vec1 vec_a;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(vec_a, input, {0,0,0,0}, simd_lane_id);
    T val = base_types::convertor<T, float>::convert(0.5f);
    add(vec_a, vec_a, val);

    store(output, vec_a, {0,0,0,0}, simd_lane_id);
}

// Exercises the rsqrt map op. Inputs are made positive (abs(x)+0.5) to stay in the
// sqrt domain; the host reference mirrors this.
template<typename T, unsigned W, typename L>
kernel void rv_rsqrt(gts_kern_PARAMS(T))
{
    using reg_vec1 = rv<T, W * 8, L>;
    using reg_vec2 = rv<T, W * 8, L>;
    using GL = gl<T, 1, 1, 1, W * 8>;
    reg_vec1 vec_a;
    reg_vec2 vec_b;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(vec_a, input, {0,0,0,0}, simd_lane_id);
    abs(vec_a, vec_a);
    T bias = base_types::convertor<T, float>::convert(0.5f);
    add(vec_a, vec_a, bias);
    rsqrt(vec_b, vec_a);
    store(output, vec_b, {0,0,0,0}, simd_lane_id);
}

// Exercises the tanh map op (domain is all reals).
template<typename T, unsigned W, typename L>
kernel void rv_tanh(gts_kern_PARAMS(T))
{
    using reg_vec1 = rv<T, W * 8, L>;
    using reg_vec2 = rv<T, W * 8, L>;
    using GL = gl<T, 1, 1, 1, W * 8>;
    reg_vec1 vec_a;
    reg_vec2 vec_b;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(vec_a, input, {0,0,0,0}, simd_lane_id);
    tanh(vec_b, vec_a);
    store(output, vec_b, {0,0,0,0}, simd_lane_id);
}


#define gen_kernels(reg_N) \
    template [[host_name("rv_exp_float_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_unary<float, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(float)); \
    template [[host_name("rv_exp_float_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_unary<float, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(float)); \
    template [[host_name("rv_exp_float_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_unary<float, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(float)); \
\
    template [[host_name("rv_exp_half_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_unary<half, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(half));  \
    template [[host_name("rv_exp_half_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_unary<half, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(half));  \
    template [[host_name("rv_exp_half_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_unary<half, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(half));  \
\
    template [[host_name("rv_exp_bf16_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_unary<bf16, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(bf16));  \
    template [[host_name("rv_exp_bf16_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_unary<bf16, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(bf16));  \
    template [[host_name("rv_exp_bf16_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_unary<bf16, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(bf16));  \
\
    template [[host_name("rv_rv_add_float_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_rv_binary<float, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(float)); \
    template [[host_name("rv_rv_add_float_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_rv_binary<float, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(float)); \
    template [[host_name("rv_rv_add_float_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_rv_binary<float, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(float)); \
\
    template [[host_name("rv_rv_add_half_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_rv_binary<half, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(half));  \
    template [[host_name("rv_rv_add_half_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_rv_binary<half, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(half));  \
    template [[host_name("rv_rv_add_half_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_rv_binary<half, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(half));  \
\
    template [[host_name("rv_rv_add_bf16_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_rv_binary<bf16, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(bf16));  \
    template [[host_name("rv_rv_add_bf16_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_rv_binary<bf16, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(bf16));  \
    template [[host_name("rv_rv_add_bf16_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_rv_binary<bf16, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(bf16));  \
\
    template [[host_name("rv_t_add_float_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_t_binary<float, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(float)); \
    template [[host_name("rv_t_add_float_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_t_binary<float, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(float)); \
    template [[host_name("rv_t_add_float_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_t_binary<float, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(float)); \
\
    template [[host_name("rv_t_add_half_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_t_binary<half, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(half));  \
    template [[host_name("rv_t_add_half_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_t_binary<half, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(half));  \
    template [[host_name("rv_t_add_half_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_t_binary<half, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(half));  \
\
    template [[host_name("rv_t_add_bf16_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_t_binary<bf16, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(bf16));  \
    template [[host_name("rv_t_add_bf16_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_t_binary<bf16, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(bf16));  \
    template [[host_name("rv_t_add_bf16_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_t_binary<bf16, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(bf16));  \
\
    template [[host_name("rv_rsqrt_float_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_rsqrt<float, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(float)); \
    template [[host_name("rv_rsqrt_float_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_rsqrt<float, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(float)); \
    template [[host_name("rv_rsqrt_float_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_rsqrt<float, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(float)); \
\
    template [[host_name("rv_rsqrt_half_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_rsqrt<half, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(half));  \
    template [[host_name("rv_rsqrt_half_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_rsqrt<half, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(half));  \
    template [[host_name("rv_rsqrt_half_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_rsqrt<half, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(half));  \
\
    template [[host_name("rv_rsqrt_bf16_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_rsqrt<bf16, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(bf16));  \
    template [[host_name("rv_rsqrt_bf16_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_rsqrt<bf16, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(bf16));  \
    template [[host_name("rv_rsqrt_bf16_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_rsqrt<bf16, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(bf16));  \
\
    template [[host_name("rv_tanh_float_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_tanh<float, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(float)); \
    template [[host_name("rv_tanh_float_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_tanh<float, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(float)); \
    template [[host_name("rv_tanh_float_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_tanh<float, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(float)); \
\
    template [[host_name("rv_tanh_half_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_tanh<half, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(half));  \
    template [[host_name("rv_tanh_half_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_tanh<half, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(half));  \
    template [[host_name("rv_tanh_half_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_tanh<half, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(half));  \
\
    template [[host_name("rv_tanh_bf16_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_tanh<bf16, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(bf16));  \
    template [[host_name("rv_tanh_bf16_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_tanh<bf16, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(bf16));  \
    template [[host_name("rv_tanh_bf16_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_tanh<bf16, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(bf16));  \


    
#if (INTENSITY_1)
    gen_kernels(1) gen_kernels(2)
#elif (INTENSITY_2)
    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4)
#elif (INTENSITY_3) // 8
    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
#elif (INTENSITY_4) // 16
    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
    gen_kernels(9) gen_kernels(10) gen_kernels(11) gen_kernels(12) gen_kernels(13) gen_kernels(14) gen_kernels(15) gen_kernels(16)
#endif

#undef gen_kernels

}

#endif
