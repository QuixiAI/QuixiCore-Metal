#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Quantized fused GEMM+GELU (Phase 6 retrofit: a real fused kernel taking quantized weights):
//   D = gelu(dequantize(Wq) @ X + bias)
// = qgemm's dequant-to-shared -> simdgroup MMA, plus flux's per-column-bias + GELU epilogue.
// W (N,K) quantized blocks (format FMT); X (K,M), bias (M,), D (N,M) all half. Demonstrates the
// dequant primitive dropping into an existing fused kernel — any weight-GEMM kernel can do this.
template<typename FMT, int N_WARPS, int BM_PER_WARP>
kernel void qflux_gelu(
    device   half*  D    [[buffer(0)]],
    device   uchar* Wq   [[buffer(1)]],
    device   half*  X    [[buffer(2)]],
    device   half*  bias [[buffer(3)]],   // (M,) per-output-column
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &M [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  warp [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    using G = group<N_WARPS>;
    constexpr const int BN = 32;
    constexpr const int BK = 32;

    using gl_h = gl<half, 1, 1, -1, -1>;
    using gl_vec = gl<half, 1, 1, 1, -1>;
    gl_h gl_x(X, nullptr, nullptr, K, M);
    gl_h gl_d(D, nullptr, nullptr, N, M);
    gl_vec gl_bias(bias, nullptr, nullptr, nullptr, M);

    threadgroup st<half, BN, BK> sW;
    rt<half, BN, BK> w_reg;
    rt<half, BK, BM_PER_WARP> x_reg;
    rt<float, BN, BM_PER_WARP> d_reg;
    zero(d_reg);

    const int by = tgid.y;
    const int bx = tgid.x;
    const int col_block = bx * N_WARPS + (int)warp;

    for (int kb = 0; kb < K / BK; kb++) {
        dequant_into_shared<FMT, BN, BK>(sW, Wq, N, K, by, kb, G::GROUP_THREADS, tid);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        load(w_reg, sW, lane);
        load(x_reg, gl_x, {0, 0, kb, col_block}, lane);
        mma_AB(d_reg, w_reg, x_reg, d_reg);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    // epilogue: + bias (per output column), then GELU
    typename rt<float, BN, BM_PER_WARP>::row_vec bias_vec;
    load(bias_vec, gl_bias, {0, 0, 0, col_block}, lane);
    add_col(d_reg, d_reg, bias_vec);
    gelu(d_reg, d_reg);
    store(gl_d, d_reg, {0, 0, by, col_block}, lane);
}

#define instantiate_qflux(name, FMT)                                          \
   template [[host_name(name)]] [[kernel]]                                    \
   void qflux_gelu<FMT, 2, 16>(                                               \
     device half* D [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     device half* bias [[buffer(3)]],                                         \
     const constant int &N [[buffer(4)]], const constant int &K [[buffer(5)]], \
     const constant int &M [[buffer(6)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint tid [[thread_index_in_threadgroup]],                               \
     uint warp [[simdgroup_index_in_threadgroup]],                           \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qflux("qflux_gelu_q8_0", q8_0);
instantiate_qflux("qflux_gelu_q4_0", q4_0);
instantiate_qflux("qflux_gelu_q4_K", q4_K);
instantiate_qflux("qflux_gelu_kU4B8", kU4B8);
instantiate_qflux("qflux_gelu_kU4", kU4);
instantiate_qflux("qflux_gelu_fp8_e4m3", fp8_e4m3);
instantiate_qflux("qflux_gelu_fp4_e2m1", fp4_e2m1);
instantiate_qflux("qflux_gelu_mxfp8", mxfp8);
instantiate_qflux("qflux_gelu_nvfp4", nvfp4);
instantiate_qflux("qflux_gelu_mxfp4", mxfp4);

}
