#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Quantized GEMM (Marlin's method, dequant-to-shared):  D = W @ X
//   W (N,K) is quantized (packed blocks, format FMT); X (K,M) and D (N,M) are half.
// Mirrors gemm_staged, but the weight block is DEQUANTIZED into the shared half tile each
// K-step instead of byte-copied — then the existing shared->register load + simdgroup MMA
// run unchanged (fp32 accumulate). BK = FMT::block_k (= 32). Shapes need N%32, M%BM, K%BK.
template<typename FMT, int N_WARPS, int BM_PER_WARP>
kernel void qgemm(
    device   half*  D  [[buffer(0)]],   // (N, M) output
    device   uchar* Wq [[buffer(1)]],   // (N, K/block_k) packed weight blocks
    device   half*  X  [[buffer(2)]],   // (K, M) activations
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    const constant int &M [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  warp [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    using G = group<N_WARPS>;
    constexpr const int BN = 32;
    constexpr const int BK = 32;             // MMA K-step (decoupled from FMT::block_k)

    using gl_h = gl<half, 1, 1, -1, -1>;
    gl_h gl_x(X, nullptr, nullptr, K, M);
    gl_h gl_d(D, nullptr, nullptr, N, M);

    threadgroup st<half, BN, BK> sW;        // dequantized weight block, shared by all warps
    rt<half, BN, BK> w_reg;
    rt<half, BK, BM_PER_WARP> x_reg;
    rt<float, BN, BM_PER_WARP> d_reg;
    zero(d_reg);

    const int by = tgid.y;                              // output row block (BN weight rows)
    const int bx = tgid.x;                              // output col block (BM cols)
    const int col_block = bx * N_WARPS + (int)warp;     // this warp's BM_PER_WARP column block

    for (int kb = 0; kb < K / BK; kb++) {
        dequant_into_shared<FMT, BN, BK>(sW, Wq, N, K, by, kb, G::GROUP_THREADS, tid);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        load(w_reg, sW, lane);                           // shared -> register (dequantized W)
        load(x_reg, gl_x, {0, 0, kb, col_block}, lane);  // this warp's X columns
        mma_AB(d_reg, w_reg, x_reg, d_reg);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);  // before sW is overwritten
    }
    store(gl_d, d_reg, {0, 0, by, col_block}, lane);
}

#define instantiate_qgemm(name, FMT, NW, BMPW)                                \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemm<FMT, NW, BMPW>(                                                 \
     device half* D [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     const constant int &M [[buffer(5)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint tid [[thread_index_in_threadgroup]],                               \
     uint warp [[simdgroup_index_in_threadgroup]],                           \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemm("qgemm_q8_0", q8_0, 2, 16);
instantiate_qgemm("qgemm_q4_0", q4_0, 2, 16);
instantiate_qgemm("qgemm_q4_K", q4_K, 2, 16);
instantiate_qgemm("qgemm_kU4B8", kU4B8, 2, 16);
instantiate_qgemm("qgemm_kU4", kU4, 2, 16);
instantiate_qgemm("qgemm_fp8_e4m3", fp8_e4m3, 2, 16);
instantiate_qgemm("qgemm_fp4_e2m1", fp4_e2m1, 2, 16);
instantiate_qgemm("qgemm_mxfp8", mxfp8, 2, 16);
instantiate_qgemm("qgemm_nvfp4", nvfp4, 2, 16);
instantiate_qgemm("qgemm_mxfp4", mxfp4, 2, 16);

}
