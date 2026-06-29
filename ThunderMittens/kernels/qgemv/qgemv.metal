#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Quantized GEMV (batch-1 decode):  d = dequantize(W) @ x,  W (N,K) quantized blocks, x (K,1).
// No MMA — one simdgroup (32 lanes) per output row: each lane dequantizes + dots a strided slice
// of the row, then simd_sum reduces. This is the memory-bound decode path where shrinking the
// weight bytes (4-8x) is the real Apple win. Mirrors llama.cpp's mul_vec_q_n.
template<typename FMT>
kernel void qgemv(
    device   half*  D  [[buffer(0)]],   // (N, 1) output
    device   uchar* Wq [[buffer(1)]],   // (N, K/block_k) packed weight blocks
    device   half*  X  [[buffer(2)]],   // (K, 1) activation vector
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;                          // one threadgroup (simdgroup) per output row
    const int bpr = K / FMT::block_k;
    device const uchar* row_base = Wq + (uint)(row * bpr) * FMT::block_bytes;

    float acc = 0.0f;
    for (int k = (int)lane; k < K; k += 32) {
        const int kb  = k / FMT::block_k;
        const int col = k % FMT::block_k;
        const half w = FMT::dequant(row_base + (uint)kb * FMT::block_bytes, col);
        acc += float(w) * float(X[k]);
    }
    acc = metal::simd_sum(acc);                      // reduce the dot across the 32 lanes
    if (lane == 0) D[row] = half(acc);
}

#define instantiate_qgemv(name, FMT)                                          \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemv<FMT>(                                                           \
     device half* D [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemv("qgemv_q8_0", q8_0);

}
