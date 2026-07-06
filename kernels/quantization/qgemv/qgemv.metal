#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Quantized GEMV (batch-1 decode):  d = dequantize(W) @ x,  W (N,K) quantized blocks, x (K,1).
// No MMA — one simdgroup (32 lanes) per output row, walked block-major: each lane owns an
// 8-col contiguous span inside a block (block_k/8 lanes cover a block; the simdgroup covers
// 32/(block_k/8) blocks per iteration). The span keeps the block-scale reads CSE-able, kills
// the per-element div/mod of the old strided walk, and lets X load as half4. This is the
// memory-bound decode path where shrinking the weight bytes (4-8x) is the real Apple win.
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

    constexpr int CPL = 8;                           // contiguous cols per lane
    constexpr int LPB = FMT::block_k / CPL;          // lanes per block (2..32)
    constexpr int BPI = 32 / LPB;                    // blocks per simdgroup iteration (1..16)
    const int b_off = (int)lane / LPB;
    const int col0  = ((int)lane % LPB) * CPL;

    float acc = 0.0f;
    for (int kb = b_off; kb < bpr; kb += BPI) {
        device const uchar* base = row_base + (uint)kb * FMT::block_bytes;
        device const half4* xv = (device const half4*)(X + kb * FMT::block_k + col0);
        const half4 x0 = xv[0], x1 = xv[1];
        half w[8];
        tk_dequant8<FMT>(base, col0, w);
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) acc += float(w[i]) * float(x0[i]);
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) acc += float(w[4 + i]) * float(x1[i]);
    }
    acc = metal::simd_sum(acc);                      // reduce the dot across the 32 lanes
    if (lane == 0) D[row] = half(acc);
}

[[host_name("qgemv_q8_0")]]
kernel void qgemv_q8_0_fast(
    device   half*  D  [[buffer(0)]],
    device   uchar* Wq [[buffer(1)]],
    device   half*  X  [[buffer(2)]],
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;
    const int bpr = K / q8_0::block_k;
    device const uchar* row_base = Wq + (uint)(row * bpr) * q8_0::block_bytes;

    const int block_offset = (int)(lane >> 2);       // 8 q8_0 blocks per simdgroup iteration
    const int chunk = (int)(lane & 3);               // 8 contiguous int8 values within the block

    float acc = 0.0f;
    for (int kb = block_offset; kb < bpr; kb += 8) {
        device const uchar* block = row_base + (uint)kb * q8_0::block_bytes;
        const float d = float(((device const half*)block)[0]);
        device const char* qs = (device const char*)(block + 2 + chunk * 8);
        const int x0 = (kb << 5) + chunk * 8;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) {
            acc += d * float(qs[i]) * float(X[x0 + i]);
        }
    }
    acc = metal::simd_sum(acc);
    if (lane == 0) D[row] = half(acc);
}

[[host_name("qgemv_q4_0")]]
kernel void qgemv_q4_0_fast(
    device   half*  D  [[buffer(0)]],
    device   uchar* Wq [[buffer(1)]],
    device   half*  X  [[buffer(2)]],
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;
    const int bpr = K / q4_0::block_k;
    device const uchar* row_base = Wq + (uint)(row * bpr) * q4_0::block_bytes;

    const int block_offset = (int)(lane >> 1);       // 16 q4_0 blocks per simdgroup iteration
    const int byte_start = (int)(lane & 1) * 8;      // each lane handles 8 packed bytes = 16 weights

    float acc = 0.0f;
    for (int kb = block_offset; kb < bpr; kb += 16) {
        device const uchar* block = row_base + (uint)kb * q4_0::block_bytes;
        const float d = float(((device const half*)block)[0]);
        device const uchar* qs = block + 2 + byte_start;
        const int x0 = (kb << 5) + byte_start;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) {
            const uchar packed = qs[i];
            acc += d * float((int)(packed & 0x0F) - 8) * float(X[x0 + i]);
            acc += d * float((int)(packed >> 4) - 8) * float(X[x0 + i + 16]);
        }
    }
    acc = metal::simd_sum(acc);
    if (lane == 0) D[row] = half(acc);
}

#define instantiate_qgemv(name, FMT)                                          \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemv<FMT>(                                                           \
     device half* D [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint lane [[thread_index_in_simdgroup]]);

// Full-weight dequant to fp16: packed (N, K/bk, bytes) -> W (N, K) half. Flat, one thread per
// 8-col span (tk_dequant8 + two half4 stores). Backs the k-quant PREFILL route: the 256-superblock
// formats' fragment-path in-GEMM dequant measured 2-2.3x slower than dequantize-then-mx.matmul,
// so qgemm routes them here for M >= 64.
template<typename FMT>
kernel void qdequant_fp16(
    device half*  W  [[buffer(0)]],   // (N, K) output
    device uchar* Wq [[buffer(1)]],   // (N, K/block_k, block_bytes)
    const constant int &N [[buffer(2)]],
    const constant int &K [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    const int spans_per_row = K / 8;
    const uint total = (uint)N * (uint)spans_per_row;
    if (tid >= total) return;
    const int row  = (int)(tid / spans_per_row);
    const int col0 = (int)(tid % spans_per_row) * 8;
    const int bpr = K / FMT::block_k;
    const int blk = col0 / FMT::block_k;
    const int cib = col0 % FMT::block_k;
    device const uchar* base = Wq + ((uint)row * bpr + blk) * FMT::block_bytes;
    half w[8];
    tk_dequant8<FMT>(base, cib, w);
    device half4* dst = (device half4*)(W + (long)row * K + col0);
    dst[0] = half4(w[0], w[1], w[2], w[3]);
    dst[1] = half4(w[4], w[5], w[6], w[7]);
}

#define instantiate_qdequant(name, FMT)                                       \
   template [[host_name(name)]] [[kernel]]                                    \
   void qdequant_fp16<FMT>(                                                   \
     device half* W [[buffer(0)]], device uchar* Wq [[buffer(1)]],            \
     const constant int &N [[buffer(2)]], const constant int &K [[buffer(3)]],\
     uint tid [[thread_position_in_grid]]);

instantiate_qdequant("qdequant_q4_K", q4_K);
instantiate_qdequant("qdequant_q5_K", q5_K);
instantiate_qdequant("qdequant_q6_K", q6_K);
instantiate_qdequant("qdequant_q2_K", q2_K);
instantiate_qdequant("qdequant_q3_K", q3_K);
instantiate_qdequant("qdequant_iq4_xs", iq4_xs);
instantiate_qdequant("qdequant_iq2_xxs", iq2_xxs);
instantiate_qdequant("qdequant_iq2_xs", iq2_xs);
instantiate_qdequant("qdequant_iq3_xxs", iq3_xxs);
instantiate_qdequant("qdequant_iq1_s", iq1_s);

instantiate_qgemv("qgemv_q8_0_small", q8_0);
instantiate_qgemv("qgemv_q4_0_small", q4_0);
instantiate_qgemv("qgemv_q4_K", q4_K);
instantiate_qgemv("qgemv_kU4B8", kU4B8);
instantiate_qgemv("qgemv_kU4", kU4);
instantiate_qgemv("qgemv_fp8_e4m3", fp8_e4m3);
instantiate_qgemv("qgemv_fp4_e2m1", fp4_e2m1);
instantiate_qgemv("qgemv_mxfp8", mxfp8);
instantiate_qgemv("qgemv_nvfp4", nvfp4);
instantiate_qgemv("qgemv_mxfp4", mxfp4);
instantiate_qgemv("qgemv_bitnet", bitnet);
instantiate_qgemv("qgemv_iq4_nl", iq4_nl);
instantiate_qgemv("qgemv_iq4_xs", iq4_xs);
instantiate_qgemv("qgemv_iq2_xxs", iq2_xxs);
instantiate_qgemv("qgemv_iq2_xs", iq2_xs);
instantiate_qgemv("qgemv_iq3_xxs", iq3_xxs);
instantiate_qgemv("qgemv_iq1_s", iq1_s);
instantiate_qgemv("qgemv_q4_1", q4_1);
instantiate_qgemv("qgemv_q5_0", q5_0);
instantiate_qgemv("qgemv_q5_1", q5_1);
instantiate_qgemv("qgemv_q2_K", q2_K);
instantiate_qgemv("qgemv_q3_K", q3_K);
instantiate_qgemv("qgemv_q5_K", q5_K);
instantiate_qgemv("qgemv_q6_K", q6_K);
instantiate_qgemv("qgemv_e5m2", e5m2);
instantiate_qgemv("qgemv_fp8_block", fp8_block);
instantiate_qgemv("qgemv_mxfp6_e3m2", mxfp6_e3m2);
instantiate_qgemv("qgemv_mxfp6_e2m3", mxfp6_e2m3);
instantiate_qgemv("qgemv_hqq", hqq);

}
