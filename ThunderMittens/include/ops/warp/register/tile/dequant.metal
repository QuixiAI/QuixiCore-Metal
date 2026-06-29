/**
 * @file
 * @brief In-register dequantization of quantized weight blocks into half tiles.
 *
 * "Marlin's method" on Apple: quantized weights are stored block-wise (a small fp16 scale +
 * packed low-precision codes), dequantized to `half` here, then fed to a standard
 * `simdgroup_matrix` MMA. The dequant math is pure IEEE-fp16 bit arithmetic, valid on Metal
 * `half` verbatim. Block layouts mirror llama.cpp's GGUF formats (ggml-common.h); the dequant
 * constants follow llama.cpp (ggml-metal.metal) and Marlin (dequant.h).
 *
 * A "format" is a small struct exposing `block_k` (weights per block), `block_bytes`, and a
 * `dequant(device const uchar* base, int col) -> half` for the weight at column `col` of the
 * block starting at byte `base`. `dequant_into_shared<FMT>` cooperatively dequantizes a tile.
 */
#pragma once
#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {

// ---- q8_0 : { half d; int8 qs[32]; }  — 34 bytes, 32 weights/block, value = d * q ----
struct q8_0 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 34;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];      // fp16 scale at offset 0
        const char q = ((device const char*)(base + 2))[col];  // signed int8 codes at offset 2
        return d * half(q);
    }
};

// Cooperatively dequantize the (by, kb) block of a (N, K) packed weight into a shared half tile.
//   Packed layout: block(n, kb) starts at byte (n*(K/FMT::block_k) + kb) * FMT::block_bytes.
//   Requires BK == FMT::block_k. `group_threads` = total threads in the threadgroup; `threadIdx`
//   = flat thread index in the threadgroup.
template<typename FMT, int BN, int BK>
METAL_FUNC void dequant_into_shared(threadgroup st<half, BN, BK>& dst,
                                    device const uchar* Wq, int N, int K,
                                    int by, int kb, int group_threads, uint threadIdx) {
    const int blocks_per_row = K / FMT::block_k;
    for (int e = (int)threadIdx; e < BN * BK; e += group_threads) {
        const int row  = e / BK;
        const int col  = e % BK;
        const int grow = by * BN + row;
        device const uchar* base = Wq + (uint)(grow * blocks_per_row + kb) * FMT::block_bytes;
        dst[int2(row, col)] = FMT::dequant(base, col);
    }
}

} // namespace mittens
