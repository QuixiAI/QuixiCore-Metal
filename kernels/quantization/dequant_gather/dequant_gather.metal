#include <metal_stdlib>
#include "tk.metal"

using namespace metal;

namespace mittens {

// The gather contract rounds only once, when the fully scaled fp32 value is
// stored as fp16. The general QuixiCore span decoder intentionally returns
// half values for matrix kernels, so keep a gather-specific fp32 decoder here
// rather than introducing an extra half rounding through tk_dequant8.
template <typename FMT>
METAL_FUNC void dequant_gather8_f32(
    device const uchar *base, int col0, thread float *values) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) values[i] = float(FMT::dequant(base, col0 + i));
}

template <>
METAL_FUNC void dequant_gather8_f32<q4_0>(
    device const uchar *base, int col0, thread float *values) {
    #pragma clang fp reassociate(off)
    const float d = float(((device const half *)base)[0]);
    device const uchar *qs = base + 2;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const int col = col0 + i;
        const int nibble = col < 16 ? (qs[col] & 0x0f) : (qs[col - 16] >> 4);
        values[i] = d * float(nibble - 8);
    }
}

template <>
METAL_FUNC void dequant_gather8_f32<q8_0>(
    device const uchar *base, int col0, thread float *values) {
    #pragma clang fp reassociate(off)
    const float d = float(((device const half *)base)[0]);
    device const char *qs = (device const char *)(base + 2);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) values[i] = d * float(qs[col0 + i]);
}

template <>
METAL_FUNC void dequant_gather8_f32<q6_K>(
    device const uchar *base, int col0, thread float *values) {
    #pragma clang fp reassociate(off)
    device const uchar *ql = base;
    device const uchar *qh = base + 128;
    device const char *sub_scales = (device const char *)(base + 192);
    const float d = float(((device const half *)(base + 208))[0]);
    const int chunk = col0 >> 7;
    const int position = col0 & 127;
    const int group = position >> 5;
    const int lane0 = position & 31;
    const float d_scale = d * float(sub_scales[
        chunk * 8 + (lane0 >> 4) + group * 2]);
    device const uchar *q = ql + chunk * 64 + lane0 + 32 * (group & 1);
    device const uchar *h = qh + chunk * 32 + lane0;
    const int high_shift = 2 * group;
    const bool high_nibble = (group & 2) != 0;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const int nibble = high_nibble ? (q[i] >> 4) : (q[i] & 0x0f);
        const int quant = (nibble | (((h[i] >> high_shift) & 3) << 4)) - 32;
        values[i] = d_scale * float(quant);
    }
}

// Gather packed GGUF rows while dequantizing directly into fp16. One thread
// owns an aligned 8-column span and emits two vector stores. Invalid ids
// produce an all-zero row.
template <typename FMT>
kernel void dequant_gather(
    device const uchar *table [[buffer(0)]],
    device const int *ids [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant int &rows [[buffer(3)]],
    constant int &columns [[buffer(4)]],
    constant int &tokens [[buffer(5)]],
    constant float &scale [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    // The standalone MPS metallib uses Metal's fast math mode.  Preserve the
    // source multiplication order so it honors the bit-exact FP32-decode/scale
    // followed by one FP16-rounding contract just like the
    // MLX build, which compiles with -fno-fast-math.
    #pragma clang fp reassociate(off)
    const int spans_per_row = columns / 8;
    const uint total = (uint)tokens * (uint)spans_per_row;
    if (tid >= total) return;
    const int token = int(tid / spans_per_row);
    const int col0 = int(tid % spans_per_row) * 8;
    const int row = ids[token];
    device half4 *dst = (device half4 *)(output + (long)token * columns + col0);
    if (row < 0 || row >= rows) {
        dst[0] = half4(0.0h);
        dst[1] = half4(0.0h);
        return;
    }

    const int blocks_per_row = columns / FMT::block_k;
    const int block = col0 / FMT::block_k;
    const int column_in_block = col0 % FMT::block_k;
    device const uchar *base =
        table + ((long)row * blocks_per_row + block) * FMT::block_bytes;
    float values[8];
    dequant_gather8_f32<FMT>(base, column_in_block, values);
    dst[0] = half4(float4(values[0], values[1], values[2], values[3]) * scale);
    dst[1] = half4(float4(values[4], values[5], values[6], values[7]) * scale);
}

#define instantiate_dequant_gather(name, FMT)                                  \
  template [[host_name("dequant_gather_" name)]] [[kernel]]                  \
  void dequant_gather<FMT>(device const uchar *table [[buffer(0)]],           \
    device const int *ids [[buffer(1)]], device half *output [[buffer(2)]],   \
    constant int &rows [[buffer(3)]], constant int &columns [[buffer(4)]],    \
    constant int &tokens [[buffer(5)]], constant float &scale [[buffer(6)]],  \
    uint tid [[thread_position_in_grid]]);

instantiate_dequant_gather("q4_0", q4_0)
instantiate_dequant_gather("q8_0", q8_0)
instantiate_dequant_gather("q6_K", q6_K)

} // namespace mittens
