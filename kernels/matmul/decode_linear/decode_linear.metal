#include <metal_stdlib>
#include "tk.metal"

using namespace metal;

namespace mittens {

// Latency-oriented linear layer for autoregressive decode. One simdgroup owns
// one (batch, output-channel) dot product. The optional erf-GELU epilogue
// removes a second launch for decoder FFN expansion.
template <typename T>
kernel void decode_linear(
    device const T *input [[buffer(0)]],
    device const T *weight [[buffer(1)]],
    device const T *bias [[buffer(2)]],
    device T *output [[buffer(3)]],
    constant int &batch [[buffer(4)]],
    constant int &in_dim [[buffer(5)]],
    constant int &out_dim [[buffer(6)]],
    constant int &gelu [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int output_channel = int(group.x);
    const int batch_index = int(group.y);
    if (output_channel >= out_dim || batch_index >= batch) return;
    const long input_base = (long)batch_index * in_dim;
    const long weight_base = (long)output_channel * in_dim;
    float accumulator = 0.0f;
    for (int k = int(lane); k < in_dim; k += 32) {
        accumulator += float(input[input_base + k]) * float(weight[weight_base + k]);
    }
    accumulator = metal::simd_sum(accumulator);
    if (lane == 0) {
        accumulator += float(bias[output_channel]);
        if (gelu != 0) accumulator = glu_gelu_erf(accumulator);
        output[(long)batch_index * out_dim + output_channel] = T(accumulator);
    }
}

#define instantiate_decode_linear(type_name, T)                                \
  template [[host_name("decode_linear_" #type_name)]] [[kernel]] void         \
  decode_linear<T>(device const T *input [[buffer(0)]],                        \
                   device const T *weight [[buffer(1)]],                       \
                   device const T *bias [[buffer(2)]],                         \
                   device T *output [[buffer(3)]],                             \
                   constant int &batch [[buffer(4)]],                          \
                   constant int &in_dim [[buffer(5)]],                         \
                   constant int &out_dim [[buffer(6)]],                        \
                   constant int &gelu [[buffer(7)]],                           \
                   uint2 group [[threadgroup_position_in_grid]],               \
                   uint lane [[thread_index_in_simdgroup]]);

instantiate_decode_linear(float32, float)
instantiate_decode_linear(bfloat16, bf16)

template <typename T>
kernel void decode_linear_residual(
    device const T *input [[buffer(0)]],
    device const T *weight [[buffer(1)]],
    device const T *bias [[buffer(2)]],
    device const T *residual [[buffer(3)]],
    device T *output [[buffer(4)]],
    constant int &batch [[buffer(5)]],
    constant int &in_dim [[buffer(6)]],
    constant int &out_dim [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int output_channel = int(group.x);
    const int batch_index = int(group.y);
    if (output_channel >= out_dim || batch_index >= batch) return;
    const long input_base = (long)batch_index * in_dim;
    const long weight_base = (long)output_channel * in_dim;
    const long output_index = (long)batch_index * out_dim + output_channel;
    float accumulator = 0.0f;
    for (int k = int(lane); k < in_dim; k += 32) {
        accumulator += float(input[input_base + k]) * float(weight[weight_base + k]);
    }
    accumulator = metal::simd_sum(accumulator);
    if (lane == 0) {
        // Match a materialized framework linear: round the linear result to T
        // before evaluating the residual addition.
        const T linear = T(accumulator + float(bias[output_channel]));
        output[output_index] = T(float(linear) + float(residual[output_index]));
    }
}

#define instantiate_decode_linear_residual(type_name, T)                       \
  template [[host_name("decode_linear_residual_" #type_name)]] [[kernel]]     \
  void decode_linear_residual<T>(device const T *input [[buffer(0)]],          \
    device const T *weight [[buffer(1)]], device const T *bias [[buffer(2)]],  \
    device const T *residual [[buffer(3)]], device T *output [[buffer(4)]],    \
    constant int &batch [[buffer(5)]], constant int &in_dim [[buffer(6)]],     \
    constant int &out_dim [[buffer(7)]],                                       \
    uint2 group [[threadgroup_position_in_grid]],                              \
    uint lane [[thread_index_in_simdgroup]]);

instantiate_decode_linear_residual(float32, float)
instantiate_decode_linear_residual(bfloat16, bf16)

// Persistent q8_0 weight path. Packed blocks are {fp16 scale; int8 codes[32]}.
template <typename T>
kernel void decode_linear_q8(
    device const T *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const T *bias [[buffer(2)]],
    device const T *residual [[buffer(3)]],
    device T *output [[buffer(4)]],
    constant int &batch [[buffer(5)]],
    constant int &in_dim [[buffer(6)]],
    constant int &out_dim [[buffer(7)]],
    constant int &gelu [[buffer(8)]],
    constant int &use_residual [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int output_channel = int(group.x);
    const int batch_index = int(group.y);
    if (output_channel >= out_dim || batch_index >= batch) return;
    const int blocks = in_dim / 32;
    const long input_base = (long)batch_index * in_dim;
    const long weight_base = (long)output_channel * blocks * 34;
    float accumulator = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        device const uchar *packed = weight + weight_base + (long)block * 34;
        const float scale = float(((device const half *)packed)[0]);
        const int code = int(((device const char *)(packed + 2))[lane]);
        accumulator += scale * float(code) * float(input[input_base + block * 32 + lane]);
    }
    accumulator = metal::simd_sum(accumulator);
    if (lane == 0) {
        const long output_index = (long)batch_index * out_dim + output_channel;
        T rounded = T(accumulator + float(bias[output_channel]));
        if (gelu != 0) rounded = T(glu_gelu_erf(float(rounded)));
        if (use_residual != 0) {
            rounded = T(float(rounded) + float(residual[output_index]));
        }
        output[output_index] = rounded;
    }
}

#define instantiate_decode_linear_q8(type_name, T)                             \
  template [[host_name("decode_linear_q8_" #type_name)]] [[kernel]]          \
  void decode_linear_q8<T>(device const T *input [[buffer(0)]],               \
    device const uchar *weight [[buffer(1)]], device const T *bias [[buffer(2)]], \
    device const T *residual [[buffer(3)]], device T *output [[buffer(4)]],   \
    constant int &batch [[buffer(5)]], constant int &in_dim [[buffer(6)]],    \
    constant int &out_dim [[buffer(7)]], constant int &gelu [[buffer(8)]],    \
    constant int &use_residual [[buffer(9)]],                                 \
    uint2 group [[threadgroup_position_in_grid]],                             \
    uint lane [[thread_index_in_simdgroup]]);

instantiate_decode_linear_q8(float32, float)
instantiate_decode_linear_q8(bfloat16, bf16)

} // namespace mittens
