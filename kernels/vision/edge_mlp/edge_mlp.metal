#include <metal_stdlib>
#include "tk.metal"

using namespace metal;

namespace mittens {

// Fixed 512->256->7 pairwise edge MLP. One 256-thread
// threadgroup owns a pair, computes the hidden activation once, then seven
// simdgroups produce the edge logits without a BxLxLx256 intermediate.
template <typename T>
kernel void edge_mlp_256x7(
    device const T *hidden [[buffer(0)]],
    device const T *first_weight [[buffer(1)]],
    device const T *first_bias [[buffer(2)]],
    device const T *second_weight [[buffer(3)]],
    device const T *second_bias [[buffer(4)]],
    device T *output [[buffer(5)]],
    constant int &batch [[buffer(6)]],
    constant int &length [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    const int batch_index = int(group.y);
    const int pair = int(group.x);
    if (batch_index >= batch || pair >= length * length) return;
    const int left_index = pair / length;
    const int right_index = pair - left_index * length;
    const int feature = int(thread_id);
    threadgroup float activation[256];

    const long left_base = ((long)batch_index * length + left_index) * 256;
    const long right_base = ((long)batch_index * length + right_index) * 256;
    const long weight_base = (long)feature * 512;
    float left = 0.0f;
    float right = float(first_bias[feature]);
    for (int dimension = 0; dimension < 256; ++dimension) {
        left += float(hidden[left_base + dimension]) *
                float(first_weight[weight_base + dimension]);
        right += float(hidden[right_base + dimension]) *
                 float(first_weight[weight_base + 256 + dimension]);
    }
    const T left_rounded = T(left);
    const T right_rounded = T(right);
    const T joined = T(float(left_rounded) + float(right_rounded));
    activation[feature] = float(T(glu_gelu_erf(float(joined))));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_id < 7) {
        const int edge_class = int(simdgroup_id);
        float accumulator = 0.0f;
        for (int index = int(lane); index < 256; index += 32) {
            accumulator += activation[index] *
                           float(second_weight[edge_class * 256 + index]);
        }
        accumulator = metal::simd_sum(accumulator);
        if (lane == 0) {
            const long destination =
                ((long)batch_index * 7 + edge_class) * length * length + pair;
            output[destination] = T(accumulator + float(second_bias[edge_class]));
        }
    }
}

#define instantiate_edge_mlp(type_name, T)                                    \
  template [[host_name("edge_mlp_256x7_" #type_name)]] [[kernel]]           \
  void edge_mlp_256x7<T>(device const T *hidden [[buffer(0)]],                \
    device const T *first_weight [[buffer(1)]],                               \
    device const T *first_bias [[buffer(2)]],                                 \
    device const T *second_weight [[buffer(3)]],                              \
    device const T *second_bias [[buffer(4)]], device T *output [[buffer(5)]], \
    constant int &batch [[buffer(6)]], constant int &length [[buffer(7)]],    \
    uint2 group [[threadgroup_position_in_grid]],                             \
    uint thread_id [[thread_index_in_threadgroup]],                           \
    uint lane [[thread_index_in_simdgroup]],                                  \
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

instantiate_edge_mlp(float32, float)
instantiate_edge_mlp(bfloat16, bf16)

} // namespace mittens
