#include <metal_stdlib>
#include "tk.metal"

using namespace metal;

namespace mittens {

// Swin 2x2 patch gather + LayerNorm. Each simdgroup produces one output patch
// and never materializes the four sliced tensors or their concatenation.
// Input: (B,H*W,C); output: (B,ceil(H/2)*ceil(W/2),4*C), ordered
// [x00,x10,x01,x11].
[[host_name("patch_merge_layernorm_bfloat16")]]
kernel void patch_merge_layernorm_bfloat16(
    device const bf16 *input [[buffer(0)]],
    device const bf16 *weight [[buffer(1)]],
    device const bf16 *bias [[buffer(2)]],
    device bf16 *output [[buffer(3)]],
    constant int &batch [[buffer(4)]],
    constant int &height [[buffer(5)]],
    constant int &width [[buffer(6)]],
    constant int &channels [[buffer(7)]],
    constant float &epsilon [[buffer(8)]],
    uint output_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int output_height = (height + 1) / 2;
    const int output_width = (width + 1) / 2;
    const int patches_per_batch = output_height * output_width;
    const int batch_index = int(output_index) / patches_per_batch;
    if (batch_index >= batch) return;
    const int patch_index = int(output_index) - batch_index * patches_per_batch;
    const int output_y = patch_index / output_width;
    const int output_x = patch_index - output_y * output_width;
    const int dimension = channels * 4;
    const long output_base = (long)output_index * dimension;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int index = int(lane); index < dimension; index += 32) {
        const int quadrant = index / channels;
        const int channel = index - quadrant * channels;
        const int source_y = output_y * 2 + ((quadrant == 1 || quadrant == 3) ? 1 : 0);
        const int source_x = output_x * 2 + ((quadrant >= 2) ? 1 : 0);
        float value = 0.0f;
        if (source_y < height && source_x < width) {
            const long source =
                ((long)batch_index * height * width + source_y * width + source_x) * channels;
            value = float(input[source + channel]);
        }
        sum += value;
        sumsq += value * value;
    }
    sum = metal::simd_sum(sum);
    sumsq = metal::simd_sum(sumsq);
    const float mean = sum / float(dimension);
    const float variance = metal::max(sumsq / float(dimension) - mean * mean, 0.0f);
    const float inverse = metal::rsqrt(variance + epsilon);

    for (int index = int(lane); index < dimension; index += 32) {
        const int quadrant = index / channels;
        const int channel = index - quadrant * channels;
        const int source_y = output_y * 2 + ((quadrant == 1 || quadrant == 3) ? 1 : 0);
        const int source_x = output_x * 2 + ((quadrant >= 2) ? 1 : 0);
        float value = 0.0f;
        if (source_y < height && source_x < width) {
            const long source =
                ((long)batch_index * height * width + source_y * width + source_x) * channels;
            value = float(input[source + channel]);
        }
        output[output_base + index] =
            bf16((value - mean) * inverse * float(weight[index]) + float(bias[index]));
    }
}

} // namespace mittens
