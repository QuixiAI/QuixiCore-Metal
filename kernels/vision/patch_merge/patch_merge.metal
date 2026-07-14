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

template <typename T>
kernel void space_to_depth_norm_linear(
    device const T *input [[buffer(0)]],
    device const T *norm_weight [[buffer(1)]],
    device const T *norm_bias [[buffer(2)]],
    device const T *projection_weight [[buffer(3)]],
    device const T *projection_bias [[buffer(4)]],
    device T *output [[buffer(5)]],
    constant int &batch [[buffer(6)]],
    constant int &height [[buffer(7)]],
    constant int &width [[buffer(8)]],
    constant int &channels [[buffer(9)]],
    constant int &out_channels [[buffer(10)]],
    constant int &block_size [[buffer(11)]],
    constant float &epsilon [[buffer(12)]],
    constant int &use_norm_bias [[buffer(13)]],
    constant int &use_projection_bias [[buffer(14)]],
    uint output_index [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float normalized[4096];
    threadgroup float partial_sum[8];
    threadgroup float partial_sumsq[8];
    threadgroup float stats[2];

    const int output_height = (height + block_size - 1) / block_size;
    const int output_width = (width + block_size - 1) / block_size;
    const int patches_per_batch = output_height * output_width;
    const int batch_index = int(output_index) / patches_per_batch;
    if (batch_index >= batch) return;
    const int patch_index = int(output_index) - batch_index * patches_per_batch;
    const int output_y = patch_index / output_width;
    const int output_x = patch_index - output_y * output_width;
    const int dimension = block_size * block_size * channels;

    float sum = 0.0f, sumsq = 0.0f;
    for (int index = int(thread_index); index < dimension; index += int(threads)) {
        const int spatial = index / channels;
        const int channel = index - spatial * channels;
        const int dy = spatial / block_size;
        const int dx = spatial - dy * block_size;
        const int source_y = output_y * block_size + dy;
        const int source_x = output_x * block_size + dx;
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
    if (lane == 0) {
        partial_sum[simd_index] = sum;
        partial_sumsq[simd_index] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_index == 0) {
        const uint simdgroups = threads / 32;
        float group_sum = lane < simdgroups ? partial_sum[lane] : 0.0f;
        float group_sumsq = lane < simdgroups ? partial_sumsq[lane] : 0.0f;
        group_sum = metal::simd_sum(group_sum);
        group_sumsq = metal::simd_sum(group_sumsq);
        if (lane == 0) {
            const float mean = group_sum / float(dimension);
            const float variance = metal::max(
                group_sumsq / float(dimension) - mean * mean, 0.0f);
            stats[0] = mean;
            stats[1] = metal::rsqrt(variance + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int index = int(thread_index); index < dimension; index += int(threads)) {
        const int spatial = index / channels;
        const int channel = index - spatial * channels;
        const int dy = spatial / block_size;
        const int dx = spatial - dy * block_size;
        const int source_y = output_y * block_size + dy;
        const int source_x = output_x * block_size + dx;
        float value = 0.0f;
        if (source_y < height && source_x < width) {
            const long source =
                ((long)batch_index * height * width + source_y * width + source_x) * channels;
            value = float(input[source + channel]);
        }
        float normed = (value - stats[0]) * stats[1] * float(norm_weight[index]);
        if (use_norm_bias != 0) normed += float(norm_bias[index]);
        normalized[index] = normed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const long output_base = (long)output_index * out_channels;
    for (int out_channel = int(thread_index); out_channel < out_channels;
         out_channel += int(threads)) {
        const long weight_base = (long)out_channel * dimension;
        float accumulator = 0.0f;
        for (int index = 0; index < dimension; ++index) {
            accumulator += normalized[index] *
                float(projection_weight[weight_base + index]);
        }
        if (use_projection_bias != 0) {
            accumulator += float(projection_bias[out_channel]);
        }
        output[output_base + out_channel] = T(accumulator);
    }
}

// Amortize projection-weight traffic across four patches. The launch path uses
// this specialization only for dimensions <= 1024, so four normalized vectors
// fit in the same 16 KiB threadgroup allocation as the generic kernel.
template <typename T>
kernel void space_to_depth_norm_linear_group4(
    device const T *input [[buffer(0)]],
    device const T *norm_weight [[buffer(1)]],
    device const T *norm_bias [[buffer(2)]],
    device const T *projection_weight [[buffer(3)]],
    device const T *projection_bias [[buffer(4)]],
    device T *output [[buffer(5)]],
    constant int &batch [[buffer(6)]],
    constant int &height [[buffer(7)]],
    constant int &width [[buffer(8)]],
    constant int &channels [[buffer(9)]],
    constant int &out_channels [[buffer(10)]],
    constant int &block_size [[buffer(11)]],
    constant float &epsilon [[buffer(12)]],
    constant int &use_norm_bias [[buffer(13)]],
    constant int &use_projection_bias [[buffer(14)]],
    uint patch_group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float normalized[4 * 1024];
    threadgroup float partial_sum[4 * 8];
    threadgroup float partial_sumsq[4 * 8];
    threadgroup float stats[4 * 2];

    const int output_height = (height + block_size - 1) / block_size;
    const int output_width = (width + block_size - 1) / block_size;
    const int patches_per_batch = output_height * output_width;
    const int total_patches = batch * patches_per_batch;
    const int first_output = int(patch_group) * 4;
    const int dimension = block_size * block_size * channels;
    const uint simdgroups = threads / 32;
    bool valid[4];
    int batch_indices[4];
    int output_ys[4];
    int output_xs[4];

    #pragma clang loop unroll(full)
    for (int patch = 0; patch < 4; ++patch) {
        const int output_index = first_output + patch;
        valid[patch] = output_index < total_patches;
        batch_indices[patch] = output_index / patches_per_batch;
        const int patch_index = output_index - batch_indices[patch] * patches_per_batch;
        output_ys[patch] = patch_index / output_width;
        output_xs[patch] = patch_index - output_ys[patch] * output_width;

        float sum = 0.0f, sumsq = 0.0f;
        if (valid[patch]) {
            for (int index = int(thread_index); index < dimension;
                 index += int(threads)) {
                const int spatial = index / channels;
                const int channel = index - spatial * channels;
                const int dy = spatial / block_size;
                const int dx = spatial - dy * block_size;
                const int source_y = output_ys[patch] * block_size + dy;
                const int source_x = output_xs[patch] * block_size + dx;
                float value = 0.0f;
                if (source_y < height && source_x < width) {
                    const long source =
                        ((long)batch_indices[patch] * height * width +
                         source_y * width + source_x) * channels;
                    value = float(input[source + channel]);
                }
                sum += value;
                sumsq += value * value;
            }
        }
        sum = metal::simd_sum(sum);
        sumsq = metal::simd_sum(sumsq);
        if (lane == 0) {
            partial_sum[patch * 8 + simd_index] = sum;
            partial_sumsq[patch * 8 + simd_index] = sumsq;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_index == 0) {
        #pragma clang loop unroll(full)
        for (int patch = 0; patch < 4; ++patch) {
            float group_sum = lane < simdgroups
                ? partial_sum[patch * 8 + lane] : 0.0f;
            float group_sumsq = lane < simdgroups
                ? partial_sumsq[patch * 8 + lane] : 0.0f;
            group_sum = metal::simd_sum(group_sum);
            group_sumsq = metal::simd_sum(group_sumsq);
            if (lane == 0) {
                const float mean = group_sum / float(dimension);
                const float variance = metal::max(
                    group_sumsq / float(dimension) - mean * mean, 0.0f);
                stats[patch * 2] = mean;
                stats[patch * 2 + 1] = metal::rsqrt(variance + epsilon);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (int patch = 0; patch < 4; ++patch) {
        if (!valid[patch]) continue;
        for (int index = int(thread_index); index < dimension;
             index += int(threads)) {
            const int spatial = index / channels;
            const int channel = index - spatial * channels;
            const int dy = spatial / block_size;
            const int dx = spatial - dy * block_size;
            const int source_y = output_ys[patch] * block_size + dy;
            const int source_x = output_xs[patch] * block_size + dx;
            float value = 0.0f;
            if (source_y < height && source_x < width) {
                const long source =
                    ((long)batch_indices[patch] * height * width +
                     source_y * width + source_x) * channels;
                value = float(input[source + channel]);
            }
            float normed = (value - stats[patch * 2]) * stats[patch * 2 + 1] *
                float(norm_weight[index]);
            if (use_norm_bias != 0) normed += float(norm_bias[index]);
            normalized[patch * 1024 + index] = normed;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int out_channel = int(thread_index); out_channel < out_channels;
         out_channel += int(threads)) {
        const long weight_base = (long)out_channel * dimension;
        float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int index = 0; index < dimension; ++index) {
            const float weight = float(projection_weight[weight_base + index]);
            #pragma clang loop unroll(full)
            for (int patch = 0; patch < 4; ++patch) {
                if (valid[patch]) {
                    accumulators[patch] +=
                        normalized[patch * 1024 + index] * weight;
                }
            }
        }
        #pragma clang loop unroll(full)
        for (int patch = 0; patch < 4; ++patch) {
            if (!valid[patch]) continue;
            float result = accumulators[patch];
            if (use_projection_bias != 0) {
                result += float(projection_bias[out_channel]);
            }
            output[(long)(first_output + patch) * out_channels + out_channel] =
                T(result);
        }
    }
}

#define instantiate_space_to_depth_norm_linear(type_name, T)                 \
  template [[host_name("space_to_depth_norm_linear_" #type_name)]]          \
  [[kernel]] void space_to_depth_norm_linear<T>(                             \
    device const T *input [[buffer(0)]], device const T *norm_weight [[buffer(1)]], \
    device const T *norm_bias [[buffer(2)]],                                 \
    device const T *projection_weight [[buffer(3)]],                         \
    device const T *projection_bias [[buffer(4)]], device T *output [[buffer(5)]], \
    constant int &batch [[buffer(6)]], constant int &height [[buffer(7)]],   \
    constant int &width [[buffer(8)]], constant int &channels [[buffer(9)]], \
    constant int &out_channels [[buffer(10)]],                              \
    constant int &block_size [[buffer(11)]], constant float &epsilon [[buffer(12)]], \
    constant int &use_norm_bias [[buffer(13)]],                              \
    constant int &use_projection_bias [[buffer(14)]],                       \
    uint output_index [[threadgroup_position_in_grid]],                      \
    uint thread_index [[thread_index_in_threadgroup]],                       \
    uint simd_index [[simdgroup_index_in_threadgroup]],                      \
    uint lane [[thread_index_in_simdgroup]],                                 \
    uint threads [[threads_per_threadgroup]]);

instantiate_space_to_depth_norm_linear(float32, float)
instantiate_space_to_depth_norm_linear(float16, half)
instantiate_space_to_depth_norm_linear(bfloat16, bf16)

#define instantiate_space_to_depth_norm_linear_group4(type_name, T)          \
  template [[host_name("space_to_depth_norm_linear_group4_" #type_name)]]   \
  [[kernel]] void space_to_depth_norm_linear_group4<T>(                      \
    device const T *input [[buffer(0)]], device const T *norm_weight [[buffer(1)]], \
    device const T *norm_bias [[buffer(2)]],                                 \
    device const T *projection_weight [[buffer(3)]],                         \
    device const T *projection_bias [[buffer(4)]], device T *output [[buffer(5)]], \
    constant int &batch [[buffer(6)]], constant int &height [[buffer(7)]],   \
    constant int &width [[buffer(8)]], constant int &channels [[buffer(9)]], \
    constant int &out_channels [[buffer(10)]],                               \
    constant int &block_size [[buffer(11)]], constant float &epsilon [[buffer(12)]], \
    constant int &use_norm_bias [[buffer(13)]],                              \
    constant int &use_projection_bias [[buffer(14)]],                        \
    uint patch_group [[threadgroup_position_in_grid]],                       \
    uint thread_index [[thread_index_in_threadgroup]],                       \
    uint simd_index [[simdgroup_index_in_threadgroup]],                      \
    uint lane [[thread_index_in_simdgroup]],                                 \
    uint threads [[threads_per_threadgroup]]);

instantiate_space_to_depth_norm_linear_group4(float32, float)
instantiate_space_to_depth_norm_linear_group4(float16, half)
instantiate_space_to_depth_norm_linear_group4(bfloat16, bf16)

} // namespace mittens
