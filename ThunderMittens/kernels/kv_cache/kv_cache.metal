#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

template <typename T>
kernel void kv_cache_zero(device T *key_cache [[buffer(0)]],
                          device T *value_cache [[buffer(1)]],
                          constant ulong &n [[buffer(2)]],
                          uint tid [[thread_position_in_grid]]) {
    if ((ulong)tid >= n) {
        return;
    }
    key_cache[tid] = T(0);
    value_cache[tid] = T(0);
}

template <typename T>
kernel void kv_cache_scatter(device const T *key [[buffer(0)]],
                             device const T *value [[buffer(1)]],
                             device const long *slot_mapping [[buffer(2)]],
                             device T *key_cache [[buffer(3)]],
                             device T *value_cache [[buffer(4)]],
                             constant int &num_heads [[buffer(5)]],
                             constant int &head_size [[buffer(6)]],
                             constant int &block_size [[buffer(7)]],
                             uint token [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tptg [[threads_per_threadgroup]]) {
    const long slot = slot_mapping[token];
    if (slot < 0) {
        return;
    }

    const long block = slot / block_size;
    const long block_offset = slot % block_size;
    const int row_elems = num_heads * head_size;
    const long src_base = (long)token * row_elems;
    const long dst_base =
        ((block * block_size + block_offset) * num_heads) * head_size;

    for (int i = (int)tid; i < row_elems; i += (int)tptg) {
        key_cache[dst_base + i] = key[src_base + i];
        value_cache[dst_base + i] = value[src_base + i];
    }
}

template <typename T>
kernel void kv_cache_gather(device const T *key_cache [[buffer(0)]],
                            device const T *value_cache [[buffer(1)]],
                            device T *key_out [[buffer(2)]],
                            device T *value_out [[buffer(3)]],
                            device const int *block_table [[buffer(4)]],
                            device const int *cu_seq_lens [[buffer(5)]],
                            constant int &num_tokens [[buffer(6)]],
                            constant int &num_seqs [[buffer(7)]],
                            constant int &block_size [[buffer(8)]],
                            constant int &block_table_stride [[buffer(9)]],
                            constant int &num_heads [[buffer(10)]],
                            constant int &head_size [[buffer(11)]],
                            uint token [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tptg [[threads_per_threadgroup]]) {
    if ((int)token >= num_tokens) {
        return;
    }

    int lo = 0;
    int hi = num_seqs;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (cu_seq_lens[mid] <= (int)token) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    const int batch = lo;
    const int local_token = (int)token - cu_seq_lens[batch];
    const int table_col = local_token / block_size;
    const int slot = local_token % block_size;
    const int block = block_table[batch * block_table_stride + table_col];
    const int row_elems = num_heads * head_size;
    const long out_base = (long)token * row_elems;

    if (block < 0) {
        for (int i = (int)tid; i < row_elems; i += (int)tptg) {
            key_out[out_base + i] = T(0);
            value_out[out_base + i] = T(0);
        }
        return;
    }

    const long cache_base =
        (((long)block * block_size + slot) * num_heads) * head_size;
    for (int i = (int)tid; i < row_elems; i += (int)tptg) {
        key_out[out_base + i] = key_cache[cache_base + i];
        value_out[out_base + i] = value_cache[cache_base + i];
    }
}

template <typename T>
kernel void kv_cache_clone(device const T *key_cache [[buffer(0)]],
                           device const T *value_cache [[buffer(1)]],
                           device T *key_out [[buffer(2)]],
                           device T *value_out [[buffer(3)]],
                           constant ulong &n [[buffer(4)]],
                           uint tid [[thread_position_in_grid]]) {
    if ((ulong)tid >= n) {
        return;
    }
    key_out[tid] = key_cache[tid];
    value_out[tid] = value_cache[tid];
}

template <typename T>
kernel void kv_cache_copy_blocks(device T *key_cache [[buffer(0)]],
                                 device T *value_cache [[buffer(1)]],
                                 device const long *block_mapping [[buffer(2)]],
                                 constant int &numel_per_block [[buffer(3)]],
                                 uint pair [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint tptg [[threads_per_threadgroup]]) {
    const long src_block = block_mapping[2 * pair];
    const long dst_block = block_mapping[2 * pair + 1];
    if (src_block < 0 || dst_block < 0) {
        return;
    }

    const long src_base = src_block * numel_per_block;
    const long dst_base = dst_block * numel_per_block;
    for (int i = (int)tid; i < numel_per_block; i += (int)tptg) {
        key_cache[dst_base + i] = key_cache[src_base + i];
        value_cache[dst_base + i] = value_cache[src_base + i];
    }
}

template <typename T>
kernel void kv_cache_scales(device const T *key [[buffer(0)]],
                            device const T *value [[buffer(1)]],
                            device float *key_scale [[buffer(2)]],
                            device float *value_scale [[buffer(3)]],
                            constant ulong &n [[buffer(4)]],
                            uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float shared_key[256];
    threadgroup float shared_value[256];

    float key_max = 0.0f;
    float value_max = 0.0f;
    for (ulong i = tid; i < n; i += 256) {
        key_max = max(key_max, abs(float(key[i])));
        value_max = max(value_max, abs(float(value[i])));
    }

    shared_key[tid] = key_max;
    shared_value[tid] = value_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_key[tid] = max(shared_key[tid], shared_key[tid + stride]);
            shared_value[tid] = max(shared_value[tid], shared_value[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        key_scale[0] = shared_key[0] / 240.0f;
        value_scale[0] = shared_value[0] / 240.0f;
    }
}

template <typename T, int D>
kernel void paged_attention(device const T *q [[buffer(0)]],
                            device const T *key_cache [[buffer(1)]],
                            device const T *value_cache [[buffer(2)]],
                            device const int *block_table [[buffer(3)]],
                            device const int *context_lens [[buffer(4)]],
                            device T *out [[buffer(5)]],
                            constant int &block_size [[buffer(6)]],
                            constant int &block_table_stride [[buffer(7)]],
                            constant float &scale [[buffer(8)]],
                            constant int &num_heads [[buffer(9)]],
                            constant int &num_kv_heads [[buffer(10)]],
                            uint3 tgid [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    constexpr int VALUES_PER_LANE = D / 32;

    const int head = (int)tgid.x;       // query head (grid x ranges over num_heads)
    const int batch = (int)tgid.y;
    // GQA/MQA: each KV head is shared by (num_heads / num_kv_heads) query heads.
    // When num_kv_heads == num_heads this is plain MHA (kv_head == head).
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const long row_base = ((long)batch * num_heads + head) * D;

    float qv[VALUES_PER_LANE];
    float acc[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        qv[i] = float(q[row_base + d]);
        acc[i] = 0.0f;
    }

    float m = -3.4028234663852886e38f;
    float l = 0.0f;

    for (int t = 0; t < context_len; ++t) {
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) {
            continue;
        }

        const long cache_base =
            (((long)block * block_size + slot) * num_kv_heads + kv_head) * D;
        float partial = 0.0f;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            partial += qv[i] * float(key_cache[cache_base + d]);
        }

        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);

        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            acc[i] = acc[i] * alpha + beta * float(value_cache[cache_base + d]);
        }
        l = l * alpha + beta;
        m = new_m;
    }

    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        out[row_base + d] = l == 0.0f ? T(0) : T(acc[i] / l);
    }
}

// --- fp8 KV cache: store e4m3 codes (uint8) with per-tensor K/V scales. ---

kernel void kv_cache_zero_u8(device uchar *key_cache [[buffer(0)]],
                             device uchar *value_cache [[buffer(1)]],
                             constant ulong &n [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
    if ((ulong)tid >= n) { return; }
    key_cache[tid] = 0;     // e4m3(0) == 0x00
    value_cache[tid] = 0;
}

template <typename T>
kernel void kv_cache_scatter_fp8(device const T *key [[buffer(0)]],
                                 device const T *value [[buffer(1)]],
                                 device const long *slot_mapping [[buffer(2)]],
                                 device uchar *key_cache [[buffer(3)]],
                                 device uchar *value_cache [[buffer(4)]],
                                 constant int &num_heads [[buffer(5)]],
                                 constant int &head_size [[buffer(6)]],
                                 constant int &block_size [[buffer(7)]],
                                 device const float *k_scale [[buffer(8)]],
                                 device const float *v_scale [[buffer(9)]],
                                 uint token [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint tptg [[threads_per_threadgroup]]) {
    const long slot = slot_mapping[token];
    if (slot < 0) { return; }
    const long block = slot / block_size;
    const long block_offset = slot % block_size;
    const int row_elems = num_heads * head_size;
    const long src_base = (long)token * row_elems;
    const long dst_base = ((block * block_size + block_offset) * num_heads) * head_size;
    // k_scale/v_scale are (num_heads,) per-head arrays; element i lives in head i/head_size.
    // (Per-tensor callers pass a broadcast array with every entry equal.)
    for (int i = (int)tid; i < row_elems; i += (int)tptg) {
        const int h = i / head_size;
        const float inv_k = k_scale[h] > 0.0f ? 1.0f / k_scale[h] : 0.0f;
        const float inv_v = v_scale[h] > 0.0f ? 1.0f / v_scale[h] : 0.0f;
        key_cache[dst_base + i] = tk_e4m3_encode(float(key[src_base + i]) * inv_k);
        value_cache[dst_base + i] = tk_e4m3_encode(float(value[src_base + i]) * inv_v);
    }
}

template <typename T, int D>
kernel void paged_attention_fp8(device const T *q [[buffer(0)]],
                                device const uchar *key_cache [[buffer(1)]],
                                device const uchar *value_cache [[buffer(2)]],
                                device const int *block_table [[buffer(3)]],
                                device const int *context_lens [[buffer(4)]],
                                device T *out [[buffer(5)]],
                                constant int &block_size [[buffer(6)]],
                                constant int &block_table_stride [[buffer(7)]],
                                constant float &scale [[buffer(8)]],
                                constant int &num_heads [[buffer(9)]],
                                constant int &num_kv_heads [[buffer(10)]],
                                device const float *k_scale [[buffer(11)]],
                                device const float *v_scale [[buffer(12)]],
                                uint3 tgid [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    constexpr int VALUES_PER_LANE = D / 32;
    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const long row_base = ((long)batch * num_heads + head) * D;

    float qv[VALUES_PER_LANE], acc[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        qv[i] = float(q[row_base + d]);
        acc[i] = 0.0f;
    }
    float m = -3.4028234663852886e38f, l = 0.0f;

    for (int t = 0; t < context_len; ++t) {
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) { continue; }
        const long cache_base =
            (((long)block * block_size + slot) * num_kv_heads + kv_head) * D;
        float partial = 0.0f;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            partial += qv[i] * (float(tk_e4m3_decode(key_cache[cache_base + d])) * k_scale[kv_head]);
        }
        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            acc[i] = acc[i] * alpha + beta * (float(tk_e4m3_decode(value_cache[cache_base + d])) * v_scale[kv_head]);
        }
        l = l * alpha + beta;
        m = new_m;
    }
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        out[row_base + d] = l == 0.0f ? T(0) : T(acc[i] / l);
    }
}

#define instantiate_kv_cache_scatter_fp8(type_name, T)                        \
  template [[host_name("kv_cache_scatter_fp8_" #type_name)]] [[kernel]] void  \
  kv_cache_scatter_fp8<T>(device const T *key [[buffer(0)]],                  \
                          device const T *value [[buffer(1)]],                \
                          device const long *slot_mapping [[buffer(2)]],      \
                          device uchar *key_cache [[buffer(3)]],              \
                          device uchar *value_cache [[buffer(4)]],            \
                          constant int &num_heads [[buffer(5)]],              \
                          constant int &head_size [[buffer(6)]],              \
                          constant int &block_size [[buffer(7)]],             \
                          device const float *k_scale [[buffer(8)]],          \
                          device const float *v_scale [[buffer(9)]],          \
                          uint token [[threadgroup_position_in_grid]],        \
                          uint tid [[thread_position_in_threadgroup]],        \
                          uint tptg [[threads_per_threadgroup]]);

#define instantiate_paged_attention_fp8(type_name, T, DVAL)                   \
  template [[host_name("paged_attention_fp8_" #type_name "_" #DVAL)]]         \
  [[kernel]] void paged_attention_fp8<T, DVAL>(                               \
      device const T *q [[buffer(0)]],                                        \
      device const uchar *key_cache [[buffer(1)]],                            \
      device const uchar *value_cache [[buffer(2)]],                          \
      device const int *block_table [[buffer(3)]],                            \
      device const int *context_lens [[buffer(4)]],                          \
      device T *out [[buffer(5)]],                                            \
      constant int &block_size [[buffer(6)]],                                 \
      constant int &block_table_stride [[buffer(7)]],                         \
      constant float &scale [[buffer(8)]],                                    \
      constant int &num_heads [[buffer(9)]],                                  \
      constant int &num_kv_heads [[buffer(10)]],                             \
      device const float *k_scale [[buffer(11)]],                             \
      device const float *v_scale [[buffer(12)]],                             \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_kv_cache_scatter_fp8(float32, float)
instantiate_kv_cache_scatter_fp8(float16, half)
instantiate_kv_cache_scatter_fp8(bfloat16, bf16)
instantiate_paged_attention_fp8(float32, float, 64)
instantiate_paged_attention_fp8(float32, float, 128)
instantiate_paged_attention_fp8(float16, half, 64)
instantiate_paged_attention_fp8(float16, half, 128)
instantiate_paged_attention_fp8(bfloat16, bf16, 64)
instantiate_paged_attention_fp8(bfloat16, bf16, 128)

#define instantiate_kv_cache_type(type_name, T)                               \
  template [[host_name("kv_cache_zero_" #type_name)]] [[kernel]] void        \
  kv_cache_zero<T>(device T *key_cache [[buffer(0)]],                         \
                   device T *value_cache [[buffer(1)]],                       \
                   constant ulong &n [[buffer(2)]],                           \
                   uint tid [[thread_position_in_grid]]);                     \
  template [[host_name("kv_cache_scatter_" #type_name)]] [[kernel]] void     \
  kv_cache_scatter<T>(device const T *key [[buffer(0)]],                      \
                      device const T *value [[buffer(1)]],                    \
                      device const long *slot_mapping [[buffer(2)]],          \
                      device T *key_cache [[buffer(3)]],                      \
                      device T *value_cache [[buffer(4)]],                    \
                      constant int &num_heads [[buffer(5)]],                  \
                      constant int &head_size [[buffer(6)]],                  \
                      constant int &block_size [[buffer(7)]],                 \
                      uint token [[threadgroup_position_in_grid]],            \
                      uint tid [[thread_position_in_threadgroup]],            \
                      uint tptg [[threads_per_threadgroup]]);                 \
  template [[host_name("kv_cache_gather_" #type_name)]] [[kernel]] void      \
  kv_cache_gather<T>(device const T *key_cache [[buffer(0)]],                 \
                     device const T *value_cache [[buffer(1)]],               \
                     device T *key_out [[buffer(2)]],                         \
                     device T *value_out [[buffer(3)]],                       \
                     device const int *block_table [[buffer(4)]],             \
                     device const int *cu_seq_lens [[buffer(5)]],             \
                     constant int &num_tokens [[buffer(6)]],                  \
                     constant int &num_seqs [[buffer(7)]],                    \
                     constant int &block_size [[buffer(8)]],                  \
                     constant int &block_table_stride [[buffer(9)]],          \
                     constant int &num_heads [[buffer(10)]],                  \
                     constant int &head_size [[buffer(11)]],                  \
                     uint token [[threadgroup_position_in_grid]],             \
                     uint tid [[thread_position_in_threadgroup]],             \
                     uint tptg [[threads_per_threadgroup]]);                  \
  template [[host_name("kv_cache_clone_" #type_name)]] [[kernel]] void       \
  kv_cache_clone<T>(device const T *key_cache [[buffer(0)]],                  \
                    device const T *value_cache [[buffer(1)]],                \
                    device T *key_out [[buffer(2)]],                          \
                    device T *value_out [[buffer(3)]],                        \
                    constant ulong &n [[buffer(4)]],                          \
                    uint tid [[thread_position_in_grid]]);                    \
  template [[host_name("kv_cache_copy_blocks_" #type_name)]] [[kernel]] void \
  kv_cache_copy_blocks<T>(device T *key_cache [[buffer(0)]],                  \
                          device T *value_cache [[buffer(1)]],                \
                          device const long *block_mapping [[buffer(2)]],     \
                          constant int &numel_per_block [[buffer(3)]],        \
                          uint pair [[threadgroup_position_in_grid]],         \
                          uint tid [[thread_position_in_threadgroup]],        \
                          uint tptg [[threads_per_threadgroup]]);             \
  template [[host_name("kv_cache_scales_" #type_name)]] [[kernel]] void      \
  kv_cache_scales<T>(device const T *key [[buffer(0)]],                       \
                     device const T *value [[buffer(1)]],                     \
                     device float *key_scale [[buffer(2)]],                   \
                     device float *value_scale [[buffer(3)]],                 \
                     constant ulong &n [[buffer(4)]],                         \
                     uint tid [[thread_position_in_threadgroup]]);

#define instantiate_paged_attention_type(type_name, T, DVAL)                 \
  template [[host_name("paged_attention_" #type_name "_" #DVAL)]]            \
  [[kernel]] void paged_attention<T, DVAL>(                                  \
      device const T *q [[buffer(0)]],                                       \
      device const T *key_cache [[buffer(1)]],                               \
      device const T *value_cache [[buffer(2)]],                             \
      device const int *block_table [[buffer(3)]],                           \
      device const int *context_lens [[buffer(4)]],                          \
      device T *out [[buffer(5)]],                                           \
      constant int &block_size [[buffer(6)]],                                \
      constant int &block_table_stride [[buffer(7)]],                        \
      constant float &scale [[buffer(8)]],                                   \
      constant int &num_heads [[buffer(9)]],                                 \
      constant int &num_kv_heads [[buffer(10)]],                             \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_kv_cache_type(float32, float)
instantiate_kv_cache_type(float16, half)
instantiate_kv_cache_type(bfloat16, bf16)

instantiate_paged_attention_type(float32, float, 64)
instantiate_paged_attention_type(float32, float, 128)
instantiate_paged_attention_type(float16, half, 64)
instantiate_paged_attention_type(float16, half, 128)
instantiate_paged_attention_type(bfloat16, bf16, 64)
instantiate_paged_attention_type(bfloat16, bf16, 128)
