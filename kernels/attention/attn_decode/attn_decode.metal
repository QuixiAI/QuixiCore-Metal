#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Batch-1 attention DECODE (built on request past train_plan §11.7's "custom
// attention: never built" line — the training path keeps SDPA; this serves the
// rollout-generate customer): one new-token query against a dense KV cache,
// GQA, online softmax. The Metal analogue of the CPU engine's
// bn_attn_decode_kv8, minus the int8 cache (rollout caches are bf16).
//
//   q (Hq, D) · kc/vc (Tk, Hkv, D) -> out (Hq, D);  head h reads kv head h/rep.
//
// One simdgroup per query head; lanes stride D (D <= 128 -> <= 4 regs/lane);
// fp32 accumulators throughout. grid (Hq, 1, 1), 32 threads.
// ---------------------------------------------------------------------------

constant float ATTN_NEG_INF = -3.4028234663852886e38f;

template <typename T>
kernel void attn_decode(device const T *q   [[buffer(0)]],
                        device const T *kc  [[buffer(1)]],
                        device const T *vc  [[buffer(2)]],
                        device T       *out [[buffer(3)]],
                        constant int &Tk  [[buffer(4)]],
                        constant int &Hq  [[buffer(5)]],
                        constant int &Hkv [[buffer(6)]],
                        constant int &D   [[buffer(7)]],
                        uint head [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const int rep = Hq / Hkv;
    const int hk = (int)head / rep;
    const float scale = metal::rsqrt(float(D));
    const int per = (D + 31) / 32;                   // regs per lane (<= 4 at D <= 128)

    float qreg[4], oacc[4];
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
        const int d = j * 32 + (int)lane;
        qreg[j] = (j < per && d < D) ? float(q[(long)head * D + d]) : 0.0f;
        oacc[j] = 0.0f;
    }

    float m = ATTN_NEG_INF, l = 0.0f;
    for (int t = 0; t < Tk; ++t) {
        const long kvbase = ((long)t * Hkv + hk) * D;
        float dot = 0.0f;
        for (int j = 0; j < per; ++j) {
            const int d = j * 32 + (int)lane;
            if (d < D) { dot += qreg[j] * float(kc[kvbase + d]); }
        }
        dot = metal::simd_sum(dot) * scale;          // identical on every lane
        const float nm   = max(m, dot);
        const float corr = exp(m - nm);
        const float p    = exp(dot - nm);
        l = l * corr + p;
        for (int j = 0; j < per; ++j) {
            const int d = j * 32 + (int)lane;
            if (d < D) { oacc[j] = oacc[j] * corr + p * float(vc[kvbase + d]); }
        }
        m = nm;
    }
    const float invl = 1.0f / l;
    for (int j = 0; j < per; ++j) {
        const int d = j * 32 + (int)lane;
        if (d < D) { out[(long)head * D + d] = T(oacc[j] * invl); }
    }
}

#define instantiate_attn_decode(type_name, T)                                   \
  template [[host_name("attn_decode_" #type_name)]] [[kernel]] void             \
  attn_decode<T>(device const T *q [[buffer(0)]],                               \
                 device const T *kc [[buffer(1)]],                              \
                 device const T *vc [[buffer(2)]],                              \
                 device T *out [[buffer(3)]],                                   \
                 constant int &Tk [[buffer(4)]],                                \
                 constant int &Hq [[buffer(5)]],                                \
                 constant int &Hkv [[buffer(6)]],                               \
                 constant int &D [[buffer(7)]],                                 \
                 uint head [[threadgroup_position_in_grid]],                    \
                 uint lane [[thread_index_in_simdgroup]]);

instantiate_attn_decode(float32, float)
instantiate_attn_decode(float16, half)
instantiate_attn_decode(bfloat16, bf16)

// Head-major batched decode. q is (B,Hq,D), while K/V use the preallocated
// decoder-cache layout (B,Hkv,cache_T,D), avoiding a history
// transpose or copy before every decode dispatch.
template <typename T>
kernel void attn_decode_bh(device const T *q [[buffer(0)]],
                           device const T *kc [[buffer(1)]],
                           device const T *vc [[buffer(2)]],
                           device T *out [[buffer(3)]],
                           constant int &Tk [[buffer(4)]],
                           constant int &Hq [[buffer(5)]],
                           constant int &Hkv [[buffer(6)]],
                           constant int &D [[buffer(7)]],
                           constant int &cache_T [[buffer(8)]],
                           uint2 pos [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const int head = int(pos.x);
    const int batch = int(pos.y);
    const int rep = Hq / Hkv;
    const int hk = head / rep;
    const float scale = metal::rsqrt(float(D));
    const int per = (D + 31) / 32;
    const long qbase = ((long)batch * Hq + head) * D;

    float qreg[4], oacc[4];
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
        const int d = j * 32 + int(lane);
        qreg[j] = (j < per && d < D) ? float(q[qbase + d]) : 0.0f;
        oacc[j] = 0.0f;
    }

    float maximum = ATTN_NEG_INF, denominator = 0.0f;
    for (int token = 0; token < Tk; ++token) {
        const long kvbase = (((long)batch * Hkv + hk) * cache_T + token) * D;
        float score = 0.0f;
        for (int j = 0; j < per; ++j) {
            const int d = j * 32 + int(lane);
            if (d < D) score += qreg[j] * float(kc[kvbase + d]);
        }
        score = metal::simd_sum(score) * scale;
        const float next_maximum = max(maximum, score);
        const float correction = exp(maximum - next_maximum);
        const float probability = exp(score - next_maximum);
        denominator = denominator * correction + probability;
        for (int j = 0; j < per; ++j) {
            const int d = j * 32 + int(lane);
            if (d < D) {
                oacc[j] = oacc[j] * correction + probability * float(vc[kvbase + d]);
            }
        }
        maximum = next_maximum;
    }

    const float inverse = 1.0f / denominator;
    for (int j = 0; j < per; ++j) {
        const int d = j * 32 + int(lane);
        if (d < D) out[qbase + d] = T(oacc[j] * inverse);
    }
}

#define instantiate_attn_decode_bh(type_name, T)                               \
  template [[host_name("attn_decode_bh_" #type_name)]] [[kernel]] void       \
  attn_decode_bh<T>(device const T *q [[buffer(0)]],                          \
                    device const T *kc [[buffer(1)]],                         \
                    device const T *vc [[buffer(2)]],                         \
                    device T *out [[buffer(3)]],                              \
                    constant int &Tk [[buffer(4)]],                           \
                    constant int &Hq [[buffer(5)]],                           \
                    constant int &Hkv [[buffer(6)]],                          \
                    constant int &D [[buffer(7)]],                            \
                    constant int &cache_T [[buffer(8)]],                      \
                    uint2 pos [[threadgroup_position_in_grid]],               \
                    uint lane [[thread_index_in_simdgroup]]);

instantiate_attn_decode_bh(float32, float)
instantiate_attn_decode_bh(float16, half)
instantiate_attn_decode_bh(bfloat16, bf16)
