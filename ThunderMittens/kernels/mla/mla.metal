#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// DeepSeek Multi-head Latent Attention (MLA) — preprocessing kernels.
//
// P1: mla_q_norm_rope — the fused Q-path. Per (token, head): optional RMSNorm over
// the full head dim (no-weight for the V4/V3.2 Q-norm, weighted for a kv_a-style
// norm, or none), then GPT-J *interleaved* RoPE on the last `rope_dim` dims (the
// `nope` prefix passes through), bf16 store. Head layout: head_dim = nope_dim +
// rope_dim (e.g. 192 = 128+64 for V2/V3, or 512 = 448+64 for V4).
//
// One warp (32 lanes) per (token, head). Each lane owns head_dim/32 CONTIGUOUS
// elements — even (head_dim % 64 == 0), so every interleaved pair (g, g+1) with g
// even is resident in a single lane (no cross-lane shuffle), and the full-head
// sum-of-squares is a per-lane contiguous sum + one simd_sum. nope_dim even ⇒ a
// pair never straddles the nope/rope boundary.
//
// cos/sin are separate (max_pos, rope_dim/2) bf16 tables (the ThunderMittens RoPE
// convention), indexed by positions[token]. Golden: rmsnorm_no_weight +
// apply_rope_gptj_last_k in vLLM's test_fused_deepseek_v4_qnorm_rope_kv_insert.
// ---------------------------------------------------------------------------
template <int D>
kernel void mla_q_norm_rope(device const bf16 *q          [[buffer(0)]],
                            device const bf16 *cosb        [[buffer(1)]],
                            device const bf16 *sinb        [[buffer(2)]],
                            device const int  *positions   [[buffer(3)]],
                            device bf16       *out         [[buffer(4)]],
                            constant int &num_heads        [[buffer(5)]],
                            constant int &nope_dim         [[buffer(6)]],
                            constant int &rope_dim         [[buffer(7)]],
                            constant int &norm_mode        [[buffer(8)]],   // 0 none,1 rms,2 rms+w
                            constant float &eps            [[buffer(9)]],
                            device const bf16 *norm_weight [[buffer(10)]],  // (D,), read iff mode 2
                            uint3 blockIdx [[threadgroup_position_in_grid]],
                            uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % 64 == 0, "mla_q_norm_rope needs head_dim divisible by 64");
    constexpr int PER_LANE = D / 32;              // contiguous, even
    const int row = blockIdx.x;                   // (token, head) flattened
    const int token = row / num_heads;
    const int pos = positions[token];
    const int rope_half = rope_dim / 2;
    const long base = (long)row * D + (long)laneId * PER_LANE;

    // Full-head RMS (no-weight) if requested.
    float rms = 1.0f;
    if (norm_mode != 0) {
        float ss = 0.0f;
        for (int k = 0; k < PER_LANE; ++k) { const float v = float(q[base + k]); ss += v * v; }
        ss = simd_sum(ss);
        rms = metal::rsqrt(ss / (float)D + eps);
    }

    const long wbase = (long)laneId * PER_LANE;   // norm_weight index for this lane's chunk
    const long csbase = (long)pos * rope_half;
    for (int k = 0; k < PER_LANE; k += 2) {
        const int g0 = (int)laneId * PER_LANE + k;   // even global index (start of a pair)
        float v0 = float(q[base + k]) * rms;
        float v1 = float(q[base + k + 1]) * rms;
        if (norm_mode == 2) {
            v0 *= float(norm_weight[wbase + k]);
            v1 *= float(norm_weight[wbase + k + 1]);
        }
        if (g0 >= nope_dim) {
            const int p = (g0 - nope_dim) / 2;       // rope pair index
            const float c = float(cosb[csbase + p]);
            const float s = float(sinb[csbase + p]);
            out[base + k]     = bf16(v0 * c - v1 * s);
            out[base + k + 1] = bf16(v0 * s + v1 * c);
        } else {
            out[base + k]     = bf16(v0);
            out[base + k + 1] = bf16(v1);
        }
    }
}

#define instantiate_mla_q_norm_rope(DVAL)                                      \
  template [[host_name("mla_q_norm_rope_" #DVAL)]] [[kernel]] void             \
  mla_q_norm_rope<DVAL>(device const bf16 *q [[buffer(0)]],                    \
                        device const bf16 *cosb [[buffer(1)]],                 \
                        device const bf16 *sinb [[buffer(2)]],                 \
                        device const int  *positions [[buffer(3)]],            \
                        device bf16       *out [[buffer(4)]],                  \
                        constant int &num_heads [[buffer(5)]],                 \
                        constant int &nope_dim [[buffer(6)]],                  \
                        constant int &rope_dim [[buffer(7)]],                  \
                        constant int &norm_mode [[buffer(8)]],                 \
                        constant float &eps [[buffer(9)]],                     \
                        device const bf16 *norm_weight [[buffer(10)]],         \
                        uint3 blockIdx [[threadgroup_position_in_grid]],       \
                        uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_mla_q_norm_rope(128);
instantiate_mla_q_norm_rope(192);
instantiate_mla_q_norm_rope(256);
instantiate_mla_q_norm_rope(512);

// ---------------------------------------------------------------------------
// P2: mla_kv_insert — classic bf16 latent KV-insert (concat_and_cache_mla). One warp per token
// writes into a paged cache kv_cache[num_blocks, block_size, LATENT + rope_dim] (MQA — one shared
// latent per token, no head axis): the compressed latent kv_c (LATENT, optionally kv_a-RMSNormed)
// at [0:LATENT], and interleaved-RoPE'd k_pe (rope_dim) at [LATENT:LATENT+rope_dim]. Clone-then-
// insert: the caller pre-populates the cache; this kernel overwrites only the mapped slots.
// LATENT % 64 == 0; rope_dim/2 <= 32 (one pair per lane).
// ---------------------------------------------------------------------------
template <int LATENT>
kernel void mla_kv_insert(device const bf16 *kv_c        [[buffer(0)]],   // (T, LATENT)
                          device const bf16 *k_pe        [[buffer(1)]],   // (T, rope_dim)
                          device const bf16 *cosb        [[buffer(2)]],
                          device const bf16 *sinb        [[buffer(3)]],
                          device const int  *positions   [[buffer(4)]],
                          device const long *slot_mapping [[buffer(5)]],
                          device bf16       *kv_cache    [[buffer(6)]],    // (nb, bs, LATENT+rope)
                          constant int &block_size       [[buffer(7)]],
                          constant int &rope_dim         [[buffer(8)]],
                          constant int &norm_mode        [[buffer(9)]],    // 0 none, 2 weighted
                          constant float &eps            [[buffer(10)]],
                          device const bf16 *norm_weight [[buffer(11)]],   // (LATENT,), mode 2
                          uint3 blockIdx [[threadgroup_position_in_grid]],
                          uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(LATENT % 64 == 0, "mla_kv_insert needs LATENT divisible by 64");
    constexpr int LPL = LATENT / 32;                 // latent elements per lane (even)
    const int token = blockIdx.x;
    const long slot = slot_mapping[token];
    if (slot < 0) { return; }
    const long block = slot / block_size;
    const long off = slot % block_size;
    const int row_width = LATENT + rope_dim;
    const long dst = ((block * block_size + off)) * (long)row_width;
    const int pos = positions[token];
    const int rope_half = rope_dim / 2;

    // Latent: optional RMSNorm over LATENT, then write to [0:LATENT].
    const long lbase = (long)token * LATENT + (long)laneId * LPL;
    float rms = 1.0f;
    if (norm_mode != 0) {
        float ss = 0.0f;
        for (int k = 0; k < LPL; ++k) { const float v = float(kv_c[lbase + k]); ss += v * v; }
        ss = simd_sum(ss);
        rms = metal::rsqrt(ss / (float)LATENT + eps);
    }
    for (int k = 0; k < LPL; ++k) {
        float v = float(kv_c[lbase + k]) * rms;
        if (norm_mode == 2) { v *= float(norm_weight[laneId * LPL + k]); }
        kv_cache[dst + laneId * LPL + k] = bf16(v);
    }

    // RoPE key: interleaved rotate on rope_dim, write to [LATENT:LATENT+rope_dim].
    if ((int)laneId < rope_half) {
        const long rbase = (long)token * rope_dim + (long)laneId * 2;
        const float e = float(k_pe[rbase]);
        const float o = float(k_pe[rbase + 1]);
        const float c = float(cosb[(long)pos * rope_half + laneId]);
        const float s = float(sinb[(long)pos * rope_half + laneId]);
        kv_cache[dst + LATENT + laneId * 2]     = bf16(e * c - o * s);
        kv_cache[dst + LATENT + laneId * 2 + 1] = bf16(e * s + o * c);
    }
}

#define instantiate_mla_kv_insert(LVAL)                                        \
  template [[host_name("mla_kv_insert_" #LVAL)]] [[kernel]] void               \
  mla_kv_insert<LVAL>(device const bf16 *kv_c [[buffer(0)]],                   \
                      device const bf16 *k_pe [[buffer(1)]],                   \
                      device const bf16 *cosb [[buffer(2)]],                   \
                      device const bf16 *sinb [[buffer(3)]],                   \
                      device const int  *positions [[buffer(4)]],              \
                      device const long *slot_mapping [[buffer(5)]],           \
                      device bf16       *kv_cache [[buffer(6)]],               \
                      constant int &block_size [[buffer(7)]],                  \
                      constant int &rope_dim [[buffer(8)]],                    \
                      constant int &norm_mode [[buffer(9)]],                   \
                      constant float &eps [[buffer(10)]],                      \
                      device const bf16 *norm_weight [[buffer(11)]],           \
                      uint3 blockIdx [[threadgroup_position_in_grid]],         \
                      uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_mla_kv_insert(128);
instantiate_mla_kv_insert(256);
instantiate_mla_kv_insert(512);

// Single-buffer bf16 copy (clone-then-insert prologue for the MLA cache).
kernel void mla_cache_clone(device const bf16 *src [[buffer(0)]],
                            device bf16       *dst [[buffer(1)]],
                            constant ulong &n      [[buffer(2)]],
                            uint tid [[thread_position_in_grid]]) {
    if ((ulong)tid < n) { dst[tid] = src[tid]; }
}
