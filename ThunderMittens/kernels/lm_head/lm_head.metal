#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Fused LM-head + sampling: pick a decode token WITHOUT materializing the (T, V)
// logits. Two stages (the paged-attn-v2 partition/reduce shape):
//   lm_head_*_partials : grid (num_vtiles, T). One simdgroup owns a TILE_V slice
//     of the vocab for token t. Each lane owns the tile's vocab rows
//     v = base + lane + 32*r and computes the full dot logit = <W[v,:], h[t,:]>
//     serially (no per-vocab reduction), applies invtemp (+ bias), and for
//     stochastic modes adds Gumbel noise indexed by the GLOBAL vocab id v so the
//     fused draw equals the unfused sampler's. h[t,:] is read from global — one
//     small K-vector reused across the tile, served from cache.
//   lm_head_*_reduce : grid (T,). Combine the per-tile partials into the final id.
//
// argmax and categorical share the kernels (a runtime use_gumbel flag); top-k has
// its own pair (k partials per tile, Gumbel-max among the merged candidates).
// TILE_V must be a multiple of 32. W is (V, K) row-major, dtype T (fp16/bf16/f32).
// ---------------------------------------------------------------------------

constant float LMH_NEG_INF = -3.4028234663852886e38f;
constant int LMH_MAX_K = 64;

// ---- argmax / categorical ----
// Grid (num_vtiles, num_tok), one simdgroup per (vocab tile, token) — max parallelism (the GEMV is
// compute-bound here, so parallelism beats W-reuse). Each lane owns the tile's vocab rows
// v = v0 + lane + 32*r and computes <W[v], h[t]> with VEC4 loads (4x fewer load instructions than
// the old scalar dot), then a cross-lane argmax. K % 4 == 0 -> the vec4 path; else a scalar tail.
template <typename T>
kernel void lm_head_argcat_partials(device const T     *h          [[buffer(0)]],   // (num_tok, K)
                                    device const T     *W          [[buffer(1)]],   // (V, K)
                                    device float       *part_val   [[buffer(2)]],   // (num_tok, num_vtiles)
                                    device int         *part_id    [[buffer(3)]],   // (num_tok, num_vtiles)
                                    device const float *bias       [[buffer(4)]],   // (V,) or dummy
                                    constant int   &V          [[buffer(5)]],
                                    constant int   &K          [[buffer(6)]],
                                    constant int   &TILE_V     [[buffer(7)]],
                                    constant int   &num_vtiles [[buffer(8)]],
                                    constant float &invtemp    [[buffer(9)]],
                                    constant uint  &seed       [[buffer(10)]],
                                    constant int   &use_gumbel [[buffer(11)]],
                                    constant int   &use_bias   [[buffer(12)]],
                                    uint2 tgid [[threadgroup_position_in_grid]],
                                    uint  lane [[thread_index_in_simdgroup]]) {
    const int vtile = (int)tgid.x;
    const int t     = (int)tgid.y;
    const int v0 = vtile * TILE_V;
    const int v1 = min(v0 + TILE_V, V);
    const int nk4 = (K % 4 == 0) ? K / 4 : 0;               // vec4 rows must be 4-element aligned
    device const T *hrow = h + (long)t * K;
    device const metal::vec<T, 4> *h4 = (device const metal::vec<T, 4>*)hrow;

    float best = LMH_NEG_INF;
    int   bi   = (v0 + (int)lane < v1) ? v0 + (int)lane : v0;
    for (int v = v0 + (int)lane; v < v1; v += 32) {
        device const T *wrow = W + (long)v * K;
        float dot_acc = 0.0f;
        device const metal::vec<T, 4> *w4 = (device const metal::vec<T, 4>*)wrow;
        for (int j = 0; j < nk4; ++j) dot_acc += dot(float4(w4[j]), float4(h4[j]));
        for (int i = nk4 * 4; i < K; ++i) dot_acc += float(wrow[i]) * float(hrow[i]);
        float ls = dot_acc * invtemp;
        if (use_bias) ls += bias[v];
        if (use_gumbel) ls += rng_gumbel(seed, (uint)t, (uint)v);
        if (ls > best || (ls == best && v < bi)) { best = ls; bi = v; }
    }
    simd_argmax(best, bi);
    if (lane == 0) {
        part_val[(long)t * num_vtiles + vtile] = best;
        part_id[(long)t * num_vtiles + vtile]  = bi;
    }
}

kernel void lm_head_argcat_reduce(device const float *part_val   [[buffer(0)]],
                                  device const int   *part_id    [[buffer(1)]],
                                  device int         *out_idx    [[buffer(2)]],
                                  constant int &num_vtiles       [[buffer(3)]],
                                  uint  t    [[threadgroup_position_in_grid]],
                                  uint  lane [[thread_index_in_simdgroup]]) {
    const long base = (long)t * num_vtiles;
    float best = LMH_NEG_INF;
    int   bi   = 0x7fffffff;
    for (int j = (int)lane; j < num_vtiles; j += 32) {
        const float v = part_val[base + j];
        const int   id = part_id[base + j];
        if (v > best || (v == best && id < bi)) { best = v; bi = id; }
    }
    simd_argmax(best, bi);
    if (lane == 0) out_idx[t] = bi;
}

// ---- argmax / categorical over QUANTIZED weights (q8_0 / q4_0) ----
// Same geometry as lm_head_argcat_partials, but W is a packed (V, K/block_k, block_bytes) uchar
// tensor dequantized on the fly (tk_dequant8<FMT> unpacks 8 columns/span). h dtype = T (f16/bf16/f32).
template <typename FMT, typename T>
kernel void lm_head_argcat_partials_q(device const T     *h          [[buffer(0)]],  // (num_tok, K)
                                      device const uchar *Wq         [[buffer(1)]],  // packed (V,K)
                                      device float       *part_val   [[buffer(2)]],
                                      device int         *part_id    [[buffer(3)]],
                                      device const float *bias       [[buffer(4)]],
                                      constant int   &V          [[buffer(5)]],
                                      constant int   &K          [[buffer(6)]],
                                      constant int   &TILE_V     [[buffer(7)]],
                                      constant int   &num_vtiles [[buffer(8)]],
                                      constant float &invtemp    [[buffer(9)]],
                                      constant uint  &seed       [[buffer(10)]],
                                      constant int   &use_gumbel [[buffer(11)]],
                                      constant int   &use_bias   [[buffer(12)]],
                                      uint2 tgid [[threadgroup_position_in_grid]],
                                      uint  lane [[thread_index_in_simdgroup]]) {
    const int vtile = (int)tgid.x;
    const int t     = (int)tgid.y;
    device const T *hrow = h + (long)t * K;
    const int v0 = vtile * TILE_V;
    const int v1 = min(v0 + TILE_V, V);
    const int bpr = K / FMT::block_k;                       // quant blocks per weight row

    float best = LMH_NEG_INF;
    int   bi   = (v0 + (int)lane < v1) ? v0 + (int)lane : v0;
    for (int v = v0 + (int)lane; v < v1; v += 32) {
        device const uchar *wq_row = Wq + (long)v * bpr * FMT::block_bytes;
        float dot_acc = 0.0f;
        for (int b = 0; b < bpr; ++b) {
            device const uchar *base = wq_row + (long)b * FMT::block_bytes;
            for (int cib = 0; cib < FMT::block_k; cib += 8) {
                half w8[8];
                tk_dequant8<FMT>(base, cib, w8);
                const int koff = b * FMT::block_k + cib;
                for (int kk = 0; kk < 8; ++kk) dot_acc += float(w8[kk]) * float(hrow[koff + kk]);
            }
        }
        float ls = dot_acc * invtemp;
        if (use_bias) ls += bias[v];
        if (use_gumbel) ls += rng_gumbel(seed, (uint)t, (uint)v);
        if (ls > best || (ls == best && v < bi)) { best = ls; bi = v; }
    }
    simd_argmax(best, bi);
    if (lane == 0) {
        part_val[(long)t * num_vtiles + vtile] = best;
        part_id[(long)t * num_vtiles + vtile]  = bi;
    }
}

#define instantiate_lm_head_q(hn, FMT, T)                                          \
  template [[host_name("lm_head_argcat_partials_q_" hn)]] [[kernel]] void          \
  lm_head_argcat_partials_q<FMT, T>(device const T *h [[buffer(0)]],               \
    device const uchar *Wq [[buffer(1)]], device float *part_val [[buffer(2)]],    \
    device int *part_id [[buffer(3)]], device const float *bias [[buffer(4)]],     \
    constant int &V [[buffer(5)]], constant int &K [[buffer(6)]],                  \
    constant int &TILE_V [[buffer(7)]], constant int &num_vtiles [[buffer(8)]],    \
    constant float &invtemp [[buffer(9)]], constant uint &seed [[buffer(10)]],     \
    constant int &use_gumbel [[buffer(11)]], constant int &use_bias [[buffer(12)]], \
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_lm_head_q("q8_0_float32", q8_0, float)
instantiate_lm_head_q("q8_0_float16", q8_0, half)
instantiate_lm_head_q("q8_0_bfloat16", q8_0, bf16)
instantiate_lm_head_q("q4_0_float32", q4_0, float)
instantiate_lm_head_q("q4_0_float16", q4_0, half)
instantiate_lm_head_q("q4_0_bfloat16", q4_0, bf16)

// ---- top-k ----
template <typename T>
kernel void lm_head_topk_partials(device const T     *h          [[buffer(0)]],
                                  device const T     *W          [[buffer(1)]],
                                  device float       *part_val   [[buffer(2)]],   // (num_tok, num_vtiles, K)
                                  device int         *part_id    [[buffer(3)]],
                                  device const float *bias       [[buffer(4)]],
                                  constant int   &V          [[buffer(5)]],
                                  constant int   &K          [[buffer(6)]],
                                  constant int   &TILE_V     [[buffer(7)]],
                                  constant int   &num_vtiles [[buffer(8)]],
                                  constant int   &topk       [[buffer(9)]],
                                  constant int   &use_bias   [[buffer(10)]],
                                  uint2 tgid [[threadgroup_position_in_grid]],
                                  uint  lane [[thread_index_in_simdgroup]]) {
    const int vtile = (int)tgid.x;
    const int t     = (int)tgid.y;
    device const T *hrow = h + (long)t * K;

    const int v0 = vtile * TILE_V;
    const int v1 = min(v0 + TILE_V, V);
    const int nk4 = (K % 4 == 0) ? K / 4 : 0;
    device const metal::vec<T, 4> *h4 = (device const metal::vec<T, 4>*)hrow;
    constexpr int MAX_PER_LANE = 2048 / 32;   // TILE_V <= 2048
    float mine_val[MAX_PER_LANE];
    int   mine_id[MAX_PER_LANE];
    bool  used[MAX_PER_LANE];
    int   nmine = 0;
    for (int v = v0 + (int)lane; v < v1; v += 32) {
        device const T *wrow = W + (long)v * K;
        device const metal::vec<T, 4> *w4 = (device const metal::vec<T, 4>*)wrow;
        float dot_acc = 0.0f;
        for (int j = 0; j < nk4; ++j) dot_acc += dot(float4(w4[j]), float4(h4[j]));
        for (int i = nk4 * 4; i < K; ++i) dot_acc += float(wrow[i]) * float(hrow[i]);
        float ls = dot_acc;
        if (use_bias) ls += bias[v];
        mine_val[nmine] = ls;
        mine_id[nmine]  = v;
        used[nmine]     = false;
        ++nmine;
    }
    const long pbase = ((long)t * num_vtiles + vtile) * topk;
    for (int kk = 0; kk < topk; ++kk) {
        float best = LMH_NEG_INF;
        int   bi   = -1;
        int   bl   = -1;
        for (int j = 0; j < nmine; ++j) {
            if (used[j]) continue;
            if (mine_val[j] > best || (mine_val[j] == best && mine_id[j] < bi)) {
                best = mine_val[j]; bi = mine_id[j]; bl = j;
            }
        }
        float gbest = best;
        int   gid   = (bi < 0) ? 0x7fffffff : bi;
        simd_argmax(gbest, gid);
        if (bl >= 0 && bi == gid) used[bl] = true;   // owner of the winner marks it used
        if (lane == 0) {
            part_val[pbase + kk] = gbest;
            part_id[pbase + kk]  = (gbest == LMH_NEG_INF) ? -1 : gid;
        }
    }
}

kernel void lm_head_topk_reduce(device const float *part_val   [[buffer(0)]],   // (num_tok, num_vtiles, K)
                                device const int   *part_id    [[buffer(1)]],
                                device int         *out_idx    [[buffer(2)]],
                                constant int &num_vtiles       [[buffer(3)]],
                                constant int &topk             [[buffer(4)]],
                                constant uint &seed            [[buffer(5)]],
                                constant float &invtemp        [[buffer(6)]],
                                uint  t    [[threadgroup_position_in_grid]],
                                uint  lane [[thread_index_in_simdgroup]]) {
    const int ncand = num_vtiles * topk;
    const long base = (long)t * ncand;
    int   chosen_id[LMH_MAX_K];
    float chosen_val[LMH_MAX_K];
    // K masked-argmax rounds over the merged per-tile partial winners (Family-A helper).
    stored_cand cand{part_val, part_id, base};
    masked_topk(cand, ncand, topk, lane, LMH_NEG_INF, chosen_id, chosen_val);
    // Gumbel-max among the k winners (global vocab id in the noise stream).
    float best = LMH_NEG_INF;
    int   bi   = chosen_id[0];
    for (int kk = 0; kk < topk; ++kk) {
        if (chosen_id[kk] < 0) continue;
        const float g = rng_gumbel(seed, (uint)t, (uint)chosen_id[kk]);
        const float p = chosen_val[kk] * invtemp + g;
        if (p > best || (p == best && chosen_id[kk] < bi)) { best = p; bi = chosen_id[kk]; }
    }
    if (lane == 0) out_idx[t] = bi;
}

// Top-p (nucleus) reduce over the merged over-selected candidate pool (num_vtiles*topk per token).
// The nucleus tokens are selected from the pool (the top-k' contains them for the peaked LM-head
// distribution), but the cumulative mass is measured against the TRUE full-vocab normalizer Z built
// from the per-tile logsumexps (part_lse) emitted by lm_head_topp_partials_q — so the nucleus cutoff
// is exact, not the pool-only approximation. Mirrors top_p_sample: bisect the tempered-logit
// threshold L for cumulative mass >= p, then Gumbel-max over {ls >= L} (noise indexed by global id).
kernel void lm_head_topp_reduce(device const float *part_val [[buffer(0)]],   // (T, num_vtiles, topk)
                                device const int   *part_id  [[buffer(1)]],
                                device int         *out_idx  [[buffer(2)]],
                                constant int   &num_vtiles   [[buffer(3)]],
                                constant int   &topk         [[buffer(4)]],
                                constant float &p            [[buffer(5)]],
                                constant uint  &seed         [[buffer(6)]],
                                constant float &invtemp      [[buffer(7)]],
                                device const float *part_lse [[buffer(8)]],   // (T, num_vtiles)
                                uint  t    [[threadgroup_position_in_grid]],
                                uint  lane [[thread_index_in_simdgroup]]) {
    const int ncand = num_vtiles * topk;
    const long base = (long)t * ncand;
    float mx = LMH_NEG_INF;
    for (int j = (int)lane; j < ncand; j += 32) {
        if (part_id[base + j] >= 0) { mx = max(mx, part_val[base + j] * invtemp); }
    }
    mx = simd_max(mx);
    // true full-vocab Z from the per-tile tempered logsumexps: Z = sum_tiles exp(part_lse - mx)
    const long lbase = (long)t * num_vtiles;
    float Z = 0.0f;
    for (int vt = (int)lane; vt < num_vtiles; vt += 32) {
        const float pl = part_lse[lbase + vt];
        if (pl > LMH_NEG_INF) { Z += exp(pl - mx); }
    }
    Z = simd_sum(Z);
    float lo = mx - 40.0f, hi = mx;             // bisect L: largest L with mass{ls>=L} >= p
    for (int it = 0; it < 32; ++it) {
        const float mid = 0.5f * (lo + hi);
        float sm = 0.0f;
        for (int j = (int)lane; j < ncand; j += 32) {
            const float ls = part_val[base + j] * invtemp;
            if (part_id[base + j] >= 0 && ls >= mid) { sm += exp(ls - mx); }
        }
        sm = simd_sum(sm) / Z;
        if (sm >= p) { lo = mid; } else { hi = mid; }
    }
    const float L = lo;
    float best = LMH_NEG_INF;
    int   bi   = -1;
    for (int j = (int)lane; j < ncand; j += 32) {
        const int id = part_id[base + j];
        const float ls = part_val[base + j] * invtemp;
        if (id < 0 || ls < L) { continue; }
        const float pert = ls + rng_gumbel(seed, (uint)t, (uint)id);
        if (pert > best || (pert == best && id < bi)) { best = pert; bi = id; }
    }
    float gbest = best;
    int   gid   = (bi < 0) ? 0x7fffffff : bi;
    simd_argmax(gbest, gid);
    if (lane == 0) { out_idx[t] = (gbest == LMH_NEG_INF) ? -1 : gid; }
}

// Quantized top-k partials: same per-lane top-k as lm_head_topk_partials, but W is packed
// (V, K/block_k, block_bytes) uchar dequantized on the fly via tk_dequant8<FMT>. Feeds the SAME
// lm_head_topk_reduce (global merge + Gumbel-max), so the fused quant top-k path avoids ever
// materializing the (T,V) logits — the point of quant decode.
template <typename FMT, typename T>
kernel void lm_head_topk_partials_q(device const T     *h          [[buffer(0)]],
                                    device const uchar *Wq         [[buffer(1)]],  // packed (V,K)
                                    device float       *part_val   [[buffer(2)]],
                                    device int         *part_id    [[buffer(3)]],
                                    device const float *bias       [[buffer(4)]],
                                    constant int   &V          [[buffer(5)]],
                                    constant int   &K          [[buffer(6)]],
                                    constant int   &TILE_V     [[buffer(7)]],
                                    constant int   &num_vtiles [[buffer(8)]],
                                    constant int   &topk       [[buffer(9)]],
                                    constant int   &use_bias   [[buffer(10)]],
                                    uint2 tgid [[threadgroup_position_in_grid]],
                                    uint  lane [[thread_index_in_simdgroup]]) {
    const int vtile = (int)tgid.x;
    const int t     = (int)tgid.y;
    device const T *hrow = h + (long)t * K;
    const int v0 = vtile * TILE_V;
    const int v1 = min(v0 + TILE_V, V);
    const int bpr = K / FMT::block_k;
    constexpr int MAX_PER_LANE = 2048 / 32;   // TILE_V <= 2048
    float mine_val[MAX_PER_LANE];
    int   mine_id[MAX_PER_LANE];
    bool  used[MAX_PER_LANE];
    int   nmine = 0;
    for (int v = v0 + (int)lane; v < v1; v += 32) {
        device const uchar *wq_row = Wq + (long)v * bpr * FMT::block_bytes;
        float dot_acc = 0.0f;
        for (int b = 0; b < bpr; ++b) {
            device const uchar *base = wq_row + (long)b * FMT::block_bytes;
            for (int cib = 0; cib < FMT::block_k; cib += 8) {
                half w8[8];
                tk_dequant8<FMT>(base, cib, w8);
                const int koff = b * FMT::block_k + cib;
                for (int kk = 0; kk < 8; ++kk) dot_acc += float(w8[kk]) * float(hrow[koff + kk]);
            }
        }
        float ls = dot_acc;
        if (use_bias) ls += bias[v];
        mine_val[nmine] = ls;
        mine_id[nmine]  = v;
        used[nmine]     = false;
        ++nmine;
    }
    const long pbase = ((long)t * num_vtiles + vtile) * topk;
    for (int kk = 0; kk < topk; ++kk) {
        float best = LMH_NEG_INF;
        int   bi   = -1;
        int   bl   = -1;
        for (int j = 0; j < nmine; ++j) {
            if (used[j]) continue;
            if (mine_val[j] > best || (mine_val[j] == best && mine_id[j] < bi)) {
                best = mine_val[j]; bi = mine_id[j]; bl = j;
            }
        }
        float gbest = best;
        int   gid   = (bi < 0) ? 0x7fffffff : bi;
        simd_argmax(gbest, gid);
        if (bl >= 0 && bi == gid) used[bl] = true;
        if (lane == 0) {
            part_val[pbase + kk] = gbest;
            part_id[pbase + kk]  = (gbest == LMH_NEG_INF) ? -1 : gid;
        }
    }
}

#define instantiate_lm_head_topk_q(hn, FMT, T)                                     \
  template [[host_name("lm_head_topk_partials_q_" hn)]] [[kernel]] void             \
  lm_head_topk_partials_q<FMT, T>(device const T *h [[buffer(0)]],                  \
    device const uchar *Wq [[buffer(1)]], device float *part_val [[buffer(2)]],     \
    device int *part_id [[buffer(3)]], device const float *bias [[buffer(4)]],      \
    constant int &V [[buffer(5)]], constant int &K [[buffer(6)]],                   \
    constant int &TILE_V [[buffer(7)]], constant int &num_vtiles [[buffer(8)]],     \
    constant int &topk [[buffer(9)]], constant int &use_bias [[buffer(10)]],        \
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_lm_head_topk_q("q8_0_float32", q8_0, float)
instantiate_lm_head_topk_q("q8_0_float16", q8_0, half)
instantiate_lm_head_topk_q("q8_0_bfloat16", q8_0, bf16)
instantiate_lm_head_topk_q("q4_0_float32", q4_0, float)
instantiate_lm_head_topk_q("q4_0_float16", q4_0, half)
instantiate_lm_head_topk_q("q4_0_bfloat16", q4_0, bf16)

// Quantized top-p partials: identical top-k selection to lm_head_topk_partials_q, but ALSO emits a
// per-tile tempered log-sum-exp (part_lse) over EVERY dequantized logit in the tile (not just the
// top-k). The reduce merges these tile-lses into the true full-vocab normalizer Z, so the nucleus
// threshold is exact (not the pool-only approximation of the shared top-k partials). Needs invtemp
// because the softmax normalizer is temperature-dependent.
template <typename FMT, typename T>
kernel void lm_head_topp_partials_q(device const T     *h          [[buffer(0)]],
                                    device const uchar *Wq         [[buffer(1)]],  // packed (V,K)
                                    device float       *part_val   [[buffer(2)]],
                                    device int         *part_id    [[buffer(3)]],
                                    device const float *bias       [[buffer(4)]],
                                    constant int   &V          [[buffer(5)]],
                                    constant int   &K          [[buffer(6)]],
                                    constant int   &TILE_V     [[buffer(7)]],
                                    constant int   &num_vtiles [[buffer(8)]],
                                    constant int   &topk       [[buffer(9)]],
                                    constant int   &use_bias   [[buffer(10)]],
                                    constant float &invtemp    [[buffer(11)]],
                                    device float       *part_lse   [[buffer(12)]],  // (T, num_vtiles)
                                    uint2 tgid [[threadgroup_position_in_grid]],
                                    uint  lane [[thread_index_in_simdgroup]]) {
    const int vtile = (int)tgid.x;
    const int t     = (int)tgid.y;
    device const T *hrow = h + (long)t * K;
    const int v0 = vtile * TILE_V;
    const int v1 = min(v0 + TILE_V, V);
    const int bpr = K / FMT::block_k;
    constexpr int MAX_PER_LANE = 2048 / 32;   // TILE_V <= 2048
    float mine_val[MAX_PER_LANE];
    int   mine_id[MAX_PER_LANE];
    bool  used[MAX_PER_LANE];
    int   nmine = 0;
    float lmax = LMH_NEG_INF, lsum = 0.0f;    // streaming tempered logsumexp over this lane's logits
    for (int v = v0 + (int)lane; v < v1; v += 32) {
        device const uchar *wq_row = Wq + (long)v * bpr * FMT::block_bytes;
        float dot_acc = 0.0f;
        for (int b = 0; b < bpr; ++b) {
            device const uchar *base = wq_row + (long)b * FMT::block_bytes;
            for (int cib = 0; cib < FMT::block_k; cib += 8) {
                half w8[8];
                tk_dequant8<FMT>(base, cib, w8);
                const int koff = b * FMT::block_k + cib;
                for (int kk = 0; kk < 8; ++kk) dot_acc += float(w8[kk]) * float(hrow[koff + kk]);
            }
        }
        float ls = dot_acc;
        if (use_bias) ls += bias[v];
        const float tl = ls * invtemp;                    // tempered logit
        if (tl > lmax) { lsum = lsum * exp(lmax - tl) + 1.0f; lmax = tl; }
        else           { lsum += exp(tl - lmax); }
        mine_val[nmine] = ls;
        mine_id[nmine]  = v;
        used[nmine]     = false;
        ++nmine;
    }
    // per-tile lse = logsumexp over all lanes' tempered logits
    const float gmax = simd_max(lmax);
    float gsum = simd_sum((lmax == LMH_NEG_INF) ? 0.0f : lsum * exp(lmax - gmax));
    if (lane == 0) {
        part_lse[(long)t * num_vtiles + vtile] = (gsum > 0.0f) ? (gmax + log(gsum)) : LMH_NEG_INF;
    }
    const long pbase = ((long)t * num_vtiles + vtile) * topk;
    for (int kk = 0; kk < topk; ++kk) {
        float best = LMH_NEG_INF;
        int   bi   = -1;
        int   bl   = -1;
        for (int j = 0; j < nmine; ++j) {
            if (used[j]) continue;
            if (mine_val[j] > best || (mine_val[j] == best && mine_id[j] < bi)) {
                best = mine_val[j]; bi = mine_id[j]; bl = j;
            }
        }
        float gbest = best;
        int   gid   = (bi < 0) ? 0x7fffffff : bi;
        simd_argmax(gbest, gid);
        if (bl >= 0 && bi == gid) used[bl] = true;
        if (lane == 0) {
            part_val[pbase + kk] = gbest;
            part_id[pbase + kk]  = (gbest == LMH_NEG_INF) ? -1 : gid;
        }
    }
}

#define instantiate_lm_head_topp_q(hn, FMT, T)                                     \
  template [[host_name("lm_head_topp_partials_q_" hn)]] [[kernel]] void             \
  lm_head_topp_partials_q<FMT, T>(device const T *h [[buffer(0)]],                  \
    device const uchar *Wq [[buffer(1)]], device float *part_val [[buffer(2)]],     \
    device int *part_id [[buffer(3)]], device const float *bias [[buffer(4)]],      \
    constant int &V [[buffer(5)]], constant int &K [[buffer(6)]],                   \
    constant int &TILE_V [[buffer(7)]], constant int &num_vtiles [[buffer(8)]],     \
    constant int &topk [[buffer(9)]], constant int &use_bias [[buffer(10)]],        \
    constant float &invtemp [[buffer(11)]], device float *part_lse [[buffer(12)]],  \
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_lm_head_topp_q("q8_0_float32", q8_0, float)
instantiate_lm_head_topp_q("q8_0_float16", q8_0, half)
instantiate_lm_head_topp_q("q8_0_bfloat16", q8_0, bf16)
instantiate_lm_head_topp_q("q4_0_float32", q4_0, float)
instantiate_lm_head_topp_q("q4_0_float16", q4_0, half)
instantiate_lm_head_topp_q("q4_0_bfloat16", q4_0, bf16)

#define instantiate_lm_head(type_name, T)                                          \
  template [[host_name("lm_head_argcat_partials_" #type_name)]] [[kernel]] void     \
  lm_head_argcat_partials<T>(device const T *h [[buffer(0)]], device const T *W [[buffer(1)]], \
    device float *part_val [[buffer(2)]], device int *part_id [[buffer(3)]],        \
    device const float *bias [[buffer(4)]], constant int &V [[buffer(5)]],          \
    constant int &K [[buffer(6)]], constant int &TILE_V [[buffer(7)]],              \
    constant int &num_vtiles [[buffer(8)]], constant float &invtemp [[buffer(9)]],  \
    constant uint &seed [[buffer(10)]], constant int &use_gumbel [[buffer(11)]],    \
    constant int &use_bias [[buffer(12)]],                                          \
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]); \
  template [[host_name("lm_head_topk_partials_" #type_name)]] [[kernel]] void        \
  lm_head_topk_partials<T>(device const T *h [[buffer(0)]], device const T *W [[buffer(1)]], \
    device float *part_val [[buffer(2)]], device int *part_id [[buffer(3)]],        \
    device const float *bias [[buffer(4)]], constant int &V [[buffer(5)]],          \
    constant int &K [[buffer(6)]], constant int &TILE_V [[buffer(7)]],              \
    constant int &num_vtiles [[buffer(8)]], constant int &topk [[buffer(9)]],       \
    constant int &use_bias [[buffer(10)]],                                          \
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_lm_head(float32, float)
instantiate_lm_head(float16, half)
instantiate_lm_head(bfloat16, bf16)
