#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Sampling kernels: keep the final decode step on-GPU. One simdgroup (32 lanes)
// per row of logits (vocab dimension V, any size, looped with stride 32).
//
// Substrate primitives reused: mittens::simd_argmax (P1, argmax-with-index) and
// mittens::rng_uniform / rng_gumbel (P4, reproducible RNG).
// ---------------------------------------------------------------------------

constant float SMP_NEG_INF = -3.4028234663852886e38f;

constant int SAMPLE_MAX_K = 64;

template <typename T>
kernel void argmax(device const T *logits  [[buffer(0)]],
                   device int     *out_idx [[buffer(1)]],
                   constant int   &V       [[buffer(2)]],
                   uint row  [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    float best = SMP_NEG_INF;
    int bi = (int)lane < V ? (int)lane : 0;
    for (int i = (int)lane; i < V; i += 32) {
        const float v = float(logits[base + i]);
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
    }
    simd_argmax(best, bi);
    if (lane == 0) {
        out_idx[row] = bi;
    }
}

// Stochastic categorical sampling via the Gumbel-max trick:
//   token = argmax_i ( logits[i]/temperature + Gumbel_i ),  Gumbel_i = -log(-log(u_i))
// which samples exactly from softmax(logits/temperature). The draw is fully determined
// by (seed, row), so a numpy reference reproducing rng_uniform/Gumbel matches exactly.
template <typename T>
kernel void sample_categorical(device const T *logits  [[buffer(0)]],
                               device int     *out_idx [[buffer(1)]],
                               constant int   &V       [[buffer(2)]],
                               constant uint  &seed    [[buffer(3)]],
                               constant float &invtemp [[buffer(4)]],
                               uint row  [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    float best = SMP_NEG_INF;
    int bi = (int)lane < V ? (int)lane : 0;
    for (int i = (int)lane; i < V; i += 32) {
        const float g = rng_gumbel(seed, (uint)row, (uint)i);   // Gumbel(0,1)
        const float p = float(logits[base + i]) * invtemp + g;
        if (p > best || (p == best && i < bi)) {
            best = p;
            bi = i;
        }
    }
    simd_argmax(best, bi);
    if (lane == 0) {
        out_idx[row] = bi;
    }
}

// Top-k sampling: restrict to the k highest-logit tokens, then Gumbel-max sample
// among them (== sampling from softmax over the top-k with temperature). The top-k
// is k iterations of argmax-with-masking; the draw is reproducible from (seed, row).
template <typename T>
kernel void top_k_sample(device const T *logits  [[buffer(0)]],
                         device int     *out_idx [[buffer(1)]],
                         constant int   &V       [[buffer(2)]],
                         constant int   &K       [[buffer(3)]],
                         constant uint  &seed    [[buffer(4)]],
                         constant float &invtemp [[buffer(5)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    int chosen_id[SAMPLE_MAX_K];
    float chosen_logit[SAMPLE_MAX_K];

    for (int k = 0; k < K; ++k) {
        float best = SMP_NEG_INF;
        int bi = (int)lane < V ? (int)lane : 0;
        for (int i = (int)lane; i < V; i += 32) {
            bool taken = false;
            for (int j = 0; j < k; ++j) {
                if (chosen_id[j] == i) { taken = true; }
            }
            if (taken) { continue; }
            const float v = float(logits[base + i]);
            if (v > best || (v == best && i < bi)) {
                best = v;
                bi = i;
            }
        }
        simd_argmax(best, bi);    // all lanes hold the k-th token
        chosen_id[k] = bi;
        chosen_logit[k] = best;
    }

    // Gumbel-max among the k selected tokens.
    float best = SMP_NEG_INF;
    int bi = chosen_id[0];
    for (int j = 0; j < K; ++j) {
        const float g = rng_gumbel(seed, (uint)row, (uint)chosen_id[j]);
        const float p = chosen_logit[j] * invtemp + g;
        if (p > best || (p == best && chosen_id[j] < bi)) {
            best = p;
            bi = chosen_id[j];
        }
    }
    if (lane == 0) {
        out_idx[row] = bi;
    }
}

// Apply temperature + repetition/presence/frequency penalties to logits, given the
// generated token history. penalty_histogram builds per-row occurrence counts (P3
// atomics) over prev_tokens (T, L); apply_penalty then transforms each logit. Order
// matches vLLM: temperature, then (if seen) repetition, presence, frequency.
kernel void penalty_histogram(device const int *prev_tokens [[buffer(0)]],
                              device atomic_int *counts      [[buffer(1)]],
                              constant int &V  [[buffer(2)]],
                              constant int &L  [[buffer(3)]],
                              constant int &TL [[buffer(4)]],
                              device const int *parent_ids   [[buffer(5)]],   // (T,) history-row map
                              uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= TL) { return; }
    const int row = (int)tid / L;
    const int col = (int)tid - row * L;
    // Beam search: row's occurrence history comes from its parent beam's prev_tokens row.
    const int tok = prev_tokens[(long)parent_ids[row] * L + col];
    if (tok >= 0 && tok < V) {
        atomic_add(counts, row * V + tok, 1);   // P3
    }
}

template <typename T>
kernel void apply_penalty(device const T     *logits   [[buffer(0)]],
                          device const int   *counts   [[buffer(1)]],
                          device T           *out      [[buffer(2)]],
                          constant int   &V        [[buffer(3)]],
                          constant float &invtemp  [[buffer(4)]],
                          constant float &rep      [[buffer(5)]],
                          constant float &presence [[buffer(6)]],
                          constant float &freq     [[buffer(7)]],
                          device const float *bias [[buffer(8)]],
                          constant int   &eos_id     [[buffer(9)]],
                          constant int   &min_length [[buffer(10)]],
                          constant int   &gen_len    [[buffer(11)]],
                          uint row  [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    const bool mask_eos = (eos_id >= 0) && (gen_len < min_length);   // forbid EOS before min_length
    for (int v = (int)lane; v < V; v += 32) {
        float ls = float(logits[base + v]) * invtemp;
        const int c = counts[base + v];
        if (c > 0) {
            ls = (ls < 0.0f) ? (ls * rep) : (ls / rep);
            ls -= presence;
            ls -= freq * float(c);
        }
        ls += bias[v];                          // per-vocab logit bias
        if (mask_eos && v == eos_id) {
            ls = SMP_NEG_INF;
        }
        out[base + v] = T(ls);
    }
}

// Top-p (nucleus) sampling without a full sort: bisection on a (temperature-scaled)
// logit threshold L finds the smallest set {l >= L} whose softmax mass >= p (each
// step is one simd-reduction of the surviving mass), then Gumbel-max samples among
// those survivors. Reproducible from (seed, row). Temperature is applied before top-p.
template <typename T>
kernel void top_p_sample(device const T *logits  [[buffer(0)]],
                         device int     *out_idx [[buffer(1)]],
                         constant int   &V       [[buffer(2)]],
                         constant float &p       [[buffer(3)]],
                         constant uint  &seed    [[buffer(4)]],
                         constant float &invtemp [[buffer(5)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;

    // max (temperature-scaled) logit and softmax denominator Z (all lanes get the reductions).
    float mx = SMP_NEG_INF;
    for (int i = (int)lane; i < V; i += 32) {
        mx = max(mx, float(logits[base + i]) * invtemp);
    }
    mx = simd_max(mx);
    float Z = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        Z += exp(float(logits[base + i]) * invtemp - mx);
    }
    Z = simd_sum(Z);

    // Bisect the threshold L: keep the largest L whose mass(L)=sum_{ls>=L} softmax >= p.
    // That yields the smallest nucleus with cumulative mass >= p.
    float lo = mx - 40.0f, hi = mx;
    for (int it = 0; it < 32; ++it) {
        const float mid = 0.5f * (lo + hi);
        float sm = 0.0f;
        for (int i = (int)lane; i < V; i += 32) {
            const float ls = float(logits[base + i]) * invtemp;
            if (ls >= mid) { sm += exp(ls - mx); }
        }
        sm = simd_sum(sm) / Z;
        if (sm >= p) { lo = mid; } else { hi = mid; }
    }
    const float L = lo;

    // Gumbel-max over the nucleus {ls >= L}.
    float best = SMP_NEG_INF;
    int bi = (int)lane < V ? (int)lane : 0;
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        if (ls < L) { continue; }
        const float g = rng_gumbel(seed, (uint)row, (uint)i);
        const float pert = ls + g;
        if (pert > best || (pert == best && i < bi)) {
            best = pert;
            bi = i;
        }
    }
    simd_argmax(best, bi);
    if (lane == 0) {
        out_idx[row] = bi;
    }
}

// ---------------------------------------------------------------------------
// Beam-search advance (two stages, the TRT-LLM / FasterTransformer recipe):
//   beam_topk_partials : grid (B*BM,), one simdgroup per beam row. Computes the row's
//     log-sum-exp, then its top-2BM candidates (2BM rounds of masked simd_argmax) and
//     emits cand_score = cum_log_probs[beam] + (logit - lse), cand_token per candidate.
//   beam_select : grid (B,), one simdgroup per batch. Global top-BM over the beam's
//     BM*2BM candidates -> next_token, parent_beam, new cum_log_probs. Keeping 2BM per
//     beam guarantees the union contains the flat top-BM over (BM*V). BM <= 16 (2BM <= 32).
// ---------------------------------------------------------------------------
template <typename T>
kernel void beam_topk_partials(device const T   *logits        [[buffer(0)]],  // (B*BM, V)
                               device const float *cum_log_probs [[buffer(1)]], // (B*BM,)
                               device float     *cand_score     [[buffer(2)]],  // (B*BM, 2BM)
                               device int       *cand_token     [[buffer(3)]],  // (B*BM, 2BM)
                               constant int &V                  [[buffer(4)]],
                               constant int &two_bm             [[buffer(5)]],
                               uint row  [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    // One pass over the vocab does BOTH the log-sum-exp (per-lane online, merged across the
    // simdgroup) AND each lane's top-2BM over its strided slice. The global top-2BM is contained in
    // the union of the per-lane sets, so a 2BM-round cross-lane merge extracts it — reading the
    // vocab ONCE instead of the old 2BM masked-argmax passes.
    float m = SMP_NEG_INF, l = 0.0f;          // lse online state
    float lv[SAMPLE_MAX_K];
    int   li[SAMPLE_MAX_K];
    for (int k = 0; k < two_bm; ++k) { lv[k] = SMP_NEG_INF; li[k] = -1; }
    float minv = SMP_NEG_INF;
    int   minp = 0;
    for (int i = (int)lane; i < V; i += 32) {
        const float x = float(logits[base + i]);
        const float nm = max(m, x);
        l = l * exp(m - nm) + exp(x - nm);
        m = nm;
        if (x > minv) {                       // beats the weakest kept candidate -> replace it
            lv[minp] = x; li[minp] = i;
            minv = lv[0]; minp = 0;           // recompute the running min over the 2BM slots
            for (int k = 1; k < two_bm; ++k)
                if (lv[k] < minv) { minv = lv[k]; minp = k; }
        }
    }
    const float M = simd_max(m);
    l = simd_sum(l * exp(m - M));
    const float lse = M + log(l);
    const float cumr = cum_log_probs[row];

    bool taken[SAMPLE_MAX_K];
    for (int k = 0; k < two_bm; ++k) taken[k] = false;
    for (int r = 0; r < two_bm; ++r) {
        float bv = SMP_NEG_INF;
        int   bi = -1, bl = -1;
        for (int k = 0; k < two_bm; ++k) {
            if (taken[k]) continue;
            if (lv[k] > bv || (lv[k] == bv && li[k] < bi)) { bv = lv[k]; bi = li[k]; bl = k; }
        }
        float gv = bv;
        int   gi = (bi < 0) ? 0x7fffffff : bi;
        simd_argmax(gv, gi);                  // global winner of this round (all lanes agree)
        if (bl >= 0 && bi == gi) taken[bl] = true;   // owner removes it from its set
        if (lane == 0) {
            cand_token[(long)row * two_bm + r] = gi;
            cand_score[(long)row * two_bm + r] = cumr + (gv - lse);
        }
    }
}

kernel void beam_select(device const float *cand_score   [[buffer(0)]],  // (B*BM, 2BM)
                        device const int   *cand_token   [[buffer(1)]],  // (B*BM, 2BM)
                        device int         *next_token   [[buffer(2)]],  // (B, BM)
                        device int         *parent_beam  [[buffer(3)]],  // (B, BM)
                        device float       *new_cum      [[buffer(4)]],  // (B, BM)
                        constant int &BM                 [[buffer(5)]],
                        constant int &two_bm             [[buffer(6)]],
                        uint b    [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const int ncand = BM * two_bm;
    const long row0 = (long)b * BM;   // first beam row of batch b
    int chosen[16];                   // BM <= 16 selected flat candidate indices
    for (int k = 0; k < BM; ++k) {
        float best = SMP_NEG_INF;
        int bc = -1;
        for (int c = (int)lane; c < ncand; c += 32) {
            bool taken = false;
            for (int mchosen = 0; mchosen < k; ++mchosen) if (chosen[mchosen] == c) taken = true;
            if (taken) continue;
            const int i = c / two_bm, j = c - i * two_bm;
            const float sc = cand_score[(row0 + i) * two_bm + j];
            if (sc > best || (sc == best && c < bc)) { best = sc; bc = c; }
        }
        int gc = (bc < 0) ? 0x7fffffff : bc;
        simd_argmax(best, gc);
        chosen[k] = gc;
        if (lane == 0) {
            const int i = gc / two_bm, j = gc - i * two_bm;
            next_token[(long)b * BM + k] = cand_token[(row0 + i) * two_bm + j];
            parent_beam[(long)b * BM + k] = i;
            new_cum[(long)b * BM + k] = best;
        }
    }
}

#define instantiate_beam(type_name, T)                                          \
  template [[host_name("beam_topk_partials_" #type_name)]] [[kernel]] void       \
  beam_topk_partials<T>(device const T *logits [[buffer(0)]],                    \
    device const float *cum_log_probs [[buffer(1)]], device float *cand_score [[buffer(2)]], \
    device int *cand_token [[buffer(3)]], constant int &V [[buffer(4)]],         \
    constant int &two_bm [[buffer(5)]],                                          \
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

#define instantiate_sampling(type_name, T)                                     \
  template [[host_name("argmax_" #type_name)]] [[kernel]] void                 \
  argmax<T>(device const T *logits [[buffer(0)]],                             \
            device int *out_idx [[buffer(1)]],                                \
            constant int &V [[buffer(2)]],                                    \
            uint row [[threadgroup_position_in_grid]],                        \
            uint lane [[thread_index_in_simdgroup]]);                         \
  template [[host_name("top_k_sample_" #type_name)]] [[kernel]] void           \
  top_k_sample<T>(device const T *logits [[buffer(0)]],                       \
                  device int *out_idx [[buffer(1)]],                          \
                  constant int &V [[buffer(2)]],                              \
                  constant int &K [[buffer(3)]],                              \
                  constant uint &seed [[buffer(4)]],                          \
                  constant float &invtemp [[buffer(5)]],                      \
                  uint row [[threadgroup_position_in_grid]],                  \
                  uint lane [[thread_index_in_simdgroup]]);                   \
  template [[host_name("top_p_sample_" #type_name)]] [[kernel]] void           \
  top_p_sample<T>(device const T *logits [[buffer(0)]],                       \
                  device int *out_idx [[buffer(1)]],                          \
                  constant int &V [[buffer(2)]],                              \
                  constant float &p [[buffer(3)]],                            \
                  constant uint &seed [[buffer(4)]],                          \
                  constant float &invtemp [[buffer(5)]],                      \
                  uint row [[threadgroup_position_in_grid]],                  \
                  uint lane [[thread_index_in_simdgroup]]);                   \
  template [[host_name("apply_penalty_" #type_name)]] [[kernel]] void          \
  apply_penalty<T>(device const T *logits [[buffer(0)]],                      \
                   device const int *counts [[buffer(1)]],                    \
                   device T *out [[buffer(2)]],                               \
                   constant int &V [[buffer(3)]],                             \
                   constant float &invtemp [[buffer(4)]],                     \
                   constant float &rep [[buffer(5)]],                         \
                   constant float &presence [[buffer(6)]],                    \
                   constant float &freq [[buffer(7)]],                        \
                   device const float *bias [[buffer(8)]],                    \
                   constant int &eos_id [[buffer(9)]],                        \
                   constant int &min_length [[buffer(10)]],                   \
                   constant int &gen_len [[buffer(11)]],                      \
                   uint row [[threadgroup_position_in_grid]],                 \
                   uint lane [[thread_index_in_simdgroup]]);                  \
  template [[host_name("sample_categorical_" #type_name)]] [[kernel]] void     \
  sample_categorical<T>(device const T *logits [[buffer(0)]],                 \
                        device int *out_idx [[buffer(1)]],                    \
                        constant int &V [[buffer(2)]],                        \
                        constant uint &seed [[buffer(3)]],                    \
                        constant float &invtemp [[buffer(4)]],                \
                        uint row [[threadgroup_position_in_grid]],            \
                        uint lane [[thread_index_in_simdgroup]]);

instantiate_sampling(float32, float)
instantiate_sampling(float16, half)
instantiate_sampling(bfloat16, bf16)

instantiate_beam(float32, float)
instantiate_beam(float16, half)
instantiate_beam(bfloat16, bf16)
