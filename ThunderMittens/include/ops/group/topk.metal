/**
 * @file
 * @brief Masked top-k selection over a simdgroup (the "Family-A" idiom).
 *
 * Several kernels (top-k sampling, MoE routing, beam select, LM-head top-k reduce) all pick the K
 * largest candidates by running K rounds of `simd_argmax`, masking out the winner of each round so
 * the next round finds the next-largest. They differ ONLY in how a candidate's (id, value, valid)
 * is read at a given scan index — so that read is abstracted into a small functor and the K-round
 * masked-argmax loop lives here once. Ties break toward the smaller id (numpy-argmax semantics),
 * matching `mittens::simd_argmax`.
 *
 * Free function (mirrors scan.metal / bitmask.metal): included from ops/ops.metal AFTER warp.metal
 * so `simd_argmax` is visible; the candidate functors are file-scope structs, not lambdas (MSL has
 * no lambdas).
 */

#pragma once

#include "../warp/register/vec/reductions.metal"   // mittens::simd_argmax (standalone-safe, pragma-once)

namespace mittens {

/**
 * @brief K rounds of masked-`simd_argmax` top-k selection over `n` candidates.
 *
 * `cand(idx, id, val, valid)` (thread-ref outputs) reports, for each scan index idx in [0,n), the
 * candidate's integer id, its float value, and whether it participates. Each round skips ids already
 * in `chosen[0..kk)`, reduces to the max (ties -> smaller id), and writes the winner into
 * `chosen[kk]` / `chosen_val[kk]`. A round with no valid unmasked candidate writes `chosen[kk] = -1`
 * and `chosen_val[kk] = neg_inf`. On return every lane holds the same `chosen[]` / `chosen_val[]`.
 * Caller supplies the sentinel (each site has its own NEG_INF constant).
 */
template <typename FN>
static METAL_FUNC void masked_topk(thread FN &cand, int n, int K, uint lane, float neg_inf,
                                   thread int *chosen, thread float *chosen_val) {
    for (int kk = 0; kk < K; ++kk) {
        float best = neg_inf;
        int   bi   = -1;
        for (int idx = (int)lane; idx < n; idx += 32) {
            int id; float v; bool valid;
            cand(idx, id, v, valid);
            if (!valid) { continue; }
            bool taken = false;
            for (int m = 0; m < kk; ++m) { if (chosen[m] == id) { taken = true; } }
            if (taken) { continue; }
            if (v > best || (v == best && id < bi)) { best = v; bi = id; }
        }
        int gid = (bi < 0) ? 0x7fffffff : bi;
        simd_argmax(best, gid);
        chosen[kk]     = (best == neg_inf) ? -1 : gid;
        chosen_val[kk] = best;
    }
}

/** Candidate accessor: id == scan index, value = float(arr[base + idx]), always valid.
 *  Used by top-k sampling and MoE routing (dense per-row logit scan). */
template <typename T>
struct indexed_cand {
    device const T *arr;
    long base;
    METAL_FUNC void operator()(int idx, thread int &id, thread float &v, thread bool &valid) const {
        id = idx;
        v = float(arr[base + idx]);
        valid = true;
    }
};

/** Candidate accessor: id read from a parallel id array (negative id = absent), value from a value
 *  array, both at base + idx. Used by the LM-head top-k reduce (merging per-tile partial winners). */
struct stored_cand {
    device const float *val;
    device const int   *ids;
    long base;
    METAL_FUNC void operator()(int idx, thread int &id, thread float &v, thread bool &valid) const {
        id = ids[base + idx];
        valid = id >= 0;
        v = val[base + idx];
    }
};

} // namespace mittens
