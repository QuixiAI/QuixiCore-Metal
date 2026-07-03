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

    // K masked-argmax rounds over the full vocab (Family-A helper).
    indexed_cand<T> cand{logits, base};
    masked_topk(cand, V, K, lane, SMP_NEG_INF, chosen_id, chosen_logit);

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

// min-p sampling: keep only tokens whose (tempered) softmax prob >= min_p * max_prob, then
// Gumbel-max sample among them. In logit space the keep test is exactly
//   logits[v]*invtemp >= max_v(logits*invtemp) + log(min_p)
// (the softmax normalizer cancels), so it is a single max-reduce + one masked Gumbel-max pass.
// Ref: flashinfer MinPOp (mask probs >= min_p*max(probs), renormalize).
template <typename T>
kernel void min_p_sample(device const T *logits  [[buffer(0)]],
                         device int     *out_idx [[buffer(1)]],
                         constant int   &V       [[buffer(2)]],
                         constant float &min_p   [[buffer(3)]],
                         constant uint  &seed    [[buffer(4)]],
                         constant float &invtemp [[buffer(5)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    float m = SMP_NEG_INF;
    for (int i = (int)lane; i < V; i += 32) {
        m = max(m, float(logits[base + i]) * invtemp);
    }
    m = simd_max(m);                                    // row max of the tempered logits
    const float thresh = m + metal::log(min_p);
    float best = SMP_NEG_INF;
    int bi = (int)lane < V ? (int)lane : 0;
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        if (ls < thresh) { continue; }
        const float pert = ls + rng_gumbel(seed, (uint)row, (uint)i);
        if (pert > best || (pert == best && i < bi)) { best = pert; bi = i; }
    }
    simd_argmax(best, bi);
    if (lane == 0) { out_idx[row] = bi; }
}

// Typical-p (locally-typical) sampling: keep the tokens whose "surprise" |(-log p_v) - H| is
// smallest until their cumulative prob reaches typical_p, then Gumbel-max sample among them. H is
// the row entropy. With p_v = exp(ls_v - mx)/Z: -log p_v = mx + logZ - ls_v, and
// H = -sum p_v log p_v = mx + logZ - sum p_v ls_v. We bisect a surprise threshold tau (mirroring
// top_p's threshold search): smallest tau s.t. mass{s_v <= tau} >= typical_p. One simdgroup/row.
// Ref: HuggingFace TypicalLogitsWarper (from definition; no local kernel reference).
template <typename T>
kernel void typical_p_sample(device const T *logits  [[buffer(0)]],
                             device int     *out_idx [[buffer(1)]],
                             constant int   &V       [[buffer(2)]],
                             constant float &typ_p   [[buffer(3)]],
                             constant uint  &seed    [[buffer(4)]],
                             constant float &invtemp [[buffer(5)]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    float mx = SMP_NEG_INF;
    for (int i = (int)lane; i < V; i += 32) { mx = max(mx, float(logits[base + i]) * invtemp); }
    mx = simd_max(mx);
    float Z = 0.0f, S1 = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        const float e = exp(ls - mx);
        Z += e;
        S1 += e * ls;                          // unnormalized sum e_v * ls_v
    }
    Z = simd_sum(Z);
    S1 = simd_sum(S1);
    const float logZ = metal::log(Z);
    const float H = mx + logZ - S1 / Z;        // row entropy
    // max surprise (upper bisection bound); mass{s<=smax} == 1 >= typ_p always.
    float smax = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        smax = max(smax, metal::abs((mx + logZ - ls) - H));
    }
    smax = simd_max(smax);
    // Smallest tau with mass{s_v <= tau} >= typ_p (mass is monotone increasing in tau).
    float lo = 0.0f, hi = smax;
    for (int it = 0; it < 32; ++it) {
        const float mid = 0.5f * (lo + hi);
        float mass = 0.0f;
        for (int i = (int)lane; i < V; i += 32) {
            const float ls = float(logits[base + i]) * invtemp;
            if (metal::abs((mx + logZ - ls) - H) <= mid) { mass += exp(ls - mx); }
        }
        mass = simd_sum(mass) / Z;
        if (mass >= typ_p) { hi = mid; } else { lo = mid; }
    }
    const float tau = hi;
    // Gumbel-max over the kept set {s_v <= tau}.
    float best = SMP_NEG_INF;
    int bi = (int)lane < V ? (int)lane : 0;
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        if (metal::abs((mx + logZ - ls) - H) > tau) { continue; }
        const float pert = ls + rng_gumbel(seed, (uint)row, (uint)i);
        if (pert > best || (pert == best && i < bi)) { best = pert; bi = i; }
    }
    simd_argmax(best, bi);
    if (lane == 0) { out_idx[row] = bi; }
}

// Grammar / structured-output masking: set logits[v] = -inf wherever the packed allow-bitmask bit
// for token v is 0 (word = bitmask[v>>5]; bit = (word >> (v&31)) & 1). One simdgroup per row; the
// bitmask is (num_tokens, ceil(V/32)) uint32. Composes before any sampler (like apply_penalty).
// Ref: sglang apply_token_bitmask_inplace_cuda.cu.
template <typename T>
kernel void apply_token_bitmask(device const T    *logits    [[buffer(0)]],   // (num_tokens, V)
                                device const uint *bitmask   [[buffer(1)]],   // (num_tokens, num_words)
                                device T          *out       [[buffer(2)]],
                                constant int      &V         [[buffer(3)]],
                                constant int      &num_words [[buffer(4)]],
                                uint row  [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const long lbase = (long)row * V;
    const long mbase = (long)row * num_words;
    device const uint *mask_row = bitmask + mbase;
    for (int v = (int)lane; v < V; v += 32) {
        out[lbase + v] = token_allowed(mask_row, v) ? logits[lbase + v] : T(SMP_NEG_INF);
    }
}

// Bad / stop-word masking: out = logits, then out[t, bad_ids[t,j]] = -inf for each j < bad_lens[t].
// One simdgroup per row: copy the row, simdgroup-barrier, then scatter -inf at the row's bad ids
// (the barrier orders the copy before the scatter so a bad id isn't overwritten by a late copy).
template <typename T>
kernel void apply_bad_words(device const T   *logits   [[buffer(0)]],   // (num_tokens, V)
                            device const int *bad_ids  [[buffer(1)]],   // (num_tokens, maxbad)
                            device const int *bad_lens [[buffer(2)]],   // (num_tokens,)
                            device T         *out      [[buffer(3)]],
                            constant int     &V        [[buffer(4)]],
                            constant int     &maxbad   [[buffer(5)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const long lbase = (long)row * V;
    for (int v = (int)lane; v < V; v += 32) { out[lbase + v] = logits[lbase + v]; }
    metal::simdgroup_barrier(metal::mem_flags::mem_device);
    const int nb = bad_lens[row];
    for (int j = (int)lane; j < nb; j += 32) {
        const int bid = bad_ids[(long)row * maxbad + j];
        if (bid >= 0 && bid < V) { out[lbase + bid] = T(SMP_NEG_INF); }
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
    int   chosen[16];                 // BM <= 16 selected flat candidate indices
    float chosen_sc[16];              // ... and their scores (new_cum)
    // The score of flat candidate c is cand_score[(row0 + c/two_bm)*two_bm + c%two_bm] ==
    // cand_score[row0*two_bm + c], so this is a plain indexed scan (Family-A helper).
    indexed_cand<float> cand{cand_score, row0 * two_bm};
    masked_topk(cand, ncand, BM, lane, SMP_NEG_INF, chosen, chosen_sc);
    if (lane == 0) {
        for (int k = 0; k < BM; ++k) {
            const int gc = chosen[k];
            const int i = gc / two_bm, j = gc - i * two_bm;
            next_token[(long)b * BM + k] = cand_token[(row0 + i) * two_bm + j];
            parent_beam[(long)b * BM + k] = i;
            new_cum[(long)b * BM + k] = chosen_sc[k];
        }
    }
}

// ---------------------------------------------------------------------------
// Speculative decoding: linear (non-tree) rejection-sampling verification, the vLLM contract.
// For each request b, walk the S draft tokens in order. Draft token dt=draft_tokens[b,i] is
// ACCEPTED iff u <= p_target/p_draft (u = accept_u[b,i]); on the first rejection, emit a "recovered"
// token sampled from the residual distribution (p_target - p_draft)+ and stop. If ALL S drafts are
// accepted, append the bonus token. out_tokens[b] = [accepted..., recovered | bonus, PLACEHOLDER...];
// accepted_cnt[b] = number of accepted drafts. The residual sample uses the same Gumbel-max trick as
// sample_categorical (argmax over log((p_t-p_d)+) + Gumbel), which draws proportional to the residual.
// One simdgroup per request; the accept decision is simdgroup-uniform so the Gumbel-max is convergent.
// Ref: vLLM v1/sample/rejection_sampler.py.
// ---------------------------------------------------------------------------
constant int SPEC_PLACEHOLDER = -1;

kernel void spec_verify_linear(device const int   *draft_tokens [[buffer(0)]],  // (B, S)
                               device const float *draft_probs  [[buffer(1)]],  // (B, S, V)
                               device const float *target_probs [[buffer(2)]],  // (B, S+1, V)
                               device const int   *bonus_tokens [[buffer(3)]],  // (B,)
                               device const float *accept_u     [[buffer(4)]],  // (B, S)
                               device int         *out_tokens   [[buffer(5)]],  // (B, S+1)
                               device int         *accepted_cnt [[buffer(6)]],  // (B,)
                               constant int  &S    [[buffer(7)]],
                               constant int  &V    [[buffer(8)]],
                               constant uint &seed [[buffer(9)]],
                               uint bidx [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const int b = (int)bidx;
    int rejected_at = S;                              // stays S == all drafts accepted
    for (int i = 0; i < S; ++i) {
        const int dt = draft_tokens[b * S + i];
        const long tbase = ((long)b * (S + 1) + i) * V;
        const long dbase = ((long)b * S + i) * V;
        const float p_t = target_probs[tbase + dt];
        const float p_d = draft_probs[dbase + dt];
        const float u = accept_u[b * S + i];
        const bool accept = (p_d <= 0.0f) ? true : (u * p_d <= p_t);   // u <= p_t/p_d
        if (accept) {
            if (lane == 0) out_tokens[b * (S + 1) + i] = dt;
            continue;
        }
        // recovered token ~ (p_t - p_d)+  via Gumbel-max (== sample_categorical over log residual)
        float best = SMP_NEG_INF;
        int bi = 0;
        for (int v = (int)lane; v < V; v += 32) {
            const float r = max(0.0f, target_probs[tbase + v] - draft_probs[dbase + v]);
            const float logit = (r > 0.0f) ? metal::log(r) : SMP_NEG_INF;
            const float g = logit + rng_gumbel(seed, (uint)(b * S + i), (uint)v);
            if (g > best || (g == best && v < bi)) { best = g; bi = v; }
        }
        simd_argmax(best, bi);
        if (lane == 0) out_tokens[b * (S + 1) + i] = bi;
        rejected_at = i;
        break;
    }
    if (rejected_at == S) {                            // all accepted -> bonus at position S
        if (lane == 0) out_tokens[b * (S + 1) + S] = bonus_tokens[b];
    } else {                                           // positions after the recovered token: empty
        for (int i = rejected_at + 1; i <= S; ++i)
            if (lane == 0) out_tokens[b * (S + 1) + i] = SPEC_PLACEHOLDER;
    }
    if (lane == 0) accepted_cnt[b] = rejected_at;
}

// Speculative TREE verification (target-only rejection, the TRT-LLM dynamicTree contract). The draft
// tree has N nodes (node 0 = root = last accepted token); node c>=1 carries draft token
// draft_tokens[c-1]; retrieve_next_token[i] = i's first child (-1 leaf), retrieve_next_sibling[i] =
// i's next sibling (-1 none). target_probs[i] is the target dist AT node i's position. Walk depth by
// depth from the root: draw a coin, accumulate the target prob of the sibling candidate tokens, and
// accept the FIRST sibling whose cumulative prob exceeds the coin (== sampling the target restricted
// to the sibling tokens); on that acceptance descend into the sibling. If every sibling is rejected,
// emit a correction token sampled from the residual target mass (target with the tried sibling
// tokens removed); at a leaf, emit the bonus token sampled from the full target at the last node.
// One simdgroup per request: lane 0 walks the (cheap, serial) tree; all 32 lanes cooperatively do
// the single full-vocab terminal sample via Gumbel-max (== proportional sampling, no cumsum scan).
// Outputs accept_index (B,N) tree positions (-1 pad), accept_token (B,N) token ids (-1 pad),
// accept_num (B,) # accepted drafts. tree_valid (B,) int: when 0 (first-gen / no tree exists) the
// request skips the walk and samples the token from the target root, accept_num=0 (TRT-LLM contract).
// The residual (all siblings rejected) excludes ALL children of `last` by re-walking the child chain
// in the terminal sample — exact for any sibling count, no cap. Ref: TRT-LLM verifyDynamicTreeRejectionKernel.
kernel void spec_verify_tree(device const int   *draft_tokens         [[buffer(0)]],  // (B, N-1)
                             device const float *target_probs         [[buffer(1)]],  // (B, N, V)
                             device const int   *retrieve_next_token   [[buffer(2)]],  // (B, N)
                             device const int   *retrieve_next_sibling [[buffer(3)]],  // (B, N)
                             device int         *accept_index         [[buffer(4)]],  // (B, N)
                             device int         *accept_token         [[buffer(5)]],  // (B, N)
                             device int         *accept_num           [[buffer(6)]],  // (B,)
                             constant int  &N          [[buffer(7)]],
                             constant int  &V          [[buffer(8)]],
                             constant uint &seed       [[buffer(9)]],
                             device const int *tree_valid [[buffer(10)]],  // (B,) 0 = no tree
                             uint b    [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int s_num_accepted, s_last, s_term;
    const long nbase = (long)b * N;
    for (int i = (int)lane; i < N; i += 32) {           // sentinel-fill the outputs
        accept_index[nbase + i] = -1;
        accept_token[nbase + i] = -1;
    }
    simdgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (lane == 0) {
        if (tree_valid[b] == 0) {                             // no tree: sample the target root token
            accept_index[nbase] = 0;
            s_num_accepted = 0; s_last = 0; s_term = 1;       // term 1 = full-target sample at node 0
        } else {
            int last = 0, num_acc = 0, term = 0;              // term: 0 none, 1 bonus(leaf), 2 residual
            accept_index[nbase] = 0;
            for (int j = 1; j < N; ++j) {
                const int firstChild = retrieve_next_token[nbase + last];
                if (firstChild == -1) { term = 1; break; }    // leaf -> bonus at `last`
                const float coin = rng_uniform(seed, (uint)b, (uint)j);
                float probAcc = 0.0f;
                bool accepted = false;
                int child = firstChild;
                device const float *parentProbs = target_probs + (nbase + last) * (long)V;
                while (child != -1) {
                    const int tok = draft_tokens[(long)b * (N - 1) + (child - 1)];
                    probAcc += parentProbs[tok];
                    if (coin <= probAcc) {
                        accept_token[nbase + num_acc] = tok;
                        num_acc += 1;
                        accept_index[nbase + num_acc] = child;
                        last = child;
                        accepted = true;
                        break;
                    }
                    child = retrieve_next_sibling[nbase + child];
                }
                if (!accepted) { term = 2; break; }           // residual at `last`, excluding all children
            }
            s_num_accepted = num_acc; s_last = last; s_term = term;
        }
    }
    simdgroup_barrier(metal::mem_flags::mem_threadgroup);

    const int term = s_term, last = s_last, num_acc = s_num_accepted;
    if (term != 0) {                                          // cooperative full-vocab terminal sample
        device const float *tp = target_probs + (nbase + last) * (long)V;
        const int firstChild = (term == 2) ? retrieve_next_token[nbase + last] : -1;  // residual siblings
        float best = SMP_NEG_INF;
        int   bi   = -1;
        for (int v = (int)lane; v < V; v += 32) {
            const float pv = tp[v];
            if (pv <= 0.0f) { continue; }
            if (term == 2) {                                  // residual: skip ANY child (tried) token
                bool tried = false;
                for (int c = firstChild; c != -1; c = retrieve_next_sibling[nbase + c]) {
                    if (draft_tokens[(long)b * (N - 1) + (c - 1)] == v) { tried = true; break; }
                }
                if (tried) { continue; }
            }
            const float g = metal::log(pv) + rng_gumbel(seed + 0x2545F491u, (uint)b, (uint)v);
            if (g > best || (g == best && v < bi)) { best = g; bi = v; }
        }
        float gb = best;
        int   gi = (bi < 0) ? 0x7fffffff : bi;
        simd_argmax(gb, gi);
        if (lane == 0) { accept_token[nbase + num_acc] = (gb == SMP_NEG_INF) ? -1 : gi; }
    }
    if (lane == 0) { accept_num[b] = num_acc; }
}

// spec_compact: gather each request's valid tokens (accepted drafts + the recovered/bonus token,
// vlen = accepted_cnt+1) from out_tokens (B, S+1) into a packed buffer with cu_accepted offsets
// (exclusive scan of vlen). packed_pos[k] = seq_lens[b] + j is the absolute KV position of the
// j-th token. packed_* are sized B*(S+1) (upper bound); cu_accepted[B] holds the real total, unused
// tail is -1. Single-threadgroup CHUNKED scan (each thread owns a contiguous batch chunk, so ANY B
// works — not capped by the thread count). Ref: vLLM rejection_sampler parse_output.
kernel void spec_compact(device const int *out_tokens   [[buffer(0)]],   // (B, S+1)
                         device const int *accepted_cnt [[buffer(1)]],   // (B,)
                         device const int *seq_lens     [[buffer(2)]],   // (B,)
                         device int *packed_tokens      [[buffer(3)]],   // (B*(S+1),)
                         device int *packed_pos         [[buffer(4)]],   // (B*(S+1),)
                         device int *cu_accepted        [[buffer(5)]],   // (B+1,)
                         constant int &B                [[buffer(6)]],
                         constant int &Sp1              [[buffer(7)]],    // S+1
                         uint tid [[thread_index_in_threadgroup]],
                         uint nthreads [[threads_per_threadgroup]]) {
    threadgroup int sg_sums[8];    // nthreads/32 <= 8
    const int chunk = (B + (int)nthreads - 1) / (int)nthreads;
    int lo = (int)tid * chunk; if (lo > B) { lo = B; }
    int hi = lo + chunk;       if (hi > B) { hi = B; }
    int local_vlen = 0;                                  // pass 1: per-thread chunk total
    for (int b = lo; b < hi; ++b) { local_vlen += accepted_cnt[b] + 1; }
    int total = 0;
    const int base = mittens::threadgroup_exclusive_scan_i32(local_vlen, tid, nthreads, sg_sums, total);
    if (tid == 0) { cu_accepted[B] = total; }
    int run = base;                                      // pass 2: re-walk chunk emitting from base
    for (int b = lo; b < hi; ++b) {
        const int vlen = accepted_cnt[b] + 1;
        const int sl = seq_lens[b];
        cu_accepted[b] = run;
        for (int j = 0; j < vlen; ++j) {                 // disjoint range [run, run+vlen)
            packed_tokens[run + j] = out_tokens[(long)b * Sp1 + j];
            packed_pos[run + j]    = sl + j;
        }
        run += vlen;
    }
    const int cap = B * Sp1;
    for (int i = (int)tid; i < cap; i += (int)nthreads) {  // sentinel-fill the unused tail
        if (i >= total) { packed_tokens[i] = -1; packed_pos[i] = -1; }
    }
}

// spec_update_kv_meta: new_seq_lens[b] = seq_lens[b] + accepted_cnt[b] + 1 (post-verify KV length).
kernel void spec_update_kv_meta(device const int *seq_lens     [[buffer(0)]],  // (B,)
                                device const int *accepted_cnt [[buffer(1)]],  // (B,)
                                device int *new_seq_lens       [[buffer(2)]],  // (B,)
                                constant int &B                [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
    if ((int)gid < B) { new_seq_lens[gid] = seq_lens[gid] + accepted_cnt[gid] + 1; }
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
                        uint lane [[thread_index_in_simdgroup]]);            \
  template [[host_name("min_p_sample_" #type_name)]] [[kernel]] void           \
  min_p_sample<T>(device const T *logits [[buffer(0)]],                       \
                  device int *out_idx [[buffer(1)]],                          \
                  constant int &V [[buffer(2)]],                              \
                  constant float &min_p [[buffer(3)]],                        \
                  constant uint &seed [[buffer(4)]],                          \
                  constant float &invtemp [[buffer(5)]],                      \
                  uint row [[threadgroup_position_in_grid]],                  \
                  uint lane [[thread_index_in_simdgroup]]);                   \
  template [[host_name("typical_p_sample_" #type_name)]] [[kernel]] void       \
  typical_p_sample<T>(device const T *logits [[buffer(0)]],                   \
                  device int *out_idx [[buffer(1)]],                          \
                  constant int &V [[buffer(2)]],                              \
                  constant float &typ_p [[buffer(3)]],                        \
                  constant uint &seed [[buffer(4)]],                          \
                  constant float &invtemp [[buffer(5)]],                      \
                  uint row [[threadgroup_position_in_grid]],                  \
                  uint lane [[thread_index_in_simdgroup]]);                   \
  template [[host_name("apply_token_bitmask_" #type_name)]] [[kernel]] void    \
  apply_token_bitmask<T>(device const T *logits [[buffer(0)]],                \
                         device const uint *bitmask [[buffer(1)]],            \
                         device T *out [[buffer(2)]],                         \
                         constant int &V [[buffer(3)]],                       \
                         constant int &num_words [[buffer(4)]],               \
                         uint row [[threadgroup_position_in_grid]],           \
                         uint lane [[thread_index_in_simdgroup]]);            \
  template [[host_name("apply_bad_words_" #type_name)]] [[kernel]] void        \
  apply_bad_words<T>(device const T *logits [[buffer(0)]],                    \
                     device const int *bad_ids [[buffer(1)]],                 \
                     device const int *bad_lens [[buffer(2)]],                \
                     device T *out [[buffer(3)]],                             \
                     constant int &V [[buffer(4)]],                           \
                     constant int &maxbad [[buffer(5)]],                      \
                     uint row [[threadgroup_position_in_grid]],               \
                     uint lane [[thread_index_in_simdgroup]]);

instantiate_sampling(float32, float)
instantiate_sampling(float16, half)
instantiate_sampling(bfloat16, bf16)

instantiate_beam(float32, float)
instantiate_beam(float16, half)
instantiate_beam(bfloat16, bf16)
