#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Mixture-of-Experts routing primitives.
//
// moe_route_topk: per token, select the top-k experts by router logit and return
// their renormalized softmax weights (== softmax over just the k selected logits,
// which equals renormalizing the top-k of the full softmax — the Mixtral rule).
//
// One simdgroup (32 lanes) per token; experts E looped with stride 32. Top-k is k
// iterations of an argmax-with-index all-reduce (butterfly, so every lane holds
// the winner and can mask it out next round). K <= MOE_MAX_K.
// ---------------------------------------------------------------------------

constant float MOE_NEG_INF = -3.4028234663852886e38f;
constant int MOE_MAX_K = 16;

// Substrate reused: mittens::simd_argmax (P1), threadgroup_exclusive_scan_i32 (P2),
// atomic_add / atomic_fetch_inc (P3).

template <typename T>
kernel void moe_route_topk(device const T *logits       [[buffer(0)]],
                           device int     *topk_ids     [[buffer(1)]],
                           device float   *topk_weights [[buffer(2)]],
                           constant int   &E            [[buffer(3)]],
                           constant int   &K            [[buffer(4)]],
                           uint token [[threadgroup_position_in_grid]],
                           uint lane  [[thread_index_in_simdgroup]]) {
    const long base = (long)token * E;
    int chosen_id[MOE_MAX_K];
    float chosen_logit[MOE_MAX_K];

    for (int k = 0; k < K; ++k) {
        float best = MOE_NEG_INF;
        int bi = (int)lane < E ? (int)lane : 0;
        for (int i = (int)lane; i < E; i += 32) {
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
        simd_argmax(best, bi);   // all lanes now hold the k-th expert (P1)
        chosen_id[k] = bi;
        chosen_logit[k] = best;
    }

    // softmax over the k selected logits (= renormalized top-k of the full softmax)
    float m = MOE_NEG_INF;
    for (int k = 0; k < K; ++k) {
        m = max(m, chosen_logit[k]);
    }
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += exp(chosen_logit[k] - m);
    }
    const float inv = 1.0f / sum;

    if (lane == 0) {
        const long ob = (long)token * K;
        for (int k = 0; k < K; ++k) {
            topk_ids[ob + k] = chosen_id[k];
            topk_weights[ob + k] = exp(chosen_logit[k] - m) * inv;
        }
    }
}

// ---------------------------------------------------------------------------
// MoE permute pipeline (P3 atomics + offset scan). Groups the T*K (token, k-slot)
// routing rows by expert id so each expert's tokens are contiguous for the GEMM.
//
//   histogram: counts[e]  via atomic add over the T*K expert ids
//   scan:      offsets[e] = exclusive prefix sum of counts (offsets[E] = T*K)
//   scatter:   pos = atomic_add(cursor[e]); sorted_row_idx[pos] = r ; inv_idx[r] = pos
//
// `r` in [0, T*K) is a flat (token=r/K, k-slot=r%K) routing row. The inverse map
// inv_idx lets the finalize step do its k-way weighted reduce with no atomics.
// (E is small, so the scan is a single-thread serial prefix sum — exact.)
// ---------------------------------------------------------------------------

kernel void moe_zero_i32(device int *p [[buffer(0)]],
                         constant int &n [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
    if ((int)tid < n) { p[tid] = 0; }
}

kernel void moe_histogram(device const int *topk_ids [[buffer(0)]],
                          device atomic_int *counts  [[buffer(1)]],
                          constant int &TK [[buffer(2)]],
                          uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= TK) { return; }
    atomic_add(counts, topk_ids[tid], 1);   // P3
}

// Single-thread exclusive prefix sum over E experts; also seeds the scatter cursor.
// Parallel exclusive prefix sum of the per-expert counts (P2 substrate scan), with a
// running prefix across tiles so any E is supported. offsets[e] = sum(counts[0..e-1]);
// offsets[E] = total; cursor seeded to offsets for the scatter. One threadgroup.
constant uint MOE_SCAN_NT = 256;

kernel void moe_scan_offsets(device const int *counts  [[buffer(0)]],
                             device int       *offsets [[buffer(1)]],   // (E+1)
                             device int       *cursor  [[buffer(2)]],   // (E)
                             constant int &E [[buffer(3)]],
                             uint tid [[thread_position_in_threadgroup]]) {
    threadgroup int sg_sums[MOE_SCAN_NT / 32];
    threadgroup int running;
    if (tid == 0) { running = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int b = 0; b < E; b += (int)MOE_SCAN_NT) {
        const int e = b + (int)tid;
        const int v = (e < E) ? counts[e] : 0;
        int total;
        const int excl = threadgroup_exclusive_scan_i32(v, tid, MOE_SCAN_NT, sg_sums, total);
        if (e < E) {
            offsets[e] = running + excl;
            cursor[e] = running + excl;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { running += total; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { offsets[E] = running; }
}

kernel void moe_scatter(device const int *topk_ids       [[buffer(0)]],
                        device atomic_int *cursor        [[buffer(1)]],
                        device int       *sorted_row_idx [[buffer(2)]],
                        device int       *inv_idx        [[buffer(3)]],
                        constant int &TK [[buffer(4)]],
                        uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= TK) { return; }
    const int pos = atomic_fetch_inc(cursor, topk_ids[tid]);   // P3 atomic cursor
    sorted_row_idx[pos] = (int)tid;
    inv_idx[tid] = pos;
}

// Finalize: per token, weighted k-way reduce of the expert outputs (permuted order),
// gathered back via inv_idx. No atomics — each token owns its K contributions.
// expert_out (T*K, Hdim) in permuted order; weights (T, K); out (T, Hdim).
template <typename T>
kernel void moe_finalize(device const T     *expert_out   [[buffer(0)]],
                         device const int   *inv_idx      [[buffer(1)]],
                         device const float *topk_weights [[buffer(2)]],
                         device T           *out          [[buffer(3)]],
                         constant int &K [[buffer(4)]],
                         constant int &Hdim [[buffer(5)]],
                         uint token [[threadgroup_position_in_grid]],
                         uint lane  [[thread_index_in_simdgroup]]) {
    const long wbase = (long)token * K;
    const long obase = (long)token * Hdim;
    for (int h = (int)lane; h < Hdim; h += 32) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int pos = inv_idx[token * K + k];
            acc += topk_weights[wbase + k] * float(expert_out[(long)pos * Hdim + h]);
        }
        out[obase + h] = T(acc);
    }
}

// ---------------------------------------------------------------------------
// Fused grouped (segmented) expert GEMM: out = permuted_input @ W[expert].
// Rows are grouped by expert with each expert's segment padded to a 32-multiple
// (moe_align pattern), so every 32-row output tile belongs to exactly one expert
// and the unmasked full-tile load/store/mma apply verbatim. Copy of matmul_custom
// (<4,2,4> -> 32x32 tile, K-step 16, fp32 accumulate) with a per-expert W base
// pointer and an expert_of_tile lookup. permuted_input/out are (total_rows, H);
// W is (E, H, H); expert_of_tile is (total_rows/32,). Requires H % 32 == 0.
// ---------------------------------------------------------------------------
template <typename T, unsigned N_BLOCK, unsigned K_BLOCK, unsigned M_BLOCK>
kernel void moe_grouped_gemm(device T *out                       [[buffer(0)]],
                             device T *A                         [[buffer(1)]],
                             device T *W                         [[buffer(2)]],
                             device const int *expert_of_tile    [[buffer(3)]],
                             constant int &total_rows            [[buffer(4)]],
                             constant int &H                     [[buffer(5)]],
                             uint3 threadgroup_id [[threadgroup_position_in_grid]],
                             uint  simd_lane_id   [[thread_index_in_simdgroup]]) {
    const int OY = (int)threadgroup_id.y;   // global row-tile (32 rows)
    const int OX = (int)threadgroup_id.x;   // output column-tile (in H)
    const int e = expert_of_tile[OY];

    using global_layout = gl<T, 1, 1, -1, -1>;
    global_layout gl_a(A, nullptr, nullptr, total_rows, H);
    global_layout gl_w(W + (long)e * H * H, nullptr, nullptr, H, H);
    global_layout gl_d(out, nullptr, nullptr, total_rows, H);

    constexpr const int N_BE = N_BLOCK * TILE_DIM;   // 32
    constexpr const int M_BE = M_BLOCK * TILE_DIM;   // 32
    constexpr const int K_BE = K_BLOCK * TILE_DIM;   // 16
    rt<T, N_BE, K_BE> a_reg;
    rt<T, K_BE, M_BE> b_reg;
    rt<float, N_BE, M_BE> d_reg;
    zero(d_reg);
    #pragma clang loop unroll(full)
    for (int k = 0; k < H / K_BE; k++) {
        load(a_reg, gl_a, {0, 0, OY, k}, simd_lane_id);
        load(b_reg, gl_w, {0, 0, k, OX}, simd_lane_id);
        mma_AB(d_reg, a_reg, b_reg, d_reg);
    }
    store(gl_d, d_reg, {0, 0, OY, OX}, simd_lane_id);
}

#define instantiate_moe_grouped_gemm(type_name, T)                             \
  template [[host_name("moe_grouped_gemm_" #type_name)]] [[kernel]] void        \
  moe_grouped_gemm<T, 4, 2, 4>(device T *out [[buffer(0)]],                     \
                               device T *A [[buffer(1)]],                       \
                               device T *W [[buffer(2)]],                       \
                               device const int *expert_of_tile [[buffer(3)]],  \
                               constant int &total_rows [[buffer(4)]],          \
                               constant int &H [[buffer(5)]],                   \
                               uint3 threadgroup_id [[threadgroup_position_in_grid]], \
                               uint simd_lane_id [[thread_index_in_simdgroup]]);

instantiate_moe_grouped_gemm(float32, float)
instantiate_moe_grouped_gemm(bfloat16, bf16)

#define instantiate_moe_finalize(type_name, T)                                 \
  template [[host_name("moe_finalize_" #type_name)]] [[kernel]] void           \
  moe_finalize<T>(device const T *expert_out [[buffer(0)]],                    \
                  device const int *inv_idx [[buffer(1)]],                     \
                  device const float *topk_weights [[buffer(2)]],              \
                  device T *out [[buffer(3)]],                                 \
                  constant int &K [[buffer(4)]],                               \
                  constant int &Hdim [[buffer(5)]],                            \
                  uint token [[threadgroup_position_in_grid]],                 \
                  uint lane [[thread_index_in_simdgroup]]);

instantiate_moe_finalize(float32, float)
instantiate_moe_finalize(float16, half)
instantiate_moe_finalize(bfloat16, bf16)

#define instantiate_moe_route(type_name, T)                                    \
  template [[host_name("moe_route_topk_" #type_name)]] [[kernel]] void         \
  moe_route_topk<T>(device const T *logits [[buffer(0)]],                      \
                    device int *topk_ids [[buffer(1)]],                        \
                    device float *topk_weights [[buffer(2)]],                  \
                    constant int &E [[buffer(3)]],                             \
                    constant int &K [[buffer(4)]],                             \
                    uint token [[threadgroup_position_in_grid]],               \
                    uint lane [[thread_index_in_simdgroup]]);

instantiate_moe_route(float32, float)
instantiate_moe_route(float16, half)
instantiate_moe_route(bfloat16, bf16)
