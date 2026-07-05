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

    // K masked-argmax rounds over the E experts (Family-A helper).
    indexed_cand<T> cand{logits, base};
    masked_topk(cand, E, K, lane, MOE_NEG_INF, chosen_id, chosen_logit);

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

// ---------------------------------------------------------------------------
// MoE padded schedule (the GPU replacement for the host-side glue): turns the compact
// per-expert layout from moe_permute into 32-row-padded segments the grouped GEMMs consume.
// MLX needs static shapes, so the padded space is allocated at the worst case
// total_pad_max = ceil32(T*K + 31*E) and marked with -1 sentinels beyond the real total:
//   expert_of_tile[t] = -1  -> the grouped GEMMs early-exit that tile
//   gather_idx[p]     = -1  -> moe_gather zero-fills that padded row
// (Mirrors vLLM's moe_align_block_size / TRT-LLM expandInputRowsKernel.)
// ---------------------------------------------------------------------------

// Single threadgroup: off_pad(E+1) = exclusive scan of ceil32(counts) (counts derived from
// the unpadded offsets), then fill expert_of_tile via binary search over off_pad and init
// gather_idx to -1. Empty experts produce zero-width segments the search skips naturally.
kernel void moe_pad_offsets(device const int *offsets        [[buffer(0)]],   // (E+1) unpadded
                            device int       *off_pad        [[buffer(1)]],   // (E+1)
                            device int       *expert_of_tile [[buffer(2)]],   // (max_tiles)
                            device int       *gather_idx     [[buffer(3)]],   // (total_pad_max)
                            constant int &E [[buffer(4)]],
                            constant int &max_tiles [[buffer(5)]],
                            constant int &total_pad_max [[buffer(6)]],
                            uint tid [[thread_position_in_threadgroup]]) {
    threadgroup int sg_sums[MOE_SCAN_NT / 32];
    threadgroup int running;
    if (tid == 0) { running = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int b = 0; b < E; b += (int)MOE_SCAN_NT) {
        const int e = b + (int)tid;
        const int count = (e < E) ? (offsets[e + 1] - offsets[e]) : 0;
        const int padded = ((count + 31) / 32) * 32;
        int total;
        const int excl = threadgroup_exclusive_scan_i32(padded, tid, MOE_SCAN_NT, sg_sums, total);
        if (e < E) { off_pad[e] = running + excl; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { running += total; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { off_pad[E] = running; }
    threadgroup_barrier(mem_flags::mem_device);

    const int total_pad = off_pad[E];
    for (int t = (int)tid; t < max_tiles; t += (int)MOE_SCAN_NT) {
        const int pos = t * 32;
        if (pos >= total_pad) { expert_of_tile[t] = -1; continue; }
        int lo = 0, hi = E;              // largest e with off_pad[e] <= pos (segment contains pos)
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (off_pad[mid] <= pos) { lo = mid; } else { hi = mid; }
        }
        expert_of_tile[t] = lo;
    }
    for (int p = (int)tid; p < total_pad_max; p += (int)MOE_SCAN_NT) {
        gather_idx[p] = -1;
    }
}

// Per compact permuted position p: place it at its padded position and record both maps —
// gather_idx[padpos] = token feeding that padded row, inv_pad[r] = padded position of routing
// row r (what finalize reads). Derives inv_pad directly, no dependence on inv_idx.
kernel void moe_pad_scatter(device const int *sorted_row_idx [[buffer(0)]],   // (TK)
                            device const int *offsets        [[buffer(1)]],   // (E+1) unpadded
                            device const int *off_pad        [[buffer(2)]],   // (E+1)
                            device int       *gather_idx     [[buffer(3)]],   // (total_pad_max)
                            device int       *inv_pad        [[buffer(4)]],   // (TK)
                            constant int &TK [[buffer(5)]],
                            constant int &E [[buffer(6)]],
                            constant int &K [[buffer(7)]],
                            uint tid [[thread_position_in_grid]]) {
    const int p = (int)tid;
    if (p >= TK) { return; }
    int lo = 0, hi = E;                  // expert whose compact segment contains p
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (offsets[mid] <= p) { lo = mid; } else { hi = mid; }
    }
    const int padpos = off_pad[lo] + (p - offsets[lo]);
    const int r = sorted_row_idx[p];     // flat routing row (token r/K, slot r%K)
    gather_idx[padpos] = r / K;
    inv_pad[r] = padpos;
}

// Gather the permuted activations: permuted_input[p, :] = x[gather_idx[p], :] (zeros for
// pad rows / the unused tail). One threadgroup of 128 threads per padded row, vec4 loads.
template <typename T>
kernel void moe_gather(device const T   *x          [[buffer(0)]],   // (T, H)
                       device const int *gather_idx [[buffer(1)]],   // (total_pad_max)
                       device T         *out        [[buffer(2)]],   // (total_pad_max, H)
                       constant int &H [[buffer(3)]],
                       uint3 tgid [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    using T4 = metal::vec<T, 4>;
    const long p = (long)tgid.x;
    const int src = gather_idx[p];
    device T* dst = out + p * H;
    if (src < 0) {
        for (int i = (int)tid * 4; i + 4 <= H; i += 128 * 4) {
            ((device T4*)(dst + i))[0] = T4(0);
        }
        for (int i = (H & ~3) + (int)tid; i < H; i += 128) { dst[i] = T(0); }
        return;
    }
    device const T* row = x + (long)src * H;
    for (int i = (int)tid * 4; i + 4 <= H; i += 128 * 4) {
        ((device T4*)(dst + i))[0] = ((device const T4*)(row + i))[0];
    }
    for (int i = (H & ~3) + (int)tid; i < H; i += 128) { dst[i] = row[i]; }
}

#define instantiate_moe_gather(type_name, T)                                   \
  template [[host_name("moe_gather_" #type_name)]] [[kernel]] void             \
  moe_gather<T>(device const T *x [[buffer(0)]],                               \
                device const int *gather_idx [[buffer(1)]],                    \
                device T *out [[buffer(2)]],                                   \
                constant int &H [[buffer(3)]],                                 \
                uint3 tgid [[threadgroup_position_in_grid]],                   \
                uint tid [[thread_index_in_threadgroup]]);

instantiate_moe_gather(float32, float)
instantiate_moe_gather(bfloat16, bf16)

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
    if (e < 0) { return; }   // padding tile beyond the real schedule (worst-case grid);
                             // its output rows are never read (inv_pad only maps real rows)

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

// ---------------------------------------------------------------------------
// Rectangular grouped GEMM: out(total_rows, N_out) = A(total_rows, K_dim) @ W[e](K_dim, N_out).
// Same segmented single-expert-per-tile structure, but the contraction (K_dim) and output width
// (N_out) are decoupled (the square moe_grouped_gemm is the K_dim==N_out==H case). Serves the MoE
// MLP's GEMM2 (inter -> H) directly. W is (E, K_dim, N_out); grid (N_out/32, total_rows/32).
// ---------------------------------------------------------------------------
template <typename T, unsigned N_BLOCK, unsigned K_BLOCK, unsigned M_BLOCK>
kernel void moe_grouped_gemm_rect(device T *out                    [[buffer(0)]],
                                  device T *A                      [[buffer(1)]],
                                  device T *W                      [[buffer(2)]],
                                  device const int *expert_of_tile [[buffer(3)]],
                                  constant int &total_rows         [[buffer(4)]],
                                  constant int &K_dim              [[buffer(5)]],
                                  constant int &N_out              [[buffer(6)]],
                                  uint3 threadgroup_id [[threadgroup_position_in_grid]],
                                  uint  simd_lane_id   [[thread_index_in_simdgroup]]) {
    const int OY = (int)threadgroup_id.y;    // row-tile
    const int OX = (int)threadgroup_id.x;    // output column-tile (in N_out)
    const int e = expert_of_tile[OY];
    if (e < 0) { return; }   // padding tile beyond the real schedule (never read downstream)

    using global_layout = gl<T, 1, 1, -1, -1>;
    global_layout gl_a(A, nullptr, nullptr, total_rows, K_dim);
    global_layout gl_w(W + (long)e * K_dim * N_out, nullptr, nullptr, K_dim, N_out);
    global_layout gl_d(out, nullptr, nullptr, total_rows, N_out);

    constexpr const int N_BE = N_BLOCK * TILE_DIM, M_BE = M_BLOCK * TILE_DIM, K_BE = K_BLOCK * TILE_DIM;
    rt<T, N_BE, K_BE> a_reg;
    rt<T, K_BE, M_BE> b_reg;
    rt<float, N_BE, M_BE> d_reg;
    zero(d_reg);
    for (int k = 0; k < K_dim / K_BE; k++) {
        load(a_reg, gl_a, {0, 0, OY, k}, simd_lane_id);
        load(b_reg, gl_w, {0, 0, k, OX}, simd_lane_id);
        mma_AB(d_reg, a_reg, b_reg, d_reg);
    }
    store(gl_d, d_reg, {0, 0, OY, OX}, simd_lane_id);
}

// ---------------------------------------------------------------------------
// Fused SiLU-GLU GEMM1: out(total_rows, inter) = silu(A @ W1_gate) * (A @ W1_up), where W1[e] is
// (H, 2*inter) laid out [gate(inter) | up(inter)]. Each inter output tile accumulates the gate and
// up 32-col tiles then applies register-tile silu + tile*tile mul — one pass, intermediate traffic
// is inter (not 2*inter). grid (inter/32, total_rows/32).
// ---------------------------------------------------------------------------
template <typename T, unsigned N_BLOCK, unsigned K_BLOCK, unsigned M_BLOCK>
kernel void moe_grouped_gemm_swiglu(device T *out                    [[buffer(0)]],
                                    device T *A                      [[buffer(1)]],
                                    device T *W1                     [[buffer(2)]],
                                    device const int *expert_of_tile [[buffer(3)]],
                                    constant int &total_rows         [[buffer(4)]],
                                    constant int &H                  [[buffer(5)]],
                                    constant int &inter              [[buffer(6)]],
                                    uint3 threadgroup_id [[threadgroup_position_in_grid]],
                                    uint  simd_lane_id   [[thread_index_in_simdgroup]]) {
    const int OY = (int)threadgroup_id.y;
    const int OX = (int)threadgroup_id.x;    // output column-tile in inter
    const int e = expert_of_tile[OY];
    if (e < 0) { return; }   // padding tile beyond the real schedule (never read downstream)

    constexpr const int N_BE = N_BLOCK * TILE_DIM, M_BE = M_BLOCK * TILE_DIM, K_BE = K_BLOCK * TILE_DIM;
    using global_layout = gl<T, 1, 1, -1, -1>;
    global_layout gl_a(A, nullptr, nullptr, total_rows, H);
    global_layout gl_w(W1 + (long)e * H * (2 * inter), nullptr, nullptr, H, 2 * inter);
    global_layout gl_d(out, nullptr, nullptr, total_rows, inter);
    const int up_tile = inter / M_BE + OX;   // up-half column-tile

    rt<T, N_BE, K_BE> a_reg;
    rt<T, K_BE, M_BE> bg_reg, bu_reg;
    rt<float, N_BE, M_BE> gate, up;
    zero(gate);
    zero(up);
    for (int k = 0; k < H / K_BE; k++) {
        load(a_reg, gl_a, {0, 0, OY, k}, simd_lane_id);
        load(bg_reg, gl_w, {0, 0, k, OX}, simd_lane_id);
        load(bu_reg, gl_w, {0, 0, k, up_tile}, simd_lane_id);
        mma_AB(gate, a_reg, bg_reg, gate);
        mma_AB(up, a_reg, bu_reg, up);
    }
    silu(gate, gate);          // silu(gate)
    mul(gate, gate, up);       // * up
    store(gl_d, gate, {0, 0, OY, OX}, simd_lane_id);
}

#define instantiate_moe_grouped_gemm_rect(type_name, T)                        \
  template [[host_name("moe_grouped_gemm_rect_" #type_name)]] [[kernel]] void   \
  moe_grouped_gemm_rect<T, 4, 2, 4>(device T *out [[buffer(0)]],               \
                                    device T *A [[buffer(1)]],                 \
                                    device T *W [[buffer(2)]],                 \
                                    device const int *expert_of_tile [[buffer(3)]], \
                                    constant int &total_rows [[buffer(4)]],    \
                                    constant int &K_dim [[buffer(5)]],         \
                                    constant int &N_out [[buffer(6)]],         \
                                    uint3 threadgroup_id [[threadgroup_position_in_grid]], \
                                    uint simd_lane_id [[thread_index_in_simdgroup]]);

#define instantiate_moe_grouped_gemm_swiglu(type_name, T)                      \
  template [[host_name("moe_grouped_gemm_swiglu_" #type_name)]] [[kernel]] void \
  moe_grouped_gemm_swiglu<T, 4, 2, 4>(device T *out [[buffer(0)]],            \
                                      device T *A [[buffer(1)]],              \
                                      device T *W1 [[buffer(2)]],             \
                                      device const int *expert_of_tile [[buffer(3)]], \
                                      constant int &total_rows [[buffer(4)]],  \
                                      constant int &H [[buffer(5)]],           \
                                      constant int &inter [[buffer(6)]],       \
                                      uint3 threadgroup_id [[threadgroup_position_in_grid]], \
                                      uint simd_lane_id [[thread_index_in_simdgroup]]);

instantiate_moe_grouped_gemm_rect(float32, float)
instantiate_moe_grouped_gemm_rect(bfloat16, bf16)
instantiate_moe_grouped_gemm_swiglu(float32, float)
instantiate_moe_grouped_gemm_swiglu(bfloat16, bf16)

// ---------------------------------------------------------------------------
// Quantized grouped expert GEMMs. Same segmented single-expert-per-tile structure as the dense
// kernels above, but W is weight-only-quantized and dequantized straight into the simdgroup
// fragment (dequant_into_register_col — no threadgroup round-trip, no barrier). Every quant
// format groups along the contraction axis, so experts are packed row-major over (N_out, K_dim)
// — the TRANSPOSE of the dense kernels' (K_dim, N_out) — and the contraction is mma_ABt
// (out = A @ Wq^T). Per-expert packed block: N_out * (K_dim/block_k) * block_bytes bytes,
// so the per-expert base offset needs long arithmetic (multi-GB expert stacks).
// Optional per-output-column bias sits in the fp32 epilogue (gpt-oss has expert biases);
// pass a 1-element dummy and has_bias=0 when absent. Requires K_dim % 32 == 0 and
// K_dim % FMT::block_k == 0. A/out are bf16 (the moe pipeline's working dtype).
// ---------------------------------------------------------------------------
template <typename FMT>
kernel void moe_grouped_gemm_rect_q(device bf16 *out                 [[buffer(0)]],
                                    device bf16 *A                   [[buffer(1)]],
                                    device const uchar *Wq           [[buffer(2)]],
                                    device const int *expert_of_tile [[buffer(3)]],
                                    device const bf16 *bias          [[buffer(4)]],
                                    constant int &total_rows         [[buffer(5)]],
                                    constant int &K_dim              [[buffer(6)]],
                                    constant int &N_out              [[buffer(7)]],
                                    constant int &has_bias           [[buffer(8)]],
                                    uint3 threadgroup_id [[threadgroup_position_in_grid]],
                                    uint  warp           [[simdgroup_index_in_threadgroup]],
                                    uint  simd_lane_id   [[thread_index_in_simdgroup]]) {
    const int OY = (int)threadgroup_id.y;    // row-tile (32 permuted rows)
    const int OX = (int)threadgroup_id.x;    // output column-tile (in N_out)
    const int e = expert_of_tile[OY];
    if (e < 0) { return; }   // padding tile beyond the real schedule (never read downstream)

    using global_layout = gl<bf16, 1, 1, -1, -1>;
    global_layout gl_a(A, nullptr, nullptr, total_rows, K_dim);
    global_layout gl_d(out, nullptr, nullptr, total_rows, N_out);
    const int blocks_per_row = K_dim / FMT::block_k;
    device const uchar *Wq_e =
        Wq + (long)e * (long)N_out * (long)blocks_per_row * FMT::block_bytes;

    // 4-warp intra-threadgroup split-K over per-warp shared dequant tiles. Measured rationale
    // (2880^2 gpt-oss tile, M4 Max): the packed path is dequant-ALU-bound, not bandwidth-bound.
    // One warp/tile can't feed the dequant pipe (frag 0.073 ms vs dense 0.062 ms at decode), and
    // a register fill pays the block-scale decode per ELEMENT — fatal for the e8m0 formats whose
    // scale is an exp2 (mxfp4 regressed to 0.109 ms under register-fill split-K while q8_0
    // improved). tk_dequant8 spans amortize the scale decode 8x, each warp owns a private sW
    // tile (no cross-warp sync in the K loop, simdgroup barriers only), and warps split K 4-way
    // into private fp32 accumulators reduced once at the end.
    constexpr const int BN = 32, BK = 32, BM = 32, N_WARPS = 4;
    threadgroup st<float, BN, BM> sAcc[N_WARPS - 1];   // 12 KB: split-K partials
    rt<bf16, BN, BK> a_reg;
    rt<bf16, BM, BK, ducks::rt_layout::col> w_reg;   // logical (N_out-tile x K), mma_ABt operand
    rt<float, BN, BM> d_reg;
    zero(d_reg);
    for (int kb = (int)warp; kb < K_dim / BK; kb += N_WARPS) {
        load(a_reg, gl_a, {0, 0, OY, kb}, simd_lane_id);
        dequant_into_register_col<FMT>(w_reg, Wq_e, N_out, K_dim, OX, kb, simd_lane_id);
        mma_ABt(d_reg, a_reg, w_reg, d_reg);
    }
    if (warp > 0) store(sAcc[warp - 1], d_reg, simd_lane_id);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (warp != 0) { return; }
    rt<float, BN, BM> part;
    #pragma clang loop unroll(full)
    for (int w = 0; w < N_WARPS - 1; w++) {
        load(part, sAcc[w], simd_lane_id);
        add(d_reg, d_reg, part);
    }
    if (has_bias != 0) {
        // per-output-column bias broadcast down the rows: each lane patches its own two
        // horizontally-adjacent fp32 accumulator elements (cols sx, sx+1 of each 8x8 subtile)
        const int qid = (int)simd_lane_id / 4;
        const int sx  = (qid & 2) * 2 + ((int)simd_lane_id % 2) * 2;
        device const bf16 *bias_e = bias + (long)e * N_out + OX * BM;
        #pragma clang loop unroll(full)
        for (int i = 0; i < d_reg.height; i++) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < d_reg.width; j++) {
                const int c = j * TILE_DIM + sx;
                d_reg.tiles[i][j].data.thread_elements()[0] += (float)bias_e[c];
                d_reg.tiles[i][j].data.thread_elements()[1] += (float)bias_e[c + 1];
            }
        }
    }
    store(gl_d, d_reg, {0, 0, OY, OX}, simd_lane_id);
}

// Fused quantized SwiGLU GEMM1: gate/up both come from one packed W1q laid out
// [gate(inter) | up(inter)] along the packed-row (N) axis, i.e. (2*inter, H) packed rows.
// act_mode 0: silu(gate) * up. act_mode 1: gpt-oss swiglu_oai — gate = min(gate, limit),
// up = clamp(up, -limit, limit), out = gate * sigmoid(alpha * gate) * (1 + up).
// Bias (E, 2*inter) is added pre-activation (has_bias gate). Epilogue is elementwise over
// the fp32 accumulator fragments (uniform branches; lane owns cols sx, sx+1 per subtile).
template <typename FMT>
kernel void moe_grouped_gemm_swiglu_q(device bf16 *out                 [[buffer(0)]],
                                      device bf16 *A                   [[buffer(1)]],
                                      device const uchar *W1q          [[buffer(2)]],
                                      device const int *expert_of_tile [[buffer(3)]],
                                      device const bf16 *bias          [[buffer(4)]],
                                      constant int &total_rows         [[buffer(5)]],
                                      constant int &H                  [[buffer(6)]],
                                      constant int &inter              [[buffer(7)]],
                                      constant int &has_bias           [[buffer(8)]],
                                      constant int &act_mode           [[buffer(9)]],
                                      constant float &alpha            [[buffer(10)]],
                                      constant float &limit            [[buffer(11)]],
                                      uint3 threadgroup_id [[threadgroup_position_in_grid]],
                                      uint  warp           [[simdgroup_index_in_threadgroup]],
                                      uint  simd_lane_id   [[thread_index_in_simdgroup]]) {
    const int OY = (int)threadgroup_id.y;
    const int OX = (int)threadgroup_id.x;    // output column-tile in inter
    const int e = expert_of_tile[OY];
    if (e < 0) { return; }

    using global_layout = gl<bf16, 1, 1, -1, -1>;
    global_layout gl_a(A, nullptr, nullptr, total_rows, H);
    global_layout gl_d(out, nullptr, nullptr, total_rows, inter);
    const int blocks_per_row = H / FMT::block_k;
    device const uchar *Wq_e =
        W1q + (long)e * (long)(2 * inter) * (long)blocks_per_row * FMT::block_bytes;

    // Same 4-warp split-K + per-warp shared-dequant geometry as the rect kernel above, with
    // two W tiles (gate/up) per warp. Threadgroup budget forces a TWO-PASS reduce sharing one
    // fp32 staging set: 8 sW tiles (16 KB) + 3 fp32 partial tiles (12 KB) = 28 KB < 32 KB.
    constexpr const int BN = 32, BK = 32, BM = 32, N_WARPS = 4;
    const int up_tile = inter / BM + OX;     // up-half packed-row tile
    threadgroup st<float, BN, BM> sAcc[N_WARPS - 1];
    rt<bf16, BN, BK> a_reg;
    rt<bf16, BM, BK, ducks::rt_layout::col> wg_reg, wu_reg;
    rt<float, BN, BM> gate, up;
    zero(gate);
    zero(up);
    for (int kb = (int)warp; kb < H / BK; kb += N_WARPS) {
        load(a_reg, gl_a, {0, 0, OY, kb}, simd_lane_id);
        dequant_into_register_col<FMT>(wg_reg, Wq_e, 2 * inter, H, OX, kb, simd_lane_id);
        dequant_into_register_col<FMT>(wu_reg, Wq_e, 2 * inter, H, up_tile, kb, simd_lane_id);
        mma_ABt(gate, a_reg, wg_reg, gate);
        mma_ABt(up, a_reg, wu_reg, up);
    }
    // pass 1: reduce gate partials into warp 0
    if (warp > 0) store(sAcc[warp - 1], gate, simd_lane_id);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (warp == 0) {
        rt<float, BN, BM> part;
        #pragma clang loop unroll(full)
        for (int w = 0; w < N_WARPS - 1; w++) {
            load(part, sAcc[w], simd_lane_id);
            add(gate, gate, part);
        }
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);   // warp 0 done reading gate partials
    // pass 2: reduce up partials, reusing the same staging tiles
    if (warp > 0) store(sAcc[warp - 1], up, simd_lane_id);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (warp != 0) { return; }
    {
        rt<float, BN, BM> part;
        #pragma clang loop unroll(full)
        for (int w = 0; w < N_WARPS - 1; w++) {
            load(part, sAcc[w], simd_lane_id);
            add(up, up, part);
        }
    }
    const int qid = (int)simd_lane_id / 4;
    const int sx  = (qid & 2) * 2 + ((int)simd_lane_id % 2) * 2;
    device const bf16 *bg = bias + (long)e * (2 * inter) + OX * BM;
    device const bf16 *bu = bias + (long)e * (2 * inter) + inter + OX * BM;
    #pragma clang loop unroll(full)
    for (int i = 0; i < gate.height; i++) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < gate.width; j++) {
            #pragma clang loop unroll(full)
            for (int el = 0; el < 2; el++) {
                const int c = j * TILE_DIM + sx + el;
                float g = gate.tiles[i][j].data.thread_elements()[el];
                float u = up.tiles[i][j].data.thread_elements()[el];
                if (has_bias != 0) { g += (float)bg[c]; u += (float)bu[c]; }
                float r;
                if (act_mode == 1) {
                    g = metal::min(g, limit);
                    u = metal::clamp(u, -limit, limit);
                    r = (g / (1.0f + metal::exp(-g * alpha))) * (1.0f + u);
                } else {
                    r = (g / (1.0f + metal::exp(-g))) * u;
                }
                gate.tiles[i][j].data.thread_elements()[el] = r;
            }
        }
    }
    store(gl_d, gate, {0, 0, OY, OX}, simd_lane_id);
}

#define instantiate_moe_gemm_q(fmt_name, FMT)                                       \
  template [[host_name("moe_grouped_gemm_rect_q_" #fmt_name)]] [[kernel]] void       \
  moe_grouped_gemm_rect_q<FMT>(device bf16 *out [[buffer(0)]],                       \
                               device bf16 *A [[buffer(1)]],                         \
                               device const uchar *Wq [[buffer(2)]],                 \
                               device const int *expert_of_tile [[buffer(3)]],       \
                               device const bf16 *bias [[buffer(4)]],                \
                               constant int &total_rows [[buffer(5)]],               \
                               constant int &K_dim [[buffer(6)]],                    \
                               constant int &N_out [[buffer(7)]],                    \
                               constant int &has_bias [[buffer(8)]],                 \
                               uint3 threadgroup_id [[threadgroup_position_in_grid]],\
                               uint warp [[simdgroup_index_in_threadgroup]],         \
                               uint simd_lane_id [[thread_index_in_simdgroup]]);      \
  template [[host_name("moe_grouped_gemm_swiglu_q_" #fmt_name)]] [[kernel]] void     \
  moe_grouped_gemm_swiglu_q<FMT>(device bf16 *out [[buffer(0)]],                     \
                                 device bf16 *A [[buffer(1)]],                       \
                                 device const uchar *W1q [[buffer(2)]],              \
                                 device const int *expert_of_tile [[buffer(3)]],     \
                                 device const bf16 *bias [[buffer(4)]],              \
                                 constant int &total_rows [[buffer(5)]],             \
                                 constant int &H [[buffer(6)]],                      \
                                 constant int &inter [[buffer(7)]],                  \
                                 constant int &has_bias [[buffer(8)]],               \
                                 constant int &act_mode [[buffer(9)]],               \
                                 constant float &alpha [[buffer(10)]],               \
                                 constant float &limit [[buffer(11)]],               \
                                 uint3 threadgroup_id [[threadgroup_position_in_grid]],\
                                 uint warp [[simdgroup_index_in_threadgroup]],       \
                                 uint simd_lane_id [[thread_index_in_simdgroup]]);

instantiate_moe_gemm_q(mxfp4, mxfp4)
instantiate_moe_gemm_q(kU4, kU4)
instantiate_moe_gemm_q(fp8_e4m3, fp8_e4m3)
instantiate_moe_gemm_q(q8_0, q8_0)
instantiate_moe_gemm_q(nvfp4, nvfp4)
instantiate_moe_gemm_q(q4_K, q4_K)

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
