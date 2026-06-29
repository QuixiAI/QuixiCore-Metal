#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Causal linear attention (identity feature map), bf16, D=64.
//   out_i = sum_{j<=i} (q_i . k_j) v_j
// Chunked running-state scan, one simdgroup per (batch, head). For each chunk
// (8 queries/keys):
//   1. inter-chunk: out_c  = q_c @ KV_state         (KV from earlier chunks)
//   2. intra-chunk: A = q_c @ k_c^T, causal-mask (lower-tri), out_c += A @ v_c
//   3. update:      KV_state += k_c^T @ v_c
// This is the causal analogue of linear_attn (the chunked scan that makes linear
// attention O(N) and causal).
template <int D>
kernel void lin_attn_causal(device   bf16     *q [[buffer(0)]],
                            device   bf16     *k [[buffer(1)]],
                            device   bf16     *v [[buffer(2)]],
                            device   bf16     *o [[buffer(3)]],
                            constant unsigned &N [[buffer(4)]],
                            constant unsigned &H [[buffer(5)]],
                            uint3 blockIdx [[threadgroup_position_in_grid]],
                            uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64, "lin_attn_causal currently supports D=64");
    using gl_t = gl<bfloat, 1, -1, -1, D>;
    gl_t gq(q, nullptr, H, N, nullptr);
    gl_t gk(k, nullptr, H, N, nullptr);
    gl_t gv(v, nullptr, H, N, nullptr);
    gl_t go(o, nullptr, H, N, nullptr);

    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int chunks = N / 8;

    rt_fl<D, D> kv;     // running KV state (sum over earlier chunks of k^T v)
    zero(kv);

    for (int c = 0; c < chunks; c++) {
        rt_bf<8, D> q_reg;
        rt_bf<8, D, ducks::rt_layout::col> k_reg;  // col layout: serves both A=Q@K^T and KV+=K^T@V
        rt_bf<8, D> v_reg;
        load(q_reg, gq, {batch, head, c, 0}, laneId);
        load(k_reg, gk, {batch, head, c, 0}, laneId);
        load(v_reg, gv, {batch, head, c, 0}, laneId);

        rt_fl<8, D> o_reg;
        zero(o_reg);

        // 1. inter-chunk: out_c = q_c @ KV_state(prev)
        rt_bf<D, D> kv_bf;
        copy(kv_bf, kv);
        mma_AB(o_reg, q_reg, kv_bf, o_reg);

        // 2. intra-chunk causal: A = q_c @ k_c^T, lower-triangular mask, out_c += A @ v_c
        rt_fl<8, 8> att;
        zero(att);
        mma_ABt(att, q_reg, k_reg, att);
        float zero_fill = 0.0f;
        make_causal(att, att, laneId, zero_fill);   // strictly-upper -> 0 (no future)
        rt_bf<8, 8> att_bf;
        copy(att_bf, att);
        mma_AB(o_reg, att_bf, v_reg, o_reg);

        store(go, o_reg, {batch, head, c, 0}, laneId);

        // 3. update state: KV_state += k_c^T @ v_c
        mma_AtB(kv, k_reg, v_reg, kv);
    }
}

#define instantiate_lin_attn_causal(D)                                  \
  template [[host_name("lin_attn_causal_" #D)]] [[kernel]] void         \
  lin_attn_causal<D>(device bf16 *q [[buffer(0)]], device bf16 *k [[buffer(1)]], \
    device bf16 *v [[buffer(2)]], device bf16 *o [[buffer(3)]], \
    constant unsigned &N [[buffer(4)]], constant unsigned &H [[buffer(5)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_lin_attn_causal(64);

}
