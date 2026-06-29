#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Causal flash-attention forward (warp-level), bf16, D in {64,128}.
// Same online-softmax structure as attn_fwd, with causal masking:
//   - only iterate kv blocks kv_idx <= q_seq (later keys are fully masked),
//   - on the diagonal block (kv_idx == q_seq) apply make_causal (set the
//     strictly-upper-triangular scores to -inf before the softmax).
constant constexpr const int TNc = 8;

template <int D>
kernel void attn_causal(device   bf16     *q [[buffer(0)]],
                        device   bf16     *k [[buffer(1)]],
                        device   bf16     *v [[buffer(2)]],
                        device   bf16     *o [[buffer(3)]],
                        constant unsigned &N [[buffer(4)]],
                        constant unsigned &H [[buffer(5)]],
                        uint3 blockIdx [[threadgroup_position_in_grid]],
                        uint laneId [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    using global_layout = gl<bfloat, 1, -1, -1, D>;
    global_layout gl_q(q, nullptr, H, N, nullptr);
    global_layout gl_k(k, nullptr, H, N, nullptr);
    global_layout gl_v(v, nullptr, H, N, nullptr);
    global_layout gl_o(o, nullptr, H, N, nullptr);
    using st_qkv     = st_bf<TNc, D>;
    using rt_qkv     = rt_bf<TNc, D>;
    using rt_k_t     = rt_bf<TNc, D, ducks::rt_layout::col>;
    using rt_att     = rt_fl<TNc, TNc>;
    using rt_o       = rt_fl<TNc, D>;
    using rv_att     = rt_fl<TNc, TNc>::col_vec;

    const int block = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq = blockIdx.x;

    rt_qkv q_reg;
    rt_k_t k_reg;
    rt_qkv v_reg;
    rt_att att_block;
    rt_o o_reg;
    rv_att max_vec_last;
    rv_att max_vec;
    rv_att norm_vec;

    load(q_reg, gl_q, {block, head, q_seq, 0}, laneId);
    neg_infty(max_vec);
    zero(norm_vec);
    zero(o_reg);
    constexpr const bf16 q_mul = ((D == 128) ? 0.08838834764bf : 0.125bf) * 1.44269504089bf;
    mul(q_reg, q_reg, q_mul);

    // Causal: only attend to key blocks at or before the query block.
    #pragma clang loop unroll(full)
    for(int kv_idx = 0; kv_idx <= q_seq; kv_idx++) {
        load(k_reg, gl_k, {block, head, kv_idx, 0}, laneId);
        zero(att_block);
        mma_ABt(att_block, q_reg, k_reg, att_block);
        if (kv_idx == q_seq) {
            // strictly-upper-triangular scores -> -inf (mask future positions)
            float neg_big = -1e30f;   // thread-space (make_causal takes thread const&)
            make_causal(att_block, att_block, laneId, neg_big);
        }
        copy(max_vec_last,  max_vec, laneId);
        row_max(max_vec, att_block, max_vec, laneId);
        sub(max_vec_last, max_vec_last, max_vec);
        exp2(max_vec_last, max_vec_last);
        sub_row(att_block, att_block, max_vec);
        exp2(att_block, att_block);
        mul(norm_vec, norm_vec, max_vec_last);
        row_sum(norm_vec, att_block, norm_vec, laneId);
        mul_row(o_reg, o_reg, max_vec_last);
        load(v_reg, gl_v, {block, head, kv_idx, 0}, laneId);
        mma_AB(o_reg, att_block, v_reg, o_reg);
    }
    div_row(o_reg, o_reg, norm_vec);
    store(gl_o, o_reg, {block, head, q_seq, 0}, laneId);
}

#define instantiate_attn_causal(D)                               \
  template [[host_name("attn_causal_" #D)]] [[kernel]] void      \
  attn_causal<D>(device   bf16     *q [[buffer(0)]], \
    device   bf16     *k [[buffer(1)]], \
    device   bf16     *v [[buffer(2)]], \
    device   bf16     *o [[buffer(3)]], \
    constant unsigned &N [[buffer(4)]], \
    constant unsigned &H [[buffer(5)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]); \

instantiate_attn_causal(64);
instantiate_attn_causal(128);

}
