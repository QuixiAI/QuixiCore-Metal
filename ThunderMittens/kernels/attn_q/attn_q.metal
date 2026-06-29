#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Quantized-KV flash attention (Phase B retrofit): K and V arrive as quantized blocks (format FMT),
// dequantized on the fly before the QK^T / AV matmuls. Q stays bf16. Structurally identical to
// attn_fwd (online softmax, one simdgroup per 8-row Q tile); the only change is the K/V loads:
//   - K (col-layout, B operand of mma_ABt) -> dequant_into_shared -> col load (the register
//     dequant only emits row-layout, so K must go through the threadgroup tile).
//   - V (row-layout, B operand of mma_AB)  -> dequant straight into the register fragment.
// Kq/Vq are uchar (B,H,N, D/block_k, block_bytes), each (b,h) slice an (N,D) matrix quantized along D.
constant constexpr const int TNQ = 8;

template <typename FMT, int D, bool CAUSAL>
kernel void attn_q(device   bf16     *q  [[buffer(0)]],
                   device   uchar    *Kq [[buffer(1)]],
                   device   uchar    *Vq [[buffer(2)]],
                   device   bf16     *o  [[buffer(3)]],
                   constant unsigned &N  [[buffer(4)]],
                   constant unsigned &H  [[buffer(5)]],
                   uint3 blockIdx [[threadgroup_position_in_grid]],
                   uint  tid      [[thread_index_in_threadgroup]],
                   uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    using global_layout = gl<bfloat, 1, -1, -1, D>;
    global_layout gl_q(q, nullptr, H, N, nullptr);
    global_layout gl_o(o, nullptr, H, N, nullptr);
    using rt_qkv = rt_bf<TNQ, D>;
    using rt_k_t = rt_bf<TNQ, D, ducks::rt_layout::col>;
    using rt_att = rt_fl<TNQ, TNQ>;
    using rt_o   = rt_fl<TNQ, D>;
    using rv_att = rt_fl<TNQ, TNQ>::col_vec;

    const int block = blockIdx.z, head = blockIdx.y, q_seq = blockIdx.x;
    const int kv_last = CAUSAL ? q_seq : (int)(N / TNQ) - 1;            // causal: attend <= q block
    const int bpr = D / FMT::block_k;                                   // quant blocks per key row
    device const uchar* Kbh = Kq + (uint)((block * (int)H + head) * (int)N) * bpr * FMT::block_bytes;
    device const uchar* Vbh = Vq + (uint)((block * (int)H + head) * (int)N) * bpr * FMT::block_bytes;

    threadgroup st<half, TNQ, D> sK;
    rt_qkv q_reg; rt_k_t k_reg; rt_qkv v_reg; rt_att att_block; rt_o o_reg;
    rv_att max_vec_last, max_vec, norm_vec;

    load(q_reg, gl_q, {block, head, q_seq, 0}, laneId);
    neg_infty(max_vec); zero(norm_vec); zero(o_reg);
    constexpr const bf16 q_mul = ((D == 128) ? 0.08838834764bf : 0.125bf) * 1.44269504089bf;
    mul(q_reg, q_reg, q_mul);

    for (int kv_idx = 0; kv_idx <= kv_last; kv_idx++) {
        dequant_into_shared<FMT, TNQ, D>(sK, Kbh, (int)N, D, kv_idx, 0, 32, tid);   // K -> shared (half)
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        load(k_reg, sK, laneId);                                                    // shared -> col reg (K^T)
        zero(att_block);
        mma_ABt(att_block, q_reg, k_reg, att_block);
        if (CAUSAL && kv_idx == q_seq) { float nb = -1e30f; make_causal(att_block, att_block, laneId, nb); }
        copy(max_vec_last, max_vec, laneId);
        row_max(max_vec, att_block, max_vec, laneId);
        sub(max_vec_last, max_vec_last, max_vec); exp2(max_vec_last, max_vec_last);
        sub_row(att_block, att_block, max_vec); exp2(att_block, att_block);
        mul(norm_vec, norm_vec, max_vec_last);
        row_sum(norm_vec, att_block, norm_vec, laneId);
        mul_row(o_reg, o_reg, max_vec_last);
        dequant_into_register<FMT>(v_reg, Vbh, (int)N, D, kv_idx, 0, laneId);       // V -> row reg
        mma_AB(o_reg, att_block, v_reg, o_reg);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);                     // before sK reuse
    }
    div_row(o_reg, o_reg, norm_vec);
    store(gl_o, o_reg, {block, head, q_seq, 0}, laneId);
}

#define instantiate_attn_q(name, FMT, D, CAUSAL)                                  \
   template [[host_name(name)]] [[kernel]] void attn_q<FMT, D, CAUSAL>(           \
     device bf16* q [[buffer(0)]], device uchar* Kq [[buffer(1)]],                \
     device uchar* Vq [[buffer(2)]], device bf16* o [[buffer(3)]],                \
     constant unsigned &N [[buffer(4)]], constant unsigned &H [[buffer(5)]],      \
     uint3 blockIdx [[threadgroup_position_in_grid]],                            \
     uint tid [[thread_index_in_threadgroup]], uint laneId [[thread_index_in_simdgroup]]);

#define instantiate_attn_q_fmt(fmt, FMT, D)                                       \
   instantiate_attn_q("attn_q_" #fmt "_" #D, FMT, D, false)                       \
   instantiate_attn_q("attn_q_causal_" #fmt "_" #D, FMT, D, true)

instantiate_attn_q_fmt(q8_0, q8_0, 64)
instantiate_attn_q_fmt(q8_0, q8_0, 128)
instantiate_attn_q_fmt(q4_0, q4_0, 64)
instantiate_attn_q_fmt(q4_0, q4_0, 128)
instantiate_attn_q_fmt(fp8_e4m3, fp8_e4m3, 64)
instantiate_attn_q_fmt(fp8_e4m3, fp8_e4m3, 128)

// ---- multi-warp variant (non-causal): NUM_WARPS simdgroups share each dequantized K/V block.
// Both K and V are cooperatively dequantized into threadgroup memory (multiwarp already stages
// both), amortizing the dequant across the warps; each warp runs its own query tile's softmax. ----
constant constexpr const int NWQ = 4;

template <typename FMT, int D>
kernel void attn_q_mw(device   bf16     *q  [[buffer(0)]],
                      device   uchar    *Kq [[buffer(1)]],
                      device   uchar    *Vq [[buffer(2)]],
                      device   bf16     *o  [[buffer(3)]],
                      constant unsigned &N  [[buffer(4)]],
                      constant unsigned &H  [[buffer(5)]],
                      uint3 blockIdx [[threadgroup_position_in_grid]],
                      uint  tid      [[thread_index_in_threadgroup]],
                      uint  warp     [[simdgroup_index_in_threadgroup]],
                      uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    using G = group<NWQ>;
    using global_layout = gl<bfloat, 1, -1, -1, D>;
    global_layout gl_q(q, nullptr, H, N, nullptr);
    global_layout gl_o(o, nullptr, H, N, nullptr);
    using rt_qkv = rt_bf<TNQ, D>;
    using rt_k_t = rt_bf<TNQ, D, ducks::rt_layout::col>;
    using rt_att = rt_fl<TNQ, TNQ>;
    using rt_o   = rt_fl<TNQ, D>;
    using rv_att = rt_fl<TNQ, TNQ>::col_vec;

    const int block = blockIdx.z, head = blockIdx.y;
    const int q_tile = blockIdx.x * NWQ + (int)warp;
    const int kv_blocks = N / TNQ;
    const int bpr = D / FMT::block_k;
    device const uchar* Kbh = Kq + (uint)((block * (int)H + head) * (int)N) * bpr * FMT::block_bytes;
    device const uchar* Vbh = Vq + (uint)((block * (int)H + head) * (int)N) * bpr * FMT::block_bytes;

    threadgroup st<half, TNQ, D> sK, sV;
    rt_qkv q_reg; rt_k_t k_reg; rt_qkv v_reg; rt_att att_block; rt_o o_reg;
    rv_att max_vec_last, max_vec, norm_vec;

    load(q_reg, gl_q, {block, head, q_tile, 0}, laneId);
    neg_infty(max_vec); zero(norm_vec); zero(o_reg);
    constexpr const bf16 q_mul = ((D == 128) ? 0.08838834764bf : 0.125bf) * 1.44269504089bf;
    mul(q_reg, q_reg, q_mul);

    for (int kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
        dequant_into_shared<FMT, TNQ, D>(sK, Kbh, (int)N, D, kv_idx, 0, G::GROUP_THREADS, tid);
        dequant_into_shared<FMT, TNQ, D>(sV, Vbh, (int)N, D, kv_idx, 0, G::GROUP_THREADS, tid);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        load(k_reg, sK, laneId);
        zero(att_block);
        mma_ABt(att_block, q_reg, k_reg, att_block);
        copy(max_vec_last, max_vec, laneId);
        row_max(max_vec, att_block, max_vec, laneId);
        sub(max_vec_last, max_vec_last, max_vec); exp2(max_vec_last, max_vec_last);
        sub_row(att_block, att_block, max_vec); exp2(att_block, att_block);
        mul(norm_vec, norm_vec, max_vec_last);
        row_sum(norm_vec, att_block, norm_vec, laneId);
        mul_row(o_reg, o_reg, max_vec_last);
        load(v_reg, sV, laneId);
        mma_AB(o_reg, att_block, v_reg, o_reg);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    div_row(o_reg, o_reg, norm_vec);
    store(gl_o, o_reg, {block, head, q_tile, 0}, laneId);
}

#define instantiate_attn_q_mw(name, FMT, D)                                       \
   template [[host_name(name)]] [[kernel]] void attn_q_mw<FMT, D>(                \
     device bf16* q [[buffer(0)]], device uchar* Kq [[buffer(1)]],                \
     device uchar* Vq [[buffer(2)]], device bf16* o [[buffer(3)]],                \
     constant unsigned &N [[buffer(4)]], constant unsigned &H [[buffer(5)]],      \
     uint3 blockIdx [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
     uint warp [[simdgroup_index_in_threadgroup]], uint laneId [[thread_index_in_simdgroup]]);

instantiate_attn_q_mw("attn_q_mw_q8_0_64", q8_0, 64);
instantiate_attn_q_mw("attn_q_mw_q8_0_128", q8_0, 128);
instantiate_attn_q_mw("attn_q_mw_fp8_e4m3_64", fp8_e4m3, 64);
instantiate_attn_q_mw("attn_q_mw_fp8_e4m3_128", fp8_e4m3, 128);

}
