#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Mamba-2 / SSD (selective state space) forward, materialized chunked form, bf16, D=64.
//   Y_t = sum_{j<=t} (C_t . B_j) * exp(cumlog_t - cumlog_j) * X_j
// where cumlog = cumsum(log a) is the running log-decay (precomputed on host, fp32).
// This is the SSD attention-equivalent: M = (C @ B^T) (.) L, L the decay-causal matrix,
// Y = M @ X. One simdgroup per (batch, head, query-chunk); loops over key-chunks <= query.
//
// The decay matrix L[i,j] = exp(cumlog_i[i] - cumlog_j[j]) is built from broadcasts:
// add_row(colvec cumlog_i) then sub_col(rowvec cumlog_j), then exp.
template <int D>
kernel void mamba2(device   bf16     *C  [[buffer(0)]],
                   device   bf16     *Bm [[buffer(1)]],
                   device   bf16     *X  [[buffer(2)]],
                   device   float    *cl [[buffer(3)]],   // cumlog (B,H,N), fp32
                   device   bf16     *Y  [[buffer(4)]],
                   constant unsigned &N  [[buffer(5)]],
                   constant unsigned &H  [[buffer(6)]],
                   uint3 blockIdx [[threadgroup_position_in_grid]],
                   uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64, "mamba2 currently supports D=64");
    using gl_t  = gl<bfloat, 1, -1, -1, D>;           // C, B, X, Y : (B,H,N,D)
    using gl_cl = gl<float, 1, -1, 1, -1>;            // cumlog (B,H,N) viewed sequence-along-cols
    gl_t  gC(C, nullptr, H, N, nullptr);
    gl_t  gB(Bm, nullptr, H, N, nullptr);
    gl_t  gX(X, nullptr, H, N, nullptr);
    gl_t  gY(Y, nullptr, H, N, nullptr);
    // gl.get<VEC> offsets by idx.c * VEC::length, so index the sequence chunk via idx.c for
    // BOTH the col_vec and row_vec loads (a vec read pulls VEC::length contiguous values).
    gl_cl gcl(cl, nullptr, H, nullptr, N);

    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int qi = blockIdx.x;                         // this query chunk

    rt_bf<8, D> c_reg;
    load(c_reg, gC, {batch, head, qi, 0}, laneId);
    typename rt_fl<8, 8>::col_vec cumlog_i;             // per query row
    load(cumlog_i, gcl, {batch, head, 0, qi}, laneId);

    rt_fl<8, D> y_reg;
    zero(y_reg);

    for (int kj = 0; kj <= qi; kj++) {
        rt_bf<8, D, ducks::rt_layout::col> b_reg;       // col layout for C @ B^T
        rt_bf<8, D> x_reg;
        load(b_reg, gB, {batch, head, kj, 0}, laneId);
        load(x_reg, gX, {batch, head, kj, 0}, laneId);
        typename rt_fl<8, 8>::row_vec cumlog_j;          // per key col
        load(cumlog_j, gcl, {batch, head, 0, kj}, laneId);

        rt_fl<8, 8> att;
        zero(att);
        mma_ABt(att, c_reg, b_reg, att);                 // C @ B^T

        // decay_log[i,j] = cumlog_i[i] - cumlog_j[j]; decay = exp(decay_log)
        rt_fl<8, 8> decay;
        zero(decay);
        add_row(decay, decay, cumlog_i);
        sub_col(decay, decay, cumlog_j);
        exp(decay, decay);
        mul(att, att, decay);

        if (kj == qi) {
            float zero_fill = 0.0f;
            make_causal(att, att, laneId, zero_fill);    // future positions -> 0
        }

        rt_bf<8, 8> att_bf;
        copy(att_bf, att);
        mma_AB(y_reg, att_bf, x_reg, y_reg);
    }
    store(gY, y_reg, {batch, head, qi, 0}, laneId);
}

#define instantiate_mamba2(D)                                  \
  template [[host_name("mamba2_" #D)]] [[kernel]] void         \
  mamba2<D>(device bf16 *C [[buffer(0)]], device bf16 *Bm [[buffer(1)]], \
    device bf16 *X [[buffer(2)]], device float *cl [[buffer(3)]], \
    device bf16 *Y [[buffer(4)]], \
    constant unsigned &N [[buffer(5)]], constant unsigned &H [[buffer(6)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_mamba2(64);

}
