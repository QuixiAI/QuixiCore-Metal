#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// GELU activation (tanh approximation), bf16 I/O, fp32 compute. Matches
// mx.nn.gelu_approx / F.gelu(approximate='tanh'):
//   y = 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
//
// Elementwise; one simdgroup per row of length D (rv_fl<D> in registers).
// ---------------------------------------------------------------------------
template <int D>
kernel void gelu(device   bf16 *x [[buffer(0)]],
                 device   bf16 *o [[buffer(1)]],
                 constant uint &M [[buffer(2)]],
                 uint3 blockIdx [[threadgroup_position_in_grid]],
                 uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;

    using row_gl = gl<bf16, 1, 1, -1, D>;
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_o(o, nullptr, nullptr, M, nullptr);

    using vecD = rv_fl<D>;
    vecD xv;
    load(xv, gl_x, {0, 0, row, 0}, laneId);
    gelu(xv, xv);                            // tanh-approx GELU (vec map)
    store(gl_o, xv, {0, 0, row, 0}, laneId);
}

#define instantiate_gelu(DVAL)                                                \
  template [[host_name("gelu_" #DVAL)]] [[kernel]] void                       \
  gelu<DVAL>(device   bf16 *x [[buffer(0)]],                                  \
             device   bf16 *o [[buffer(1)]],                                  \
             constant uint &M [[buffer(2)]],                                  \
             uint3 blockIdx [[threadgroup_position_in_grid]],                 \
             uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_gelu(256);
instantiate_gelu(512);
instantiate_gelu(768);
instantiate_gelu(1024);

}
