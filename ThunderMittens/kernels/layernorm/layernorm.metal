#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// LayerNorm (forward), bf16 I/O, fp32 compute.
//
// One simdgroup (32 lanes) processes one row of length D. The whole row fits
// in registers (D/32 floats per lane), so there is no threadgroup memory, no
// async pipeline, and no threadgroup_barrier: every cross-lane exchange happens
// inside the warp-level `sum` reduction via simd shuffles.
//
//   y = (x - mean(x)) * rsqrt(var(x) + eps) * weight + bias
//
// This is a faithful-but-minimal port of ThunderKittens kernels/layernorm:
// dropout and the fused residual add are intentionally dropped for v1.
// ---------------------------------------------------------------------------
template <int D>
kernel void layernorm(device   bf16  *x      [[buffer(0)]],
                      device   bf16  *weight [[buffer(1)]],
                      device   bf16  *bias   [[buffer(2)]],
                      device   bf16  *o      [[buffer(3)]],
                      constant uint  &M      [[buffer(4)]],   // total rows = prod(shape[:-1])
                      constant float &eps    [[buffer(5)]],
                      uint3 blockIdx [[threadgroup_position_in_grid]],
                      uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;

    using row_gl = gl<bf16, 1, 1, -1, D>;   // (M, D) — rows runtime, cols compile-time
    using vec_gl = gl<bf16, 1, 1,  1, D>;   // (1, D) — weight / bias
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_o(o, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight, nullptr, nullptr, nullptr, nullptr);
    vec_gl gl_b(bias,   nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;                   // naive layout, fp32 compute
    vecD xv, wv, bv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);  // bf16 -> fp32 on the fly
    load(wv, gl_w, {0, 0, 0,   0}, laneId);
    load(bv, gl_b, {0, 0, 0,   0}, laneId);

    // mean
    float mean = 0.f;
    sum(mean, xv, laneId);
    mean /= (float)D;
    sub(xv, xv, mean);                       // x - mean (vec - scalar, broadcast)

    // variance
    float var = 0.f;
    mul(sq, xv, xv);                         // (x - mean)^2
    sum(var, sq, laneId);
    var /= (float)D;
    float inv = metal::rsqrt(var + eps);     // scalar rsqrt (no substrate op needed)

    // normalize, scale, shift
    mul(xv, xv, inv);                        // * 1/std (vec - scalar)
    mul(xv, xv, wv);                         // * weight (channelwise vec - vec)
    add(xv, xv, bv);                         // + bias   (channelwise vec - vec)
    store(gl_o, xv, {0, 0, row, 0}, laneId); // fp32 -> bf16 on the fly
}

#define instantiate_layernorm(DVAL)                                            \
  template [[host_name("layernorm_" #DVAL)]] [[kernel]] void                   \
  layernorm<DVAL>(device   bf16  *x      [[buffer(0)]],                        \
                  device   bf16  *weight [[buffer(1)]],                        \
                  device   bf16  *bias   [[buffer(2)]],                        \
                  device   bf16  *o      [[buffer(3)]],                        \
                  constant uint  &M      [[buffer(4)]],                        \
                  constant float &eps    [[buffer(5)]],                        \
                  uint3 blockIdx [[threadgroup_position_in_grid]],             \
                  uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_layernorm(256);
instantiate_layernorm(512);
instantiate_layernorm(768);
instantiate_layernorm(1024);

}
