#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Inverted dropout (training). keep_i = rng_uniform(seed, i) >= p; out_i = keep_i ? x_i/(1-p) : 0.
// The keep-mask is a pure function of (seed, flat index) via the counter-based RNG, so the backward
// recomputes the SAME mask from the same seed — no mask tensor is stored. p in [0,1); inv_keep =
// 1/(1-p) is passed from the host. T templated (fp16/bf16/fp32). One thread per element.
// ---------------------------------------------------------------------------

template <typename T>
kernel void dropout_fwd(device const T *x        [[buffer(0)]],
                        device T       *out      [[buffer(1)]],
                        constant uint  &seed     [[buffer(2)]],
                        constant float &p        [[buffer(3)]],
                        constant float &inv_keep [[buffer(4)]],
                        constant uint  &n        [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) { return; }
    const float u = rng_uniform(seed, gid, 0u);
    out[gid] = (u >= p) ? T(float(x[gid]) * inv_keep) : T(0.0f);
}

// Backward: dx_i = keep_i ? dy_i/(1-p) : 0 (same mask recomputed from seed).
template <typename T>
kernel void dropout_bwd(device const T *dy       [[buffer(0)]],
                        device T       *dx       [[buffer(1)]],
                        constant uint  &seed     [[buffer(2)]],
                        constant float &p        [[buffer(3)]],
                        constant float &inv_keep [[buffer(4)]],
                        constant uint  &n        [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) { return; }
    const float u = rng_uniform(seed, gid, 0u);
    dx[gid] = (u >= p) ? T(float(dy[gid]) * inv_keep) : T(0.0f);
}

#define instantiate_dropout(type_name, T)                                        \
  template [[host_name("dropout_fwd_" #type_name)]] [[kernel]] void               \
  dropout_fwd<T>(device const T *x [[buffer(0)]], device T *out [[buffer(1)]],     \
    constant uint &seed [[buffer(2)]], constant float &p [[buffer(3)]],            \
    constant float &inv_keep [[buffer(4)]], constant uint &n [[buffer(5)]],        \
    uint gid [[thread_position_in_grid]]);                                         \
  template [[host_name("dropout_bwd_" #type_name)]] [[kernel]] void               \
  dropout_bwd<T>(device const T *dy [[buffer(0)]], device T *dx [[buffer(1)]],     \
    constant uint &seed [[buffer(2)]], constant float &p [[buffer(3)]],            \
    constant float &inv_keep [[buffer(4)]], constant uint &n [[buffer(5)]],        \
    uint gid [[thread_position_in_grid]]);

instantiate_dropout(float32, float)
instantiate_dropout(float16, half)
instantiate_dropout(bfloat16, bf16)
