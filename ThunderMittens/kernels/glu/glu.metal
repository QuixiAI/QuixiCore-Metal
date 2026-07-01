#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

constant float GLU_GELU_COEF_A = 0.044715f;
constant float GLU_GELU_QUICK_COEF = -1.702f;
constant float GLU_SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
constant float GLU_SQRT_2_INV = 0.70710678118654752440084436210484f;

constant float GLU_ERF_P = 0.3275911f;
constant float GLU_ERF_A1 = 0.254829592f;
constant float GLU_ERF_A2 = -0.284496736f;
constant float GLU_ERF_A3 = 1.421413741f;
constant float GLU_ERF_A4 = -1.453152027f;
constant float GLU_ERF_A5 = 1.061405429f;

template <typename T>
METAL_FUNC float to_float(T x) {
    return float(x);
}

METAL_FUNC float glu_tanh(float x) {
    return 1.0f - 2.0f / (metal::exp(x + x) + 1.0f);
}

METAL_FUNC float glu_erf_approx(float x) {
    const float sx = x < 0.0f ? -1.0f : 1.0f;
    x = metal::abs(x);
    const float t = 1.0f / (1.0f + GLU_ERF_P * x);
    const float poly = (((((GLU_ERF_A5 * t + GLU_ERF_A4) * t) + GLU_ERF_A3) * t + GLU_ERF_A2) * t + GLU_ERF_A1) * t;
    const float y = 1.0f - poly * metal::exp(-x * x);
    return sx * y;
}

METAL_FUNC float glu_gelu_tanh(float x) {
    const float inner = GLU_SQRT_2_OVER_PI * x * (1.0f + GLU_GELU_COEF_A * x * x);
    return 0.5f * x * (1.0f + glu_tanh(inner));
}

METAL_FUNC float glu_gelu_erf(float x) {
    return 0.5f * x * (1.0f + glu_erf_approx(x * GLU_SQRT_2_INV));
}

METAL_FUNC float glu_eval(int mode, float x0, float x1, float alpha, float limit) {
    if (mode == 0) {
        return x0 * x1 * (x0 > 0.0f ? 1.0f : 0.0f);
    }
    if (mode == 1) {
        return glu_gelu_tanh(x0) * x1;
    }
    if (mode == 2) {
        return (x0 / (1.0f + metal::exp(-x0))) * x1;
    }
    if (mode == 3) {
        x0 = metal::min(x0, limit);
        x1 = metal::max(metal::min(x1, limit), -limit);
        return (x0 / (1.0f + metal::exp(-x0 * alpha))) * (1.0f + x1);
    }
    if (mode == 4) {
        return glu_gelu_erf(x0) * x1;
    }
    return (x0 * (1.0f / (1.0f + metal::exp(GLU_GELU_QUICK_COEF * x0)))) * x1;
}

// One thread per 4 elements (vec4 loads/stores; the scalar version measured 103 GB/s vs the
// ~400 GB/s the same shape reaches vectorized). The last thread of a ragged n takes the
// scalar tail.
template <typename T, int MODE>
kernel void glu(device const T *x [[buffer(0)]],
                device const T *gate [[buffer(1)]],
                device T *out [[buffer(2)]],
                constant uint &n [[buffer(3)]],
                constant float &alpha [[buffer(4)]],
                constant float &limit [[buffer(5)]],
                uint tid [[thread_position_in_grid]]) {
    using T4 = metal::vec<T, 4>;
    const uint base = tid * 4;
    if (base + 4 <= n) {
        const float4 x0 = float4(((device const T4*)(x + base))[0]);
        const float4 x1 = float4(((device const T4*)(gate + base))[0]);
        float4 r;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) r[i] = glu_eval(MODE, x0[i], x1[i], alpha, limit);
        ((device T4*)(out + base))[0] = T4(r);
    } else {
        for (uint i = base; i < n; ++i)
            out[i] = T(glu_eval(MODE, to_float(x[i]), to_float(gate[i]), alpha, limit));
    }
}

#define instantiate_glu(MODE_NAME, MODE_ID, type_name, T)                    \
  template [[host_name("glu_" #MODE_NAME "_" #type_name)]] [[kernel]] void  \
  glu<T, MODE_ID>(device const T *x [[buffer(0)]],                           \
                  device const T *gate [[buffer(1)]],                        \
                  device T *out [[buffer(2)]],                               \
                  constant uint &n [[buffer(3)]],                            \
                  constant float &alpha [[buffer(4)]],                       \
                  constant float &limit [[buffer(5)]],                       \
                  uint tid [[thread_position_in_grid]]);

#define instantiate_glu_mode(MODE_NAME, MODE_ID)                             \
  instantiate_glu(MODE_NAME, MODE_ID, float32, float)                        \
  instantiate_glu(MODE_NAME, MODE_ID, float16, half)                         \
  instantiate_glu(MODE_NAME, MODE_ID, bfloat16, bf16)

instantiate_glu_mode(reglu, 0)
instantiate_glu_mode(geglu, 1)
instantiate_glu_mode(swiglu, 2)
instantiate_glu_mode(swiglu_oai, 3)
instantiate_glu_mode(geglu_erf, 4)
instantiate_glu_mode(geglu_quick, 5)

} // namespace mittens
