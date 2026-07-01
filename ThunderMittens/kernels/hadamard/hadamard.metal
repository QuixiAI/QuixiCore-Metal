#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// Walsh-Hadamard transform over the final axis, one SIMDGROUP (32 lanes) per row.
// Each lane owns E = D/32 CONSECUTIVE elements: the low log2(E) index bits are
// butterflied locally in registers, the 5 lane bits via simd_shuffle_xor — no
// threadgroup memory and no barriers. (FWHT butterfly stages act on independent
// index bits, so the stage order is free.) The previous version used one
// D-thread threadgroup per row with log2(D) full barriers and half the threads
// idle per round; at D=128 it measured 0.44x of a plain matmul against H.
template <typename T, int D>
kernel void hadamard(device const T *x [[buffer(0)]],
                     device T *out [[buffer(1)]],
                     constant float &scale [[buffer(2)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    constexpr int E = D / 32;
    static_assert(E >= 1, "D must be >= 32");
    float v[E];
    const long base = (long)row * D + (long)lane * E;
    #pragma clang loop unroll(full)
    for (int i = 0; i < E; ++i) v[i] = float(x[base + i]);

    // local butterflies over the in-lane index bits
    #pragma clang loop unroll(full)
    for (int h = 1; h < E; h <<= 1) {
        #pragma clang loop unroll(full)
        for (int i = 0; i < E; ++i) {
            if ((i & h) == 0) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
        }
    }
    // cross-lane butterflies over the 5 lane bits
    #pragma clang loop unroll(full)
    for (int m = 1; m < 32; m <<= 1) {
        const bool upper = (lane & (uint)m) != 0;
        #pragma clang loop unroll(full)
        for (int i = 0; i < E; ++i) {
            const float p = metal::simd_shuffle_xor(v[i], (ushort)m);
            v[i] = upper ? (p - v[i]) : (v[i] + p);
        }
    }
    #pragma clang loop unroll(full)
    for (int i = 0; i < E; ++i) out[base + i] = T(v[i] * scale);
}

#define instantiate_hadamard(type_name, T, DVAL)                            \
  template [[host_name("hadamard_" #type_name "_" #DVAL)]] [[kernel]] void \
  hadamard<T, DVAL>(device const T *x [[buffer(0)]],                        \
                    device T *out [[buffer(1)]],                            \
                    constant float &scale [[buffer(2)]],                    \
                    uint row [[threadgroup_position_in_grid]],              \
                    uint lane [[thread_index_in_simdgroup]]);

#define instantiate_hadamard_type(type_name, T) \
  instantiate_hadamard(type_name, T, 64)        \
  instantiate_hadamard(type_name, T, 128)       \
  instantiate_hadamard(type_name, T, 256)       \
  instantiate_hadamard(type_name, T, 512)

instantiate_hadamard_type(float32, float)
instantiate_hadamard_type(float16, half)
instantiate_hadamard_type(bfloat16, bf16)
