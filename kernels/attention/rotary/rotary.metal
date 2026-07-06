#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// Rotary positional embedding (RoPE), split-half / GPT-NeoX convention,
// matching mx.fast.rope(..., traditional=False). bf16 I/O, fp32 compute.
//
// With halves x1 = x[..., :D/2], x2 = x[..., D/2:] and per-position cos/sin of
// shape (N, D/2):
//     o1 = x1*cos - x2*sin
//     o2 = x2*cos + x1*sin
//
// cos/sin are precomputed and passed in (the kernel needs no trig op).
// Geometry: FLAT — one thread per 4 rotation pairs (vectorized bf16_4
// loads/stores), 256-thread groups. The previous one-simdgroup-per-row layout
// gave each threadgroup only D elements of work and scalar substrate loads,
// measuring ~2.3x slower than mx.fast.rope; this shape matches it.
// x is flattened to (M, D) with M = B*H*N; the sequence position is row % N.
// ---------------------------------------------------------------------------
template <int D>
kernel void rotary(device   bf16 *x    [[buffer(0)]],
                   device   bf16 *cosb [[buffer(1)]],
                   device   bf16 *sinb [[buffer(2)]],
                   device   bf16 *o    [[buffer(3)]],
                   constant uint &N    [[buffer(4)]],   // sequence length
                   constant uint &M    [[buffer(5)]],   // rows = B*H*N
                   uint tid [[thread_position_in_grid]]) {
    constexpr int D2 = D / 2;
    static_assert(D2 % 4 == 0, "D/2 must be divisible by 4");
    constexpr int QPR = D2 / 4;                  // 4-pair quads per row
    if (tid >= M * (uint)QPR) return;
    const int row = (int)(tid / QPR);
    const int p4  = (int)(tid % QPR) * 4;        // first pair index of this quad
    const long xb = (long)row * D;
    const long cs = (long)(row % (int)N) * D2 + p4;

    const float4 a = float4(((device const bf16_4*)(x + xb + p4))[0]);
    const float4 b = float4(((device const bf16_4*)(x + xb + D2 + p4))[0]);
    const float4 c = float4(((device const bf16_4*)(cosb + cs))[0]);
    const float4 s = float4(((device const bf16_4*)(sinb + cs))[0]);
    ((device bf16_4*)(o + xb + p4))[0]      = bf16_4(a * c - b * s);
    ((device bf16_4*)(o + xb + D2 + p4))[0] = bf16_4(b * c + a * s);
}

#define instantiate_rotary(DVAL)                                              \
  template [[host_name("rotary_" #DVAL)]] [[kernel]] void                     \
  rotary<DVAL>(device   bf16 *x    [[buffer(0)]],                             \
               device   bf16 *cosb [[buffer(1)]],                            \
               device   bf16 *sinb [[buffer(2)]],                            \
               device   bf16 *o    [[buffer(3)]],                            \
               constant uint &N    [[buffer(4)]],                            \
               constant uint &M    [[buffer(5)]],                            \
               uint tid [[thread_position_in_grid]]);

instantiate_rotary(64);
instantiate_rotary(128);

// ---------------------------------------------------------------------------
// Rotary, GPT-J *interleaved* convention, matching mx.fast.rope(traditional=True).
// Rotates adjacent pairs (x[2p], x[2p+1]) rather than the two halves:
//     o[2p]   = x[2p]*cos[p] - x[2p+1]*sin[p]
//     o[2p+1] = x[2p]*sin[p] + x[2p+1]*cos[p]
// cos/sin are (N, D/2) (one entry per pair). Same flat geometry: one thread
// per 4 pairs = 8 contiguous elements (two bf16_4 loads), pairs stay in-lane.
// ---------------------------------------------------------------------------
template <int D>
kernel void rotary_interleaved(device   bf16 *x    [[buffer(0)]],
                               device   bf16 *cosb [[buffer(1)]],
                               device   bf16 *sinb [[buffer(2)]],
                               device   bf16 *o    [[buffer(3)]],
                               constant uint &N    [[buffer(4)]],
                               constant uint &M    [[buffer(5)]],
                               uint tid [[thread_position_in_grid]]) {
    constexpr int D2 = D / 2;
    static_assert(D % 8 == 0, "interleaved rotary needs D divisible by 8");
    constexpr int QPR = D2 / 4;
    if (tid >= M * (uint)QPR) return;
    const int row = (int)(tid / QPR);
    const int p4  = (int)(tid % QPR) * 4;
    const long xb = (long)row * D + 2 * p4;      // 8 contiguous elements
    const long cs = (long)(row % (int)N) * D2 + p4;

    const float4 e0 = float4(((device const bf16_4*)(x + xb))[0]);      // pairs p4, p4+1
    const float4 e1 = float4(((device const bf16_4*)(x + xb + 4))[0]);  // pairs p4+2, p4+3
    const float4 c = float4(((device const bf16_4*)(cosb + cs))[0]);
    const float4 s = float4(((device const bf16_4*)(sinb + cs))[0]);
    ((device bf16_4*)(o + xb))[0] = bf16_4(float4(
        e0.x * c.x - e0.y * s.x, e0.x * s.x + e0.y * c.x,
        e0.z * c.y - e0.w * s.y, e0.z * s.y + e0.w * c.y));
    ((device bf16_4*)(o + xb + 4))[0] = bf16_4(float4(
        e1.x * c.z - e1.y * s.z, e1.x * s.z + e1.y * c.z,
        e1.z * c.w - e1.w * s.w, e1.z * s.w + e1.w * c.w));
}

#define instantiate_rotary_interleaved(DVAL)                                   \
  template [[host_name("rotary_interleaved_" #DVAL)]] [[kernel]] void          \
  rotary_interleaved<DVAL>(device   bf16 *x    [[buffer(0)]],                  \
                           device   bf16 *cosb [[buffer(1)]],                  \
                           device   bf16 *sinb [[buffer(2)]],                  \
                           device   bf16 *o    [[buffer(3)]],                  \
                           constant uint &N    [[buffer(4)]],                  \
                           constant uint &M    [[buffer(5)]],                  \
                           uint tid [[thread_position_in_grid]]);

instantiate_rotary_interleaved(64);
instantiate_rotary_interleaved(128);

}
