/**
 * @file
 * @brief In-register dequantization of quantized weight blocks into half tiles.
 *
 * "Marlin's method" on Apple: quantized weights are stored block-wise (a small fp16 scale +
 * packed low-precision codes), dequantized to `half` here, then fed to a standard
 * `simdgroup_matrix` MMA. The dequant math is pure IEEE-fp16 bit arithmetic, valid on Metal
 * `half` verbatim. Block layouts mirror llama.cpp's GGUF formats (ggml-common.h); the dequant
 * constants follow llama.cpp (ggml-metal.metal) and Marlin (dequant.h).
 *
 * A "format" is a small struct exposing `block_k` (weights per block), `block_bytes`, and a
 * `dequant(device const uchar* base, int col) -> half` for the weight at column `col` of the
 * block starting at byte `base`. `dequant_into_shared<FMT>` cooperatively dequantizes a tile.
 */
#pragma once
#include "../../../../common/common.metal"
#include "../../../../types/types.metal"
#include "dequant_tables.metal"   // GGUF i-quant lattice/codebook constant tables (namespace mittens)

namespace mittens {

// ---- integer dot primitive (the "idot/imma" unlock) ------------------------------------------
// Apple's simdgroup_matrix has no integer path, so activation-quantized kernels accumulate in
// int32 on the ALUs. idot4 is the dp4a equivalent (4 packed signed int8 per uint -> int32),
// modeled on BitNet's __dp4a; a per-lane int8 GEMV then simd_sum-reduces (see qgemv_w8a8/_w2a8).
METAL_FUNC int idot4(uint a, uint b) {
    int s = 0;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 4; i++) {
        s += (int)(char)((a >> (8 * i)) & 0xffu) * (int)(char)((b >> (8 * i)) & 0xffu);
    }
    return s;
}

// ---- codebook / lookup-table dequant primitive (the second new style) -------------------------
// Packed bits index a constant table instead of doing bit arithmetic. kvalues_iq4nl is GGUF's
// 16-entry non-linear fp codebook (ggml-common.h); a nibble indexes it, then * the block scale.
constant const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// ---- iq4_nl : { half d; uint8 qs[16]; }  — 18 bytes, 32 weights, value = d * kvalues_iq4nl[nib]
//   (q4_0-style nibble layout: col<16 -> low nibble of qs[col], else high nibble of qs[col-16]). ----
struct iq4_nl {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 18;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        device const uchar* qs = base + 2;
        const int nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return d * half(kvalues_iq4nl[nib]);
    }
};

// ---- iq4_xs : 256-superblock IQ4_NL. { half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]; }
//   = 136 bytes. 8 sub-blocks of 32; each has a 6-bit scale ls = (4 low bits in scales_l | 2 high
//   bits in scales_h) − 32, so value = d·ls · kvalues_iq4nl[nibble]. (ggml-common.h block_iq4_xs.) ----
struct iq4_xs {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 136;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        const ushort scales_h = ((device const ushort*)(base + 2))[0];
        device const uchar* scales_l = base + 4;       // 4 bytes
        device const uchar* qs = base + 8;             // 128 bytes
        const int ib = col >> 5;                       // sub-block 0..7
        const int local = col & 31;
        const int sl = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
        const int sh = (scales_h >> (2 * ib)) & 0x3;
        const int ls = (sl | (sh << 4)) - 32;          // 6-bit signed sub-scale
        const half dl = d * half(ls);
        const int nib = (local < 16) ? (qs[16 * ib + local] & 0x0F)
                                     : (qs[16 * ib + (local - 16)] >> 4);
        return dl * half(kvalues_iq4nl[nib]);
    }
};

// ---- iq2_xxs : E8-lattice 2.0625 bpw. { half d; uint16 qs[32]; } = 66 bytes, 256 weights.
//   Per block-of-32 (4 groups of 8): 4 uint16 = grid indices (aux_g) + signs/scale (aux_s). Each
//   8-bit grid index selects an iq2xxs_grid entry (8 packed uint8 magnitudes); a 7-bit ksigns index
//   gives the 8 signs; the top 4 bits of aux_s give the sub-scale. (ggml-metal dequantize_iq2_xxs.) ----
struct iq2_xxs {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 66;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        device const ushort* qs = (device const ushort*)(base + 2);
        const int ib32 = col >> 5, p = col & 31, sub = p >> 3, elem = p & 7;
        device const ushort* q2 = qs + 4 * ib32;
        const uint aux_g = (uint)q2[0] | ((uint)q2[1] << 16);
        const uint aux_s = (uint)q2[2] | ((uint)q2[3] << 16);
        const uint g = (aux_g >> (8 * sub)) & 0xff;
        const uint gv = (uint)((iq2xxs_grid[g] >> (8 * elem)) & 0xffUL);
        const uchar signs = ksigns_iq2xs[(aux_s >> (7 * sub)) & 127];
        const half dl = d * (0.5h + half((aux_s >> 28) & 0xf)) * 0.25h;
        const half sgn = (signs & kmask_iq2xs[elem]) ? -1.0h : 1.0h;
        return dl * half(gv) * sgn;
    }
};

// ---- iq2_xs : E8-lattice 2.3125 bpw. { half d; uint16 qs[32]; uint8 scales[8]; } = 74 bytes, 256
//   weights. Each uint16: low 9 bits = iq2xs_grid index (512), high 7 = ksigns index; 4-bit
//   per-half scale from scales[ib32]. (ggml-metal dequantize_iq2_xs.) ----
struct iq2_xs {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 74;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        device const ushort* qs = (device const ushort*)(base + 2);
        device const uchar* scales = base + 66;
        const int ib32 = col >> 5, p = col & 31, il = p >> 4, sub2 = (p & 15) >> 3, elem = p & 7;
        const ushort idx16 = qs[4 * ib32 + 2 * il + sub2];
        const uint g = idx16 & 511;
        const uchar signs = ksigns_iq2xs[idx16 >> 9];
        const int sc = (scales[ib32] >> (4 * il)) & 0xF;
        const half dl = d * (0.5h + half(sc)) * 0.25h;
        const uint gv = (uint)((iq2xs_grid[g] >> (8 * elem)) & 0xffUL);
        const half sgn = (signs & kmask_iq2xs[elem]) ? -1.0h : 1.0h;
        return dl * half(gv) * sgn;
    }
};

// ---- iq3_xxs : E8-lattice 3.0625 bpw. { half d; uint8 qs[96]; } = 98 bytes, 256 weights. First
//   64 bytes of qs = 8-bit grid indices (8 per block-of-32); the next 32 = uint16 sign/scale (gas).
//   Each iq3xxs_grid entry is a uint32 of 4 magnitudes. (ggml-metal dequantize_iq3_xxs.) ----
struct iq3_xxs {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 98;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        device const uchar* qs = base + 2;
        const int ib32 = col >> 5, p = col & 31, il = p >> 4, w = p & 15, r = w >> 2, i = w & 3;
        device const uchar* q3 = qs + 8 * ib32;
        device const ushort* gas = (device const ushort*)(qs + 64) + 2 * ib32;
        const uint aux32 = (uint)gas[0] | ((uint)gas[1] << 16);
        const uint gv = (iq3xxs_grid[q3[4 * il + r]] >> (8 * i)) & 0xff;
        const uchar signs = ksigns_iq2xs[(aux32 >> (14 * il + 7 * (r >> 1))) & 127];
        const half dl = d * (0.5h + half(aux32 >> 28)) * 0.5h;
        const half sgn = (signs & kmask_iq2xs[i + 4 * (r & 1)]) ? -1.0h : 1.0h;
        return dl * half(gv) * sgn;
    }
};

// ---- iq1_s : 1.5625 bpw. { half d; uint8 qs[32]; uint16 qh[8]; } = 50 bytes, 256 weights. Per
//   half: two iq1s_grid_gpu entries (index = qs byte | high bits from qh); 3-bit scale + a sign in
//   qh give dl and the ml offset (value = dl·nibble + ml). (ggml-metal dequantize_iq1_s.) ----
struct iq1_s {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 50;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        device const uchar* qs = base + 2;
        device const ushort* qh = (device const ushort*)(base + 34);
        const int ib32 = col >> 5, p = col & 31, il = p >> 4, w = p & 15;
        const int which = w >> 2, i = w & 3;        // which: 0/1 -> grid1 lo/hi, 2/3 -> grid2 lo/hi
        device const uchar* qsp = qs + 4 * ib32 + 2 * il;
        const ushort qhv = qh[ib32];
        const half dl = d * half(2 * ((qhv >> 12) & 7) + 1);
        const half ml = dl * ((qhv & 0x8000) ? half(-1.0h - IQ1S_DELTA) : half(-1.0h + IQ1S_DELTA));
        const uint h = (uint)(qhv >> (6 * il));
        const uint gi = (which >> 1) == 0 ? (qsp[0] | ((h << 8) & 0x700))
                                          : (qsp[1] | ((h << 5) & 0x700));
        const uint b = (iq1s_grid_gpu[gi] >> (8 * i)) & 0xff;
        const uint nib = (which & 1) ? (b >> 4) : (b & 0xF);
        return dl * half(nib) + ml;
    }
};

// ---- q8_0 : { half d; int8 qs[32]; }  — 34 bytes, 32 weights/block, value = d * q ----
struct q8_0 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 34;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];      // fp16 scale at offset 0
        const char q = ((device const char*)(base + 2))[col];  // signed int8 codes at offset 2
        return d * half(q);
    }
    // integer path: the raw int8 code and the per-group (block) scale, kept separate.
    static METAL_FUNC int  code(device const uchar* base, int col) { return (int)((device const char*)(base + 2))[col]; }
    static METAL_FUNC half gscale(device const uchar* base)        { return ((device const half*)base)[0]; }
};

// ---- q4_0 : { half d; uint8 qs[16]; } — 18 bytes, 32 weights/block. Nibble packing (ggml):
//   weight col<16 -> qs[col]&0xF ; col>=16 -> qs[col-16]>>4 ; value = d * (nibble - 8). ----
struct q4_0 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 18;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];      // fp16 scale at offset 0
        device const uchar* qs = base + 2;                 // 16 packed-nibble bytes
        const int nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return d * half(nib - 8);
    }
};

// ---- q4_K : { half d; half dmin; uint8 scales[12]; uint8 qs[128]; } — 144 bytes, 256/block.
//   8 sub-blocks of 32; each has a 6-bit scale `sc` and 6-bit min `m` (packed in `scales`,
//   extracted GGUF-style). value = (d*sc)*nibble - (dmin*m). ----
struct q4_K {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 144;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d    = ((device const half*)base)[0];
        const half dmin = ((device const half*)base)[1];
        device const uchar* scales = base + 4;
        device const uchar* qs     = base + 16;
        const int chunk = col / 64;        // 0..3
        const int pos   = col % 64;        // 0..63
        int sub, nib;
        if (pos < 32) { sub = chunk * 2;     nib = qs[chunk * 32 + pos]        & 0x0F; }
        else          { sub = chunk * 2 + 1; nib = qs[chunk * 32 + (pos - 32)] >> 4;   }
        // get_scale_min_k4(sub): unpack the 6-bit scale `sc` and min `m`
        uchar sc, m;
        if (sub < 4) { sc = scales[sub] & 63; m = scales[sub + 4] & 63; }
        else {
            sc = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4);
            m  = (scales[sub + 4] >> 4)   | ((scales[sub]     >> 6) << 4);
        }
        return d * half(sc) * half(nib) - dmin * half(m);
    }
};

// ---- kU4B8 : GPTQ/Marlin grouped int4, group=128. { half scale; uint8 qs[64]; } — 66 bytes.
//   unsigned 4-bit with bias 8; value = scale * (nibble - 8). Nibble packing like q4_0 (col<64 ->
//   qs[col]&0xF ; col>=64 -> qs[col-64]>>4). Larger group than q4_0 (more compression). ----
struct kU4B8 {
    constant static constexpr const int block_k     = 128;
    constant static constexpr const int block_bytes = 66;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = ((device const half*)base)[0];
        device const uchar* qs = base + 2;
        const int nib = (col < 64) ? (qs[col] & 0x0F) : (qs[col - 64] >> 4);
        return scale * half(nib - 8);
    }
};

// ---- kU4 : AWQ grouped int4, group=128, per-group zero-point. { half scale; half zp;
//   uint8 qs[64]; } — 68 bytes. value = scale * (nibble - zp). ----
struct kU4 {
    constant static constexpr const int block_k     = 128;
    constant static constexpr const int block_bytes = 68;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = ((device const half*)base)[0];
        const half zp    = ((device const half*)base)[1];
        device const uchar* qs = base + 4;
        const int nib = (col < 64) ? (qs[col] & 0x0F) : (qs[col - 64] >> 4);
        return scale * (half(nib) - zp);
    }
};

// ---- float-code decoders (pure IEEE bit/field math; widen to half) ----
// fp8 e4m3 (1-4-3, bias 7, no inf): value = (-1)^s * (1 + m/8) * 2^(e-7), subnormal at e==0.
METAL_FUNC half tk_e4m3_decode(uchar v) {
    const uint e = (v >> 3) & 0xF;
    const uint m = v & 0x7;
    half val = (e == 0) ? (half(m) * 0.125h * 0.015625h)      // (m/8) * 2^-6
                        : (1.0h + half(m) * 0.125h) * metal::exp2(half((int)e - 7));
    return ((v >> 7) & 1) ? -val : val;
}
// fp4 e2m1 (1-2-1, bias 1): values 0,.5,1,1.5,2,3,4,6 (+sign).
METAL_FUNC half tk_e2m1_decode(uint nib) {
    const uint e = (nib >> 1) & 0x3;
    const uint m = nib & 0x1;
    half val = (e == 0) ? (m ? 0.5h : 0.0h)
                        : (1.0h + half(m) * 0.5h) * metal::exp2(half((int)e - 1));
    return ((nib >> 3) & 1) ? -val : val;
}

// ---- fp8_e4m3 : per-group (32) half-scaled fp8. { half scale; uint8 qs[32]; } — 34 bytes.
//   value = scale * e4m3(q). ----
struct fp8_e4m3 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 34;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        return ((device const half*)base)[0] * tk_e4m3_decode((base + 2)[col]);
    }
};

// ---- fp4_e2m1 : per-group (32) half-scaled fp4 (nibbles, q4_0-style packing). 18 bytes. ----
struct fp4_e2m1 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 18;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        device const uchar* qs = base + 2;
        const uint nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return ((device const half*)base)[0] * tk_e2m1_decode(nib);
    }
};

// ---- mxfp8 : OCP microscaling — 32-element block, e8m0 power-of-two block scale + fp8 e4m3.
//   { uint8 e8m0; uint8 qs[32]; } — 33 bytes. value = 2^(e8m0-127) * e4m3(q). ----
struct mxfp8 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 33;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = metal::exp2(half((int)base[0] - 127));
        return scale * tk_e4m3_decode((base + 1)[col]);
    }
};

// ---- nvfp4 : 16-element block, fp8 e4m3 block scale + fp4 e2m1 codes (nibbles).
//   { uint8 e4m3_scale; uint8 qs[8]; } — 9 bytes. value = e4m3(scale) * e2m1(nib). ----
struct nvfp4 {
    constant static constexpr const int block_k     = 16;
    constant static constexpr const int block_bytes = 9;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = tk_e4m3_decode(base[0]);
        device const uchar* qs = base + 1;
        const uint nib = (col < 8) ? (qs[col] & 0x0F) : (qs[col - 8] >> 4);
        return scale * tk_e2m1_decode(nib);
    }
};

// ---- mxfp4 : OCP microscaling — 32-element block, e8m0 power-of-two block scale + fp4 e2m1 codes
//   (nibbles). { uint8 e8m0; uint8 qs[16]; } — 17 bytes. value = 2^(e8m0-127) * e2m1(nib). ----
struct mxfp4 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 17;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = metal::exp2(half((int)base[0] - 127));
        device const uchar* qs = base + 1;
        const uint nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return scale * tk_e2m1_decode(nib);
    }
};

// ---- bitnet : BitNet b1.58 ternary weights {-1,0,+1}, group 32, per-group absmean scale.
//   2-bit codes packed 4/byte (code in {0,1,2} -> value scale*(code-1)). { half scale; uint8 qs[8]; }
//   = 10 bytes. (BitNet's GPU kernel uses int8×int2 dp4a; Apple has no int matmul, so we dequant
//   ternary -> half and use the standard simdgroup MMA, like every other format here.) ----
struct bitnet {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 10;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = ((device const half*)base)[0];
        device const uchar* qs = base + 2;                 // 8 bytes, 4 ternary codes each
        const uint code = (qs[col >> 2] >> ((col & 3) * 2)) & 0x3;
        return scale * half((int)code - 1);                // 0->-1, 1->0, 2->+1
    }
    // integer path (W2A8): the ternary code in {-1,0,+1} and the per-group absmean scale.
    static METAL_FUNC int code(device const uchar* base, int col) {
        device const uchar* qs = base + 2;
        return (int)((qs[col >> 2] >> ((col & 3) * 2)) & 0x3) - 1;
    }
    static METAL_FUNC half gscale(device const uchar* base) { return ((device const half*)base)[0]; }
};

// ============================ Phase 3: GGUF k-quant + legacy fan-out ============================
// Byte layouts match ggml-common.h; per-column decoders mirror the ggml CPU dequantize_row_* refs.

// ---- q4_1 : { half d; half m; uint8 qs[16]; } — 20 bytes, 32/block. value = d*nibble + m. ----
struct q4_1 {
    constant static constexpr const int block_k = 32, block_bytes = 20;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0], m = ((device const half*)base)[1];
        device const uchar* qs = base + 4;
        const int nib = (col < 16) ? (qs[col] & 0xF) : (qs[col - 16] >> 4);
        return d * half(nib) + m;
    }
};

// ---- q5_0 : { half d; uint8 qh[4]; uint8 qs[16]; } — 22 bytes. value = d*(q-16), q = nibble |
//   (5th bit = bit `col` of the qh uint32). ----
struct q5_0 {
    constant static constexpr const int block_k = 32, block_bytes = 22;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        const uint qh = (uint)base[2] | ((uint)base[3] << 8) | ((uint)base[4] << 16) | ((uint)base[5] << 24);
        device const uchar* qs = base + 6;
        const int nib = (col < 16) ? (qs[col] & 0xF) : (qs[col - 16] >> 4);
        const int q = nib | (((qh >> col) & 1) << 4);
        return d * half(q - 16);
    }
};

// ---- q5_1 : { half d; half m; uint8 qh[4]; uint8 qs[16]; } — 24 bytes. value = d*q + m. ----
struct q5_1 {
    constant static constexpr const int block_k = 32, block_bytes = 24;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0], m = ((device const half*)base)[1];
        const uint qh = (uint)base[4] | ((uint)base[5] << 8) | ((uint)base[6] << 16) | ((uint)base[7] << 24);
        device const uchar* qs = base + 8;
        const int nib = (col < 16) ? (qs[col] & 0xF) : (qs[col - 16] >> 4);
        const int q = nib | (((qh >> col) & 1) << 4);
        return d * half(q) + m;
    }
};

// ---- q2_K : { uint8 scales[16]; uint8 qs[64]; half d; half dmin; } — 84 bytes, 256/block.
//   16 sub-blocks of 16; scales byte = 4-bit dl-scale | 4-bit min. value = d*sc*q - dmin*m. ----
struct q2_K {
    constant static constexpr const int block_k = 256, block_bytes = 84;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        device const uchar* scales = base; device const uchar* qs = base + 16;
        const half d = ((device const half*)(base + 80))[0], dmin = ((device const half*)(base + 82))[0];
        const int chunk = col >> 7, pos = col & 127, sidx = pos >> 5, sub = (pos >> 4) & 1, l = pos & 15;
        const int is = chunk * 8 + sidx * 2 + sub;
        const int q = (qs[chunk * 32 + sub * 16 + l] >> (2 * sidx)) & 3;
        return d * half(scales[is] & 0xF) * half(q) - dmin * half(scales[is] >> 4);
    }
};

// ---- q3_K : { uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; half d; } — 110 bytes, 256/block.
//   low 2 bits in qs, high bit in hmask; 16 6-bit signed scales packed (kmask). value = d*(sc-32)*q3. ----
struct q3_K {
    constant static constexpr const int block_k = 256, block_bytes = 110;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        device const uchar* hmask = base; device const uchar* qs = base + 32; device const uchar* sca = base + 96;
        const half d = ((device const half*)(base + 108))[0];
        const int chunk = col >> 7, pos = col & 127, sidx = pos >> 5, sub = (pos >> 4) & 1, l = pos & 15;
        const int is = chunk * 8 + sidx * 2 + sub;
        const int low2 = (qs[chunk * 32 + sub * 16 + l] >> (2 * sidx)) & 3;
        const int hb = (hmask[sub * 16 + l] & (1 << (chunk * 4 + sidx))) ? 1 : 0;
        const int q3v = (low2 | (hb << 2)) - 4;
        const int w = is >> 2, b = is & 3; int s;
        if (w == 0)      s = (sca[b] & 0xF)        | ((sca[8 + b] & 3) << 4);
        else if (w == 1) s = (sca[4 + b] & 0xF)    | (((sca[8 + b] >> 2) & 3) << 4);
        else if (w == 2) s = ((sca[b] >> 4) & 0xF) | (((sca[8 + b] >> 4) & 3) << 4);
        else             s = ((sca[4 + b] >> 4) & 0xF) | (((sca[8 + b] >> 6) & 3) << 4);
        return d * half(s - 32) * half(q3v);
    }
};

// ---- q5_K : { half d; half dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]; } — 176 bytes.
//   8 sub-blocks of 32; 6-bit scale+min (get_scale_min_k4, as q4_K); 5-bit q = nibble | (qh bit)<<4. ----
struct q5_K {
    constant static constexpr const int block_k = 256, block_bytes = 176;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0], dmin = ((device const half*)(base + 2))[0];
        device const uchar* sca = base + 4; device const uchar* qh = base + 16; device const uchar* qs = base + 48;
        const int chunk = col >> 6, pos = col & 63, sub = pos >> 5, l = pos & 31;
        const int is = 2 * chunk + sub;
        const int nib = sub ? (qs[chunk * 32 + l] >> 4) : (qs[chunk * 32 + l] & 0xF);
        const int hb = (qh[l] & (1 << (2 * chunk + sub))) ? 1 : 0;
        const int q = nib + hb * 16;
        int sc, mn;
        if (is < 4) { sc = sca[is] & 63; mn = sca[is + 4] & 63; }
        else { sc = (sca[is + 4] & 0xF) | ((sca[is - 4] >> 6) << 4); mn = (sca[is + 4] >> 4) | ((sca[is] >> 6) << 4); }
        return d * half(sc) * half(q) - dmin * half(mn);
    }
};

// ---- q6_K : { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d; } — 210 bytes, 256/block.
//   16 sub-blocks of 16; 6-bit q = (4 low in ql | 2 high in qh) - 32; int8 scales. value = d*sc*q. ----
struct q6_K {
    constant static constexpr const int block_k = 256, block_bytes = 210;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        device const uchar* ql = base; device const uchar* qh = base + 128;
        device const char* sca = (device const char*)(base + 192);
        const half d = ((device const half*)(base + 208))[0];
        const int chunk = col >> 7, pos = col & 127, group = pos >> 5, l = pos & 31;
        const int ql_byte = ql[chunk * 64 + l + 32 * (group & 1)];
        const int nib = (group & 2) ? (ql_byte >> 4) : (ql_byte & 0xF);
        const int hbits = (qh[chunk * 32 + l] >> (2 * group)) & 3;
        const int q = (nib | (hbits << 4)) - 32;
        const int sc_idx = chunk * 8 + (l >> 4) + group * 2;
        return d * half((int)sca[sc_idx]) * half(q);
    }
};

// Cooperatively dequantize an (BN x BK) weight tile into a shared half tile. `kb` is the K-tile
// index in units of BK (the MMA K-step). The quant grouping (FMT::block_k) is DECOUPLED from BK:
// each tile column maps to its quant block via the global K index, so large blocks (e.g. q4_K's
// 256) work with a small BK. Requires FMT::block_k % BK == 0 and K % FMT::block_k == 0.
//   Packed layout: block(n, b) starts at byte (n*(K/FMT::block_k) + b) * FMT::block_bytes.
//   `group_threads` = total threads in the threadgroup; `threadIdx` = flat thread index.
template<typename FMT, int BN, int BK>
METAL_FUNC void dequant_into_shared(threadgroup st<half, BN, BK>& dst,
                                    device const uchar* Wq, int N, int K,
                                    int by, int kb, int group_threads, uint threadIdx) {
    const int blocks_per_row = K / FMT::block_k;
    for (int e = (int)threadIdx; e < BN * BK; e += group_threads) {
        const int row  = e / BK;
        const int tcol = e % BK;
        const int grow = by * BN + row;
        const int gk   = kb * BK + tcol;                 // global K column
        const int blk  = gk / FMT::block_k;              // which quant block
        const int cib  = gk % FMT::block_k;              // column within that block
        device const uchar* base = Wq + (uint)(grow * blocks_per_row + blk) * FMT::block_bytes;
        dst[int2(row, tcol)] = FMT::dequant(base, cib);
    }
}

// Dequantize an (RT::rows x RT::cols) weight tile DIRECTLY into the simdgroup register fragment —
// no threadgroup round-trip, no barrier (Marlin's "zero-shuffle" idea on Apple). Each lane fills
// only its own 2 elements per 8x8 subtile, using the substrate's lane->(row,col) fragment map
// (mirrors load(rt, gl) in global_to_register.metal): thread_elements()[0]/[1] = the weights at
// (row = by*rows + i*8 + simd_y, col = kb*cols + j*8 + simd_x [+1]). `by`/`kb` are tile-block indices.
template<typename FMT, typename RT>
METAL_FUNC void dequant_into_register(thread RT& dst, device const uchar* Wq, int N, int K,
                                      int by, int kb, uint laneid) {
    const int qid    = (int)laneid / 4;
    const int simd_y = (qid & 4) + ((int)laneid / 2) % 4;
    const int simd_x = (qid & 2) * 2 + ((int)laneid % 2) * 2;
    const int bpr = K / FMT::block_k;
    #pragma clang loop unroll(full)
    for (int i = 0; i < RT::height; i++) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < RT::width; j++) {
            const int grow = by * RT::rows + i * mittens::TILE_DIM + simd_y;
            const int gc   = kb * RT::cols + j * mittens::TILE_DIM + simd_x;
            const int blk0 = gc / FMT::block_k,       cib0 = gc % FMT::block_k;
            const int blk1 = (gc + 1) / FMT::block_k, cib1 = (gc + 1) % FMT::block_k;
            device const uchar* b0 = Wq + (uint)(grow * bpr + blk0) * FMT::block_bytes;
            device const uchar* b1 = Wq + (uint)(grow * bpr + blk1) * FMT::block_bytes;
            dst.tiles[i][j].data.thread_elements()[0] = FMT::dequant(b0, cib0);
            dst.tiles[i][j].data.thread_elements()[1] = FMT::dequant(b1, cib1);
        }
    }
}

} // namespace mittens
