#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// ---------------------------------------------------------------------------
// GDN / GatedDeltaNet linear attention (Qwen3-Next / Kimi-Linear-class hybrid mixer):
// per-timestep delta-rule recurrence over a per-(head, dv) state row S[dv, 0:Dk]:
//
//   S      *= g[t, hv]                      (scalar gate/decay, a multiplier in (0,1])
//   kv_mem  = k[t] . S                      (simd_sum over Dk lanes)
//   delta   = (v[t, dv] - kv_mem) * beta[t, hv]
//   S      += k[t] * delta                  (rank-1 delta correction)
//   y[t,dv] = q[t] . S                      (simd_sum)
//
// One simdgroup per (request, hv, dv): grid (Dv, 1, R*Hv), 32 lanes partition Dk
// (DK/32 fp32 state elements per lane; DK in {64, 128} compile-time). Varlen packed
// inputs via cu_seqlens; persistent fp32 state pool (num_slots, Hv, Dv, Dk) indexed by
// slot_mapping[req] — each simdgroup owns its (hv, dv) row exclusively, so the in-place
// pool update is race-free. GQA: hk = hv / (Hv/Hk). load_initial == 0 starts fresh
// prefills at S = 0 (the pool row is still overwritten at the end).
// Port of metal-forge gdn_linear_attention with the state promoted to fp32.
// ---------------------------------------------------------------------------
template <typename T, int DK>
kernel void gdn_recur(device const T *q            [[buffer(0)]],
                      device const T *k            [[buffer(1)]],
                      device const T *v            [[buffer(2)]],
                      device const T *g            [[buffer(3)]],
                      device const T *beta         [[buffer(4)]],
                      device float   *state_pool   [[buffer(5)]],
                      device const int *cu_seqlens   [[buffer(6)]],
                      device const int *slot_mapping [[buffer(7)]],
                      device T       *y            [[buffer(8)]],
                      constant int   &num_requests [[buffer(9)]],
                      constant int   &Hk           [[buffer(10)]],
                      constant int   &Hv           [[buffer(11)]],
                      constant int   &Dv           [[buffer(12)]],
                      constant int   &load_initial [[buffer(13)]],
                      uint3 gid [[threadgroup_position_in_grid]],
                      uint  lane [[thread_index_in_simdgroup]]) {
    static_assert(DK == 64 || DK == 128, "gdn_recur supports Dk in {64, 128}");
    constexpr int N_PER_T = DK / 32;
    const int req_idx = (int)gid.z / Hv;
    const int hv_idx = (int)gid.z % Hv;
    const int dv_idx = (int)gid.x;
    const int dk0 = (int)lane * N_PER_T;
    if (req_idx >= num_requests || dv_idx >= Dv) { return; }

    const int hk_idx = hv_idx / (Hv / Hk);
    const int seq_start = cu_seqlens[req_idx];
    const int seq_len = cu_seqlens[req_idx + 1] - seq_start;
    const long slot = slot_mapping[req_idx];
    device float *state_ptr = state_pool + ((slot * Hv + hv_idx) * (long)Dv + dv_idx) * DK;

    float state[N_PER_T];
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_PER_T; ++i) {
        state[i] = (load_initial != 0) ? state_ptr[dk0 + i] : 0.0f;
    }

    device const T *q_ = q + (long)seq_start * Hk * DK + hk_idx * DK;
    device const T *k_ = k + (long)seq_start * Hk * DK + hk_idx * DK;
    device const T *v_ = v + (long)seq_start * Hv * Dv + hv_idx * Dv;
    device const T *g_ = g + (long)seq_start * Hv;
    device const T *beta_ = beta + (long)seq_start * Hv;
    device T *y_ = y + (long)seq_start * Hv * Dv + hv_idx * Dv;

    using TN = metal::vec<T, N_PER_T>;   // lane owns N_PER_T CONTIGUOUS Dk elems -> vec load
    for (int t = 0; t < seq_len; ++t) {
        const float g_val = float(g_[hv_idx]);
        const TN kvec = ((device const TN*)(k_ + dk0))[0];   // k read once, reused below
        const TN qvec = ((device const TN*)(q_ + dk0))[0];
        float kv_mem = 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < N_PER_T; ++i) {
            state[i] *= g_val;
            kv_mem += state[i] * float(kvec[i]);
        }
        kv_mem = metal::simd_sum(kv_mem);

        const float delta = (float(v_[dv_idx]) - kv_mem) * float(beta_[hv_idx]);

        float out = 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < N_PER_T; ++i) {
            state[i] += float(kvec[i]) * delta;
            out += state[i] * float(qvec[i]);
        }
        out = metal::simd_sum(out);
        if (lane == 0) {
            y_[dv_idx] = T(out);
        }

        q_ += Hk * DK;
        k_ += Hk * DK;
        v_ += Hv * Dv;
        y_ += Hv * Dv;
        g_ += Hv;
        beta_ += Hv;
    }

    #pragma clang loop unroll(full)
    for (int i = 0; i < N_PER_T; ++i) {
        state_ptr[dk0 + i] = state[i];
    }
}

#define instantiate_gdn_recur(type_name, T, DKVAL)                              \
  template [[host_name("gdn_recur_" #type_name "_d" #DKVAL)]] [[kernel]] void   \
  gdn_recur<T, DKVAL>(device const T *q [[buffer(0)]],                           \
                      device const T *k [[buffer(1)]],                           \
                      device const T *v [[buffer(2)]],                           \
                      device const T *g [[buffer(3)]],                           \
                      device const T *beta [[buffer(4)]],                        \
                      device float *state_pool [[buffer(5)]],                    \
                      device const int *cu_seqlens [[buffer(6)]],                \
                      device const int *slot_mapping [[buffer(7)]],              \
                      device T *y [[buffer(8)]],                                 \
                      constant int &num_requests [[buffer(9)]],                  \
                      constant int &Hk [[buffer(10)]],                           \
                      constant int &Hv [[buffer(11)]],                           \
                      constant int &Dv [[buffer(12)]],                           \
                      constant int &load_initial [[buffer(13)]],                 \
                      uint3 gid [[threadgroup_position_in_grid]],                \
                      uint lane [[thread_index_in_simdgroup]]);

instantiate_gdn_recur(float32, float, 64)
instantiate_gdn_recur(float32, float, 128)
instantiate_gdn_recur(float16, half, 64)
instantiate_gdn_recur(float16, half, 128)
instantiate_gdn_recur(bfloat16, bf16, 64)
instantiate_gdn_recur(bfloat16, bf16, 128)

}
