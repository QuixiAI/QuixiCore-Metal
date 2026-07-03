#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Token embedding + multimodal span merge (GPU-resident, no host preprocessing bubble).
//
//   embedding_lookup:  out[t] = scale * table[token_ids[t]]  (+ pos_table[t] if use_pos).
//                      A padded/negative token id emits zeros. Pure gather.
//   merge_multimodal_spans:  out[t] = src[t] >= 0 ? modal[src[t]] : text[t].  The per-row src map
//                      encodes which text positions are replaced by image/audio/video embeddings
//                      (built host-side from span offsets/lengths); the kernel is a per-row select.
// Both are flat one-thread-per-element (t*D + d); T templated (fp16/bf16/fp32).
// Ref: FasterTransformer gpt_kernels invokeInputIdsEmbeddingLookupPosEncoding.
// ---------------------------------------------------------------------------

template <typename T>
kernel void embedding_lookup(device const int *token_ids [[buffer(0)]],  // (num_tok,)
                             device const T   *table     [[buffer(1)]],  // (vocab, D)
                             device const T   *pos_table [[buffer(2)]],  // (num_tok, D) or dummy
                             device T         *out       [[buffer(3)]],  // (num_tok, D)
                             constant int   &D           [[buffer(4)]],
                             constant int   &vocab       [[buffer(5)]],
                             constant int   &n_tok       [[buffer(6)]],
                             constant float &scale       [[buffer(7)]],
                             constant int   &use_pos     [[buffer(8)]],
                             uint gid [[thread_position_in_grid]]) {
    const long total = (long)n_tok * D;
    if ((long)gid >= total) return;
    const int t = (int)((long)gid / D);
    const int d = (int)((long)gid % D);
    const int tok = token_ids[t];
    float v = 0.0f;
    if (tok >= 0 && tok < vocab) v = float(table[(long)tok * D + d]) * scale;
    if (use_pos) v += float(pos_table[(long)t * D + d]);
    out[gid] = T(v);
}

// out[t] = (src[t] >= 0) ? modal[src[t]] : text[t]. src (num_tok,) int: -1 keeps the text embedding,
// >=0 gathers row src[t] of the modal embeddings. One thread per (t, d) element.
template <typename T>
kernel void merge_multimodal_spans(device const T   *text  [[buffer(0)]],   // (num_tok, D)
                                   device const T   *modal [[buffer(1)]],   // (num_modal, D)
                                   device const int *src   [[buffer(2)]],   // (num_tok,)
                                   device T         *out   [[buffer(3)]],   // (num_tok, D)
                                   constant int &D         [[buffer(4)]],
                                   constant int &n_tok     [[buffer(5)]],
                                   constant int &n_modal   [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    const long total = (long)n_tok * D;
    if ((long)gid >= total) return;
    const int t = (int)((long)gid / D);
    const int d = (int)((long)gid % D);
    const int sm = src[t];
    out[gid] = (sm >= 0 && sm < n_modal) ? modal[(long)sm * D + d] : text[gid];
}

#define instantiate_embedding(type_name, T)                                        \
  template [[host_name("embedding_lookup_" #type_name)]] [[kernel]] void            \
  embedding_lookup<T>(device const int *token_ids [[buffer(0)]],                    \
    device const T *table [[buffer(1)]], device const T *pos_table [[buffer(2)]],   \
    device T *out [[buffer(3)]], constant int &D [[buffer(4)]],                     \
    constant int &vocab [[buffer(5)]], constant int &n_tok [[buffer(6)]],           \
    constant float &scale [[buffer(7)]], constant int &use_pos [[buffer(8)]],       \
    uint gid [[thread_position_in_grid]]);                                          \
  template [[host_name("merge_multimodal_spans_" #type_name)]] [[kernel]] void      \
  merge_multimodal_spans<T>(device const T *text [[buffer(0)]],                     \
    device const T *modal [[buffer(1)]], device const int *src [[buffer(2)]],       \
    device T *out [[buffer(3)]], constant int &D [[buffer(4)]],                     \
    constant int &n_tok [[buffer(5)]], constant int &n_modal [[buffer(6)]],         \
    uint gid [[thread_position_in_grid]]);

instantiate_embedding(float32, float)
instantiate_embedding(float16, half)
instantiate_embedding(bfloat16, bf16)
