// Shared, framework-agnostic launch logic for the ThunderMittens kernels.
//
// This header is the SINGLE SOURCE OF TRUTH for each kernel's host ABI — the kernel
// name, the buffer index mapping, the scalar parameters, and the grid/threadgroup
// geometry. Both backends drive it through a small "encoder" adapter:
//   - MLX  (<kernel>.cpp via MLXEncoder in tk_mlx_launch.h): binds with set_input_array
//     so MLX's residency/scheduling bookkeeping is preserved.
//   - Torch (tk_torch/torch_kernels.mm via TorchEncoder): binds the MTLBuffer directly.
//
// An adapter `E` must provide:
//   typedefs E::in_t, E::out_t                      (input / output buffer handle types)
//   void pipeline(const std::string& kernel_name)   (set the compute pipeline state)
//   void in(E::in_t, int index)                     (bind an input buffer)
//   void out(E::out_t, int index)                   (bind the output buffer)
//   template<class T> void bytes(const T&, int idx) (set inline scalar bytes)
//   void dispatch(int gx,int gy,int gz, int tx,int ty,int tz)  (dispatch threadgroups)
//
// Pure C++: depends on neither MLX nor Metal, so it compiles in both .cpp and .mm.

#pragma once
#include <cstdint>
#include <string>

namespace tk {

// ----- kernel-name helpers (must match the [[host_name(...)]] in <kernel>.metal) -----
inline std::string layernorm_kernel_name(int D) { return "layernorm_" + std::to_string(D); }
inline std::string attn_fwd_kernel_name(int D) { return "attn_fwd_" + std::to_string(D); }
inline std::string add_rt_kernel_name(const std::string& t) { return "add_rt_" + t; }
inline std::string matmul_custom_kernel_name(const std::string& t) { return "matmul_custom_" + t; }
inline std::string rms_norm_kernel_name(int D) { return "rms_norm_" + std::to_string(D); }
inline std::string rms_norm_add_kernel_name(int D) { return "rms_norm_add_" + std::to_string(D); }
inline std::string layernorm_add_kernel_name(int D) { return "layernorm_add_" + std::to_string(D); }
inline std::string rope_kv_insert_kernel_name(const std::string& t, int D) {
  return "rope_kv_insert_" + t + "_" + std::to_string(D);
}
inline std::string rope_kv_insert_norm_kernel_name(const std::string& t, int D) {
  return "rope_kv_insert_norm_" + t + "_" + std::to_string(D);
}
inline std::string rms_norm_add_fp8_kernel_name(int D) { return "rms_norm_add_fp8_" + std::to_string(D); }
inline std::string rms_norm_add_fp8_dyn_kernel_name(int D) { return "rms_norm_add_fp8_dyn_" + std::to_string(D); }
inline std::string layernorm_add_fp8_kernel_name(int D) { return "layernorm_add_fp8_" + std::to_string(D); }
inline std::string layernorm_add_fp8_dyn_kernel_name(int D) { return "layernorm_add_fp8_dyn_" + std::to_string(D); }
inline std::string argmax_kernel_name(const std::string& t) { return "argmax_" + t; }
inline std::string moe_route_topk_kernel_name(const std::string& t) { return "moe_route_topk_" + t; }
inline std::string moe_finalize_kernel_name(const std::string& t) { return "moe_finalize_" + t; }
inline std::string moe_grouped_gemm_kernel_name(const std::string& t) { return "moe_grouped_gemm_" + t; }
inline std::string moe_grouped_gemm_rect_kernel_name(const std::string& t) { return "moe_grouped_gemm_rect_" + t; }
inline std::string moe_grouped_gemm_swiglu_kernel_name(const std::string& t) { return "moe_grouped_gemm_swiglu_" + t; }
inline std::string sample_categorical_kernel_name(const std::string& t) { return "sample_categorical_" + t; }
inline std::string top_k_sample_kernel_name(const std::string& t) { return "top_k_sample_" + t; }
inline std::string top_p_sample_kernel_name(const std::string& t) { return "top_p_sample_" + t; }
inline std::string apply_penalty_kernel_name(const std::string& t) { return "apply_penalty_" + t; }
inline std::string quant_tensor_absmax_kernel_name(const std::string& t) { return "quant_tensor_absmax_" + t; }
inline std::string quant_tensor_encode_fp8_kernel_name(const std::string& t) { return "quant_tensor_encode_fp8_" + t; }
inline std::string quant_tensor_encode_int8_kernel_name(const std::string& t) { return "quant_tensor_encode_int8_" + t; }
inline std::string quantize_per_token_fp8_kernel_name(const std::string& t) { return "quantize_per_token_fp8_" + t; }
inline std::string quantize_per_token_int8_kernel_name(const std::string& t) { return "quantize_per_token_int8_" + t; }
inline std::string softmax_kernel_name(int D) { return "softmax_" + std::to_string(D); }
inline std::string rotary_kernel_name(int D) { return "rotary_" + std::to_string(D); }
inline std::string rotary_interleaved_kernel_name(int D) { return "rotary_interleaved_" + std::to_string(D); }
inline std::string mla_q_norm_rope_kernel_name(int D) { return "mla_q_norm_rope_" + std::to_string(D); }
inline std::string mla_kv_insert_kernel_name(int L) { return "mla_kv_insert_" + std::to_string(L); }
inline std::string mla_decode_kernel_name(int L, int R) {
  return "mla_decode_" + std::to_string(L) + "_" + std::to_string(R);
}
inline std::string gelu_kernel_name(int D) { return "gelu_" + std::to_string(D); }
inline std::string glu_kernel_name(const std::string& mode, const std::string& t) { return "glu_" + mode + "_" + t; }
inline std::string hadamard_kernel_name(const std::string& t, int D) {
  return "hadamard_" + t + "_" + std::to_string(D);
}
inline std::string kv_cache_kernel_name(const std::string& op, const std::string& t) {
  return "kv_cache_" + op + "_" + t;
}
inline std::string paged_attention_kernel_name(const std::string& t, int D) {
  return "paged_attention_" + t + "_" + std::to_string(D);
}
inline std::string paged_attention_gqa_staged_kernel_name(const std::string& t, int D) {
  return "paged_attention_gqa_staged_" + t + "_" + std::to_string(D);
}
inline std::string paged_attention_xcache_kernel_name(const std::string& t, int D) {
  return "paged_attention_xcache_" + t + "_" + std::to_string(D);
}
inline std::string kv_cache_scatter_fp8_kernel_name(const std::string& t) { return "kv_cache_scatter_fp8_" + t; }
inline std::string paged_attention_fp8_kernel_name(const std::string& t, int D) {
  return "paged_attention_fp8_" + t + "_" + std::to_string(D);
}
inline std::string paged_attention_partition_kernel_name(const std::string& t, int D) {
  return "paged_attention_partition_" + t + "_" + std::to_string(D);
}
inline std::string paged_attention_partition_fp8_kernel_name(const std::string& t, int D) {
  return "paged_attention_partition_fp8_" + t + "_" + std::to_string(D);
}
inline std::string paged_attention_reduce_kernel_name(const std::string& t, int D) {
  return "paged_attention_reduce_" + t + "_" + std::to_string(D);
}
inline std::string attn_causal_kernel_name(int D) { return "attn_causal_" + std::to_string(D); }
inline std::string flux_gelu_kernel_name(const std::string& t) { return "flux_gelu_" + t; }
inline std::string flux_gate_kernel_name(const std::string& t) { return "flux_gate_" + t; }
inline std::string gemm_staged_kernel_name(const std::string& t) { return "gemm_staged_" + t; }
inline std::string attn_multiwarp_kernel_name(int D) { return "attn_multiwarp_" + std::to_string(D); }
inline std::string attn_q_kernel_name(const std::string& fmt, int D, bool causal) {
  return std::string("attn_q_") + (causal ? "causal_" : "") + fmt + "_" + std::to_string(D);
}
inline std::string linear_attn_kernel_name(int D) { return "linear_attn_" + std::to_string(D); }
inline std::string hedgehog_kernel_name(int D) { return "hedgehog_" + std::to_string(D); }
inline std::string lin_attn_causal_kernel_name(int D) { return "lin_attn_causal_" + std::to_string(D); }
inline std::string mamba2_kernel_name(int D) { return "mamba2_" + std::to_string(D); }
inline std::string mamba2_bwd_i_kernel_name(int D) { return "mamba2_bwd_i_" + std::to_string(D); }
inline std::string mamba2_bwd_j_kernel_name(int D) { return "mamba2_bwd_j_" + std::to_string(D); }
inline std::string lin_attn_decay_kernel_name(int D) { return "lin_attn_decay_" + std::to_string(D); }
inline std::string based_kernel_name(int DQK, int DVO) {
  return "based_" + std::to_string(DQK) + "_" + std::to_string(DVO);
}
inline std::string cmplx_matmul_kernel_name(const std::string& t) { return "cmplx_matmul_" + t; }
inline std::string fftconv_kernel_name(int S) { return "fftconv_" + std::to_string(S); }
inline std::string qgemm_kernel_name(const std::string& fmt) { return "qgemm_" + fmt; }
inline std::string qgemv_kernel_name(const std::string& fmt) { return "qgemv_" + fmt; }
inline std::string qflux_gelu_kernel_name(const std::string& fmt) { return "qflux_gelu_" + fmt; }
inline std::string qgemm_frag_kernel_name(const std::string& fmt) { return "qgemm_frag_" + fmt; }
inline std::string qgemm_actorder_kernel_name(const std::string& fmt) { return "qgemm_actorder_" + fmt; }

// ----- LayerNorm: x@0 w@1 b@2 -> o@3 ; M@4(u32) eps@5(f32) ; grid (M,1,1) group (32,1,1) -----
template <class E>
void launch_layernorm(E& e, typename E::in_t x, typename E::in_t w, typename E::in_t b,
                      typename E::out_t o, uint32_t M, int D, float eps) {
  e.pipeline(layernorm_kernel_name(D));
  e.in(x, 0); e.in(w, 1); e.in(b, 2); e.out(o, 3);
  e.bytes(M, 4); e.bytes(eps, 5);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

// ----- add_rt: x@0 y@1 -> out@2 ; rows@3(i32) cols@4(i32) ; grid (cols/8, rows/8, 1) -----
template <class E>
void launch_add_rt(E& e, typename E::in_t x, typename E::in_t y, typename E::out_t o,
                   int rows, int cols, const std::string& type_name) {
  e.pipeline(add_rt_kernel_name(type_name));
  e.in(x, 0); e.in(y, 1); e.out(o, 2);
  e.bytes(rows, 3); e.bytes(cols, 4);
  e.dispatch(cols / 8, rows / 8, 1, 32, 1, 1);
}

// ----- matmul_custom: D(out)@0 A@1 B@2 ; N@3 K@4 M@5 (i32) ; grid (M/32, N/32, 1) -----
// A is (N,K), B is (K,M), out is (N,M).
template <class E>
void launch_matmul_custom(E& e, typename E::out_t o, typename E::in_t a, typename E::in_t b,
                          int N, int K, int M, const std::string& type_name) {
  e.pipeline(matmul_custom_kernel_name(type_name));
  e.out(o, 0); e.in(a, 1); e.in(b, 2);
  e.bytes(N, 3); e.bytes(K, 4); e.bytes(M, 5);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- attn_fwd: q@0 k@1 v@2 -> o@3 ; N@4(u32) H@5(u32) ; grid (N/8, H, B) group (32,1,1) -----
template <class E>
void launch_attn_fwd(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                     typename E::out_t o, unsigned N, unsigned H, int B, int D) {
  e.pipeline(attn_fwd_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- attn_q (quantized-KV attention): q@0(bf16) Kq@1(uchar) Vq@2(uchar) o@3(bf16) ; N@4 H@5 ;
//        grid (N/8, H, B), 32 threads. Same online-softmax flow as attn_fwd, K/V dequantized. -----
template <class E>
void launch_attn_q(E& e, typename E::in_t q, typename E::in_t kq, typename E::in_t vq,
                   typename E::out_t o, unsigned N, unsigned H, int B, int D,
                   const std::string& fmt, bool causal, bool multiwarp) {
  const int NW = 4;  // attn_q_mw warps
  e.pipeline(multiwarp ? ("attn_q_mw_" + fmt + "_" + std::to_string(D))
                       : attn_q_kernel_name(fmt, D, causal));
  e.in(q, 0); e.in(kq, 1); e.in(vq, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  if (multiwarp)
    e.dispatch(static_cast<int>(N) / (8 * NW), static_cast<int>(H), B, 32 * NW, 1, 1);
  else
    e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- rms_norm: x@0 w@1 -> o@2 ; M@3(u32) eps@4(f32) ; grid (M,1,1) group (32,1,1) -----
template <class E>
void launch_rms_norm(E& e, typename E::in_t x, typename E::in_t w,
                     typename E::out_t o, uint32_t M, int D, float eps) {
  e.pipeline(rms_norm_kernel_name(D));
  e.in(x, 0); e.in(w, 1); e.out(o, 2);
  e.bytes(M, 3); e.bytes(eps, 4);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

// ----- rms_norm_add: x@0 residual@1 w@2 -> o@3 res_out@4 ; M@5(u32) eps@6(f32) ;
//        grid (M,1,1) group (32,1,1). o = rms_norm(x+residual)*w ; res_out = x+residual -----
template <class E>
void launch_rms_norm_add(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                         typename E::out_t o, typename E::out_t res_out,
                         uint32_t M, int D, float eps) {
  e.pipeline(rms_norm_add_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.out(o, 3); e.out(res_out, 4);
  e.bytes(M, 5); e.bytes(eps, 6);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

// ----- layernorm_add: x@0 residual@1 w@2 b@3 -> o@4 res_out@5 ; M@6(u32) eps@7(f32) ;
//        grid (M,1,1) group (32,1,1). o = layernorm(x+residual)*w+b ; res_out = x+residual -----
template <class E>
void launch_layernorm_add(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                          typename E::in_t b, typename E::out_t o, typename E::out_t res_out,
                          uint32_t M, int D, float eps) {
  e.pipeline(layernorm_add_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.in(b, 3); e.out(o, 4); e.out(res_out, 5);
  e.bytes(M, 6); e.bytes(eps, 7);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

// ----- rope_kv_insert: k@0 v@1 cos@2 sin@3 positions@4(i32) slot_mapping@5(i64) ->
//        key_cache@6 value_cache@7 ; num_kv_heads@8(i32) block_size@9(i32) ; grid (M,1,1).
//        M = num_tokens*num_kv_heads. caches must be pre-cloned (insert overwrites slot rows). -----
template <class E>
void launch_rope_kv_insert(E& e, typename E::in_t k, typename E::in_t v,
                           typename E::in_t cos, typename E::in_t sin,
                           typename E::in_t positions, typename E::in_t slot_mapping,
                           typename E::out_t key_cache, typename E::out_t value_cache,
                           int M, int num_kv_heads, int block_size, int D,
                           const std::string& type_name) {
  e.pipeline(rope_kv_insert_kernel_name(type_name, D));
  e.in(k, 0); e.in(v, 1); e.in(cos, 2); e.in(sin, 3);
  e.in(positions, 4); e.in(slot_mapping, 5);
  e.out(key_cache, 6); e.out(value_cache, 7);
  e.bytes(num_kv_heads, 8); e.bytes(block_size, 9);
  e.dispatch(M, 1, 1, 32, 1, 1);
}

// ----- rope_kv_insert_norm: adds K RMSNorm (weight@8, eps@11, gemma@12) before RoPE+insert. -----
template <class E>
void launch_rope_kv_insert_norm(E& e, typename E::in_t k, typename E::in_t v,
                                typename E::in_t cos, typename E::in_t sin,
                                typename E::in_t positions, typename E::in_t slot_mapping,
                                typename E::out_t key_cache, typename E::out_t value_cache,
                                typename E::in_t norm_weight, int M, int num_kv_heads,
                                int block_size, int D, float eps, int gemma,
                                const std::string& type_name) {
  e.pipeline(rope_kv_insert_norm_kernel_name(type_name, D));
  e.in(k, 0); e.in(v, 1); e.in(cos, 2); e.in(sin, 3);
  e.in(positions, 4); e.in(slot_mapping, 5);
  e.out(key_cache, 6); e.out(value_cache, 7); e.in(norm_weight, 8);
  e.bytes(num_kv_heads, 9); e.bytes(block_size, 10); e.bytes(eps, 11); e.bytes(gemma, 12);
  e.dispatch(M, 1, 1, 32, 1, 1);
}

// ----- moe_route_topk: logits@0(T,E) -> topk_ids@1(i32) topk_weights@2(f32), both (T,K) ;
//        E@3(i32) K@4(i32) ; grid (num_tokens,1,1), 32 thr. Top-k experts + renormalized softmax. -----
template <class Enc>
void launch_moe_route_topk(Enc& e, typename Enc::in_t logits, typename Enc::out_t topk_ids,
                           typename Enc::out_t topk_weights, int num_tokens, int num_experts,
                           int k, const std::string& type_name) {
  e.pipeline(moe_route_topk_kernel_name(type_name));
  e.in(logits, 0); e.out(topk_ids, 1); e.out(topk_weights, 2);
  e.bytes(num_experts, 3); e.bytes(k, 4);
  e.dispatch(num_tokens, 1, 1, 32, 1, 1);
}

// ----- MoE permute pipeline (all int32). Run in order in one (serial) encoder. -----
template <class Enc>
void launch_moe_zero_i32(Enc& e, typename Enc::out_t p, int n) {
  e.pipeline("moe_zero_i32");
  e.out(p, 0); e.bytes(n, 1);
  e.dispatch((n + 255) / 256, 1, 1, 256, 1, 1);
}
template <class Enc>
void launch_moe_histogram(Enc& e, typename Enc::in_t topk_ids, typename Enc::out_t counts, int TK) {
  e.pipeline("moe_histogram");
  e.in(topk_ids, 0); e.out(counts, 1); e.bytes(TK, 2);
  e.dispatch((TK + 255) / 256, 1, 1, 256, 1, 1);
}
template <class Enc>
void launch_moe_scan_offsets(Enc& e, typename Enc::in_t counts, typename Enc::out_t offsets,
                             typename Enc::out_t cursor, int num_experts) {
  e.pipeline("moe_scan_offsets");
  e.in(counts, 0); e.out(offsets, 1); e.out(cursor, 2); e.bytes(num_experts, 3);
  e.dispatch(1, 1, 1, 256, 1, 1);   // one threadgroup; parallel P2 scan
}
template <class Enc>
void launch_moe_scatter(Enc& e, typename Enc::in_t topk_ids, typename Enc::out_t cursor,
                        typename Enc::out_t sorted_row_idx, typename Enc::out_t inv_idx, int TK) {
  e.pipeline("moe_scatter");
  e.in(topk_ids, 0); e.out(cursor, 1); e.out(sorted_row_idx, 2); e.out(inv_idx, 3); e.bytes(TK, 4);
  e.dispatch((TK + 255) / 256, 1, 1, 256, 1, 1);
}

// ----- moe_grouped_gemm: out@0 A@1(permuted_input) W@2(E,H,H) expert_of_tile@3(i32) ;
//        total_rows@4 H@5 ; grid (H/32, total_rows/32, 1), 32 thr. out = A @ W[expert]. -----
template <class Enc>
void launch_moe_grouped_gemm(Enc& e, typename Enc::out_t out, typename Enc::in_t A,
                             typename Enc::in_t W, typename Enc::in_t expert_of_tile,
                             int total_rows, int H, const std::string& type_name) {
  e.pipeline(moe_grouped_gemm_kernel_name(type_name));
  e.out(out, 0); e.in(A, 1); e.in(W, 2); e.in(expert_of_tile, 3);
  e.bytes(total_rows, 4); e.bytes(H, 5);
  e.dispatch(H / 32, total_rows / 32, 1, 32, 1, 1);
}

// Rectangular grouped GEMM: out(total_rows,N_out)=A(total_rows,K_dim)@W[e](K_dim,N_out). grid (N_out/32, rows/32).
template <class Enc>
void launch_moe_grouped_gemm_rect(Enc& e, typename Enc::out_t out, typename Enc::in_t A,
                                  typename Enc::in_t W, typename Enc::in_t expert_of_tile,
                                  int total_rows, int K_dim, int N_out, const std::string& type_name) {
  e.pipeline(moe_grouped_gemm_rect_kernel_name(type_name));
  e.out(out, 0); e.in(A, 1); e.in(W, 2); e.in(expert_of_tile, 3);
  e.bytes(total_rows, 4); e.bytes(K_dim, 5); e.bytes(N_out, 6);
  e.dispatch(N_out / 32, total_rows / 32, 1, 32, 1, 1);
}

// Fused SiLU-GLU GEMM1: out(total_rows,inter)=silu(A@W1_gate)*(A@W1_up), W1[e] (H,2*inter). grid (inter/32, rows/32).
template <class Enc>
void launch_moe_grouped_gemm_swiglu(Enc& e, typename Enc::out_t out, typename Enc::in_t A,
                                    typename Enc::in_t W1, typename Enc::in_t expert_of_tile,
                                    int total_rows, int H, int inter, const std::string& type_name) {
  e.pipeline(moe_grouped_gemm_swiglu_kernel_name(type_name));
  e.out(out, 0); e.in(A, 1); e.in(W1, 2); e.in(expert_of_tile, 3);
  e.bytes(total_rows, 4); e.bytes(H, 5); e.bytes(inter, 6);
  e.dispatch(inter / 32, total_rows / 32, 1, 32, 1, 1);
}

// ----- moe_finalize: expert_out@0 inv_idx@1(i32) topk_weights@2(f32) -> out@3 ; K@4 Hdim@5 ;
//        grid (num_tokens,1,1), 32 thr. k-way weighted reduce via inv_idx (no atomics). -----
template <class Enc>
void launch_moe_finalize(Enc& e, typename Enc::in_t expert_out, typename Enc::in_t inv_idx,
                         typename Enc::in_t topk_weights, typename Enc::out_t out,
                         int num_tokens, int k, int Hdim, const std::string& type_name) {
  e.pipeline(moe_finalize_kernel_name(type_name));
  e.in(expert_out, 0); e.in(inv_idx, 1); e.in(topk_weights, 2); e.out(out, 3);
  e.bytes(k, 4); e.bytes(Hdim, 5);
  e.dispatch(num_tokens, 1, 1, 32, 1, 1);
}

// ----- argmax (greedy sampling): logits@0 -> out_idx@1(i32) ; V@2(i32) ; grid (rows,1,1), 32 thr.
//        One simdgroup per row finds the argmax token over the vocab dim V. -----
template <class E>
void launch_argmax(E& e, typename E::in_t logits, typename E::out_t out_idx,
                   int rows, int V, const std::string& type_name) {
  e.pipeline(argmax_kernel_name(type_name));
  e.in(logits, 0); e.out(out_idx, 1);
  e.bytes(V, 2);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- sample_categorical: logits@0 -> out_idx@1(i32) ; V@2(i32) seed@3(u32) invtemp@4(f32) ;
//        grid (rows,1,1), 32 thr. Gumbel-max sampling from softmax(logits/temperature). -----
template <class E>
void launch_sample_categorical(E& e, typename E::in_t logits, typename E::out_t out_idx,
                               int rows, int V, uint32_t seed, float invtemp,
                               const std::string& type_name) {
  e.pipeline(sample_categorical_kernel_name(type_name));
  e.in(logits, 0); e.out(out_idx, 1);
  e.bytes(V, 2); e.bytes(seed, 3); e.bytes(invtemp, 4);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- top_k_sample: logits@0 -> out_idx@1(i32) ; V@2 K@3(i32) seed@4(u32) invtemp@5(f32) ;
//        grid (rows,1,1), 32 thr. Gumbel-max sampling restricted to the top-k logits. -----
template <class E>
void launch_top_k_sample(E& e, typename E::in_t logits, typename E::out_t out_idx,
                         int rows, int V, int k, uint32_t seed, float invtemp,
                         const std::string& type_name) {
  e.pipeline(top_k_sample_kernel_name(type_name));
  e.in(logits, 0); e.out(out_idx, 1);
  e.bytes(V, 2); e.bytes(k, 3); e.bytes(seed, 4); e.bytes(invtemp, 5);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- top_p_sample: logits@0 -> out_idx@1(i32) ; V@2(i32) p@3(f32) seed@4(u32) invtemp@5(f32) ;
//        grid (rows,1,1), 32 thr. Gumbel-max sampling from the nucleus (cumulative prob >= p). -----
template <class E>
void launch_top_p_sample(E& e, typename E::in_t logits, typename E::out_t out_idx,
                         int rows, int V, float p, uint32_t seed, float invtemp,
                         const std::string& type_name) {
  e.pipeline(top_p_sample_kernel_name(type_name));
  e.in(logits, 0); e.out(out_idx, 1);
  e.bytes(V, 2); e.bytes(p, 3); e.bytes(seed, 4); e.bytes(invtemp, 5);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- penalty_histogram: prev_tokens@0(i32) -> counts@1(atomic i32) ; V@2 L@3 TL@4 ; grid (TL).
//        counts[(row,tok)] += 1 for each valid history token. Zero counts first. -----
template <class E>
void launch_penalty_histogram(E& e, typename E::in_t prev_tokens, typename E::out_t counts,
                              int V, int L, int TL) {
  e.pipeline("penalty_histogram");
  e.in(prev_tokens, 0); e.out(counts, 1);
  e.bytes(V, 2); e.bytes(L, 3); e.bytes(TL, 4);
  e.dispatch((TL + 255) / 256, 1, 1, 256, 1, 1);
}

// ----- apply_penalty: logits@0 counts@1(i32) -> out@2 ; V@3 invtemp@4 rep@5 presence@6 freq@7 ;
//        grid (rows,1,1), 32 thr. temperature + repetition/presence/frequency penalties. -----
template <class E>
void launch_apply_penalty(E& e, typename E::in_t logits, typename E::in_t counts,
                          typename E::out_t out, typename E::in_t bias, int rows, int V,
                          float invtemp, float rep, float presence, float freq,
                          int eos_id, int min_length, int gen_len, const std::string& type_name) {
  e.pipeline(apply_penalty_kernel_name(type_name));
  e.in(logits, 0); e.in(counts, 1); e.out(out, 2);
  e.bytes(V, 3); e.bytes(invtemp, 4); e.bytes(rep, 5); e.bytes(presence, 6); e.bytes(freq, 7);
  e.in(bias, 8); e.bytes(eos_id, 9); e.bytes(min_length, 10); e.bytes(gen_len, 11);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- per-tensor dynamic quant (2 passes): absmax@1(atomic_uint) reduce, then encode. -----
template <class E>
void launch_quant_tensor_absmax(E& e, typename E::in_t x, typename E::out_t scale_u, int n,
                                const std::string& type_name) {
  e.pipeline(quant_tensor_absmax_kernel_name(type_name));
  e.in(x, 0); e.out(scale_u, 1); e.bytes(n, 2);
  e.dispatch((n + 255) / 256, 1, 1, 256, 1, 1);
}
template <class E>
void launch_quant_tensor_encode(E& e, typename E::in_t x, typename E::in_t scale_u,
                                typename E::out_t codes, typename E::out_t scale_out, int n,
                                bool is_int8, const std::string& type_name) {
  e.pipeline(is_int8 ? quant_tensor_encode_int8_kernel_name(type_name)
                     : quant_tensor_encode_fp8_kernel_name(type_name));
  e.in(x, 0); e.in(scale_u, 1); e.out(codes, 2); e.out(scale_out, 3); e.bytes(n, 4);
  e.dispatch((n + 255) / 256, 1, 1, 256, 1, 1);
}

// ----- quantize_per_token_fp8: x@0 -> codes@1(uint8) scale@2(f32) ; D@3(i32) ; grid (rows,1,1).
//        Per-row absmax -> scale=absmax/448 ; codes = e4m3(x/scale). -----
template <class E>
void launch_quantize_per_token_fp8(E& e, typename E::in_t x, typename E::out_t codes,
                                   typename E::out_t scale, int rows, int D,
                                   const std::string& type_name) {
  e.pipeline(quantize_per_token_fp8_kernel_name(type_name));
  e.in(x, 0); e.out(codes, 1); e.out(scale, 2);
  e.bytes(D, 3);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- quantize_per_token_int8: x@0 -> codes@1(int8) scale@2(f32) ; D@3(i32) ; grid (rows,1,1).
//        Per-row absmax -> scale=absmax/127 ; codes = round_clamp(x/scale). -----
template <class E>
void launch_quantize_per_token_int8(E& e, typename E::in_t x, typename E::out_t codes,
                                    typename E::out_t scale, int rows, int D,
                                    const std::string& type_name) {
  e.pipeline(quantize_per_token_int8_kernel_name(type_name));
  e.in(x, 0); e.out(codes, 1); e.out(scale, 2);
  e.bytes(D, 3);
  e.dispatch(rows, 1, 1, 32, 1, 1);
}

// ----- fp8 norm epilogues: codes=e4m3(norm(x+residual)*w[+b]/scale). res_out=x+residual (bf16).
//        static: inv_scale param; dyn: per-row absmax/448 -> scale output. grid (M,1,1). -----
template <class E>
void launch_rms_norm_add_fp8(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                             typename E::out_t codes, typename E::out_t res_out,
                             uint32_t M, int D, float eps, float inv_scale) {
  e.pipeline(rms_norm_add_fp8_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.out(codes, 3); e.out(res_out, 4);
  e.bytes(M, 5); e.bytes(eps, 6); e.bytes(inv_scale, 7);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
template <class E>
void launch_rms_norm_add_fp8_dyn(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                                 typename E::out_t codes, typename E::out_t res_out,
                                 typename E::out_t scale, uint32_t M, int D, float eps) {
  e.pipeline(rms_norm_add_fp8_dyn_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.out(codes, 3); e.out(res_out, 4); e.out(scale, 5);
  e.bytes(M, 6); e.bytes(eps, 7);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
template <class E>
void launch_layernorm_add_fp8(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                              typename E::in_t b, typename E::out_t codes, typename E::out_t res_out,
                              uint32_t M, int D, float eps, float inv_scale) {
  e.pipeline(layernorm_add_fp8_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.in(b, 3); e.out(codes, 4); e.out(res_out, 5);
  e.bytes(M, 6); e.bytes(eps, 7); e.bytes(inv_scale, 8);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}
template <class E>
void launch_layernorm_add_fp8_dyn(E& e, typename E::in_t x, typename E::in_t r, typename E::in_t w,
                                  typename E::in_t b, typename E::out_t codes, typename E::out_t res_out,
                                  typename E::out_t scale, uint32_t M, int D, float eps) {
  e.pipeline(layernorm_add_fp8_dyn_kernel_name(D));
  e.in(x, 0); e.in(r, 1); e.in(w, 2); e.in(b, 3); e.out(codes, 4); e.out(res_out, 5); e.out(scale, 6);
  e.bytes(M, 7); e.bytes(eps, 8);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

// ----- softmax (last axis): x@0 -> o@1 ; M@2(u32) ; grid (M,1,1) group (32,1,1) -----
template <class E>
void launch_softmax(E& e, typename E::in_t x, typename E::out_t o, uint32_t M, int D) {
  e.pipeline(softmax_kernel_name(D));
  e.in(x, 0); e.out(o, 1);
  e.bytes(M, 2);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

// ----- rotary (split-half RoPE): x@0 cos@1 sin@2 -> o@3 ; N@4(u32) ;
//        grid (M,1,1) group (32,1,1). x is (M=B*H*N, D) flattened; cos/sin are
//        (N, D/2); each row uses seq position n = row % N. -----
template <class E>
void launch_rotary(E& e, typename E::in_t x, typename E::in_t cos, typename E::in_t sin,
                   typename E::out_t o, uint32_t M, unsigned N, int D, bool interleaved = false) {
  e.pipeline(interleaved ? rotary_interleaved_kernel_name(D) : rotary_kernel_name(D));
  e.in(x, 0); e.in(cos, 1); e.in(sin, 2); e.out(o, 3);
  e.bytes(N, 4);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

// ----- MLA Q-path: q@0 cos@1 sin@2 positions@3 -> out@4 ; num_heads@5 nope@6 rope@7 norm_mode@8
//        eps@9 ; norm_weight@10 (read iff mode 2) ; grid (M=tokens*heads,1,1) group (32,1,1). -----
template <class E>
void launch_mla_q_norm_rope(E& e, typename E::in_t q, typename E::in_t cos, typename E::in_t sin,
                            typename E::in_t positions, typename E::in_t norm_weight,
                            typename E::out_t out, int M, int num_heads, int nope_dim, int rope_dim,
                            int norm_mode, float eps, int head_dim) {
  e.pipeline(mla_q_norm_rope_kernel_name(head_dim));
  e.in(q, 0); e.in(cos, 1); e.in(sin, 2); e.in(positions, 3); e.out(out, 4);
  e.bytes(num_heads, 5); e.bytes(nope_dim, 6); e.bytes(rope_dim, 7); e.bytes(norm_mode, 8);
  e.bytes(eps, 9); e.in(norm_weight, 10);
  e.dispatch(M, 1, 1, 32, 1, 1);
}

// ----- MLA classic KV insert: kv_c@0 k_pe@1 cos@2 sin@3 positions@4 slot_mapping@5 -> kv_cache@6 ;
//        block_size@7 rope_dim@8 norm_mode@9 eps@10 ; norm_weight@11 ; grid (T,1,1) group (32,1,1). -----
template <class E>
void launch_mla_cache_clone(E& e, typename E::in_t src, typename E::out_t dst, uint64_t n) {
  e.pipeline("mla_cache_clone");
  e.in(src, 0); e.out(dst, 1); e.bytes(n, 2);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

template <class E>
void launch_mla_cache_clone_u8(E& e, typename E::in_t src, typename E::out_t dst, uint64_t n) {
  e.pipeline("mla_cache_clone_u8");
  e.in(src, 0); e.out(dst, 1); e.bytes(n, 2);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

// ----- MLA V4 packed fp8 insert: kv@0(512) cos@1 sin@2 positions@3 slot@4 -> data_cache@5(u8,576)
//        scale_cache@6(u8,8) ; block_size@7 ; grid (T,1,1) group (32,1,1). -----
template <class E>
void launch_mla_kv_insert_fp8(E& e, typename E::in_t kv, typename E::in_t cos, typename E::in_t sin,
                              typename E::in_t positions, typename E::in_t slot_mapping,
                              typename E::out_t data_cache, typename E::out_t scale_cache,
                              int num_tokens, int block_size) {
  e.pipeline("mla_kv_insert_fp8");
  e.in(kv, 0); e.in(cos, 1); e.in(sin, 2); e.in(positions, 3); e.in(slot_mapping, 4);
  e.out(data_cache, 5); e.out(scale_cache, 6); e.bytes(block_size, 7);
  e.dispatch(num_tokens, 1, 1, 32, 1, 1);
}

template <class E>
void launch_mla_kv_insert(E& e, typename E::in_t kv_c, typename E::in_t k_pe, typename E::in_t cos,
                          typename E::in_t sin, typename E::in_t positions,
                          typename E::in_t slot_mapping, typename E::out_t kv_cache,
                          typename E::in_t norm_weight, int num_tokens, int block_size,
                          int rope_dim, int norm_mode, float eps, int latent) {
  e.pipeline(mla_kv_insert_kernel_name(latent));
  e.in(kv_c, 0); e.in(k_pe, 1); e.in(cos, 2); e.in(sin, 3); e.in(positions, 4);
  e.in(slot_mapping, 5); e.out(kv_cache, 6);
  e.bytes(block_size, 7); e.bytes(rope_dim, 8); e.bytes(norm_mode, 9); e.bytes(eps, 10);
  e.in(norm_weight, 11);
  e.dispatch(num_tokens, 1, 1, 32, 1, 1);
}

// ----- MLA latent decode (MQA): q@0(B,N,QK) kv_cache@1(nb,bs,QK) block_table@2 context_lens@3 ->
//        out@4(B,N,LATENT) ; block_size@5 stride@6 scale@7 num_heads@8 ; grid (num_heads,B) 32 thr. -----
template <class E>
void launch_mla_decode(E& e, typename E::in_t q, typename E::in_t kv_cache,
                       typename E::in_t block_table, typename E::in_t context_lens,
                       typename E::out_t out, int batch, int num_heads, int block_size,
                       int block_table_stride, float scale, int latent, int rope) {
  e.pipeline(mla_decode_kernel_name(latent, rope));
  e.in(q, 0); e.in(kv_cache, 1); e.in(block_table, 2); e.in(context_lens, 3); e.out(out, 4);
  e.bytes(block_size, 5); e.bytes(block_table_stride, 6); e.bytes(scale, 7); e.bytes(num_heads, 8);
  e.dispatch(num_heads, batch, 1, 32, 1, 1);
}

// ----- gelu (elementwise, last axis): x@0 -> o@1 ; M@2(u32) ; grid (M,1,1) group (32,1,1) -----
template <class E>
void launch_gelu(E& e, typename E::in_t x, typename E::out_t o, uint32_t M, int D) {
  e.pipeline(gelu_kernel_name(D));
  e.in(x, 0); e.out(o, 1);
  e.bytes(M, 2);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

// ----- glu family: x@0 gate@1 -> out@2 ; n@3(uint32) alpha@4 limit@5 ; flat elementwise. -----
// Modes mirror llama.cpp's ReGLU/GEGLU/SwiGLU kernels. alpha/limit are only used by swiglu_oai.
template <class E>
void launch_glu(E& e, typename E::in_t x, typename E::in_t gate, typename E::out_t o,
                uint32_t n, const std::string& mode, const std::string& type_name,
                float alpha, float limit) {
  e.pipeline(glu_kernel_name(mode, type_name));
  e.in(x, 0); e.in(gate, 1); e.out(o, 2);
  e.bytes(n, 3); e.bytes(alpha, 4); e.bytes(limit, 5);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

// ----- Hadamard/FWHT over the final axis: x@0 -> out@1 ; scale@2. D in {64,128,256,512}. -----
template <class E>
void launch_hadamard(
    E& e,
    typename E::in_t x,
    typename E::out_t out,
    int rows,
    int D,
    float scale,
    const std::string& type_name) {
  e.pipeline(hadamard_kernel_name(type_name, D));
  e.in(x, 0);
  e.out(out, 1);
  e.bytes(scale, 2);
  e.dispatch(rows, 1, 1, D, 1, 1);
}

// ----- KV cache zero: key_cache@0 value_cache@1 ; n@2(ulong). Flat memset for fresh caches. -----
template <class E>
void launch_kv_cache_zero(
    E& e,
    typename E::out_t key_cache,
    typename E::out_t value_cache,
    uint64_t n,
    const std::string& type_name) {
  e.pipeline(kv_cache_kernel_name("zero", type_name));
  e.out(key_cache, 0);
  e.out(value_cache, 1);
  e.bytes(n, 2);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

// ----- KV cache scatter: key@0 value@1 slot_mapping@2 -> key_cache@3 value_cache@4.
// key/value are (T,H,D); caches are (num_blocks, block_size, H, D). -----
template <class E>
void launch_kv_cache_scatter(
    E& e,
    typename E::in_t key,
    typename E::in_t value,
    typename E::in_t slot_mapping,
    typename E::out_t key_cache,
    typename E::out_t value_cache,
    int num_tokens,
    int num_heads,
    int head_size,
    int block_size,
    const std::string& type_name) {
  e.pipeline(kv_cache_kernel_name("scatter", type_name));
  e.in(key, 0);
  e.in(value, 1);
  e.in(slot_mapping, 2);
  e.out(key_cache, 3);
  e.out(value_cache, 4);
  e.bytes(num_heads, 5);
  e.bytes(head_size, 6);
  e.bytes(block_size, 7);
  e.dispatch(num_tokens, 1, 1, 256, 1, 1);
}

// ----- KV cache gather: key_cache@0 value_cache@1 -> key_out@2 value_out@3.
// block_table@4 cu_seq_lens@5; outputs are (num_tokens,H,D). -----
template <class E>
void launch_kv_cache_gather(
    E& e,
    typename E::in_t key_cache,
    typename E::in_t value_cache,
    typename E::out_t key_out,
    typename E::out_t value_out,
    typename E::in_t block_table,
    typename E::in_t cu_seq_lens,
    int num_tokens,
    int num_seqs,
    int block_size,
    int block_table_stride,
    int num_heads,
    int head_size,
    const std::string& type_name) {
  e.pipeline(kv_cache_kernel_name("gather", type_name));
  e.in(key_cache, 0);
  e.in(value_cache, 1);
  e.out(key_out, 2);
  e.out(value_out, 3);
  e.in(block_table, 4);
  e.in(cu_seq_lens, 5);
  e.bytes(num_tokens, 6);
  e.bytes(num_seqs, 7);
  e.bytes(block_size, 8);
  e.bytes(block_table_stride, 9);
  e.bytes(num_heads, 10);
  e.bytes(head_size, 11);
  e.dispatch(num_tokens, 1, 1, 256, 1, 1);
}

// ----- KV cache clone: key_cache@0 value_cache@1 -> key_out@2 value_out@3 ; n@4. -----
template <class E>
void launch_kv_cache_clone(
    E& e,
    typename E::in_t key_cache,
    typename E::in_t value_cache,
    typename E::out_t key_out,
    typename E::out_t value_out,
    uint64_t n,
    const std::string& type_name) {
  e.pipeline(kv_cache_kernel_name("clone", type_name));
  e.in(key_cache, 0);
  e.in(value_cache, 1);
  e.out(key_out, 2);
  e.out(value_out, 3);
  e.bytes(n, 4);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

// ----- KV cache block copy: in-place over output caches. mapping is (num_pairs,2) int64. -----
template <class E>
void launch_kv_cache_copy_blocks(
    E& e,
    typename E::out_t key_cache,
    typename E::out_t value_cache,
    typename E::in_t block_mapping,
    int num_pairs,
    int numel_per_block,
    const std::string& type_name) {
  e.pipeline(kv_cache_kernel_name("copy_blocks", type_name));
  e.out(key_cache, 0);
  e.out(value_cache, 1);
  e.in(block_mapping, 2);
  e.bytes(numel_per_block, 3);
  e.dispatch(num_pairs, 1, 1, 256, 1, 1);
}

// ----- KV cache scales: key@0 value@1 -> key_scale@2 value_scale@3 ; n@4.
// Single threadgroup scans the arrays and emits absmax / 240, matching vLLM's fp8 scale convention. -----
template <class E>
void launch_kv_cache_scales(
    E& e,
    typename E::in_t key,
    typename E::in_t value,
    typename E::out_t key_scale,
    typename E::out_t value_scale,
    uint64_t n,
    const std::string& type_name) {
  e.pipeline(kv_cache_kernel_name("scales", type_name));
  e.in(key, 0);
  e.in(value, 1);
  e.out(key_scale, 2);
  e.out(value_scale, 3);
  e.bytes(n, 4);
  e.dispatch(1, 1, 1, 256, 1, 1);
}

// ----- Paged decode attention: q@0 cacheK@1 cacheV@2 block_table@3 context_lens@4 -> out@5.
// q/out are (B, num_heads, D); caches are (num_blocks, block_size, num_kv_heads, D), D in {64,128}.
// GQA/MQA: num_heads may be a multiple of num_kv_heads (kv_head = head / (num_heads/num_kv_heads)). -----
template <class E>
void launch_paged_attention(
    E& e,
    typename E::in_t q,
    typename E::in_t key_cache,
    typename E::in_t value_cache,
    typename E::in_t block_table,
    typename E::in_t context_lens,
    typename E::out_t out,
    int batch,
    int num_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int block_table_stride,
    float scale,
    typename E::in_t alibi_slopes,
    int use_alibi,
    typename E::in_t block_mask,
    int use_mask,
    const std::string& type_name) {
  e.pipeline(paged_attention_kernel_name(type_name, head_size));
  e.in(q, 0);
  e.in(key_cache, 1);
  e.in(value_cache, 2);
  e.in(block_table, 3);
  e.in(context_lens, 4);
  e.out(out, 5);
  e.bytes(block_size, 6);
  e.bytes(block_table_stride, 7);
  e.bytes(scale, 8);
  e.bytes(num_heads, 9);
  e.bytes(num_kv_heads, 10);
  e.in(alibi_slopes, 11);
  e.bytes(use_alibi, 12);
  e.in(block_mask, 13);
  e.bytes(use_mask, 14);
  e.dispatch(num_heads, batch, 1, 32, 1, 1);
}

// vLLM x-packed cache decode: same grid as paged_attention; caches use vLLM's memory order,
// x@11 (= 16/sizeof(dtype)) selects the packed head-dim stride.
template <class E>
void launch_paged_attention_xcache(
    E& e, typename E::in_t q, typename E::in_t key_cache, typename E::in_t value_cache,
    typename E::in_t block_table, typename E::in_t context_lens, typename E::out_t out,
    int batch, int num_heads, int num_kv_heads, int head_size, int block_size,
    int block_table_stride, float scale, int x, const std::string& type_name) {
  e.pipeline(paged_attention_xcache_kernel_name(type_name, head_size));
  e.in(q, 0); e.in(key_cache, 1); e.in(value_cache, 2);
  e.in(block_table, 3); e.in(context_lens, 4); e.out(out, 5);
  e.bytes(block_size, 6); e.bytes(block_table_stride, 7); e.bytes(scale, 8);
  e.bytes(num_heads, 9); e.bytes(num_kv_heads, 10); e.bytes(x, 11);
  e.dispatch(num_heads, batch, 1, 32, 1, 1);
}

// GQA KV-reuse staged decode: grid (num_kv_heads, batch, 1), threadgroup (32, group_size, 1) —
// group_size simdgroups share one staged KV vector. Same buffer ABI as launch_paged_attention.
template <class E>
void launch_paged_attention_gqa_staged(
    E& e, typename E::in_t q, typename E::in_t key_cache, typename E::in_t value_cache,
    typename E::in_t block_table, typename E::in_t context_lens, typename E::out_t out,
    int batch, int num_heads, int num_kv_heads, int head_size, int block_size,
    int block_table_stride, float scale, const std::string& type_name) {
  const int group_size = num_heads / num_kv_heads;
  e.pipeline(paged_attention_gqa_staged_kernel_name(type_name, head_size));
  e.in(q, 0); e.in(key_cache, 1); e.in(value_cache, 2);
  e.in(block_table, 3); e.in(context_lens, 4); e.out(out, 5);
  e.bytes(block_size, 6); e.bytes(block_table_stride, 7); e.bytes(scale, 8);
  e.bytes(num_heads, 9); e.bytes(num_kv_heads, 10);
  e.dispatch(num_kv_heads, batch, 1, 32, group_size, 1);
}

// ----- fp8 KV cache: zero (uint8), scatter-with-encode, and dequant-on-read paged attention. -----
template <class E>
void launch_kv_cache_zero_u8(E& e, typename E::out_t key_cache, typename E::out_t value_cache,
                             uint64_t n) {
  e.pipeline("kv_cache_zero_u8");
  e.out(key_cache, 0); e.out(value_cache, 1); e.bytes(n, 2);
  constexpr int threads = 256;
  e.dispatch(static_cast<int>((n + threads - 1) / threads), 1, 1, threads, 1, 1);
}

template <class E>
void launch_kv_cache_scatter_fp8(E& e, typename E::in_t key, typename E::in_t value,
                                 typename E::in_t slot_mapping, typename E::out_t key_cache,
                                 typename E::out_t value_cache, int num_tokens, int num_heads,
                                 int head_size, int block_size, typename E::in_t k_scale,
                                 typename E::in_t v_scale, int fmt, const std::string& type_name) {
  e.pipeline(kv_cache_scatter_fp8_kernel_name(type_name));
  e.in(key, 0); e.in(value, 1); e.in(slot_mapping, 2);
  e.out(key_cache, 3); e.out(value_cache, 4);
  e.bytes(num_heads, 5); e.bytes(head_size, 6); e.bytes(block_size, 7);
  e.in(k_scale, 8); e.in(v_scale, 9); e.bytes(fmt, 10);
  e.dispatch(num_tokens, 1, 1, 256, 1, 1);
}

template <class E>
void launch_paged_attention_fp8(E& e, typename E::in_t q, typename E::in_t key_cache,
                                typename E::in_t value_cache, typename E::in_t block_table,
                                typename E::in_t context_lens, typename E::out_t out,
                                int batch, int num_heads, int num_kv_heads, int head_size,
                                int block_size, int block_table_stride, float scale,
                                typename E::in_t k_scale, typename E::in_t v_scale, int fmt,
                                const std::string& type_name) {
  e.pipeline(paged_attention_fp8_kernel_name(type_name, head_size));
  e.in(q, 0); e.in(key_cache, 1); e.in(value_cache, 2);
  e.in(block_table, 3); e.in(context_lens, 4); e.out(out, 5);
  e.bytes(block_size, 6); e.bytes(block_table_stride, 7); e.bytes(scale, 8);
  e.bytes(num_heads, 9); e.bytes(num_kv_heads, 10);
  e.in(k_scale, 11); e.in(v_scale, 12); e.bytes(fmt, 13);
  e.dispatch(num_heads, batch, 1, 32, 1, 1);
}

// ----- Paged attention v2 partition: q@0 cacheK@1 cacheV@2 block_table@3 context_lens@4 ->
//        tmp_out@5 max_logits@6 exp_sums@7 (all fp32) ; scalars 8..14 ; grid (H, B, P), 32 thr.
//        Each (head,batch,partition) does a local softmax over its KV slice. GQA-aware. -----
template <class E>
void launch_paged_attention_partition(
    E& e, typename E::in_t q, typename E::in_t key_cache, typename E::in_t value_cache,
    typename E::in_t block_table, typename E::in_t context_lens,
    typename E::out_t tmp_out, typename E::out_t max_logits, typename E::out_t exp_sums,
    int batch, int num_heads, int num_kv_heads, int head_size, int block_size,
    int block_table_stride, float scale, int num_partitions, int partition_size,
    const std::string& type_name) {
  e.pipeline(paged_attention_partition_kernel_name(type_name, head_size));
  e.in(q, 0); e.in(key_cache, 1); e.in(value_cache, 2);
  e.in(block_table, 3); e.in(context_lens, 4);
  e.out(tmp_out, 5); e.out(max_logits, 6); e.out(exp_sums, 7);
  e.bytes(block_size, 8); e.bytes(block_table_stride, 9); e.bytes(scale, 10);
  e.bytes(num_heads, 11); e.bytes(num_kv_heads, 12);
  e.bytes(num_partitions, 13); e.bytes(partition_size, 14);
  e.dispatch(num_heads, batch, num_partitions, 32, 1, 1);
}

// fp8 partition: uint8 caches, per-head k_scale/v_scale@15,16 (in), fmt@17. Reduce is reused.
template <class E>
void launch_paged_attention_partition_fp8(
    E& e, typename E::in_t q, typename E::in_t key_cache, typename E::in_t value_cache,
    typename E::in_t block_table, typename E::in_t context_lens,
    typename E::out_t tmp_out, typename E::out_t max_logits, typename E::out_t exp_sums,
    int batch, int num_heads, int num_kv_heads, int head_size, int block_size,
    int block_table_stride, float scale, int num_partitions, int partition_size,
    typename E::in_t k_scale, typename E::in_t v_scale, int fmt, const std::string& type_name) {
  e.pipeline(paged_attention_partition_fp8_kernel_name(type_name, head_size));
  e.in(q, 0); e.in(key_cache, 1); e.in(value_cache, 2);
  e.in(block_table, 3); e.in(context_lens, 4);
  e.out(tmp_out, 5); e.out(max_logits, 6); e.out(exp_sums, 7);
  e.bytes(block_size, 8); e.bytes(block_table_stride, 9); e.bytes(scale, 10);
  e.bytes(num_heads, 11); e.bytes(num_kv_heads, 12);
  e.bytes(num_partitions, 13); e.bytes(partition_size, 14);
  e.in(k_scale, 15); e.in(v_scale, 16); e.bytes(fmt, 17);
  e.dispatch(num_heads, batch, num_partitions, 32, 1, 1);
}

// ----- Paged attention v2 reduce: tmp_out@0 max_logits@1 exp_sums@2 (fp32) -> out@3 ;
//        num_heads@4 num_partitions@5 ; grid (H, B, 1), 32 threads. LSE merge over partitions. -----
template <class E>
void launch_paged_attention_reduce(
    E& e, typename E::in_t tmp_out, typename E::in_t max_logits, typename E::in_t exp_sums,
    typename E::out_t out, int batch, int num_heads, int head_size, int num_partitions,
    const std::string& type_name) {
  e.pipeline(paged_attention_reduce_kernel_name(type_name, head_size));
  e.in(tmp_out, 0); e.in(max_logits, 1); e.in(exp_sums, 2); e.out(out, 3);
  e.bytes(num_heads, 4); e.bytes(num_partitions, 5);
  e.dispatch(num_heads, batch, 1, 32, 1, 1);
}

// ----- attn_causal: q@0 k@1 v@2 -> o@3 ; N@4(u32) H@5(u32) ; grid (N/8, H, B) group (32,1,1) -----
// Same as attn_fwd but with causal masking (lower-triangular).
template <class E>
void launch_attn_causal(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                        typename E::out_t o, unsigned N, unsigned H, int B, int D) {
  e.pipeline(attn_causal_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- flux_gelu: D@0 A@1 B@2 bias@3 ; N@4 K@5 M@6 (i32) ; grid (M/32, N/32, 1) -----
// out = gelu(A@B + bias); A (N,K), B (K,M), bias (M,).
template <class E>
void launch_flux_gelu(E& e, typename E::out_t d, typename E::in_t a, typename E::in_t b,
                      typename E::in_t bias, int N, int K, int M, const std::string& t) {
  e.pipeline(flux_gelu_kernel_name(t));
  e.out(d, 0); e.in(a, 1); e.in(b, 2); e.in(bias, 3);
  e.bytes(N, 4); e.bytes(K, 5); e.bytes(M, 6);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- flux_gate: D@0 A@1 B@2 bias@3 gate@4 residual@5 ; N@6 K@7 M@8 ; grid (M/32, N/32, 1) -----
// out = (A@B + bias) * gate + residual.
template <class E>
void launch_flux_gate(E& e, typename E::out_t d, typename E::in_t a, typename E::in_t b,
                      typename E::in_t bias, typename E::in_t gate, typename E::in_t resid,
                      int N, int K, int M, const std::string& t) {
  e.pipeline(flux_gate_kernel_name(t));
  e.out(d, 0); e.in(a, 1); e.in(b, 2); e.in(bias, 3); e.in(gate, 4); e.in(resid, 5);
  e.bytes(N, 6); e.bytes(K, 7); e.bytes(M, 8);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- gemm_staged: D@0 A@1 B@2 ; N@3 K@4 M@5 (i32) ; grid (M/32, N/32, 1), 64 threads
//        (2 simdgroups) per threadgroup. A (N,K), B (K,M), out (N,M). A bigger 4-simdgroup
//        BM=128 tile was benchmarked and is slower (see gemm_staged.metal). -----
template <class E>
void launch_gemm_staged(E& e, typename E::out_t d, typename E::in_t a, typename E::in_t b,
                        int N, int K, int M, const std::string& t) {
  e.pipeline(gemm_staged_kernel_name(t));
  e.out(d, 0); e.in(a, 1); e.in(b, 2);
  e.bytes(N, 3); e.bytes(K, 4); e.bytes(M, 5);
  e.dispatch(M / 32, N / 32, 1, 64, 1, 1);  // 64 threads = 2 simdgroups
}

// ----- attn_multiwarp: q@0 k@1 v@2 -> o@3 ; N@4(u32) H@5(u32) ; grid (N/32, H, B),
//        128 threads (4 simdgroups) per threadgroup; shared K/V across warps. -----
template <class E>
void launch_attn_multiwarp(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                           typename E::out_t o, unsigned N, unsigned H, int B, int D) {
  constexpr int NUM_WARPS = 4;  // 2 vs 4 benchmarked equivalent (both ~5% behind attn_fwd)
  e.pipeline(attn_multiwarp_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.dispatch(static_cast<int>(N) / (8 * NUM_WARPS), static_cast<int>(H), B,
             32 * NUM_WARPS, 1, 1);
}

// ----- linear_attn: q@0 k@1 v@2 -> o@3 ; N@4(u32) H@5(u32) ; grid (1, H, B) group (32,1,1).
//        Non-causal linear attention out = Q @ (K^T @ V). q,k,v,o (B,H,N,D), D=64. -----
template <class E>
void launch_linear_attn(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                        typename E::out_t o, unsigned N, unsigned H, int B, int D) {
  e.pipeline(linear_attn_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.dispatch(1, static_cast<int>(H), B, 32, 1, 1);
}

// ----- hedgehog: q@0 k@1 v@2 -> o@3 ; N@4(u32) H@5(u32) ; grid (1, H, B) group (32,1,1).
//        Feature-map linear attention out = phi(Q) @ (phi(K)^T @ V), D=64. -----
template <class E>
void launch_hedgehog(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                     typename E::out_t o, unsigned N, unsigned H, int B, int D) {
  e.pipeline(hedgehog_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.dispatch(1, static_cast<int>(H), B, 32, 1, 1);
}

// ----- lin_attn_causal: q@0 k@1 v@2 -> o@3 ; N@4(u32) H@5(u32) ; grid (1, H, B) group (32,1,1).
//        Causal linear attention (chunked running-KV scan), D=64. -----
template <class E>
void launch_lin_attn_causal(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                            typename E::out_t o, unsigned N, unsigned H, int B, int D) {
  e.pipeline(lin_attn_causal_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.dispatch(1, static_cast<int>(H), B, 32, 1, 1);
}

// ----- mamba2 (SSD): C@0 B@1 X@2 cumlog@3 -> Y@4 ; N@5(u32) H@6(u32) ;
//        grid (N/8, H, B) group (32,1,1). C,B,X,Y (B,H,N,D) bf16; cumlog (B,H,N) fp32. -----
template <class E>
void launch_mamba2(E& e, typename E::in_t C, typename E::in_t Bm, typename E::in_t X,
                   typename E::in_t cumlog, typename E::out_t Y, unsigned N, unsigned H,
                   int B, int D) {
  e.pipeline(mamba2_kernel_name(D));
  e.in(C, 0); e.in(Bm, 1); e.in(X, 2); e.in(cumlog, 3); e.out(Y, 4);
  e.bytes(N, 5); e.bytes(H, 6);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- mamba2 backward. C@0 Bm@1 X@2 dY@3 cumlog@4 -> dC@5 (bwd_i) / dB@5 dX@6 (bwd_j). -----
template <class E>
void launch_mamba2_bwd_i(E& e, typename E::in_t C, typename E::in_t Bm, typename E::in_t X,
                         typename E::in_t dY, typename E::in_t cumlog, typename E::out_t dC,
                         unsigned N, unsigned H, int B, int D) {
  e.pipeline(mamba2_bwd_i_kernel_name(D));
  e.in(C, 0); e.in(Bm, 1); e.in(X, 2); e.in(dY, 3); e.in(cumlog, 4); e.out(dC, 5);
  e.bytes(N, 6); e.bytes(H, 7);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}
template <class E>
void launch_mamba2_bwd_j(E& e, typename E::in_t C, typename E::in_t Bm, typename E::in_t X,
                         typename E::in_t dY, typename E::in_t cumlog, typename E::out_t dB,
                         typename E::out_t dX, unsigned N, unsigned H, int B, int D) {
  e.pipeline(mamba2_bwd_j_kernel_name(D));
  e.in(C, 0); e.in(Bm, 1); e.in(X, 2); e.in(dY, 3); e.in(cumlog, 4); e.out(dB, 5); e.out(dX, 6);
  e.bytes(N, 7); e.bytes(H, 8);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- attn backward family (FlashAttention-2 bwd). All grid (N/8, H, B) group (32,1,1). -----
template <class E>
void launch_attn_fwd_l(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                       typename E::out_t o, typename E::out_t L, unsigned N, unsigned H, int B, int D,
                       bool causal) {
  e.pipeline("attn_fwd_l_" + std::string(causal ? "causal_" : "noncausal_") + std::to_string(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3); e.out(L, 4);
  e.bytes(N, 5); e.bytes(H, 6);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}
template <class E>
void launch_attn_bwd_prep(E& e, typename E::in_t o, typename E::in_t ddo, typename E::out_t delta,
                          unsigned N, unsigned H, int B, int D) {
  e.pipeline("attn_bwd_prep_" + std::to_string(D));
  e.in(o, 0); e.in(ddo, 1); e.out(delta, 2);
  e.bytes(N, 3); e.bytes(H, 4);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}
template <class E>
void launch_attn_bwd_dq(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                        typename E::in_t ddo, typename E::in_t L, typename E::in_t delta,
                        typename E::out_t dq, unsigned N, unsigned H, int B, int D, bool causal) {
  e.pipeline("attn_bwd_dq_" + std::string(causal ? "causal_" : "noncausal_") + std::to_string(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.in(ddo, 3); e.in(L, 4); e.in(delta, 5); e.out(dq, 6);
  e.bytes(N, 7); e.bytes(H, 8);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}
template <class E>
void launch_attn_bwd_dkv(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                         typename E::in_t ddo, typename E::in_t L, typename E::in_t delta,
                         typename E::out_t dk, typename E::out_t dv, unsigned N, unsigned H, int B,
                         int D, bool causal) {
  e.pipeline("attn_bwd_dkv_" + std::string(causal ? "causal_" : "noncausal_") + std::to_string(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.in(ddo, 3); e.in(L, 4); e.in(delta, 5); e.out(dk, 6); e.out(dv, 7);
  e.bytes(N, 8); e.bytes(H, 9);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- lin_attn_decay (retention): q@0 k@1 v@2 cl@3(=-slope*pos) -> o@4 ; N@5(u32) H@6(u32) ;
//        grid (N/8, H, B) group (32,1,1). q,k,v,o (B,H,N,D) bf16; cl (B,H,N) fp32. -----
template <class E>
void launch_lin_attn_decay(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                           typename E::in_t cl, typename E::out_t o, unsigned N, unsigned H,
                           int B, int D) {
  e.pipeline(lin_attn_decay_kernel_name(D));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.in(cl, 3); e.out(o, 4);
  e.bytes(N, 5); e.bytes(H, 6);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- based (Taylor feature-map linear attention): q@0 k@1 (D_QK) v@2 (D_VO) -> o@3 ; N@4 H@5 ;
//        grid (N/8, H, B) group (32,1,1). q,k (B,H,N,16) v,o (B,H,N,64) bf16. -----
template <class E>
void launch_based(E& e, typename E::in_t q, typename E::in_t k, typename E::in_t v,
                  typename E::out_t o, unsigned N, unsigned H, int B, int DQK, int DVO) {
  e.pipeline(based_kernel_name(DQK, DVO));
  e.in(q, 0); e.in(k, 1); e.in(v, 2); e.out(o, 3);
  e.bytes(N, 4); e.bytes(H, 5);
  e.dispatch(static_cast<int>(N) / 8, static_cast<int>(H), B, 32, 1, 1);
}

// ----- cmplx_matmul: D@0 A@1 B@2 ; N@3 K@4 M@5 (i32) ; grid (M/32, N/32, 1) group (32,1,1).
//        Complex GEMM D = A @ B; each operand has a leading size-2 (real,imag) axis:
//        A (2,N,K), B (2,K,M), D (2,N,M). Uses the complex_mma_AB primitive. -----
template <class E>
void launch_cmplx_matmul(E& e, typename E::out_t d, typename E::in_t a, typename E::in_t b,
                         int N, int K, int M, const std::string& t) {
  const bool use_small = K < 512;
  e.pipeline(cmplx_matmul_kernel_name(t) + (use_small ? "_small" : ""));
  e.out(d, 0); e.in(a, 1); e.in(b, 2);
  e.bytes(N, 3); e.bytes(K, 4); e.bytes(M, 5);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- fftconv (Monarch FFT convolution): OUT@0 X@1 F@2 TWF@3 FINV@4 TWI@5 KF@6 ;
//        BH@7 H@8 (i32) ; grid (BH,1,1) group (32,1,1). N = S*S; S in {16,32}.
//        Complex arrays carry a leading size-2 (real,imag) axis; OUT is real (BH,S,S). -----
template <class E>
void launch_fftconv(E& e, typename E::out_t out, typename E::in_t x, typename E::in_t F,
                    typename E::in_t twf, typename E::in_t finv, typename E::in_t twi,
                    typename E::in_t kf, int BH, int H, int S) {
  e.pipeline(fftconv_kernel_name(S));
  e.out(out, 0); e.in(x, 1); e.in(F, 2); e.in(twf, 3);
  e.in(finv, 4); e.in(twi, 5); e.in(kf, 6);
  e.bytes(BH, 7); e.bytes(H, 8);
  e.dispatch(BH, 1, 1, 32, 1, 1);
}

// ----- qgemm (quantized GEMM, dequant-to-shared): D@0 Wq@1 X@2 ; N@3 K@4 M@5 (i32) ;
//        grid (M/32, N/32, 1), 64 threads (2 simdgroups). D=W@X, W (N,K) quantized blocks
//        (format `fmt`), X (K,M) half, D (N,M) half. -----
template <class E>
void launch_qgemm(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t x,
                  int N, int K, int M, const std::string& fmt) {
  e.pipeline(qgemm_kernel_name(fmt));
  e.out(d, 0); e.in(wq, 1); e.in(x, 2);
  e.bytes(N, 3); e.bytes(K, 4); e.bytes(M, 5);
  e.dispatch(M / 32, N / 32, 1, 64, 1, 1);  // 64 threads = 2 simdgroups, BM=32
}

// ----- qgemm_actorder: GPTQ act-order, in-kernel g_idx gather. D@0 Wq@1 X@2 perm@3(int) ; N@4 K@5
//        M@6 ; grid (M/32, N/32, 1), 32 threads. Gathers X K-rows by perm during the X load. -----
template <class E>
void launch_qgemm_actorder(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t x,
                           typename E::in_t perm, int N, int K, int M, const std::string& fmt) {
  e.pipeline(qgemm_actorder_kernel_name(fmt));
  e.out(d, 0); e.in(wq, 1); e.in(x, 2); e.in(perm, 3);
  e.bytes(N, 4); e.bytes(K, 5); e.bytes(M, 6);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- qgemm_fp8_scaled: both operands fp8 e4m3, rank-1 scaled. D@0 Wq@1(N,K fp8) Xq@2(K,M fp8)
//        w_scale@3(N) a_scale@4(M) ; N@5 K@6 M@7 ; grid (M/32, N/32, 1), 32 threads. -----
template <class E>
void launch_qgemm_fp8_scaled(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t xq,
                             typename E::in_t wscale, typename E::in_t ascale, int N, int K, int M) {
  e.pipeline("mittens::qgemm_fp8_scaled");
  e.out(d, 0); e.in(wq, 1); e.in(xq, 2); e.in(wscale, 3); e.in(ascale, 4);
  e.bytes(N, 5); e.bytes(K, 6); e.bytes(M, 7);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- qgemm_blockscale (fp8_block2d): D@0 Wq@1(codes) X@2 scale2d@3 ; N@4 K@5 M@6 ; grid
//        (M/32, N/32, 1), 32 threads. Separate (N/128,K/128) tile scale. -----
template <class E>
void launch_qgemm_blockscale(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t x,
                             typename E::in_t scale2d, int N, int K, int M) {
  e.pipeline("qgemm_blockscale_fp8_raw");
  e.out(d, 0); e.in(wq, 1); e.in(x, 2); e.in(scale2d, 3);
  e.bytes(N, 4); e.bytes(K, 5); e.bytes(M, 6);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);
}

// ----- qgemm_frag: dequant-direct-to-fragment. D@0 Wq@1 X@2 ; N@3 K@4 M@5 ; grid (M/32, N/32, 1),
//        32 threads (1 simdgroup) per 32x32 output tile. No shared staging / barrier. -----
template <class E>
void launch_qgemm_frag(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t x,
                       int N, int K, int M, const std::string& fmt) {
  e.pipeline(qgemm_frag_kernel_name(fmt));
  e.out(d, 0); e.in(wq, 1); e.in(x, 2);
  e.bytes(N, 3); e.bytes(K, 4); e.bytes(M, 5);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);  // 32 threads = 1 simdgroup
}

// ----- qgemv (quantized GEMV, batch-1 decode): D@0 Wq@1 X@2 ; N@3 K@4 (i32) ;
//        grid (N,1,1), 32 threads (1 simdgroup) per output row. d = W @ x, x (K,1) half. -----
template <class E>
void launch_qgemv(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t x,
                  int N, int K, const std::string& fmt) {
  const bool use_small = K <= 512 && (fmt == "q8_0" || fmt == "q4_0");
  e.pipeline(use_small ? qgemv_kernel_name(fmt) + "_small" : qgemv_kernel_name(fmt));
  e.out(d, 0); e.in(wq, 1); e.in(x, 2);
  e.bytes(N, 3); e.bytes(K, 4);
  e.dispatch(N, 1, 1, 32, 1, 1);  // one simdgroup per output row
}

// ----- qgemv_w8a8 (W8A8 int8xint8 decode): D@0 Wq@1(int8) Xq@2(int8) w_scale@3 a_scale@4 ;
//        N@5 K@6 (i32) ; grid (N,1,1) 32 threads. int32 accumulate then *w_scale[n]*a_scale. -----
template <class E>
void launch_qgemv_w8a8(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t xq,
                       typename E::in_t wscale, typename E::in_t ascale, int N, int K) {
  e.pipeline("mittens::qgemv_w8a8");  // non-template kernel keeps its namespaced symbol
  e.out(d, 0); e.in(wq, 1); e.in(xq, 2); e.in(wscale, 3); e.in(ascale, 4);
  e.bytes(N, 5); e.bytes(K, 6);
  e.dispatch(N, 1, 1, 32, 1, 1);
}

// ----- qgemv_w2a8 (BitNet W2A8 int2xint8 decode): D@0 Wq@1(bitnet blocks) Xq@2(int8) a_scale@3 ;
//        N@4 K@5 (i32) ; grid (N,1,1) 32 threads. per-group int32 sums * absmean scale * a_scale. -----
template <class E>
void launch_qgemv_w2a8(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t xq,
                       typename E::in_t ascale, int N, int K) {
  e.pipeline("mittens::qgemv_w2a8");  // non-template kernel keeps its namespaced symbol
  e.out(d, 0); e.in(wq, 1); e.in(xq, 2); e.in(ascale, 3);
  e.bytes(N, 4); e.bytes(K, 5);
  e.dispatch(N, 1, 1, 32, 1, 1);
}

// ----- qgemm_w8a8 (W8A8 int8xint8 PREFILL, M>1): D@0 Wq@1(int8 N,K) Xq@2(int8 M,K) w_scale@3
//        a_scale@4 ; N@5 K@6 M@7 ; grid (N,1,1) 32 threads. Exact int32, scaled once. -----
template <class E>
void launch_qgemm_w8a8(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t xq,
                       typename E::in_t wscale, typename E::in_t ascale, int N, int K, int M) {
  e.pipeline("mittens::qgemm_w8a8");
  e.out(d, 0); e.in(wq, 1); e.in(xq, 2); e.in(wscale, 3); e.in(ascale, 4);
  e.bytes(N, 5); e.bytes(K, 6); e.bytes(M, 7);
  e.dispatch(N, 1, 1, 32, 1, 1);
}

// ----- qgemm_w2a8 (BitNet W2A8 prefill): D@0 Wq@1(blocks) Xq@2(int8 M,K) a_scale@3 ; N@4 K@5 M@6. ---
template <class E>
void launch_qgemm_w2a8(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t xq,
                       typename E::in_t ascale, int N, int K, int M) {
  e.pipeline("mittens::qgemm_w2a8");
  e.out(d, 0); e.in(wq, 1); e.in(xq, 2); e.in(ascale, 3);
  e.bytes(N, 4); e.bytes(K, 5); e.bytes(M, 6);
  e.dispatch(N, 1, 1, 32, 1, 1);
}

// ----- qflux_gelu (quantized fused GEMM+GELU): D@0 Wq@1 X@2 bias@3 ; N@4 K@5 M@6 (i32) ;
//        grid (M/32, N/32, 1), 32 threads (1 simdgroup, dequant-direct-to-fragment). -----
template <class E>
void launch_qflux_gelu(E& e, typename E::out_t d, typename E::in_t wq, typename E::in_t x,
                       typename E::in_t bias, int N, int K, int M, const std::string& fmt) {
  e.pipeline(qflux_gelu_kernel_name(fmt));
  e.out(d, 0); e.in(wq, 1); e.in(x, 2); e.in(bias, 3);
  e.bytes(N, 4); e.bytes(K, 5); e.bytes(M, 6);
  e.dispatch(M / 32, N / 32, 1, 32, 1, 1);  // 1 simdgroup per 32x32 tile
}

} // namespace tk
