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
inline std::string softmax_kernel_name(int D) { return "softmax_" + std::to_string(D); }
inline std::string rotary_kernel_name(int D) { return "rotary_" + std::to_string(D); }
inline std::string gelu_kernel_name(int D) { return "gelu_" + std::to_string(D); }
inline std::string attn_causal_kernel_name(int D) { return "attn_causal_" + std::to_string(D); }
inline std::string flux_gelu_kernel_name(const std::string& t) { return "flux_gelu_" + t; }
inline std::string flux_gate_kernel_name(const std::string& t) { return "flux_gate_" + t; }
inline std::string gemm_staged_kernel_name(const std::string& t) { return "gemm_staged_" + t; }
inline std::string attn_multiwarp_kernel_name(int D) { return "attn_multiwarp_" + std::to_string(D); }
inline std::string linear_attn_kernel_name(int D) { return "linear_attn_" + std::to_string(D); }
inline std::string hedgehog_kernel_name(int D) { return "hedgehog_" + std::to_string(D); }
inline std::string lin_attn_causal_kernel_name(int D) { return "lin_attn_causal_" + std::to_string(D); }
inline std::string mamba2_kernel_name(int D) { return "mamba2_" + std::to_string(D); }
inline std::string cmplx_matmul_kernel_name(const std::string& t) { return "cmplx_matmul_" + t; }
inline std::string fftconv_kernel_name(int S) { return "fftconv_" + std::to_string(S); }
inline std::string qgemm_kernel_name(const std::string& fmt) { return "qgemm_" + fmt; }
inline std::string qgemv_kernel_name(const std::string& fmt) { return "qgemv_" + fmt; }
inline std::string qflux_gelu_kernel_name(const std::string& fmt) { return "qflux_gelu_" + fmt; }
inline std::string qgemm_frag_kernel_name(const std::string& fmt) { return "qgemm_frag_" + fmt; }

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

// ----- rms_norm: x@0 w@1 -> o@2 ; M@3(u32) eps@4(f32) ; grid (M,1,1) group (32,1,1) -----
template <class E>
void launch_rms_norm(E& e, typename E::in_t x, typename E::in_t w,
                     typename E::out_t o, uint32_t M, int D, float eps) {
  e.pipeline(rms_norm_kernel_name(D));
  e.in(x, 0); e.in(w, 1); e.out(o, 2);
  e.bytes(M, 3); e.bytes(eps, 4);
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
                   typename E::out_t o, uint32_t M, unsigned N, int D) {
  e.pipeline(rotary_kernel_name(D));
  e.in(x, 0); e.in(cos, 1); e.in(sin, 2); e.out(o, 3);
  e.bytes(N, 4);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
}

// ----- gelu (elementwise, last axis): x@0 -> o@1 ; M@2(u32) ; grid (M,1,1) group (32,1,1) -----
template <class E>
void launch_gelu(E& e, typename E::in_t x, typename E::out_t o, uint32_t M, int D) {
  e.pipeline(gelu_kernel_name(D));
  e.in(x, 0); e.out(o, 1);
  e.bytes(M, 2);
  e.dispatch(static_cast<int>(M), 1, 1, 32, 1, 1);
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

// ----- cmplx_matmul: D@0 A@1 B@2 ; N@3 K@4 M@5 (i32) ; grid (M/32, N/32, 1) group (32,1,1).
//        Complex GEMM D = A @ B; each operand has a leading size-2 (real,imag) axis:
//        A (2,N,K), B (2,K,M), D (2,N,M). Uses the complex_mma_AB primitive. -----
template <class E>
void launch_cmplx_matmul(E& e, typename E::out_t d, typename E::in_t a, typename E::in_t b,
                         int N, int K, int M, const std::string& t) {
  e.pipeline(cmplx_matmul_kernel_name(t));
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
  e.pipeline(qgemv_kernel_name(fmt));
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
