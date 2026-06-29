// PyTorch MPS backend for the ThunderMittens kernels.
//
// The compute lives in the shared, framework-agnostic .metal kernels (compiled to a
// .metallib). This file is the thin host glue that dispatches those kernels onto
// PyTorch's MPS stream — the analogue of the MLX Primitive `eval_gpu` in <kernel>.cpp.
//
// The per-kernel host ABI (name, buffer indices, params, grid/threadgroup geometry)
// is the single source of truth in ../tk_launch.h; this file only provides a Torch
// "encoder adapter" and the tensor<->buffer plumbing.

#include <torch/extension.h>
#include <torch/mps.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>
#include <unordered_map>

#include "tk_launch.h"

// The MTLBuffer backing an MPS tensor's storage (documented PyTorch pattern).
static inline id<MTLBuffer> mtl_buffer(const at::Tensor& t) {
  return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
}
static inline NSUInteger byte_offset(const at::Tensor& t) {
  return static_cast<NSUInteger>(t.storage_offset()) * t.element_size();
}
static inline std::string tk_type_name(const at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kFloat: return "float32";
    case at::kHalf: return "float16";
    case at::kBFloat16: return "bfloat16";
    default: TORCH_CHECK(false, "tk_torch: unsupported dtype ", t.scalar_type());
  }
}

// ---- lazily-loaded metallib + pipeline-state cache (keyed by function name) ----
static std::string g_metallib_path;
static id<MTLLibrary> g_library = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;

static void tk_set_library(const std::string& path) {
  g_metallib_path = path;
  g_library = nil;
  g_pipelines.clear();
}

static id<MTLComputePipelineState> tk_pipeline(id<MTLDevice> device, NSString* name) {
  std::string key = name.UTF8String;
  auto it = g_pipelines.find(key);
  if (it != g_pipelines.end()) return it->second;

  NSError* err = nil;
  if (g_library == nil) {
    TORCH_CHECK(!g_metallib_path.empty(),
                "tk_torch: metallib path not set; call _set_library() first");
    NSString* p = [NSString stringWithUTF8String:g_metallib_path.c_str()];
    g_library = [device newLibraryWithURL:[NSURL fileURLWithPath:p] error:&err];
    TORCH_CHECK(g_library != nil, "tk_torch: failed to load metallib at ", g_metallib_path);
  }
  id<MTLFunction> fn = [g_library newFunctionWithName:name];
  TORCH_CHECK(fn != nil, "tk_torch: kernel function not found: ", name.UTF8String);
  id<MTLComputePipelineState> pso =
      [device newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso != nil, "tk_torch: failed to create pipeline for ", name.UTF8String);
  g_pipelines[key] = pso;
  return pso;
}

// ---- Torch encoder adapter: drives tk::launch_<name>() (see tk_launch.h) ----
struct TorchEncoder {
  using in_t = const at::Tensor&;
  using out_t = const at::Tensor&;
  id<MTLComputeCommandEncoder> enc;
  id<MTLDevice> device;
  void pipeline(const std::string& name) {
    [enc setComputePipelineState:tk_pipeline(device,
                                             [NSString stringWithUTF8String:name.c_str()])];
  }
  void in(const at::Tensor& t, int i) {
    [enc setBuffer:mtl_buffer(t) offset:byte_offset(t) atIndex:i];
  }
  void out(const at::Tensor& t, int i) {
    [enc setBuffer:mtl_buffer(t) offset:byte_offset(t) atIndex:i];
  }
  template <class T>
  void bytes(const T& v, int i) {
    [enc setBytes:&v length:sizeof(T) atIndex:i];
  }
  void dispatch(int gx, int gy, int gz, int tx, int ty, int tz) {
    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz)
        threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
  }
};

// Run `fn(encoder)` on torch's MPS stream. The command buffer is torch's current one;
// it is committed at the next stream sync (e.g. .cpu()/torch.mps.synchronize()).
template <class F>
static void tk_encode(F fn) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
    dispatch_queue_t q = torch::mps::get_dispatch_queue();
    id<MTLDevice> dev = cb.device;
    dispatch_sync(q, ^{
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      TorchEncoder e{enc, dev};
      fn(e);
      [enc endEncoding];
    });
  }
}

// ----------------------------- kernels -----------------------------
static at::Tensor layernorm_mps(const at::Tensor& x_in, const at::Tensor& w_in,
                                const at::Tensor& b_in, double eps) {
  TORCH_CHECK(x_in.device().is_mps(), "layernorm: x must be an MPS tensor");
  TORCH_CHECK(x_in.scalar_type() == at::kBFloat16, "layernorm: x must be bfloat16");
  auto x = x_in.contiguous(), w = w_in.contiguous(), b = b_in.contiguous();
  const int D = x.size(-1);
  TORCH_CHECK(D == 256 || D == 512 || D == 768 || D == 1024,
              "layernorm: last dim must be 256/512/768/1024");
  const uint32_t M = static_cast<uint32_t>(x.numel() / D);
  auto out = at::empty_like(x);
  const float eps_f = static_cast<float>(eps);
  tk_encode([&](TorchEncoder& e) { tk::launch_layernorm(e, x, w, b, out, M, D, eps_f); });
  return out;
}

static at::Tensor add_rt_mps(const at::Tensor& x_in, const at::Tensor& y_in) {
  TORCH_CHECK(x_in.device().is_mps(), "add_rt: x must be an MPS tensor");
  TORCH_CHECK(x_in.sizes() == y_in.sizes(), "add_rt: x and y must have the same shape");
  auto x = x_in.contiguous(), y = y_in.contiguous();
  TORCH_CHECK(x.dim() == 2, "add_rt: expects 2D inputs");
  const int rows = x.size(0), cols = x.size(1);
  TORCH_CHECK(rows % 8 == 0 && cols % 8 == 0, "add_rt: both dims must be multiples of 8");
  auto out = at::empty_like(x);
  const std::string tn = tk_type_name(x);
  tk_encode([&](TorchEncoder& e) { tk::launch_add_rt(e, x, y, out, rows, cols, tn); });
  return out;
}

static at::Tensor matmul_custom_mps(const at::Tensor& x_in, const at::Tensor& y_in) {
  TORCH_CHECK(x_in.device().is_mps(), "matmul_custom: x must be an MPS tensor");
  auto x = x_in.contiguous(), y = y_in.contiguous();
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && x.size(1) == y.size(0),
              "matmul_custom: expects (N,K) @ (K,M)");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
              "matmul_custom: dtype must be float32 or bfloat16");
  const int N = x.size(0), K = x.size(1), M = y.size(1);
  TORCH_CHECK(N % 32 == 0 && M % 32 == 0 && K % 16 == 0,
              "matmul_custom: requires N%32==0, M%32==0, K%16==0");
  auto out = at::empty({N, M}, x.options());
  const std::string tn = tk_type_name(x);
  tk_encode([&](TorchEncoder& e) { tk::launch_matmul_custom(e, out, x, y, N, K, M, tn); });
  return out;
}

static at::Tensor attn_fwd_mps(const at::Tensor& q_in, const at::Tensor& k_in,
                               const at::Tensor& v_in) {
  TORCH_CHECK(q_in.device().is_mps(), "attn_fwd: q must be an MPS tensor");
  TORCH_CHECK(q_in.scalar_type() == at::kBFloat16, "attn_fwd: q must be bfloat16");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  TORCH_CHECK(q.dim() == 4, "attn_fwd: expects (B,H,N,D)");
  const int B = q.size(0), H = q.size(1);
  const unsigned N = static_cast<unsigned>(q.size(2));
  const int D = q.size(3);
  TORCH_CHECK(D == 64 || D == 128, "attn_fwd: D must be 64 or 128");
  TORCH_CHECK(N % 8 == 0, "attn_fwd: N must be a multiple of 8");
  auto out = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) { tk::launch_attn_fwd(e, q, k, v, out, N, H, B, D); });
  return out;
}

static at::Tensor rms_norm_mps(const at::Tensor& x_in, const at::Tensor& w_in, double eps) {
  TORCH_CHECK(x_in.device().is_mps(), "rms_norm: x must be an MPS tensor");
  TORCH_CHECK(x_in.scalar_type() == at::kBFloat16, "rms_norm: x must be bfloat16");
  auto x = x_in.contiguous(), w = w_in.contiguous();
  const int D = x.size(-1);
  TORCH_CHECK(D == 256 || D == 512 || D == 768 || D == 1024,
              "rms_norm: last dim must be 256/512/768/1024");
  const uint32_t M = static_cast<uint32_t>(x.numel() / D);
  auto out = at::empty_like(x);
  const float eps_f = static_cast<float>(eps);
  tk_encode([&](TorchEncoder& e) { tk::launch_rms_norm(e, x, w, out, M, D, eps_f); });
  return out;
}

static at::Tensor softmax_mps(const at::Tensor& x_in) {
  TORCH_CHECK(x_in.device().is_mps(), "softmax: x must be an MPS tensor");
  TORCH_CHECK(x_in.scalar_type() == at::kBFloat16, "softmax: x must be bfloat16");
  auto x = x_in.contiguous();
  const int D = x.size(-1);
  TORCH_CHECK(D == 256 || D == 512 || D == 768 || D == 1024,
              "softmax: last dim must be 256/512/768/1024");
  const uint32_t M = static_cast<uint32_t>(x.numel() / D);
  auto out = at::empty_like(x);
  tk_encode([&](TorchEncoder& e) { tk::launch_softmax(e, x, out, M, D); });
  return out;
}

static at::Tensor rotary_mps(const at::Tensor& x_in, const at::Tensor& cos_in,
                             const at::Tensor& sin_in) {
  TORCH_CHECK(x_in.device().is_mps(), "rotary: x must be an MPS tensor");
  TORCH_CHECK(x_in.scalar_type() == at::kBFloat16, "rotary: x must be bfloat16");
  TORCH_CHECK(x_in.dim() == 4, "rotary: x must be (B,H,N,D)");
  auto x = x_in.contiguous(), cos = cos_in.contiguous(), sin = sin_in.contiguous();
  const int D = x.size(-1);
  const unsigned N = static_cast<unsigned>(x.size(-2));
  TORCH_CHECK(D == 64 || D == 128, "rotary: head dim must be 64 or 128");
  TORCH_CHECK(cos.size(-1) == D / 2 && sin.size(-1) == D / 2 &&
              cos.size(-2) == (int64_t)N && sin.size(-2) == (int64_t)N,
              "rotary: cos/sin must be (N, D/2)");
  const uint32_t M = static_cast<uint32_t>(x.numel() / D);
  auto out = at::empty_like(x);
  tk_encode([&](TorchEncoder& e) { tk::launch_rotary(e, x, cos, sin, out, M, N, D); });
  return out;
}

static at::Tensor gelu_mps(const at::Tensor& x_in) {
  TORCH_CHECK(x_in.device().is_mps(), "gelu: x must be an MPS tensor");
  TORCH_CHECK(x_in.scalar_type() == at::kBFloat16, "gelu: x must be bfloat16");
  auto x = x_in.contiguous();
  const int D = x.size(-1);
  TORCH_CHECK(D == 256 || D == 512 || D == 768 || D == 1024,
              "gelu: last dim must be 256/512/768/1024");
  const uint32_t M = static_cast<uint32_t>(x.numel() / D);
  auto out = at::empty_like(x);
  tk_encode([&](TorchEncoder& e) { tk::launch_gelu(e, x, out, M, D); });
  return out;
}

static at::Tensor attn_causal_mps(const at::Tensor& q_in, const at::Tensor& k_in,
                                  const at::Tensor& v_in) {
  TORCH_CHECK(q_in.device().is_mps(), "attn_causal: q must be an MPS tensor");
  TORCH_CHECK(q_in.scalar_type() == at::kBFloat16, "attn_causal: q must be bfloat16");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  TORCH_CHECK(q.dim() == 4, "attn_causal: expects (B,H,N,D)");
  const int B = q.size(0), H = q.size(1);
  const unsigned N = static_cast<unsigned>(q.size(2));
  const int D = q.size(3);
  TORCH_CHECK(D == 64 || D == 128, "attn_causal: D must be 64 or 128");
  TORCH_CHECK(N % 8 == 0, "attn_causal: N must be a multiple of 8");
  auto out = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) { tk::launch_attn_causal(e, q, k, v, out, N, H, B, D); });
  return out;
}

static at::Tensor flux_gelu_mps(const at::Tensor& x_in, const at::Tensor& w_in,
                                const at::Tensor& bias_in) {
  TORCH_CHECK(x_in.device().is_mps(), "flux_gelu: x must be an MPS tensor");
  auto x = x_in.contiguous(), w = w_in.contiguous(), bias = bias_in.contiguous();
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(0), "flux_gelu: (N,K)@(K,M)");
  const int N = x.size(0), K = x.size(1), M = w.size(1);
  TORCH_CHECK(N % 32 == 0 && M % 32 == 0 && K % 16 == 0, "flux_gelu: N%32,M%32,K%16");
  auto out = at::empty({N, M}, x.options());
  const std::string tn = tk_type_name(x);
  tk_encode([&](TorchEncoder& e) { tk::launch_flux_gelu(e, out, x, w, bias, N, K, M, tn); });
  return out;
}

static at::Tensor flux_gate_mps(const at::Tensor& x_in, const at::Tensor& w_in,
                                const at::Tensor& bias_in, const at::Tensor& gate_in,
                                const at::Tensor& res_in) {
  TORCH_CHECK(x_in.device().is_mps(), "flux_gate: x must be an MPS tensor");
  auto x = x_in.contiguous(), w = w_in.contiguous(), bias = bias_in.contiguous();
  auto gate = gate_in.contiguous(), res = res_in.contiguous();
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(0), "flux_gate: (N,K)@(K,M)");
  const int N = x.size(0), K = x.size(1), M = w.size(1);
  TORCH_CHECK(N % 32 == 0 && M % 32 == 0 && K % 16 == 0, "flux_gate: N%32,M%32,K%16");
  auto out = at::empty({N, M}, x.options());
  const std::string tn = tk_type_name(x);
  tk_encode([&](TorchEncoder& e) { tk::launch_flux_gate(e, out, x, w, bias, gate, res, N, K, M, tn); });
  return out;
}

static at::Tensor gemm_staged_mps(const at::Tensor& x_in, const at::Tensor& y_in) {
  TORCH_CHECK(x_in.device().is_mps(), "gemm_staged: x must be an MPS tensor");
  auto x = x_in.contiguous(), y = y_in.contiguous();
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && x.size(1) == y.size(0), "gemm_staged: (N,K)@(K,M)");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
              "gemm_staged: dtype float32 or bfloat16");
  const int N = x.size(0), K = x.size(1), M = y.size(1);
  TORCH_CHECK(N % 32 == 0 && M % 32 == 0 && K % 16 == 0, "gemm_staged: N%32,M%32,K%16");
  auto out = at::empty({N, M}, x.options());
  const std::string tn = tk_type_name(x);
  tk_encode([&](TorchEncoder& e) { tk::launch_gemm_staged(e, out, x, y, N, K, M, tn); });
  return out;
}

static at::Tensor attn_multiwarp_mps(const at::Tensor& q_in, const at::Tensor& k_in,
                                     const at::Tensor& v_in) {
  TORCH_CHECK(q_in.device().is_mps(), "attn_multiwarp: q must be an MPS tensor");
  TORCH_CHECK(q_in.scalar_type() == at::kBFloat16, "attn_multiwarp: q must be bfloat16");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  TORCH_CHECK(q.dim() == 4, "attn_multiwarp: expects (B,H,N,D)");
  const int B = q.size(0), H = q.size(1);
  const unsigned N = static_cast<unsigned>(q.size(2));
  const int D = q.size(3);
  TORCH_CHECK(D == 64 || D == 128, "attn_multiwarp: D must be 64 or 128");
  TORCH_CHECK(N % 32 == 0, "attn_multiwarp: N must be a multiple of 32");
  auto out = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) { tk::launch_attn_multiwarp(e, q, k, v, out, N, H, B, D); });
  return out;
}

static at::Tensor linear_attn_mps(const at::Tensor& q_in, const at::Tensor& k_in,
                                  const at::Tensor& v_in) {
  TORCH_CHECK(q_in.device().is_mps(), "linear_attn: q must be an MPS tensor");
  TORCH_CHECK(q_in.scalar_type() == at::kBFloat16, "linear_attn: q must be bfloat16");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  TORCH_CHECK(q.dim() == 4, "linear_attn: expects (B,H,N,D)");
  const int B = q.size(0), H = q.size(1);
  const unsigned N = static_cast<unsigned>(q.size(2));
  const int D = q.size(3);
  TORCH_CHECK(D == 64, "linear_attn: D must be 64");
  TORCH_CHECK(N % 8 == 0, "linear_attn: N must be a multiple of 8");
  auto out = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) { tk::launch_linear_attn(e, q, k, v, out, N, H, B, D); });
  return out;
}

static at::Tensor hedgehog_mps(const at::Tensor& q_in, const at::Tensor& k_in,
                               const at::Tensor& v_in) {
  TORCH_CHECK(q_in.device().is_mps(), "hedgehog: q must be an MPS tensor");
  TORCH_CHECK(q_in.scalar_type() == at::kBFloat16, "hedgehog: q must be bfloat16");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  TORCH_CHECK(q.dim() == 4, "hedgehog: expects (B,H,N,D)");
  const int B = q.size(0), H = q.size(1);
  const unsigned N = static_cast<unsigned>(q.size(2));
  const int D = q.size(3);
  TORCH_CHECK(D == 64, "hedgehog: D must be 64");
  TORCH_CHECK(N % 8 == 0, "hedgehog: N must be a multiple of 8");
  auto out = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) { tk::launch_hedgehog(e, q, k, v, out, N, H, B, D); });
  return out;
}

static at::Tensor lin_attn_causal_mps(const at::Tensor& q_in, const at::Tensor& k_in,
                                      const at::Tensor& v_in) {
  TORCH_CHECK(q_in.device().is_mps(), "lin_attn_causal: q must be an MPS tensor");
  TORCH_CHECK(q_in.scalar_type() == at::kBFloat16, "lin_attn_causal: q must be bfloat16");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  TORCH_CHECK(q.dim() == 4, "lin_attn_causal: expects (B,H,N,D)");
  const int B = q.size(0), H = q.size(1);
  const unsigned N = static_cast<unsigned>(q.size(2));
  const int D = q.size(3);
  TORCH_CHECK(D == 64, "lin_attn_causal: D must be 64");
  TORCH_CHECK(N % 8 == 0, "lin_attn_causal: N must be a multiple of 8");
  auto out = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) { tk::launch_lin_attn_causal(e, q, k, v, out, N, H, B, D); });
  return out;
}

static at::Tensor mamba2_mps(const at::Tensor& C_in, const at::Tensor& B_in,
                             const at::Tensor& X_in, const at::Tensor& cl_in) {
  TORCH_CHECK(C_in.device().is_mps(), "mamba2: C must be an MPS tensor");
  TORCH_CHECK(C_in.scalar_type() == at::kBFloat16, "mamba2: C,B,X must be bfloat16");
  TORCH_CHECK(cl_in.scalar_type() == at::kFloat, "mamba2: cumlog must be float32");
  auto C = C_in.contiguous(), B = B_in.contiguous(), X = X_in.contiguous(), cl = cl_in.contiguous();
  TORCH_CHECK(C.dim() == 4, "mamba2: C,B,X expect (B,H,N,D)");
  const int Bsz = C.size(0), H = C.size(1);
  const unsigned N = static_cast<unsigned>(C.size(2));
  const int D = C.size(3);
  TORCH_CHECK(D == 64, "mamba2: D must be 64");
  TORCH_CHECK(N % 8 == 0, "mamba2: N must be a multiple of 8");
  auto out = at::empty_like(C);
  tk_encode([&](TorchEncoder& e) { tk::launch_mamba2(e, C, B, X, cl, out, N, H, Bsz, D); });
  return out;
}

static at::Tensor cmplx_matmul_mps(const at::Tensor& a_in, const at::Tensor& b_in) {
  TORCH_CHECK(a_in.device().is_mps(), "cmplx_matmul: a must be an MPS tensor");
  TORCH_CHECK(a_in.scalar_type() == at::kFloat || a_in.scalar_type() == at::kBFloat16,
              "cmplx_matmul: dtype float32 or bfloat16");
  auto a = a_in.contiguous(), b = b_in.contiguous();
  TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && a.size(0) == 2 && b.size(0) == 2 &&
              a.size(2) == b.size(1), "cmplx_matmul: a (2,N,K), b (2,K,M)");
  const int N = a.size(1), K = a.size(2), M = b.size(2);
  TORCH_CHECK(N % 32 == 0 && M % 32 == 0 && K % 16 == 0, "cmplx_matmul: N%32,M%32,K%16");
  auto out = at::empty({2, N, M}, a.options());
  const std::string tn = tk_type_name(a);
  tk_encode([&](TorchEncoder& e) { tk::launch_cmplx_matmul(e, out, a, b, N, K, M, tn); });
  return out;
}

static at::Tensor fftconv_mps(const at::Tensor& x_in, const at::Tensor& F_in,
                              const at::Tensor& twf_in, const at::Tensor& finv_in,
                              const at::Tensor& twi_in, const at::Tensor& kf_in) {
  TORCH_CHECK(x_in.device().is_mps(), "fftconv: x must be an MPS tensor");
  TORCH_CHECK(x_in.scalar_type() == at::kFloat, "fftconv: inputs must be float32");
  auto x = x_in.contiguous(), F = F_in.contiguous(), twf = twf_in.contiguous();
  auto finv = finv_in.contiguous(), twi = twi_in.contiguous(), kf = kf_in.contiguous();
  TORCH_CHECK(x.dim() == 5 && x.size(0) == 2, "fftconv: x must be (2,B,H,S,S)");
  const int B = x.size(1), H = x.size(2), S = x.size(3);
  TORCH_CHECK(x.size(4) == S && (S == 16 || S == 32), "fftconv: S must be 16 or 32");
  auto out = at::empty({B, H, S, S}, x.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_fftconv(e, out, x, F, twf, finv, twi, kf, B * H, H, S);
  });
  return out;
}

static at::Tensor qgemm_mps(const at::Tensor& wq_in, const at::Tensor& x_in,
                            const std::string& format) {
  TORCH_CHECK(wq_in.device().is_mps(), "qgemm: wq must be an MPS tensor");
  TORCH_CHECK(wq_in.scalar_type() == at::kByte, "qgemm: wq must be uint8 packed blocks");
  TORCH_CHECK(x_in.scalar_type() == at::kHalf, "qgemm: x must be float16");
  auto wq = wq_in.contiguous(), x = x_in.contiguous();
  TORCH_CHECK(wq.dim() == 3 && x.dim() == 2, "qgemm: wq (N,K/bk,bytes), x (K,M)");
  const int block_k = (format == "q4_K") ? 256 : 32;
  const int N = wq.size(0), K = (int)wq.size(1) * block_k, M = x.size(1);
  TORCH_CHECK(x.size(0) == K && N % 32 == 0 && M % 32 == 0, "qgemm: N%32,M%32, x rows==K");
  auto out = at::empty({N, M}, x.options());
  tk_encode([&](TorchEncoder& e) { tk::launch_qgemm(e, out, wq, x, N, K, M, format); });
  return out;
}

static at::Tensor qgemv_mps(const at::Tensor& wq_in, const at::Tensor& x_in,
                            const std::string& format) {
  TORCH_CHECK(wq_in.device().is_mps(), "qgemv: wq must be an MPS tensor");
  TORCH_CHECK(wq_in.scalar_type() == at::kByte, "qgemv: wq must be uint8 packed blocks");
  TORCH_CHECK(x_in.scalar_type() == at::kHalf, "qgemv: x must be float16");
  auto wq = wq_in.contiguous(), x = x_in.contiguous();
  TORCH_CHECK(wq.dim() == 3 && x.dim() == 2 && x.size(1) == 1, "qgemv: wq (N,K/bk,bytes), x (K,1)");
  const int block_k = (format == "q4_K") ? 256 : 32;
  const int N = wq.size(0), K = (int)wq.size(1) * block_k;
  TORCH_CHECK(x.size(0) == K, "qgemv: x rows must equal K");
  auto out = at::empty({N, 1}, x.options());
  tk_encode([&](TorchEncoder& e) { tk::launch_qgemv(e, out, wq, x, N, K, format); });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_set_library", &tk_set_library, "set the metallib path");
  m.def("layernorm", &layernorm_mps, "ThunderMittens LayerNorm (MPS)");
  m.def("add_rt", &add_rt_mps, "ThunderMittens add_rt elementwise add (MPS)");
  m.def("matmul_custom", &matmul_custom_mps, "ThunderMittens matmul_custom GEMM (MPS)");
  m.def("attn_fwd", &attn_fwd_mps, "ThunderMittens attention forward (MPS)");
  m.def("rms_norm", &rms_norm_mps, "ThunderMittens RMSNorm (MPS)");
  m.def("softmax", &softmax_mps, "ThunderMittens softmax (MPS)");
  m.def("rotary", &rotary_mps, "ThunderMittens rotary/RoPE (MPS)");
  m.def("gelu", &gelu_mps, "ThunderMittens GELU (MPS)");
  m.def("attn_causal", &attn_causal_mps, "ThunderMittens causal attention (MPS)");
  m.def("flux_gelu", &flux_gelu_mps, "ThunderMittens fused GEMM+GELU (MPS)");
  m.def("flux_gate", &flux_gate_mps, "ThunderMittens fused GEMM+gate+residual (MPS)");
  m.def("gemm_staged", &gemm_staged_mps, "ThunderMittens staged multi-simdgroup GEMM (MPS)");
  m.def("attn_multiwarp", &attn_multiwarp_mps, "ThunderMittens multi-warp attention (MPS)");
  m.def("linear_attn", &linear_attn_mps, "ThunderMittens non-causal linear attention (MPS)");
  m.def("hedgehog", &hedgehog_mps, "ThunderMittens hedgehog linear attention (MPS)");
  m.def("lin_attn_causal", &lin_attn_causal_mps, "ThunderMittens causal linear attention (MPS)");
  m.def("mamba2", &mamba2_mps, "ThunderMittens Mamba-2 / SSD forward (MPS)");
  m.def("cmplx_matmul", &cmplx_matmul_mps, "ThunderMittens complex GEMM (MPS)");
  m.def("fftconv", &fftconv_mps, "ThunderMittens Monarch FFT convolution (MPS)");
  m.def("qgemm", &qgemm_mps, "ThunderMittens quantized GEMM (MPS)");
  m.def("qgemv", &qgemv_mps, "ThunderMittens quantized GEMV decode (MPS)");
}
