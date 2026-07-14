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

#include <cmath>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "tk_launch.h"

// The MTLBuffer backing an MPS tensor's storage (documented PyTorch pattern).
static inline id<MTLBuffer> mtl_buffer(const at::Tensor& t) {
  return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
}
static inline NSUInteger byte_offset(const at::Tensor& t) {
  return static_cast<NSUInteger>(t.storage_offset()) * t.element_size();
}
static bool tk_is_float_dtype(const at::Tensor& t);   // defined below

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
  TORCH_CHECK(x_in.device().is_mps() && w_in.device().is_mps() &&
              b_in.device().is_mps(), "layernorm: inputs must be MPS tensors");
  TORCH_CHECK(x_in.scalar_type() == at::kBFloat16 &&
              w_in.scalar_type() == at::kBFloat16 &&
              b_in.scalar_type() == at::kBFloat16 && x_in.dim() > 0,
              "layernorm: inputs must be bfloat16 and x must have at least one dimension");
  const int D = x_in.size(-1);
  TORCH_CHECK(D > 0 && D % 4 == 0 && w_in.dim() == 1 && b_in.dim() == 1 &&
              w_in.size(0) == D && b_in.size(0) == D,
              "layernorm: weight/bias must match a positive last dim divisible by 4");
  auto x = x_in.contiguous(), w = w_in.contiguous(), b = b_in.contiguous();
  const uint32_t M = static_cast<uint32_t>(x.numel() / D);
  auto out = at::empty_like(x);
  const float eps_f = static_cast<float>(eps);
  tk_encode([&](TorchEncoder& e) {
    if (D == 256 || D == 512 || D == 768 || D == 1024) {
      tk::launch_layernorm(e, x, w, b, out, M, D, eps_f);
    } else {
      tk::launch_layernorm_dyn(e, x, w, b, out, M, D, eps_f);
    }
  });
  return out;
}

static at::Tensor layernorm_bwd_dx_mps(const at::Tensor& x_in, const at::Tensor& w_in,
                                       const at::Tensor& dy_in, const at::Tensor& mean_in,
                                       const at::Tensor& rstd_in) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in),
              "layernorm_bwd_dx: x must be a float MPS tensor");
  auto x = x_in.contiguous();
  auto w = w_in.to(x.scalar_type()).contiguous();
  auto dy = dy_in.to(x.scalar_type()).contiguous();
  auto mean = mean_in.to(at::kFloat).contiguous();
  auto rstd = rstd_in.to(at::kFloat).contiguous();
  const int D = x.size(-1);
  const int rows = static_cast<int>(x.numel() / D);
  auto dx = at::empty_like(x);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_layernorm_bwd_dx(e, x, w, dy, mean, rstd, dx, rows, D, tk_type_name(x));
  });
  return dx;
}

static std::vector<at::Tensor> layernorm_bwd_fused_mps(const at::Tensor& x_in, const at::Tensor& w_in,
                                                       const at::Tensor& dy_in, double eps) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in),
              "layernorm_bwd_fused: x must be a float MPS tensor");
  auto x = x_in.contiguous();
  auto w = w_in.to(x.scalar_type()).contiguous();
  auto dy = dy_in.to(x.scalar_type()).contiguous();
  const int D = x.size(-1);
  const int rows = static_cast<int>(x.numel() / D);
  auto dx = at::empty_like(x);
  auto f32 = x.options().dtype(at::kFloat);
  auto dweight = at::zeros({D}, f32);                   // atomic accumulators (pre-zeroed)
  auto dbias = at::zeros({D}, f32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_layernorm_bwd_fused(e, x, w, dy, dx, dweight, dbias, rows, D,
                                   static_cast<float>(eps), tk_type_name(x));
  });
  return {dx, dweight, dbias};
}

static std::vector<at::Tensor> rms_norm_bwd_fused_mps(const at::Tensor& x_in, const at::Tensor& w_in,
                                                      const at::Tensor& dy_in, double eps) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in),
              "rms_norm_bwd_fused: x must be a float MPS tensor");
  auto x = x_in.contiguous();
  auto w = w_in.to(x.scalar_type()).contiguous();
  auto dy = dy_in.to(x.scalar_type()).contiguous();
  const int D = x.size(-1);
  const int rows = static_cast<int>(x.numel() / D);
  auto dx = at::empty_like(x);
  auto dweight = at::zeros({D}, x.options().dtype(at::kFloat));   // atomic accumulator (pre-zeroed)
  tk_encode([&](TorchEncoder& e) {
    tk::launch_rms_norm_bwd_fused(e, x, w, dy, dx, dweight, rows, D, static_cast<float>(eps),
                                  tk_type_name(x));
  });
  return {dx, dweight};
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
                               const at::Tensor& v_in, double softcap,
                               const c10::optional<at::Tensor>& sinks_in) {
  TORCH_CHECK(q_in.device().is_mps(), "attn_fwd: q must be an MPS tensor");
  TORCH_CHECK(q_in.scalar_type() == at::kBFloat16, "attn_fwd: q must be bfloat16");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  TORCH_CHECK(q.dim() == 4, "attn_fwd: expects (B,H,N,D)");
  const int B = q.size(0), H = q.size(1);
  const unsigned N = static_cast<unsigned>(q.size(2));
  const int D = q.size(3);
  TORCH_CHECK(D == 64 || D == 128, "attn_fwd: D must be 64 or 128");
  TORCH_CHECK(N % 8 == 0, "attn_fwd: N must be a multiple of 8");
  const bool has_sink = sinks_in.has_value();
  at::Tensor sinks = q;   // placeholder binding when absent (kernel never reads it)
  if (has_sink) {
    TORCH_CHECK(sinks_in->dim() == 1 && sinks_in->size(0) == H, "attn_fwd: sinks must be (H,)");
    sinks = sinks_in->to(at::kFloat).contiguous();
  }
  auto out = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_attn_fwd(e, q, k, v, out, N, H, B, D, static_cast<float>(softcap), sinks,
                        has_sink ? 1 : 0);
  });
  return out;
}

static at::Tensor rms_norm_mps(const at::Tensor& x_in, const at::Tensor& w_in, double eps) {
  TORCH_CHECK(x_in.device().is_mps(), "rms_norm: x must be an MPS tensor");
  TORCH_CHECK(x_in.scalar_type() == at::kBFloat16, "rms_norm: x must be bfloat16");
  auto x = x_in.contiguous(), w = w_in.contiguous();
  const int D = x.size(-1);
  TORCH_CHECK(D % 4 == 0, "rms_norm: last dim must be a multiple of 4");
  const uint32_t M = static_cast<uint32_t>(x.numel() / D);
  auto out = at::empty_like(x);
  const float eps_f = static_cast<float>(eps);
  tk_encode([&](TorchEncoder& e) {
    if (D == 256 || D == 512 || D == 768 || D == 1024) {
      tk::launch_rms_norm(e, x, w, out, M, D, eps_f);
    } else {
      tk::launch_rms_norm_dyn(e, x, w, out, M, D, eps_f);
    }
  });
  return out;
}

static at::Tensor rms_norm_bwd_dx_mps(const at::Tensor& x_in, const at::Tensor& w_in,
                                      const at::Tensor& dy_in, const at::Tensor& rstd_in) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in),
              "rms_norm_bwd_dx: x must be a float MPS tensor");
  auto x = x_in.contiguous();
  auto w = w_in.to(x.scalar_type()).contiguous();
  auto dy = dy_in.to(x.scalar_type()).contiguous();
  auto rstd = rstd_in.to(at::kFloat).contiguous();
  const int D = x.size(-1);
  const int rows = static_cast<int>(x.numel() / D);
  auto dx = at::empty_like(x);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_rms_norm_bwd_dx(e, x, w, dy, rstd, dx, rows, D, tk_type_name(x));
  });
  return dx;
}

// Fused residual-add + RMSNorm. Returns (out, x+residual).
static std::tuple<at::Tensor, at::Tensor> rms_norm_add_mps(
    const at::Tensor& x_in, const at::Tensor& r_in, const at::Tensor& w_in, double eps) {
  TORCH_CHECK(x_in.device().is_mps(), "rms_norm_add: x must be an MPS tensor");
  TORCH_CHECK(x_in.scalar_type() == at::kBFloat16, "rms_norm_add: x must be bfloat16");
  TORCH_CHECK(r_in.sizes() == x_in.sizes(), "rms_norm_add: residual must match x shape");
  auto x = x_in.contiguous(), r = r_in.contiguous(), w = w_in.contiguous();
  const int D = x.size(-1);
  TORCH_CHECK(D == 256 || D == 512 || D == 768 || D == 1024,
              "rms_norm_add: last dim must be 256/512/768/1024");
  const uint32_t M = static_cast<uint32_t>(x.numel() / D);
  auto out = at::empty_like(x);
  auto res_out = at::empty_like(x);
  const float eps_f = static_cast<float>(eps);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_rms_norm_add(e, x, r, w, out, res_out, M, D, eps_f);
  });
  return {out, res_out};
}

// Fused residual-add + LayerNorm. Returns (out, x+residual).
static std::tuple<at::Tensor, at::Tensor> layernorm_add_mps(
    const at::Tensor& x_in, const at::Tensor& r_in, const at::Tensor& w_in,
    const at::Tensor& b_in, double eps) {
  TORCH_CHECK(x_in.device().is_mps(), "layernorm_add: x must be an MPS tensor");
  TORCH_CHECK(x_in.scalar_type() == at::kBFloat16, "layernorm_add: x must be bfloat16");
  TORCH_CHECK(r_in.sizes() == x_in.sizes(), "layernorm_add: residual must match x shape");
  auto x = x_in.contiguous(), r = r_in.contiguous(), w = w_in.contiguous(), b = b_in.contiguous();
  const int D = x.size(-1);
  TORCH_CHECK(D == 256 || D == 512 || D == 768 || D == 1024,
              "layernorm_add: last dim must be 256/512/768/1024");
  const uint32_t M = static_cast<uint32_t>(x.numel() / D);
  auto out = at::empty_like(x);
  auto res_out = at::empty_like(x);
  const float eps_f = static_cast<float>(eps);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_layernorm_add(e, x, r, w, b, out, res_out, M, D, eps_f);
  });
  return {out, res_out};
}

// fp8 norm epilogues (MPS). Static returns (codes, res_out); dynamic returns (codes, res_out, scale).
static void anfp8_check(const at::Tensor& x, const at::Tensor& r, int& D, uint32_t& M) {
  TORCH_CHECK(x.device().is_mps() && x.scalar_type() == at::kBFloat16, "fp8 norm: x must be bf16 MPS");
  TORCH_CHECK(r.sizes() == x.sizes(), "fp8 norm: residual must match x");
  D = x.size(-1);
  TORCH_CHECK(D == 256 || D == 512 || D == 768 || D == 1024, "fp8 norm: D in {256,512,768,1024}");
  M = static_cast<uint32_t>(x.numel() / D);
}
static std::tuple<at::Tensor, at::Tensor> rms_norm_add_fp8_mps(
    const at::Tensor& x_in, const at::Tensor& r_in, const at::Tensor& w_in, double eps, double scale) {
  int D; uint32_t M; anfp8_check(x_in, r_in, D, M);
  auto x = x_in.contiguous(), r = r_in.contiguous(), w = w_in.contiguous();
  auto codes = at::empty(x.sizes(), x.options().dtype(at::kByte));
  auto res_out = at::empty_like(x);
  const float inv = scale > 0.0 ? 1.0f / static_cast<float>(scale) : 0.0f;
  tk_encode([&](TorchEncoder& e) {
    tk::launch_rms_norm_add_fp8(e, x, r, w, codes, res_out, M, D, static_cast<float>(eps), inv);
  });
  return {codes, res_out};
}
static std::tuple<at::Tensor, at::Tensor, at::Tensor> rms_norm_add_fp8_dyn_mps(
    const at::Tensor& x_in, const at::Tensor& r_in, const at::Tensor& w_in, double eps) {
  int D; uint32_t M; anfp8_check(x_in, r_in, D, M);
  auto x = x_in.contiguous(), r = r_in.contiguous(), w = w_in.contiguous();
  auto codes = at::empty(x.sizes(), x.options().dtype(at::kByte));
  auto res_out = at::empty_like(x);
  std::vector<int64_t> sshape(x.sizes().begin(), x.sizes().end() - 1);
  if (sshape.empty()) sshape.push_back(1);
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_rms_norm_add_fp8_dyn(e, x, r, w, codes, res_out, scale, M, D, static_cast<float>(eps));
  });
  return {codes, res_out, scale};
}
static std::tuple<at::Tensor, at::Tensor, at::Tensor> rms_norm_add_int8_dyn_mps(
    const at::Tensor& x_in, const at::Tensor& r_in, const at::Tensor& w_in, double eps) {
  int D; uint32_t M; anfp8_check(x_in, r_in, D, M);
  auto x = x_in.contiguous(), r = r_in.contiguous(), w = w_in.contiguous();
  auto codes = at::empty(x.sizes(), x.options().dtype(at::kChar));
  auto res_out = at::empty_like(x);
  std::vector<int64_t> sshape(x.sizes().begin(), x.sizes().end() - 1);
  if (sshape.empty()) sshape.push_back(1);
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_rms_norm_add_int8_dyn(e, x, r, w, codes, res_out, scale, M, D, static_cast<float>(eps));
  });
  return {codes, res_out, scale};
}

// fused gated-activation -> dynamic quant epilogues (buffer order matches tk.glu(x, gate)).
static std::tuple<at::Tensor, at::Tensor> silu_mul_quant_fp8_mps(
    const at::Tensor& x_in, const at::Tensor& gate_in, int64_t mode, double alpha, double limit) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in) && x_in.sizes() == gate_in.sizes(),
              "silu_mul_quant_fp8: x/gate must be same-shape float MPS tensors");
  auto x = x_in.contiguous(), gate = gate_in.to(x_in.scalar_type()).contiguous();
  const int D = x.size(-1);
  const int rows = x.numel() / D;
  auto codes = at::empty(x.sizes(), x.options().dtype(at::kByte));
  std::vector<int64_t> sshape(x.sizes().begin(), x.sizes().end() - 1);
  if (sshape.empty()) sshape.push_back(1);
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_silu_mul_quant_fp8(e, x, gate, codes, scale, rows, D, (int)mode,
                                  (float)alpha, (float)limit, tk_type_name(x));
  });
  return {codes, scale};
}

static std::tuple<at::Tensor, at::Tensor> silu_mul_quant_int8_mps(
    const at::Tensor& x_in, const at::Tensor& gate_in, int64_t mode, double alpha, double limit) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in) && x_in.sizes() == gate_in.sizes(),
              "silu_mul_quant_int8: x/gate must be same-shape float MPS tensors");
  auto x = x_in.contiguous(), gate = gate_in.to(x_in.scalar_type()).contiguous();
  const int D = x.size(-1);
  const int rows = x.numel() / D;
  auto codes = at::empty(x.sizes(), x.options().dtype(at::kChar));
  std::vector<int64_t> sshape(x.sizes().begin(), x.sizes().end() - 1);
  if (sshape.empty()) sshape.push_back(1);
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_silu_mul_quant_int8(e, x, gate, codes, scale, rows, D, (int)mode,
                                   (float)alpha, (float)limit, tk_type_name(x));
  });
  return {codes, scale};
}

static std::tuple<at::Tensor, at::Tensor> silu_mul_quant_fp8_group_mps(
    const at::Tensor& x_in, const at::Tensor& gate_in, int64_t group_size, bool ue8m0,
    int64_t mode, double alpha, double limit) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in) && x_in.sizes() == gate_in.sizes(),
              "silu_mul_quant_fp8_group: x/gate must be same-shape float MPS tensors");
  auto x = x_in.contiguous(), gate = gate_in.to(x_in.scalar_type()).contiguous();
  const int D = x.size(-1);
  TORCH_CHECK(group_size > 0 && group_size % 4 == 0 && D % group_size == 0,
              "silu_mul_quant_fp8_group: bad group_size");
  const int rows = x.numel() / D;
  auto codes = at::empty(x.sizes(), x.options().dtype(at::kByte));
  auto sshape = x.sizes().vec(); sshape.back() = D / group_size;
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_silu_mul_quant_fp8_group(e, x, gate, codes, scale, rows, D, (int)group_size,
                                        ue8m0 ? 1 : 0, (int)mode, (float)alpha, (float)limit,
                                        tk_type_name(x));
  });
  return {codes, scale};
}

static std::tuple<at::Tensor, at::Tensor> layernorm_add_fp8_mps(
    const at::Tensor& x_in, const at::Tensor& r_in, const at::Tensor& w_in, const at::Tensor& b_in,
    double eps, double scale) {
  int D; uint32_t M; anfp8_check(x_in, r_in, D, M);
  auto x = x_in.contiguous(), r = r_in.contiguous(), w = w_in.contiguous(), b = b_in.contiguous();
  auto codes = at::empty(x.sizes(), x.options().dtype(at::kByte));
  auto res_out = at::empty_like(x);
  const float inv = scale > 0.0 ? 1.0f / static_cast<float>(scale) : 0.0f;
  tk_encode([&](TorchEncoder& e) {
    tk::launch_layernorm_add_fp8(e, x, r, w, b, codes, res_out, M, D, static_cast<float>(eps), inv);
  });
  return {codes, res_out};
}
static std::tuple<at::Tensor, at::Tensor, at::Tensor> layernorm_add_fp8_dyn_mps(
    const at::Tensor& x_in, const at::Tensor& r_in, const at::Tensor& w_in, const at::Tensor& b_in,
    double eps) {
  int D; uint32_t M; anfp8_check(x_in, r_in, D, M);
  auto x = x_in.contiguous(), r = r_in.contiguous(), w = w_in.contiguous(), b = b_in.contiguous();
  auto codes = at::empty(x.sizes(), x.options().dtype(at::kByte));
  auto res_out = at::empty_like(x);
  std::vector<int64_t> sshape(x.sizes().begin(), x.sizes().end() - 1);
  if (sshape.empty()) sshape.push_back(1);
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_layernorm_add_fp8_dyn(e, x, r, w, b, codes, res_out, scale, M, D, static_cast<float>(eps));
  });
  return {codes, res_out, scale};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> layernorm_add_int8_dyn_mps(
    const at::Tensor& x_in, const at::Tensor& r_in, const at::Tensor& w_in, const at::Tensor& b_in,
    double eps) {
  int D; uint32_t M; anfp8_check(x_in, r_in, D, M);
  auto x = x_in.contiguous(), r = r_in.contiguous(), w = w_in.contiguous(), b = b_in.contiguous();
  auto codes = at::empty(x.sizes(), x.options().dtype(at::kChar));
  auto res_out = at::empty_like(x);
  std::vector<int64_t> sshape(x.sizes().begin(), x.sizes().end() - 1);
  if (sshape.empty()) sshape.push_back(1);
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_layernorm_add_int8_dyn(e, x, r, w, b, codes, res_out, scale, M, D, static_cast<float>(eps));
  });
  return {codes, res_out, scale};
}

// per-block (per-128-group) dynamic norm-quant. int8=true -> char codes, else uchar (fp8).
static std::tuple<at::Tensor, at::Tensor, at::Tensor> rms_norm_add_per_block_mps(
    const at::Tensor& x_in, const at::Tensor& r_in, const at::Tensor& w_in, double eps,
    bool int8, bool ue8m0) {
  int D; uint32_t M; anfp8_check(x_in, r_in, D, M);
  TORCH_CHECK(D % 128 == 0, "rms_norm_add_per_block: D must be a multiple of 128");
  auto x = x_in.contiguous(), r = r_in.contiguous(), w = w_in.contiguous();
  auto codes = at::empty(x.sizes(), x.options().dtype(int8 ? at::kChar : at::kByte));
  auto res_out = at::empty_like(x);
  std::vector<int64_t> sshape(x.sizes().begin(), x.sizes().end());
  sshape.back() = D / 128;
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_rms_norm_add_per_block(e, x, r, w, codes, res_out, scale, M, D,
                                      static_cast<float>(eps), int8, ue8m0 ? 1 : 0);
  });
  return {codes, res_out, scale};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> layernorm_add_per_block_mps(
    const at::Tensor& x_in, const at::Tensor& r_in, const at::Tensor& w_in, const at::Tensor& b_in,
    double eps, bool int8, bool ue8m0) {
  int D; uint32_t M; anfp8_check(x_in, r_in, D, M);
  TORCH_CHECK(D % 128 == 0, "layernorm_add_per_block: D must be a multiple of 128");
  auto x = x_in.contiguous(), r = r_in.contiguous(), w = w_in.contiguous(), b = b_in.contiguous();
  auto codes = at::empty(x.sizes(), x.options().dtype(int8 ? at::kChar : at::kByte));
  auto res_out = at::empty_like(x);
  std::vector<int64_t> sshape(x.sizes().begin(), x.sizes().end());
  sshape.back() = D / 128;
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_layernorm_add_per_block(e, x, r, w, b, codes, res_out, scale, M, D,
                                       static_cast<float>(eps), int8, ue8m0 ? 1 : 0);
  });
  return {codes, res_out, scale};
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
                             const at::Tensor& sin_in, bool interleaved) {
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
  tk_encode([&](TorchEncoder& e) { tk::launch_rotary(e, x, cos, sin, out, M, N, D, interleaved); });
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

static at::Tensor gelu_bwd_mps(const at::Tensor& x_in, const at::Tensor& dy_in) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in),
              "gelu_bwd: x must be a float MPS tensor");
  auto x = x_in.contiguous();
  auto dy = dy_in.to(x.scalar_type()).contiguous();
  const int n = static_cast<int>(x.numel());
  auto dx = at::empty_like(x);
  tk_encode([&](TorchEncoder& e) { tk::launch_gelu_bwd(e, x, dy, dx, n, tk_type_name(x)); });
  return dx;
}

static at::Tensor dropout_mps(const at::Tensor& x_in, double p, int64_t seed, bool bwd) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in),
              "dropout: x must be a float MPS tensor");
  TORCH_CHECK(p >= 0.0 && p < 1.0, "dropout: p must be in [0, 1)");
  auto x = x_in.contiguous();
  auto out = at::empty_like(x);
  const uint32_t n = static_cast<uint32_t>(x.numel());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_dropout(e, x, out, static_cast<uint32_t>(seed), static_cast<float>(p), n, bwd,
                       tk_type_name(x));
  });
  return out;
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> adamw_mps(
    const at::Tensor& param_in, const at::Tensor& grad_in, const at::Tensor& m_in,
    const at::Tensor& v_in, double lr, double beta1, double beta2, double eps,
    double weight_decay, int64_t step) {
  TORCH_CHECK(param_in.device().is_mps() && tk_is_float_dtype(param_in),
              "adamw: param must be a float MPS tensor");
  TORCH_CHECK(m_in.scalar_type() == at::kFloat && v_in.scalar_type() == at::kFloat,
              "adamw: moment state m, v must be float32");
  TORCH_CHECK(param_in.sizes() == grad_in.sizes() && param_in.sizes() == m_in.sizes() &&
              param_in.sizes() == v_in.sizes(), "adamw: param, grad, m, v shapes must match");
  TORCH_CHECK(step >= 1, "adamw: step (t) must be >= 1");
  auto param = param_in.contiguous();
  auto grad = grad_in.to(param.scalar_type()).contiguous();
  auto m = m_in.contiguous(), v = v_in.contiguous();
  auto p_out = at::empty_like(param), m_out = at::empty_like(m), v_out = at::empty_like(v);
  const float bc1 = 1.0f - std::pow(static_cast<float>(beta1), static_cast<float>(step));
  const float bc2 = 1.0f - std::pow(static_cast<float>(beta2), static_cast<float>(step));
  const uint32_t n = static_cast<uint32_t>(param.numel());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_adamw(e, param, grad, m, v, p_out, m_out, v_out, static_cast<float>(lr),
                     static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(eps),
                     static_cast<float>(weight_decay), bc1, bc2, n, tk_type_name(param));
  });
  return {p_out, m_out, v_out};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> adamw_masked_mps(
    const at::Tensor& param_in, const at::Tensor& grad_in, const at::Tensor& m_in,
    const at::Tensor& v_in, double lr, double beta1, double beta2, double eps,
    double weight_decay, int64_t step, const at::Tensor& mask_in, int64_t seg_size,
    int64_t mask_mode) {
  TORCH_CHECK(param_in.device().is_mps() && tk_is_float_dtype(param_in),
              "adamw_masked: param must be a float MPS tensor");
  TORCH_CHECK(m_in.scalar_type() == at::kFloat && v_in.scalar_type() == at::kFloat,
              "adamw_masked: moment state m, v must be float32");
  TORCH_CHECK(param_in.sizes() == grad_in.sizes() && param_in.sizes() == m_in.sizes() &&
              param_in.sizes() == v_in.sizes(), "adamw_masked: shapes must match");
  TORCH_CHECK(step >= 1, "adamw_masked: step (t) must be >= 1");
  TORCH_CHECK(mask_in.scalar_type() == at::kByte, "adamw_masked: mask must be uint8");
  TORCH_CHECK(seg_size >= 1, "adamw_masked: seg_size must be >= 1");
  TORCH_CHECK(mask_mode == 0 || mask_mode == 1, "adamw_masked: mask_mode must be 0 or 1");
  auto param = param_in.contiguous();
  auto grad = grad_in.to(param.scalar_type()).contiguous();
  auto m = m_in.contiguous(), v = v_in.contiguous(), mask = mask_in.contiguous();
  const uint32_t n = static_cast<uint32_t>(param.numel());
  TORCH_CHECK((int64_t)mask.numel() * seg_size >= (int64_t)n,
              "adamw_masked: mask covers fewer than numel/seg_size segments");
  auto p_out = at::empty_like(param), m_out = at::empty_like(m), v_out = at::empty_like(v);
  const float bc1 = 1.0f - std::pow(static_cast<float>(beta1), static_cast<float>(step));
  const float bc2 = 1.0f - std::pow(static_cast<float>(beta2), static_cast<float>(step));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_adamw_masked(e, param, grad, m, v, p_out, m_out, v_out, static_cast<float>(lr),
                            static_cast<float>(beta1), static_cast<float>(beta2),
                            static_cast<float>(eps), static_cast<float>(weight_decay), bc1, bc2,
                            n, mask, static_cast<uint32_t>(seg_size),
                            static_cast<int>(mask_mode), tk_type_name(param));
  });
  return {p_out, m_out, v_out};
}

static at::Tensor embedding_lookup_mps(const at::Tensor& token_ids_in, const at::Tensor& table_in,
                                       const at::Tensor& pos_table_in, double scale) {
  TORCH_CHECK(token_ids_in.device().is_mps() && token_ids_in.dim() == 1,
              "embedding_lookup: token_ids must be a 1-D MPS tensor");
  TORCH_CHECK(table_in.dim() == 2 && tk_is_float_dtype(table_in),
              "embedding_lookup: table must be (vocab, D) float");
  auto token_ids = token_ids_in.to(at::kInt).contiguous();
  auto table = table_in.contiguous();
  const int n_tok = token_ids.size(0), vocab = table.size(0), D = table.size(1);
  const bool use_pos = pos_table_in.numel() > 1;
  auto pos_table = use_pos ? pos_table_in.to(table.scalar_type()).contiguous()
                           : at::zeros({1}, table.options());
  if (use_pos) {
    TORCH_CHECK(pos_table.dim() == 2 && pos_table.size(0) == n_tok && pos_table.size(1) == D,
                "embedding_lookup: pos_table must be (num_tok, D)");
  }
  auto out = at::empty({n_tok, D}, table.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_embedding_lookup(e, token_ids, table, pos_table, out, D, vocab, n_tok,
                                static_cast<float>(scale), use_pos ? 1 : 0, tk_type_name(table));
  });
  return out;
}

static at::Tensor embedding_backward_mps(const at::Tensor& token_ids_in, const at::Tensor& dY_in,
                                         int64_t vocab, double scale) {
  TORCH_CHECK(token_ids_in.device().is_mps() && token_ids_in.dim() == 1,
              "embedding_backward: token_ids must be a 1-D MPS tensor");
  TORCH_CHECK(dY_in.dim() == 2 && tk_is_float_dtype(dY_in),
              "embedding_backward: dY must be (num_tok, D) float");
  auto token_ids = token_ids_in.to(at::kInt).contiguous();
  auto dY = dY_in.contiguous();
  const int n_tok = token_ids.size(0), D = dY.size(1);
  TORCH_CHECK(dY.size(0) == n_tok, "embedding_backward: dY num_tok must match token_ids");
  // at::zeros gives the zeroed fp32 accumulator (no separate zero kernel needed on torch).
  auto dtable = at::zeros({(long)vocab, D}, dY.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_embedding_backward(e, token_ids, dY, dtable, D, (int)vocab, n_tok,
                                  static_cast<float>(scale), tk_type_name(dY));
  });
  return dtable;
}

static at::Tensor embedding_backward_sorted_mps(const at::Tensor& sorted_ids_in,
    const at::Tensor& perm_in, const at::Tensor& dY_in, int64_t vocab, double scale) {
  TORCH_CHECK(sorted_ids_in.device().is_mps() && sorted_ids_in.dim() == 1 && perm_in.dim() == 1,
              "embedding_backward_sorted: sorted_ids / perm must be 1-D MPS tensors");
  TORCH_CHECK(dY_in.dim() == 2 && tk_is_float_dtype(dY_in),
              "embedding_backward_sorted: dY must be (num_tok, D) float");
  auto sorted_ids = sorted_ids_in.to(at::kInt).contiguous();
  auto perm = perm_in.to(at::kInt).contiguous();
  auto dY = dY_in.contiguous();
  const int n_tok = sorted_ids.size(0), D = dY.size(1);
  TORCH_CHECK(dY.size(0) == n_tok && perm.size(0) == n_tok,
              "embedding_backward_sorted: num_tok mismatch");
  auto dtable = at::zeros({(long)vocab, D}, dY.options().dtype(at::kFloat));  // absent ids stay 0
  tk_encode([&](TorchEncoder& e) {
    tk::launch_embedding_backward_sorted(e, sorted_ids, perm, dY, dtable, D, (int)vocab, n_tok,
                                         static_cast<float>(scale), tk_type_name(dY));
  });
  return dtable;
}

static at::Tensor build_multimodal_src_mps(const at::Tensor& so_in, const at::Tensor& sl_in,
                                           const at::Tensor& ms_in, int64_t num_tok) {
  TORCH_CHECK(so_in.device().is_mps() && so_in.dim() == 1,
              "build_multimodal_src: span_offsets must be (num_spans,) MPS");
  auto so = so_in.to(at::kInt).contiguous();
  auto sl = sl_in.to(at::kInt).contiguous();
  auto ms = ms_in.to(at::kInt).contiguous();
  const int num_spans = so.size(0);
  auto src = at::empty({(long)num_tok}, so.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_build_multimodal_src(e, so, sl, ms, src, num_spans, static_cast<int>(num_tok));
  });
  return src;
}

static at::Tensor merge_multimodal_spans_mps(const at::Tensor& text_in, const at::Tensor& modal_in,
                                             const at::Tensor& src_in) {
  TORCH_CHECK(text_in.device().is_mps() && text_in.dim() == 2 && tk_is_float_dtype(text_in),
              "merge_multimodal_spans: text must be (num_tok, D) float MPS");
  TORCH_CHECK(modal_in.dim() == 2 && modal_in.size(1) == text_in.size(1) &&
                  modal_in.scalar_type() == text_in.scalar_type(),
              "merge_multimodal_spans: modal must be (num_modal, D), same D/dtype");
  TORCH_CHECK(src_in.dim() == 1 && src_in.size(0) == text_in.size(0),
              "merge_multimodal_spans: src must be (num_tok,)");
  auto text = text_in.contiguous();
  auto modal = modal_in.contiguous();
  auto src = src_in.to(at::kInt).contiguous();
  const int n_tok = text.size(0), D = text.size(1), n_modal = modal.size(0);
  auto out = at::empty_like(text);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_merge_multimodal_spans(e, text, modal, src, out, D, n_tok, n_modal,
                                      tk_type_name(text));
  });
  return out;
}

static bool valid_glu_mode(const std::string& mode) {
  return mode == "reglu" || mode == "geglu" || mode == "swiglu" ||
         mode == "swiglu_oai" || mode == "geglu_erf" ||
         mode == "geglu_quick";
}

static at::Tensor glu_mps(const at::Tensor& x_in, const at::Tensor& gate_in,
                          const std::string& mode, double alpha, double limit) {
  TORCH_CHECK(x_in.device().is_mps(), "glu: x must be an MPS tensor");
  TORCH_CHECK(gate_in.device().is_mps(), "glu: gate must be an MPS tensor");
  TORCH_CHECK(x_in.sizes() == gate_in.sizes(), "glu: x and gate must have the same shape");
  TORCH_CHECK(valid_glu_mode(mode), "glu: unsupported mode ", mode);
  TORCH_CHECK(x_in.scalar_type() == gate_in.scalar_type(), "glu: x and gate dtypes must match");
  TORCH_CHECK(x_in.scalar_type() == at::kFloat || x_in.scalar_type() == at::kHalf ||
              x_in.scalar_type() == at::kBFloat16,
              "glu: dtype must be float32, float16, or bfloat16");
  auto x = x_in.contiguous(), gate = gate_in.contiguous();
  auto out = at::empty_like(x);
  const uint32_t n = static_cast<uint32_t>(out.numel());
  const std::string tn = tk_type_name(x);
  const float alpha_f = static_cast<float>(alpha);
  const float limit_f = static_cast<float>(limit);
  tk_encode([&](TorchEncoder& e) { tk::launch_glu(e, x, gate, out, n, mode, tn, alpha_f, limit_f); });
  return out;
}

static std::tuple<at::Tensor, at::Tensor> glu_bwd_mps(const at::Tensor& x_in,
    const at::Tensor& gate_in, const at::Tensor& dc_in, const std::string& mode,
    double alpha, double limit) {
  TORCH_CHECK(x_in.device().is_mps() && gate_in.device().is_mps() && dc_in.device().is_mps(),
              "glu_backward: x, gate, dc must be MPS tensors");
  TORCH_CHECK(x_in.sizes() == gate_in.sizes() && x_in.sizes() == dc_in.sizes(),
              "glu_backward: x, gate, dc must have the same shape");
  TORCH_CHECK(valid_glu_mode(mode), "glu_backward: unsupported mode ", mode);
  TORCH_CHECK(x_in.scalar_type() == at::kFloat || x_in.scalar_type() == at::kHalf ||
              x_in.scalar_type() == at::kBFloat16,
              "glu_backward: dtype must be float32, float16, or bfloat16");
  auto x = x_in.contiguous(), gate = gate_in.contiguous(), dc = dc_in.to(x_in.scalar_type()).contiguous();
  auto da = at::empty_like(x), db = at::empty_like(x);
  const uint32_t n = static_cast<uint32_t>(x.numel());
  const std::string tn = tk_type_name(x);
  const float alpha_f = static_cast<float>(alpha), limit_f = static_cast<float>(limit);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_glu_bwd(e, x, gate, dc, da, db, n, mode, tn, alpha_f, limit_f);
  });
  return {da, db};
}

static at::Tensor hadamard_mps(const at::Tensor& x_in, double scale) {
  TORCH_CHECK(x_in.device().is_mps(), "hadamard: x must be an MPS tensor");
  TORCH_CHECK(x_in.numel() > 0 && x_in.dim() > 0,
              "hadamard: input must be non-empty with a final axis");
  TORCH_CHECK(x_in.scalar_type() == at::kFloat || x_in.scalar_type() == at::kHalf ||
              x_in.scalar_type() == at::kBFloat16,
              "hadamard: dtype must be float32, float16, or bfloat16");
  auto x = x_in.contiguous();
  const int D = static_cast<int>(x.size(-1));
  TORCH_CHECK(D == 64 || D == 128 || D == 256 || D == 512,
              "hadamard: final axis must be 64, 128, 256, or 512");
  const int rows = static_cast<int>(x.numel() / D);
  auto out = at::empty_like(x);
  const float scale_f = scale > 0.0 ? static_cast<float>(scale)
                                    : 1.0f / std::sqrt(static_cast<float>(D));
  const std::string tn = tk_type_name(x);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_hadamard(e, x, out, rows, D, scale_f, tn);
  });
  return out;
}

static bool tk_is_float_dtype(const at::Tensor& t) {
  return t.scalar_type() == at::kFloat || t.scalar_type() == at::kHalf ||
         t.scalar_type() == at::kBFloat16;
}

static std::tuple<at::Tensor, at::Tensor> kv_cache_scatter_mps(
    const at::Tensor& key_in, const at::Tensor& value_in,
    const at::Tensor& slot_mapping_in, int64_t num_blocks, int64_t block_size) {
  TORCH_CHECK(key_in.device().is_mps(), "kv_cache_scatter: key must be an MPS tensor");
  TORCH_CHECK(value_in.device().is_mps() && slot_mapping_in.device().is_mps(),
              "kv_cache_scatter: all inputs must be MPS tensors");
  TORCH_CHECK(key_in.dim() == 3 && value_in.sizes() == key_in.sizes(),
              "kv_cache_scatter: key/value must have shape (num_tokens, num_heads, head_size)");
  TORCH_CHECK(slot_mapping_in.dim() == 1 && slot_mapping_in.size(0) == key_in.size(0),
              "kv_cache_scatter: slot_mapping must have shape (num_tokens,)");
  TORCH_CHECK(key_in.scalar_type() == value_in.scalar_type() && tk_is_float_dtype(key_in),
              "kv_cache_scatter: key/value must share float32, float16, or bfloat16 dtype");
  TORCH_CHECK(slot_mapping_in.scalar_type() == at::kLong,
              "kv_cache_scatter: slot_mapping must be int64/torch.long");
  TORCH_CHECK(num_blocks > 0 && block_size > 0,
              "kv_cache_scatter: num_blocks and block_size must be positive");

  auto key = key_in.contiguous();
  auto value = value_in.contiguous();
  auto slot_mapping = slot_mapping_in.contiguous();
  const int T = key.size(0), H = key.size(1), D = key.size(2);
  auto key_cache = at::empty({num_blocks, block_size, H, D}, key.options());
  auto value_cache = at::empty({num_blocks, block_size, H, D}, key.options());
  const std::string tn = tk_type_name(key);
  const uint64_t total = static_cast<uint64_t>(key_cache.numel());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kv_cache_zero(e, key_cache, value_cache, total, tn);
    tk::launch_kv_cache_scatter(e, key, value, slot_mapping, key_cache, value_cache,
                                T, H, D, static_cast<int>(block_size), tn);
  });
  return {key_cache, value_cache};
}

static std::tuple<at::Tensor, at::Tensor> kv_cache_gather_mps(
    const at::Tensor& key_cache_in, const at::Tensor& value_cache_in,
    const at::Tensor& block_table_in, const at::Tensor& cu_seq_lens_in,
    int64_t num_tokens) {
  TORCH_CHECK(key_cache_in.device().is_mps(), "kv_cache_gather: key_cache must be MPS");
  TORCH_CHECK(value_cache_in.device().is_mps() && block_table_in.device().is_mps() &&
                  cu_seq_lens_in.device().is_mps(),
              "kv_cache_gather: all inputs must be MPS tensors");
  TORCH_CHECK(key_cache_in.dim() == 4 && value_cache_in.sizes() == key_cache_in.sizes(),
              "kv_cache_gather: caches must have shape (num_blocks, block_size, num_heads, head_size)");
  TORCH_CHECK(block_table_in.dim() == 2 && cu_seq_lens_in.dim() == 1 &&
                  cu_seq_lens_in.size(0) == block_table_in.size(0) + 1,
              "kv_cache_gather: block_table (B,max_blocks), cu_seq_lens (B+1,)");
  TORCH_CHECK(key_cache_in.scalar_type() == value_cache_in.scalar_type() && tk_is_float_dtype(key_cache_in),
              "kv_cache_gather: caches must share float32, float16, or bfloat16 dtype");
  TORCH_CHECK(block_table_in.scalar_type() == at::kInt && cu_seq_lens_in.scalar_type() == at::kInt,
              "kv_cache_gather: block_table and cu_seq_lens must be int32");
  TORCH_CHECK(num_tokens >= 0, "kv_cache_gather: num_tokens must be non-negative");

  auto key_cache = key_cache_in.contiguous();
  auto value_cache = value_cache_in.contiguous();
  auto block_table = block_table_in.contiguous();
  auto cu_seq_lens = cu_seq_lens_in.contiguous();
  const int H = key_cache.size(2), D = key_cache.size(3);
  auto key_out = at::empty({num_tokens, H, D}, key_cache.options());
  auto value_out = at::empty({num_tokens, H, D}, key_cache.options());
  const std::string tn = tk_type_name(key_cache);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kv_cache_gather(e, key_cache, value_cache, key_out, value_out,
                               block_table, cu_seq_lens, static_cast<int>(num_tokens),
                               static_cast<int>(cu_seq_lens.size(0) - 1),
                               static_cast<int>(key_cache.size(1)),
                               static_cast<int>(block_table.size(1)), H, D, tn);
  });
  return {key_out, value_out};
}

static std::tuple<at::Tensor, at::Tensor> kv_cache_copy_blocks_mps(
    const at::Tensor& key_cache_in, const at::Tensor& value_cache_in,
    const at::Tensor& block_mapping_in) {
  TORCH_CHECK(key_cache_in.device().is_mps(), "kv_cache_copy_blocks: key_cache must be MPS");
  TORCH_CHECK(value_cache_in.device().is_mps() && block_mapping_in.device().is_mps(),
              "kv_cache_copy_blocks: all inputs must be MPS tensors");
  TORCH_CHECK(key_cache_in.dim() == 4 && value_cache_in.sizes() == key_cache_in.sizes(),
              "kv_cache_copy_blocks: caches must have shape (num_blocks, block_size, num_heads, head_size)");
  TORCH_CHECK(block_mapping_in.dim() == 2 && block_mapping_in.size(1) == 2,
              "kv_cache_copy_blocks: block_mapping must have shape (num_pairs, 2)");
  TORCH_CHECK(key_cache_in.scalar_type() == value_cache_in.scalar_type() && tk_is_float_dtype(key_cache_in),
              "kv_cache_copy_blocks: caches must share float32, float16, or bfloat16 dtype");
  TORCH_CHECK(block_mapping_in.scalar_type() == at::kLong,
              "kv_cache_copy_blocks: block_mapping must be int64/torch.long");

  auto key_cache = key_cache_in.contiguous();
  auto value_cache = value_cache_in.contiguous();
  auto block_mapping = block_mapping_in.contiguous();
  auto key_out = at::empty_like(key_cache);
  auto value_out = at::empty_like(value_cache);
  const std::string tn = tk_type_name(key_cache);
  const uint64_t total = static_cast<uint64_t>(key_cache.numel());
  const int numel_per_block = key_cache.size(1) * key_cache.size(2) * key_cache.size(3);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kv_cache_clone(e, key_cache, value_cache, key_out, value_out, total, tn);
    tk::launch_kv_cache_copy_blocks(e, key_cache, value_cache, key_out, value_out, block_mapping,
                                    static_cast<int>(block_mapping.size(0)),
                                    numel_per_block, tn);
  });
  return {key_out, value_out};
}

static at::Tensor beam_build_copy_pairs_mps(
    const at::Tensor& parent_beam_in, const at::Tensor& block_table_in,
    const at::Tensor& seq_lens_in, int64_t block_size) {
  TORCH_CHECK(parent_beam_in.device().is_mps() && block_table_in.device().is_mps() &&
              seq_lens_in.device().is_mps(), "beam_build_copy_pairs: all inputs must be MPS tensors");
  TORCH_CHECK(parent_beam_in.dim() == 2, "beam_build_copy_pairs: parent_beam must be (B, BM)");
  TORCH_CHECK(block_table_in.dim() == 2, "beam_build_copy_pairs: block_table must be (B*BM, max_blocks)");
  auto parent_beam = parent_beam_in.to(at::kInt).contiguous();
  auto block_table = block_table_in.to(at::kInt).contiguous();
  const int B = parent_beam.size(0), BM = parent_beam.size(1);
  const int max_blocks = block_table.size(1);
  auto seq_lens = seq_lens_in.reshape({(long)(B * BM)}).to(at::kInt).contiguous();
  const int n_slots = B * BM * max_blocks;
  auto pairs = at::empty({(long)n_slots, 2}, parent_beam.options().dtype(at::kLong));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_beam_build_copy_pairs(e, parent_beam, block_table, seq_lens, pairs,
                                     BM, max_blocks, static_cast<int>(block_size), n_slots);
  });
  return pairs;
}

static at::Tensor beam_remap_block_table_mps(const at::Tensor& block_table_in,
                                             const at::Tensor& parent_beam_in) {
  TORCH_CHECK(block_table_in.device().is_mps() && parent_beam_in.device().is_mps(),
              "beam_remap_block_table: inputs must be MPS tensors");
  TORCH_CHECK(parent_beam_in.dim() == 2 && block_table_in.dim() == 2,
              "beam_remap_block_table: parent_beam (B,BM), block_table (B*BM,max_blocks)");
  auto block_table = block_table_in.to(at::kInt).contiguous();
  auto parent_beam = parent_beam_in.to(at::kInt).contiguous();
  const int BM = parent_beam.size(1), nrows = block_table.size(0), max_blocks = block_table.size(1);
  auto out = at::empty({nrows, max_blocks}, block_table.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_beam_remap_block_table(e, block_table, parent_beam, out, nrows, BM, max_blocks);
  });
  return out;
}

static std::tuple<at::Tensor, at::Tensor> kv_cache_scales_mps(
    const at::Tensor& key_in, const at::Tensor& value_in) {
  TORCH_CHECK(key_in.device().is_mps(), "kv_cache_scales: key must be MPS");
  TORCH_CHECK(value_in.device().is_mps(), "kv_cache_scales: value must be MPS");
  TORCH_CHECK(key_in.sizes() == value_in.sizes(), "kv_cache_scales: key/value shape mismatch");
  TORCH_CHECK(key_in.scalar_type() == value_in.scalar_type() && tk_is_float_dtype(key_in),
              "kv_cache_scales: key/value must share float32, float16, or bfloat16 dtype");
  auto key = key_in.contiguous();
  auto value = value_in.contiguous();
  auto key_scale = at::empty({1}, key.options().dtype(at::kFloat));
  auto value_scale = at::empty({1}, key.options().dtype(at::kFloat));
  const std::string tn = tk_type_name(key);
  const uint64_t n = static_cast<uint64_t>(key.numel());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kv_cache_scales(e, key, value, key_scale, value_scale, n, tn);
  });
  return {key_scale, value_scale};
}

static std::tuple<at::Tensor, at::Tensor> kv_cache_gather_fp8_mps(
    const at::Tensor& kc_in, const at::Tensor& vc_in, const at::Tensor& bt_in,
    const at::Tensor& cu_in, const at::Tensor& ks_in, const at::Tensor& vs_in,
    int64_t num_tokens, int64_t fmt) {
  TORCH_CHECK(kc_in.device().is_mps() && kc_in.dim() == 4 && vc_in.sizes() == kc_in.sizes(),
              "kv_cache_gather_fp8: caches must be (num_blocks, block_size, num_kv_heads, head_size)");
  auto kc = kc_in.to(at::kByte).contiguous(), vc = vc_in.to(at::kByte).contiguous();
  auto bt = bt_in.to(at::kInt).contiguous(), cu = cu_in.to(at::kInt).contiguous();
  auto ks = ks_in.to(at::kFloat).contiguous(), vs = vs_in.to(at::kFloat).contiguous();
  const int H = kc.size(2), D = kc.size(3);
  auto opts = at::TensorOptions().dtype(at::kBFloat16).device(kc.device());
  auto key_out = at::empty({num_tokens, H, D}, opts);
  auto value_out = at::empty({num_tokens, H, D}, opts);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kv_cache_gather_fp8(e, kc, vc, key_out, value_out, bt, cu, ks, vs,
                                   (int)num_tokens, (int)(cu.size(0) - 1), (int)kc.size(1),
                                   (int)bt.size(1), H, D, (int)fmt, "bfloat16");
  });
  return {key_out, value_out};
}

static std::tuple<at::Tensor, at::Tensor> kv_cache_scale_update_mps(
    const at::Tensor& key_in, const at::Tensor& value_in, const at::Tensor& oks_in,
    const at::Tensor& ovs_in) {
  TORCH_CHECK(key_in.device().is_mps() && key_in.sizes() == value_in.sizes() &&
                  tk_is_float_dtype(key_in),
              "kv_cache_scale_update: key/value must be same-shape float MPS tensors");
  auto key = key_in.contiguous(), value = value_in.contiguous();
  auto oks = oks_in.to(at::kFloat).contiguous(), ovs = ovs_in.to(at::kFloat).contiguous();
  auto nks = at::empty({1}, key.options().dtype(at::kFloat));
  auto nvs = at::empty({1}, key.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kv_cache_scale_update(e, key, value, oks, ovs, nks, nvs,
                                     (uint64_t)key.numel(), tk_type_name(key));
  });
  return {nks, nvs};
}

// fp8 KV cache: scatter K/V into a uint8 (e4m3) paged cache with per-tensor scales.
static std::tuple<at::Tensor, at::Tensor> kv_cache_scatter_fp8_mps(
    const at::Tensor& key_in, const at::Tensor& value_in, const at::Tensor& slot_in,
    int64_t num_blocks, int64_t block_size, const at::Tensor& k_scale_in,
    const at::Tensor& v_scale_in, int64_t fmt) {
  TORCH_CHECK(key_in.device().is_mps(), "kv_cache_scatter_fp8: key must be an MPS tensor");
  TORCH_CHECK(key_in.dim() == 3 && value_in.sizes() == key_in.sizes(),
              "kv_cache_scatter_fp8: key/value must be (num_tokens, num_heads, head_size)");
  TORCH_CHECK(tk_is_float_dtype(key_in), "kv_cache_scatter_fp8: key/value must be float");
  TORCH_CHECK(slot_in.dim() == 1 && slot_in.size(0) == key_in.size(0),
              "kv_cache_scatter_fp8: slot_mapping must be (num_tokens,)");
  auto key = key_in.contiguous(), value = value_in.contiguous();
  auto slot = slot_in.to(at::kLong).contiguous();
  const int T = key.size(0), H = key.size(1), D = key.size(2);
  TORCH_CHECK(k_scale_in.dim() == 1 && k_scale_in.size(0) == H && v_scale_in.sizes() == k_scale_in.sizes(),
              "kv_cache_scatter_fp8: k_scale/v_scale must be (num_heads,)");
  auto ks = k_scale_in.to(at::kFloat).contiguous(), vs = v_scale_in.to(at::kFloat).contiguous();
  auto kc = at::empty({num_blocks, block_size, H, D}, key.options().dtype(at::kByte));
  auto vc = at::empty({num_blocks, block_size, H, D}, key.options().dtype(at::kByte));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kv_cache_zero_u8(e, kc, vc, static_cast<uint64_t>(kc.numel()));
    tk::launch_kv_cache_scatter_fp8(e, key, value, slot, kc, vc, T, H, D,
                                    static_cast<int>(block_size), ks, vs,
                                    static_cast<int>(fmt), tk_type_name(key));
  });
  return {kc, vc};
}

static at::Tensor paged_attention_fp8_mps(
    const at::Tensor& q_in, const at::Tensor& key_cache_in, const at::Tensor& value_cache_in,
    const at::Tensor& block_table_in, const at::Tensor& context_lens_in,
    const at::Tensor& k_scale_in, const at::Tensor& v_scale_in, double scale, int64_t fmt,
    int64_t window) {
  TORCH_CHECK(q_in.device().is_mps() && tk_is_float_dtype(q_in), "paged_attention_fp8: q must be float MPS");
  TORCH_CHECK(q_in.dim() == 3, "paged_attention_fp8: q must be (B,H,D)");
  TORCH_CHECK(key_cache_in.dim() == 4 && value_cache_in.sizes() == key_cache_in.sizes(),
              "paged_attention_fp8: caches must be (num_blocks, block_size, num_kv_heads, D)");
  TORCH_CHECK(key_cache_in.scalar_type() == at::kByte && value_cache_in.scalar_type() == at::kByte,
              "paged_attention_fp8: caches must be uint8 (e4m3 codes)");
  TORCH_CHECK(key_cache_in.size(3) == q_in.size(2), "paged_attention_fp8: head_size mismatch");
  TORCH_CHECK(key_cache_in.size(2) > 0 && q_in.size(1) % key_cache_in.size(2) == 0,
              "paged_attention_fp8: num_q_heads must be a positive multiple of num_kv_heads");
  TORCH_CHECK(block_table_in.scalar_type() == at::kInt && context_lens_in.scalar_type() == at::kInt,
              "paged_attention_fp8: block_table and context_lens must be int32");
  const int B = q_in.size(0), H = q_in.size(1), D = q_in.size(2);
  const int H_KV = key_cache_in.size(2), block_size = key_cache_in.size(1);
  TORCH_CHECK(D == 64 || D == 128, "paged_attention_fp8: head_size must be 64 or 128");
  TORCH_CHECK(k_scale_in.dim() == 1 && k_scale_in.size(0) == H_KV && v_scale_in.sizes() == k_scale_in.sizes(),
              "paged_attention_fp8: k_scale/v_scale must be (num_kv_heads,)");
  auto q = q_in.contiguous();
  auto kc = key_cache_in.contiguous(), vc = value_cache_in.contiguous();
  auto bt = block_table_in.contiguous(), cl = context_lens_in.contiguous();
  auto ks = k_scale_in.to(at::kFloat).contiguous(), vs = v_scale_in.to(at::kFloat).contiguous();
  auto out = at::empty_like(q);
  const float scale_f = scale > 0.0 ? static_cast<float>(scale)
                                    : 1.0f / std::sqrt(static_cast<float>(D));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_paged_attention_fp8(e, q, kc, vc, bt, cl, out, B, H, H_KV, D, block_size,
                                   static_cast<int>(bt.size(1)), scale_f, ks, vs,
                                   static_cast<int>(fmt), static_cast<int>(window), tk_type_name(q));
  });
  return out;
}

// Fused K RMSNorm + RoPE + paged-KV insert. Returns the two updated caches.
static std::tuple<at::Tensor, at::Tensor> rope_kv_insert_norm_mps(
    const at::Tensor& k_in, const at::Tensor& v_in, const at::Tensor& cos_in,
    const at::Tensor& sin_in, const at::Tensor& positions_in, const at::Tensor& slot_mapping_in,
    const at::Tensor& key_cache_in, const at::Tensor& value_cache_in, const at::Tensor& nw_in,
    double eps, bool gemma) {
  TORCH_CHECK(k_in.device().is_mps() && tk_is_float_dtype(k_in),
              "rope_kv_insert_norm: k must be float32/float16/bfloat16 MPS");
  TORCH_CHECK(k_in.dim() == 3 && v_in.sizes() == k_in.sizes(), "rope_kv_insert_norm: k/v (T,H,D)");
  const int num_tokens = k_in.size(0), num_kv_heads = k_in.size(1), D = k_in.size(2);
  TORCH_CHECK(D == 64 || D == 128, "rope_kv_insert_norm: D must be 64 or 128");
  TORCH_CHECK(key_cache_in.dim() == 4 && key_cache_in.size(2) == num_kv_heads &&
                  key_cache_in.size(3) == D, "rope_kv_insert_norm: cache mismatch");
  TORCH_CHECK(nw_in.dim() == 1 && nw_in.size(0) == D, "rope_kv_insert_norm: norm_weight (D,)");
  const int block_size = key_cache_in.size(1);
  const auto dt = k_in.scalar_type();
  auto k = k_in.contiguous(), v = v_in.contiguous();
  auto cos = cos_in.to(dt).contiguous(), sin = sin_in.to(dt).contiguous();
  auto positions = positions_in.to(at::kInt).contiguous();
  auto slot_mapping = slot_mapping_in.to(at::kLong).contiguous();
  auto nw = nw_in.to(dt).contiguous();
  auto key_out = key_cache_in.to(dt).contiguous().clone();
  auto value_out = value_cache_in.to(dt).contiguous().clone();
  tk_encode([&](TorchEncoder& e) {
    tk::launch_rope_kv_insert_norm(e, k, v, cos, sin, positions, slot_mapping, key_out, value_out,
                                   nw, num_tokens * num_kv_heads, num_kv_heads, block_size, D,
                                   static_cast<float>(eps), gemma ? 1 : 0, tk_type_name(k));
  });
  return {key_out, value_out};
}

// Q-path RoPE (+ optional weighted RMSNorm) into a contiguous q_out.
static at::Tensor rope_q_mps(
    const at::Tensor& q_in, const at::Tensor& cos_in, const at::Tensor& sin_in,
    const at::Tensor& positions_in, const at::Tensor& nw_in, bool do_norm, bool gemma, double eps) {
  TORCH_CHECK(q_in.device().is_mps() && tk_is_float_dtype(q_in), "rope_q: q must be float MPS");
  TORCH_CHECK(q_in.dim() == 3, "rope_q: q must be (num_tokens, num_q_heads, D)");
  const int num_heads = q_in.size(1), D = q_in.size(2);
  TORCH_CHECK(D == 64 || D == 128, "rope_q: D must be 64 or 128");
  TORCH_CHECK(cos_in.size(-1) == D / 2 && sin_in.size(-1) == D / 2, "rope_q: cos/sin (P, D/2)");
  const auto dt = q_in.scalar_type();
  auto q = q_in.contiguous();
  auto cos = cos_in.to(dt).contiguous(), sin = sin_in.to(dt).contiguous();
  auto positions = positions_in.to(at::kInt).contiguous();
  auto nw = nw_in.to(dt).contiguous();
  auto out = at::empty_like(q);
  const int M = static_cast<int>(q.numel() / D);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_rope_q(e, q, cos, sin, positions, out, nw, M, num_heads, do_norm ? 1 : 0,
                      gemma ? 1 : 0, static_cast<float>(eps), D, tk_type_name(q));
  });
  return out;
}

// DeepSeek MLA Q-path: optional RMSNorm + GPT-J interleaved RoPE on the last rope_dim dims.
static at::Tensor mla_q_norm_rope_mps(
    const at::Tensor& q_in, const at::Tensor& cos_in, const at::Tensor& sin_in,
    const at::Tensor& positions_in, const at::Tensor& nw_in, int64_t num_heads,
    int64_t nope_dim, int64_t rope_dim, int64_t norm_mode, double eps) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.scalar_type() == at::kBFloat16,
              "mla_q_norm_rope: q must be bf16 MPS");
  const int head_dim = q_in.size(-1);
  TORCH_CHECK(head_dim % 64 == 0 && head_dim == nope_dim + rope_dim,
              "mla_q_norm_rope: head_dim must be nope+rope and %64==0");
  TORCH_CHECK(cos_in.size(-1) == rope_dim / 2 && sin_in.size(-1) == rope_dim / 2,
              "mla_q_norm_rope: cos/sin must be (max_pos, rope_dim/2)");
  auto q = q_in.contiguous();
  auto cos = cos_in.to(at::kBFloat16).contiguous(), sin = sin_in.to(at::kBFloat16).contiguous();
  auto positions = positions_in.to(at::kInt).contiguous();
  auto nw = nw_in.to(at::kBFloat16).contiguous();
  auto out = at::empty_like(q);
  const int M = static_cast<int>(q.numel() / head_dim);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_mla_q_norm_rope(e, q, cos, sin, positions, nw, out, M,
                               static_cast<int>(num_heads), static_cast<int>(nope_dim),
                               static_cast<int>(rope_dim), static_cast<int>(norm_mode),
                               static_cast<float>(eps), head_dim);
  });
  return out;
}

// DeepSeek MLA classic KV-insert (clone-then-insert into a paged bf16 cache).
static at::Tensor mla_kv_insert_mps(
    const at::Tensor& kv_c_in, const at::Tensor& k_pe_in, const at::Tensor& cos_in,
    const at::Tensor& sin_in, const at::Tensor& positions_in, const at::Tensor& slot_in,
    const at::Tensor& cache_in, const at::Tensor& nw_in, int64_t rope_dim, int64_t norm_mode,
    double eps) {
  TORCH_CHECK(kv_c_in.device().is_mps() && kv_c_in.scalar_type() == at::kBFloat16,
              "mla_kv_insert: kv_c must be bf16 MPS");
  const int latent = kv_c_in.size(-1);
  TORCH_CHECK(latent % 64 == 0, "mla_kv_insert: LATENT must be %64==0");
  TORCH_CHECK(k_pe_in.size(-1) == rope_dim && rope_dim % 2 == 0 && rope_dim / 2 <= 32,
              "mla_kv_insert: k_pe last dim must be rope_dim (even, /2<=32)");
  TORCH_CHECK(cache_in.dim() == 3 && cache_in.size(2) == latent + rope_dim,
              "mla_kv_insert: kv_cache must be (nb, bs, LATENT+rope_dim)");
  auto kv_c = kv_c_in.contiguous(), k_pe = k_pe_in.contiguous();
  auto cos = cos_in.to(at::kBFloat16).contiguous(), sin = sin_in.to(at::kBFloat16).contiguous();
  auto positions = positions_in.to(at::kInt).contiguous();
  auto slot = slot_in.to(at::kLong).contiguous();
  auto nw = nw_in.to(at::kBFloat16).contiguous();
  auto out = cache_in.contiguous().clone();
  const int num_tokens = static_cast<int>(kv_c.numel() / latent);
  const int block_size = cache_in.size(1);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_mla_kv_insert(e, kv_c, k_pe, cos, sin, positions, slot, out, nw, num_tokens,
                             block_size, static_cast<int>(rope_dim), static_cast<int>(norm_mode),
                             static_cast<float>(eps), latent);
  });
  return out;
}

// DeepSeek MLA absorb-path latent decode (MQA). q (B,N,576), cache (nb,bs,576) -> o (B,N,512).
static at::Tensor mla_decode_mps(
    const at::Tensor& q_in, const at::Tensor& cache_in, const at::Tensor& block_table_in,
    const at::Tensor& context_lens_in, double scale) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.scalar_type() == at::kBFloat16,
              "mla_decode: q must be bf16 MPS");
  TORCH_CHECK(q_in.dim() == 3 && q_in.size(2) == 576, "mla_decode: q must be (B, N, 576)");
  TORCH_CHECK(cache_in.dim() == 3 && cache_in.size(2) == 576, "mla_decode: cache must be (nb, bs, 576)");
  TORCH_CHECK(block_table_in.scalar_type() == at::kInt && context_lens_in.scalar_type() == at::kInt,
              "mla_decode: block_table and context_lens must be int32");
  const int B = q_in.size(0), N = q_in.size(1);
  const int latent = 512;
  auto q = q_in.contiguous();
  auto cache = cache_in.contiguous();
  auto bt = block_table_in.contiguous(), cl = context_lens_in.contiguous();
  auto out = at::empty({B, N, latent}, q.options());
  const float scale_f = scale > 0.0 ? static_cast<float>(scale) : 1.0f / std::sqrt(576.0f);
  // P4v2 route: partitioned multi-head decode + paged-v2 reduce (see mla.metal for rationale)
  const int block_size = static_cast<int>(cache.size(1));
  const int max_ctx = static_cast<int>(bt.size(1)) * block_size;
  int psize = ((512 + block_size - 1) / block_size) * block_size;
  const int P = std::max(1, (max_ctx + psize - 1) / psize);
  auto f32 = q.options().dtype(at::kFloat);
  auto tmp_out = at::empty({B, N, P, latent}, f32);
  auto max_logits = at::empty({B, N, P}, f32);
  auto exp_sums = at::empty({B, N, P}, f32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_mla_decode_partition(e, q, cache, bt, cl, tmp_out, max_logits, exp_sums, B, N,
                                    block_size, static_cast<int>(bt.size(1)), scale_f, latent,
                                    64, P, psize);
    tk::launch_paged_attention_reduce(e, tmp_out, max_logits, exp_sums, out, B, N, latent, P, tmp_out, 0, "bfloat16");
  });
  return out;
}

// DeepSeek-V4 dense latent decode over the packed cache. q (B,N,512) -> o (B,N,512).
static at::Tensor mla_decode_fp8_mps(
    const at::Tensor& q_in, const at::Tensor& data_in, const at::Tensor& scale_in,
    const at::Tensor& block_table_in, const at::Tensor& context_lens_in, double scale) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.scalar_type() == at::kBFloat16,
              "mla_decode_fp8: q must be bf16 MPS");
  TORCH_CHECK(q_in.dim() == 3 && q_in.size(2) == 512, "mla_decode_fp8: q must be (B, N, 512)");
  TORCH_CHECK(data_in.dim() == 3 && data_in.size(2) == 576 && data_in.scalar_type() == at::kByte,
              "mla_decode_fp8: data_cache must be (nb, bs, 576) uint8");
  TORCH_CHECK(scale_in.dim() == 3 && scale_in.size(2) == 8 && scale_in.scalar_type() == at::kByte,
              "mla_decode_fp8: scale_cache must be (nb, bs, 8) uint8");
  TORCH_CHECK(block_table_in.scalar_type() == at::kInt && context_lens_in.scalar_type() == at::kInt,
              "mla_decode_fp8: block_table and context_lens must be int32");
  const int B = q_in.size(0), N = q_in.size(1);
  auto q = q_in.contiguous();
  auto data = data_in.contiguous(), sc = scale_in.contiguous();
  auto bt = block_table_in.contiguous(), cl = context_lens_in.contiguous();
  auto out = at::empty({B, N, 512}, q.options());
  const float scale_f = scale > 0.0 ? static_cast<float>(scale) : 1.0f / std::sqrt(512.0f);
  // P4a-v2 route: partitioned decode + paged-v2 reduce (same upgrade as bf16 mla_decode)
  const int block_size = static_cast<int>(data.size(1));
  const int max_ctx = static_cast<int>(bt.size(1)) * block_size;
  const int psize = ((512 + block_size - 1) / block_size) * block_size;
  const int P = std::max(1, (max_ctx + psize - 1) / psize);
  auto f32 = q.options().dtype(at::kFloat);
  auto tmp_out = at::empty({B, N, P, 512}, f32);
  auto max_logits = at::empty({B, N, P}, f32);
  auto exp_sums = at::empty({B, N, P}, f32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_mla_decode_fp8_partition(e, q, data, sc, bt, cl, tmp_out, max_logits, exp_sums,
                                        B, N, block_size, static_cast<int>(bt.size(1)), scale_f,
                                        P, psize);
    tk::launch_paged_attention_reduce(e, tmp_out, max_logits, exp_sums, out, B, N, 512, P, tmp_out, 0, "bfloat16");
  });
  return out;
}

// DeepSeek-V4 sparse latent decode: attend only indices[b, 0:topk_length[b]].
static at::Tensor mla_decode_fp8_sparse_mps(
    const at::Tensor& q_in, const at::Tensor& data_in, const at::Tensor& scale_in,
    const at::Tensor& block_table_in, const at::Tensor& indices_in, const at::Tensor& lens_in,
    double scale) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.scalar_type() == at::kBFloat16,
              "mla_decode_fp8_sparse: q must be bf16 MPS");
  TORCH_CHECK(q_in.dim() == 3 && q_in.size(2) == 512, "mla_decode_fp8_sparse: q must be (B,N,512)");
  TORCH_CHECK(data_in.dim() == 3 && data_in.size(2) == 576 && data_in.scalar_type() == at::kByte,
              "mla_decode_fp8_sparse: data_cache (nb,bs,576) uint8");
  TORCH_CHECK(scale_in.dim() == 3 && scale_in.size(2) == 8 && scale_in.scalar_type() == at::kByte,
              "mla_decode_fp8_sparse: scale_cache (nb,bs,8) uint8");
  TORCH_CHECK(indices_in.dim() == 2 && indices_in.size(0) == q_in.size(0),
              "mla_decode_fp8_sparse: indices (B, max_topk)");
  const int B = q_in.size(0), N = q_in.size(1), max_topk = indices_in.size(1);
  auto q = q_in.contiguous();
  auto data = data_in.contiguous(), sc = scale_in.contiguous();
  auto bt = block_table_in.contiguous();
  auto idx = indices_in.to(at::kInt).contiguous(), lens = lens_in.to(at::kInt).contiguous();
  auto out = at::empty({B, N, 512}, q.options());
  const float scale_f = scale > 0.0 ? static_cast<float>(scale) : 1.0f / std::sqrt(512.0f);
  // P4b-v2 route: partition the top-k index list + paged-v2 reduce
  const int psize = 512;
  const int P = std::max(1, (max_topk + psize - 1) / psize);
  auto f32 = q.options().dtype(at::kFloat);
  auto tmp_out = at::empty({B, N, P, 512}, f32);
  auto max_logits = at::empty({B, N, P}, f32);
  auto exp_sums = at::empty({B, N, P}, f32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_mla_decode_fp8_sparse_partition(e, q, data, sc, bt, idx, lens, tmp_out,
                                               max_logits, exp_sums, B, N,
                                               static_cast<int>(data.size(1)),
                                               static_cast<int>(bt.size(1)), scale_f, max_topk,
                                               P, psize);
    tk::launch_paged_attention_reduce(e, tmp_out, max_logits, exp_sums, out, B, N, 512, P, tmp_out, 0, "bfloat16");
  });
  return out;
}

// DeepSeek-V4 packed MLA KV-insert. Returns (data_cache uint8 (…,576), scale_cache uint8 (…,8)).
static std::tuple<at::Tensor, at::Tensor> mla_kv_insert_fp8_mps(
    const at::Tensor& kv_in, const at::Tensor& cos_in, const at::Tensor& sin_in,
    const at::Tensor& positions_in, const at::Tensor& slot_in, const at::Tensor& data_in,
    const at::Tensor& scale_in) {
  TORCH_CHECK(kv_in.device().is_mps() && kv_in.scalar_type() == at::kBFloat16,
              "mla_kv_insert_fp8: kv must be bf16 MPS");
  TORCH_CHECK(kv_in.size(-1) == 512, "mla_kv_insert_fp8: kv must be (…, 512)");
  TORCH_CHECK(data_in.dim() == 3 && data_in.size(2) == 576 && data_in.scalar_type() == at::kByte,
              "mla_kv_insert_fp8: data_cache must be (nb, bs, 576) uint8");
  TORCH_CHECK(scale_in.dim() == 3 && scale_in.size(2) == 8 && scale_in.scalar_type() == at::kByte,
              "mla_kv_insert_fp8: scale_cache must be (nb, bs, 8) uint8");
  auto kv = kv_in.contiguous();
  auto cos = cos_in.to(at::kBFloat16).contiguous(), sin = sin_in.to(at::kBFloat16).contiguous();
  auto positions = positions_in.to(at::kInt).contiguous();
  auto slot = slot_in.to(at::kLong).contiguous();
  auto data_out = data_in.contiguous().clone();
  auto scale_out = scale_in.contiguous().clone();
  const int num_tokens = static_cast<int>(kv.numel() / 512);
  const int block_size = data_in.size(1);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_mla_kv_insert_fp8(e, kv, cos, sin, positions, slot, data_out, scale_out,
                                 num_tokens, block_size);
  });
  return {data_out, scale_out};
}

static at::Tensor paged_attention_mps(
    const at::Tensor& q_in, const at::Tensor& key_cache_in,
    const at::Tensor& value_cache_in, const at::Tensor& block_table_in,
    const at::Tensor& context_lens_in, double scale, int64_t window) {
  TORCH_CHECK(q_in.device().is_mps(), "paged_attention: q must be an MPS tensor");
  TORCH_CHECK(key_cache_in.device().is_mps() && value_cache_in.device().is_mps() &&
                  block_table_in.device().is_mps() && context_lens_in.device().is_mps(),
              "paged_attention: all inputs must be MPS tensors");
  TORCH_CHECK(q_in.dim() == 3, "paged_attention: q must have shape (B,H,D)");
  TORCH_CHECK(key_cache_in.dim() == 4 && value_cache_in.sizes() == key_cache_in.sizes(),
              "paged_attention: caches must have shape (num_blocks, block_size, H, D)");
  TORCH_CHECK(key_cache_in.size(3) == q_in.size(2),
              "paged_attention: q head_size must match caches");
  TORCH_CHECK(key_cache_in.size(2) > 0 && q_in.size(1) % key_cache_in.size(2) == 0,
              "paged_attention: num_q_heads must be a positive multiple of num_kv_heads (GQA/MQA)");
  TORCH_CHECK(block_table_in.dim() == 2 && block_table_in.size(0) == q_in.size(0),
              "paged_attention: block_table must have shape (B, max_blocks)");
  TORCH_CHECK(context_lens_in.dim() == 1 && context_lens_in.size(0) == q_in.size(0),
              "paged_attention: context_lens must have shape (B,)");
  TORCH_CHECK(q_in.scalar_type() == key_cache_in.scalar_type() &&
                  q_in.scalar_type() == value_cache_in.scalar_type() && tk_is_float_dtype(q_in),
              "paged_attention: q/cache dtype must be float32, float16, or bfloat16");
  TORCH_CHECK(block_table_in.scalar_type() == at::kInt && context_lens_in.scalar_type() == at::kInt,
              "paged_attention: block_table and context_lens must be int32");
  const int B = q_in.size(0), H = q_in.size(1), D = q_in.size(2);
  const int H_KV = key_cache_in.size(2);
  TORCH_CHECK(D == 64 || D == 128, "paged_attention: head_size must be 64 or 128");

  auto q = q_in.contiguous();
  auto key_cache = key_cache_in.contiguous();
  auto value_cache = value_cache_in.contiguous();
  auto block_table = block_table_in.contiguous();
  auto context_lens = context_lens_in.contiguous();
  auto out = at::empty_like(q);
  auto no_alibi = at::zeros({1}, q.options().dtype(at::kFloat));  // buffer 11 placeholder
  auto no_mask = at::zeros({1}, q.options().dtype(at::kInt));     // buffer 13 placeholder
  const float scale_f = scale > 0.0 ? static_cast<float>(scale)
                                    : 1.0f / std::sqrt(static_cast<float>(D));
  const std::string tn = tk_type_name(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_paged_attention(e, q,
        key_cache,
        value_cache,
        block_table,
        context_lens,
        out,
        B,
        H,
        H_KV,
        D,
        static_cast<int>(key_cache.size(1)),
        static_cast<int>(block_table.size(1)),
        scale_f,
        no_alibi,
        0,
        no_mask,
        0,
        static_cast<int>(window), 1, tn);
  });
  return out;
}

// Block-sparse paged decode: block_mask (batch, max_blocks) int32 (1=attend, 0=skip) per KV block.
static at::Tensor paged_attention_block_sparse_mps(
    const at::Tensor& q_in, const at::Tensor& key_cache_in,
    const at::Tensor& value_cache_in, const at::Tensor& block_table_in,
    const at::Tensor& context_lens_in, const at::Tensor& block_mask_in, double scale,
    int64_t window) {
  TORCH_CHECK(q_in.device().is_mps() && tk_is_float_dtype(q_in), "paged_attention_block_sparse: q must be float MPS");
  TORCH_CHECK(q_in.dim() == 3, "paged_attention_block_sparse: q must be (B,H,D)");
  TORCH_CHECK(key_cache_in.dim() == 4 && value_cache_in.sizes() == key_cache_in.sizes(),
              "paged_attention_block_sparse: caches must be (num_blocks, block_size, H, D)");
  TORCH_CHECK(key_cache_in.size(3) == q_in.size(2), "paged_attention_block_sparse: head_size mismatch");
  TORCH_CHECK(key_cache_in.size(2) > 0 && q_in.size(1) % key_cache_in.size(2) == 0,
              "paged_attention_block_sparse: num_q_heads must be a positive multiple of num_kv_heads");
  TORCH_CHECK(block_table_in.scalar_type() == at::kInt && context_lens_in.scalar_type() == at::kInt,
              "paged_attention_block_sparse: block_table and context_lens must be int32");
  TORCH_CHECK((block_mask_in.dim() == 2 && block_mask_in.sizes() == block_table_in.sizes()) ||
                  (block_mask_in.dim() == 3 && block_mask_in.size(0) == q_in.size(0) &&
                   block_mask_in.size(1) == q_in.size(1) &&
                   block_mask_in.size(2) == block_table_in.size(1)),
              "paged_attention_block_sparse: block_mask must be (B, max_blocks) or "
              "(B, num_heads, max_blocks)");
  const int B = q_in.size(0), H = q_in.size(1), D = q_in.size(2);
  const int mask_heads = block_mask_in.dim() == 3 ? H : 1;
  const int H_KV = key_cache_in.size(2);
  TORCH_CHECK(D == 64 || D == 128, "paged_attention_block_sparse: head_size must be 64 or 128");

  auto q = q_in.contiguous();
  auto key_cache = key_cache_in.contiguous();
  auto value_cache = value_cache_in.contiguous();
  auto block_table = block_table_in.contiguous();
  auto context_lens = context_lens_in.contiguous();
  auto mask = block_mask_in.to(at::kInt).contiguous();
  auto out = at::empty_like(q);
  auto no_alibi = at::zeros({1}, q.options().dtype(at::kFloat));
  const float scale_f = scale > 0.0 ? static_cast<float>(scale)
                                    : 1.0f / std::sqrt(static_cast<float>(D));
  const std::string tn = tk_type_name(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_paged_attention(e, q,
        key_cache,
        value_cache,
        block_table,
        context_lens,
        out,
        B,
        H,
        H_KV,
        D,
        static_cast<int>(key_cache.size(1)),
        static_cast<int>(block_table.size(1)),
        scale_f,
        no_alibi,
        0,
        mask,
        1,
        static_cast<int>(window), mask_heads, tn);
  });
  return out;
}

// Paged decode with a per-head ALiBi linear position bias (alibi_slopes is (num_heads,)).
static at::Tensor paged_attention_alibi_mps(
    const at::Tensor& q_in, const at::Tensor& key_cache_in,
    const at::Tensor& value_cache_in, const at::Tensor& block_table_in,
    const at::Tensor& context_lens_in, const at::Tensor& alibi_slopes_in, double scale,
    int64_t window) {
  TORCH_CHECK(q_in.device().is_mps() && tk_is_float_dtype(q_in), "paged_attention_alibi: q must be float MPS");
  TORCH_CHECK(q_in.dim() == 3, "paged_attention_alibi: q must be (B,H,D)");
  TORCH_CHECK(key_cache_in.dim() == 4 && value_cache_in.sizes() == key_cache_in.sizes(),
              "paged_attention_alibi: caches must be (num_blocks, block_size, H, D)");
  TORCH_CHECK(key_cache_in.size(3) == q_in.size(2), "paged_attention_alibi: head_size mismatch");
  TORCH_CHECK(key_cache_in.size(2) > 0 && q_in.size(1) % key_cache_in.size(2) == 0,
              "paged_attention_alibi: num_q_heads must be a positive multiple of num_kv_heads");
  TORCH_CHECK(block_table_in.scalar_type() == at::kInt && context_lens_in.scalar_type() == at::kInt,
              "paged_attention_alibi: block_table and context_lens must be int32");
  const int B = q_in.size(0), H = q_in.size(1), D = q_in.size(2);
  const int H_KV = key_cache_in.size(2);
  TORCH_CHECK(D == 64 || D == 128, "paged_attention_alibi: head_size must be 64 or 128");
  TORCH_CHECK(alibi_slopes_in.dim() == 1 && alibi_slopes_in.size(0) == H,
              "paged_attention_alibi: alibi_slopes must be (num_heads,)");

  auto q = q_in.contiguous();
  auto key_cache = key_cache_in.contiguous();
  auto value_cache = value_cache_in.contiguous();
  auto block_table = block_table_in.contiguous();
  auto context_lens = context_lens_in.contiguous();
  auto slopes = alibi_slopes_in.to(at::kFloat).contiguous();
  auto out = at::empty_like(q);
  auto no_mask = at::zeros({1}, q.options().dtype(at::kInt));
  const float scale_f = scale > 0.0 ? static_cast<float>(scale)
                                    : 1.0f / std::sqrt(static_cast<float>(D));
  const std::string tn = tk_type_name(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_paged_attention(e, q,
        key_cache,
        value_cache,
        block_table,
        context_lens,
        out,
        B,
        H,
        H_KV,
        D,
        static_cast<int>(key_cache.size(1)),
        static_cast<int>(block_table.size(1)),
        scale_f,
        slopes,
        1,
        no_mask,
        0,
        static_cast<int>(window), 1, tn);
  });
  return out;
}

// vLLM x-packed cache decode: key (nb, nkv, hd/x, bs, x), value (nb, nkv, hd, bs).
static at::Tensor paged_attention_xcache_mps(
    const at::Tensor& q_in, const at::Tensor& key_cache_in,
    const at::Tensor& value_cache_in, const at::Tensor& block_table_in,
    const at::Tensor& context_lens_in, double scale) {
  TORCH_CHECK(q_in.device().is_mps() && tk_is_float_dtype(q_in), "paged_attention_xcache: q must be float MPS");
  TORCH_CHECK(q_in.dim() == 3, "paged_attention_xcache: q must be (B,H,D)");
  TORCH_CHECK(key_cache_in.dim() == 5, "paged_attention_xcache: key_cache must be (nb, nkv, hd/x, bs, x)");
  TORCH_CHECK(value_cache_in.dim() == 4, "paged_attention_xcache: value_cache must be (nb, nkv, hd, bs)");
  const int B = q_in.size(0), H = q_in.size(1), D = q_in.size(2);
  const int H_KV = key_cache_in.size(1), block_size = key_cache_in.size(3), x = key_cache_in.size(4);
  TORCH_CHECK(D == 64 || D == 128, "paged_attention_xcache: head_size must be 64 or 128");
  TORCH_CHECK(x > 0 && D % x == 0 && key_cache_in.size(2) == D / x,
              "paged_attention_xcache: key_cache head_size/x split inconsistent with q head_size");
  TORCH_CHECK(value_cache_in.size(1) == H_KV && value_cache_in.size(2) == D && value_cache_in.size(3) == block_size,
              "paged_attention_xcache: value_cache shape inconsistent with key_cache");
  TORCH_CHECK(H_KV > 0 && H % H_KV == 0, "paged_attention_xcache: num_q_heads must be a positive multiple of num_kv_heads");
  TORCH_CHECK(block_table_in.scalar_type() == at::kInt && context_lens_in.scalar_type() == at::kInt,
              "paged_attention_xcache: block_table and context_lens must be int32");

  auto q = q_in.contiguous();
  auto key_cache = key_cache_in.contiguous();
  auto value_cache = value_cache_in.contiguous();
  auto block_table = block_table_in.contiguous();
  auto context_lens = context_lens_in.contiguous();
  auto out = at::empty_like(q);
  const float scale_f = scale > 0.0 ? static_cast<float>(scale)
                                    : 1.0f / std::sqrt(static_cast<float>(D));
  const std::string tn = tk_type_name(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_paged_attention_xcache(e, q, key_cache, value_cache, block_table, context_lens,
                                      out, B, H, H_KV, D, block_size,
                                      static_cast<int>(block_table.size(1)), scale_f, x, tn);
  });
  return out;
}

// GQA KV-reuse staged decode (bit-equivalent to paged_attention_mps; different memory shape).
static at::Tensor paged_attention_staged_mps(
    const at::Tensor& q_in, const at::Tensor& key_cache_in,
    const at::Tensor& value_cache_in, const at::Tensor& block_table_in,
    const at::Tensor& context_lens_in, double scale) {
  TORCH_CHECK(q_in.device().is_mps() && tk_is_float_dtype(q_in), "paged_attention_staged: q must be float MPS");
  TORCH_CHECK(q_in.dim() == 3, "paged_attention_staged: q must have shape (B,H,D)");
  TORCH_CHECK(key_cache_in.dim() == 4 && value_cache_in.sizes() == key_cache_in.sizes(),
              "paged_attention_staged: caches must have shape (num_blocks, block_size, H, D)");
  TORCH_CHECK(key_cache_in.size(3) == q_in.size(2), "paged_attention_staged: head_size mismatch");
  TORCH_CHECK(key_cache_in.size(2) > 0 && q_in.size(1) % key_cache_in.size(2) == 0,
              "paged_attention_staged: num_q_heads must be a positive multiple of num_kv_heads");
  TORCH_CHECK(q_in.scalar_type() == key_cache_in.scalar_type() &&
                  q_in.scalar_type() == value_cache_in.scalar_type(),
              "paged_attention_staged: q/cache dtype must match");
  TORCH_CHECK(block_table_in.scalar_type() == at::kInt && context_lens_in.scalar_type() == at::kInt,
              "paged_attention_staged: block_table and context_lens must be int32");
  const int B = q_in.size(0), H = q_in.size(1), D = q_in.size(2);
  const int H_KV = key_cache_in.size(2);
  TORCH_CHECK(D == 64 || D == 128, "paged_attention_staged: head_size must be 64 or 128");

  auto q = q_in.contiguous();
  auto key_cache = key_cache_in.contiguous();
  auto value_cache = value_cache_in.contiguous();
  auto block_table = block_table_in.contiguous();
  auto context_lens = context_lens_in.contiguous();
  auto out = at::empty_like(q);
  const float scale_f = scale > 0.0 ? static_cast<float>(scale)
                                    : 1.0f / std::sqrt(static_cast<float>(D));
  const std::string tn = tk_type_name(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_paged_attention_gqa_staged(e, q, key_cache, value_cache, block_table, context_lens,
                                          out, B, H, H_KV, D, static_cast<int>(key_cache.size(1)),
                                          static_cast<int>(block_table.size(1)), scale_f, tn);
  });
  return out;
}

// Fused RoPE on K + paged-KV insert. Returns the two updated caches.
static std::tuple<at::Tensor, at::Tensor> rope_kv_insert_mps(
    const at::Tensor& k_in, const at::Tensor& v_in, const at::Tensor& cos_in,
    const at::Tensor& sin_in, const at::Tensor& positions_in, const at::Tensor& slot_mapping_in,
    const at::Tensor& key_cache_in, const at::Tensor& value_cache_in) {
  TORCH_CHECK(k_in.device().is_mps(), "rope_kv_insert: inputs must be MPS tensors");
  TORCH_CHECK(k_in.dim() == 3 && v_in.sizes() == k_in.sizes(),
              "rope_kv_insert: k/v must be (num_tokens, num_kv_heads, D)");
  TORCH_CHECK(key_cache_in.dim() == 4 && value_cache_in.sizes() == key_cache_in.sizes(),
              "rope_kv_insert: caches must be (num_blocks, block_size, num_kv_heads, D)");
  TORCH_CHECK(tk_is_float_dtype(k_in), "rope_kv_insert: k must be float32/float16/bfloat16");
  const int num_tokens = k_in.size(0), num_kv_heads = k_in.size(1), D = k_in.size(2);
  TORCH_CHECK(D == 64 || D == 128, "rope_kv_insert: D must be 64 or 128");
  TORCH_CHECK(key_cache_in.size(2) == num_kv_heads && key_cache_in.size(3) == D,
              "rope_kv_insert: cache heads/head_size must match k");
  const int block_size = key_cache_in.size(1);
  const auto dt = k_in.scalar_type();

  auto k = k_in.contiguous(), v = v_in.contiguous();
  auto cos = cos_in.to(dt).contiguous(), sin = sin_in.to(dt).contiguous();
  auto positions = positions_in.to(at::kInt).contiguous();
  auto slot_mapping = slot_mapping_in.to(at::kLong).contiguous();
  // Copy the existing caches through; the insert overwrites only the slot rows.
  auto key_out = key_cache_in.to(dt).contiguous().clone();
  auto value_out = value_cache_in.to(dt).contiguous().clone();
  tk_encode([&](TorchEncoder& e) {
    tk::launch_rope_kv_insert(e, k, v, cos, sin, positions, slot_mapping, key_out, value_out,
                              num_tokens * num_kv_heads, num_kv_heads, block_size, D, tk_type_name(k));
  });
  return {key_out, value_out};
}

// Long-context paged decode attention (partition/reduce). GQA/MQA aware.
static at::Tensor paged_attention_v2_mps(
    const at::Tensor& q_in, const at::Tensor& key_cache_in, const at::Tensor& value_cache_in,
    const at::Tensor& block_table_in, const at::Tensor& context_lens_in,
    double scale, int64_t partition_size, int64_t window, double softcap,
    const c10::optional<at::Tensor>& sinks_in) {
  TORCH_CHECK(q_in.device().is_mps(), "paged_attention_v2: q must be an MPS tensor");
  TORCH_CHECK(q_in.dim() == 3, "paged_attention_v2: q must be (B,H,D)");
  TORCH_CHECK(key_cache_in.dim() == 4 && value_cache_in.sizes() == key_cache_in.sizes(),
              "paged_attention_v2: caches must be (num_blocks, block_size, num_kv_heads, D)");
  TORCH_CHECK(key_cache_in.size(3) == q_in.size(2),
              "paged_attention_v2: q head_size must match caches");
  TORCH_CHECK(key_cache_in.size(2) > 0 && q_in.size(1) % key_cache_in.size(2) == 0,
              "paged_attention_v2: num_q_heads must be a positive multiple of num_kv_heads");
  TORCH_CHECK(q_in.scalar_type() == key_cache_in.scalar_type() &&
                  q_in.scalar_type() == value_cache_in.scalar_type() && tk_is_float_dtype(q_in),
              "paged_attention_v2: q/cache dtype must be float32, float16, or bfloat16");
  TORCH_CHECK(block_table_in.scalar_type() == at::kInt && context_lens_in.scalar_type() == at::kInt,
              "paged_attention_v2: block_table and context_lens must be int32");
  const int B = q_in.size(0), H = q_in.size(1), D = q_in.size(2);
  const int H_KV = key_cache_in.size(2), block_size = key_cache_in.size(1);
  TORCH_CHECK(D == 64 || D == 128, "paged_attention_v2: head_size must be 64 or 128");
  TORCH_CHECK(partition_size > 0 && partition_size % block_size == 0,
              "paged_attention_v2: partition_size must be a positive multiple of block_size");

  auto q = q_in.contiguous();
  auto key_cache = key_cache_in.contiguous();
  auto value_cache = value_cache_in.contiguous();
  auto block_table = block_table_in.contiguous();
  auto context_lens = context_lens_in.contiguous();
  const int max_ctx = static_cast<int>(block_table.size(1)) * block_size;
  const int num_partitions = std::max(1, (max_ctx + static_cast<int>(partition_size) - 1) /
                                             static_cast<int>(partition_size));
  auto f32 = q.options().dtype(at::kFloat);
  auto tmp_out = at::empty({B, H, num_partitions, D}, f32);
  auto max_logits = at::empty({B, H, num_partitions}, f32);
  auto exp_sums = at::empty({B, H, num_partitions}, f32);
  auto out = at::empty_like(q);
  const float scale_f = scale > 0.0 ? static_cast<float>(scale)
                                    : 1.0f / std::sqrt(static_cast<float>(D));
  const bool has_sink = sinks_in.has_value();
  at::Tensor sinks = tmp_out;   // placeholder binding when absent (never read)
  if (has_sink) {
    TORCH_CHECK(sinks_in->dim() == 1 && sinks_in->size(0) == H,
                "paged_attention_v2: sinks must be (H,)");
    sinks = sinks_in->to(at::kFloat).contiguous();
  }
  const std::string tn = tk_type_name(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_paged_attention_partition(
        e, q, key_cache, value_cache, block_table, context_lens, tmp_out, max_logits, exp_sums,
        B, H, H_KV, D, block_size, static_cast<int>(block_table.size(1)), scale_f,
        num_partitions, static_cast<int>(partition_size), static_cast<int>(window),
        static_cast<float>(softcap), tn);
    tk::launch_paged_attention_reduce(
        e, tmp_out, max_logits, exp_sums, out, B, H, D, num_partitions, sinks,
        has_sink ? 1 : 0, tn);
  });
  return out;
}

static at::Tensor cascade_attention_mps(
    const at::Tensor& q_in, const at::Tensor& prefix_k_in, const at::Tensor& prefix_v_in,
    const at::Tensor& key_cache_in, const at::Tensor& value_cache_in,
    const at::Tensor& block_table_in, const at::Tensor& context_lens_in,
    double scale, int64_t partition_size) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.dim() == 3, "cascade_attention: q must be (B,H,D) MPS");
  TORCH_CHECK(prefix_k_in.dim() == 3 && prefix_v_in.sizes() == prefix_k_in.sizes(),
              "cascade_attention: prefix_k/prefix_v must be (prefix_len, num_kv_heads, D)");
  TORCH_CHECK(key_cache_in.dim() == 4 && value_cache_in.sizes() == key_cache_in.sizes(),
              "cascade_attention: caches must be (num_blocks, block_size, num_kv_heads, D)");
  TORCH_CHECK(q_in.scalar_type() == key_cache_in.scalar_type() &&
                  q_in.scalar_type() == prefix_k_in.scalar_type() && tk_is_float_dtype(q_in),
              "cascade_attention: q/prefix/cache must share float32/float16/bfloat16");
  const int B = q_in.size(0), H = q_in.size(1), D = q_in.size(2);
  const int H_KV = key_cache_in.size(2), block_size = key_cache_in.size(1);
  const int prefix_len = prefix_k_in.size(0);
  TORCH_CHECK(D == 64 || D == 128, "cascade_attention: head_size must be 64 or 128");
  TORCH_CHECK(prefix_k_in.size(1) == H_KV, "cascade_attention: prefix/suffix num_kv_heads mismatch");
  TORCH_CHECK(partition_size > 0 && partition_size % block_size == 0,
              "cascade_attention: partition_size must be a positive multiple of block_size");
  auto q = q_in.contiguous();
  auto prefix_k = prefix_k_in.contiguous(), prefix_v = prefix_v_in.contiguous();
  auto key_cache = key_cache_in.contiguous(), value_cache = value_cache_in.contiguous();
  auto block_table = block_table_in.to(at::kInt).contiguous();
  auto context_lens = context_lens_in.to(at::kInt).contiguous();
  const int ps = static_cast<int>(partition_size);
  const int Pp = std::max(1, (prefix_len + ps - 1) / ps);
  const int max_suffix = static_cast<int>(block_table.size(1)) * block_size;
  const int Ps = std::max(1, (max_suffix + ps - 1) / ps);
  auto f32 = q.options().dtype(at::kFloat);
  auto p_tmp = at::empty({B, H, Pp, D}, f32), p_ml = at::empty({B, H, Pp}, f32),
       p_es = at::empty({B, H, Pp}, f32);
  auto s_tmp = at::empty({B, H, Ps, D}, f32), s_ml = at::empty({B, H, Ps}, f32),
       s_es = at::empty({B, H, Ps}, f32);
  auto out = at::empty_like(q);
  const float scale_f = scale > 0.0 ? static_cast<float>(scale)
                                    : 1.0f / std::sqrt(static_cast<float>(D));
  const std::string tn = tk_type_name(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_cascade_prefix_partition(e, q, prefix_k, prefix_v, p_tmp, p_ml, p_es, B, H, H_KV, D,
                                        prefix_len, scale_f, Pp, ps, tn);
    tk::launch_paged_attention_partition(e, q, key_cache, value_cache, block_table, context_lens, s_tmp, s_ml, s_es, B, H, H_KV, D, block_size, static_cast<int>(block_table.size(1)), scale_f, Ps, ps, 0, 0.0f, tn);
  });
  auto tmp = at::cat({p_tmp, s_tmp}, 2);
  auto ml = at::cat({p_ml, s_ml}, 2);
  auto es = at::cat({p_es, s_es}, 2);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_paged_attention_reduce(e, tmp, ml, es, out, B, H, D, Pp + Ps, tmp, 0, tn);
  });
  return out;
}

static at::Tensor cascade_attention_multi_mps(
    const at::Tensor& q_in, const std::vector<at::Tensor>& prefix_ks,
    const std::vector<at::Tensor>& prefix_vs, const at::Tensor& key_cache_in,
    const at::Tensor& value_cache_in, const at::Tensor& block_table_in,
    const at::Tensor& context_lens_in, double scale, int64_t partition_size) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.dim() == 3, "cascade_multi: q must be (B,H,D) MPS");
  TORCH_CHECK(!prefix_ks.empty() && prefix_ks.size() == prefix_vs.size(),
              "cascade_multi: need >=1 matching prefix level");
  const int B = q_in.size(0), H = q_in.size(1), D = q_in.size(2);
  const int H_KV = key_cache_in.size(2), block_size = key_cache_in.size(1);
  TORCH_CHECK(D == 64 || D == 128, "cascade_multi: head_size must be 64 or 128");
  TORCH_CHECK(partition_size > 0 && partition_size % block_size == 0,
              "cascade_multi: partition_size must be a positive multiple of block_size");
  auto q = q_in.contiguous();
  auto key_cache = key_cache_in.contiguous(), value_cache = value_cache_in.contiguous();
  auto block_table = block_table_in.to(at::kInt).contiguous();
  auto context_lens = context_lens_in.to(at::kInt).contiguous();
  const int ps = static_cast<int>(partition_size);
  const int max_suffix = static_cast<int>(block_table.size(1)) * block_size;
  const int Ps = std::max(1, (max_suffix + ps - 1) / ps);
  auto f32 = q.options().dtype(at::kFloat);
  const float scale_f = scale > 0.0 ? static_cast<float>(scale)
                                    : 1.0f / std::sqrt(static_cast<float>(D));
  const std::string tn = tk_type_name(q);
  std::vector<at::Tensor> tmps, mls, ess;
  int total_parts = 0;
  tk_encode([&](TorchEncoder& e) {
    for (size_t i = 0; i < prefix_ks.size(); ++i) {
      auto pk = prefix_ks[i].contiguous(), pv = prefix_vs[i].contiguous();
      const int plen = pk.size(0);
      const int Pp = std::max(1, (plen + ps - 1) / ps);
      auto p_tmp = at::empty({B, H, Pp, D}, f32), p_ml = at::empty({B, H, Pp}, f32),
           p_es = at::empty({B, H, Pp}, f32);
      tk::launch_cascade_prefix_partition(e, q, pk, pv, p_tmp, p_ml, p_es, B, H, H_KV, D, plen,
                                          scale_f, Pp, ps, tn);
      tmps.push_back(p_tmp); mls.push_back(p_ml); ess.push_back(p_es);
      total_parts += Pp;
    }
    auto s_tmp = at::empty({B, H, Ps, D}, f32), s_ml = at::empty({B, H, Ps}, f32),
         s_es = at::empty({B, H, Ps}, f32);
    tk::launch_paged_attention_partition(e, q, key_cache, value_cache, block_table, context_lens, s_tmp, s_ml, s_es, B, H, H_KV, D, block_size, static_cast<int>(block_table.size(1)), scale_f, Ps, ps, 0, 0.0f, tn);
    tmps.push_back(s_tmp); mls.push_back(s_ml); ess.push_back(s_es);
    total_parts += Ps;
  });
  auto tmp = at::cat(tmps, 2), ml = at::cat(mls, 2), es = at::cat(ess, 2);
  auto out = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_paged_attention_reduce(e, tmp, ml, es, out, B, H, D, total_parts, tmp, 0, tn);
  });
  return out;
}

static at::Tensor cascade_attention_fp8_mps(
    const at::Tensor& q_in, const at::Tensor& pk_in, const at::Tensor& pv_in,
    const at::Tensor& key_cache_in, const at::Tensor& value_cache_in, const at::Tensor& bt_in,
    const at::Tensor& cl_in, const at::Tensor& ks_in, const at::Tensor& vs_in, double scale,
    int64_t partition_size, int64_t fmt) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.dim() == 3, "cascade_fp8: q must be (B,H,D) MPS");
  TORCH_CHECK(pk_in.scalar_type() == at::kByte && pv_in.scalar_type() == at::kByte,
              "cascade_fp8: prefix_k/prefix_v must be uint8 (fp8 codes)");
  const int B = q_in.size(0), H = q_in.size(1), D = q_in.size(2);
  const int H_KV = key_cache_in.size(2), block_size = key_cache_in.size(1);
  const int prefix_len = pk_in.size(0);
  TORCH_CHECK(D == 64 || D == 128, "cascade_fp8: head_size must be 64 or 128");
  TORCH_CHECK(partition_size > 0 && partition_size % block_size == 0,
              "cascade_fp8: partition_size must be a positive multiple of block_size");
  auto q = q_in.contiguous();
  auto pk = pk_in.contiguous(), pv = pv_in.contiguous();
  auto key_cache = key_cache_in.contiguous(), value_cache = value_cache_in.contiguous();
  auto bt = bt_in.to(at::kInt).contiguous();
  auto cl = cl_in.to(at::kInt).contiguous();
  auto ks = ks_in.to(at::kFloat).contiguous(), vs = vs_in.to(at::kFloat).contiguous();
  const int ps = static_cast<int>(partition_size);
  const int Pp = std::max(1, (prefix_len + ps - 1) / ps);
  const int max_suffix = static_cast<int>(bt.size(1)) * block_size;
  const int Ps = std::max(1, (max_suffix + ps - 1) / ps);
  auto f32 = q.options().dtype(at::kFloat);
  auto p_tmp = at::empty({B, H, Pp, D}, f32), p_ml = at::empty({B, H, Pp}, f32),
       p_es = at::empty({B, H, Pp}, f32);
  auto s_tmp = at::empty({B, H, Ps, D}, f32), s_ml = at::empty({B, H, Ps}, f32),
       s_es = at::empty({B, H, Ps}, f32);
  const float scale_f = scale > 0.0 ? static_cast<float>(scale)
                                     : 1.0f / std::sqrt(static_cast<float>(D));
  const std::string tn = tk_type_name(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_cascade_prefix_partition_fp8(e, q, pk, pv, p_tmp, p_ml, p_es, B, H, H_KV, D,
                                            prefix_len, scale_f, Pp, ps, ks, vs,
                                            static_cast<int>(fmt), tn);
    tk::launch_paged_attention_partition(e, q, key_cache, value_cache, bt, cl, s_tmp, s_ml, s_es, B, H, H_KV, D, block_size, static_cast<int>(bt.size(1)), scale_f, Ps, ps, 0, 0.0f, tn);
  });
  auto tmp = at::cat({p_tmp, s_tmp}, 2), ml = at::cat({p_ml, s_ml}, 2), es = at::cat({p_es, s_es}, 2);
  auto out = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_paged_attention_reduce(e, tmp, ml, es, out, B, H, D, Pp + Ps, tmp, 0, tn);
  });
  return out;
}

// Long-context paged decode over an fp8 (uint8) cache, dequantized on read with per-head scales.
static at::Tensor paged_attention_v2_fp8_mps(
    const at::Tensor& q_in, const at::Tensor& key_cache_in, const at::Tensor& value_cache_in,
    const at::Tensor& block_table_in, const at::Tensor& context_lens_in,
    const at::Tensor& k_scale_in, const at::Tensor& v_scale_in,
    double scale, int64_t partition_size, int64_t fmt, int64_t window, double softcap,
    const c10::optional<at::Tensor>& sinks_in) {
  TORCH_CHECK(q_in.device().is_mps() && tk_is_float_dtype(q_in), "paged_attention_v2_fp8: q must be float MPS");
  TORCH_CHECK(q_in.dim() == 3, "paged_attention_v2_fp8: q must be (B,H,D)");
  TORCH_CHECK(key_cache_in.dim() == 4 && value_cache_in.sizes() == key_cache_in.sizes(),
              "paged_attention_v2_fp8: caches must be (num_blocks, block_size, num_kv_heads, D)");
  TORCH_CHECK(key_cache_in.scalar_type() == at::kByte && value_cache_in.scalar_type() == at::kByte,
              "paged_attention_v2_fp8: caches must be uint8 (fp8 codes)");
  TORCH_CHECK(key_cache_in.size(3) == q_in.size(2), "paged_attention_v2_fp8: head_size mismatch");
  TORCH_CHECK(key_cache_in.size(2) > 0 && q_in.size(1) % key_cache_in.size(2) == 0,
              "paged_attention_v2_fp8: num_q_heads must be a positive multiple of num_kv_heads");
  TORCH_CHECK(block_table_in.scalar_type() == at::kInt && context_lens_in.scalar_type() == at::kInt,
              "paged_attention_v2_fp8: block_table and context_lens must be int32");
  const int B = q_in.size(0), H = q_in.size(1), D = q_in.size(2);
  const int H_KV = key_cache_in.size(2), block_size = key_cache_in.size(1);
  TORCH_CHECK(D == 64 || D == 128, "paged_attention_v2_fp8: head_size must be 64 or 128");
  TORCH_CHECK(partition_size > 0 && partition_size % block_size == 0,
              "paged_attention_v2_fp8: partition_size must be a positive multiple of block_size");
  TORCH_CHECK(k_scale_in.dim() == 1 && k_scale_in.size(0) == H_KV && v_scale_in.sizes() == k_scale_in.sizes(),
              "paged_attention_v2_fp8: k_scale/v_scale must be (num_kv_heads,)");

  auto q = q_in.contiguous();
  auto key_cache = key_cache_in.contiguous();
  auto value_cache = value_cache_in.contiguous();
  auto block_table = block_table_in.contiguous();
  auto context_lens = context_lens_in.contiguous();
  auto ks = k_scale_in.to(at::kFloat).contiguous(), vs = v_scale_in.to(at::kFloat).contiguous();
  const int max_ctx = static_cast<int>(block_table.size(1)) * block_size;
  const int num_partitions = std::max(1, (max_ctx + static_cast<int>(partition_size) - 1) /
                                             static_cast<int>(partition_size));
  auto f32 = q.options().dtype(at::kFloat);
  auto tmp_out = at::empty({B, H, num_partitions, D}, f32);
  auto max_logits = at::empty({B, H, num_partitions}, f32);
  auto exp_sums = at::empty({B, H, num_partitions}, f32);
  auto out = at::empty_like(q);
  const float scale_f = scale > 0.0 ? static_cast<float>(scale)
                                    : 1.0f / std::sqrt(static_cast<float>(D));
  const bool has_sink = sinks_in.has_value();
  at::Tensor sinks = tmp_out;   // placeholder binding when absent (never read)
  if (has_sink) {
    TORCH_CHECK(sinks_in->dim() == 1 && sinks_in->size(0) == H,
                "paged_attention_v2_fp8: sinks must be (H,)");
    sinks = sinks_in->to(at::kFloat).contiguous();
  }
  const std::string tn = tk_type_name(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_paged_attention_partition_fp8(
        e, q, key_cache, value_cache, block_table, context_lens, tmp_out, max_logits, exp_sums,
        B, H, H_KV, D, block_size, static_cast<int>(block_table.size(1)), scale_f,
        num_partitions, static_cast<int>(partition_size), ks, vs, static_cast<int>(fmt),
        static_cast<int>(window), static_cast<float>(softcap), tn);
    tk::launch_paged_attention_reduce(
        e, tmp_out, max_logits, exp_sums, out, B, H, D, num_partitions, sinks,
        has_sink ? 1 : 0, tn);
  });
  return out;
}

// MoE routing: top-k experts + renormalized softmax weights. Returns (ids int32, weights f32).
static std::tuple<at::Tensor, at::Tensor> moe_route_topk_mps(const at::Tensor& logits_in, int64_t k) {
  TORCH_CHECK(logits_in.device().is_mps(), "moe_route_topk: logits must be an MPS tensor");
  TORCH_CHECK(logits_in.dim() == 2, "moe_route_topk: logits must be (num_tokens, num_experts)");
  TORCH_CHECK(tk_is_float_dtype(logits_in), "moe_route_topk: logits must be float");
  auto logits = logits_in.contiguous();
  const int T = logits.size(0), E = logits.size(1);
  TORCH_CHECK(k > 0 && k <= 16 && k <= E, "moe_route_topk: require 1 <= k <= min(16, num_experts)");
  auto ids = at::empty({T, (int64_t)k}, logits.options().dtype(at::kInt));
  auto weights = at::empty({T, (int64_t)k}, logits.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_route_topk(e, logits, ids, weights, T, E, static_cast<int>(k), tk_type_name(logits));
  });
  return {ids, weights};
}

// MoE permute: group T*k routing rows by expert. Returns (sorted_row_idx, offsets, inv_idx).
static std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_permute_mps(
    const at::Tensor& topk_ids_in, int64_t num_experts) {
  TORCH_CHECK(topk_ids_in.device().is_mps(), "moe_permute: topk_ids must be an MPS tensor");
  TORCH_CHECK(topk_ids_in.dim() == 2, "moe_permute: topk_ids must be (num_tokens, k)");
  TORCH_CHECK(num_experts > 0, "moe_permute: num_experts must be positive");
  auto ids = topk_ids_in.to(at::kInt).contiguous();
  const int T = ids.size(0), K = ids.size(1), TK = T * K;
  auto opt = ids.options();
  auto sorted = at::empty({TK}, opt);
  auto offsets = at::empty({num_experts + 1}, opt);
  auto inv = at::empty({TK}, opt);
  auto counts = at::empty({num_experts}, opt);
  auto cursor = at::empty({num_experts}, opt);
  const int E = static_cast<int>(num_experts);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_zero_i32(e, counts, E);
    tk::launch_moe_histogram(e, ids, counts, TK);
    tk::launch_moe_scan_offsets(e, counts, offsets, cursor, E);
    tk::launch_moe_scatter(e, ids, cursor, sorted, inv, TK);
  });
  return {sorted, offsets, inv};
}

// MoE padded schedule: 32-row-padded per-expert segments for the grouped GEMMs.
// Returns (expert_of_tile, gather_idx, inv_pad, off_pad); -1 sentinels mark pad tiles/rows.
static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> moe_pad_schedule_mps(
    const at::Tensor& sorted_in, const at::Tensor& offsets_in, int64_t k) {
  TORCH_CHECK(sorted_in.device().is_mps(), "moe_pad_schedule: sorted_row_idx must be MPS");
  TORCH_CHECK(sorted_in.dim() == 1 && offsets_in.dim() == 1,
              "moe_pad_schedule: sorted_row_idx (TK,), offsets (E+1,)");
  TORCH_CHECK(k > 0, "moe_pad_schedule: k must be positive");
  auto sorted = sorted_in.to(at::kInt).contiguous();
  auto offsets = offsets_in.to(at::kInt).contiguous();
  const int TK = sorted.size(0);
  const int E = static_cast<int>(offsets.size(0)) - 1;
  const int total_pad_max = ((TK + 31 * E + 31) / 32) * 32;
  const int max_tiles = total_pad_max / 32;
  auto opt = sorted.options();
  auto expert_of_tile = at::empty({max_tiles}, opt);
  auto gather_idx = at::empty({total_pad_max}, opt);
  auto inv_pad = at::empty({TK}, opt);
  auto off_pad = at::empty({E + 1}, opt);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_pad_offsets(e, offsets, off_pad, expert_of_tile, gather_idx,
                               E, max_tiles, total_pad_max);
    tk::launch_moe_pad_scatter(e, sorted, offsets, off_pad, gather_idx, inv_pad,
                               TK, E, static_cast<int>(k));
  });
  return {expert_of_tile, gather_idx, inv_pad, off_pad};
}

// MoE gather: out[p, :] = x[gather_idx[p], :] (zeros where gather_idx[p] < 0).
static at::Tensor moe_gather_mps(const at::Tensor& x_in, const at::Tensor& gather_idx_in) {
  TORCH_CHECK(x_in.device().is_mps(), "moe_gather: x must be an MPS tensor");
  TORCH_CHECK(x_in.dim() == 2 && gather_idx_in.dim() == 1, "moe_gather: x (T,H), gather_idx (P,)");
  TORCH_CHECK(x_in.scalar_type() == at::kFloat || x_in.scalar_type() == at::kBFloat16,
              "moe_gather: x must be float32 or bfloat16");
  auto x = x_in.contiguous();
  auto gi = gather_idx_in.to(at::kInt).contiguous();
  const int H = x.size(1);
  const int total_pad_max = gi.size(0);
  auto out = at::empty({total_pad_max, H}, x.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_gather(e, x, gi, out, H, total_pad_max, tk_type_name(x));
  });
  return out;
}

// Fused grouped expert GEMM: out = permuted_input @ W[expert]. Returns (total_rows, H).
static at::Tensor moe_grouped_gemm_mps(const at::Tensor& pi_in, const at::Tensor& W_in,
                                       const at::Tensor& eot_in) {
  TORCH_CHECK(pi_in.device().is_mps(), "moe_grouped_gemm: permuted_input must be an MPS tensor");
  TORCH_CHECK(pi_in.dim() == 2 && W_in.dim() == 3, "moe_grouped_gemm: shapes (total_rows,H) / (E,H,H)");
  TORCH_CHECK(tk_is_float_dtype(pi_in), "moe_grouped_gemm: dtype must be float32/float16/bfloat16");
  const int total_rows = pi_in.size(0), H = pi_in.size(1);
  TORCH_CHECK(total_rows % 32 == 0 && H % 32 == 0, "moe_grouped_gemm: total_rows,H must be %32");
  TORCH_CHECK(W_in.size(1) == H && W_in.size(2) == H, "moe_grouped_gemm: W must be (E,H,H)");
  auto pi = pi_in.contiguous(), W = W_in.contiguous();
  auto eot = eot_in.to(at::kInt).contiguous();
  auto out = at::empty({total_rows, H}, pi.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_grouped_gemm(e, out, pi, W, eot, total_rows, H, tk_type_name(pi));
  });
  return out;
}

static at::Tensor moe_grouped_gemm_rect_mps(const at::Tensor& A_in, const at::Tensor& W_in,
                                            const at::Tensor& eot_in) {
  TORCH_CHECK(A_in.device().is_mps() && tk_is_float_dtype(A_in), "moe_grouped_gemm_rect: A float MPS");
  TORCH_CHECK(A_in.dim() == 2 && W_in.dim() == 3, "moe_grouped_gemm_rect: A (rows,K), W (E,K,N)");
  const int total_rows = A_in.size(0), K_dim = A_in.size(1), N_out = W_in.size(2);
  TORCH_CHECK(total_rows % 32 == 0 && K_dim % 16 == 0 && N_out % 32 == 0,
              "moe_grouped_gemm_rect: rows%32, K%16, N%32");
  TORCH_CHECK(W_in.size(1) == K_dim, "moe_grouped_gemm_rect: W must be (E,K_dim,N_out)");
  auto A = A_in.contiguous(), W = W_in.contiguous();
  auto eot = eot_in.to(at::kInt).contiguous();
  auto out = at::empty({total_rows, N_out}, A.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_grouped_gemm_rect(e, out, A, W, eot, total_rows, K_dim, N_out, tk_type_name(A));
  });
  return out;
}

static at::Tensor moe_grouped_gemm_swiglu_mps(const at::Tensor& A_in, const at::Tensor& W1_in,
                                              const at::Tensor& eot_in) {
  TORCH_CHECK(A_in.device().is_mps() && tk_is_float_dtype(A_in), "moe_grouped_gemm_swiglu: A float MPS");
  TORCH_CHECK(A_in.dim() == 2 && W1_in.dim() == 3, "moe_grouped_gemm_swiglu: A (rows,H), W1 (E,H,2*inter)");
  const int total_rows = A_in.size(0), H = A_in.size(1);
  TORCH_CHECK(W1_in.size(2) % 2 == 0, "moe_grouped_gemm_swiglu: W1 last dim must be 2*inter");
  const int inter = W1_in.size(2) / 2;
  TORCH_CHECK(total_rows % 32 == 0 && H % 16 == 0 && inter % 32 == 0,
              "moe_grouped_gemm_swiglu: rows%32, H%16, inter%32");
  TORCH_CHECK(W1_in.size(1) == H, "moe_grouped_gemm_swiglu: W1 must be (E,H,2*inter)");
  auto A = A_in.contiguous(), W1 = W1_in.contiguous();
  auto eot = eot_in.to(at::kInt).contiguous();
  auto out = at::empty({total_rows, inter}, A.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_grouped_gemm_swiglu(e, out, A, W1, eot, total_rows, H, inter, tk_type_name(A));
  });
  return out;
}

// marginal utilities: tau_tail, packbits, segment_packbits, permute_cols.
static at::Tensor tau_tail_mps(const at::Tensor& qkv_in, const at::Tensor& gate_in,
                               const at::Tensor& tau_in, const at::Tensor& pos_in,
                               int64_t n_heads, int64_t head_dim) {
  TORCH_CHECK(qkv_in.device().is_mps() && qkv_in.dim() == 2 && tk_is_float_dtype(qkv_in),
              "tau_tail: qkv must be a float (T, 3*q_dim) MPS tensor");
  const int T = qkv_in.size(0), q_dim = qkv_in.size(1) / 3;
  TORCH_CHECK(n_heads * head_dim == q_dim, "tau_tail: n_heads*head_dim must equal q_dim");
  auto out = qkv_in.contiguous().clone();
  auto gate = gate_in.to(qkv_in.scalar_type()).contiguous();
  auto tau = tau_in.to(qkv_in.scalar_type()).contiguous();
  auto pos = pos_in.to(at::kInt).contiguous();
  tk_encode([&](TorchEncoder& e) {
    tk::launch_tau_tail(e, out, gate, tau, pos, T * (int)n_heads * (int)head_dim,
                        (int)n_heads, (int)head_dim, q_dim, tk_type_name(qkv_in));
  });
  return out;
}
static at::Tensor packbits_mps(const at::Tensor& x_in, bool bit_order_big) {
  TORCH_CHECK(x_in.device().is_mps(), "packbits: x must be an MPS tensor");
  auto x = x_in.to(at::kByte).contiguous().view({-1});
  const int n = x.numel();
  auto out = at::empty({(n + 7) / 8}, x.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_packbits(e, x, out, n, bit_order_big ? 1 : 0);
  });
  return out;
}
static at::Tensor segment_packbits_mps(const at::Tensor& x_in, const at::Tensor& in_ind,
                                       const at::Tensor& out_ind, int64_t total_output_bytes,
                                       bool bit_order_big) {
  TORCH_CHECK(x_in.device().is_mps(), "segment_packbits: x must be an MPS tensor");
  auto x = x_in.to(at::kByte).contiguous().view({-1});
  auto ii = in_ind.to(at::kInt).contiguous(), oi = out_ind.to(at::kInt).contiguous();
  const int num_segments = ii.numel() - 1;
  auto out = at::empty({total_output_bytes}, x.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_segment_packbits(e, x, ii, oi, out, num_segments, (int)total_output_bytes,
                                bit_order_big ? 1 : 0);
  });
  return out;
}
static at::Tensor permute_cols_mps(const at::Tensor& x_in, const at::Tensor& perm_in) {
  TORCH_CHECK(x_in.device().is_mps() && x_in.dim() == 2 && x_in.element_size() == 2,
              "permute_cols: x must be a 2-D 16-bit MPS tensor");
  auto x = x_in.contiguous();
  auto perm = perm_in.to(at::kInt).contiguous();
  const int rows = x.size(0), cols = x.size(1);
  auto out = at::empty_like(x);
  auto xv = x.view(at::kShort), ov = out.view(at::kShort);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_permute_cols(e, xv, perm, ov, rows, cols);
  });
  return out;
}

// TurboQuant KV codec (encode + decode).
static std::vector<at::Tensor> tq_encode_mps(
    const at::Tensor& key_in, const at::Tensor& value_in, const at::Tensor& kc_in,
    const at::Tensor& vc_in, const at::Tensor& ks_in, const at::Tensor& vs_in,
    const at::Tensor& kz_in, const at::Tensor& slots_in, const at::Tensor& cent_in,
    const at::Tensor& signs_in, int64_t block_size, int64_t k_bits, bool k_signed,
    int64_t v_bits) {
  TORCH_CHECK(key_in.device().is_mps() && key_in.dim() == 3 && tk_is_float_dtype(key_in),
              "tq_encode: key must be a float (tokens, num_kv_heads, head_size) MPS tensor");
  const int num_tokens = key_in.size(0), num_kv_heads = key_in.size(1), hs = key_in.size(2);
  TORCH_CHECK(hs == 64 || hs == 128 || hs == 256, "tq_encode: head_size must be 64/128/256");
  auto key = key_in.contiguous(), value = value_in.to(key_in.scalar_type()).contiguous();
  auto kc = kc_in.to(at::kByte).contiguous().clone();
  auto vc = vc_in.to(at::kByte).contiguous().clone();
  auto ks = ks_in.to(at::kHalf).contiguous().clone();
  auto vs = vs_in.to(at::kHalf).contiguous().clone();
  auto kz = kz_in.to(at::kHalf).contiguous().clone();
  auto slots = slots_in.to(at::kInt).contiguous();
  auto cent = cent_in.to(at::kFloat).contiguous(), signs = signs_in.to(at::kFloat).contiguous();
  tk_encode([&](TorchEncoder& e) {
    tk::launch_tq_encode(e, key, value, kc, vc, ks, vs, kz, slots, cent, signs, num_tokens,
                         num_kv_heads, hs, (int)block_size, (int)k_bits, k_signed ? 1 : 0,
                         (int)v_bits, tk_type_name(key));
  });
  return {kc, vc, ks, vs, kz};
}

static std::vector<at::Tensor> tq_decode_mps(
    const at::Tensor& kc_in, const at::Tensor& vc_in, const at::Tensor& ks_in,
    const at::Tensor& vs_in, const at::Tensor& kz_in, const at::Tensor& slots_in,
    const at::Tensor& cent_in, const at::Tensor& signs_in, int64_t num_kv_heads,
    int64_t head_size, int64_t block_size, int64_t k_bits, bool k_signed, int64_t v_bits) {
  TORCH_CHECK(kc_in.device().is_mps(), "tq_decode: caches must be MPS tensors");
  const int n = slots_in.size(0);
  auto kc = kc_in.to(at::kByte).contiguous(), vc = vc_in.to(at::kByte).contiguous();
  auto ks = ks_in.to(at::kHalf).contiguous(), vs = vs_in.to(at::kHalf).contiguous();
  auto kz = kz_in.to(at::kHalf).contiguous();
  auto slots = slots_in.to(at::kInt).contiguous();
  auto cent = cent_in.to(at::kFloat).contiguous(), signs = signs_in.to(at::kFloat).contiguous();
  auto opts = at::TensorOptions().dtype(at::kFloat).device(kc.device());
  auto k_out = at::empty({n, num_kv_heads, head_size}, opts);
  auto v_out = at::empty({n, num_kv_heads, head_size}, opts);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_tq_decode(e, kc, vc, ks, vs, kz, slots, cent, signs, k_out, v_out, n,
                         (int)num_kv_heads, (int)head_size, (int)block_size, (int)k_bits,
                         k_signed ? 1 : 0, (int)v_bits, "float32");
  });
  return {k_out, v_out};
}

// DeepSeek-V3.2 indexer K quant-and-cache (+ gather).
static std::vector<at::Tensor> indexer_k_quant_and_cache_mps(
    const at::Tensor& k_in, const at::Tensor& slot_in, const at::Tensor& code_in,
    const at::Tensor& scale_in, int64_t quant_block_size, bool ue8m0) {
  TORCH_CHECK(k_in.device().is_mps() && k_in.dim() == 2 && tk_is_float_dtype(k_in),
              "indexer_k_quant_and_cache: k must be a float (tokens, head_dim) MPS tensor");
  const int num_tokens = k_in.size(0), head_dim = k_in.size(1);
  const int nq = (head_dim + (int)quant_block_size - 1) / (int)quant_block_size;
  auto k = k_in.contiguous();
  auto slot = slot_in.to(at::kInt).contiguous();
  auto code = code_in.to(at::kByte).contiguous().clone();
  auto scale = scale_in.to(at::kFloat).contiguous().clone();
  tk_encode([&](TorchEncoder& e) {
    tk::launch_indexer_k_quant_and_cache(e, k, slot, code, scale, num_tokens, head_dim, nq,
                                         (int)quant_block_size, ue8m0 ? 1 : 0, tk_type_name(k));
  });
  return {code, scale};
}

static at::Tensor indexer_k_gather_mps(const at::Tensor& code_in, const at::Tensor& scale_in,
                                       const at::Tensor& slots_in, int64_t head_dim,
                                       int64_t quant_block_size) {
  TORCH_CHECK(code_in.device().is_mps(), "indexer_k_gather: cache must be MPS");
  const int n = slots_in.size(0);
  const int nq = (head_dim + (int)quant_block_size - 1) / (int)quant_block_size;
  auto code = code_in.to(at::kByte).contiguous(), scale = scale_in.to(at::kFloat).contiguous();
  auto slots = slots_in.to(at::kInt).contiguous();
  auto opts = at::TensorOptions().dtype(at::kBFloat16).device(code.device());
  auto k_out = at::empty({n, head_dim}, opts);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_indexer_k_gather(e, code, scale, slots, k_out, n, (int)head_dim, nq,
                                (int)quant_block_size, "bfloat16");
  });
  return k_out;
}

// MInference decode block-mask builder.
static at::Tensor minference_block_mask_mps(
    const at::Tensor& vert_in, const at::Tensor& slash_in, const at::Tensor& lens_in,
    int64_t max_blocks, int64_t block_size, int64_t vertical_topk, int64_t slash_topk,
    int64_t last_n_blocks) {
  TORCH_CHECK(vert_in.device().is_mps() && vert_in.dim() == 3 && slash_in.dim() == 3 &&
                  vert_in.size(0) == slash_in.size(0) && vert_in.size(1) == slash_in.size(1),
              "minference_block_mask: vertical/slash must be (B, H, nnz) MPS tensors");
  auto vert = vert_in.to(at::kInt).contiguous(), slash = slash_in.to(at::kInt).contiguous();
  auto lens = lens_in.to(at::kInt).contiguous();
  const int B = vert.size(0), H = vert.size(1);
  auto mask = at::empty({B, H, max_blocks}, vert.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_minference_block_mask(e, vert, slash, lens, mask, B, H, (int)vert.size(2),
                                     (int)slash.size(2), (int)vertical_topk, (int)slash_topk,
                                     (int)block_size, (int)max_blocks, (int)last_n_blocks);
  });
  return mask;
}

// sampler-zoo logit/prob transforms (one simdgroup per row; see sampling_transforms.metal).
static void st_shape(const at::Tensor& x, int& rows, int& V, const char* name) {
  TORCH_CHECK(x.device().is_mps() && x.dim() == 2 && tk_is_float_dtype(x),
              std::string(name) + ": input must be a float (rows, V) MPS tensor");
  rows = x.size(0); V = x.size(1);
}
static float st_invtemp(double t) { return 1.0f / std::max((float)t, 1e-6f); }

static at::Tensor quadratic_transform_mps(const at::Tensor& x_in, double factor, double curve,
                                          double temperature) {
  int rows, V; st_shape(x_in, rows, V, "quadratic_transform");
  auto x = x_in.contiguous(); auto out = at::empty_like(x);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_quadratic_transform(e, x, out, rows, V, (float)factor, (float)curve,
                                   st_invtemp(temperature), tk_type_name(x));
  });
  return out;
}
static at::Tensor logit_mask1_mps(const char* kernel, const at::Tensor& x_in, double p0,
                                  double temperature) {
  int rows, V; st_shape(x_in, rows, V, kernel);
  auto x = x_in.contiguous(); auto out = at::empty_like(x);
  const std::string k = std::string(kernel) + "_" + tk_type_name(x);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_logit_mask1(e, k, x, out, rows, V, (float)p0, st_invtemp(temperature));
  });
  return out;
}
static at::Tensor top_nsigma_mask_mps(const at::Tensor& x, double p, double t) {
  return logit_mask1_mps("top_nsigma_mask", x, p, t); }
static at::Tensor top_a_mask_mps(const at::Tensor& x, double p, double t) {
  return logit_mask1_mps("top_a_mask", x, p, t); }
static at::Tensor epsilon_cutoff_mask_mps(const at::Tensor& x, double p, double t) {
  return logit_mask1_mps("epsilon_cutoff_mask", x, p, t); }
static at::Tensor eta_cutoff_mask_mps(const at::Tensor& x, double p, double t) {
  return logit_mask1_mps("eta_cutoff_mask", x, p, t); }
static at::Tensor xtc_mask_mps(const at::Tensor& x_in, double threshold, double probability,
                               int64_t seed, double temperature) {
  int rows, V; st_shape(x_in, rows, V, "xtc_mask");
  auto x = x_in.contiguous(); auto out = at::empty_like(x);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_xtc_mask(e, x, out, rows, V, (float)threshold, (float)probability,
                        st_invtemp(temperature), (uint32_t)seed, tk_type_name(x));
  });
  return out;
}
static at::Tensor skew_transform_mps(const at::Tensor& x_in, double skew) {
  int rows, V; st_shape(x_in, rows, V, "skew_transform");
  auto x = x_in.contiguous(); auto out = at::empty_like(x);
  const std::string k = "skew_transform_" + tk_type_name(x);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_prob_transform1(e, k, x, out, rows, V, (float)skew);
  });
  return out;
}
static at::Tensor top_k_renorm_mps(const at::Tensor& x_in, int64_t k) {
  int rows, V; st_shape(x_in, rows, V, "top_k_renorm");
  TORCH_CHECK(k > 0 && k <= 64, "top_k_renorm: 1 <= k <= 64");
  auto x = x_in.contiguous(); auto out = at::empty_like(x);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_top_k_renorm(e, x, out, rows, V, (int)k, tk_type_name(x));
  });
  return out;
}
static at::Tensor top_p_renorm_mps(const at::Tensor& x_in, double p) {
  int rows, V; st_shape(x_in, rows, V, "top_p_renorm");
  auto x = x_in.contiguous(); auto out = at::empty_like(x);
  const std::string k = "top_p_renorm_probs_" + tk_type_name(x);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_prob_transform1(e, k, x, out, rows, V, (float)p);
  });
  return out;
}
static at::Tensor no_repeat_ngram_mask_mps(const at::Tensor& x_in, const at::Tensor& prev_in,
                                           const at::Tensor& lens_in, int64_t ngram,
                                           double temperature) {
  int rows, V; st_shape(x_in, rows, V, "no_repeat_ngram_mask");
  TORCH_CHECK(ngram >= 2, "no_repeat_ngram_mask: ngram_size >= 2");
  auto x = x_in.contiguous();
  auto prev = prev_in.to(at::kInt).contiguous(), lens = lens_in.to(at::kInt).contiguous();
  auto out = at::empty_like(x);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_no_repeat_ngram_mask(e, x, prev, lens, out, rows, V, (int)prev.size(1),
                                    (int)ngram, st_invtemp(temperature), tk_type_name(x));
  });
  return out;
}
static at::Tensor dry_penalty_mps(const at::Tensor& x_in, const at::Tensor& prev_in,
                                  const at::Tensor& lens_in, const at::Tensor& breakers_in,
                                  double multiplier, double base, int64_t allowed,
                                  int64_t range, int64_t max_ngram, int64_t max_occ,
                                  int64_t early_exit, double temperature) {
  int rows, V; st_shape(x_in, rows, V, "dry_penalty");
  auto x = x_in.contiguous();
  auto prev = prev_in.to(at::kInt).contiguous(), lens = lens_in.to(at::kInt).contiguous();
  auto brk = breakers_in.to(at::kInt).contiguous();
  auto out = at::empty_like(x);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_dry_penalty(e, x, prev, lens, brk, out, rows, V, (int)prev.size(1),
                           (int)brk.size(0), (float)multiplier, (float)base, (int)allowed,
                           (int)range, (int)max_ngram, (int)max_occ, (int)early_exit,
                           st_invtemp(temperature), tk_type_name(x));
  });
  return out;
}

// per-group / azp activation quantizers + azp-corrected W8A8 GEMM.
static std::tuple<at::Tensor, at::Tensor> quantize_per_group_fp8_mps(
    const at::Tensor& x_in, int64_t group_size, bool ue8m0) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in), "quantize_per_group_fp8: float MPS");
  auto x = x_in.contiguous();
  const int D = x.size(-1);
  TORCH_CHECK(group_size > 0 && group_size % 4 == 0 && D % group_size == 0,
              "quantize_per_group_fp8: bad group_size");
  const int rows = x.numel() / D;
  auto codes = at::empty_like(x, x.options().dtype(at::kByte));
  auto sshape = x.sizes().vec(); sshape.back() = D / group_size;
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_quantize_per_group_fp8(e, x, codes, scale, rows, D, (int)group_size,
                                      ue8m0 ? 1 : 0, tk_type_name(x));
  });
  return {codes, scale};
}

static std::tuple<at::Tensor, at::Tensor> quantize_per_group_int8_mps(
    const at::Tensor& x_in, int64_t group_size) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in), "quantize_per_group_int8: float MPS");
  auto x = x_in.contiguous();
  const int D = x.size(-1);
  TORCH_CHECK(group_size > 0 && group_size % 4 == 0 && D % group_size == 0,
              "quantize_per_group_int8: bad group_size");
  const int rows = x.numel() / D;
  auto codes = at::empty_like(x, x.options().dtype(at::kChar));
  auto sshape = x.sizes().vec(); sshape.back() = D / group_size;
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_quantize_per_group_int8(e, x, codes, scale, rows, D, (int)group_size,
                                       tk_type_name(x));
  });
  return {codes, scale};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> quantize_per_token_int8_azp_mps(
    const at::Tensor& x_in) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in),
              "quantize_per_token_int8_azp: float MPS");
  auto x = x_in.contiguous();
  const int D = x.size(-1);
  const int rows = x.numel() / D;
  auto codes = at::empty_like(x, x.options().dtype(at::kChar));
  auto sshape = x.sizes().vec(); sshape.pop_back();
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  auto azp = at::empty(sshape, x.options().dtype(at::kInt));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_quantize_per_token_int8_azp(e, x, codes, scale, azp, rows, D, tk_type_name(x));
  });
  return {codes, scale, azp};
}

static at::Tensor qgemm_w8a8_azp_mps(const at::Tensor& wq_in, const at::Tensor& xq_in,
                                     const at::Tensor& ws_in, const at::Tensor& as_in,
                                     const at::Tensor& rsum_in, const at::Tensor& azp_in) {
  TORCH_CHECK(wq_in.device().is_mps() && wq_in.scalar_type() == at::kChar &&
              xq_in.scalar_type() == at::kChar, "qgemm_w8a8_azp: wq/xq int8 MPS");
  auto wq = wq_in.contiguous(), xq = xq_in.contiguous();
  auto ws = ws_in.to(at::kHalf).contiguous(), as = as_in.to(at::kFloat).contiguous();
  auto rsum = rsum_in.to(at::kInt).contiguous(), azp = azp_in.to(at::kInt).contiguous();
  const int N = wq.size(0), K = wq.size(1), M = xq.size(0);
  auto out = at::empty({N, M}, wq.options().dtype(at::kHalf));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_qgemm_w8a8_azp(e, out, wq, xq, ws, as, rsum, azp, N, K, M);
  });
  return out;
}

// GatedDeltaNet delta-rule linear attention (varlen packed + persistent fp32 state pool).
static std::tuple<at::Tensor, at::Tensor> gdn_recur_mps(
    const at::Tensor& q_in, const at::Tensor& k_in, const at::Tensor& v_in,
    const at::Tensor& g_in, const at::Tensor& beta_in, const at::Tensor& pool_in,
    const at::Tensor& cu_in, const at::Tensor& slots_in, bool load_initial) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.dim() == 3 && tk_is_float_dtype(q_in),
              "gdn_recur: q must be a float (total_tokens, Hk, Dk) MPS tensor");
  const int Hk = q_in.size(1), Dk = q_in.size(2);
  const int Hv = v_in.size(1), Dv = v_in.size(2);
  TORCH_CHECK(Dk == 64 || Dk == 128, "gdn_recur: Dk must be 64 or 128");
  TORCH_CHECK(Hv % Hk == 0, "gdn_recur: Hv must be a multiple of Hk");
  auto q = q_in.contiguous(), k = k_in.to(q_in.scalar_type()).contiguous();
  auto v = v_in.to(q_in.scalar_type()).contiguous();
  auto g = g_in.to(q_in.scalar_type()).contiguous();
  auto beta = beta_in.to(q_in.scalar_type()).contiguous();
  auto cu = cu_in.to(at::kInt).contiguous(), slots = slots_in.to(at::kInt).contiguous();
  const int R = cu.size(0) - 1;
  auto pool = pool_in.to(at::kFloat).contiguous().clone();
  auto y = at::empty_like(v);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_gdn_recur(e, q, k, v, g, beta, pool, cu, slots, y, R, Hk, Hv, Dv, Dk,
                         load_initial ? 1 : 0, tk_type_name(q));
  });
  return {y, pool};
}

// Mamba-1 (S6) selective scan (dense + varlen). Functional: state is cloned and the clone
// updated in place; returns (out, new_state).
static std::tuple<at::Tensor, at::Tensor> selective_scan_mps(
    const at::Tensor& u_in, const at::Tensor& delta_in, const at::Tensor& A_in,
    const at::Tensor& B_in, const at::Tensor& C_in, const at::Tensor& state_in,
    const c10::optional<at::Tensor>& D_in, const c10::optional<at::Tensor>& bias_in,
    const c10::optional<at::Tensor>& z_in, bool delta_softplus) {
  TORCH_CHECK(u_in.device().is_mps() && u_in.dim() == 3 && tk_is_float_dtype(u_in),
              "selective_scan: u must be a float (batch, dim, seqlen) MPS tensor");
  const int batch = u_in.size(0), dim = u_in.size(1), seqlen = u_in.size(2);
  TORCH_CHECK(B_in.dim() == 4 && B_in.sizes() == C_in.sizes(),
              "selective_scan: B/C must be (batch, n_groups, dstate, seqlen)");
  const int n_groups = B_in.size(1), dstate = B_in.size(2);
  TORCH_CHECK(dstate <= 256 && dim % n_groups == 0,
              "selective_scan: dstate <= 256, dim %% n_groups == 0");
  auto u = u_in.contiguous(), delta = delta_in.to(u_in.scalar_type()).contiguous();
  auto A = A_in.to(at::kFloat).contiguous();
  auto B = B_in.to(u_in.scalar_type()).contiguous(), C = C_in.to(u_in.scalar_type()).contiguous();
  const bool has_d = D_in.has_value(), has_bias = bias_in.has_value(), has_z = z_in.has_value();
  auto dummy_f = at::zeros({1}, u.options().dtype(at::kFloat));
  auto dummy_io = at::zeros({1}, u.options());
  auto D = has_d ? D_in->to(at::kFloat).contiguous() : dummy_f;
  auto bias = has_bias ? bias_in->to(at::kFloat).contiguous() : dummy_f;
  auto z = has_z ? z_in->to(u_in.scalar_type()).contiguous() : dummy_io;
  auto state = state_in.to(at::kFloat).contiguous().clone();
  auto out = at::empty_like(u);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_selective_scan_dense(e, u, delta, A, B, C, D, bias, z, out, state,
                                    batch, dim, seqlen, dstate, n_groups,
                                    has_d ? 1 : 0, has_bias ? 1 : 0, has_z ? 1 : 0,
                                    delta_softplus ? 1 : 0, tk_type_name(u));
  });
  return {out, state};
}

static std::tuple<at::Tensor, at::Tensor> selective_scan_varlen_mps(
    const at::Tensor& u_in, const at::Tensor& delta_in, const at::Tensor& A_in,
    const at::Tensor& B_in, const at::Tensor& C_in, const at::Tensor& qsl_in,
    const at::Tensor& state_in, const c10::optional<at::Tensor>& D_in,
    const c10::optional<at::Tensor>& bias_in, const c10::optional<at::Tensor>& z_in,
    const c10::optional<at::Tensor>& cidx_in, const c10::optional<at::Tensor>& his_in,
    bool delta_softplus, int64_t null_block_id) {
  TORCH_CHECK(u_in.device().is_mps() && u_in.dim() == 2 && tk_is_float_dtype(u_in),
              "selective_scan_varlen: u must be a float (dim, total_tokens) MPS tensor");
  const int dim = u_in.size(0), total_tokens = u_in.size(1);
  TORCH_CHECK(B_in.dim() == 3 && B_in.sizes() == C_in.sizes(),
              "selective_scan_varlen: B/C must be (n_groups, dstate, total_tokens)");
  const int n_groups = B_in.size(0), dstate = B_in.size(1);
  TORCH_CHECK(dstate <= 256 && dim % n_groups == 0,
              "selective_scan_varlen: dstate <= 256, dim %% n_groups == 0");
  auto u = u_in.contiguous(), delta = delta_in.to(u_in.scalar_type()).contiguous();
  auto A = A_in.to(at::kFloat).contiguous();
  auto B = B_in.to(u_in.scalar_type()).contiguous(), C = C_in.to(u_in.scalar_type()).contiguous();
  auto qsl = qsl_in.to(at::kInt).contiguous();
  const int batch = qsl.size(0) - 1;
  const bool has_d = D_in.has_value(), has_bias = bias_in.has_value(), has_z = z_in.has_value();
  const bool use_ci = cidx_in.has_value(), use_his = his_in.has_value();
  auto dummy_f = at::zeros({1}, u.options().dtype(at::kFloat));
  auto dummy_io = at::zeros({1}, u.options());
  auto dummy_i = at::zeros({1}, u.options().dtype(at::kInt));
  auto dummy_u8 = at::zeros({1}, u.options().dtype(at::kByte));
  auto D = has_d ? D_in->to(at::kFloat).contiguous() : dummy_f;
  auto bias = has_bias ? bias_in->to(at::kFloat).contiguous() : dummy_f;
  auto z = has_z ? z_in->to(u_in.scalar_type()).contiguous() : dummy_io;
  auto cidx = use_ci ? cidx_in->to(at::kInt).contiguous() : dummy_i;
  auto his = use_his ? his_in->to(at::kByte).contiguous() : dummy_u8;
  auto state = state_in.to(at::kFloat).contiguous().clone();
  auto out = at::empty_like(u);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_selective_scan_varlen(e, u, delta, A, B, C, D, bias, z, qsl, cidx, his,
                                     out, state, batch, dim, total_tokens, dstate, n_groups,
                                     has_d ? 1 : 0, has_bias ? 1 : 0, has_z ? 1 : 0,
                                     delta_softplus ? 1 : 0, use_ci ? 1 : 0, use_his ? 1 : 0,
                                     static_cast<int>(null_block_id), tk_type_name(u));
  });
  return {out, state};
}

static std::tuple<at::Tensor, at::Tensor> selective_scan_varlen_apc_mps(
    const at::Tensor& u_in, const at::Tensor& delta_in, const at::Tensor& A_in,
    const at::Tensor& B_in, const at::Tensor& C_in, const at::Tensor& qsl_in,
    const at::Tensor& cidx_in, const at::Tensor& his_in, const at::Tensor& state_in,
    const at::Tensor& bif_in, const at::Tensor& bil_in, const at::Tensor& isi_in,
    const at::Tensor& ccs_in, const at::Tensor& lci_in, int64_t block_size,
    int64_t cache_indices_stride, bool use_chunk_metadata,
    const c10::optional<at::Tensor>& D_in, const c10::optional<at::Tensor>& bias_in,
    const c10::optional<at::Tensor>& z_in, bool delta_softplus, int64_t null_block_id) {
  TORCH_CHECK(u_in.device().is_mps() && u_in.dim() == 2 && tk_is_float_dtype(u_in),
              "selective_scan_varlen_apc: u must be a float (dim, total_tokens) MPS tensor");
  const int dim = u_in.size(0), total_tokens = u_in.size(1);
  const int n_groups = B_in.size(0), dstate = B_in.size(1);
  auto u = u_in.contiguous(), delta = delta_in.to(u_in.scalar_type()).contiguous();
  auto A = A_in.to(at::kFloat).contiguous();
  auto B = B_in.to(u_in.scalar_type()).contiguous(), C = C_in.to(u_in.scalar_type()).contiguous();
  auto qsl = qsl_in.to(at::kInt).contiguous();
  const int batch = qsl.size(0) - 1;
  const bool has_d = D_in.has_value(), has_bias = bias_in.has_value(), has_z = z_in.has_value();
  auto dummy_f = at::zeros({1}, u.options().dtype(at::kFloat));
  auto dummy_io = at::zeros({1}, u.options());
  auto D = has_d ? D_in->to(at::kFloat).contiguous() : dummy_f;
  auto bias = has_bias ? bias_in->to(at::kFloat).contiguous() : dummy_f;
  auto z = has_z ? z_in->to(u_in.scalar_type()).contiguous() : dummy_io;
  auto cidx = cidx_in.to(at::kInt).contiguous(), his = his_in.to(at::kByte).contiguous();
  auto bif = bif_in.to(at::kInt).contiguous(), bil = bil_in.to(at::kInt).contiguous();
  auto isi = isi_in.to(at::kInt).contiguous(), ccs = ccs_in.to(at::kInt).contiguous();
  auto lci = lci_in.to(at::kInt).contiguous();
  auto state = state_in.to(at::kFloat).contiguous().clone();
  auto out = at::empty_like(u);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_selective_scan_varlen_apc(
        e, u, delta, A, B, C, D, bias, z, qsl, cidx, his, out, state, bif, bil, isi, ccs, lci,
        batch, dim, total_tokens, dstate, n_groups, has_d ? 1 : 0, has_bias ? 1 : 0,
        has_z ? 1 : 0, delta_softplus ? 1 : 0, static_cast<int>(null_block_id),
        (int)block_size, (int)cache_indices_stride, use_chunk_metadata ? 1 : 0,
        tk_type_name(u));
  });
  return {out, state};
}

// Fused per-head QK-RMSNorm + RoPE over packed QKV; V heads copied through.
static at::Tensor qk_norm_rope_mps(const at::Tensor& qkv_in, const at::Tensor& qw_in,
                                   const at::Tensor& kw_in, const at::Tensor& cos_in,
                                   const at::Tensor& sin_in, const at::Tensor& pos_in,
                                   int64_t hq, int64_t hk, int64_t hv, double eps,
                                   bool interleaved, bool gemma) {
  TORCH_CHECK(qkv_in.device().is_mps() && qkv_in.scalar_type() == at::kBFloat16 &&
              qkv_in.dim() == 2, "qk_norm_rope: qkv must be a bf16 (T, HT*D) MPS tensor");
  const int HT = static_cast<int>(hq + hk + hv);
  TORCH_CHECK(HT > 0 && qkv_in.size(1) % HT == 0, "qk_norm_rope: head counts must divide width");
  const int T = qkv_in.size(0), D = qkv_in.size(1) / HT;
  TORCH_CHECK(D == 64 || D == 128 || D == 256, "qk_norm_rope: head_dim must be 64/128/256");
  auto qkv = qkv_in.contiguous();
  auto qw = qw_in.to(at::kBFloat16).contiguous(), kw = kw_in.to(at::kBFloat16).contiguous();
  auto cosb = cos_in.to(at::kBFloat16).contiguous(), sinb = sin_in.to(at::kBFloat16).contiguous();
  auto pos = pos_in.to(at::kInt).contiguous();
  auto out = at::empty_like(qkv);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_qk_norm_rope(e, qkv, qw, kw, cosb, sinb, pos, out, T,
                            static_cast<int>(hq), static_cast<int>(hk), static_cast<int>(hv),
                            D, static_cast<float>(eps), interleaved ? 1 : 0, gemma ? 1 : 0);
  });
  return out;
}

// DeepSeek-style grouped routing (noaux_tc). Returns (topk_ids int32, topk_weights f32).
static std::tuple<at::Tensor, at::Tensor> moe_route_grouped_mps(
    const at::Tensor& logits_in, const c10::optional<at::Tensor>& bias_in, int64_t k,
    int64_t n_group, int64_t topk_group, bool renormalize, double routed_scaling_factor,
    int64_t scoring_func) {
  TORCH_CHECK(logits_in.device().is_mps() && logits_in.dim() == 2 && tk_is_float_dtype(logits_in),
              "moe_route_grouped: logits must be a float (T,E) MPS tensor");
  auto logits = logits_in.contiguous();
  const int T = logits.size(0), E = logits.size(1);
  TORCH_CHECK(E <= 512 && n_group >= 1 && n_group <= 32 && E % n_group == 0,
              "moe_route_grouped: need E <= 512, 1 <= n_group <= 32, E % n_group == 0");
  TORCH_CHECK(topk_group >= 1 && topk_group <= n_group, "moe_route_grouped: bad topk_group");
  TORCH_CHECK(k > 0 && k <= 16 && k <= E, "moe_route_grouped: require 1 <= k <= min(16, E)");
  TORCH_CHECK(scoring_func >= 0 && scoring_func <= 2, "moe_route_grouped: bad scoring_func");
  const bool has_bias = bias_in.has_value();
  at::Tensor bias;
  if (has_bias) {
    TORCH_CHECK(bias_in->dim() == 1 && bias_in->size(0) == E,
                "moe_route_grouped: bias must be (E,)");
    bias = bias_in->to(at::kFloat).contiguous();
  } else {
    bias = at::zeros({1}, logits.options().dtype(at::kFloat));
  }
  auto ids = at::empty({T, (int64_t)k}, logits.options().dtype(at::kInt));
  auto weights = at::empty({T, (int64_t)k}, logits.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_route_grouped(e, logits, bias, ids, weights, T, E,
                                 static_cast<int>(n_group), static_cast<int>(topk_group),
                                 static_cast<int>(k), renormalize ? 1 : 0,
                                 static_cast<float>(routed_scaling_factor),
                                 static_cast<int>(scoring_func), has_bias ? 1 : 0,
                                 tk_type_name(logits));
  });
  return {ids, weights};
}

// Quantized grouped expert GEMMs: Wq is the packed (E, N_out, row_bytes) uint8 expert stack
// (quant groups along K — tk.quant.quantize_expert_stack layout), contracted as A @ Wq^T.
// bias: pass a 1-element dummy with has_bias=false when absent. bf16 activations only.
static at::Tensor moe_grouped_gemm_rect_q_mps(const at::Tensor& A_in, const at::Tensor& Wq_in,
                                              const at::Tensor& eot_in, const at::Tensor& bias_in,
                                              bool has_bias, const std::string& format) {
  TORCH_CHECK(A_in.device().is_mps() && A_in.scalar_type() == at::kBFloat16,
              "moe_grouped_gemm_rect_q: A must be a bfloat16 MPS tensor");
  TORCH_CHECK(A_in.dim() == 2 && Wq_in.dim() == 3 && Wq_in.scalar_type() == at::kByte,
              "moe_grouped_gemm_rect_q: A (rows,K), Wq (E,N_out,row_bytes) uint8");
  const int total_rows = A_in.size(0), K_dim = A_in.size(1), N_out = Wq_in.size(1);
  TORCH_CHECK(total_rows % 32 == 0 && K_dim % 32 == 0 && N_out % 32 == 0,
              "moe_grouped_gemm_rect_q: rows%32, K%32, N%32");
  if (has_bias) {
    TORCH_CHECK(bias_in.dim() == 2 && bias_in.size(0) == Wq_in.size(0) &&
                bias_in.size(1) == N_out, "moe_grouped_gemm_rect_q: bias must be (E, N_out)");
  }
  auto A = A_in.contiguous(), Wq = Wq_in.contiguous();
  auto eot = eot_in.to(at::kInt).contiguous();
  auto bias = bias_in.to(at::kBFloat16).contiguous();
  auto out = at::empty({total_rows, N_out}, A.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_grouped_gemm_rect_q(e, out, A, Wq, eot, bias, total_rows, K_dim, N_out,
                                       has_bias ? 1 : 0, format);
  });
  return out;
}

static at::Tensor moe_grouped_gemm_swiglu_q_mps(const at::Tensor& A_in, const at::Tensor& W1q_in,
                                                const at::Tensor& eot_in, const at::Tensor& bias_in,
                                                bool has_bias, int64_t act_mode, double alpha,
                                                double limit, const std::string& format) {
  TORCH_CHECK(A_in.device().is_mps() && A_in.scalar_type() == at::kBFloat16,
              "moe_grouped_gemm_swiglu_q: A must be a bfloat16 MPS tensor");
  TORCH_CHECK(A_in.dim() == 2 && W1q_in.dim() == 3 && W1q_in.scalar_type() == at::kByte,
              "moe_grouped_gemm_swiglu_q: A (rows,H), W1q (E,2*inter,row_bytes) uint8");
  const int total_rows = A_in.size(0), H = A_in.size(1);
  TORCH_CHECK(W1q_in.size(1) % 2 == 0, "moe_grouped_gemm_swiglu_q: W1q dim 1 must be 2*inter");
  const int inter = W1q_in.size(1) / 2;
  TORCH_CHECK(total_rows % 32 == 0 && H % 32 == 0 && inter % 32 == 0,
              "moe_grouped_gemm_swiglu_q: rows%32, H%32, inter%32");
  if (has_bias) {
    TORCH_CHECK(bias_in.dim() == 2 && bias_in.size(0) == W1q_in.size(0) &&
                bias_in.size(1) == 2 * inter,
                "moe_grouped_gemm_swiglu_q: bias must be (E, 2*inter)");
  }
  auto A = A_in.contiguous(), W1q = W1q_in.contiguous();
  auto eot = eot_in.to(at::kInt).contiguous();
  auto bias = bias_in.to(at::kBFloat16).contiguous();
  auto out = at::empty({total_rows, inter}, A.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_grouped_gemm_swiglu_q(e, out, A, W1q, eot, bias, total_rows, H, inter,
                                         has_bias ? 1 : 0, static_cast<int>(act_mode),
                                         static_cast<float>(alpha), static_cast<float>(limit),
                                         format);
  });
  return out;
}

// MoE finalize: out[t] = sum_k weight[t,k] * expert_out[inv_idx[t*k+k]]. Returns (T, Hdim).
static at::Tensor moe_finalize_mps(const at::Tensor& expert_out_in, const at::Tensor& inv_in,
                                   const at::Tensor& w_in, int64_t k) {
  TORCH_CHECK(expert_out_in.device().is_mps(), "moe_finalize: expert_out must be an MPS tensor");
  TORCH_CHECK(expert_out_in.dim() == 2, "moe_finalize: expert_out must be (T*k, Hdim)");
  TORCH_CHECK(tk_is_float_dtype(expert_out_in), "moe_finalize: expert_out must be float");
  auto eo = expert_out_in.contiguous();
  auto inv = inv_in.to(at::kInt).contiguous();
  auto w = w_in.to(at::kFloat).contiguous();
  const int T = w.size(0), Hdim = eo.size(1);
  auto out = at::empty({T, Hdim}, eo.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_finalize(e, eo, inv, w, out, T, static_cast<int>(k), Hdim, tk_type_name(eo));
  });
  return out;
}

// Greedy sampling: argmax token index over the last (vocab) axis. Returns int32.
static at::Tensor argmax_sample_mps(const at::Tensor& logits_in) {
  TORCH_CHECK(logits_in.device().is_mps(), "argmax_sample: logits must be an MPS tensor");
  TORCH_CHECK(tk_is_float_dtype(logits_in), "argmax_sample: logits must be float32/float16/bfloat16");
  auto logits = logits_in.contiguous();
  const int V = logits.size(-1);
  const int rows = static_cast<int>(logits.numel() / V);
  std::vector<int64_t> oshape(logits.sizes().begin(), logits.sizes().end() - 1);
  if (oshape.empty()) oshape.push_back(1);
  auto out = at::empty(oshape, logits.options().dtype(at::kInt));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_argmax(e, logits, out, rows, V, tk_type_name(logits));
  });
  return out;
}

// Gumbel-max categorical sampling from softmax(logits/temperature). Returns int32.
static at::Tensor sample_categorical_mps(const at::Tensor& logits_in, double temperature,
                                         int64_t seed) {
  TORCH_CHECK(logits_in.device().is_mps(), "sample_categorical: logits must be an MPS tensor");
  TORCH_CHECK(tk_is_float_dtype(logits_in), "sample_categorical: logits must be float");
  TORCH_CHECK(temperature > 0.0, "sample_categorical: temperature must be > 0");
  auto logits = logits_in.contiguous();
  const int V = logits.size(-1);
  const int rows = static_cast<int>(logits.numel() / V);
  std::vector<int64_t> oshape(logits.sizes().begin(), logits.sizes().end() - 1);
  if (oshape.empty()) oshape.push_back(1);
  auto out = at::empty(oshape, logits.options().dtype(at::kInt));
  const uint32_t seed_u = static_cast<uint32_t>(seed);
  const float invtemp = 1.0f / static_cast<float>(temperature);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_sample_categorical(e, logits, out, rows, V, seed_u, invtemp, tk_type_name(logits));
  });
  return out;
}

// Top-k sampling: Gumbel-max from softmax over the k highest logits. Returns int32.
static at::Tensor top_k_sample_mps(const at::Tensor& logits_in, int64_t k, double temperature,
                                   int64_t seed) {
  TORCH_CHECK(logits_in.device().is_mps(), "top_k_sample: logits must be an MPS tensor");
  TORCH_CHECK(tk_is_float_dtype(logits_in), "top_k_sample: logits must be float");
  TORCH_CHECK(temperature > 0.0, "top_k_sample: temperature must be > 0");
  auto logits = logits_in.contiguous();
  const int V = logits.size(-1);
  TORCH_CHECK(k > 0 && k <= 64 && k <= V, "top_k_sample: require 1 <= k <= min(64, vocab)");
  const int rows = static_cast<int>(logits.numel() / V);
  std::vector<int64_t> oshape(logits.sizes().begin(), logits.sizes().end() - 1);
  if (oshape.empty()) oshape.push_back(1);
  auto out = at::empty(oshape, logits.options().dtype(at::kInt));
  const uint32_t seed_u = static_cast<uint32_t>(seed);
  const float invtemp = 1.0f / static_cast<float>(temperature);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_top_k_sample(e, logits, out, rows, V, static_cast<int>(k), seed_u, invtemp,
                            tk_type_name(logits));
  });
  return out;
}

// Temperature + repetition/presence/frequency penalties. Returns the penalized logits.
// Beam-search advance: fused log-softmax + cumulative score + top-beam_width with parent tracking.
static std::tuple<at::Tensor, at::Tensor, at::Tensor> beam_advance_mps(
    const at::Tensor& logits_in, const at::Tensor& cum_in, int64_t beam_width) {
  TORCH_CHECK(logits_in.device().is_mps() && tk_is_float_dtype(logits_in),
              "beam_advance: logits must be a float MPS tensor");
  TORCH_CHECK(logits_in.dim() == 2 && cum_in.dim() == 2 && cum_in.size(1) == beam_width,
              "beam_advance: logits (B*BM, V), cum_log_probs (B, BM)");
  TORCH_CHECK(beam_width >= 1 && beam_width <= 16, "beam_advance: beam_width must be in [1, 16]");
  const int BM = static_cast<int>(beam_width);
  const int B = cum_in.size(0), BR = logits_in.size(0), V = logits_in.size(1);
  TORCH_CHECK(BR == B * BM, "beam_advance: logits rows must equal B * beam_width");
  const int two_bm = 2 * BM;
  TORCH_CHECK(V >= two_bm, "beam_advance: vocab size V must be >= 2 * beam_width");
  auto logits = logits_in.contiguous();
  auto cum = cum_in.to(at::kFloat).contiguous().view({BR});
  auto f32 = logits.options().dtype(at::kFloat);
  auto i32 = logits.options().dtype(at::kInt);
  auto cand_score = at::empty({BR, two_bm}, f32);
  auto cand_token = at::empty({BR, two_bm}, i32);
  auto next_token = at::empty({B, BM}, i32);
  auto parent_beam = at::empty({B, BM}, i32);
  auto new_cum = at::empty({B, BM}, f32);
  const std::string tn = tk_type_name(logits);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_beam_topk_partials(e, logits, cum, cand_score, cand_token, BR, V, two_bm, tn);
    tk::launch_beam_select(e, cand_score, cand_token, next_token, parent_beam, new_cum, B, BM,
                           two_bm);
  });
  return {next_token, parent_beam, new_cum};
}

static std::vector<at::Tensor> spec_verify_linear_mps(
    const at::Tensor& draft_tokens_in, const at::Tensor& draft_probs_in,
    const at::Tensor& target_probs_in, const at::Tensor& bonus_tokens_in,
    const at::Tensor& accept_u_in, int64_t seed) {
  TORCH_CHECK(draft_tokens_in.device().is_mps() && draft_tokens_in.dim() == 2,
              "spec_verify_linear: draft_tokens must be (B,S) MPS");
  TORCH_CHECK(draft_probs_in.dim() == 3 && target_probs_in.dim() == 3,
              "spec_verify_linear: draft_probs (B,S,V), target_probs (B,S+1,V)");
  const int B = draft_tokens_in.size(0), S = draft_tokens_in.size(1), V = draft_probs_in.size(2);
  TORCH_CHECK(target_probs_in.size(1) == S + 1 && target_probs_in.size(2) == V,
              "spec_verify_linear: target_probs must be (B,S+1,V)");
  auto dt = draft_tokens_in.to(at::kInt).contiguous();
  auto dp = draft_probs_in.to(at::kFloat).contiguous();
  auto tp = target_probs_in.to(at::kFloat).contiguous();
  auto bt = bonus_tokens_in.to(at::kInt).contiguous();
  auto au = accept_u_in.to(at::kFloat).contiguous();
  auto i32 = dt.options();
  auto out_tokens = at::empty({B, S + 1}, i32);
  auto accepted_cnt = at::empty({B}, i32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_spec_verify_linear(e, dt, dp, tp, bt, au, out_tokens, accepted_cnt, B, S, V,
                                  static_cast<unsigned>(seed));
  });
  return {out_tokens, accepted_cnt};
}

static std::vector<at::Tensor> spec_verify_tree_mps(const at::Tensor& draft_tokens_in,
    const at::Tensor& target_probs_in, const at::Tensor& rt_in, const at::Tensor& rs_in,
    const at::Tensor& tree_valid_in, int64_t seed) {
  TORCH_CHECK(target_probs_in.device().is_mps() && target_probs_in.dim() == 3,
              "spec_verify_tree: target_probs must be (B, N, V) MPS");
  const int B = target_probs_in.size(0), N = target_probs_in.size(1), V = target_probs_in.size(2);
  TORCH_CHECK(draft_tokens_in.dim() == 2 && draft_tokens_in.size(0) == B &&
              draft_tokens_in.size(1) == N - 1, "spec_verify_tree: draft_tokens must be (B, N-1)");
  TORCH_CHECK(tree_valid_in.dim() == 1 && tree_valid_in.size(0) == B,
              "spec_verify_tree: tree_valid must be (B,)");
  auto dt = draft_tokens_in.to(at::kInt).contiguous();
  auto tp = target_probs_in.to(at::kFloat).contiguous();
  auto rt = rt_in.to(at::kInt).contiguous();
  auto rs = rs_in.to(at::kInt).contiguous();
  auto tv = tree_valid_in.to(at::kInt).contiguous();
  auto i32 = dt.options();
  auto accept_index = at::empty({B, N}, i32);
  auto accept_token = at::empty({B, N}, i32);
  auto accept_num = at::empty({B}, i32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_spec_verify_tree(e, dt, tp, rt, rs, accept_index, accept_token, accept_num, tv, B, N,
                                V, static_cast<unsigned>(seed));
  });
  return {accept_index, accept_token, accept_num};
}

static std::vector<at::Tensor> spec_compact_mps(const at::Tensor& out_tokens_in,
    const at::Tensor& accepted_cnt_in, const at::Tensor& seq_lens_in) {
  TORCH_CHECK(out_tokens_in.device().is_mps() && out_tokens_in.dim() == 2,
              "spec_compact: out_tokens must be (B, S+1) MPS");
  const int B = out_tokens_in.size(0), Sp1 = out_tokens_in.size(1);   // chunked scan: any B
  auto ot = out_tokens_in.to(at::kInt).contiguous();
  auto ac = accepted_cnt_in.to(at::kInt).contiguous();
  auto sl = seq_lens_in.to(at::kInt).contiguous();
  auto i32 = ot.options();
  auto packed_tokens = at::empty({B * Sp1}, i32);
  auto packed_pos = at::empty({B * Sp1}, i32);
  auto cu_accepted = at::empty({B + 1}, i32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_spec_compact(e, ot, ac, sl, packed_tokens, packed_pos, cu_accepted, B, Sp1);
  });
  return {packed_tokens, packed_pos, cu_accepted};
}

// vLLM v1 ragged rejection samplers (int32 ids, external-noise buffers).
static at::Tensor rejection_greedy_sample_mps(
    const at::Tensor& cu_in, const at::Tensor& draft_in, const at::Tensor& targ_in,
    const at::Tensor& bonus_in, int64_t max_draft, const c10::optional<at::Tensor>& greedy_in) {
  TORCH_CHECK(cu_in.device().is_mps() && cu_in.dim() == 1 && cu_in.size(0) >= 2,
              "rejection_greedy_sample: cu_num_draft_tokens must be (B+1,) MPS");
  const int B = cu_in.size(0) - 1, S1 = max_draft + 1;
  auto cu = cu_in.to(at::kInt).contiguous(), draft = draft_in.to(at::kInt).contiguous();
  auto targ = targ_in.to(at::kInt).contiguous(), bonus = bonus_in.to(at::kInt).contiguous();
  const bool hg = greedy_in.has_value();
  auto greedy = hg ? greedy_in->to(at::kByte).contiguous() : at::zeros({1}, cu.options().dtype(at::kByte));
  auto out = at::empty({B, S1}, cu.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_rejection_greedy_sample(e, out, cu, draft, targ, bonus, greedy, B, S1, hg ? 1 : 0);
  });
  return out;
}
static at::Tensor rejection_random_sample_mps(
    const at::Tensor& cu_in, const at::Tensor& draft_in, const at::Tensor& tp_in,
    const at::Tensor& bonus_in, const at::Tensor& rec_in, const at::Tensor& unif_in,
    int64_t max_draft, const c10::optional<at::Tensor>& dp_in,
    const c10::optional<at::Tensor>& greedy_in) {
  const int B = cu_in.size(0) - 1, S1 = max_draft + 1, V = tp_in.size(1);
  auto cu = cu_in.to(at::kInt).contiguous(), draft = draft_in.to(at::kInt).contiguous();
  auto tp = tp_in.to(at::kFloat).contiguous(), bonus = bonus_in.to(at::kInt).contiguous();
  auto rec = rec_in.to(at::kInt).contiguous(), unif = unif_in.to(at::kFloat).contiguous();
  const bool ndp = !dp_in.has_value(), hg = greedy_in.has_value();
  auto dp = ndp ? at::zeros({1}, tp.options()) : dp_in->to(at::kFloat).contiguous();
  auto greedy = hg ? greedy_in->to(at::kByte).contiguous() : at::zeros({1}, cu.options().dtype(at::kByte));
  auto out = at::empty({B, S1}, cu.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_rejection_random_sample(e, out, cu, draft, dp, tp, bonus, rec, unif, greedy, B,
                                       S1, V, ndp ? 1 : 0, hg ? 1 : 0);
  });
  return out;
}
static at::Tensor sample_recovered_tokens_mps(
    const at::Tensor& cu_in, const at::Tensor& draft_in, const at::Tensor& tp_in,
    const at::Tensor& iq_in, const c10::optional<at::Tensor>& dp_in) {
  const int B = cu_in.size(0) - 1, total = draft_in.size(0), V = tp_in.size(1);
  auto cu = cu_in.to(at::kInt).contiguous(), draft = draft_in.to(at::kInt).contiguous();
  auto tp = tp_in.to(at::kFloat).contiguous(), iq = iq_in.to(at::kFloat).contiguous();
  const bool ndp = !dp_in.has_value();
  auto dp = ndp ? at::zeros({1}, tp.options()) : dp_in->to(at::kFloat).contiguous();
  auto out = at::empty({total}, cu.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_sample_recovered_tokens(e, out, cu, draft, dp, tp, iq, B, total, V, ndp ? 1 : 0);
  });
  return out;
}

// EAGLE input-prep metadata builders (int32, per request).
static std::vector<at::Tensor> eagle_prepare_inputs_padded_mps(
    const at::Tensor& cu_in, const at::Tensor& vc_in, const at::Tensor& qsl_in) {
  const int B = cu_in.size(0) - 1;
  auto cu = cu_in.to(at::kInt).contiguous(), vc = vc_in.to(at::kInt).contiguous();
  auto qsl = qsl_in.to(at::kInt).contiguous();
  auto ti = at::empty({B}, cu.options()), nr = at::empty({B}, cu.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_eagle_prepare_inputs_padded(e, cu, vc, qsl, ti, nr, B);
  });
  return {ti, nr};
}
static std::vector<at::Tensor> eagle_prepare_next_token_padded_mps(
    const at::Tensor& si_in, const at::Tensor& dm_in, const at::Tensor& bk_in, int64_t vocab) {
  const int B = si_in.size(0), ns = si_in.size(1);
  auto si = si_in.to(at::kInt).contiguous(), dm = dm_in.to(at::kByte).contiguous();
  auto bk = bk_in.to(at::kInt).contiguous();
  auto nt = at::empty({B}, si.options()), vc = at::empty({B}, si.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_eagle_prepare_next_token_padded(e, si, dm, bk, nt, vc, (int)vocab, ns, B);
  });
  return {nt, vc};
}
static std::vector<at::Tensor> eagle_step_slot_mapping_metadata_mps(
    const at::Tensor& pos_in, const at::Tensor& bt_in, const at::Tensor& sl_in,
    int64_t block_size, int64_t max_model_len, int64_t pad_id, int64_t input_batch_size) {
  const int B = pos_in.size(0), stride = bt_in.size(1);
  const int ib = input_batch_size < 0 ? B : (int)input_batch_size;
  auto pos = pos_in.to(at::kInt).contiguous(), bt = bt_in.to(at::kInt).contiguous();
  auto sl = sl_in.to(at::kInt).contiguous();
  auto cp = at::empty({ib}, pos.options()), sm = at::empty({ib}, pos.options());
  auto nsl = at::empty({B}, pos.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_eagle_step_slot_mapping_metadata(e, pos, bt, sl, cp, sm, nsl, (int)block_size,
                                                (int)max_model_len, (int)pad_id, B, ib, stride,
                                                stride);
  });
  return {cp, sm, nsl};
}
static at::Tensor eagle_expand_int32_mps(const at::Tensor& in_in, const at::Tensor& cu_in,
                                         int64_t total, int64_t rf, int64_t rt) {
  const int B = cu_in.size(0) - 1;
  auto in = in_in.to(at::kInt).contiguous(), cu = cu_in.to(at::kInt).contiguous();
  auto out = at::empty({total}, cu.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_eagle_expand_int32(e, out, in, cu, (int)rf, (int)rt, B);
  });
  return out;
}

static std::vector<at::Tensor> build_dynamic_tree_mps(const at::Tensor& parents_in) {
  TORCH_CHECK(parents_in.device().is_mps() && parents_in.dim() == 2,
              "build_dynamic_tree: parents must be (B, N) MPS");
  const int B = parents_in.size(0), N = parents_in.size(1);
  auto p = parents_in.to(at::kInt).contiguous();
  auto i32 = p.options();
  auto rt = at::empty({B, N}, i32);
  auto rs = at::empty({B, N}, i32);
  auto positions = at::empty({B, N}, i32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_build_dynamic_tree(e, p, rt, rs, positions, B, N);
  });
  return {rt, rs, positions};
}

static at::Tensor spec_update_kv_meta_mps(const at::Tensor& seq_lens_in,
                                          const at::Tensor& accepted_cnt_in) {
  TORCH_CHECK(seq_lens_in.device().is_mps() && seq_lens_in.dim() == 1,
              "spec_update_kv_meta: seq_lens must be (B,) MPS");
  const int B = seq_lens_in.size(0);
  auto sl = seq_lens_in.to(at::kInt).contiguous();
  auto ac = accepted_cnt_in.to(at::kInt).contiguous();
  auto out = at::empty({B}, sl.options());
  tk_encode([&](TorchEncoder& e) { tk::launch_spec_update_kv_meta(e, sl, ac, out, B); });
  return out;
}

static at::Tensor apply_penalty_mps(const at::Tensor& logits_in, const at::Tensor& prev_in,
                                    const at::Tensor& bias_in, const at::Tensor& parent_in,
                                    double temperature,
                                    double repetition_penalty, double presence_penalty,
                                    double frequency_penalty, int64_t eos_id, int64_t min_length,
                                    int64_t gen_len) {
  TORCH_CHECK(logits_in.device().is_mps(), "apply_penalty: logits must be an MPS tensor");
  TORCH_CHECK(logits_in.dim() == 2, "apply_penalty: logits must be (num_tokens, vocab)");
  TORCH_CHECK(tk_is_float_dtype(logits_in), "apply_penalty: logits must be float");
  TORCH_CHECK(prev_in.dim() == 2 && prev_in.size(0) == logits_in.size(0),
              "apply_penalty: prev_tokens must be (num_tokens, history_len)");
  TORCH_CHECK(temperature > 0.0, "apply_penalty: temperature must be > 0");
  auto logits = logits_in.contiguous();
  auto prev = prev_in.to(at::kInt).contiguous();
  const int T = logits.size(0), V = logits.size(1), L = prev.size(1);
  TORCH_CHECK(bias_in.dim() == 1 && bias_in.size(0) == V, "apply_penalty: bias must be (vocab,)");
  TORCH_CHECK(parent_in.dim() == 1 && parent_in.size(0) == T, "apply_penalty: parent_ids (num_tokens,)");
  auto bias = bias_in.to(at::kFloat).contiguous();
  auto parent = parent_in.to(at::kInt).contiguous();
  auto out = at::empty_like(logits);
  auto counts = at::empty({T, V}, logits.options().dtype(at::kInt));
  const float invtemp = 1.0f / static_cast<float>(temperature);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_zero_i32(e, counts, T * V);
    tk::launch_penalty_histogram(e, prev, counts, V, L, T * L, parent);
    tk::launch_apply_penalty(e, logits, counts, out, bias, T, V, invtemp,
                             static_cast<float>(repetition_penalty),
                             static_cast<float>(presence_penalty),
                             static_cast<float>(frequency_penalty), static_cast<int>(eos_id),
                             static_cast<int>(min_length), static_cast<int>(gen_len),
                             tk_type_name(logits));
  });
  return out;
}

// Top-p (nucleus) sampling. Returns int32.
static at::Tensor top_p_sample_mps(const at::Tensor& logits_in, double p, double temperature,
                                   int64_t seed) {
  TORCH_CHECK(logits_in.device().is_mps(), "top_p_sample: logits must be an MPS tensor");
  TORCH_CHECK(tk_is_float_dtype(logits_in), "top_p_sample: logits must be float");
  TORCH_CHECK(temperature > 0.0, "top_p_sample: temperature must be > 0");
  TORCH_CHECK(p > 0.0 && p <= 1.0, "top_p_sample: p must be in (0, 1]");
  auto logits = logits_in.contiguous();
  const int V = logits.size(-1);
  const int rows = static_cast<int>(logits.numel() / V);
  std::vector<int64_t> oshape(logits.sizes().begin(), logits.sizes().end() - 1);
  if (oshape.empty()) oshape.push_back(1);
  auto out = at::empty(oshape, logits.options().dtype(at::kInt));
  const uint32_t seed_u = static_cast<uint32_t>(seed);
  const float invtemp = 1.0f / static_cast<float>(temperature);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_top_p_sample(e, logits, out, rows, V, static_cast<float>(p), seed_u, invtemp,
                            tk_type_name(logits));
  });
  return out;
}

static at::Tensor min_p_sample_mps(const at::Tensor& logits_in, double min_p, double temperature,
                                   int64_t seed) {
  TORCH_CHECK(logits_in.device().is_mps() && tk_is_float_dtype(logits_in),
              "min_p_sample: logits must be a float MPS tensor");
  TORCH_CHECK(temperature > 0.0, "min_p_sample: temperature must be > 0");
  TORCH_CHECK(min_p > 0.0 && min_p <= 1.0, "min_p_sample: min_p must be in (0, 1]");
  auto logits = logits_in.contiguous();
  const int V = logits.size(-1);
  const int rows = static_cast<int>(logits.numel() / V);
  std::vector<int64_t> oshape(logits.sizes().begin(), logits.sizes().end() - 1);
  if (oshape.empty()) oshape.push_back(1);
  auto out = at::empty(oshape, logits.options().dtype(at::kInt));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_min_p_sample(e, logits, out, rows, V, static_cast<float>(min_p),
                            static_cast<uint32_t>(seed), 1.0f / static_cast<float>(temperature),
                            tk_type_name(logits));
  });
  return out;
}

static at::Tensor typical_p_sample_mps(const at::Tensor& logits_in, double typical_p,
                                       double temperature, int64_t seed) {
  TORCH_CHECK(logits_in.device().is_mps() && tk_is_float_dtype(logits_in),
              "typical_p_sample: logits must be a float MPS tensor");
  TORCH_CHECK(temperature > 0.0, "typical_p_sample: temperature must be > 0");
  TORCH_CHECK(typical_p > 0.0 && typical_p <= 1.0, "typical_p_sample: typical_p must be in (0, 1]");
  auto logits = logits_in.contiguous();
  const int V = logits.size(-1);
  const int rows = static_cast<int>(logits.numel() / V);
  std::vector<int64_t> oshape(logits.sizes().begin(), logits.sizes().end() - 1);
  if (oshape.empty()) oshape.push_back(1);
  auto out = at::empty(oshape, logits.options().dtype(at::kInt));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_typical_p_sample(e, logits, out, rows, V, static_cast<float>(typical_p),
                                static_cast<uint32_t>(seed), 1.0f / static_cast<float>(temperature),
                                tk_type_name(logits));
  });
  return out;
}

static at::Tensor apply_token_bitmask_mps(const at::Tensor& logits_in, const at::Tensor& bitmask_in) {
  TORCH_CHECK(logits_in.device().is_mps() && tk_is_float_dtype(logits_in),
              "apply_token_bitmask: logits must be a float MPS tensor");
  auto logits = logits_in.contiguous();
  const int V = logits.size(-1);
  const int num_words = (V + 31) / 32;
  TORCH_CHECK(bitmask_in.size(-1) == num_words,
              "apply_token_bitmask: bitmask last dim must be ceil(V/32)");
  auto bitmask = bitmask_in.to(at::kInt).contiguous();   // raw bytes read as uint in the kernel
  const int rows = static_cast<int>(logits.numel() / V);
  auto out = at::empty_like(logits);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_apply_token_bitmask(e, logits, bitmask, out, rows, V, num_words, tk_type_name(logits));
  });
  return out;
}

static at::Tensor apply_bad_words_mps(const at::Tensor& logits_in, const at::Tensor& bad_ids_in,
                                      const at::Tensor& bad_lens_in) {
  TORCH_CHECK(logits_in.device().is_mps() && tk_is_float_dtype(logits_in),
              "apply_bad_words: logits must be a float MPS tensor");
  auto logits = logits_in.contiguous();
  const int V = logits.size(-1);
  const int rows = static_cast<int>(logits.numel() / V);
  auto bad_ids = bad_ids_in.to(at::kInt).contiguous();
  auto bad_lens = bad_lens_in.to(at::kInt).contiguous();
  TORCH_CHECK(bad_ids.dim() == 2 && bad_ids.size(0) == rows,
              "apply_bad_words: bad_ids must be (num_tokens, maxbad)");
  TORCH_CHECK(bad_lens.dim() == 1 && bad_lens.size(0) == rows,
              "apply_bad_words: bad_lens must be (num_tokens,)");
  const int maxbad = bad_ids.size(1);
  auto out = at::empty_like(logits);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_apply_bad_words(e, logits, bad_ids, bad_lens, out, rows, V, maxbad,
                               tk_type_name(logits));
  });
  return out;
}

// Per-tensor (global) dynamic quant via atomic-max. Returns (codes, scale scalar).
static std::tuple<at::Tensor, at::Tensor> quantize_per_tensor_mps(const at::Tensor& x_in,
                                                                  bool is_int8) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in),
              "quantize_per_tensor: x must be float MPS");
  auto x = x_in.contiguous();
  const int n = static_cast<int>(x.numel());
  auto codes = at::empty(x.sizes(), x.options().dtype(is_int8 ? at::kChar : at::kByte));
  auto scale = at::empty({1}, x.options().dtype(at::kFloat));
  auto scale_u = at::empty({1}, x.options().dtype(at::kInt));   // 4-byte atomic scratch
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_zero_i32(e, scale_u, 1);
    tk::launch_quant_tensor_absmax(e, x, scale_u, n, tk_type_name(x));
    tk::launch_quant_tensor_encode(e, x, scale_u, codes, scale, n, is_int8, tk_type_name(x));
  });
  return {codes, scale};
}
static std::tuple<at::Tensor, at::Tensor> quantize_per_tensor_fp8_mps(const at::Tensor& x) {
  return quantize_per_tensor_mps(x, false);
}
static std::tuple<at::Tensor, at::Tensor> quantize_per_tensor_int8_mps(const at::Tensor& x) {
  return quantize_per_tensor_mps(x, true);
}

// Runtime per-row fp8 e4m3 quantization. Returns (codes uint8, scale f32).
static std::tuple<at::Tensor, at::Tensor> quantize_per_token_fp8_mps(const at::Tensor& x_in) {
  TORCH_CHECK(x_in.device().is_mps(), "quantize_per_token_fp8: x must be an MPS tensor");
  TORCH_CHECK(tk_is_float_dtype(x_in), "quantize_per_token_fp8: x must be float32/float16/bfloat16");
  auto x = x_in.contiguous();
  const int D = x.size(-1);
  const int rows = static_cast<int>(x.numel() / D);
  auto codes = at::empty(x.sizes(), x.options().dtype(at::kByte));
  std::vector<int64_t> sshape(x.sizes().begin(), x.sizes().end() - 1);
  if (sshape.empty()) sshape.push_back(1);
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_quantize_per_token_fp8(e, x, codes, scale, rows, D, tk_type_name(x));
  });
  return {codes, scale};
}

// Runtime per-row symmetric int8 quantization. Returns (codes int8, scale f32).
static std::tuple<at::Tensor, at::Tensor> quantize_per_token_int8_mps(const at::Tensor& x_in) {
  TORCH_CHECK(x_in.device().is_mps(), "quantize_per_token_int8: x must be an MPS tensor");
  TORCH_CHECK(tk_is_float_dtype(x_in), "quantize_per_token_int8: x must be float32/float16/bfloat16");
  auto x = x_in.contiguous();
  const int D = x.size(-1);
  const int rows = static_cast<int>(x.numel() / D);
  auto codes = at::empty(x.sizes(), x.options().dtype(at::kChar));
  std::vector<int64_t> sshape(x.sizes().begin(), x.sizes().end() - 1);
  if (sshape.empty()) sshape.push_back(1);
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_quantize_per_token_int8(e, x, codes, scale, rows, D, tk_type_name(x));
  });
  return {codes, scale};
}

static std::tuple<at::Tensor, at::Tensor, int, int, int, at::Tensor> weight_quant_prep(
    const at::Tensor& w_in, const char* name) {
  TORCH_CHECK(w_in.device().is_mps() && tk_is_float_dtype(w_in),
              name, ": W must be float32/float16/bfloat16 MPS");
  TORCH_CHECK(w_in.dim() == 2 || w_in.dim() == 3, name, ": W must be (N,K) or (E,N,K)");
  auto w = w_in.contiguous();
  const int E = w.dim() == 3 ? w.size(0) : 1;
  const int N = w.size(-2), K = w.size(-1);
  TORCH_CHECK(K % 32 == 0, name, ": K must be a multiple of 32");
  std::vector<int64_t> qshape(w.sizes().begin(), w.sizes().end() - 1);
  qshape.push_back(K / 32);
  qshape.push_back(10);
  auto wq = at::empty(qshape, w.options().dtype(at::kByte));
  auto w_deq = at::empty(w.sizes(), w.options().dtype(at::kBFloat16));
  return {wq, w_deq, E, N, K, w};
}

static std::tuple<at::Tensor, at::Tensor> weight_quant_ternary_mps(
    const at::Tensor& w_in, int64_t group_k) {
  auto [wq, w_deq, E, N, K, w] = weight_quant_prep(w_in, "weight_quant_ternary");
  TORCH_CHECK(group_k >= 32 && group_k % 32 == 0 && K % group_k == 0,
              "weight_quant_ternary: group_k must be a multiple of 32 that divides K");
  tk_encode([&](TorchEncoder& e) {
    tk::launch_weight_quant_ternary(e, w, wq, w_deq, E, N, K, static_cast<int>(group_k),
                                    tk_type_name(w));
  });
  return {wq, w_deq};
}

static std::tuple<at::Tensor, at::Tensor> weight_quant_ternary_pt_mps(const at::Tensor& w_in) {
  auto [wq, w_deq, E, N, K, w] = weight_quant_prep(w_in, "weight_quant_ternary_pt");
  auto abssum = at::zeros({E}, w.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_weight_quant_ternary_abssum(e, w, abssum, E, N * K, tk_type_name(w));
    tk::launch_weight_quant_ternary_pt_encode(e, w, abssum, wq, w_deq, E, N, K, tk_type_name(w));
  });
  return {wq, w_deq};
}

static at::Tensor gemm_v3_mps(const at::Tensor& a_in, const at::Tensor& b_in) {
  TORCH_CHECK(a_in.device().is_mps() && tk_is_float_dtype(a_in), "gemm_v3: A must be float MPS");
  TORCH_CHECK(a_in.scalar_type() == b_in.scalar_type(), "gemm_v3: dtypes must match");
  auto a = a_in.contiguous();
  auto b = b_in.contiguous();
  const int N = a.size(0), K = a.size(1), M = b.size(1);
  TORCH_CHECK(b.size(0) == K && N % 64 == 0 && M % 64 == 0 && K % 32 == 0,
              "gemm_v3: A (N,K) B (K,M), N%64, M%64, K%32");
  auto out = at::empty({N, M}, a.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_gemm_v3(e, out, a, b, N, K, M, tk_type_name(a));
  });
  return out;
}

static at::Tensor qgemm_bwd_mps(const at::Tensor& g_in, const at::Tensor& wq_in) {
  TORCH_CHECK(g_in.device().is_mps() && g_in.scalar_type() == at::kHalf,
              "qgemm_bwd: grad_y must be a float16 MPS tensor");
  TORCH_CHECK(wq_in.scalar_type() == at::kByte && wq_in.dim() >= 2, "qgemm_bwd: wq uint8");
  auto g = g_in.contiguous();
  auto wq = wq_in.contiguous();
  const int M = g.size(0), N = wq.size(0);
  const long row_bytes = wq.numel() / N;
  const int K = static_cast<int>(row_bytes / 10 * 32);
  TORCH_CHECK(g.size(1) == N && M % 32 == 0 && N % 32 == 0 && K % 32 == 0,
              "qgemm_bwd: g (M,N), M/N/K % 32 == 0");
  auto out = at::empty({M, K}, g.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_qgemm_bwd(e, out, g, wq, M, N, K, "bitnet");
  });
  return out;
}

static std::tuple<at::Tensor, at::Tensor> kd_kl_topk_fwd_mps(
    const at::Tensor& logits_in, const at::Tensor& t_idx_in, const at::Tensor& t_prob_in,
    double invtemp, int64_t tail_mode) {
  TORCH_CHECK(logits_in.device().is_mps() && tk_is_float_dtype(logits_in),
              "kd_kl_topk_fwd: logits must be a float MPS tensor");
  TORCH_CHECK(t_idx_in.scalar_type() == at::kInt && t_prob_in.scalar_type() == at::kFloat,
              "kd_kl_topk_fwd: t_idx int32, t_prob float32");
  TORCH_CHECK(logits_in.dim() == 2 && t_idx_in.dim() == 2 && t_idx_in.sizes() == t_prob_in.sizes()
              && t_idx_in.size(0) == logits_in.size(0), "kd_kl_topk_fwd: shape mismatch");
  TORCH_CHECK(tail_mode == 0 || tail_mode == 1, "kd_kl_topk_fwd: tail_mode must be 0 or 1");
  auto logits = logits_in.contiguous();
  auto t_idx = t_idx_in.contiguous();
  auto t_prob = t_prob_in.contiguous();
  const int rows = logits.size(0), V = logits.size(1), K = t_idx.size(1);
  auto loss = at::empty({rows}, logits.options().dtype(at::kFloat));
  auto lse = at::empty({rows}, logits.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kd_kl_topk_fwd(e, logits, t_idx, t_prob, loss, lse, rows, V, K,
                              static_cast<float>(invtemp), static_cast<int>(tail_mode),
                              tk_type_name(logits));
  });
  return {loss, lse};
}

static at::Tensor kd_kl_topk_bwd_mps(
    const at::Tensor& logits_in, const at::Tensor& t_idx_in, const at::Tensor& t_prob_in,
    const at::Tensor& lse_in, const at::Tensor& grad_out_in, double invtemp,
    int64_t tail_mode) {
  TORCH_CHECK(logits_in.device().is_mps() && tk_is_float_dtype(logits_in),
              "kd_kl_topk_bwd: logits must be a float MPS tensor");
  TORCH_CHECK(t_idx_in.scalar_type() == at::kInt && t_prob_in.scalar_type() == at::kFloat,
              "kd_kl_topk_bwd: t_idx int32, t_prob float32");
  TORCH_CHECK(tail_mode == 0 || tail_mode == 1, "kd_kl_topk_bwd: tail_mode must be 0 or 1");
  auto logits = logits_in.contiguous();
  auto t_idx = t_idx_in.contiguous();
  auto t_prob = t_prob_in.contiguous();
  auto lse = lse_in.to(at::kFloat).contiguous();
  auto grad_out = grad_out_in.to(at::kFloat).contiguous();
  const int rows = logits.size(0), V = logits.size(1), K = t_idx.size(1);
  auto grad = at::empty_like(logits);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kd_kl_topk_bwd(e, logits, t_idx, t_prob, lse, grad_out, grad, rows, V, K,
                              static_cast<float>(invtemp), static_cast<int>(tail_mode),
                              tk_type_name(logits));
  });
  return grad;
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> fake_quant_int8_mps(
    const at::Tensor& x_in) {
  TORCH_CHECK(x_in.device().is_mps(), "fake_quant_int8: x must be an MPS tensor");
  TORCH_CHECK(tk_is_float_dtype(x_in), "fake_quant_int8: x must be float32/float16/bfloat16");
  auto x = x_in.contiguous();
  const int D = x.size(-1);
  TORCH_CHECK(D % 4 == 0, "fake_quant_int8: last dimension must be a multiple of 4");
  const int rows = static_cast<int>(x.numel() / D);
  auto x_q = at::empty(x.sizes(), x.options().dtype(at::kBFloat16));
  auto codes = at::empty(x.sizes(), x.options().dtype(at::kChar));
  std::vector<int64_t> sshape(x.sizes().begin(), x.sizes().end() - 1);
  if (sshape.empty()) sshape.push_back(1);
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_fake_quant_int8(e, x, x_q, codes, scale, rows, D, tk_type_name(x));
  });
  return {x_q, codes, scale};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> silu_mul_fake_quant_int8_mps(
    const at::Tensor& x_in, const at::Tensor& gate_in, int64_t mode, double alpha, double limit) {
  TORCH_CHECK(x_in.device().is_mps(), "silu_mul_fake_quant_int8: x must be an MPS tensor");
  TORCH_CHECK(tk_is_float_dtype(x_in), "silu_mul_fake_quant_int8: x must be float");
  TORCH_CHECK(x_in.sizes() == gate_in.sizes() && x_in.scalar_type() == gate_in.scalar_type(),
              "silu_mul_fake_quant_int8: x/gate must match");
  auto x = x_in.contiguous();
  auto gate = gate_in.contiguous();
  const int D = x.size(-1);
  TORCH_CHECK(D % 4 == 0, "silu_mul_fake_quant_int8: last dimension must be a multiple of 4");
  const int rows = static_cast<int>(x.numel() / D);
  auto x_q = at::empty(x.sizes(), x.options().dtype(at::kBFloat16));
  auto codes = at::empty(x.sizes(), x.options().dtype(at::kChar));
  std::vector<int64_t> sshape(x.sizes().begin(), x.sizes().end() - 1);
  if (sshape.empty()) sshape.push_back(1);
  auto scale = at::empty(sshape, x.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_silu_mul_fake_quant_int8(e, x, gate, x_q, codes, scale, rows, D,
                                        static_cast<int>(mode), static_cast<float>(alpha),
                                        static_cast<float>(limit), tk_type_name(x));
  });
  return {x_q, codes, scale};
}

static std::tuple<at::Tensor, at::Tensor> quantize_tq2_0_mps(const at::Tensor& w_in) {
  TORCH_CHECK(w_in.device().is_mps() && tk_is_float_dtype(w_in),
              "quantize_tq2_0: W must be float MPS");
  TORCH_CHECK(w_in.dim() == 2 || w_in.dim() == 3, "quantize_tq2_0: W (N,K) or (E,N,K)");
  auto w = w_in.contiguous();
  const int E = w.dim() == 3 ? w.size(0) : 1;
  const int N = w.size(-2), K = w.size(-1);
  TORCH_CHECK(K % 256 == 0, "quantize_tq2_0: K must be a multiple of 256");
  std::vector<int64_t> qshape(w.sizes().begin(), w.sizes().end() - 1);
  qshape.push_back(K / 256);
  qshape.push_back(66);
  auto wq = at::empty(qshape, w.options().dtype(at::kByte));
  auto w_deq = at::empty(w.sizes(), w.options().dtype(at::kBFloat16));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_quantize_tq2_0(e, w, wq, w_deq, E, N, K, tk_type_name(w));
  });
  return {wq, w_deq};
}

static at::Tensor qdequant_mps(const at::Tensor& wq_in, const std::string& format) {
  TORCH_CHECK(wq_in.device().is_mps() && wq_in.scalar_type() == at::kByte,
              "qdequant: wq must be uint8 MPS");
  TORCH_CHECK(wq_in.dim() == 3, "qdequant: wq (N, K/block_k, block_bytes)");
  auto wq = wq_in.contiguous();
  const int block_k = (format == "q4_K" || format == "iq4_xs" || format == "iq2_xxs" ||
                       format == "iq2_xs" || format == "iq3_xxs" || format == "iq1_s" ||
                       format == "q2_K" || format == "q3_K" || format == "q5_K" ||
                       format == "q6_K" || format == "tq2_0") ? 256
                    : (format == "kU4B8" || format == "kU4" || format == "fp8_block") ? 128
                    : (format == "hqq") ? 64
                    : (format == "nvfp4") ? 16 : 32;
  const int N = wq.size(0), K = static_cast<int>(wq.size(1)) * block_k;
  auto w = at::empty({N, K}, wq.options().dtype(at::kHalf));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_qdequant_fp16(e, w, wq, N, K, format);
  });
  return w;
}

static at::Tensor ternary_stats_mps(const at::Tensor& wq_in) {
  TORCH_CHECK(wq_in.device().is_mps() && wq_in.scalar_type() == at::kByte,
              "ternary_stats: wq must be a uint8 MPS tensor");
  TORCH_CHECK(wq_in.dim() >= 2 && wq_in.size(-1) == 10,
              "ternary_stats: wq must be (..., nblocks, 10) bitnet blocks");
  auto wq = wq_in.contiguous();
  const int nblocks = wq.size(-2);
  const int rows = static_cast<int>(wq.numel() / (static_cast<int64_t>(nblocks) * 10));
  auto counts = at::empty({rows, 3}, wq.options().dtype(at::kInt));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_ternary_stats(e, wq, counts, rows, nblocks);
  });
  return counts;
}

static at::Tensor code_flip_count_mps(const at::Tensor& a_in, const at::Tensor& b_in) {
  TORCH_CHECK(a_in.device().is_mps() && a_in.scalar_type() == at::kByte &&
              b_in.scalar_type() == at::kByte, "code_flip_count: packs must be uint8 MPS");
  TORCH_CHECK(a_in.sizes() == b_in.sizes() && a_in.dim() >= 2 && a_in.size(-1) == 10,
              "code_flip_count: packs must be identically-shaped (..., nblocks, 10)");
  auto a = a_in.contiguous();
  auto b = b_in.contiguous();
  const int nblocks = a.size(-2);
  const int rows = static_cast<int>(a.numel() / (static_cast<int64_t>(nblocks) * 10));
  auto flips = at::empty({rows}, a.options().dtype(at::kInt));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_code_flip_count(e, a, b, flips, rows, nblocks);
  });
  return flips;
}

static std::tuple<at::Tensor, at::Tensor> fake_quant_fp8_mps(const at::Tensor& x_in) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in),
              "fake_quant_fp8: x must be float MPS");
  auto x = x_in.contiguous();
  const int n = static_cast<int>(x.numel());
  auto x_fq = at::empty(x.sizes(), x.options());
  auto scale = at::empty({1}, x.options().dtype(at::kFloat));
  auto scale_u = at::empty({1}, x.options().dtype(at::kInt));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_zero_i32(e, scale_u, 1);
    tk::launch_quant_tensor_absmax(e, x, scale_u, n, tk_type_name(x));
    tk::launch_fake_quant_fp8(e, x, scale_u, x_fq, scale, n, tk_type_name(x));
  });
  return {x_fq, scale};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> kd_kl_dense_fwd_mps(
    const at::Tensor& t_in, const at::Tensor& s_in, double invtemp) {
  TORCH_CHECK(t_in.device().is_mps() && tk_is_float_dtype(t_in),
              "kd_kl_dense_fwd: logits must be float MPS");
  TORCH_CHECK(t_in.sizes() == s_in.sizes() && t_in.scalar_type() == s_in.scalar_type() &&
              t_in.dim() == 2, "kd_kl_dense_fwd: teacher/student (Tn, V) must match");
  auto t = t_in.contiguous();
  auto s = s_in.contiguous();
  const int rows = t.size(0), V = t.size(1);
  auto loss = at::empty({rows}, t.options().dtype(at::kFloat));
  auto lse_t = at::empty({rows}, t.options().dtype(at::kFloat));
  auto lse_s = at::empty({rows}, t.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kd_kl_dense_fwd(e, t, s, loss, lse_t, lse_s, rows, V,
                               static_cast<float>(invtemp), tk_type_name(t));
  });
  return {loss, lse_t, lse_s};
}

static at::Tensor kd_kl_dense_bwd_mps(const at::Tensor& t_in, const at::Tensor& s_in,
                                      const at::Tensor& lse_t_in, const at::Tensor& lse_s_in,
                                      const at::Tensor& grad_out_in, double invtemp) {
  auto t = t_in.contiguous();
  auto s = s_in.contiguous();
  auto lse_t = lse_t_in.contiguous();
  auto lse_s = lse_s_in.contiguous();
  auto grad_out = grad_out_in.to(at::kFloat).contiguous();
  const int rows = t.size(0), V = t.size(1);
  auto grad = at::empty_like(s);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kd_kl_dense_bwd(e, t, s, lse_t, lse_s, grad_out, grad, rows, V,
                               static_cast<float>(invtemp), tk_type_name(t));
  });
  return grad;
}

static std::vector<at::Tensor> kd_ce_fused_fwd_mps(
    const at::Tensor& t_in, const at::Tensor& s_in, const at::Tensor& targets_in,
    double invtemp) {
  TORCH_CHECK(t_in.device().is_mps() && tk_is_float_dtype(t_in),
              "kd_ce_fused_fwd: logits must be float MPS");
  TORCH_CHECK(t_in.sizes() == s_in.sizes() && t_in.scalar_type() == s_in.scalar_type() &&
              t_in.dim() == 2, "kd_ce_fused_fwd: teacher/student (Tn, V) must match");
  TORCH_CHECK(targets_in.scalar_type() == at::kInt && targets_in.numel() == t_in.size(0),
              "kd_ce_fused_fwd: targets must be int32 (Tn,)");
  auto t = t_in.contiguous();
  auto s = s_in.contiguous();
  auto tg = targets_in.contiguous();
  const int rows = t.size(0), V = t.size(1);
  auto opts = t.options().dtype(at::kFloat);
  auto ce = at::empty({rows}, opts);
  auto kd = at::empty({rows}, opts);
  auto lse_sr = at::empty({rows}, opts);
  auto lse_st = at::empty({rows}, opts);
  auto lse_t = at::empty({rows}, opts);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kd_ce_fused_fwd(e, t, s, tg, ce, kd, lse_sr, lse_st, lse_t, rows, V,
                               static_cast<float>(invtemp), tk_type_name(t));
  });
  return {ce, kd, lse_sr, lse_st, lse_t};
}

static at::Tensor kd_ce_fused_bwd_mps(
    const at::Tensor& t_in, const at::Tensor& s_in, const at::Tensor& targets_in,
    const at::Tensor& lse_sr_in, const at::Tensor& lse_st_in, const at::Tensor& lse_t_in,
    const at::Tensor& go_ce_in, const at::Tensor& go_kd_in, double invtemp) {
  auto t = t_in.contiguous();
  auto s = s_in.contiguous();
  auto tg = targets_in.contiguous();
  auto lse_sr = lse_sr_in.contiguous();
  auto lse_st = lse_st_in.contiguous();
  auto lse_t = lse_t_in.contiguous();
  auto go_ce = go_ce_in.to(at::kFloat).contiguous();
  auto go_kd = go_kd_in.to(at::kFloat).contiguous();
  const int rows = t.size(0), V = t.size(1);
  auto grad = at::empty_like(s);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kd_ce_fused_bwd(e, t, s, tg, lse_sr, lse_st, lse_t, go_ce, go_kd, grad,
                               rows, V, static_cast<float>(invtemp), tk_type_name(t));
  });
  return grad;
}

static at::Tensor qgemm_w2a8_fused_mps(const at::Tensor& wq_in, const at::Tensor& x_in) {
  TORCH_CHECK(x_in.device().is_mps() && tk_is_float_dtype(x_in),
              "qgemm_w2a8_fused: x must be float MPS");
  TORCH_CHECK(wq_in.scalar_type() == at::kByte && wq_in.dim() >= 2,
              "qgemm_w2a8_fused: wq uint8");
  auto x = x_in.contiguous();
  auto wq = wq_in.contiguous();
  const int M = x.size(0), K = x.size(1);
  const int N = wq.size(0);
  const long row_bytes = wq.numel() / N;
  TORCH_CHECK(static_cast<int>(row_bytes / 10 * 32) == K,
              "qgemm_w2a8_fused: wq K does not match x");
  TORCH_CHECK(K % 32 == 0 && K <= 8192, "qgemm_w2a8_fused: K % 32 == 0 and K <= 8192");
  auto out = at::empty({M, N}, x.options().dtype(at::kHalf));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_qgemm_w2a8_fused(e, out, wq, x, M, N, K, tk_type_name(x));
  });
  return out;
}

static at::Tensor attn_decode_mps(const at::Tensor& q_in, const at::Tensor& k_in,
                                  const at::Tensor& v_in) {
  TORCH_CHECK(q_in.device().is_mps() && tk_is_float_dtype(q_in),
              "attn_decode: q must be float MPS");
  TORCH_CHECK(q_in.dim() == 2 && k_in.dim() == 3 && v_in.dim() == 3,
              "attn_decode: q (Hq,D), kc/vc (Tk,Hkv,D)");
  TORCH_CHECK(q_in.scalar_type() == k_in.scalar_type() && k_in.scalar_type() == v_in.scalar_type(),
              "attn_decode: dtypes must match");
  auto q = q_in.contiguous();
  auto k = k_in.contiguous();
  auto v = v_in.contiguous();
  const int Hq = q.size(0), D = q.size(1);
  const int Tk = k.size(0), Hkv = k.size(1);
  TORCH_CHECK(k.size(2) == D && v.sizes() == k.sizes(), "attn_decode: shape mismatch");
  TORCH_CHECK(D <= 128, "attn_decode: D <= 128");
  TORCH_CHECK(Hq % Hkv == 0, "attn_decode: Hq must be a multiple of Hkv");
  auto out = at::empty({Hq, D}, q.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_attn_decode(e, q, k, v, out, Tk, Hq, Hkv, D, tk_type_name(q));
  });
  return out;
}

static at::Tensor moe_grouped_gemm_bwd_dx_mps(const at::Tensor& dy_in, const at::Tensor& W_in,
                                              const at::Tensor& eot_in) {
  TORCH_CHECK(dy_in.device().is_mps() && tk_is_float_dtype(dy_in),
              "moe_grouped_gemm_bwd_dx: dy must be float MPS");
  TORCH_CHECK(dy_in.dim() == 2 && W_in.dim() == 3,
              "moe_grouped_gemm_bwd_dx: dy (rows,N), W (E,K,N)");
  TORCH_CHECK(dy_in.scalar_type() == W_in.scalar_type(), "moe_grouped_gemm_bwd_dx: dtypes must match");
  const int total_rows = dy_in.size(0), N_out = dy_in.size(1), K_dim = W_in.size(1);
  TORCH_CHECK(W_in.size(2) == N_out, "moe_grouped_gemm_bwd_dx: W must be (E,K_dim,N_out)");
  TORCH_CHECK(total_rows % 32 == 0 && K_dim % 32 == 0 && N_out % 16 == 0,
              "moe_grouped_gemm_bwd_dx: rows%32, K%32, N%16");
  auto dy = dy_in.contiguous();
  auto W = W_in.contiguous();
  auto eot = eot_in.to(at::kInt).contiguous();
  auto dx = at::empty({total_rows, K_dim}, dy.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_grouped_gemm_bwd_dx(e, dx, dy, W, eot, total_rows, K_dim, N_out,
                                       tk_type_name(dy));
  });
  return dx;
}

static at::Tensor moe_grouped_gemm_bwd_dw_mps(const at::Tensor& A_in, const at::Tensor& dy_in,
                                              const at::Tensor& off_pad_in, int64_t num_experts) {
  TORCH_CHECK(A_in.device().is_mps() && tk_is_float_dtype(A_in),
              "moe_grouped_gemm_bwd_dw: A must be float MPS");
  TORCH_CHECK(A_in.dim() == 2 && dy_in.dim() == 2 && A_in.size(0) == dy_in.size(0),
              "moe_grouped_gemm_bwd_dw: A (rows,K), dy (rows,N)");
  TORCH_CHECK(A_in.scalar_type() == dy_in.scalar_type(), "moe_grouped_gemm_bwd_dw: dtypes must match");
  const int total_rows = A_in.size(0), K_dim = A_in.size(1), N_out = dy_in.size(1);
  const int E = static_cast<int>(num_experts);
  TORCH_CHECK(off_pad_in.numel() == E + 1, "moe_grouped_gemm_bwd_dw: off_pad must be (E+1,)");
  TORCH_CHECK(total_rows % 32 == 0 && K_dim % 32 == 0 && N_out % 32 == 0,
              "moe_grouped_gemm_bwd_dw: rows/K/N % 32 == 0");
  auto A = A_in.contiguous();
  auto dy = dy_in.contiguous();
  auto off_pad = off_pad_in.to(at::kInt).contiguous();
  auto dw = at::empty({E, K_dim, N_out}, A.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_grouped_gemm_bwd_dw(e, dw, A, dy, off_pad, E, total_rows, K_dim, N_out,
                                       tk_type_name(A));
  });
  return dw;
}

static std::tuple<at::Tensor, at::Tensor> moe_finalize_bwd_mps(
    const at::Tensor& grad_out_in, const at::Tensor& expert_out_in,
    const at::Tensor& inv_in, const at::Tensor& weights_in) {
  TORCH_CHECK(grad_out_in.device().is_mps() && tk_is_float_dtype(grad_out_in),
              "moe_finalize_bwd: grad_out must be float MPS");
  TORCH_CHECK(grad_out_in.dim() == 2 && expert_out_in.dim() == 2 && weights_in.dim() == 2,
              "moe_finalize_bwd: grad_out (T,H), expert_out (P,H), weights (T,K)");
  TORCH_CHECK(grad_out_in.scalar_type() == expert_out_in.scalar_type(),
              "moe_finalize_bwd: dtypes must match");
  const int T = grad_out_in.size(0), Hdim = grad_out_in.size(1), K = weights_in.size(1);
  TORCH_CHECK(inv_in.numel() == static_cast<int64_t>(T) * K, "moe_finalize_bwd: inv_pad must be (T*K,)");
  auto grad_out = grad_out_in.contiguous();
  auto expert_out = expert_out_in.contiguous();
  auto inv = inv_in.to(at::kInt).contiguous();
  auto weights = weights_in.to(at::kFloat).contiguous();
  auto grad_eo = at::zeros(expert_out.sizes(), expert_out.options());
  auto grad_w = at::empty({T, K}, weights.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_finalize_bwd(e, grad_out, expert_out, inv, grad_eo, grad_w, weights,
                                T, K, Hdim, tk_type_name(grad_out));
  });
  return {grad_eo, grad_w};
}

static at::Tensor moe_gather_bwd_mps(const at::Tensor& dA_in, const at::Tensor& inv_in,
                                     int64_t k) {
  TORCH_CHECK(dA_in.device().is_mps() && tk_is_float_dtype(dA_in),
              "moe_gather_bwd: dA must be float MPS");
  TORCH_CHECK(dA_in.dim() == 2 && inv_in.dim() == 1 && k > 0 && inv_in.numel() % k == 0,
              "moe_gather_bwd: dA (P,H), inv_pad (T*k,)");
  const int Hdim = dA_in.size(1);
  const int T = static_cast<int>(inv_in.numel() / k);
  auto dA = dA_in.contiguous();
  auto inv = inv_in.to(at::kInt).contiguous();
  auto dx = at::empty({T, Hdim}, dA.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_moe_gather_bwd(e, dA, inv, dx, T, static_cast<int>(k), Hdim,
                              tk_type_name(dA));
  });
  return dx;
}

static at::Tensor attn_causal_mps(const at::Tensor& q_in, const at::Tensor& k_in,
                                  const at::Tensor& v_in, double softcap,
                                  const c10::optional<at::Tensor>& sinks_in) {
  TORCH_CHECK(q_in.device().is_mps(), "attn_causal: q must be an MPS tensor");
  TORCH_CHECK(q_in.scalar_type() == at::kBFloat16, "attn_causal: q must be bfloat16");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  TORCH_CHECK(q.dim() == 4, "attn_causal: expects (B,H,N,D)");
  const int B = q.size(0), H = q.size(1);
  const unsigned N = static_cast<unsigned>(q.size(2));
  const int D = q.size(3);
  TORCH_CHECK(D == 64 || D == 128, "attn_causal: D must be 64 or 128");
  TORCH_CHECK(N % 8 == 0, "attn_causal: N must be a multiple of 8");
  const bool has_sink = sinks_in.has_value();
  at::Tensor sinks = q;
  if (has_sink) {
    TORCH_CHECK(sinks_in->dim() == 1 && sinks_in->size(0) == H, "attn_causal: sinks must be (H,)");
    sinks = sinks_in->to(at::kFloat).contiguous();
  }
  auto out = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_attn_causal(e, q, k, v, out, N, H, B, D, static_cast<float>(softcap), sinks,
                           has_sink ? 1 : 0);
  });
  return out;
}

static at::Tensor attn_window_mps(const at::Tensor& q_in, const at::Tensor& k_in,
                                  const at::Tensor& v_in, int64_t window, double softcap,
                                  const c10::optional<at::Tensor>& sinks_in) {
  TORCH_CHECK(q_in.device().is_mps(), "attn_window: q must be an MPS tensor");
  TORCH_CHECK(q_in.scalar_type() == at::kBFloat16, "attn_window: q must be bfloat16");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  TORCH_CHECK(q.dim() == 4, "attn_window: expects (B,H,N,D)");
  const int B = q.size(0), H = q.size(1);
  const unsigned N = static_cast<unsigned>(q.size(2));
  const int D = q.size(3);
  TORCH_CHECK(D == 64 || D == 128, "attn_window: D must be 64 or 128");
  TORCH_CHECK(N % 8 == 0, "attn_window: N must be a multiple of 8");
  const bool has_sink = sinks_in.has_value();
  at::Tensor sinks = q;
  if (has_sink) {
    TORCH_CHECK(sinks_in->dim() == 1 && sinks_in->size(0) == H, "attn_window: sinks must be (H,)");
    sinks = sinks_in->to(at::kFloat).contiguous();
  }
  auto out = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_attn_window(e, q, k, v, out, N, H, B, D, static_cast<int>(window),
                           static_cast<float>(softcap), sinks, has_sink ? 1 : 0);
  });
  return out;
}

static at::Tensor attn_varlen_prefill_mps(
    const at::Tensor& q_hm_in, const at::Tensor& key_cache_in, const at::Tensor& value_cache_in,
    const at::Tensor& block_table_in, const at::Tensor& context_lens_in,
    const at::Tensor& tile_seq_in, const at::Tensor& tile_local0_in, const at::Tensor& seq_qlen_in,
    double scale, double softcap, const c10::optional<at::Tensor>& sinks_in) {
  TORCH_CHECK(q_hm_in.device().is_mps(), "attn_varlen_prefill: q_hm must be an MPS tensor");
  TORCH_CHECK(q_hm_in.scalar_type() == at::kBFloat16 &&
              key_cache_in.scalar_type() == at::kBFloat16 &&
              value_cache_in.scalar_type() == at::kBFloat16,
              "attn_varlen_prefill: q_hm/key_cache/value_cache must be bfloat16");
  auto q_hm = q_hm_in.contiguous();
  auto key_cache = key_cache_in.contiguous(), value_cache = value_cache_in.contiguous();
  auto block_table = block_table_in.to(at::kInt).contiguous();
  auto context_lens = context_lens_in.to(at::kInt).contiguous();
  auto tile_seq = tile_seq_in.to(at::kInt).contiguous();
  auto tile_local0 = tile_local0_in.to(at::kInt).contiguous();
  auto seq_qlen = seq_qlen_in.to(at::kInt).contiguous();
  TORCH_CHECK(q_hm.dim() == 3, "attn_varlen_prefill: q_hm must be (H, total_padded, D)");
  const int H = q_hm.size(0), total_padded = q_hm.size(1), D = q_hm.size(2);
  const int H_KV = key_cache.size(2), block_size = key_cache.size(1);
  const int bt_stride = block_table.size(1), n_tiles = tile_seq.size(0);
  TORCH_CHECK(D == 64 || D == 128, "attn_varlen_prefill: D must be 64 or 128");
  TORCH_CHECK(block_size % 8 == 0, "attn_varlen_prefill: block_size must be a multiple of 8");
  const bool has_sink = sinks_in.has_value();
  at::Tensor sinks = q_hm;   // placeholder binding when absent (never read)
  if (has_sink) {
    TORCH_CHECK(sinks_in->dim() == 1 && sinks_in->size(0) == H,
                "attn_varlen_prefill: sinks must be (H,)");
    sinks = sinks_in->to(at::kFloat).contiguous();
  }
  auto out = at::empty_like(q_hm);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_attn_varlen_prefill(e, q_hm, key_cache, value_cache, block_table, context_lens,
                                   tile_seq, tile_local0, seq_qlen, out, n_tiles, total_padded,
                                   H, H_KV, block_size, bt_stride, static_cast<float>(scale), D,
                                   static_cast<float>(softcap), sinks, has_sink ? 1 : 0);
  });
  return out;
}

static std::vector<at::Tensor> varlen_build_worklist_mps(
    const at::Tensor& cu_seqlens_in, int64_t max_tiles) {
  TORCH_CHECK(cu_seqlens_in.device().is_mps() && cu_seqlens_in.dim() == 1,
              "varlen_build_worklist: cu_seqlens must be a 1-D MPS tensor (B+1,)");
  auto cu = cu_seqlens_in.to(at::kInt).contiguous();
  const int B = static_cast<int>(cu.size(0)) - 1;
  TORCH_CHECK(B >= 1, "varlen_build_worklist: B must be >= 1");   // chunked scan: any B
  auto i32 = cu.options().dtype(at::kInt);
  auto qlens = at::empty({(long)B}, i32);
  auto pad_off = at::empty({(long)(B + 1)}, i32);
  auto tile_seq = at::empty({max_tiles}, i32);
  auto tile_local0 = at::empty({max_tiles}, i32);
  auto n_tiles = at::empty({1}, i32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_varlen_build_worklist(e, cu, qlens, pad_off, tile_seq, tile_local0, n_tiles,
                                     B, static_cast<int>(max_tiles));
  });
  return {qlens, pad_off, tile_seq, tile_local0, n_tiles};
}

static at::Tensor varlen_pad_q_mps(const at::Tensor& q_in, const at::Tensor& cu_in,
                                   const at::Tensor& po_in, int64_t total_padded) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.dim() == 3 && q_in.scalar_type() == at::kBFloat16,
              "varlen_pad_q: q_packed must be (total_q, H, D) bf16 MPS");
  auto q = q_in.contiguous();
  auto cu = cu_in.to(at::kInt).contiguous(), po = po_in.to(at::kInt).contiguous();
  const int total_q = q.size(0), H = q.size(1), D = q.size(2), B = static_cast<int>(cu.size(0)) - 1;
  auto q_hm = at::empty({H, (long)total_padded, D}, q.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_varlen_pack_gather(e, q, cu, po, q_hm, total_q, B, H, D,
                                  static_cast<int>(total_padded), false);
  });
  return q_hm;
}

static at::Tensor varlen_regather_o_mps(const at::Tensor& o_in, const at::Tensor& cu_in,
                                        const at::Tensor& po_in, int64_t total_q) {
  TORCH_CHECK(o_in.device().is_mps() && o_in.dim() == 3 && o_in.scalar_type() == at::kBFloat16,
              "varlen_regather_o: o_hm must be (H, total_padded, D) bf16 MPS");
  auto o = o_in.contiguous();
  auto cu = cu_in.to(at::kInt).contiguous(), po = po_in.to(at::kInt).contiguous();
  const int H = o.size(0), total_padded = o.size(1), D = o.size(2), B = static_cast<int>(cu.size(0)) - 1;
  auto o_packed = at::empty({(long)total_q, H, D}, o.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_varlen_pack_gather(e, o, cu, po, o_packed, static_cast<int>(total_q), B, H, D,
                                  total_padded, true);
  });
  return o_packed;
}

// Fused LM-head + sampling: token id per row of h without materializing the (T, V) logits.
static at::Tensor lm_head_sample_mps(const at::Tensor& h_in, const at::Tensor& W_in,
                                     const at::Tensor& bias_in, int64_t mode, int64_t k,
                                     double temperature, int64_t seed) {
  TORCH_CHECK(h_in.device().is_mps(), "lm_head_sample: h must be an MPS tensor");
  TORCH_CHECK(h_in.dim() == 2 && W_in.dim() == 2 && h_in.size(1) == W_in.size(1),
              "lm_head_sample: h (T,K) and W (V,K) must share K");
  TORCH_CHECK(tk_is_float_dtype(h_in) && h_in.scalar_type() == W_in.scalar_type(),
              "lm_head_sample: h/W must be the same float dtype (f32/f16/bf16)");
  TORCH_CHECK(mode >= 0 && mode <= 2, "lm_head_sample: mode must be 0/1/2");
  constexpr int TILE_V = 256;
  auto h = h_in.contiguous();
  auto W = W_in.contiguous();
  auto bias = bias_in.to(at::kFloat).contiguous();
  const int T = h.size(0), K = h.size(1), V = W.size(0);
  const int num_vtiles = (V + TILE_V - 1) / TILE_V;
  const float invtemp = 1.0f / static_cast<float>(temperature);
  const int use_bias = bias.numel() > 1 ? 1 : 0;
  const std::string tn = tk_type_name(h);
  auto i32 = h.options().dtype(at::kInt);
  auto f32 = h.options().dtype(at::kFloat);
  auto out = at::empty({T}, i32);

  if (mode == 2) {
    TORCH_CHECK(k >= 1 && k <= 64 && k <= TILE_V, "lm_head_sample: topk k must be in [1, 64]");
    TORCH_CHECK(k <= V, "lm_head_sample: topk k must be <= vocab size V");
    auto part_val = at::empty({T, num_vtiles, static_cast<long>(k)}, f32);
    auto part_id = at::empty({T, num_vtiles, static_cast<long>(k)}, i32);
    tk_encode([&](TorchEncoder& e) {
      tk::launch_lm_head_topk_partials(e, h, W, part_val, part_id, bias, V, K, TILE_V, num_vtiles,
                                       static_cast<int>(k), use_bias, T, tn);
      tk::launch_lm_head_topk_reduce(e, part_val, part_id, out, num_vtiles, static_cast<int>(k),
                                     static_cast<unsigned>(seed), invtemp, T);
    });
    return out;
  }

  const int use_gumbel = (mode == 1) ? 1 : 0;
  auto part_val = at::empty({T, num_vtiles}, f32);
  auto part_id = at::empty({T, num_vtiles}, i32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_lm_head_argcat_partials(e, h, W, part_val, part_id, bias, V, K, TILE_V, num_vtiles,
                                       invtemp, static_cast<unsigned>(seed), use_gumbel, use_bias,
                                       T, tn);
    tk::launch_lm_head_argcat_reduce(e, part_val, part_id, out, num_vtiles, T);
  });
  return out;
}

// Fused LM-head + sampling over quantized q8_0/q4_0/q6_K weights (dequantized on read).
static at::Tensor lm_head_sample_q_mps(const at::Tensor& h_in, const at::Tensor& Wq_in,
                                       const at::Tensor& bias_in, int64_t V, int64_t K,
                                       const std::string& fmt, int64_t mode, int64_t topk,
                                       double temperature, int64_t seed, double top_p) {
  TORCH_CHECK(h_in.device().is_mps() && Wq_in.device().is_mps() &&
              bias_in.device().is_mps() && tk_is_float_dtype(h_in),
              "lm_head_sample_q: inputs must be MPS and h must be floating point");
  TORCH_CHECK(h_in.dim() == 2 && h_in.size(1) == K, "lm_head_sample_q: h must be (T, K)");
  TORCH_CHECK(mode >= 0 && mode <= 3,
              "lm_head_sample_q: mode 0=argmax, 1=categorical, 2=topk, 3=topp");
  TORCH_CHECK(temperature > 0.0, "lm_head_sample_q: temperature must be positive");
  int block_k = 0, block_bytes = 0;
  if (fmt == "q8_0") { block_k = 32; block_bytes = 34; }
  else if (fmt == "q4_0") { block_k = 32; block_bytes = 18; }
  else if (fmt == "q6_K") {
    block_k = 256; block_bytes = 210;
    TORCH_CHECK(h_in.scalar_type() == at::kFloat && mode <= 1,
                "lm_head_sample_q: q6_K requires fp32 h and argmax/categorical mode");
  } else {
    TORCH_CHECK(false, "lm_head_sample_q: format must be q8_0, q4_0, or q6_K");
  }
  TORCH_CHECK(K % block_k == 0 && Wq_in.scalar_type() == at::kByte && Wq_in.dim() == 3 &&
              Wq_in.size(0) == V && Wq_in.size(1) == K / block_k &&
              Wq_in.size(2) == block_bytes,
              "lm_head_sample_q: packed weight shape does not match V/K/format");
  constexpr int TILE_V = 256;
  auto h = h_in.contiguous();
  auto Wq = Wq_in.to(at::kByte).contiguous();
  auto bias = bias_in.to(at::kFloat).contiguous();
  const int T = h.size(0);
  TORCH_CHECK(T > 0 && V > 0 && K > 0,
              "lm_head_sample_q: T, V, and K must be positive");
  const int num_vtiles = (static_cast<int>(V) + TILE_V - 1) / TILE_V;
  const int use_bias = bias.numel() > 1 ? 1 : 0;
  const std::string tn = tk_type_name(h);
  auto i32 = h.options().dtype(at::kInt);
  auto f32 = h.options().dtype(at::kFloat);
  auto out = at::empty({T}, i32);
  const float invtemp = 1.0f / static_cast<float>(temperature);
  if (mode == 2 || mode == 3) {
    const int k = static_cast<int>(topk);
    TORCH_CHECK(k >= 1 && k <= 64 && k <= TILE_V, "lm_head_sample_q: k must be in [1, 64]");
    TORCH_CHECK(k <= static_cast<int>(V), "lm_head_sample_q: k must be <= vocab size V");
    TORCH_CHECK(mode != 3 || (top_p > 0.0 && top_p <= 1.0),
                "lm_head_sample_q: topp requires top_p in (0, 1]");
    auto part_val = at::empty({T, num_vtiles, k}, f32);
    auto part_id = at::empty({T, num_vtiles, k}, i32);
    auto part_lse = at::empty({T, num_vtiles}, f32);   // top-p only (per-tile logsumexp for true Z)
    tk_encode([&](TorchEncoder& e) {
      if (mode == 2) {
        tk::launch_lm_head_topk_partials_q(e, h, Wq, part_val, part_id, bias, static_cast<int>(V),
                                           static_cast<int>(K), TILE_V, num_vtiles, k, use_bias, T,
                                           fmt, tn);
        tk::launch_lm_head_topk_reduce(e, part_val, part_id, out, num_vtiles, k,
                                       static_cast<unsigned>(seed), invtemp, T);
      } else {
        // top-p: dedicated partials also emit the per-tile logsumexp for the true full-vocab Z
        tk::launch_lm_head_topp_partials_q(e, h, Wq, part_val, part_id, bias, static_cast<int>(V),
                                           static_cast<int>(K), TILE_V, num_vtiles, k, use_bias,
                                           invtemp, part_lse, T, fmt, tn);
        tk::launch_lm_head_topp_reduce(e, part_val, part_id, out, num_vtiles, k,
                                       static_cast<float>(top_p), static_cast<unsigned>(seed),
                                       invtemp, part_lse, T);
      }
    });
    return out;
  }
  const int use_gumbel = (mode == 1) ? 1 : 0;
  auto part_val = at::empty({T, num_vtiles}, f32);
  auto part_id = at::empty({T, num_vtiles}, i32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_lm_head_argcat_partials_q(e, h, Wq, part_val, part_id, bias, static_cast<int>(V),
                                         static_cast<int>(K), TILE_V, num_vtiles, invtemp,
                                         static_cast<unsigned>(seed), use_gumbel, use_bias, T, fmt,
                                         tn);
    tk::launch_lm_head_argcat_reduce(e, part_val, part_id, out, num_vtiles, T);
  });
  return out;
}

// Fused cross-entropy forward: per-row [loss, lse] without storing (T, V) probabilities.
static std::tuple<at::Tensor, at::Tensor> cross_entropy_fwd_mps(
    const at::Tensor& logits_in, const at::Tensor& targets_in, int64_t ignore_index,
    double label_smoothing, double z_loss, double softcap) {
  TORCH_CHECK(logits_in.device().is_mps() && tk_is_float_dtype(logits_in),
              "cross_entropy_fwd: logits must be a float MPS tensor");
  TORCH_CHECK(logits_in.dim() == 2, "cross_entropy_fwd: logits must be (T, V)");
  auto logits = logits_in.contiguous();
  auto targets = targets_in.to(at::kInt).contiguous();
  const int T = logits.size(0), V = logits.size(1);
  auto f32 = logits.options().dtype(at::kFloat);
  auto loss = at::empty({T}, f32);
  auto lse = at::empty({T}, f32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_cross_entropy_fwd(e, logits, targets, loss, lse, V, static_cast<int>(ignore_index),
                                 static_cast<float>(label_smoothing), static_cast<float>(z_loss),
                                 static_cast<float>(softcap), T, T < 512 && V >= 8192,
                                 tk_type_name(logits));
  });
  return {loss, lse};
}

// Fused cross-entropy backward: grad_logits (T, V), out-of-place.
static at::Tensor cross_entropy_bwd_mps(
    const at::Tensor& logits_in, const at::Tensor& targets_in, const at::Tensor& lse_in,
    const at::Tensor& grad_out_in, int64_t ignore_index, double label_smoothing, double z_loss,
    double softcap) {
  TORCH_CHECK(logits_in.device().is_mps() && tk_is_float_dtype(logits_in),
              "cross_entropy_bwd: logits must be a float MPS tensor");
  TORCH_CHECK(logits_in.dim() == 2, "cross_entropy_bwd: logits must be (T, V)");
  auto logits = logits_in.contiguous();
  auto targets = targets_in.to(at::kInt).contiguous();
  auto lse = lse_in.to(at::kFloat).contiguous();
  auto grad_out = grad_out_in.to(at::kFloat).contiguous();
  const int T = logits.size(0), V = logits.size(1);
  auto grad_logits = at::empty_like(logits);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_cross_entropy_bwd(e, logits, targets, lse, grad_out, grad_logits, V,
                                 static_cast<int>(ignore_index), static_cast<float>(label_smoothing),
                                 static_cast<float>(z_loss), static_cast<float>(softcap), T,
                                 T < 512 && V >= 8192, tk_type_name(logits));
  });
  return grad_logits;
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
  constexpr unsigned L = 64;   // chunk rows (must match LIN_CHUNK_L in the metal)
  if (N % L != 0 || N < 2 * L) {
    // small/ragged N: the serial single-simdgroup scan
    tk_encode([&](TorchEncoder& e) { tk::launch_lin_attn_causal(e, q, k, v, out, N, H, B, D); });
    return out;
  }
  // chunked-parallel: per-chunk KV -> exclusive chunk prefix -> seeded per-chunk scan
  const int C = static_cast<int>(N / L);
  auto f32 = q.options().dtype(at::kFloat);
  auto s_raw = at::empty({B, H, C, D, D}, f32);
  auto s_ex = at::empty({B, H, C, D, D}, f32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_lin_chunk_kv(e, k, v, s_raw, N, H, B, C, D);
    tk::launch_lin_chunk_scan(e, s_raw, s_ex, static_cast<unsigned>(C), B * H, D);
    tk::launch_lin_chunk_out(e, q, k, v, s_ex, out, N, H, B, C, D);
  });
  return out;
}

// Shared SSD dispatch (mamba2 / lin_attn_decay — identical math): the quadratic materialized
// kernel for small/ragged N, the chunked linear-time 3-kernel pipeline otherwise. The chunked
// D x D state is 64x64 quadrant-tiled (QB = D/64), so both head dims run chunked; the scanned
// state is stored BF16 (the out mma consumes bf16 anyway — identical results, half the state-
// read traffic) and the out kernel is COOPERATIVE (one threadgroup per chunk shares the staged
// state across its 8 query tiles). Route thresholds are MEASURED (M-series): the quadratic
// kernel wins below N=2048 at D=64 and below N=4096 at D=128.
static constexpr unsigned kSsdChunkL = 64;   // must match SSD_CHUNK_L in the metal
static inline unsigned ssd_chunk_min(int D) { return (D == 64) ? 2048 : 4096; }

static at::Tensor ssd_chunked(const at::Tensor& C, const at::Tensor& B, const at::Tensor& X,
                              const at::Tensor& cl, int Bsz, int H, unsigned N, int D) {
  TORCH_CHECK(N % kSsdChunkL == 0 && N >= 2 * kSsdChunkL,
              "ssd_chunked: N must be a multiple of 64 and >= 128");
  const int Cn = static_cast<int>(N / kSsdChunkL);
  auto out = at::empty_like(C);
  auto s_raw = at::empty({Bsz, H, Cn, D, D}, C.options().dtype(at::kFloat));
  auto s_ex = at::empty({Bsz, H, Cn, D, D}, C.options());   // bf16 (see routing note above)
  tk_encode([&](TorchEncoder& e) {
    tk::launch_ssd_chunk_kv(e, B, X, cl, s_raw, N, H, Bsz, Cn, D);
    tk::launch_ssd_chunk_scan(e, s_raw, cl, s_ex, static_cast<unsigned>(Cn), N, Bsz * H, D);
    tk::launch_ssd_chunk_out(e, C, B, X, cl, s_ex, out, N, H, Bsz, D);
  });
  return out;
}

static at::Tensor ssd_dispatch(const at::Tensor& C, const at::Tensor& B, const at::Tensor& X,
                               const at::Tensor& cl, int Bsz, int H, unsigned N, int D,
                               bool decay_kernel_name) {
  if (N % kSsdChunkL != 0 || N < ssd_chunk_min(D)) {
    auto out = at::empty_like(C);
    tk_encode([&](TorchEncoder& e) {
      if (decay_kernel_name)
        tk::launch_lin_attn_decay(e, C, B, X, cl, out, N, H, Bsz, D);
      else
        tk::launch_mamba2(e, C, B, X, cl, out, N, H, Bsz, D);
    });
    return out;
  }
  return ssd_chunked(C, B, X, cl, Bsz, H, N, D);
}

// Mamba-2 / SSD backward: [dC, dB, dX, dcumlog]. dcumlog = rowsum(M) - colsum(M), M = dSt∘S.
// Chunked (linear-time): recompute the forward chunk states (kv -> scan), build the gradient
// states G_c and reverse-scan dKV (gstate -> rscan, ONE reverse chain — bwd_col loads dKV in
// both row and col register layouts instead of materializing a transposed copy), then the
// cooperative chunk-bounded row/col kernels. dcl comes out in-kernel, split into intra + inter
// halves: rowsum(M) = r + ri and colsum(M) = cc + ci, so the host just combines.
static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ssd_chunked_bwd(
    const at::Tensor& C, const at::Tensor& B, const at::Tensor& X,
    const at::Tensor& cl, const at::Tensor& dY) {
  const int Bsz = C.size(0), H = C.size(1), D = C.size(3);
  const unsigned N = static_cast<unsigned>(C.size(2));
  TORCH_CHECK(N % kSsdChunkL == 0 && N >= 2 * kSsdChunkL,
              "ssd_chunked_bwd: N must be a multiple of 64 and >= 128");
  const int Cn = static_cast<int>(N / kSsdChunkL);
  auto f32 = C.options().dtype(at::kFloat);
  auto dC = at::empty_like(C), dB = at::empty_like(C), dX = at::empty_like(C);
  auto s_raw = at::empty({Bsz, H, Cn, D, D}, f32);
  auto s_ex = at::empty({Bsz, H, Cn, D, D}, C.options());          // bf16
  auto dkv = at::empty({Bsz, H, Cn, D, D}, C.options());           // bf16
  auto r = at::empty({Bsz, H, (long)N}, f32), ri = at::empty({Bsz, H, (long)N}, f32);
  auto cc = at::empty({Bsz, H, (long)N}, f32), ci = at::empty({Bsz, H, (long)N}, f32);
  tk_encode([&](TorchEncoder& e) {
    // forward chunk states (recompute), then G_c reuses s_raw (dead after the scan)
    tk::launch_ssd_chunk_kv(e, B, X, cl, s_raw, N, H, Bsz, Cn, D);
    tk::launch_ssd_chunk_scan(e, s_raw, cl, s_ex, static_cast<unsigned>(Cn), N, Bsz * H, D);
    tk::launch_ssd_chunk_gstate(e, C, dY, cl, s_raw, N, H, Bsz, Cn, D);
    tk::launch_ssd_chunk_rscan(e, s_raw, cl, dkv, static_cast<unsigned>(Cn), N, Bsz * H, D);
    tk::launch_ssd_chunk_bwd_row(e, C, B, X, cl, dY, s_ex, dC, r, ri, N, H, Bsz, D);
    tk::launch_ssd_chunk_bwd_col(e, C, B, X, cl, dY, dkv, dB, dX, cc, ci, N, H, Bsz, D);
  });
  return {dC, dB, dX, (r + ri) - (cc + ci)};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mamba2_bwd_mps(
    const at::Tensor& C_in, const at::Tensor& B_in, const at::Tensor& X_in,
    const at::Tensor& cl_in, const at::Tensor& dY_in, bool force_quadratic = false) {
  TORCH_CHECK(C_in.device().is_mps() && C_in.scalar_type() == at::kBFloat16,
              "mamba2_bwd: C,B,X,dY must be bfloat16 MPS");
  TORCH_CHECK(cl_in.scalar_type() == at::kFloat, "mamba2_bwd: cumlog must be float32");
  auto C = C_in.contiguous(), B = B_in.contiguous(), X = X_in.contiguous();
  auto cl = cl_in.contiguous(), dY = dY_in.contiguous();
  TORCH_CHECK(C.dim() == 4, "mamba2_bwd: C,B,X,dY expect (B,H,N,D)");
  const int Bsz = C.size(0), H = C.size(1), D = C.size(3);
  const unsigned N = static_cast<unsigned>(C.size(2));
  TORCH_CHECK(D == 64 || D == 128, "mamba2_bwd: D must be 64 or 128");
  TORCH_CHECK(N % 8 == 0, "mamba2_bwd: N must be a multiple of 8");
  // Chunked linear-time backward above the same measured crossovers as the forward; quadratic
  // (in-kernel fp32 dcumlog) below. force_quadratic pins the quadratic path (matches MLX).
  if (!force_quadratic && N % kSsdChunkL == 0 && N >= ssd_chunk_min(D)) {
    return ssd_chunked_bwd(C, B, X, cl, dY);
  }
  auto f32 = C.options().dtype(at::kFloat);
  auto dC = at::empty_like(C), dB = at::empty_like(C), dX = at::empty_like(C);
  auto r = at::empty({Bsz, H, (long)N}, f32);
  auto cc = at::empty({Bsz, H, (long)N}, f32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_mamba2_bwd_row(e, C, B, X, cl, dY, dC, r, N, H, Bsz, D);
    tk::launch_mamba2_bwd_col(e, C, B, X, cl, dY, dB, dX, cc, N, H, Bsz, D);
  });
  return {dC, dB, dX, r - cc};
}

// Forced chunked routes (tests / benchmarks): same math as the auto route, no threshold gate.
static at::Tensor mamba2_chunked_mps(const at::Tensor& C_in, const at::Tensor& B_in,
                                     const at::Tensor& X_in, const at::Tensor& cl_in) {
  TORCH_CHECK(C_in.device().is_mps() && C_in.scalar_type() == at::kBFloat16,
              "mamba2_chunked: C,B,X must be bfloat16 MPS");
  TORCH_CHECK(cl_in.scalar_type() == at::kFloat, "mamba2_chunked: cumlog must be float32");
  auto C = C_in.contiguous(), B = B_in.contiguous(), X = X_in.contiguous(), cl = cl_in.contiguous();
  TORCH_CHECK(C.dim() == 4, "mamba2_chunked: C,B,X expect (B,H,N,D)");
  const int Bsz = C.size(0), H = C.size(1), D = C.size(3);
  const unsigned N = static_cast<unsigned>(C.size(2));
  TORCH_CHECK(D == 64 || D == 128, "mamba2_chunked: D must be 64 or 128");
  return ssd_chunked(C, B, X, cl, Bsz, H, N, D);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mamba2_bwd_chunked_mps(
    const at::Tensor& C_in, const at::Tensor& B_in, const at::Tensor& X_in,
    const at::Tensor& cl_in, const at::Tensor& dY_in) {
  TORCH_CHECK(C_in.device().is_mps() && C_in.scalar_type() == at::kBFloat16,
              "mamba2_bwd_chunked: C,B,X,dY must be bfloat16 MPS");
  TORCH_CHECK(cl_in.scalar_type() == at::kFloat, "mamba2_bwd_chunked: cumlog must be float32");
  auto C = C_in.contiguous(), B = B_in.contiguous(), X = X_in.contiguous();
  auto cl = cl_in.contiguous(), dY = dY_in.contiguous();
  TORCH_CHECK(C.dim() == 4, "mamba2_bwd_chunked: C,B,X,dY expect (B,H,N,D)");
  const int D = C.size(3);
  TORCH_CHECK(D == 64 || D == 128, "mamba2_bwd_chunked: D must be 64 or 128");
  return ssd_chunked_bwd(C, B, X, cl, dY);
}

// ssd_decode: single-token SSD decode step. S is updated IN PLACE (Sin aliases Sout) and
// returned alongside y — the O(D^2) generation step for mamba2 / lin_attn_decay.
static std::tuple<at::Tensor, at::Tensor> ssd_decode_mps(
    const at::Tensor& S_in, const at::Tensor& a_in, const at::Tensor& x_in,
    const at::Tensor& k_in, const at::Tensor& q_in) {
  TORCH_CHECK(S_in.device().is_mps(), "ssd_decode: tensors must be MPS");
  TORCH_CHECK(S_in.scalar_type() == at::kFloat && a_in.scalar_type() == at::kFloat &&
              x_in.scalar_type() == at::kFloat && k_in.scalar_type() == at::kFloat &&
              q_in.scalar_type() == at::kFloat, "ssd_decode: all inputs must be float32");
  auto S = S_in.contiguous(), a = a_in.contiguous(), x = x_in.contiguous(),
       k = k_in.contiguous(), q = q_in.contiguous();
  TORCH_CHECK(S.dim() == 4, "ssd_decode: S expects (B,H,D,D)");
  const int Bsz = S.size(0), H = S.size(1), D = S.size(2);
  TORCH_CHECK(S.size(3) == D, "ssd_decode: S must be square");
  TORCH_CHECK(D == 64 || D == 128, "ssd_decode: D must be 64 or 128");
  auto y = at::empty({Bsz, H, D}, S.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_ssd_decode(e, S, a, x, k, q, S, y, static_cast<unsigned>(H), Bsz, D);
  });
  return {y, S};
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
  TORCH_CHECK(D == 64 || D == 128, "mamba2: D must be 64 or 128");
  TORCH_CHECK(N % 8 == 0, "mamba2: N must be a multiple of 8");
  return ssd_dispatch(C, B, X, cl, Bsz, H, N, D, /*decay_kernel_name=*/false);
}

static at::Tensor lin_attn_decay_mps(const at::Tensor& q_in, const at::Tensor& k_in,
                                     const at::Tensor& v_in, const at::Tensor& cl_in) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.scalar_type() == at::kBFloat16, "lin_attn_decay: q,k,v bf16 MPS");
  TORCH_CHECK(cl_in.scalar_type() == at::kFloat, "lin_attn_decay: cl must be float32");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous(), cl = cl_in.contiguous();
  TORCH_CHECK(q.dim() == 4, "lin_attn_decay: q,k,v expect (B,H,N,D)");
  const int Bsz = q.size(0), H = q.size(1);
  const unsigned N = static_cast<unsigned>(q.size(2));
  const int D = q.size(3);
  TORCH_CHECK(D == 64 && N % 8 == 0, "lin_attn_decay: D=64, N%8==0");
  return ssd_dispatch(q, k, v, cl, Bsz, H, N, D, /*decay_kernel_name=*/true);
}

static at::Tensor based_mps(const at::Tensor& q_in, const at::Tensor& k_in, const at::Tensor& v_in) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.scalar_type() == at::kBFloat16, "based: q,k,v bf16 MPS");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  TORCH_CHECK(q.dim() == 4, "based: q,k,v expect (B,H,N,D)");
  const int Bsz = q.size(0), H = q.size(1);
  const unsigned N = static_cast<unsigned>(q.size(2));
  const int DQK = q.size(3), DVO = v.size(3);
  TORCH_CHECK(DQK == 16 && DVO == 64 && N % 8 == 0, "based: D_QK=16, D_VO=64, N%8==0");
  auto out = at::empty_like(v);
  tk_encode([&](TorchEncoder& e) { tk::launch_based(e, q, k, v, out, N, H, Bsz, DQK, DVO); });
  return out;
}

static std::tuple<at::Tensor, at::Tensor> attn_fwd_l_mps(const at::Tensor& q_in, const at::Tensor& k_in,
                                                         const at::Tensor& v_in, bool causal) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.scalar_type() == at::kBFloat16, "attn_fwd_l: q,k,v bf16 MPS");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  const int B = q.size(0), H = q.size(1), D = q.size(3);
  const unsigned N = static_cast<unsigned>(q.size(2));
  TORCH_CHECK((D == 64 || D == 128) && N % 8 == 0, "attn_fwd_l: D in {64,128}, N%8==0");
  auto o = at::empty_like(q);
  auto L = at::empty({B, H, (int)N}, q.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) { tk::launch_attn_fwd_l(e, q, k, v, o, L, N, H, B, D, causal); });
  return {o, L};
}

static at::Tensor attn_bwd_prep_mps(const at::Tensor& o_in, const at::Tensor& do_in) {
  TORCH_CHECK(o_in.device().is_mps() && o_in.scalar_type() == at::kBFloat16, "attn_bwd_prep: o,do bf16 MPS");
  auto o = o_in.contiguous(), dd = do_in.contiguous();
  const int B = o.size(0), H = o.size(1), D = o.size(3);
  const unsigned N = static_cast<unsigned>(o.size(2));
  auto delta = at::empty({B, H, (int)N}, o.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) { tk::launch_attn_bwd_prep(e, o, dd, delta, N, H, B, D); });
  return delta;
}

static at::Tensor attn_bwd_dq_mps(const at::Tensor& q_in, const at::Tensor& k_in, const at::Tensor& v_in,
                                  const at::Tensor& do_in, const at::Tensor& L_in, const at::Tensor& delta_in,
                                  bool causal) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.scalar_type() == at::kBFloat16, "attn_bwd_dq: q bf16 MPS");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous(), dd = do_in.contiguous();
  auto L = L_in.contiguous(), delta = delta_in.contiguous();
  const int B = q.size(0), H = q.size(1), D = q.size(3);
  const unsigned N = static_cast<unsigned>(q.size(2));
  auto dq = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) { tk::launch_attn_bwd_dq(e, q, k, v, dd, L, delta, dq, N, H, B, D, causal); });
  return dq;
}

static std::tuple<at::Tensor, at::Tensor> attn_bwd_dkv_mps(
    const at::Tensor& q_in, const at::Tensor& k_in, const at::Tensor& v_in, const at::Tensor& do_in,
    const at::Tensor& L_in, const at::Tensor& delta_in, bool causal) {
  TORCH_CHECK(q_in.device().is_mps() && q_in.scalar_type() == at::kBFloat16, "attn_bwd_dkv: q bf16 MPS");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous(), dd = do_in.contiguous();
  auto L = L_in.contiguous(), delta = delta_in.contiguous();
  const int B = q.size(0), H = q.size(1), D = q.size(3);
  const unsigned N = static_cast<unsigned>(q.size(2));
  auto dk = at::empty_like(k), dv = at::empty_like(v);
  tk_encode([&](TorchEncoder& e) { tk::launch_attn_bwd_dkv(e, q, k, v, dd, L, delta, dk, dv, N, H, B, D, causal); });
  return {dk, dv};
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
  const int block_k = (format == "q4_K" || format == "iq4_xs" || format == "iq2_xxs" || format == "iq2_xs" || format == "iq3_xxs" || format == "iq1_s" || format == "q2_K" || format == "q3_K" || format == "q5_K" || format == "q6_K" || format == "tq2_0") ? 256
                    : (format == "kU4B8" || format == "kU4" || format == "fp8_block") ? 128
                    : (format == "hqq") ? 64
                    : (format == "nvfp4") ? 16 : 32;
  const int N = wq.size(0), K = (int)wq.size(1) * block_k, M = x.size(1);
  TORCH_CHECK(x.size(0) == K && N % 32 == 0 && M % 32 == 0, "qgemm: N%32,M%32, x rows==K");
  // k-quant (256-superblock) prefill route: in-GEMM fragment dequant of these formats measured
  // 2-2.3x slower than dequantize-then-matmul — use the span-decode dequant + torch GEMM.
  if (block_k == 256 && M >= 64) {
    auto w = at::empty({N, K}, x.options());
    tk_encode([&](TorchEncoder& e) { tk::launch_qdequant_fp16(e, w, wq, N, K, format); });
    return at::matmul(w, x);
  }
  auto out = at::empty({N, M}, x.options());
  // dequant-direct-to-fragment (Marlin zero-shuffle) — ~40% faster than the staged path, same result.
  tk_encode([&](TorchEncoder& e) { tk::launch_qgemm_frag(e, out, wq, x, N, K, M, format); });
  return out;
}

static at::Tensor qgemm_blockscale_mps(const at::Tensor& wq_in, const at::Tensor& x_in,
                                       const at::Tensor& sc_in) {
  TORCH_CHECK(wq_in.device().is_mps() && wq_in.scalar_type() == at::kByte, "qgemm_blockscale: wq uint8");
  TORCH_CHECK(x_in.scalar_type() == at::kHalf && sc_in.scalar_type() == at::kHalf, "qgemm_blockscale: x,scale2d f16");
  auto wq = wq_in.contiguous(), x = x_in.contiguous(), sc = sc_in.contiguous();
  const int N = wq.size(0), K = (int)wq.size(1) * 128, M = x.size(1);
  TORCH_CHECK(x.size(0) == K && N % 32 == 0 && M % 32 == 0, "qgemm_blockscale: shapes");
  auto out = at::empty({N, M}, x.options());
  tk_encode([&](TorchEncoder& e) { tk::launch_qgemm_blockscale(e, out, wq, x, sc, N, K, M); });
  return out;
}

static at::Tensor qgemm_fp8_scaled_mps(const at::Tensor& wq_in, const at::Tensor& xq_in,
                                       const at::Tensor& ws_in, const at::Tensor& as_in) {
  TORCH_CHECK(wq_in.device().is_mps() && wq_in.scalar_type() == at::kByte && xq_in.scalar_type() == at::kByte,
              "qgemm_fp8_scaled: wq, xq must be uint8 (fp8 codes) MPS");
  TORCH_CHECK(ws_in.scalar_type() == at::kHalf && as_in.scalar_type() == at::kHalf,
              "qgemm_fp8_scaled: w_scale, a_scale f16");
  auto wq = wq_in.contiguous(), xq = xq_in.contiguous(), ws = ws_in.contiguous(), as = as_in.contiguous();
  const int N = wq.size(0), K = wq.size(1), M = xq.size(1);
  TORCH_CHECK(xq.size(0) == K && N % 32 == 0 && M % 32 == 0 && K % 32 == 0, "qgemm_fp8_scaled: shapes");
  auto out = at::empty({N, M}, ws.options());
  tk_encode([&](TorchEncoder& e) { tk::launch_qgemm_fp8_scaled(e, out, wq, xq, ws, as, N, K, M); });
  return out;
}

static at::Tensor qgemm_actorder_k_mps(const at::Tensor& wq_in, const at::Tensor& x_in,
                                       const at::Tensor& perm_in, const std::string& format) {
  TORCH_CHECK(wq_in.device().is_mps() && wq_in.scalar_type() == at::kByte, "qgemm_actorder_k: wq uint8 MPS");
  TORCH_CHECK(x_in.scalar_type() == at::kHalf && perm_in.scalar_type() == at::kInt,
              "qgemm_actorder_k: x float16, perm int32");
  auto wq = wq_in.contiguous(), x = x_in.contiguous(), perm = perm_in.contiguous();
  const int block_k = (format == "kU4B8" || format == "kU4") ? 128 : 32;
  const int N = wq.size(0), K = (int)wq.size(1) * block_k, M = x.size(1);
  TORCH_CHECK(x.size(0) == K && perm.size(0) == K && N % 32 == 0 && M % 32 == 0, "qgemm_actorder_k: shapes");
  auto out = at::empty({N, M}, x.options());
  tk_encode([&](TorchEncoder& e) { tk::launch_qgemm_actorder(e, out, wq, x, perm, N, K, M, format); });
  return out;
}

static at::Tensor qgemv_mps(const at::Tensor& wq_in, const at::Tensor& x_in,
                            const std::string& format) {
  TORCH_CHECK(wq_in.device().is_mps() && x_in.device().is_mps(),
              "qgemv: inputs must be MPS tensors");
  TORCH_CHECK(wq_in.scalar_type() == at::kByte, "qgemv: wq must be uint8 packed blocks");
  TORCH_CHECK(x_in.scalar_type() == at::kHalf ||
              (x_in.scalar_type() == at::kFloat &&
               (format == "q4_0" || format == "q6_K")),
              "qgemv: x must be float16, or fp32 for q4_0/q6_K");
  auto wq = wq_in.contiguous(), x = x_in.contiguous();
  TORCH_CHECK(wq.dim() == 3 && x.dim() == 2 && x.size(1) == 1, "qgemv: wq (N,K/bk,bytes), x (K,1)");
  const int block_k = (format == "q4_K" || format == "iq4_xs" || format == "iq2_xxs" || format == "iq2_xs" || format == "iq3_xxs" || format == "iq1_s" || format == "q2_K" || format == "q3_K" || format == "q5_K" || format == "q6_K" || format == "tq2_0") ? 256
                    : (format == "kU4B8" || format == "kU4" || format == "fp8_block") ? 128
                    : (format == "hqq") ? 64
                    : (format == "nvfp4") ? 16 : 32;
  const int N = wq.size(0), K = (int)wq.size(1) * block_k;
  TORCH_CHECK(x.size(0) == K && N > 0 && K > 0,
              "qgemv: x rows must equal K; N and K must be positive");
  TORCH_CHECK(x.scalar_type() != at::kFloat ||
              (format == "q4_0" && wq.size(2) == 18) ||
              (format == "q6_K" && wq.size(2) == 210),
              "qgemv: fp32 packed block width does not match format");
  auto out = at::empty({N, 1}, x.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_qgemv(e, out, wq, x, N, K, format, tk_type_name(x));
  });
  return out;
}

static at::Tensor qflux_gelu_mps(const at::Tensor& wq_in, const at::Tensor& x_in,
                                 const at::Tensor& bias_in, const std::string& format) {
  TORCH_CHECK(wq_in.device().is_mps(), "qflux_gelu: wq must be an MPS tensor");
  TORCH_CHECK(wq_in.scalar_type() == at::kByte, "qflux_gelu: wq must be uint8 packed blocks");
  TORCH_CHECK(x_in.scalar_type() == at::kHalf && bias_in.scalar_type() == at::kHalf,
              "qflux_gelu: x and bias must be float16");
  auto wq = wq_in.contiguous(), x = x_in.contiguous(), bias = bias_in.contiguous();
  TORCH_CHECK(wq.dim() == 3 && x.dim() == 2 && bias.dim() == 1, "qflux_gelu: wq (N,K/bk,bytes), x (K,M), bias (M)");
  const int block_k = (format == "q4_K" || format == "iq4_xs" || format == "iq2_xxs" || format == "iq2_xs" || format == "iq3_xxs" || format == "iq1_s" || format == "q2_K" || format == "q3_K" || format == "q5_K" || format == "q6_K") ? 256
                    : (format == "kU4B8" || format == "kU4" || format == "fp8_block") ? 128
                    : (format == "hqq") ? 64
                    : (format == "nvfp4") ? 16 : 32;
  const int N = wq.size(0), K = (int)wq.size(1) * block_k, M = x.size(1);
  TORCH_CHECK(x.size(0) == K && bias.size(0) == M && N % 32 == 0 && M % 32 == 0, "qflux_gelu: shapes");
  auto out = at::empty({N, M}, x.options());
  tk_encode([&](TorchEncoder& e) { tk::launch_qflux_gelu(e, out, wq, x, bias, N, K, M, format); });
  return out;
}

static at::Tensor attn_q_mps(const at::Tensor& q_in, const at::Tensor& kq_in,
                             const at::Tensor& vq_in, const std::string& format, bool causal, bool multiwarp) {
  TORCH_CHECK(q_in.device().is_mps(), "attn_q: q must be an MPS tensor");
  TORCH_CHECK(q_in.scalar_type() == at::kBFloat16, "attn_q: q must be bfloat16");
  TORCH_CHECK(kq_in.scalar_type() == at::kByte && vq_in.scalar_type() == at::kByte,
              "attn_q: kq, vq must be uint8 packed blocks");
  auto q = q_in.contiguous(), kq = kq_in.contiguous(), vq = vq_in.contiguous();
  TORCH_CHECK(q.dim() == 4 && kq.dim() == 5, "attn_q: q (B,H,N,D), kq (B,H,N,D/bk,bytes)");
  const int B = q.size(0), H = q.size(1), D = q.size(3);
  const unsigned N = static_cast<unsigned>(q.size(2));
  TORCH_CHECK((D == 64 || D == 128) && N % 8 == 0, "attn_q: D in {64,128}, N%8==0");
  auto out = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) { tk::launch_attn_q(e, q, kq, vq, out, N, H, B, D, format, causal, multiwarp); });
  return out;
}

static at::Tensor qgemv_w8a8_mps(const at::Tensor& wq_in, const at::Tensor& xq_in,
                                 const at::Tensor& ws_in, const at::Tensor& as_in) {
  TORCH_CHECK(wq_in.device().is_mps(), "qgemv_w8a8: wq must be an MPS tensor");
  TORCH_CHECK(wq_in.scalar_type() == at::kChar && xq_in.scalar_type() == at::kChar,
              "qgemv_w8a8: wq, xq must be int8");
  TORCH_CHECK(ws_in.scalar_type() == at::kHalf && as_in.scalar_type() == at::kHalf,
              "qgemv_w8a8: scales must be float16");
  auto wq = wq_in.contiguous(), xq = xq_in.contiguous(), ws = ws_in.contiguous(), as = as_in.contiguous();
  const int N = wq.size(0), K = wq.size(1);
  TORCH_CHECK(K % 4 == 0 && xq.size(0) == K, "qgemv_w8a8: K%4==0, xq rows==K");
  auto out = at::empty({N, 1}, ws.options());
  tk_encode([&](TorchEncoder& e) { tk::launch_qgemv_w8a8(e, out, wq, xq, ws, as, N, K); });
  return out;
}

static at::Tensor qgemv_w2a8_mps(const at::Tensor& wq_in, const at::Tensor& xq_in,
                                 const at::Tensor& as_in, int64_t version) {
  TORCH_CHECK(wq_in.device().is_mps(), "qgemv_w2a8: wq must be an MPS tensor");
  TORCH_CHECK(wq_in.scalar_type() == at::kByte && xq_in.scalar_type() == at::kChar,
              "qgemv_w2a8: wq uint8 (bitnet blocks), xq int8");
  TORCH_CHECK(as_in.scalar_type() == at::kHalf, "qgemv_w2a8: a_scale float16");
  auto wq = wq_in.contiguous(), xq = xq_in.contiguous(), as = as_in.contiguous();
  const int N = wq.size(0), K = (int)wq.size(1) * 32;
  TORCH_CHECK(xq.size(0) == K, "qgemv_w2a8: xq rows==K");
  auto out = at::empty({N, 1}, as.options());
  tk_encode([&](TorchEncoder& e) {
    if (version == 2) {
      tk::launch_qgemv_w2a8_v2(e, out, wq, xq, as, N, K);
    } else {
      tk::launch_qgemv_w2a8(e, out, wq, xq, as, N, K);
    }
  });
  return out;
}

static at::Tensor qgemv_w2a8_v2_mps(const at::Tensor& wq_in, const at::Tensor& xq_in,
                                    const at::Tensor& as_in) {
  TORCH_CHECK(wq_in.device().is_mps(), "qgemv_w2a8_v2: wq must be an MPS tensor");
  TORCH_CHECK(wq_in.scalar_type() == at::kByte && xq_in.scalar_type() == at::kChar,
              "qgemv_w2a8_v2: wq uint8 (bitnet blocks), xq int8");
  TORCH_CHECK(as_in.scalar_type() == at::kHalf, "qgemv_w2a8_v2: a_scale float16");
  auto wq = wq_in.contiguous();
  auto xq = xq_in.contiguous();
  auto as = as_in.contiguous();
  const int N = wq.size(0), K = static_cast<int>(wq.size(1)) * 32;
  TORCH_CHECK(xq.size(0) == K, "qgemv_w2a8_v2: xq rows==K");
  auto out = at::empty({N, 1}, as.options());
  tk_encode([&](TorchEncoder& e) { tk::launch_qgemv_w2a8_v2(e, out, wq, xq, as, N, K); });
  return out;
}

static at::Tensor qgemm_w8a8_mps(const at::Tensor& wq_in, const at::Tensor& xq_in,
                                 const at::Tensor& ws_in, const at::Tensor& as_in) {
  TORCH_CHECK(wq_in.device().is_mps() && wq_in.scalar_type() == at::kChar && xq_in.scalar_type() == at::kChar,
              "qgemm_w8a8: wq, xq must be int8 MPS");
  auto wq = wq_in.contiguous(), xq = xq_in.contiguous(), ws = ws_in.contiguous(), as = as_in.contiguous();
  const int N = wq.size(0), K = wq.size(1), M = xq.size(0);
  TORCH_CHECK(K % 4 == 0 && xq.size(1) == K, "qgemm_w8a8: K%4==0, xq (M,K)");
  auto out = at::empty({N, M}, ws.options());
  tk_encode([&](TorchEncoder& e) { tk::launch_qgemm_w8a8(e, out, wq, xq, ws, as, N, K, M); });
  return out;
}

static at::Tensor qgemm_w2a8_mps(const at::Tensor& wq_in, const at::Tensor& xq_in,
                                 const at::Tensor& as_in) {
  TORCH_CHECK(wq_in.device().is_mps() && wq_in.scalar_type() == at::kByte && xq_in.scalar_type() == at::kChar,
              "qgemm_w2a8: wq uint8, xq int8 MPS");
  auto wq = wq_in.contiguous(), xq = xq_in.contiguous(), as = as_in.contiguous();
  const int N = wq.size(0), K = (int)wq.size(1) * 32, M = xq.size(0);
  TORCH_CHECK(xq.size(1) == K, "qgemm_w2a8: xq (M,K)");
  auto out = at::empty({N, M}, as.options());
  tk_encode([&](TorchEncoder& e) { tk::launch_qgemm_w2a8(e, out, wq, xq, as, N, K, M); });
  return out;
}

// ---------------------- Fused and specialized kernels ----------------------

static std::tuple<at::Tensor, at::Tensor> decode_layernorm_add_mps(
    const at::Tensor& x_in, const at::Tensor& residual_in,
    const at::Tensor& weight_in, const at::Tensor& bias_in, double eps) {
  TORCH_CHECK(x_in.device().is_mps() && residual_in.device().is_mps() &&
              weight_in.device().is_mps() && bias_in.device().is_mps() &&
              tk_is_float_dtype(x_in) && x_in.dim() > 0 && x_in.numel() > 0,
              "decode_layernorm_add: inputs must be non-empty MPS tensors");
  TORCH_CHECK(x_in.scalar_type() == at::kFloat || x_in.scalar_type() == at::kBFloat16,
              "decode_layernorm_add: dtype must be fp32 or bf16");
  TORCH_CHECK(residual_in.sizes() == x_in.sizes() &&
              residual_in.scalar_type() == x_in.scalar_type(),
              "decode_layernorm_add: residual must match x");
  const int dimension = x_in.size(-1);
  TORCH_CHECK(dimension > 0 && weight_in.dim() == 1 && bias_in.dim() == 1 &&
              weight_in.size(0) == dimension && bias_in.size(0) == dimension &&
              weight_in.scalar_type() == x_in.scalar_type() &&
              bias_in.scalar_type() == x_in.scalar_type(),
              "decode_layernorm_add: weight/bias must match final dimension and dtype");
  auto x = x_in.contiguous(), residual = residual_in.contiguous();
  auto weight = weight_in.contiguous(), bias = bias_in.contiguous();
  auto normalized = at::empty_like(x), summed = at::empty_like(x);
  const int rows = x.numel() / dimension;
  tk_encode([&](TorchEncoder& e) {
    tk::launch_decode_layernorm_add(
        e, x, residual, weight, bias, normalized, summed, rows, dimension,
        static_cast<float>(eps), tk_type_name(x));
  });
  return {normalized, summed};
}

static at::Tensor attn_decode_bh_mps(
    const at::Tensor& q_in, const at::Tensor& k_in, const at::Tensor& v_in,
    int64_t context_length) {
  TORCH_CHECK(q_in.device().is_mps() && tk_is_float_dtype(q_in),
              "attn_decode_bh: q must be float MPS");
  TORCH_CHECK(k_in.device().is_mps() && v_in.device().is_mps() &&
              q_in.dim() == 3 && k_in.dim() == 4 && v_in.sizes() == k_in.sizes(),
              "attn_decode_bh: q (B,Hq,D), k/v (B,Hkv,cache_T,D)");
  TORCH_CHECK(q_in.scalar_type() == k_in.scalar_type() &&
              k_in.scalar_type() == v_in.scalar_type(), "attn_decode_bh: dtypes must match");
  const int B = q_in.size(0), Hq = q_in.size(1), D = q_in.size(2);
  const int Hkv = k_in.size(1), cache_T = k_in.size(2);
  TORCH_CHECK(B > 0 && Hq > 0 && D > 0 && k_in.size(0) == B &&
              k_in.size(3) == D && D <= 128 && cache_T > 0 && Hkv > 0 &&
              Hq % Hkv == 0 && context_length > 0 && context_length <= cache_T,
              "attn_decode_bh: incompatible shapes/context length");
  auto q = q_in.contiguous(), k = k_in.contiguous(), v = v_in.contiguous();
  auto out = at::empty_like(q);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_attn_decode_bh(e, q, k, v, out, B, static_cast<int>(context_length),
                              cache_T, Hq, Hkv, D, tk_type_name(q));
  });
  return out;
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> decode_cache_attention_mps(
    const at::Tensor& q_in, const at::Tensor& new_k_in,
    const at::Tensor& new_v_in, const at::Tensor& cos_in,
    const at::Tensor& sin_in, const at::Tensor& positions_in,
    const at::Tensor& context_lengths_in, const at::Tensor& q_weight_in,
    const at::Tensor& k_weight_in, const at::Tensor& key_cache_in,
    const at::Tensor& value_cache_in, double eps, bool do_q_norm,
    bool do_k_norm, bool gemma, double softmax_scale) {
  TORCH_CHECK(q_in.device().is_mps() && tk_is_float_dtype(q_in) && q_in.dim() == 3 &&
              q_in.size(0) > 0 && q_in.size(1) > 0 &&
              (q_in.size(2) == 64 || q_in.size(2) == 128),
              "decode_cache_attention: q must be (B,Hq,D) float with D=64 or 128");
  const int B = q_in.size(0), Hq = q_in.size(1), D = q_in.size(2);
  TORCH_CHECK(new_k_in.device().is_mps() && new_v_in.device().is_mps() &&
              new_k_in.scalar_type() == q_in.scalar_type() &&
              new_v_in.scalar_type() == q_in.scalar_type() && new_k_in.dim() == 3 &&
              new_v_in.sizes() == new_k_in.sizes() && new_k_in.size(0) == B &&
              new_k_in.size(1) > 0 && new_k_in.size(2) == D &&
              Hq % new_k_in.size(1) == 0,
              "decode_cache_attention: new_k/new_v must be (B,Hkv,D), Hq%Hkv=0");
  const int Hkv = new_k_in.size(1);
  TORCH_CHECK(key_cache_in.device().is_mps() && value_cache_in.device().is_mps() &&
              key_cache_in.scalar_type() == q_in.scalar_type() &&
              value_cache_in.scalar_type() == q_in.scalar_type() &&
              key_cache_in.dim() == 4 && value_cache_in.sizes() == key_cache_in.sizes() &&
              key_cache_in.size(0) == B && key_cache_in.size(1) == Hkv &&
              key_cache_in.size(2) > 0 && key_cache_in.size(3) == D,
              "decode_cache_attention: caches must be (B,Hkv,cache_T,D)");
  TORCH_CHECK(cos_in.device().is_mps() && sin_in.device().is_mps() &&
              cos_in.scalar_type() == q_in.scalar_type() &&
              sin_in.scalar_type() == q_in.scalar_type() && cos_in.dim() == 2 &&
              sin_in.sizes() == cos_in.sizes() && cos_in.size(0) > 0 &&
              cos_in.size(1) == D / 2,
              "decode_cache_attention: cos/sin must be (P,D/2) with q dtype");
  TORCH_CHECK(positions_in.device().is_mps() && context_lengths_in.device().is_mps() &&
              positions_in.dim() == 1 && positions_in.size(0) == B &&
              context_lengths_in.dim() == 1 && context_lengths_in.size(0) == B,
              "decode_cache_attention: positions/context_lengths must be (B,)");
  TORCH_CHECK(!do_q_norm ||
              (q_weight_in.device().is_mps() && q_weight_in.scalar_type() == q_in.scalar_type() &&
               q_weight_in.dim() == 1 && q_weight_in.size(0) == D),
              "decode_cache_attention: q_weight must be (D,)");
  TORCH_CHECK(!do_k_norm ||
              (k_weight_in.device().is_mps() && k_weight_in.scalar_type() == q_in.scalar_type() &&
               k_weight_in.dim() == 1 && k_weight_in.size(0) == D),
              "decode_cache_attention: k_weight must be (D,)");
  TORCH_CHECK(eps > 0.0 && softmax_scale >= 0.0,
              "decode_cache_attention: eps positive and softmax_scale non-negative required");
  auto q = q_in.contiguous(), new_k = new_k_in.contiguous(), new_v = new_v_in.contiguous();
  auto cos = cos_in.contiguous(), sin = sin_in.contiguous();
  auto positions = positions_in.to(at::kInt).contiguous();
  auto context_lengths = context_lengths_in.to(at::kInt).contiguous();
  auto q_weight = q_weight_in.to(q.scalar_type()).contiguous();
  auto k_weight = k_weight_in.to(q.scalar_type()).contiguous();
  auto key_cache_input = key_cache_in.contiguous();
  auto value_cache_input = value_cache_in.contiguous();
  auto output = at::empty_like(q), key_cache = at::empty_like(key_cache_input);
  auto value_cache = at::empty_like(value_cache_input);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_kv_cache_clone(
        e, key_cache_input, value_cache_input, key_cache, value_cache,
        static_cast<uint64_t>(key_cache_input.numel()), tk_type_name(q));
    tk::launch_decode_cache_attention(
        e, q, new_k, new_v, cos, sin, positions, context_lengths,
        q_weight, k_weight, key_cache, value_cache, output,
        B, Hq, Hkv, key_cache.size(2), D, static_cast<float>(eps),
        do_q_norm, do_k_norm, gemma, static_cast<float>(softmax_scale),
        tk_type_name(q));
  });
  return {output, key_cache, value_cache};
}

static at::Tensor swin_attn_d32_mps(
    const at::Tensor& qkv_in, const at::Tensor& relative_bias_in,
    const at::Tensor& mask_in, int64_t windows_per_image) {
  TORCH_CHECK(qkv_in.device().is_mps() && relative_bias_in.device().is_mps() &&
              mask_in.device().is_mps() && tk_is_float_dtype(qkv_in),
              "swin_attn_d32: inputs must be MPS and qkv must be floating point");
  TORCH_CHECK(qkv_in.dim() == 5 && qkv_in.size(2) == 3 && qkv_in.size(4) == 32,
              "swin_attn_d32: qkv must be (BW,N,3,H,32)");
  const int BW = qkv_in.size(0), N = qkv_in.size(1), H = qkv_in.size(3);
  TORCH_CHECK(BW > 0 && N > 0 && H > 0 && windows_per_image >= 0,
              "swin_attn_d32: BW, N, and H must be positive; windows_per_image cannot be negative");
  TORCH_CHECK(relative_bias_in.scalar_type() == qkv_in.scalar_type() &&
              relative_bias_in.dim() == 3 && relative_bias_in.size(0) == H &&
              relative_bias_in.size(1) == N && relative_bias_in.size(2) == N,
              "swin_attn_d32: relative_bias must be (H,N,N) with qkv dtype");
  const bool has_mask = mask_in.dim() == 3;
  TORCH_CHECK((!has_mask && mask_in.numel() == 1) ||
              (has_mask && windows_per_image > 0 &&
               mask_in.size(0) == windows_per_image &&
               mask_in.size(1) == N && mask_in.size(2) == N),
              "swin_attn_d32: mask must be (windows_per_image,N,N) or scalar placeholder");
  auto qkv = qkv_in.contiguous(), relative_bias = relative_bias_in.contiguous();
  auto mask = mask_in.to(at::kFloat).contiguous();
  auto output = at::empty({BW, N, H, 32}, qkv.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_swin_attn_d32(e, qkv, relative_bias, mask, output, BW, N, H,
                             static_cast<int>(windows_per_image), has_mask ? 1 : 0,
                             tk_type_name(qkv));
  });
  return output;
}

static at::Tensor patch_merge_layernorm_mps(
    const at::Tensor& input_in, const at::Tensor& weight_in,
    const at::Tensor& bias_in, int64_t height, int64_t width, double eps) {
  TORCH_CHECK(input_in.device().is_mps() && weight_in.device().is_mps() &&
              bias_in.device().is_mps() && input_in.scalar_type() == at::kBFloat16 &&
              input_in.dim() == 3 && height > 0 && width > 0,
              "patch_merge_layernorm: input must be bf16 (B,H*W,C)");
  const int B = input_in.size(0), C = input_in.size(2), D = 4 * C;
  TORCH_CHECK(B > 0 && C > 0 && input_in.size(1) == height * width && weight_in.dim() == 1 &&
              bias_in.dim() == 1 && weight_in.size(0) == D && bias_in.size(0) == D &&
              weight_in.scalar_type() == at::kBFloat16 &&
              bias_in.scalar_type() == at::kBFloat16,
              "patch_merge_layernorm: spatial or weight/bias shape mismatch");
  auto input = input_in.contiguous(), weight = weight_in.contiguous(), bias = bias_in.contiguous();
  const int patches = ((static_cast<int>(height) + 1) / 2) *
                      ((static_cast<int>(width) + 1) / 2);
  auto output = at::empty({B, patches, D}, input.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_patch_merge_layernorm(e, input, weight, bias, output, B,
                                     static_cast<int>(height), static_cast<int>(width), C,
                                     static_cast<float>(eps));
  });
  return output;
}

static at::Tensor space_to_depth_norm_linear_mps(
    const at::Tensor& input_in, const at::Tensor& norm_weight_in,
    const at::Tensor& norm_bias_in, const at::Tensor& projection_weight_in,
    const at::Tensor& projection_bias_in, int64_t height, int64_t width,
    int64_t block_size, double eps, bool use_norm_bias,
    bool use_projection_bias) {
  TORCH_CHECK(input_in.device().is_mps() && tk_is_float_dtype(input_in) &&
              input_in.dim() == 3 && input_in.size(0) > 0 && input_in.size(2) > 0 &&
              height > 0 && width > 0 && input_in.size(1) == height * width &&
              (block_size == 2 || block_size == 4),
              "space_to_depth_norm_linear: input must be (B,H*W,C) float; block is 2 or 4");
  const int B = input_in.size(0), C = input_in.size(2);
  const int dimension = block_size * block_size * C;
  TORCH_CHECK(dimension <= 4096 && norm_weight_in.device().is_mps() &&
              norm_weight_in.scalar_type() == input_in.scalar_type() &&
              norm_weight_in.dim() == 1 && norm_weight_in.size(0) == dimension,
              "space_to_depth_norm_linear: norm_weight mismatch or dimension > 4096");
  TORCH_CHECK(!use_norm_bias ||
              (norm_bias_in.device().is_mps() &&
               norm_bias_in.scalar_type() == input_in.scalar_type() &&
               norm_bias_in.dim() == 1 && norm_bias_in.size(0) == dimension),
              "space_to_depth_norm_linear: norm_bias must be (block^2*C,)");
  TORCH_CHECK(projection_weight_in.device().is_mps() &&
              projection_weight_in.scalar_type() == input_in.scalar_type() &&
              projection_weight_in.dim() == 2 && projection_weight_in.size(0) > 0 &&
              projection_weight_in.size(1) == dimension,
              "space_to_depth_norm_linear: projection_weight must be (O,block^2*C)");
  const int O = projection_weight_in.size(0);
  TORCH_CHECK(!use_projection_bias ||
              (projection_bias_in.device().is_mps() &&
               projection_bias_in.scalar_type() == input_in.scalar_type() &&
               projection_bias_in.dim() == 1 && projection_bias_in.size(0) == O),
              "space_to_depth_norm_linear: projection_bias must be (O,)");
  auto input = input_in.contiguous(), norm_weight = norm_weight_in.contiguous();
  auto norm_bias = norm_bias_in.to(input.scalar_type()).contiguous();
  auto projection_weight = projection_weight_in.contiguous();
  auto projection_bias = projection_bias_in.to(input.scalar_type()).contiguous();
  const int patches = ((static_cast<int>(height) + block_size - 1) / block_size) *
                      ((static_cast<int>(width) + block_size - 1) / block_size);
  auto output = at::empty({B, patches, O}, input.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_space_to_depth_norm_linear(
        e, input, norm_weight, norm_bias, projection_weight, projection_bias,
        output, B, static_cast<int>(height), static_cast<int>(width), C, O,
        static_cast<int>(block_size), static_cast<float>(eps), use_norm_bias,
        use_projection_bias, tk_type_name(input));
  });
  return output;
}

static at::Tensor edge_mlp_256x7_mps(
    const at::Tensor& hidden_in, const at::Tensor& first_weight_in,
    const at::Tensor& first_bias_in, const at::Tensor& second_weight_in,
    const at::Tensor& second_bias_in) {
  TORCH_CHECK(hidden_in.device().is_mps() && first_weight_in.device().is_mps() &&
              first_bias_in.device().is_mps() && second_weight_in.device().is_mps() &&
              second_bias_in.device().is_mps() &&
              (hidden_in.scalar_type() == at::kFloat ||
               hidden_in.scalar_type() == at::kBFloat16) &&
              hidden_in.dim() == 3 && hidden_in.size(2) == 256,
              "edge_mlp_256x7: hidden must be fp32/bf16 (B,L,256)");
  const auto dtype = hidden_in.scalar_type();
  TORCH_CHECK(hidden_in.size(0) > 0 && hidden_in.size(1) > 0 &&
              first_weight_in.sizes() == at::IntArrayRef({256, 512}) &&
              first_bias_in.sizes() == at::IntArrayRef({256}) &&
              second_weight_in.sizes() == at::IntArrayRef({7, 256}) &&
              second_bias_in.sizes() == at::IntArrayRef({7}) &&
              first_weight_in.scalar_type() == dtype && first_bias_in.scalar_type() == dtype &&
              second_weight_in.scalar_type() == dtype && second_bias_in.scalar_type() == dtype,
              "edge_mlp_256x7: expected weights (256,512)/(7,256) and biases (256,)/(7,)");
  auto hidden = hidden_in.contiguous(), first_weight = first_weight_in.contiguous();
  auto first_bias = first_bias_in.contiguous(), second_weight = second_weight_in.contiguous();
  auto second_bias = second_bias_in.contiguous();
  const int B = hidden.size(0), L = hidden.size(1);
  auto output = at::empty({B, 7, L, L}, hidden.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_edge_mlp_256x7(e, hidden, first_weight, first_bias, second_weight,
                              second_bias, output, B, L, tk_type_name(hidden));
  });
  return output;
}

static void check_decode_linear(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias,
    const char* name) {
  TORCH_CHECK(x.device().is_mps() && weight.device().is_mps() && bias.device().is_mps() &&
              (x.scalar_type() == at::kFloat ||
              x.scalar_type() == at::kBFloat16) && x.dim() == 2 && weight.dim() == 2 &&
              bias.dim() == 1 && x.scalar_type() == weight.scalar_type() &&
              x.scalar_type() == bias.scalar_type() && x.size(1) == weight.size(1) &&
              bias.size(0) == weight.size(0) && x.size(0) > 0 && x.size(1) > 0 &&
              weight.size(0) > 0, name,
              ": expected x (B,K), weight (N,K), bias (N,), same fp32/bf16 dtype");
}

static at::Tensor decode_linear_mps(
    const at::Tensor& x_in, const at::Tensor& weight_in,
    const at::Tensor& bias_in, bool gelu) {
  check_decode_linear(x_in, weight_in, bias_in, "decode_linear");
  auto x = x_in.contiguous(), weight = weight_in.contiguous(), bias = bias_in.contiguous();
  const int B = x.size(0), K = x.size(1), N = weight.size(0);
  auto output = at::empty({B, N}, x.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_decode_linear(e, x, weight, bias, output, B, K, N, gelu,
                             tk_type_name(x));
  });
  return output;
}

static at::Tensor decode_linear_residual_mps(
    const at::Tensor& x_in, const at::Tensor& weight_in,
    const at::Tensor& bias_in, const at::Tensor& residual_in) {
  check_decode_linear(x_in, weight_in, bias_in, "decode_linear_residual");
  TORCH_CHECK(residual_in.device().is_mps() && residual_in.dim() == 2 &&
              residual_in.size(0) == x_in.size(0) &&
              residual_in.size(1) == weight_in.size(0) &&
              residual_in.scalar_type() == x_in.scalar_type(),
              "decode_linear_residual: residual must be (B,N) with x dtype");
  auto x = x_in.contiguous(), weight = weight_in.contiguous(), bias = bias_in.contiguous();
  auto residual = residual_in.contiguous();
  const int B = x.size(0), K = x.size(1), N = weight.size(0);
  auto output = at::empty({B, N}, x.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_decode_linear_residual(e, x, weight, bias, residual, output, B, K, N,
                                      tk_type_name(x));
  });
  return output;
}

static at::Tensor decode_linear_q8_mps(
    const at::Tensor& x_in, const at::Tensor& weight_in, const at::Tensor& bias_in,
    const at::Tensor& residual_in, bool gelu, bool use_residual) {
  TORCH_CHECK(x_in.device().is_mps() && weight_in.device().is_mps() &&
              bias_in.device().is_mps() && residual_in.device().is_mps() &&
              (x_in.scalar_type() == at::kFloat ||
              x_in.scalar_type() == at::kBFloat16) && x_in.dim() == 2 &&
              weight_in.scalar_type() == at::kByte && weight_in.dim() == 3 &&
              weight_in.size(2) == 34, "decode_linear_q8: bad input/packed weight");
  const int B = x_in.size(0), K = x_in.size(1), N = weight_in.size(0);
  TORCH_CHECK(B > 0 && K > 0 && N > 0 && K % 32 == 0 &&
              weight_in.size(1) == K / 32 && bias_in.dim() == 1 &&
              bias_in.size(0) == N && bias_in.scalar_type() == x_in.scalar_type(),
              "decode_linear_q8: packed K or bias mismatch");
  TORCH_CHECK(!use_residual || (residual_in.dim() == 2 && residual_in.size(0) == B &&
              residual_in.size(1) == N && residual_in.scalar_type() == x_in.scalar_type()),
              "decode_linear_q8: residual must be (B,N) with x dtype");
  auto x = x_in.contiguous(), weight = weight_in.contiguous(), bias = bias_in.contiguous();
  auto residual = residual_in.contiguous();
  auto output = at::empty({B, N}, x.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_decode_linear_q8(e, x, weight, bias, residual, output, B, K, N,
                                gelu, use_residual, tk_type_name(x));
  });
  return output;
}

static int check_decode_epilogue_weight(
    const at::Tensor& x, const at::Tensor& weight,
    const std::string& format, const char* name) {
  TORCH_CHECK(x.device().is_mps() && weight.device().is_mps() &&
              tk_is_float_dtype(x) && x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              name, ": x must be non-empty (B,K) fp32/fp16/bf16");
  const int K = x.size(1);
  if (format.empty()) {
    TORCH_CHECK(weight.scalar_type() == x.scalar_type() && weight.dim() == 2 &&
                weight.size(0) > 0 && weight.size(1) == K,
                name, ": dense weight must be (N,K) with x dtype");
    return weight.size(0);
  }
  int block_k = 0, block_bytes = 0;
  if (format == "q4_0") { block_k = 32; block_bytes = 18; }
  else if (format == "q8_0") { block_k = 32; block_bytes = 34; }
  else if (format == "q6_K") { block_k = 256; block_bytes = 210; }
  else {
    TORCH_CHECK(false,
                name, ": format must be empty/dense, q4_0, q8_0, or q6_K");
  }
  TORCH_CHECK(weight.scalar_type() == at::kByte && weight.dim() == 3 &&
              weight.size(0) > 0 && K % block_k == 0 &&
              weight.size(1) == K / block_k && weight.size(2) == block_bytes,
              name, ": packed weight must be uint8 (N,K/block_k,block_bytes)");
  return weight.size(0);
}

static at::Tensor decode_linear_epilogue_mps(
    const at::Tensor& x_in, const at::Tensor& weight_in,
    const at::Tensor& bias_in, const at::Tensor& residual_in,
    const std::string& format, int64_t activation,
    bool use_bias, bool use_residual) {
  TORCH_CHECK(activation >= 0 && activation <= 2,
              "decode_linear_epilogue: activation must be 0, 1, or 2");
  const int N = check_decode_epilogue_weight(
      x_in, weight_in, format, "decode_linear_epilogue");
  const int B = x_in.size(0), K = x_in.size(1);
  TORCH_CHECK(!use_bias || (bias_in.device().is_mps() && bias_in.dim() == 1 &&
              bias_in.size(0) == N && bias_in.scalar_type() == x_in.scalar_type()),
              "decode_linear_epilogue: bias must be (N,) with x dtype");
  TORCH_CHECK(!use_residual || (residual_in.device().is_mps() && residual_in.dim() == 2 &&
              residual_in.size(0) == B && residual_in.size(1) == N &&
              residual_in.scalar_type() == x_in.scalar_type()),
              "decode_linear_epilogue: residual must be (B,N) with x dtype");
  auto x = x_in.contiguous(), weight = weight_in.contiguous();
  auto bias = bias_in.to(x.scalar_type()).contiguous();
  auto residual = residual_in.to(x.scalar_type()).contiguous();
  auto output = at::empty({B, N}, x.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_decode_linear_epilogue(
        e, x, weight, bias, residual, output, B, K, N,
        static_cast<int>(activation), use_bias, use_residual, format,
        tk_type_name(x));
  });
  return output;
}

static at::Tensor decode_swiglu_mps(
    const at::Tensor& x_in, const at::Tensor& gate_weight_in,
    const at::Tensor& up_weight_in, const at::Tensor& gate_bias_in,
    const at::Tensor& up_bias_in, const std::string& format, bool use_bias) {
  const int N = check_decode_epilogue_weight(
      x_in, gate_weight_in, format, "decode_swiglu");
  const int up_N = check_decode_epilogue_weight(
      x_in, up_weight_in, format, "decode_swiglu");
  TORCH_CHECK(up_N == N && up_weight_in.sizes() == gate_weight_in.sizes(),
              "decode_swiglu: gate and up weights must have identical shapes");
  TORCH_CHECK(!use_bias ||
              (gate_bias_in.device().is_mps() && up_bias_in.device().is_mps() &&
               gate_bias_in.scalar_type() == x_in.scalar_type() &&
               up_bias_in.scalar_type() == x_in.scalar_type() &&
               gate_bias_in.dim() == 1 && up_bias_in.dim() == 1 &&
               gate_bias_in.size(0) == N && up_bias_in.size(0) == N),
              "decode_swiglu: both biases must be (N,) with x dtype");
  auto x = x_in.contiguous(), gate_weight = gate_weight_in.contiguous();
  auto up_weight = up_weight_in.contiguous();
  auto gate_bias = gate_bias_in.to(x.scalar_type()).contiguous();
  auto up_bias = up_bias_in.to(x.scalar_type()).contiguous();
  const int B = x.size(0), K = x.size(1);
  auto output = at::empty({B, N}, x.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_decode_swiglu(
        e, x, gate_weight, up_weight, gate_bias, up_bias, output,
        B, K, N, use_bias, format, tk_type_name(x));
  });
  return output;
}

static at::Tensor dequant_gather_mps(
    const at::Tensor& table_in, const at::Tensor& ids_in,
    const std::string& format, double scale) {
  int block_k = 0, block_bytes = 0;
  if (format == "q4_0") { block_k = 32; block_bytes = 18; }
  else if (format == "q8_0") { block_k = 32; block_bytes = 34; }
  else if (format == "q6_K") { block_k = 256; block_bytes = 210; }
  else { TORCH_CHECK(false, "dequant_gather: format must be q4_0, q8_0, or q6_K"); }
  TORCH_CHECK(table_in.device().is_mps() && ids_in.device().is_mps() &&
              table_in.scalar_type() == at::kByte && table_in.dim() == 3 &&
              table_in.size(0) > 0 && table_in.size(1) > 0 &&
              table_in.size(2) == block_bytes && ids_in.numel() > 0,
              "dequant_gather: table must be packed uint8 (rows,blocks,block_bytes)");
  auto table = table_in.contiguous(), ids = ids_in.to(at::kInt).contiguous();
  const int rows = table.size(0), columns = table.size(1) * block_k;
  auto shape = ids.sizes().vec();
  shape.push_back(columns);
  auto output = at::empty(shape, table.options().dtype(at::kHalf));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_dequant_gather(e, table, ids, output, rows, columns, ids.numel(),
                              static_cast<float>(scale), format);
  });
  return output;
}

static std::tuple<int, int> quantized_embedding_layout(const std::string& format) {
  if (format == "q4_0") return {32, 18};
  if (format == "q8_0") return {32, 34};
  if (format == "q4_K") return {256, 144};
  if (format == "q5_K") return {256, 176};
  if (format == "q6_K") return {256, 210};
  if (format == "q2_K") return {256, 84};
  if (format == "q3_K") return {256, 110};
  if (format == "iq4_nl") return {32, 18};
  if (format == "iq4_xs") return {256, 136};
  if (format == "kU4B8") return {128, 66};
  if (format == "kU4") return {128, 68};
  if (format == "hqq") return {64, 36};
  if (format == "fp8_e4m3") return {32, 34};
  if (format == "nvfp4") return {16, 9};
  if (format == "mxfp4") return {32, 17};
  TORCH_CHECK(false, "quantized_embedding: unsupported packed format '", format, "'");
}

static at::ScalarType quantized_embedding_dtype(const std::string& name) {
  if (name == "float16") return at::kHalf;
  if (name == "bfloat16") return at::kBFloat16;
  if (name == "float32") return at::kFloat;
  TORCH_CHECK(false,
              "quantized_embedding: output_dtype must be float16, bfloat16, or float32");
}

static at::Tensor quantized_embedding_mps(
    const at::Tensor& table_in, const at::Tensor& ids_in,
    const at::Tensor& add_in, const std::string& format, double scale,
    bool use_add, const std::string& output_dtype) {
  auto [block_k, block_bytes] = quantized_embedding_layout(format);
  TORCH_CHECK(table_in.device().is_mps() && ids_in.device().is_mps() &&
              add_in.device().is_mps() && table_in.scalar_type() == at::kByte &&
              table_in.dim() == 3 && table_in.size(0) > 0 && table_in.size(1) > 0 &&
              table_in.size(2) == block_bytes && ids_in.numel() > 0,
              "quantized_embedding: table must be packed uint8 (rows,blocks,block_bytes)");
  const auto dtype = quantized_embedding_dtype(output_dtype);
  const int rows = table_in.size(0), columns = table_in.size(1) * block_k;
  auto shape = ids_in.sizes().vec();
  shape.push_back(columns);
  TORCH_CHECK(!use_add || (add_in.scalar_type() == dtype && add_in.sizes() == shape),
              "quantized_embedding: add must match output shape and output_dtype");
  auto table = table_in.contiguous(), ids = ids_in.to(at::kInt).contiguous();
  auto add = add_in.to(dtype).contiguous();
  auto output = at::empty(shape, table.options().dtype(dtype));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_quantized_embedding(
        e, table, ids, add, output, rows, columns, ids.numel(),
        static_cast<float>(scale), use_add, format, tk_type_name(output));
  });
  return output;
}

static at::Tensor quantized_embedding_bag_mps(
    const at::Tensor& table_in, const at::Tensor& ids_in,
    const at::Tensor& offsets_in, const at::Tensor& sample_weights_in,
    const std::string& format, double scale, bool use_weights,
    bool mean_mode, const std::string& output_dtype) {
  auto [block_k, block_bytes] = quantized_embedding_layout(format);
  TORCH_CHECK(table_in.device().is_mps() && ids_in.device().is_mps() &&
              offsets_in.device().is_mps() && sample_weights_in.device().is_mps() &&
              table_in.scalar_type() == at::kByte && table_in.dim() == 3 &&
              table_in.size(0) > 0 && table_in.size(1) > 0 &&
              table_in.size(2) == block_bytes && ids_in.dim() == 1 &&
              offsets_in.dim() == 1 && offsets_in.size(0) >= 2,
              "quantized_embedding_bag: expected packed table, flat ids, and offsets");
  TORCH_CHECK(!use_weights ||
              (sample_weights_in.dim() == 1 && sample_weights_in.numel() == ids_in.numel()),
              "quantized_embedding_bag: sample_weights must have one value per id");
  const auto dtype = quantized_embedding_dtype(output_dtype);
  const int rows = table_in.size(0), columns = table_in.size(1) * block_k;
  const int bags = offsets_in.size(0) - 1;
  auto table = table_in.contiguous(), ids = ids_in.to(at::kInt).contiguous();
  auto offsets = offsets_in.to(at::kInt).contiguous();
  auto weights = sample_weights_in.to(at::kFloat).contiguous();
  auto output = at::empty({bags, columns}, table.options().dtype(dtype));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_quantized_embedding_bag(
        e, table, ids, offsets, weights, output, rows, columns, ids.numel(), bags,
        static_cast<float>(scale), use_weights, mean_mode, format,
        tk_type_name(output));
  });
  return output;
}

static std::tuple<at::Tensor, at::Tensor> lm_head_constrained_mps(
    const at::Tensor& h_in, const at::Tensor& weight_in, const at::Tensor& bias_in,
    const at::Tensor& forbidden_in, const at::Tensor& previous_in,
    int64_t eos_id, bool forbid_eos) {
  TORCH_CHECK(h_in.device().is_mps() && weight_in.device().is_mps() &&
              bias_in.device().is_mps() && forbidden_in.device().is_mps() &&
              previous_in.device().is_mps() && tk_is_float_dtype(h_in) && h_in.dim() == 2 &&
              weight_in.dim() == 2 && h_in.size(1) == weight_in.size(1) &&
              h_in.scalar_type() == weight_in.scalar_type(),
              "lm_head_constrained: h (T,K), weight (V,K), same float dtype");
  const int T = h_in.size(0), K = h_in.size(1), V = weight_in.size(0);
  TORCH_CHECK(T > 0 && K > 0 && V > 0,
              "lm_head_constrained: T, K, and V must be positive");
  TORCH_CHECK(forbidden_in.scalar_type() == at::kByte && forbidden_in.dim() == 2 &&
              forbidden_in.size(0) == V && forbidden_in.size(1) == V,
              "lm_head_constrained: forbidden must be uint8 (V,V)");
  TORCH_CHECK(previous_in.numel() == T, "lm_head_constrained: previous must have T ids");
  TORCH_CHECK(!forbid_eos || (eos_id >= 0 && eos_id < V),
              "lm_head_constrained: invalid eos_id");
  const int use_bias = bias_in.numel() > 1 ? 1 : 0;
  TORCH_CHECK(!use_bias || (bias_in.dim() == 1 && bias_in.size(0) == V),
              "lm_head_constrained: bias must be (V,) or scalar placeholder");
  constexpr int TILE_V = 256;
  const int num_vtiles = (V + TILE_V - 1) / TILE_V;
  auto h = h_in.contiguous(), weight = weight_in.contiguous();
  auto bias = bias_in.to(at::kFloat).contiguous();
  auto forbidden = forbidden_in.contiguous(), previous = previous_in.to(at::kInt).contiguous();
  auto f32 = h.options().dtype(at::kFloat), i32 = h.options().dtype(at::kInt);
  auto part_max = at::empty({T, num_vtiles}, f32);
  auto part_sum = at::empty({T, num_vtiles}, f32);
  auto part_best = at::empty({T, num_vtiles}, f32);
  auto part_id = at::empty({T, num_vtiles}, i32);
  auto token = at::empty({T}, i32), logprob = at::empty({T}, f32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_lm_head_constrained_partials(
        e, h, weight, bias, forbidden, previous, part_max, part_sum, part_best, part_id,
        V, K, TILE_V, num_vtiles, use_bias, static_cast<int>(eos_id),
        forbid_eos ? 1 : 0, T, tk_type_name(h));
    tk::launch_lm_head_constrained_reduce(
        e, part_max, part_sum, part_best, part_id, token, logprob, num_vtiles, T);
  });
  return {token, logprob};
}

static int check_masked_lm_weight(
    const at::Tensor& h, const at::Tensor& weight,
    const std::string& format, const char* name) {
  TORCH_CHECK(h.device().is_mps() && weight.device().is_mps() &&
              tk_is_float_dtype(h) && h.dim() == 2 &&
              h.size(0) > 0 && h.size(1) > 0,
              name, ": h must be non-empty (T,K) fp32/fp16/bf16");
  const int K = h.size(1);
  if (format.empty()) {
    TORCH_CHECK(weight.scalar_type() == h.scalar_type() && weight.dim() == 2 &&
                weight.size(0) > 0 && weight.size(1) == K,
                name, ": dense weight must be (V,K) with h dtype");
    return weight.size(0);
  }
  int block_k = 0, block_bytes = 0;
  if (format == "q4_0") { block_k = 32; block_bytes = 18; }
  else if (format == "q8_0") { block_k = 32; block_bytes = 34; }
  else if (format == "q6_K") { block_k = 256; block_bytes = 210; }
  else {
    TORCH_CHECK(false,
                name, ": format must be empty/dense, q4_0, q8_0, or q6_K");
  }
  TORCH_CHECK(weight.scalar_type() == at::kByte && weight.dim() == 3 &&
              weight.size(0) > 0 && K % block_k == 0 &&
              weight.size(1) == K / block_k && weight.size(2) == block_bytes,
              name, ": invalid packed weight shape");
  return weight.size(0);
}

static std::tuple<at::Tensor, at::Tensor> lm_head_masked_mps(
    const at::Tensor& h_in, const at::Tensor& weight_in,
    const at::Tensor& bias_in, const at::Tensor& allow_mask_in,
    const std::string& format, int64_t topk, bool normalize_allowed) {
  const int V = check_masked_lm_weight(
      h_in, weight_in, format, "lm_head_masked");
  const int T = h_in.size(0), K = h_in.size(1), words = (V + 31) / 32;
  TORCH_CHECK(topk >= 1 && topk <= 8 && topk <= V,
              "lm_head_masked: topk must be in [1,min(8,V)]");
  TORCH_CHECK(allow_mask_in.device().is_mps() && allow_mask_in.dim() == 2 &&
              allow_mask_in.size(0) == T && allow_mask_in.size(1) == words,
              "lm_head_masked: allow_mask must be (T,ceil(V/32))");
  const bool use_bias = bias_in.numel() > 1;
  TORCH_CHECK(!use_bias || (bias_in.dim() == 1 && bias_in.size(0) == V),
              "lm_head_masked: bias must be (V,) or scalar placeholder");
  constexpr int MASKED_TILE_V = 128;
  const int num_vtiles = (V + MASKED_TILE_V - 1) / MASKED_TILE_V;
  auto h = h_in.contiguous(), weight = weight_in.contiguous();
  auto bias = use_bias ? bias_in.to(at::kFloat).contiguous()
                       : at::zeros({1}, h.options().dtype(at::kFloat));
  auto allow_mask = allow_mask_in.to(at::kInt).contiguous();
  auto f32 = h.options().dtype(at::kFloat), i32 = h.options().dtype(at::kInt);
  auto part_val = at::empty({T, num_vtiles, topk}, f32);
  auto part_id = at::empty({T, num_vtiles, topk}, i32);
  auto part_max = at::empty({T, num_vtiles}, f32);
  auto part_sum = at::empty({T, num_vtiles}, f32);
  auto ids = at::empty({T, topk}, i32), logprobs = at::empty({T, topk}, f32);
  tk_encode([&](TorchEncoder& e) {
    tk::launch_lm_head_masked_partials(
        e, h, weight, bias, allow_mask, part_val, part_id, part_max, part_sum,
        V, K, MASKED_TILE_V, num_vtiles, static_cast<int>(topk), use_bias,
        normalize_allowed, words, T, format, tk_type_name(h));
    tk::launch_lm_head_masked_reduce(
        e, part_val, part_id, part_max, part_sum, ids, logprobs,
        num_vtiles, static_cast<int>(topk), T);
  });
  return {ids, logprobs};
}

static std::tuple<at::Tensor, at::Tensor> lm_head_candidates_mps(
    const at::Tensor& h_in, const at::Tensor& weight_in,
    const at::Tensor& bias_in, const at::Tensor& candidate_ids_in,
    const at::Tensor& offsets_in, const std::string& format, int64_t topk) {
  const int V = check_masked_lm_weight(
      h_in, weight_in, format, "lm_head_candidates");
  const int T = h_in.size(0), K = h_in.size(1);
  TORCH_CHECK(topk >= 1 && topk <= 8 && topk <= V,
              "lm_head_candidates: topk must be in [1,min(8,V)]");
  TORCH_CHECK(candidate_ids_in.device().is_mps() && offsets_in.device().is_mps() &&
              candidate_ids_in.dim() == 1 && offsets_in.dim() == 1 &&
              offsets_in.size(0) == T + 1,
              "lm_head_candidates: flat candidate_ids and T+1 offsets required");
  const bool use_bias = bias_in.numel() > 1;
  TORCH_CHECK(!use_bias || (bias_in.dim() == 1 && bias_in.size(0) == V),
              "lm_head_candidates: bias must be (V,) or scalar placeholder");
  auto h = h_in.contiguous(), weight = weight_in.contiguous();
  auto candidate_ids = candidate_ids_in.to(at::kInt).contiguous();
  auto offsets = offsets_in.to(at::kInt).contiguous();
  auto bias = use_bias ? bias_in.to(at::kFloat).contiguous()
                       : at::zeros({1}, h.options().dtype(at::kFloat));
  auto ids = at::empty({T, topk}, h.options().dtype(at::kInt));
  auto logprobs = at::empty({T, topk}, h.options().dtype(at::kFloat));
  tk_encode([&](TorchEncoder& e) {
    tk::launch_lm_head_candidates(
        e, h, weight, candidate_ids, offsets, bias, ids, logprobs,
        V, K, static_cast<int>(topk), use_bias, T, format, tk_type_name(h));
  });
  return {ids, logprobs};
}

static at::Tensor flux_gelu_erf_mps(
    const at::Tensor& x_in, const at::Tensor& weight_in, const at::Tensor& bias_in) {
  TORCH_CHECK(x_in.device().is_mps() && weight_in.device().is_mps() &&
              bias_in.device().is_mps() && x_in.dim() == 2 && weight_in.dim() == 2 &&
              x_in.size(1) == weight_in.size(0) && x_in.scalar_type() == weight_in.scalar_type() &&
              bias_in.dim() == 1 && bias_in.size(0) == weight_in.size(1) &&
              bias_in.scalar_type() == x_in.scalar_type() &&
              (x_in.scalar_type() == at::kFloat || x_in.scalar_type() == at::kBFloat16),
              "flux_gelu_erf: expected compatible fp32/bf16 GEMM inputs");
  auto x = x_in.contiguous(), weight = weight_in.contiguous(), bias = bias_in.contiguous();
  const int N = x.size(0), K = x.size(1), M = weight.size(1);
  TORCH_CHECK(N > 0 && K > 0 && M > 0 && N % 32 == 0 && M % 32 == 0 && K % 16 == 0,
              "flux_gelu_erf: N%32, M%32, K%16 required");
  auto output = at::empty({N, M}, x.options());
  tk_encode([&](TorchEncoder& e) {
    tk::launch_flux_gelu_erf(e, output, x, weight, bias, N, K, M, tk_type_name(x));
  });
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_set_library", &tk_set_library, "set the metallib path");
  m.def("layernorm", &layernorm_mps, "ThunderMittens LayerNorm (MPS)");
  m.def("layernorm_bwd_dx", &layernorm_bwd_dx_mps, "ThunderMittens LayerNorm backward dX (MPS)");
  m.def("layernorm_bwd_fused", &layernorm_bwd_fused_mps,
        "ThunderMittens fused LayerNorm backward -> [dX, dweight, dbias] (MPS)");
  m.def("add_rt", &add_rt_mps, "ThunderMittens add_rt elementwise add (MPS)");
  m.def("matmul_custom", &matmul_custom_mps, "ThunderMittens matmul_custom GEMM (MPS)");
  m.def("attn_fwd", &attn_fwd_mps, "ThunderMittens attention forward (MPS)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("softcap") = 0.0, pybind11::arg("sinks") = pybind11::none());
  m.def("rms_norm", &rms_norm_mps, "ThunderMittens RMSNorm (MPS)");
  m.def("rms_norm_bwd_dx", &rms_norm_bwd_dx_mps, "ThunderMittens RMSNorm backward dX (MPS)");
  m.def("rms_norm_bwd_fused", &rms_norm_bwd_fused_mps,
        "ThunderMittens fused RMSNorm backward -> [dX, dweight] (MPS)");
  m.def("rms_norm_add", &rms_norm_add_mps, "ThunderMittens fused residual-add + RMSNorm (MPS)");
  m.def("layernorm_add", &layernorm_add_mps, "ThunderMittens fused residual-add + LayerNorm (MPS)");
  m.def("rms_norm_add_fp8", &rms_norm_add_fp8_mps, "ThunderMittens fused add+rms_norm static fp8 (MPS)");
  m.def("rms_norm_add_fp8_dyn", &rms_norm_add_fp8_dyn_mps, "ThunderMittens fused add+rms_norm dyn fp8 (MPS)");
  m.def("layernorm_add_fp8", &layernorm_add_fp8_mps, "ThunderMittens fused add+layernorm static fp8 (MPS)");
  m.def("layernorm_add_fp8_dyn", &layernorm_add_fp8_dyn_mps, "ThunderMittens fused add+layernorm dyn fp8 (MPS)");
  m.def("softmax", &softmax_mps, "ThunderMittens softmax (MPS)");
  m.def("rotary", &rotary_mps, "ThunderMittens rotary/RoPE (MPS)");
  m.def("gelu", &gelu_mps, "ThunderMittens GELU (MPS)");
  m.def("gelu_bwd", &gelu_bwd_mps, "ThunderMittens GELU backward (MPS)");
  m.def("dropout", &dropout_mps, "ThunderMittens inverted dropout fwd/bwd (MPS)");
  m.def("adamw", &adamw_mps, "ThunderMittens AdamW optimizer step (MPS)");
  m.def("adamw_masked", &adamw_masked_mps, "ThunderMittens masked AdamW optimizer step (MPS)");
  m.def("embedding_lookup", &embedding_lookup_mps, "ThunderMittens token embedding lookup (MPS)");
  m.def("embedding_backward", &embedding_backward_mps,
        "ThunderMittens embedding backward / scatter-add grad (MPS)");
  m.def("embedding_backward_sorted", &embedding_backward_sorted_mps,
        "ThunderMittens embedding backward (sorted-segment, atomic-free) (MPS)");
  m.def("build_multimodal_src", &build_multimodal_src_mps,
        "ThunderMittens on-device multimodal src builder (MPS)");
  m.def("merge_multimodal_spans", &merge_multimodal_spans_mps,
        "ThunderMittens multimodal span merge (MPS)");
  m.def("glu", &glu_mps, "ThunderMittens GLU-family activation (MPS)");
  m.def("glu_backward", &glu_bwd_mps, "ThunderMittens GLU-family backward (MPS)");
  m.def("hadamard", &hadamard_mps, "ThunderMittens Hadamard/FWHT (MPS)");
  m.def("kv_cache_scatter", &kv_cache_scatter_mps, "ThunderMittens KV cache scatter (MPS)");
  m.def("kv_cache_gather", &kv_cache_gather_mps, "ThunderMittens KV cache gather (MPS)");
  m.def("kv_cache_copy_blocks", &kv_cache_copy_blocks_mps, "ThunderMittens KV cache block copy (MPS)");
  m.def("beam_build_copy_pairs", &beam_build_copy_pairs_mps, "ThunderMittens beam KV reorder copy-pair builder (MPS)");
  m.def("beam_remap_block_table", &beam_remap_block_table_mps, "ThunderMittens zero-copy beam block-table remap (MPS)");
  m.def("kv_cache_scales", &kv_cache_scales_mps, "ThunderMittens KV cache fp8 scales (MPS)");
  m.def("kv_cache_gather_fp8", &kv_cache_gather_fp8_mps, "fp8 KV gather + upconvert (MPS)",
        pybind11::arg("key_cache"), pybind11::arg("value_cache"), pybind11::arg("block_table"),
        pybind11::arg("cu_seq_lens"), pybind11::arg("k_scale"), pybind11::arg("v_scale"),
        pybind11::arg("num_tokens"), pybind11::arg("fmt") = 0);
  m.def("kv_cache_scale_update", &kv_cache_scale_update_mps,
        "incremental KV scale running-max update (MPS)");
  m.def("paged_attention", &paged_attention_mps, "ThunderMittens paged decode attention (MPS)");
  m.def("paged_attention_alibi", &paged_attention_alibi_mps, "ThunderMittens paged decode with ALiBi (MPS)");
  m.def("paged_attention_block_sparse", &paged_attention_block_sparse_mps, "ThunderMittens block-sparse paged decode (MPS)");
  m.def("paged_attention_staged", &paged_attention_staged_mps, "ThunderMittens GQA KV-reuse staged decode (MPS)");
  m.def("paged_attention_xcache", &paged_attention_xcache_mps, "ThunderMittens vLLM x-packed cache decode (MPS)");
  m.def("kv_cache_scatter_fp8", &kv_cache_scatter_fp8_mps, "ThunderMittens fp8 KV cache scatter (MPS)");
  m.def("paged_attention_fp8", &paged_attention_fp8_mps, "ThunderMittens fp8 paged attention (MPS)");
  m.def("rope_kv_insert", &rope_kv_insert_mps, "ThunderMittens fused RoPE + paged-KV insert (MPS)");
  m.def("rope_kv_insert_norm", &rope_kv_insert_norm_mps, "ThunderMittens fused K-norm + RoPE + KV insert (MPS)");
  m.def("rope_q", &rope_q_mps, "ThunderMittens Q-path RoPE (+optional norm) (MPS)");
  m.def("mla_q_norm_rope", &mla_q_norm_rope_mps, "ThunderMittens DeepSeek MLA Q-path norm+interleaved-rope (MPS)");
  m.def("mla_kv_insert", &mla_kv_insert_mps, "ThunderMittens DeepSeek MLA classic KV-insert (MPS)");
  m.def("mla_decode", &mla_decode_mps, "ThunderMittens DeepSeek MLA latent flash-decode (MPS)");
  m.def("mla_decode_fp8", &mla_decode_fp8_mps, "ThunderMittens DeepSeek-V4 packed fp8 latent decode (MPS)");
  m.def("mla_decode_fp8_sparse", &mla_decode_fp8_sparse_mps, "ThunderMittens DeepSeek-V4 sparse latent decode (MPS)");
  m.def("mla_kv_insert_fp8", &mla_kv_insert_fp8_mps, "ThunderMittens DeepSeek-V4 packed fp8 MLA KV-insert (MPS)");
  m.def("paged_attention_v2", &paged_attention_v2_mps, "ThunderMittens long-context paged attention (MPS)",
        pybind11::arg("q"), pybind11::arg("key_cache"), pybind11::arg("value_cache"),
        pybind11::arg("block_table"), pybind11::arg("context_lens"), pybind11::arg("scale"),
        pybind11::arg("partition_size"), pybind11::arg("window"),
        pybind11::arg("softcap") = 0.0, pybind11::arg("sinks") = pybind11::none());
  m.def("cascade_attention", &cascade_attention_mps, "ThunderMittens cascade / shared-prefix attention (MPS)");
  m.def("cascade_attention_multi", &cascade_attention_multi_mps, "ThunderMittens N-level cascade attention (MPS)");
  m.def("cascade_attention_fp8", &cascade_attention_fp8_mps, "ThunderMittens fp8-prefix cascade attention (MPS)");
  m.def("paged_attention_v2_fp8", &paged_attention_v2_fp8_mps, "ThunderMittens long-context fp8 paged attention (MPS)",
        pybind11::arg("q"), pybind11::arg("key_cache"), pybind11::arg("value_cache"),
        pybind11::arg("block_table"), pybind11::arg("context_lens"), pybind11::arg("k_scale"),
        pybind11::arg("v_scale"), pybind11::arg("scale"), pybind11::arg("partition_size"),
        pybind11::arg("fmt"), pybind11::arg("window"),
        pybind11::arg("softcap") = 0.0, pybind11::arg("sinks") = pybind11::none());
  m.def("moe_route_topk", &moe_route_topk_mps, "ThunderMittens MoE top-k routing (MPS)");
  m.def("moe_permute", &moe_permute_mps, "ThunderMittens MoE permute (MPS)");
  m.def("moe_pad_schedule", &moe_pad_schedule_mps, "ThunderMittens MoE padded schedule (MPS)");
  m.def("moe_gather", &moe_gather_mps, "ThunderMittens MoE permuted-input gather (MPS)");
  m.def("moe_grouped_gemm", &moe_grouped_gemm_mps, "ThunderMittens MoE grouped expert GEMM (MPS)");
  m.def("moe_grouped_gemm_rect", &moe_grouped_gemm_rect_mps, "ThunderMittens MoE rectangular grouped GEMM (MPS)");
  m.def("moe_grouped_gemm_swiglu", &moe_grouped_gemm_swiglu_mps, "ThunderMittens MoE fused SiLU-GLU GEMM1 (MPS)");
  m.def("tau_tail", &tau_tail_mps, "tau_tail QKV tail scaling (MPS)");
  m.def("packbits", &packbits_mps, "packbits (MPS)", pybind11::arg("x"),
        pybind11::arg("bit_order_big") = true);
  m.def("segment_packbits", &segment_packbits_mps, "segment_packbits (MPS)",
        pybind11::arg("x"), pybind11::arg("input_indptr"), pybind11::arg("output_indptr"),
        pybind11::arg("total_output_bytes"), pybind11::arg("bit_order_big") = true);
  m.def("permute_cols", &permute_cols_mps, "permute_cols 16-bit column gather (MPS)");
  m.def("tq_encode", &tq_encode_mps, "TurboQuant KV encode (MPS)");
  m.def("tq_decode", &tq_decode_mps, "TurboQuant KV decode (MPS)");
  m.def("indexer_k_quant_and_cache", &indexer_k_quant_and_cache_mps,
        "DeepSeek-V3.2 indexer K quant-and-cache (MPS)", pybind11::arg("k"),
        pybind11::arg("slot_mapping"), pybind11::arg("code_cache"), pybind11::arg("scale_cache"),
        pybind11::arg("quant_block_size") = 128, pybind11::arg("ue8m0") = false);
  m.def("indexer_k_gather", &indexer_k_gather_mps, "indexer cache gather + dequant (MPS)",
        pybind11::arg("code_cache"), pybind11::arg("scale_cache"), pybind11::arg("slots"),
        pybind11::arg("head_dim"), pybind11::arg("quant_block_size") = 128);
  m.def("minference_block_mask", &minference_block_mask_mps,
        "MInference decode block-mask builder (MPS)", pybind11::arg("vertical_indexes"),
        pybind11::arg("slash_indexes"), pybind11::arg("context_lens"),
        pybind11::arg("max_blocks"), pybind11::arg("block_size"),
        pybind11::arg("vertical_topk") = 1 << 30, pybind11::arg("slash_topk") = 1 << 30,
        pybind11::arg("last_n_blocks") = 1);
  m.def("quadratic_transform", &quadratic_transform_mps, "quadratic logit transform (MPS)");
  m.def("top_nsigma_mask", &top_nsigma_mask_mps, "top-nsigma mask (MPS)");
  m.def("top_a_mask", &top_a_mask_mps, "top-A mask (MPS)");
  m.def("epsilon_cutoff_mask", &epsilon_cutoff_mask_mps, "epsilon-cutoff mask (MPS)");
  m.def("eta_cutoff_mask", &eta_cutoff_mask_mps, "eta-cutoff mask (MPS)");
  m.def("xtc_mask", &xtc_mask_mps, "XTC mask (MPS)");
  m.def("skew_transform", &skew_transform_mps, "skew prob transform (MPS)");
  m.def("top_k_renorm", &top_k_renorm_mps, "top-k renormalized probs (MPS)");
  m.def("top_p_renorm", &top_p_renorm_mps, "top-p renormalized probs (MPS)");
  m.def("no_repeat_ngram_mask", &no_repeat_ngram_mask_mps, "no-repeat-ngram mask (MPS)");
  m.def("dry_penalty", &dry_penalty_mps, "DRY repetition penalty (MPS)");
  m.def("rms_norm_add_int8_dyn", &rms_norm_add_int8_dyn_mps,
        "fused add + rms_norm + dynamic per-row int8 (MPS)");
  m.def("layernorm_add_int8_dyn", &layernorm_add_int8_dyn_mps,
        "fused add + layernorm + dynamic per-row int8 (MPS)");
  m.def("rms_norm_add_per_block", &rms_norm_add_per_block_mps,
        "fused add + rms_norm + per-128-block quant (MPS)", pybind11::arg("x"),
        pybind11::arg("residual"), pybind11::arg("weight"), pybind11::arg("eps"),
        pybind11::arg("int8"), pybind11::arg("ue8m0") = false);
  m.def("layernorm_add_per_block", &layernorm_add_per_block_mps,
        "fused add + layernorm + per-128-block quant (MPS)", pybind11::arg("x"),
        pybind11::arg("residual"), pybind11::arg("weight"), pybind11::arg("bias"),
        pybind11::arg("eps"), pybind11::arg("int8"), pybind11::arg("ue8m0") = false);
  m.def("silu_mul_quant_fp8", &silu_mul_quant_fp8_mps,
        "fused gated-activation -> per-token fp8 (MPS)", pybind11::arg("x"),
        pybind11::arg("gate"), pybind11::arg("mode") = 0, pybind11::arg("alpha") = 1.702,
        pybind11::arg("limit") = 7.0);
  m.def("silu_mul_quant_int8", &silu_mul_quant_int8_mps,
        "fused gated-activation -> per-token int8 (MPS)", pybind11::arg("x"),
        pybind11::arg("gate"), pybind11::arg("mode") = 0, pybind11::arg("alpha") = 1.702,
        pybind11::arg("limit") = 7.0);
  m.def("silu_mul_quant_fp8_group", &silu_mul_quant_fp8_group_mps,
        "fused gated-activation -> per-group fp8 (MPS)", pybind11::arg("x"),
        pybind11::arg("gate"), pybind11::arg("group_size") = 128,
        pybind11::arg("ue8m0") = false, pybind11::arg("mode") = 0,
        pybind11::arg("alpha") = 1.702, pybind11::arg("limit") = 7.0);
  m.def("fake_quant_int8", &fake_quant_int8_mps,
        "BitNet one-pass per-token int8 fake quant (MPS)");
  m.def("silu_mul_fake_quant_int8", &silu_mul_fake_quant_int8_mps,
        "BitNet fused gated activation + int8 fake quant (MPS)",
        pybind11::arg("x"), pybind11::arg("gate"), pybind11::arg("mode") = 0,
        pybind11::arg("alpha") = 1.702, pybind11::arg("limit") = 7.0);
  m.def("weight_quant_ternary", &weight_quant_ternary_mps,
        "BitNet ternary weight quantization (MPS)", pybind11::arg("w"),
        pybind11::arg("group_k") = 32);
  m.def("weight_quant_ternary_pt", &weight_quant_ternary_pt_mps,
        "BitNet per-tensor ternary weight quantization (MPS)");
  m.def("gemm_v3", &gemm_v3_mps, "2x2-warp staged GEMM variant (MPS)");
  m.def("qgemm_bwd", &qgemm_bwd_mps,
        "BitNet ternary backward GEMM: grad_y @ dequant(bitnet blocks) (MPS)");
  m.def("quantize_tq2_0", &quantize_tq2_0_mps,
        "llama.cpp-native TQ2_0 pack: W -> (ggml block_tq2_0 bytes, w_deq bf16) (MPS)");
  m.def("qdequant", &qdequant_mps,
        "Packed quant blocks -> fp16 dense for qdequant-instantiated formats (MPS)");
  m.def("ternary_stats", &ternary_stats_mps,
        "Ternary health: packed wq -> per-row {-1,0,+1} code counts int32 (MPS)");
  m.def("code_flip_count", &code_flip_count_mps,
        "Ternary health: per-row count of differing codes between two packs (MPS)");
  m.def("fake_quant_fp8", &fake_quant_fp8_mps,
        "Per-tensor e4m3 fake quant -> (x_fq, scale) (MPS)");
  m.def("qgemm_w2a8_fused", &qgemm_w2a8_fused_mps,
        "BitNet fused per-token int8 act-quant + W2A8 GEMM (MPS)");
  m.def("attn_decode", &attn_decode_mps,
        "Batch-1 GQA attention decode against dense KV cache (MPS)");
  m.def("quantize_per_group_fp8", &quantize_per_group_fp8_mps,
        "per-group dynamic fp8 quant (MPS)", pybind11::arg("x"),
        pybind11::arg("group_size") = 128, pybind11::arg("ue8m0") = false);
  m.def("quantize_per_group_int8", &quantize_per_group_int8_mps,
        "per-group dynamic int8 quant (MPS)", pybind11::arg("x"),
        pybind11::arg("group_size") = 128);
  m.def("quantize_per_token_int8_azp", &quantize_per_token_int8_azp_mps,
        "asymmetric per-token int8 quant (MPS)");
  m.def("qgemm_w8a8_azp", &qgemm_w8a8_azp_mps, "azp-corrected W8A8 GEMM (MPS)");
  m.def("gdn_recur", &gdn_recur_mps,
        "ThunderMittens GatedDeltaNet linear attention (MPS)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("g"),
        pybind11::arg("beta"), pybind11::arg("state_pool"), pybind11::arg("cu_seqlens"),
        pybind11::arg("slot_mapping"), pybind11::arg("load_initial") = true);
  m.def("selective_scan", &selective_scan_mps,
        "ThunderMittens Mamba-1 selective scan (MPS)",
        pybind11::arg("u"), pybind11::arg("delta"), pybind11::arg("A"), pybind11::arg("B"),
        pybind11::arg("C"), pybind11::arg("state"), pybind11::arg("D") = pybind11::none(),
        pybind11::arg("delta_bias") = pybind11::none(), pybind11::arg("z") = pybind11::none(),
        pybind11::arg("delta_softplus") = true);
  m.def("selective_scan_varlen_apc", &selective_scan_varlen_apc_mps,
        "ThunderMittens varlen S6 scan with APC paged state (MPS)",
        pybind11::arg("u"), pybind11::arg("delta"), pybind11::arg("A"), pybind11::arg("B"),
        pybind11::arg("C"), pybind11::arg("query_start_loc"), pybind11::arg("cache_indices"),
        pybind11::arg("has_initial_state"), pybind11::arg("state"),
        pybind11::arg("block_idx_first_scheduled_token"),
        pybind11::arg("block_idx_last_scheduled_token"), pybind11::arg("initial_state_idx"),
        pybind11::arg("cu_chunk_seqlen"), pybind11::arg("last_chunk_indices"),
        pybind11::arg("block_size"), pybind11::arg("cache_indices_stride"),
        pybind11::arg("use_chunk_metadata"), pybind11::arg("D") = c10::nullopt,
        pybind11::arg("delta_bias") = c10::nullopt, pybind11::arg("z") = c10::nullopt,
        pybind11::arg("delta_softplus") = true, pybind11::arg("null_block_id") = -1);
  m.def("selective_scan_varlen", &selective_scan_varlen_mps,
        "ThunderMittens varlen Mamba-1 selective scan (MPS)",
        pybind11::arg("u"), pybind11::arg("delta"), pybind11::arg("A"), pybind11::arg("B"),
        pybind11::arg("C"), pybind11::arg("query_start_loc"), pybind11::arg("state"),
        pybind11::arg("D") = pybind11::none(), pybind11::arg("delta_bias") = pybind11::none(),
        pybind11::arg("z") = pybind11::none(), pybind11::arg("cache_indices") = pybind11::none(),
        pybind11::arg("has_initial_state") = pybind11::none(),
        pybind11::arg("delta_softplus") = true, pybind11::arg("null_block_id") = -1);
  m.def("qk_norm_rope", &qk_norm_rope_mps,
        "ThunderMittens fused per-head QK-RMSNorm + RoPE on packed QKV (MPS)",
        pybind11::arg("qkv"), pybind11::arg("q_weight"), pybind11::arg("k_weight"),
        pybind11::arg("cos"), pybind11::arg("sin"), pybind11::arg("positions"),
        pybind11::arg("num_heads_q"), pybind11::arg("num_heads_k"), pybind11::arg("num_heads_v"),
        pybind11::arg("eps") = 1e-6, pybind11::arg("interleaved") = false,
        pybind11::arg("gemma") = false);
  m.def("moe_route_grouped", &moe_route_grouped_mps,
        "ThunderMittens DeepSeek-style grouped MoE routing (MPS)",
        pybind11::arg("logits"), pybind11::arg("bias"), pybind11::arg("k"),
        pybind11::arg("n_group"), pybind11::arg("topk_group"), pybind11::arg("renormalize"),
        pybind11::arg("routed_scaling_factor"), pybind11::arg("scoring_func"));
  m.def("moe_grouped_gemm_rect_q", &moe_grouped_gemm_rect_q_mps, "ThunderMittens quantized grouped expert GEMM (MPS)");
  m.def("moe_grouped_gemm_swiglu_q", &moe_grouped_gemm_swiglu_q_mps, "ThunderMittens quantized fused SwiGLU expert GEMM1 (MPS)");
  m.def("moe_finalize", &moe_finalize_mps, "ThunderMittens MoE finalize reduce (MPS)");
  m.def("moe_grouped_gemm_bwd_dx", &moe_grouped_gemm_bwd_dx_mps,
        "MoE backward: dX = dY @ W[e]^T over the padded schedule (MPS)");
  m.def("moe_grouped_gemm_bwd_dw", &moe_grouped_gemm_bwd_dw_mps,
        "MoE backward: dW[e] = A^T @ dY per padded expert segment (MPS)",
        pybind11::arg("A"), pybind11::arg("dy"), pybind11::arg("off_pad"),
        pybind11::arg("num_experts"));
  m.def("moe_finalize_bwd", &moe_finalize_bwd_mps,
        "MoE backward of finalize -> (grad_expert_out zero-padded, grad_weights) (MPS)");
  m.def("moe_gather_bwd", &moe_gather_bwd_mps,
        "MoE backward of gather: dx[t] = sum_k dA[inv_pad[t*k+j]] (MPS)",
        pybind11::arg("dA"), pybind11::arg("inv_pad"), pybind11::arg("k"));
  m.def("argmax_sample", &argmax_sample_mps, "ThunderMittens greedy argmax sampling (MPS)");
  m.def("sample_categorical", &sample_categorical_mps, "ThunderMittens Gumbel-max sampling (MPS)");
  m.def("top_k_sample", &top_k_sample_mps, "ThunderMittens top-k sampling (MPS)");
  m.def("top_p_sample", &top_p_sample_mps, "ThunderMittens top-p nucleus sampling (MPS)");
  m.def("min_p_sample", &min_p_sample_mps, "ThunderMittens min-p sampling (MPS)");
  m.def("typical_p_sample", &typical_p_sample_mps, "ThunderMittens typical-p sampling (MPS)");
  m.def("apply_token_bitmask", &apply_token_bitmask_mps, "ThunderMittens grammar bitmask masking (MPS)");
  m.def("apply_bad_words", &apply_bad_words_mps, "ThunderMittens bad/stop-word masking (MPS)");
  m.def("beam_advance", &beam_advance_mps, "ThunderMittens beam-search advance (MPS)");
  m.def("spec_verify_tree", &spec_verify_tree_mps, "ThunderMittens spec tree verification (MPS)");
  m.def("spec_compact", &spec_compact_mps, "ThunderMittens spec accepted-token compaction (MPS)");
  m.def("rejection_greedy_sample", &rejection_greedy_sample_mps, "vLLM greedy rejection (MPS)",
        pybind11::arg("cu_num_draft_tokens"), pybind11::arg("draft_token_ids"),
        pybind11::arg("target_argmax"), pybind11::arg("bonus_token_ids"),
        pybind11::arg("max_draft"), pybind11::arg("is_greedy") = c10::nullopt);
  m.def("rejection_random_sample", &rejection_random_sample_mps, "vLLM random rejection (MPS)",
        pybind11::arg("cu_num_draft_tokens"), pybind11::arg("draft_token_ids"),
        pybind11::arg("target_probs"), pybind11::arg("bonus_token_ids"),
        pybind11::arg("recovered_token_ids"), pybind11::arg("uniform_probs"),
        pybind11::arg("max_draft"), pybind11::arg("draft_probs") = c10::nullopt,
        pybind11::arg("is_greedy") = c10::nullopt);
  m.def("sample_recovered_tokens", &sample_recovered_tokens_mps, "recovered token sampler (MPS)",
        pybind11::arg("cu_num_draft_tokens"), pybind11::arg("draft_token_ids"),
        pybind11::arg("target_probs"), pybind11::arg("inv_q"),
        pybind11::arg("draft_probs") = c10::nullopt);
  m.def("eagle_prepare_inputs_padded", &eagle_prepare_inputs_padded_mps, "EAGLE prep inputs (MPS)");
  m.def("eagle_prepare_next_token_padded", &eagle_prepare_next_token_padded_mps,
        "EAGLE next seed token (MPS)");
  m.def("eagle_step_slot_mapping_metadata", &eagle_step_slot_mapping_metadata_mps,
        "EAGLE step slot mapping (MPS)");
  m.def("eagle_expand_int32", &eagle_expand_int32_mps, "EAGLE ragged broadcast (MPS)");
  m.def("build_dynamic_tree", &build_dynamic_tree_mps, "ThunderMittens device draft-tree pointers (MPS)");
  m.def("spec_update_kv_meta", &spec_update_kv_meta_mps, "ThunderMittens spec KV meta update (MPS)");
  m.def("spec_verify_linear", &spec_verify_linear_mps,
        "ThunderMittens speculative linear rejection-sampling verification (MPS)");
  m.def("apply_penalty", &apply_penalty_mps, "ThunderMittens logit penalties (MPS)");
  m.def("quantize_per_tensor_fp8", &quantize_per_tensor_fp8_mps, "ThunderMittens per-tensor fp8 quant (MPS)");
  m.def("quantize_per_tensor_int8", &quantize_per_tensor_int8_mps, "ThunderMittens per-tensor int8 quant (MPS)");
  m.def("quantize_per_token_fp8", &quantize_per_token_fp8_mps, "ThunderMittens per-row fp8 quant (MPS)");
  m.def("quantize_per_token_int8", &quantize_per_token_int8_mps, "ThunderMittens per-row int8 quant (MPS)");
  m.def("attn_causal", &attn_causal_mps, "ThunderMittens causal attention (MPS)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("softcap") = 0.0, pybind11::arg("sinks") = pybind11::none());
  m.def("attn_window", &attn_window_mps, "ThunderMittens sliding-window causal attention (MPS)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"), pybind11::arg("window"),
        pybind11::arg("softcap") = 0.0, pybind11::arg("sinks") = pybind11::none());
  m.def("attn_varlen_prefill", &attn_varlen_prefill_mps,
        "ThunderMittens varlen/paged-prefill causal attention (MPS)",
        pybind11::arg("q_hm"), pybind11::arg("key_cache"), pybind11::arg("value_cache"),
        pybind11::arg("block_table"), pybind11::arg("context_lens"), pybind11::arg("tile_seq"),
        pybind11::arg("tile_local0"), pybind11::arg("seq_qlen"), pybind11::arg("scale"),
        pybind11::arg("softcap") = 0.0, pybind11::arg("sinks") = pybind11::none());
  m.def("varlen_pad_q", &varlen_pad_q_mps, "ThunderMittens device varlen Q pad/gather (MPS)");
  m.def("varlen_regather_o", &varlen_regather_o_mps, "ThunderMittens device varlen output re-gather (MPS)");
  m.def("varlen_build_worklist", &varlen_build_worklist_mps,
        "ThunderMittens on-device varlen prefill worklist builder (MPS)");
  m.def("lm_head_sample", &lm_head_sample_mps, "ThunderMittens fused LM-head + sampling (MPS)");
  m.def("lm_head_sample_q", &lm_head_sample_q_mps,
        "ThunderMittens fused LM-head + sampling over quantized weights (MPS)");
  m.def("cross_entropy_fwd", &cross_entropy_fwd_mps, "ThunderMittens fused cross-entropy fwd (MPS)");
  m.def("cross_entropy_bwd", &cross_entropy_bwd_mps, "ThunderMittens fused cross-entropy bwd (MPS)");
  m.def("kd_kl_topk_fwd", &kd_kl_topk_fwd_mps, "BitNet sparse-teacher KD-KL fwd (MPS)",
        pybind11::arg("logits"), pybind11::arg("t_idx"), pybind11::arg("t_prob"),
        pybind11::arg("invtemp") = 1.0, pybind11::arg("tail_mode") = 0);
  m.def("kd_kl_topk_bwd", &kd_kl_topk_bwd_mps, "BitNet sparse-teacher KD-KL bwd (MPS)",
        pybind11::arg("logits"), pybind11::arg("t_idx"), pybind11::arg("t_prob"),
        pybind11::arg("lse"), pybind11::arg("grad_out"), pybind11::arg("invtemp") = 1.0,
        pybind11::arg("tail_mode") = 0);
  m.def("kd_kl_dense_fwd", &kd_kl_dense_fwd_mps,
        "Dense-teacher KD-KL forward -> (loss, lse_t, lse_s) (MPS)",
        pybind11::arg("t_logits"), pybind11::arg("s_logits"), pybind11::arg("invtemp") = 1.0);
  m.def("kd_kl_dense_bwd", &kd_kl_dense_bwd_mps,
        "Dense-teacher KD-KL backward -> grad_s (MPS)",
        pybind11::arg("t_logits"), pybind11::arg("s_logits"), pybind11::arg("lse_t"),
        pybind11::arg("lse_s"), pybind11::arg("grad_out"), pybind11::arg("invtemp") = 1.0);
  m.def("kd_ce_fused_fwd", &kd_ce_fused_fwd_mps,
        "Fused CE + dense-KD forward -> [ce, kd, lse_sr, lse_st, lse_t] (MPS)",
        pybind11::arg("t_logits"), pybind11::arg("s_logits"), pybind11::arg("targets"),
        pybind11::arg("invtemp") = 1.0);
  m.def("kd_ce_fused_bwd", &kd_ce_fused_bwd_mps,
        "Fused CE + dense-KD backward -> combined grad_s (MPS)",
        pybind11::arg("t_logits"), pybind11::arg("s_logits"), pybind11::arg("targets"),
        pybind11::arg("lse_sr"), pybind11::arg("lse_st"), pybind11::arg("lse_t"),
        pybind11::arg("go_ce"), pybind11::arg("go_kd"), pybind11::arg("invtemp") = 1.0);
  m.def("flux_gelu", &flux_gelu_mps, "ThunderMittens fused GEMM+GELU (MPS)");
  m.def("flux_gate", &flux_gate_mps, "ThunderMittens fused GEMM+gate+residual (MPS)");
  m.def("gemm_staged", &gemm_staged_mps, "ThunderMittens staged multi-simdgroup GEMM (MPS)");
  m.def("attn_multiwarp", &attn_multiwarp_mps, "ThunderMittens multi-warp attention (MPS)");
  m.def("linear_attn", &linear_attn_mps, "ThunderMittens non-causal linear attention (MPS)");
  m.def("hedgehog", &hedgehog_mps, "ThunderMittens hedgehog linear attention (MPS)");
  m.def("lin_attn_causal", &lin_attn_causal_mps, "ThunderMittens causal linear attention (MPS)");
  m.def("mamba2", &mamba2_mps, "ThunderMittens Mamba-2 / SSD forward (MPS)");
  m.def("mamba2_bwd", &mamba2_bwd_mps, "ThunderMittens Mamba-2 / SSD backward (MPS)",
        pybind11::arg("C"), pybind11::arg("B"), pybind11::arg("X"), pybind11::arg("cumlog"),
        pybind11::arg("dY"), pybind11::arg("force_quadratic") = false);
  m.def("mamba2_chunked", &mamba2_chunked_mps,
        "ThunderMittens Mamba-2 / SSD forward, forced chunked route (MPS)");
  m.def("mamba2_bwd_chunked", &mamba2_bwd_chunked_mps,
        "ThunderMittens Mamba-2 / SSD backward, forced chunked route (MPS)");
  m.def("ssd_decode", &ssd_decode_mps,
        "ThunderMittens SSD single-token decode step (MPS, in-place state)");
  m.def("lin_attn_decay", &lin_attn_decay_mps, "ThunderMittens decay/retention linear attention (MPS)");
  m.def("based", &based_mps, "ThunderMittens Based Taylor-map linear attention (MPS)");
  m.def("attn_fwd_l", &attn_fwd_l_mps, "ThunderMittens flash-attn forward + L (MPS)");
  m.def("attn_bwd_prep", &attn_bwd_prep_mps, "ThunderMittens flash-attn backward prep delta (MPS)");
  m.def("attn_bwd_dq", &attn_bwd_dq_mps, "ThunderMittens flash-attn backward dQ (MPS)");
  m.def("attn_bwd_dkv", &attn_bwd_dkv_mps, "ThunderMittens flash-attn backward dK,dV (MPS)");
  m.def("cmplx_matmul", &cmplx_matmul_mps, "ThunderMittens complex GEMM (MPS)");
  m.def("fftconv", &fftconv_mps, "ThunderMittens Monarch FFT convolution (MPS)");
  m.def("qgemm", &qgemm_mps, "ThunderMittens quantized GEMM (MPS)");
  m.def("qgemm_blockscale", &qgemm_blockscale_mps, "ThunderMittens fp8_block2d GEMM (MPS)");
  m.def("qgemm_fp8_scaled", &qgemm_fp8_scaled_mps, "ThunderMittens fp8 rank-1 scaled GEMM (MPS)");
  m.def("qgemm_actorder_k", &qgemm_actorder_k_mps, "ThunderMittens GPTQ act-order qgemm, in-kernel gather (MPS)");
  m.def("qgemv", &qgemv_mps, "ThunderMittens quantized GEMV decode (MPS)");
  m.def("qflux_gelu", &qflux_gelu_mps, "ThunderMittens quantized fused GEMM+GELU (MPS)");
  m.def("attn_q", &attn_q_mps, "ThunderMittens quantized-KV flash attention (MPS)");
  m.def("qgemv_w8a8", &qgemv_w8a8_mps, "ThunderMittens W8A8 int8xint8 decode GEMV (MPS)");
  m.def("qgemv_w2a8", &qgemv_w2a8_mps, "ThunderMittens BitNet W2A8 int decode GEMV (MPS)",
        pybind11::arg("wq"), pybind11::arg("xq"), pybind11::arg("a_scale"),
        pybind11::arg("version") = 1);
  m.def("qgemv_w2a8_v2", &qgemv_w2a8_v2_mps, "ThunderMittens BitNet W2A8 int decode GEMV v2 (MPS)");
  m.def("qgemm_w8a8", &qgemm_w8a8_mps, "ThunderMittens W8A8 int8 prefill GEMM (MPS)");
  m.def("qgemm_w2a8", &qgemm_w2a8_mps, "ThunderMittens BitNet W2A8 prefill GEMM (MPS)");
  m.def("decode_layernorm_add", &decode_layernorm_add_mps,
        "decode-compatible residual-add + LayerNorm (MPS)");
  m.def("attn_decode_bh", &attn_decode_bh_mps,
        "batched head-major GQA attention decode (MPS)");
  m.def("decode_cache_attention", &decode_cache_attention_mps,
        "fused norm + RoPE + cache append + decode attention (MPS)",
        pybind11::arg("q"), pybind11::arg("new_k"), pybind11::arg("new_v"),
        pybind11::arg("cos"), pybind11::arg("sin"), pybind11::arg("positions"),
        pybind11::arg("context_lengths"), pybind11::arg("q_weight"),
        pybind11::arg("k_weight"), pybind11::arg("key_cache"),
        pybind11::arg("value_cache"), pybind11::arg("eps") = 1e-6,
        pybind11::arg("do_q_norm") = false, pybind11::arg("do_k_norm") = false,
        pybind11::arg("gemma") = false, pybind11::arg("softmax_scale") = 0.0);
  m.def("swin_attn_d32", &swin_attn_d32_mps,
        "Swin packed-QKV window attention, D=32 (MPS)");
  m.def("patch_merge_layernorm", &patch_merge_layernorm_mps,
        "fused Swin patch merge + LayerNorm (MPS)");
  m.def("space_to_depth_norm_linear", &space_to_depth_norm_linear_mps,
        "fused space-to-depth + norm + projection (MPS)",
        pybind11::arg("input"), pybind11::arg("norm_weight"),
        pybind11::arg("norm_bias"), pybind11::arg("projection_weight"),
        pybind11::arg("projection_bias"), pybind11::arg("height"),
        pybind11::arg("width"), pybind11::arg("block_size") = 2,
        pybind11::arg("eps") = 1e-5, pybind11::arg("use_norm_bias") = true,
        pybind11::arg("use_projection_bias") = false);
  m.def("edge_mlp_256x7", &edge_mlp_256x7_mps,
        "fixed pairwise 512-to-256-to-7 edge MLP (MPS)");
  m.def("decode_linear", &decode_linear_mps,
        "latency-oriented decode linear with optional erf GELU (MPS)");
  m.def("decode_linear_residual", &decode_linear_residual_mps,
        "decode linear + materialized-order residual (MPS)");
  m.def("decode_linear_q8", &decode_linear_q8_mps,
        "q8_0 decode linear with optional GELU/residual (MPS)");
  m.def("decode_linear_epilogue", &decode_linear_epilogue_mps,
        "dense/packed decode linear epilogue (MPS)",
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
        pybind11::arg("residual"), pybind11::arg("format") = "",
        pybind11::arg("activation") = 0, pybind11::arg("use_bias") = false,
        pybind11::arg("use_residual") = false);
  m.def("decode_swiglu", &decode_swiglu_mps,
        "dense/packed decode pair with SwiGLU (MPS)",
        pybind11::arg("x"), pybind11::arg("gate_weight"),
        pybind11::arg("up_weight"), pybind11::arg("gate_bias"),
        pybind11::arg("up_bias"), pybind11::arg("format") = "",
        pybind11::arg("use_bias") = false);
  m.def("dequant_gather", &dequant_gather_mps,
        "packed GGUF gather + direct fp16 dequantization (MPS)");
  m.def("quantized_embedding", &quantized_embedding_mps,
        "packed embedding lookup with optional additive epilogue (MPS)",
        pybind11::arg("table"), pybind11::arg("ids"), pybind11::arg("add"),
        pybind11::arg("format"), pybind11::arg("scale") = 1.0,
        pybind11::arg("use_add") = false,
        pybind11::arg("output_dtype") = "float16");
  m.def("quantized_embedding_bag", &quantized_embedding_bag_mps,
        "CSR embedding bag over packed rows (MPS)",
        pybind11::arg("table"), pybind11::arg("ids"), pybind11::arg("offsets"),
        pybind11::arg("sample_weights"), pybind11::arg("format"),
        pybind11::arg("scale") = 1.0, pybind11::arg("use_weights") = false,
        pybind11::arg("mean_mode") = false,
        pybind11::arg("output_dtype") = "float16");
  m.def("lm_head_constrained", &lm_head_constrained_mps,
        "dense constrained LM head returning token and logprob (MPS)");
  m.def("lm_head_masked", &lm_head_masked_mps,
        "masked dense/packed LM head returning top-k and logprobs (MPS)",
        pybind11::arg("h"), pybind11::arg("weight"), pybind11::arg("bias"),
        pybind11::arg("allow_mask"), pybind11::arg("format") = "",
        pybind11::arg("topk") = 1, pybind11::arg("normalize_allowed") = true);
  m.def("lm_head_candidates", &lm_head_candidates_mps,
        "CSR candidate dense/packed LM head returning top-k and logprobs (MPS)",
        pybind11::arg("h"), pybind11::arg("weight"), pybind11::arg("bias"),
        pybind11::arg("candidate_ids"), pybind11::arg("offsets"),
        pybind11::arg("format") = "", pybind11::arg("topk") = 1);
  m.def("flux_gelu_erf", &flux_gelu_erf_mps,
        "fused GEMM + erf GELU (MPS)");
}
