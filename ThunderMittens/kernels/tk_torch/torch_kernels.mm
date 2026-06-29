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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_set_library", &tk_set_library, "set the metallib path");
  m.def("layernorm", &layernorm_mps, "ThunderMittens LayerNorm (MPS)");
  m.def("add_rt", &add_rt_mps, "ThunderMittens add_rt elementwise add (MPS)");
  m.def("matmul_custom", &matmul_custom_mps, "ThunderMittens matmul_custom GEMM (MPS)");
  m.def("attn_fwd", &attn_fwd_mps, "ThunderMittens attention forward (MPS)");
}
