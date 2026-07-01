// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "add_norm/add_norm.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static void check_norm_shapes(const array& x, const array& residual, int D, const char* name) {
  if (x.shape() != residual.shape()) {
    throw std::invalid_argument(std::string(name) + ": x and residual must have the same shape");
  }
  if (!(D == 256 || D == 512 || D == 768 || D == 1024)) {
    throw std::invalid_argument(std::string(name) + ": last dim must be 256, 512, 768, or 1024");
  }
  if (x.dtype() != bfloat16 || residual.dtype() != bfloat16) {
    throw std::invalid_argument(std::string(name) + ": x and residual must be bfloat16");
  }
}

///////////////////////////////////////////////////////////////////////////////
// Operation Implementations
///////////////////////////////////////////////////////////////////////////////

std::vector<array> rms_norm_add(
    const array& x,
    const array& residual,
    const array& weight,
    float eps /* = 1e-5f */,
    StreamOrDevice s /* = {} */) {
  const int D = x.shape(-1);
  check_norm_shapes(x, residual, D, "rms_norm_add");
  if (weight.ndim() != 1 || weight.shape(0) != D || weight.dtype() != bfloat16) {
    throw std::invalid_argument("rms_norm_add: weight must be bfloat16 with shape (D,)");
  }
  return array::make_arrays(
      {x.shape(), x.shape()},
      {bfloat16, bfloat16},
      std::make_shared<RMSNormAdd>(to_stream(s), eps),
      {x, residual, weight});
}

std::vector<array> layernorm_add(
    const array& x,
    const array& residual,
    const array& weight,
    const array& bias,
    float eps /* = 1e-5f */,
    StreamOrDevice s /* = {} */) {
  const int D = x.shape(-1);
  check_norm_shapes(x, residual, D, "layernorm_add");
  if (weight.ndim() != 1 || weight.shape(0) != D || weight.dtype() != bfloat16) {
    throw std::invalid_argument("layernorm_add: weight must be bfloat16 with shape (D,)");
  }
  if (bias.ndim() != 1 || bias.shape(0) != D || bias.dtype() != bfloat16) {
    throw std::invalid_argument("layernorm_add: bias must be bfloat16 with shape (D,)");
  }
  return array::make_arrays(
      {x.shape(), x.shape()},
      {bfloat16, bfloat16},
      std::make_shared<LayerNormAdd>(to_stream(s), eps),
      {x, residual, weight, bias});
}

///////////////////////////////////////////////////////////////////////////////
// CPU (no fallback yet — use mx.fast.{rms_norm,layer_norm} on (x+residual))
///////////////////////////////////////////////////////////////////////////////

void RMSNormAdd::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("RMSNormAdd has no CPU implementation.");
}

void LayerNormAdd::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LayerNormAdd has no CPU implementation.");
}

///////////////////////////////////////////////////////////////////////////////
// Metal Backend
///////////////////////////////////////////////////////////////////////////////

void RMSNormAdd::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& x = inputs[0];
  auto& residual = inputs[1];
  auto& weight = inputs[2];
  auto& out = outputs[0];
  auto& res_out = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  res_out.set_data(allocator::malloc_or_wait(res_out.nbytes()));

  const int D = x.shape(-1);
  const uint32_t M = static_cast<uint32_t>(x.size() / D);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_rms_norm_add(enc, x, residual, weight, out, res_out, M, D, eps_);
}

void LayerNormAdd::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 4);
  auto& x = inputs[0];
  auto& residual = inputs[1];
  auto& weight = inputs[2];
  auto& bias = inputs[3];
  auto& out = outputs[0];
  auto& res_out = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  res_out.set_data(allocator::malloc_or_wait(res_out.nbytes()));

  const int D = x.shape(-1);
  const uint32_t M = static_cast<uint32_t>(x.size() / D);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_layernorm_add(enc, x, residual, weight, bias, out, res_out, M, D, eps_);
}

///////////////////////////////////////////////////////////////////////////////
// Transforms (no autodiff)
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// fp8 epilogue ops
///////////////////////////////////////////////////////////////////////////////

static std::vector<int> anfp8_scale_shape(const array& x) {
  std::vector<int> sh(x.shape().begin(), x.shape().end() - 1);
  if (sh.empty()) sh.push_back(1);
  return sh;
}

std::vector<array> rms_norm_add_fp8(
    const array& x, const array& residual, const array& weight, float eps, float scale,
    StreamOrDevice s) {
  const int D = x.shape(-1);
  check_norm_shapes(x, residual, D, "rms_norm_add_fp8");
  if (weight.ndim() != 1 || weight.shape(0) != D || weight.dtype() != bfloat16) {
    throw std::invalid_argument("rms_norm_add_fp8: weight must be bfloat16 (D,)");
  }
  const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
  return array::make_arrays(
      {x.shape(), x.shape()}, {uint8, bfloat16},
      std::make_shared<AddNormFp8>(to_stream(s), false, false, eps, inv), {x, residual, weight});
}

std::vector<array> rms_norm_add_fp8_dyn(
    const array& x, const array& residual, const array& weight, float eps, StreamOrDevice s) {
  const int D = x.shape(-1);
  check_norm_shapes(x, residual, D, "rms_norm_add_fp8_dyn");
  if (weight.ndim() != 1 || weight.shape(0) != D || weight.dtype() != bfloat16) {
    throw std::invalid_argument("rms_norm_add_fp8_dyn: weight must be bfloat16 (D,)");
  }
  return array::make_arrays(
      {x.shape(), x.shape(), anfp8_scale_shape(x)}, {uint8, bfloat16, float32},
      std::make_shared<AddNormFp8>(to_stream(s), false, true, eps, 0.0f), {x, residual, weight});
}

std::vector<array> layernorm_add_fp8(
    const array& x, const array& residual, const array& weight, const array& bias, float eps,
    float scale, StreamOrDevice s) {
  const int D = x.shape(-1);
  check_norm_shapes(x, residual, D, "layernorm_add_fp8");
  if (weight.shape(0) != D || bias.shape(0) != D) {
    throw std::invalid_argument("layernorm_add_fp8: weight/bias must be (D,)");
  }
  const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
  return array::make_arrays(
      {x.shape(), x.shape()}, {uint8, bfloat16},
      std::make_shared<AddNormFp8>(to_stream(s), true, false, eps, inv),
      {x, residual, weight, bias});
}

std::vector<array> layernorm_add_fp8_dyn(
    const array& x, const array& residual, const array& weight, const array& bias, float eps,
    StreamOrDevice s) {
  const int D = x.shape(-1);
  check_norm_shapes(x, residual, D, "layernorm_add_fp8_dyn");
  if (weight.shape(0) != D || bias.shape(0) != D) {
    throw std::invalid_argument("layernorm_add_fp8_dyn: weight/bias must be (D,)");
  }
  return array::make_arrays(
      {x.shape(), x.shape(), anfp8_scale_shape(x)}, {uint8, bfloat16, float32},
      std::make_shared<AddNormFp8>(to_stream(s), true, true, eps, 0.0f),
      {x, residual, weight, bias});
}

void AddNormFp8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("AddNormFp8 has no CPU implementation.");
}

void AddNormFp8::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& residual = inputs[1];
  auto& weight = inputs[2];
  auto& codes = outputs[0];
  auto& res_out = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  codes.set_data(allocator::malloc_or_wait(codes.nbytes()));
  res_out.set_data(allocator::malloc_or_wait(res_out.nbytes()));

  const int D = x.shape(-1);
  const uint32_t M = static_cast<uint32_t>(x.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);

  if (layernorm_) {
    auto& bias = inputs[3];
    if (dynamic_) {
      auto& scale = outputs[2];
      scale.set_data(allocator::malloc_or_wait(scale.nbytes()));
      tk::launch_layernorm_add_fp8_dyn(enc, x, residual, weight, bias, codes, res_out, scale, M, D, eps_);
    } else {
      tk::launch_layernorm_add_fp8(enc, x, residual, weight, bias, codes, res_out, M, D, eps_, inv_scale_);
    }
  } else {
    if (dynamic_) {
      auto& scale = outputs[2];
      scale.set_data(allocator::malloc_or_wait(scale.nbytes()));
      tk::launch_rms_norm_add_fp8_dyn(enc, x, residual, weight, codes, res_out, scale, M, D, eps_);
    } else {
      tk::launch_rms_norm_add_fp8(enc, x, residual, weight, codes, res_out, M, D, eps_, inv_scale_);
    }
  }
}

std::vector<array> AddNormFp8::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AddNormFp8 has no jvp implementation.");
}
std::vector<array> AddNormFp8::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("AddNormFp8 has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> AddNormFp8::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AddNormFp8 has no vmap implementation.");
}

#define TK_NORM_NO_AUTODIFF(CLASS, LABEL)                                    \
  std::vector<array> CLASS::jvp(                                             \
      const std::vector<array>&,                                             \
      const std::vector<array>&,                                             \
      const std::vector<int>&) {                                             \
    throw std::runtime_error(LABEL " has no jvp implementation.");           \
  }                                                                          \
  std::vector<array> CLASS::vjp(                                             \
      const std::vector<array>&,                                             \
      const std::vector<array>&,                                             \
      const std::vector<int>&,                                               \
      const std::vector<array>&) {                                           \
    throw std::runtime_error(LABEL " has no vjp implementation.");           \
  }                                                                          \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(               \
      const std::vector<array>&,                                             \
      const std::vector<int>&) {                                             \
    throw std::runtime_error(LABEL " has no vmap implementation.");          \
  }                                                                          \
  bool CLASS::is_equivalent(const Primitive& other) const {                  \
    return eps_ == static_cast<const CLASS&>(other).eps_;                    \
  }

TK_NORM_NO_AUTODIFF(RMSNormAdd, "RMSNormAdd")
TK_NORM_NO_AUTODIFF(LayerNormAdd, "LayerNormAdd")

} // namespace mlx::core
