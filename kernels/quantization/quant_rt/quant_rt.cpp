// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "quant_rt/quant_rt.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static std::vector<int> qrt_scale_shape(const array& x) {
  std::vector<int> sh(x.shape().begin(), x.shape().end() - 1);
  if (sh.empty()) {
    sh.push_back(1);
  }
  return sh;
}

static void qrt_check(const array& x, const char* name) {
  if (x.ndim() < 1) {
    throw std::invalid_argument(std::string(name) + ": x must have at least 1 dimension");
  }
  if (!(x.dtype() == float32 || x.dtype() == float16 || x.dtype() == bfloat16)) {
    throw std::invalid_argument(std::string(name) + ": x must be float32, float16, or bfloat16");
  }
}

std::vector<array> quantize_per_token_fp8(const array& x, StreamOrDevice s /* = {} */) {
  qrt_check(x, "quantize_per_token_fp8");
  auto x_c = contiguous(x, false, s);
  return array::make_arrays(
      {x.shape(), qrt_scale_shape(x)},
      {uint8, float32},
      std::make_shared<QuantizePerTokenFp8>(to_stream(s)),
      {x_c});
}

std::vector<array> quantize_per_token_int8(const array& x, StreamOrDevice s /* = {} */) {
  qrt_check(x, "quantize_per_token_int8");
  auto x_c = contiguous(x, false, s);
  return array::make_arrays(
      {x.shape(), qrt_scale_shape(x)},
      {int8, float32},
      std::make_shared<QuantizePerTokenInt8>(to_stream(s)),
      {x_c});
}

std::vector<array> quantize_per_group_fp8(const array& x, int group_size, bool ue8m0,
                                          StreamOrDevice s) {
  qrt_check(x, "quantize_per_group_fp8");
  const int D = x.shape(-1);
  if (group_size <= 0 || group_size % 4 != 0 || D % group_size != 0) {
    throw std::invalid_argument("quantize_per_group_fp8: need group_size % 4 == 0, D % group_size == 0");
  }
  auto scale_shape = x.shape();
  scale_shape.back() = D / group_size;
  return array::make_arrays(
      {x.shape(), scale_shape},
      {uint8, float32},
      std::make_shared<QuantizePerGroupFp8>(to_stream(s), group_size, ue8m0),
      {contiguous(x, false, s)});
}

std::vector<array> quantize_per_group_int8(const array& x, int group_size, StreamOrDevice s) {
  qrt_check(x, "quantize_per_group_int8");
  const int D = x.shape(-1);
  if (group_size <= 0 || group_size % 4 != 0 || D % group_size != 0) {
    throw std::invalid_argument("quantize_per_group_int8: need group_size % 4 == 0, D % group_size == 0");
  }
  auto scale_shape = x.shape();
  scale_shape.back() = D / group_size;
  return array::make_arrays(
      {x.shape(), scale_shape},
      {int8, float32},
      std::make_shared<QuantizePerGroupInt8>(to_stream(s), group_size),
      {contiguous(x, false, s)});
}

std::vector<array> quantize_per_token_int8_azp(const array& x, StreamOrDevice s) {
  qrt_check(x, "quantize_per_token_int8_azp");
  return array::make_arrays(
      {x.shape(), qrt_scale_shape(x), qrt_scale_shape(x)},
      {int8, float32, int32},
      std::make_shared<QuantizePerTokenInt8Azp>(to_stream(s)),
      {contiguous(x, false, s)});
}

void QuantizePerGroupFp8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QuantizePerGroupFp8 has no CPU implementation.");
}
void QuantizePerGroupFp8::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& codes = outputs[0];
  auto& scale = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  codes.set_data(allocator::malloc_or_wait(codes.nbytes()));
  scale.set_data(allocator::malloc_or_wait(scale.nbytes()));
  const int D = x.shape(-1);
  const int rows = static_cast<int>(x.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_quantize_per_group_fp8(enc, x, codes, scale, rows, D, g_, ue8m0_ ? 1 : 0,
                                    type_to_name(x));
}

void QuantizePerGroupInt8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QuantizePerGroupInt8 has no CPU implementation.");
}
void QuantizePerGroupInt8::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& codes = outputs[0];
  auto& scale = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  codes.set_data(allocator::malloc_or_wait(codes.nbytes()));
  scale.set_data(allocator::malloc_or_wait(scale.nbytes()));
  const int D = x.shape(-1);
  const int rows = static_cast<int>(x.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_quantize_per_group_int8(enc, x, codes, scale, rows, D, g_, type_to_name(x));
}

void QuantizePerTokenInt8Azp::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QuantizePerTokenInt8Azp has no CPU implementation.");
}
void QuantizePerTokenInt8Azp::eval_gpu(const std::vector<array>& inputs,
                                       std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& codes = outputs[0];
  auto& scale = outputs[1];
  auto& azp = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  codes.set_data(allocator::malloc_or_wait(codes.nbytes()));
  scale.set_data(allocator::malloc_or_wait(scale.nbytes()));
  azp.set_data(allocator::malloc_or_wait(azp.nbytes()));
  const int D = x.shape(-1);
  const int rows = static_cast<int>(x.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_quantize_per_token_int8_azp(enc, x, codes, scale, azp, rows, D, type_to_name(x));
}

void QuantizePerTokenFp8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QuantizePerTokenFp8 has no CPU implementation.");
}
void QuantizePerTokenInt8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QuantizePerTokenInt8 has no CPU implementation.");
}

void QuantizePerTokenFp8::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& codes = outputs[0];
  auto& scale = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  codes.set_data(allocator::malloc_or_wait(codes.nbytes()));
  scale.set_data(allocator::malloc_or_wait(scale.nbytes()));
  const int D = x.shape(-1);
  const int rows = static_cast<int>(x.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_quantize_per_token_fp8(enc, x, codes, scale, rows, D, type_to_name(x));
}

void QuantizePerTokenInt8::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& codes = outputs[0];
  auto& scale = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  codes.set_data(allocator::malloc_or_wait(codes.nbytes()));
  scale.set_data(allocator::malloc_or_wait(scale.nbytes()));
  const int D = x.shape(-1);
  const int rows = static_cast<int>(x.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_quantize_per_token_int8(enc, x, codes, scale, rows, D, type_to_name(x));
}

// ------------------------- per-tensor dynamic quant -------------------------

static std::vector<array> quantize_per_tensor_impl(const array& x, bool is_int8, StreamOrDevice s) {
  qrt_check(x, is_int8 ? "quantize_per_tensor_int8" : "quantize_per_tensor_fp8");
  auto x_c = contiguous(x, false, s);
  return array::make_arrays(
      {x.shape(), {1}, {1}},
      {is_int8 ? int8 : uint8, float32, uint32},
      std::make_shared<QuantizePerTensor>(to_stream(s), is_int8),
      {x_c});
}
std::vector<array> quantize_per_tensor_fp8(const array& x, StreamOrDevice s) {
  return quantize_per_tensor_impl(x, false, s);
}
std::vector<array> quantize_per_tensor_int8(const array& x, StreamOrDevice s) {
  return quantize_per_tensor_impl(x, true, s);
}

void QuantizePerTensor::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QuantizePerTensor has no CPU implementation.");
}

void QuantizePerTensor::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& codes = outputs[0];
  auto& scale = outputs[1];
  auto& scale_u = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  codes.set_data(allocator::malloc_or_wait(codes.nbytes()));
  scale.set_data(allocator::malloc_or_wait(scale.nbytes()));
  scale_u.set_data(allocator::malloc_or_wait(scale_u.nbytes()));
  const int n = static_cast<int>(x.size());
  const std::string tn = type_to_name(x);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_moe_zero_i32(enc, scale_u, 1);   // zero the atomic accumulator (uint 0 < any orderable)
  tk::launch_quant_tensor_absmax(enc, x, scale_u, n, tn);
  tk::launch_quant_tensor_encode(enc, x, scale_u, codes, scale, n, is_int8_, tn);
}

#define TK_QRT_NO_AUTODIFF(CLASS, LABEL)                                     \
  std::vector<array> CLASS::jvp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&) {                                             \
    throw std::runtime_error(LABEL " has no jvp implementation.");           \
  }                                                                          \
  std::vector<array> CLASS::vjp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&, const std::vector<array>&) {                  \
    throw std::runtime_error(LABEL " has no vjp implementation.");           \
  }                                                                          \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(               \
      const std::vector<array>&, const std::vector<int>&) {                  \
    throw std::runtime_error(LABEL " has no vmap implementation.");          \
  }

TK_QRT_NO_AUTODIFF(QuantizePerGroupFp8, "QuantizePerGroupFp8")
TK_QRT_NO_AUTODIFF(QuantizePerGroupInt8, "QuantizePerGroupInt8")
TK_QRT_NO_AUTODIFF(QuantizePerTokenInt8Azp, "QuantizePerTokenInt8Azp")
TK_QRT_NO_AUTODIFF(QuantizePerTokenFp8, "QuantizePerTokenFp8")
TK_QRT_NO_AUTODIFF(QuantizePerTokenInt8, "QuantizePerTokenInt8")
TK_QRT_NO_AUTODIFF(QuantizePerTensor, "QuantizePerTensor")

} // namespace mlx::core
