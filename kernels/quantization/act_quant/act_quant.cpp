// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "act_quant/act_quant.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
void actq_check(const array& x, const array& gate, int mode, const char* name) {
  if (x.ndim() < 1 || x.shape() != gate.shape()) {
    throw std::invalid_argument(std::string(name) + ": x and gate must have the same shape");
  }
  if (!(x.dtype() == float32 || x.dtype() == float16 || x.dtype() == bfloat16)) {
    throw std::invalid_argument(std::string(name) + ": dtype must be f32/f16/bf16");
  }
  if (mode != 0 && mode != 1) {
    throw std::invalid_argument(std::string(name) + ": mode must be 0 (swiglu) or 1 (swiglu_oai)");
  }
}
} // namespace

std::vector<array> silu_mul_quant_fp8(
    const array& x, const array& gate, int mode, float alpha, float limit, StreamOrDevice s) {
  actq_check(x, gate, mode, "silu_mul_quant_fp8");
  auto sshape = x.shape();
  sshape.pop_back();
  return array::make_arrays(
      {x.shape(), sshape},
      {uint8, float32},
      std::make_shared<ActQuant>(to_stream(s), 0, mode, alpha, limit, 0, false),
      {contiguous(x, false, s), contiguous(astype(gate, x.dtype(), s), false, s)});
}

std::vector<array> silu_mul_quant_int8(
    const array& x, const array& gate, int mode, float alpha, float limit, StreamOrDevice s) {
  actq_check(x, gate, mode, "silu_mul_quant_int8");
  auto sshape = x.shape();
  sshape.pop_back();
  return array::make_arrays(
      {x.shape(), sshape},
      {int8, float32},
      std::make_shared<ActQuant>(to_stream(s), 1, mode, alpha, limit, 0, false),
      {contiguous(x, false, s), contiguous(astype(gate, x.dtype(), s), false, s)});
}

std::vector<array> silu_mul_quant_fp8_group(
    const array& x, const array& gate, int group_size, bool ue8m0, int mode, float alpha,
    float limit, StreamOrDevice s) {
  actq_check(x, gate, mode, "silu_mul_quant_fp8_group");
  const int D = x.shape(-1);
  if (group_size <= 0 || group_size % 4 != 0 || D % group_size != 0) {
    throw std::invalid_argument(
        "silu_mul_quant_fp8_group: need group_size % 4 == 0 and D % group_size == 0");
  }
  auto sshape = x.shape();
  sshape.back() = D / group_size;
  return array::make_arrays(
      {x.shape(), sshape},
      {uint8, float32},
      std::make_shared<ActQuant>(to_stream(s), 2, mode, alpha, limit, group_size, ue8m0),
      {contiguous(x, false, s), contiguous(astype(gate, x.dtype(), s), false, s)});
}

void ActQuant::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("ActQuant has no CPU implementation.");
}

void ActQuant::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& gate = inputs[1];
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
  if (kind_ == 0) {
    tk::launch_silu_mul_quant_fp8(enc, x, gate, codes, scale, rows, D, mode_, alpha_, limit_,
                                  type_to_name(x));
  } else if (kind_ == 1) {
    tk::launch_silu_mul_quant_int8(enc, x, gate, codes, scale, rows, D, mode_, alpha_, limit_,
                                   type_to_name(x));
  } else {
    tk::launch_silu_mul_quant_fp8_group(enc, x, gate, codes, scale, rows, D, group_size_,
                                        ue8m0_ ? 1 : 0, mode_, alpha_, limit_,
                                        type_to_name(x));
  }
}

std::vector<array> ActQuant::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("ActQuant has no jvp implementation.");
}
std::vector<array> ActQuant::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("ActQuant has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> ActQuant::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("ActQuant has no vmap implementation.");
}

} // namespace mlx::core
