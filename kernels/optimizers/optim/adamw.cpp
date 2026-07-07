// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <cmath>
#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "optim/adamw.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static bool adamw_is_float(Dtype dt) {
  return dt == float32 || dt == float16 || dt == bfloat16;
}

std::vector<array> adamw(
    const array& param, const array& grad, const array& m, const array& v,
    float lr, float beta1, float beta2, float eps, float weight_decay, int step,
    StreamOrDevice s /* = {} */) {
  if (!adamw_is_float(param.dtype())) {
    throw std::invalid_argument("adamw: param must be float (fp16/bf16/fp32)");
  }
  if (param.shape() != grad.shape() || param.shape() != m.shape() || param.shape() != v.shape()) {
    throw std::invalid_argument("adamw: param, grad, m, v must have the same shape");
  }
  if (m.dtype() != float32 || v.dtype() != float32) {
    throw std::invalid_argument("adamw: moment state m, v must be float32");
  }
  if (step < 1) {
    throw std::invalid_argument("adamw: step (t) must be >= 1");
  }
  auto p_c = contiguous(param, false, s);
  auto g_c = contiguous(astype(grad, param.dtype(), s), false, s);
  auto m_c = contiguous(m, false, s);
  auto v_c = contiguous(v, false, s);
  return array::make_arrays(
      {param.shape(), param.shape(), param.shape()}, {param.dtype(), float32, float32},
      std::make_shared<AdamW>(to_stream(s), lr, beta1, beta2, eps, weight_decay, step),
      {p_c, g_c, m_c, v_c});
}

std::vector<array> adamw_masked(
    const array& param, const array& grad, const array& m, const array& v,
    float lr, float beta1, float beta2, float eps, float weight_decay, int step,
    const array& mask, int seg_size, int mask_mode, StreamOrDevice s /* = {} */) {
  if (!adamw_is_float(param.dtype())) {
    throw std::invalid_argument("adamw_masked: param must be float (fp16/bf16/fp32)");
  }
  if (param.shape() != grad.shape() || param.shape() != m.shape() || param.shape() != v.shape()) {
    throw std::invalid_argument("adamw_masked: param, grad, m, v must have the same shape");
  }
  if (m.dtype() != float32 || v.dtype() != float32) {
    throw std::invalid_argument("adamw_masked: moment state m, v must be float32");
  }
  if (step < 1) {
    throw std::invalid_argument("adamw_masked: step (t) must be >= 1");
  }
  if (mask.dtype() != uint8) {
    throw std::invalid_argument("adamw_masked: mask must be uint8");
  }
  if (seg_size < 1) {
    throw std::invalid_argument("adamw_masked: seg_size must be >= 1");
  }
  if (mask_mode != 0 && mask_mode != 1) {
    throw std::invalid_argument("adamw_masked: mask_mode must be 0 or 1");
  }
  if (static_cast<int64_t>(mask.size()) * seg_size < static_cast<int64_t>(param.size())) {
    throw std::invalid_argument("adamw_masked: mask covers fewer than numel/seg_size segments");
  }
  auto p_c = contiguous(param, false, s);
  auto g_c = contiguous(astype(grad, param.dtype(), s), false, s);
  auto m_c = contiguous(m, false, s);
  auto v_c = contiguous(v, false, s);
  auto mask_c = contiguous(mask, false, s);
  return array::make_arrays(
      {param.shape(), param.shape(), param.shape()}, {param.dtype(), float32, float32},
      std::make_shared<AdamWMasked>(
          to_stream(s), lr, beta1, beta2, eps, weight_decay, step, seg_size, mask_mode),
      {p_c, g_c, m_c, v_c, mask_c});
}

void AdamW::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("AdamW has no CPU implementation.");
}
void AdamW::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& param = inputs[0];
  auto& grad = inputs[1];
  auto& m = inputs[2];
  auto& v = inputs[3];
  auto& p_out = outputs[0];
  auto& m_out = outputs[1];
  auto& v_out = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  p_out.set_data(allocator::malloc_or_wait(p_out.nbytes()));
  m_out.set_data(allocator::malloc_or_wait(m_out.nbytes()));
  v_out.set_data(allocator::malloc_or_wait(v_out.nbytes()));
  const float bc1 = 1.0f - std::pow(b1_, static_cast<float>(step_));
  const float bc2 = 1.0f - std::pow(b2_, static_cast<float>(step_));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_adamw(enc, param, grad, m, v, p_out, m_out, v_out, lr_, b1_, b2_, eps_, wd_, bc1, bc2,
                   static_cast<uint32_t>(param.size()), type_to_name(param));
}

void AdamWMasked::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("AdamWMasked has no CPU implementation.");
}
void AdamWMasked::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& param = inputs[0];
  auto& grad = inputs[1];
  auto& m = inputs[2];
  auto& v = inputs[3];
  auto& mask = inputs[4];
  auto& p_out = outputs[0];
  auto& m_out = outputs[1];
  auto& v_out = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  p_out.set_data(allocator::malloc_or_wait(p_out.nbytes()));
  m_out.set_data(allocator::malloc_or_wait(m_out.nbytes()));
  v_out.set_data(allocator::malloc_or_wait(v_out.nbytes()));
  const float bc1 = 1.0f - std::pow(b1_, static_cast<float>(step_));
  const float bc2 = 1.0f - std::pow(b2_, static_cast<float>(step_));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_adamw_masked(
      enc, param, grad, m, v, p_out, m_out, v_out, lr_, b1_, b2_, eps_, wd_, bc1, bc2,
      static_cast<uint32_t>(param.size()), mask, static_cast<uint32_t>(seg_size_),
      mask_mode_, type_to_name(param));
}
std::vector<array> AdamW::jvp(const std::vector<array>&, const std::vector<array>&,
                              const std::vector<int>&) {
  throw std::runtime_error("AdamW has no jvp implementation.");
}
std::vector<array> AdamW::vjp(const std::vector<array>&, const std::vector<array>&,
                              const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("AdamW has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> AdamW::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AdamW has no vmap implementation.");
}

std::vector<array> AdamWMasked::jvp(const std::vector<array>&, const std::vector<array>&,
                                    const std::vector<int>&) {
  throw std::runtime_error("AdamWMasked has no jvp implementation.");
}
std::vector<array> AdamWMasked::vjp(const std::vector<array>&, const std::vector<array>&,
                                    const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("AdamWMasked has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> AdamWMasked::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AdamWMasked has no vmap implementation.");
}

} // namespace mlx::core
