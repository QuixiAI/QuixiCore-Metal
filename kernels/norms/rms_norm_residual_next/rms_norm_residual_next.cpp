// Copyright © 2024 Apple Inc.

#include <cassert>
#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "rms_norm_residual_next/rms_norm_residual_next.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

std::vector<array> rms_norm_residual_next(
    const array& x,
    const array& post_weight,
    const array& residual,
    const array& next_weight,
    float eps /* = 1e-5f */,
    StreamOrDevice s /* = {} */) {
  const int D = x.shape(-1);
  if (x.dtype() != bfloat16 || residual.dtype() != bfloat16 ||
      residual.shape() != x.shape() || x.ndim() < 1) {
    throw std::invalid_argument(
        "rms_norm_residual_next: x/residual must be bfloat16 with the same shape");
  }
  auto check_weight = [&](const array& w, const char* which) {
    if (w.ndim() != 1 || w.shape(0) != D || w.dtype() != bfloat16) {
      throw std::invalid_argument(std::string("rms_norm_residual_next: ") + which +
                                  " must be bfloat16 with shape (D,)");
    }
  };
  check_weight(post_weight, "post_weight");
  check_weight(next_weight, "next_weight");
  if (!(D == 256 || D == 512 || D == 768 || D == 1024)) {
    throw std::invalid_argument(
        "rms_norm_residual_next: last dim must be 256, 512, 768, or 1024");
  }
  return array::make_arrays(
      {x.shape(), x.shape()},
      {bfloat16, bfloat16},
      std::make_shared<RMSNormResidualNext>(to_stream(s), eps),
      {x, post_weight, residual, next_weight});
}

void RMSNormResidualNext::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("RMSNormResidualNext has no CPU implementation.");
}

bool RMSNormResidualNext::is_equivalent(const Primitive& other) const {
  return eps_ == static_cast<const RMSNormResidualNext&>(other).eps_;
}

void RMSNormResidualNext::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 4);
  auto& x = inputs[0];
  auto& post_weight = inputs[1];
  auto& residual = inputs[2];
  auto& next_weight = inputs[3];
  auto& res_out = outputs[0];
  auto& next_out = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  res_out.set_data(allocator::malloc_or_wait(res_out.nbytes()));
  next_out.set_data(allocator::malloc_or_wait(next_out.nbytes()));

  const int D = x.shape(-1);
  const uint32_t M = static_cast<uint32_t>(x.size() / D);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_rms_norm_residual_next(enc, x, post_weight, residual, next_weight,
                                    res_out, next_out, M, D, eps_);
}

}  // namespace mlx::core
