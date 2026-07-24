// Copyright © 2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "mean_pool_rms_l2/mean_pool_rms_l2.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation Implementation
///////////////////////////////////////////////////////////////////////////////

array mean_pool_rms_l2(
    const array& x,
    const array& weight,
    float eps /* = 1e-5f */,
    StreamOrDevice s /* = {} */
) {
  assert(x.dtype() == bfloat16 && weight.dtype() == bfloat16);
  assert(x.ndim() == 2);
  const int D = x.shape(-1);
  assert(weight.ndim() == 1 && weight.shape(0) == D);
  if (!(D == 256 || D == 512 || D == 768 || D == 1024)) {
    throw std::invalid_argument(
        "mean_pool_rms_l2: last dim must be 256, 512, 768, or 1024");
  }

  return array(
      /* const std::vector<int>& shape = */ {D},
      /* Dtype dtype = */ bfloat16,
      /* std::shared_ptr<Primitive> primitive = */
      std::make_shared<MeanPoolRmsL2>(to_stream(s), eps),
      /* const std::vector<array>& inputs = */ {x, weight});
}

array masked_mean_pool_rms_l2(
    const array& x, const array& mask, const array& weight,
    float eps, StreamOrDevice s) {
  if (x.dtype() != bfloat16 || x.ndim() != 3) {
    throw std::invalid_argument("masked_mean_pool_rms_l2: x must be BF16 (B,T,D)");
  }
  const int B = x.shape(0), T = x.shape(1), D = x.shape(2);
  if (!(D == 256 || D == 512 || D == 768 || D == 1024)) {
    throw std::invalid_argument("masked_mean_pool_rms_l2: D must be 256/512/768/1024");
  }
  if (mask.ndim() != 2 || mask.shape(0) != B || mask.shape(1) != T ||
      weight.ndim() != 1 || weight.shape(0) != D || weight.dtype() != bfloat16) {
    throw std::invalid_argument("masked_mean_pool_rms_l2: need mask(B,T) and BF16 weight(D)");
  }
  return array(
      {B, D}, bfloat16,
      std::make_shared<MaskedMeanPoolRmsL2>(to_stream(s), eps),
      {contiguous(x, false, s), contiguous(astype(mask, int32, s), false, s),
       contiguous(weight, false, s)});
}

void MaskedMeanPoolRmsL2::eval_cpu(
    const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MaskedMeanPoolRmsL2 has no CPU implementation.");
}
void MaskedMeanPoolRmsL2::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0]; auto& mask = inputs[1]; auto& weight = inputs[2]; auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index); MLXEncoder enc(d, ce);
  tk::launch_masked_mean_pool_rms_l2(
      enc, x, mask, weight, out, x.shape(0), x.shape(1), x.shape(2), eps_);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

void MeanPoolRmsL2::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // No CPU fallback yet; compose mean/rms_norm/l2 in the framework for a reference.
  assert(false);
}

void MeanPoolRmsL2::eval_cpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

bool MeanPoolRmsL2::is_equivalent(const Primitive& other) const {
  const MeanPoolRmsL2& o = static_cast<const MeanPoolRmsL2&>(other);
  return eps_ == o.eps_;
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

void MeanPoolRmsL2::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  auto& x = inputs[0];
  auto& weight = inputs[1];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int D = x.shape(-1);
  const uint32_t M = static_cast<uint32_t>(x.size() / D);  // pooled rows

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mean_pool_rms_l2(enc, x, weight, out, M, D, eps_);
}

}  // namespace mlx::core
