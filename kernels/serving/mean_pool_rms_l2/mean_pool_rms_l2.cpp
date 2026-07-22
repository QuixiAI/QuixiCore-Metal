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
