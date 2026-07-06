// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "softmax/softmax.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation Implementation
///////////////////////////////////////////////////////////////////////////////

array softmax_tk(const array& x, StreamOrDevice s /* = {} */) {
  assert(x.dtype() == bfloat16);
  const int D = x.shape(-1);
  assert((D == 256 || D == 512 || D == 768 || D == 1024) &&
         "softmax: last dim must be 256, 512, 768, or 1024");

  return array(
      /* const std::vector<int>& shape = */ x.shape(),
      /* Dtype dtype = */ bfloat16,
      /* std::shared_ptr<Primitive> primitive = */
      std::make_shared<TKSoftmax>(to_stream(s)),
      /* const std::vector<array>& inputs = */ {x});
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

void TKSoftmax::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // No CPU fallback yet; use mx.softmax for a reference.
  assert(false);
}

void TKSoftmax::eval_cpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

void TKSoftmax::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  auto& x = inputs[0];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int D = x.shape(-1);
  const uint32_t M = static_cast<uint32_t>(x.size() / D);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_softmax(enc, x, out, M, D);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

std::vector<array> TKSoftmax::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("TKSoftmax has no jvp implementation.");
}

std::vector<array> TKSoftmax::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  throw std::runtime_error("TKSoftmax has no vjp implementation.");
}

std::pair<std::vector<array>, std::vector<int>> TKSoftmax::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("TKSoftmax has no vmap implementation.");
}

bool TKSoftmax::is_equivalent(const Primitive& other) const {
  return true;
}

} // namespace mlx::core
