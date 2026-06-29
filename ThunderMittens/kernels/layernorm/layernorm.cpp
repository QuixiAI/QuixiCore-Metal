// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "layernorm/layernorm.h"

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

/**
 *  LayerNorm over the last axis:
 *      y = (x - mean(x)) * rsqrt(var(x) + eps) * weight + bias
 *
 *  Inputs are assumed bf16 and row-contiguous. The last dim D must be one of
 *  the instantiated widths {256, 512, 768, 1024}.
 **/
array layernorm(
    const array& x,
    const array& weight,
    const array& bias,
    float eps /* = 1e-5f */,
    StreamOrDevice s /* = {} */
) {
  assert(x.dtype() == bfloat16 && weight.dtype() == bfloat16 &&
         bias.dtype() == bfloat16);
  const int D = x.shape(-1);
  assert(weight.ndim() == 1 && weight.shape(0) == D);
  assert(bias.ndim() == 1 && bias.shape(0) == D);
  assert((D == 256 || D == 512 || D == 768 || D == 1024) &&
         "layernorm: last dim must be 256, 512, 768, or 1024");

  return array(
      /* const std::vector<int>& shape = */ x.shape(),
      /* Dtype dtype = */ bfloat16,
      /* std::shared_ptr<Primitive> primitive = */
      std::make_shared<LayerNorm>(to_stream(s), eps),
      /* const std::vector<array>& inputs = */ {x, weight, bias});
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

/** Fall back implementation for evaluation on CPU */
void LayerNorm::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // No CPU fallback yet; use mx.fast.layer_norm for a reference.
  assert(false);
}

void LayerNorm::eval_cpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

/** Evaluate primitive on GPU */
void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& x = inputs[0];
  auto& weight = inputs[1];
  auto& bias = inputs[2];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int D = x.shape(-1);
  const uint32_t M = static_cast<uint32_t>(x.size() / D); // rows = prod(shape[:-1])

  // Dispatch via the shared host ABI (one simdgroup per row; M threadgroups).
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_layernorm(enc, x, weight, bias, out, M, D, eps_);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

std::vector<array> LayerNorm::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("LayerNorm has no jvp implementation.");
}

std::vector<array> LayerNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  throw std::runtime_error("LayerNorm has no vjp implementation.");
}

std::pair<std::vector<array>, std::vector<int>> LayerNorm::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("LayerNorm has no vmap implementation.");
}

/** Equivalence check **/
bool LayerNorm::is_equivalent(const Primitive& other) const {
  const LayerNorm& r_other = static_cast<const LayerNorm&>(other);
  return eps_ == r_other.eps_;
}

} // namespace mlx::core
