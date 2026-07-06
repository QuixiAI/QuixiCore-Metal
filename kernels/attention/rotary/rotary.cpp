// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "rotary/rotary.h"

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

array rotary(
    const array& x,
    const array& cos,
    const array& sin,
    bool interleaved /* = false */,
    StreamOrDevice s /* = {} */
) {
  assert(x.dtype() == bfloat16 && cos.dtype() == bfloat16 && sin.dtype() == bfloat16);
  assert(x.ndim() == 4 && "rotary: x must be (B, H, N, D)");
  const int D = x.shape(-1);
  const int N = x.shape(-2);
  assert((D == 64 || D == 128) && "rotary: head dim must be 64 or 128");
  assert(cos.shape(-1) == D / 2 && sin.shape(-1) == D / 2 &&
         cos.shape(-2) == N && sin.shape(-2) == N &&
         "rotary: cos/sin must be (N, D/2)");

  return array(
      /* const std::vector<int>& shape = */ x.shape(),
      /* Dtype dtype = */ bfloat16,
      /* std::shared_ptr<Primitive> primitive = */
      std::make_shared<Rotary>(to_stream(s), interleaved),
      /* const std::vector<array>& inputs = */ {x, cos, sin});
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

void Rotary::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // No CPU fallback yet; use mx.fast.rope for a reference.
  assert(false);
}

void Rotary::eval_cpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

void Rotary::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& x = inputs[0];
  auto& cos = inputs[1];
  auto& sin = inputs[2];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int D = x.shape(-1);
  const unsigned N = static_cast<unsigned>(x.shape(-2));
  const uint32_t M = static_cast<uint32_t>(x.size() / D); // B*H*N rows

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_rotary(enc, x, cos, sin, out, M, N, D, interleaved_);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

std::vector<array> Rotary::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("Rotary has no jvp implementation.");
}

std::vector<array> Rotary::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  throw std::runtime_error("Rotary has no vjp implementation.");
}

std::pair<std::vector<array>, std::vector<int>> Rotary::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("Rotary has no vmap implementation.");
}

bool Rotary::is_equivalent(const Primitive& other) const {
  return interleaved_ == static_cast<const Rotary&>(other).interleaved_;
}

} // namespace mlx::core
