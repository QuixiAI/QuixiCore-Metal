// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "gelu/gelu.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array gelu(const array& x, StreamOrDevice s /* = {} */) {
  assert(x.dtype() == bfloat16);
  const int D = x.shape(-1);
  assert((D == 256 || D == 512 || D == 768 || D == 1024) &&
         "gelu: last dim must be 256, 512, 768, or 1024");
  return array(
      x.shape(), bfloat16,
      std::make_shared<Gelu>(to_stream(s)),
      {x});
}

void Gelu::eval(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(false); // no CPU fallback; use mx.nn.gelu_approx for a reference.
}

void Gelu::eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void Gelu::eval_gpu(
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
  tk::launch_gelu(enc, x, out, M, D);
}

std::vector<array> Gelu::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("Gelu has no jvp implementation.");
}
std::vector<array> Gelu::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("Gelu has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> Gelu::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("Gelu has no vmap implementation.");
}
// ---- GELU backward (elementwise dx = dy * gelu'(x)) ----
array gelu_bwd(const array& x, const array& dy, StreamOrDevice s /* = {} */) {
  const Dtype dt = x.dtype();
  auto x_c = contiguous(x, false, s);
  auto dy_c = contiguous(astype(dy, dt, s), false, s);
  return array(x.shape(), dt, std::make_shared<GeluBwd>(to_stream(s)), {x_c, dy_c});
}

void GeluBwd::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("GeluBwd has no CPU implementation.");
}
void GeluBwd::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& dy = inputs[1];
  auto& dx = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  dx.set_data(allocator::malloc_or_wait(dx.nbytes()));
  const int n = static_cast<int>(x.size());
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_gelu_bwd(enc, x, dy, dx, n, type_to_name(x));
}
std::vector<array> GeluBwd::jvp(const std::vector<array>&, const std::vector<array>&,
                                const std::vector<int>&) {
  throw std::runtime_error("GeluBwd has no jvp implementation.");
}
std::vector<array> GeluBwd::vjp(const std::vector<array>&, const std::vector<array>&,
                                const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("GeluBwd has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> GeluBwd::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("GeluBwd has no vmap implementation.");
}

bool Gelu::is_equivalent(const Primitive& other) const {
  return true;
}

} // namespace mlx::core
