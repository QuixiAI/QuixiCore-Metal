// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "attn_causal/attn_causal.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array attn_causal(
    const array& q,
    const array& k,
    const array& v,
    StreamOrDevice s /* = {} */
) {
  assert(q.dtype() == bfloat16 && k.dtype() == bfloat16 && v.dtype() == bfloat16);
  assert(q.shape() == k.shape() && k.shape() == v.shape());
  const int D = q.shape(3);
  assert(D == 64 || D == 128);

  return array(
      q.shape(), bfloat16,
      std::make_shared<AttnCausal>(to_stream(s)),
      {q, k, v});
}

void AttnCausal::eval(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(false); // no CPU fallback.
}
void AttnCausal::eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void AttnCausal::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int N = q.shape(2);
  const int D = q.shape(3);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_attn_causal(enc, q, k, v, out, static_cast<unsigned>(N),
                         static_cast<unsigned>(H), B, D);
}

std::vector<array> AttnCausal::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnCausal has no jvp implementation.");
}
std::vector<array> AttnCausal::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("AttnCausal has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> AttnCausal::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnCausal has no vmap implementation.");
}
bool AttnCausal::is_equivalent(const Primitive& other) const {
  return true;
}

} // namespace mlx::core
