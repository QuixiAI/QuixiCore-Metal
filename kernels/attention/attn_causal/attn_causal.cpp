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
    float softcap /* = 0.0f */,
    const std::optional<array>& sinks /* = std::nullopt */,
    StreamOrDevice s /* = {} */
) {
  assert(q.dtype() == bfloat16 && k.dtype() == bfloat16 && v.dtype() == bfloat16);
  assert(q.shape() == k.shape() && k.shape() == v.shape());
  const int H = q.shape(1);
  const int D = q.shape(3);
  assert(D == 64 || D == 128);

  const bool has_sink = sinks.has_value();
  if (has_sink && (sinks->ndim() != 1 || sinks->shape(0) != H)) {
    throw std::invalid_argument("attn_causal: sinks must be (H,)");
  }
  auto sink_arr = has_sink ? astype(*sinks, float32, s) : q;
  return array(
      q.shape(), bfloat16,
      std::make_shared<AttnCausal>(to_stream(s), softcap, has_sink),
      {q, k, v, sink_arr});
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
  assert(inputs.size() == 4);
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& sinks = inputs[3];   // == q (placeholder) when has_sink_ is false
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
                         static_cast<unsigned>(H), B, D, softcap_, sinks,
                         has_sink_ ? 1 : 0);
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
  auto& o = static_cast<const AttnCausal&>(other);
  return softcap_ == o.softcap_ && has_sink_ == o.has_sink_;
}

array attn_window(
    const array& q,
    const array& k,
    const array& v,
    int window,
    float softcap /* = 0.0f */,
    const std::optional<array>& sinks /* = std::nullopt */,
    StreamOrDevice s /* = {} */
) {
  assert(q.dtype() == bfloat16 && k.dtype() == bfloat16 && v.dtype() == bfloat16);
  assert(q.shape() == k.shape() && k.shape() == v.shape());
  const int H = q.shape(1);
  const int D = q.shape(3);
  assert(D == 64 || D == 128);
  assert(q.shape(2) % 8 == 0 && "attn_window: N must be a multiple of 8");

  const bool has_sink = sinks.has_value();
  if (has_sink && (sinks->ndim() != 1 || sinks->shape(0) != H)) {
    throw std::invalid_argument("attn_window: sinks must be (H,)");
  }
  auto sink_arr = has_sink ? astype(*sinks, float32, s) : q;
  return array(
      q.shape(), bfloat16,
      std::make_shared<AttnWindow>(to_stream(s), window, softcap, has_sink),
      {q, k, v, sink_arr});
}

void AttnWindow::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("AttnWindow has no CPU implementation.");
}

void AttnWindow::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 4);
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& sinks = inputs[3];   // == q (placeholder) when has_sink_ is false
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
  tk::launch_attn_window(enc, q, k, v, out, static_cast<unsigned>(N),
                         static_cast<unsigned>(H), B, D, window_, softcap_, sinks,
                         has_sink_ ? 1 : 0);
}

std::vector<array> AttnWindow::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnWindow has no jvp implementation.");
}
std::vector<array> AttnWindow::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("AttnWindow has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> AttnWindow::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnWindow has no vmap implementation.");
}

} // namespace mlx::core
