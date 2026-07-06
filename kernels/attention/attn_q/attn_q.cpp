// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "attn_q/attn_q.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array attn_q(const array& q, const array& kq, const array& vq,
             const std::string& format, bool causal, bool multiwarp, StreamOrDevice s) {
  assert(q.dtype() == bfloat16 && kq.dtype() == uint8 && vq.dtype() == uint8);
  assert(q.ndim() == 4 && kq.ndim() == 5 && vq.ndim() == 5);
  const int D = q.shape(3);
  assert((D == 64 || D == 128) && q.shape(2) % 8 == 0);
  (void)D;
  return array(q.shape(), bfloat16,
               std::make_shared<AttnQ>(to_stream(s), format, causal, multiwarp), {q, kq, vq});
}

void AttnQ::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void AttnQ::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void AttnQ::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& q = inputs[0]; auto& kq = inputs[1]; auto& vq = inputs[2];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_attn_q(enc, q, kq, vq, out, static_cast<unsigned>(N),
                    static_cast<unsigned>(H), B, D, fmt_, causal_, mw_);
}

std::vector<array> AttnQ::jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnQ has no jvp."); }
std::vector<array> AttnQ::vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("AttnQ has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> AttnQ::vmap(const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnQ has no vmap."); }
bool AttnQ::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const AttnQ&>(other);
  return fmt_ == o.fmt_ && causal_ == o.causal_ && mw_ == o.mw_;
}

} // namespace mlx::core
