// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

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

namespace {
std::pair<int, int> attn_q_format_layout(const std::string& format) {
  if (format == "q8_0") return {32, 34};
  if (format == "q4_0") return {32, 18};
  if (format == "fp8_e4m3") return {32, 34};
  if (format == "mxfp8") return {32, 33};
  throw std::invalid_argument(
      "attn_q: format must be q8_0, q4_0, fp8_e4m3, or mxfp8");
}
} // namespace

array attn_q(const array& q, const array& kq, const array& vq,
             const std::string& format, bool causal, bool multiwarp, StreamOrDevice s) {
  const auto [block_k, block_bytes] = attn_q_format_layout(format);
  if (q.dtype() != bfloat16 || kq.dtype() != uint8 || vq.dtype() != uint8) {
    throw std::invalid_argument("attn_q: q must be bfloat16 and kq/vq uint8");
  }
  if (q.ndim() != 4 || kq.ndim() != 5 || vq.ndim() != 5) {
    throw std::invalid_argument(
        "attn_q: q must be (B,H,N,D), kq/vq (B,H,N,D/block_k,block_bytes)");
  }
  const int D = q.shape(3);
  const int N = q.shape(2);
  if ((D != 64 && D != 128) || N % 8 != 0 || D % block_k != 0) {
    throw std::invalid_argument("attn_q: D must be 64 or 128 and N must be divisible by 8");
  }
  if (kq.shape() != vq.shape() || kq.shape(0) != q.shape(0) ||
      kq.shape(1) != q.shape(1) || kq.shape(2) != N ||
      kq.shape(3) != D / block_k || kq.shape(4) != block_bytes) {
    throw std::invalid_argument(
        "attn_q: packed kq/vq shape does not match q and format");
  }
  if (multiwarp && (causal || N % 32 != 0)) {
    throw std::invalid_argument(
        "attn_q: multiwarp requires non-causal attention and N divisible by 32");
  }
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
