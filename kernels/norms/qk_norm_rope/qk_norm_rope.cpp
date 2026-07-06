// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "qk_norm_rope/qk_norm_rope.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array qk_norm_rope(
    const array& qkv,
    const array& q_weight,
    const array& k_weight,
    const array& cosb,
    const array& sinb,
    const array& positions,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    float eps /* = 1e-6f */,
    bool interleaved /* = false */,
    bool gemma /* = false */,
    StreamOrDevice s /* = {} */) {
  if (qkv.ndim() != 2 || qkv.dtype() != bfloat16) {
    throw std::invalid_argument("qk_norm_rope: qkv must be (T, (Hq+Hk+Hv)*D) bfloat16");
  }
  const int HT = num_heads_q + num_heads_k + num_heads_v;
  if (HT <= 0 || num_heads_q <= 0 || num_heads_k <= 0 || qkv.shape(1) % HT != 0) {
    throw std::invalid_argument("qk_norm_rope: head counts must divide qkv width");
  }
  const int D = qkv.shape(1) / HT;
  if (!(D == 64 || D == 128 || D == 256)) {
    throw std::invalid_argument("qk_norm_rope: head_dim must be 64, 128 or 256");
  }
  if (q_weight.ndim() != 1 || q_weight.shape(0) != D ||
      k_weight.ndim() != 1 || k_weight.shape(0) != D) {
    throw std::invalid_argument("qk_norm_rope: q_weight/k_weight must be (D,)");
  }
  if (cosb.ndim() != 2 || cosb.shape(1) != D / 2 || sinb.shape() != cosb.shape()) {
    throw std::invalid_argument("qk_norm_rope: cos/sin must be (max_pos, D/2)");
  }
  if (positions.ndim() != 1 || positions.shape(0) != qkv.shape(0)) {
    throw std::invalid_argument("qk_norm_rope: positions must be (T,)");
  }
  return array(
      qkv.shape(), bfloat16,
      std::make_shared<QkNormRope>(to_stream(s), num_heads_q, num_heads_k, num_heads_v, eps,
                                   interleaved, gemma),
      {contiguous(qkv, false, s),
       contiguous(astype(q_weight, bfloat16, s), false, s),
       contiguous(astype(k_weight, bfloat16, s), false, s),
       contiguous(astype(cosb, bfloat16, s), false, s),
       contiguous(astype(sinb, bfloat16, s), false, s),
       contiguous(astype(positions, int32, s), false, s)});
}

void QkNormRope::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QkNormRope has no CPU implementation.");
}

void QkNormRope::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& qkv = inputs[0];
  auto& qw = inputs[1];
  auto& kw = inputs[2];
  auto& cosb = inputs[3];
  auto& sinb = inputs[4];
  auto& pos = inputs[5];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int T = qkv.shape(0);
  const int HT = hq_ + hk_ + hv_;
  const int D = qkv.shape(1) / HT;
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qk_norm_rope(enc, qkv, qw, kw, cosb, sinb, pos, out, T, hq_, hk_, hv_, D, eps_,
                          interleaved_ ? 1 : 0, gemma_ ? 1 : 0);
}

std::vector<array> QkNormRope::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QkNormRope has no jvp implementation.");
}
std::vector<array> QkNormRope::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("QkNormRope has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> QkNormRope::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QkNormRope has no vmap implementation.");
}

} // namespace mlx::core
