// Copyright © 2024 Apple Inc.

#include <cassert>
#include <stdexcept>

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

std::vector<array> qk_norm_rope_kv_f16(
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
    throw std::invalid_argument("qk_norm_rope_kv_f16: qkv must be (T, (Hq+Hk+Hv)*D) bfloat16");
  }
  const int HT = num_heads_q + num_heads_k + num_heads_v;
  if (HT <= 0 || num_heads_q <= 0 || num_heads_k <= 0 || num_heads_v <= 0 ||
      qkv.shape(1) % HT != 0) {
    throw std::invalid_argument("qk_norm_rope_kv_f16: head counts must divide qkv width");
  }
  const int T = qkv.shape(0);
  const int D = qkv.shape(1) / HT;
  if (!(D == 64 || D == 128 || D == 256)) {
    throw std::invalid_argument("qk_norm_rope_kv_f16: head_dim must be 64, 128 or 256");
  }
  if (q_weight.ndim() != 1 || q_weight.shape(0) != D ||
      k_weight.ndim() != 1 || k_weight.shape(0) != D) {
    throw std::invalid_argument("qk_norm_rope_kv_f16: q_weight/k_weight must be (D,)");
  }
  if (cosb.ndim() != 2 || cosb.shape(1) != D / 2 || sinb.shape() != cosb.shape()) {
    throw std::invalid_argument("qk_norm_rope_kv_f16: cos/sin must be (max_pos, D/2)");
  }
  if (positions.ndim() != 1 || positions.shape(0) != T) {
    throw std::invalid_argument("qk_norm_rope_kv_f16: positions must be (T,)");
  }
  return array::make_arrays(
      {{T, num_heads_q * D}, {T, num_heads_k * D}, {T, num_heads_v * D}},
      {bfloat16, float16, float16},
      std::make_shared<QkNormRopeKvF16>(to_stream(s), num_heads_q, num_heads_k, num_heads_v, eps,
                                        interleaved, gemma),
      {contiguous(qkv, false, s),
       contiguous(astype(q_weight, bfloat16, s), false, s),
       contiguous(astype(k_weight, bfloat16, s), false, s),
       contiguous(astype(cosb, bfloat16, s), false, s),
       contiguous(astype(sinb, bfloat16, s), false, s),
       contiguous(astype(positions, int32, s), false, s)});
}

void QkNormRopeKvF16::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QkNormRopeKvF16 has no CPU implementation.");
}

void QkNormRopeKvF16::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 6);
  auto& qkv = inputs[0];
  auto& qw = inputs[1];
  auto& kw = inputs[2];
  auto& cosb = inputs[3];
  auto& sinb = inputs[4];
  auto& pos = inputs[5];
  auto& q_out = outputs[0];
  auto& k_out = outputs[1];
  auto& v_out = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  q_out.set_data(allocator::malloc_or_wait(q_out.nbytes()));
  k_out.set_data(allocator::malloc_or_wait(k_out.nbytes()));
  v_out.set_data(allocator::malloc_or_wait(v_out.nbytes()));
  const int T = qkv.shape(0);
  const int HT = hq_ + hk_ + hv_;
  const int D = qkv.shape(1) / HT;
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qk_norm_rope_kv_f16(enc, qkv, qw, kw, cosb, sinb, pos, q_out, k_out, v_out, T,
                                 hq_, hk_, hv_, D, eps_, interleaved_ ? 1 : 0, gemma_ ? 1 : 0);
}

}  // namespace mlx::core
