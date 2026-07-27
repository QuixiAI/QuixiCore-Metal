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

array qk_norm_rope_positioned(
    const array& qkv,
    const array& q_weight,
    const array& k_weight,
    const array& cosb,
    const array& sinb,
    const array& positions,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    int rotary_dim,
    float eps,
    bool interleaved,
    float norm_weight_offset,
    const std::vector<int>& mrope_sections,
    bool section_interleaved,
    StreamOrDevice s) {
  if (qkv.ndim() != 2 || qkv.dtype() != bfloat16) {
    throw std::invalid_argument(
        "qk_norm_rope_positioned: qkv must be (T, (Hq+Hk+Hv)*D) bfloat16");
  }
  const int HT = num_heads_q + num_heads_k + num_heads_v;
  if (num_heads_q <= 0 || num_heads_k <= 0 || num_heads_v < 0 || HT <= 0 ||
      qkv.shape(1) % HT != 0) {
    throw std::invalid_argument(
        "qk_norm_rope_positioned: nonnegative head counts must divide qkv width");
  }
  const int D = qkv.shape(1) / HT;
  if (!(D == 64 || D == 128 || D == 256 || D == 512)) {
    throw std::invalid_argument(
        "qk_norm_rope_positioned: head_dim must be 64, 128, 256 or 512");
  }
  const int rd = rotary_dim == 0 ? D : rotary_dim;
  if (rd <= 0 || rd > D || rd % 2 != 0) {
    throw std::invalid_argument(
        "qk_norm_rope_positioned: rotary_dim must be positive, even, and <= head_dim");
  }
  if (q_weight.ndim() != 1 || q_weight.shape(0) != D ||
      k_weight.ndim() != 1 || k_weight.shape(0) != D) {
    throw std::invalid_argument(
        "qk_norm_rope_positioned: q_weight/k_weight must both be (D,)");
  }
  if (cosb.ndim() != 2 || sinb.ndim() != 2 || cosb.shape() != sinb.shape() ||
      cosb.shape(1) != rd / 2) {
    throw std::invalid_argument(
        "qk_norm_rope_positioned: cos/sin must be (max_pos, rotary_dim/2)");
  }
  const int T = qkv.shape(0);
  const bool multimodal = !mrope_sections.empty();
  if ((!multimodal && (positions.ndim() != 1 || positions.shape(0) != T)) ||
      (multimodal && (positions.ndim() != 2 || positions.shape(0) != 3 ||
                      positions.shape(1) != T))) {
    throw std::invalid_argument(
        "qk_norm_rope_positioned: positions must be (T,), or (3,T) with M-RoPE sections");
  }
  if (multimodal) {
    if (interleaved) {
      throw std::invalid_argument(
          "qk_norm_rope_positioned: M-RoPE uses split-half pairing; interleaved is an axis mode");
    }
    if (mrope_sections.size() != 3 || mrope_sections[0] < 0 || mrope_sections[1] < 0 ||
        mrope_sections[2] < 0 ||
        mrope_sections[0] + mrope_sections[1] + mrope_sections[2] != rd / 2) {
      throw std::invalid_argument(
          "qk_norm_rope_positioned: M-RoPE sections must sum to rotary_dim/2");
    }
    if (section_interleaved &&
        (mrope_sections[0] != (rd / 2 + 2) / 3 ||
         mrope_sections[1] != (rd / 2 + 1) / 3 ||
         mrope_sections[2] != (rd / 2) / 3)) {
      throw std::invalid_argument(
          "qk_norm_rope_positioned: interleaved sections must describe THWTHW... counts");
    }
  } else if (section_interleaved) {
    throw std::invalid_argument(
        "qk_norm_rope_positioned: section_interleaved requires M-RoPE sections");
  }
  return array(
      qkv.shape(), bfloat16,
      std::make_shared<QkNormRopePositioned>(
          to_stream(s), num_heads_q, num_heads_k, num_heads_v, rd, eps, interleaved,
          norm_weight_offset, mrope_sections, section_interleaved),
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

void QkNormRopePositioned::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QkNormRopePositioned has no CPU implementation.");
}

void QkNormRopePositioned::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& qkv = inputs[0];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int T = qkv.shape(0);
  const int D = qkv.shape(1) / (hq_ + hk_ + hv_);
  const int position_mode = sections_.empty() ? 0 : (section_interleaved_ ? 2 : 1);
  const int st = sections_.empty() ? 0 : sections_[0];
  const int sh = sections_.empty() ? 0 : sections_[1];
  const int sw = sections_.empty() ? 0 : sections_[2];
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qk_norm_rope_positioned(
      enc, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], out,
      T, hq_, hk_, hv_, D, eps_, rotary_dim_, interleaved_ ? 1 : 0,
      weight_offset_, position_mode, st, sh, sw);
}

std::vector<array> QkNormRopePositioned::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QkNormRopePositioned has no jvp implementation.");
}
std::vector<array> QkNormRopePositioned::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("QkNormRopePositioned has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> QkNormRopePositioned::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("QkNormRopePositioned has no vmap implementation.");
}
bool QkNormRopePositioned::is_equivalent(const Primitive& other) const {
  auto& o = static_cast<const QkNormRopePositioned&>(other);
  return hq_ == o.hq_ && hk_ == o.hk_ && hv_ == o.hv_ &&
         rotary_dim_ == o.rotary_dim_ && eps_ == o.eps_ &&
         interleaved_ == o.interleaved_ && weight_offset_ == o.weight_offset_ &&
         sections_ == o.sections_ && section_interleaved_ == o.section_interleaved_;
}

} // namespace mlx::core
