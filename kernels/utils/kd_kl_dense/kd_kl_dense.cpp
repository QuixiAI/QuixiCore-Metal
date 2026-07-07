// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "kd_kl_dense/kd_kl_dense.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
bool kd_dense_float(Dtype d) {
  return d == float32 || d == float16 || d == bfloat16;
}

void kd_dense_check_pair(const array& t, const array& s, const char* name) {
  if (t.ndim() != 2 || s.shape() != t.shape() || t.dtype() != s.dtype() ||
      !kd_dense_float(t.dtype())) {
    throw std::invalid_argument(std::string(name) + ": teacher/student must match float (T,V)");
  }
}

void kd_dense_check_vec(const array& v, int rows, const char* name, const char* field) {
  if (v.shape() != std::vector<int>{rows}) {
    throw std::invalid_argument(std::string(name) + ": " + field + " must be (T,)");
  }
}
} // namespace

std::vector<array> kd_kl_dense_fwd(
    const array& t_logits, const array& s_logits, float invtemp, StreamOrDevice s) {
  kd_dense_check_pair(t_logits, s_logits, "kd_kl_dense_fwd");
  const int T = t_logits.shape(0);
  return array::make_arrays(
      {{T}, {T}, {T}},
      {float32, float32, float32},
      std::make_shared<KdKlDenseFwd>(to_stream(s), invtemp),
      {contiguous(t_logits, false, s), contiguous(s_logits, false, s)});
}

array kd_kl_dense_bwd(
    const array& t_logits, const array& s_logits, const array& lse_t, const array& lse_s,
    const array& grad_out, float invtemp, StreamOrDevice s) {
  kd_dense_check_pair(t_logits, s_logits, "kd_kl_dense_bwd");
  const int T = t_logits.shape(0);
  kd_dense_check_vec(lse_t, T, "kd_kl_dense_bwd", "lse_t");
  kd_dense_check_vec(lse_s, T, "kd_kl_dense_bwd", "lse_s");
  kd_dense_check_vec(grad_out, T, "kd_kl_dense_bwd", "grad_out");
  return array(
      s_logits.shape(), s_logits.dtype(), std::make_shared<KdKlDenseBwd>(to_stream(s), invtemp),
      {contiguous(t_logits, false, s), contiguous(s_logits, false, s),
       contiguous(astype(lse_t, float32, s), false, s),
       contiguous(astype(lse_s, float32, s), false, s),
       contiguous(astype(grad_out, float32, s), false, s)});
}

std::vector<array> kd_ce_fused_fwd(
    const array& t_logits, const array& s_logits, const array& targets,
    float invtemp, StreamOrDevice s) {
  kd_dense_check_pair(t_logits, s_logits, "kd_ce_fused_fwd");
  const int T = t_logits.shape(0);
  if (targets.size() != T) {
    throw std::invalid_argument("kd_ce_fused_fwd: targets must have T elements");
  }
  return array::make_arrays(
      {{T}, {T}, {T}, {T}, {T}},
      {float32, float32, float32, float32, float32},
      std::make_shared<KdCeFusedFwd>(to_stream(s), invtemp),
      {contiguous(t_logits, false, s), contiguous(s_logits, false, s),
       contiguous(astype(targets, int32, s), false, s)});
}

array kd_ce_fused_bwd(
    const array& t_logits, const array& s_logits, const array& targets, const array& lse_sr,
    const array& lse_st, const array& lse_t, const array& go_ce, const array& go_kd,
    float invtemp, StreamOrDevice s) {
  kd_dense_check_pair(t_logits, s_logits, "kd_ce_fused_bwd");
  const int T = t_logits.shape(0);
  if (targets.size() != T) {
    throw std::invalid_argument("kd_ce_fused_bwd: targets must have T elements");
  }
  kd_dense_check_vec(lse_sr, T, "kd_ce_fused_bwd", "lse_sr");
  kd_dense_check_vec(lse_st, T, "kd_ce_fused_bwd", "lse_st");
  kd_dense_check_vec(lse_t, T, "kd_ce_fused_bwd", "lse_t");
  kd_dense_check_vec(go_ce, T, "kd_ce_fused_bwd", "go_ce");
  kd_dense_check_vec(go_kd, T, "kd_ce_fused_bwd", "go_kd");
  return array(
      s_logits.shape(), s_logits.dtype(), std::make_shared<KdCeFusedBwd>(to_stream(s), invtemp),
      {contiguous(t_logits, false, s), contiguous(s_logits, false, s),
       contiguous(astype(targets, int32, s), false, s),
       contiguous(astype(lse_sr, float32, s), false, s),
       contiguous(astype(lse_st, float32, s), false, s),
       contiguous(astype(lse_t, float32, s), false, s),
       contiguous(astype(go_ce, float32, s), false, s),
       contiguous(astype(go_kd, float32, s), false, s)});
}

void KdKlDenseFwd::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("KdKlDenseFwd has no CPU implementation.");
}
void KdKlDenseBwd::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("KdKlDenseBwd has no CPU implementation.");
}
void KdCeFusedFwd::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("KdCeFusedFwd has no CPU implementation.");
}
void KdCeFusedBwd::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("KdCeFusedBwd has no CPU implementation.");
}

void KdKlDenseFwd::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& t = inputs[0];
  auto& s_logits = inputs[1];
  auto& loss = outputs[0];
  auto& lse_t = outputs[1];
  auto& lse_s = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  for (auto* o : {&loss, &lse_t, &lse_s}) {
    o->set_data(allocator::malloc_or_wait(o->nbytes()));
  }
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_kd_kl_dense_fwd(
      enc, t, s_logits, loss, lse_t, lse_s, t.shape(0), t.shape(1), invtemp_, type_to_name(t));
}

void KdKlDenseBwd::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& t = inputs[0];
  auto& s_logits = inputs[1];
  auto& grad = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  grad.set_data(allocator::malloc_or_wait(grad.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_kd_kl_dense_bwd(
      enc, t, s_logits, inputs[2], inputs[3], inputs[4], grad,
      t.shape(0), t.shape(1), invtemp_, type_to_name(t));
}

void KdCeFusedFwd::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& t = inputs[0];
  auto& s_logits = inputs[1];
  auto& targets = inputs[2];
  auto& ce_loss = outputs[0];
  auto& kd = outputs[1];
  auto& lse_sr = outputs[2];
  auto& lse_st = outputs[3];
  auto& lse_t = outputs[4];
  auto& s = stream();
  auto& d = metal::device(s.device);
  for (auto* o : {&ce_loss, &kd, &lse_sr, &lse_st, &lse_t}) {
    o->set_data(allocator::malloc_or_wait(o->nbytes()));
  }
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_kd_ce_fused_fwd(
      enc, t, s_logits, targets, ce_loss, kd, lse_sr, lse_st, lse_t,
      t.shape(0), t.shape(1), invtemp_, type_to_name(t));
}

void KdCeFusedBwd::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& t = inputs[0];
  auto& grad = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  grad.set_data(allocator::malloc_or_wait(grad.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_kd_ce_fused_bwd(
      enc, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
      inputs[6], inputs[7], grad, t.shape(0), t.shape(1), invtemp_, type_to_name(t));
}

#define TK_KD_DENSE_NO_AUTODIFF(CLASS, LABEL)                                 \
  std::vector<array> CLASS::jvp(                                               \
      const std::vector<array>&, const std::vector<array>&,                    \
      const std::vector<int>&) {                                               \
    throw std::runtime_error(LABEL " has no jvp implementation.");             \
  }                                                                            \
  std::vector<array> CLASS::vjp(                                               \
      const std::vector<array>&, const std::vector<array>&,                    \
      const std::vector<int>&, const std::vector<array>&) {                    \
    throw std::runtime_error(LABEL " has no vjp implementation.");             \
  }                                                                            \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(                 \
      const std::vector<array>&, const std::vector<int>&) {                    \
    throw std::runtime_error(LABEL " has no vmap implementation.");            \
  }

TK_KD_DENSE_NO_AUTODIFF(KdKlDenseFwd, "KdKlDenseFwd")
TK_KD_DENSE_NO_AUTODIFF(KdKlDenseBwd, "KdKlDenseBwd")
TK_KD_DENSE_NO_AUTODIFF(KdCeFusedFwd, "KdCeFusedFwd")
TK_KD_DENSE_NO_AUTODIFF(KdCeFusedBwd, "KdCeFusedBwd")

} // namespace mlx::core
