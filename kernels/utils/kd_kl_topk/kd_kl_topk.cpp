#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "kd_kl_topk/kd_kl_topk.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {

bool kd_is_float(Dtype d) {
  return d == float32 || d == float16 || d == bfloat16;
}

void kd_check_common(const array& logits, const array& t_idx, const array& t_prob, const char* name) {
  if (logits.ndim() != 2 || t_idx.ndim() != 2 || t_prob.ndim() != 2 ||
      t_idx.shape() != t_prob.shape() || t_idx.shape(0) != logits.shape(0)) {
    throw std::invalid_argument(std::string(name) + ": expected logits (T,V), t_idx/t_prob (T,K)");
  }
  if (!kd_is_float(logits.dtype())) {
    throw std::invalid_argument(std::string(name) + ": logits must be float32/float16/bfloat16");
  }
  if (t_idx.dtype() != int32 || t_prob.dtype() != float32) {
    throw std::invalid_argument(std::string(name) + ": t_idx must be int32 and t_prob float32");
  }
}

void kd_check_tail(int tail_mode, const char* name) {
  if (tail_mode != 0 && tail_mode != 1) {
    throw std::invalid_argument(std::string(name) + ": tail_mode must be 0 or 1");
  }
}

} // namespace

std::vector<array> kd_kl_topk_fwd(
    const array& logits, const array& t_idx, const array& t_prob, float invtemp, int tail_mode,
    StreamOrDevice s) {
  kd_check_common(logits, t_idx, t_prob, "kd_kl_topk_fwd");
  kd_check_tail(tail_mode, "kd_kl_topk_fwd");
  const int T = logits.shape(0);
  return array::make_arrays(
      {{T}, {T}},
      {float32, float32},
      std::make_shared<KdKlTopkFwd>(to_stream(s), invtemp, tail_mode),
      {contiguous(logits, false, s), contiguous(t_idx, false, s), contiguous(t_prob, false, s)});
}

array kd_kl_topk_bwd(
    const array& logits, const array& t_idx, const array& t_prob, const array& lse,
    const array& grad_out, float invtemp, int tail_mode, StreamOrDevice s) {
  kd_check_common(logits, t_idx, t_prob, "kd_kl_topk_bwd");
  kd_check_tail(tail_mode, "kd_kl_topk_bwd");
  const int T = logits.shape(0);
  if (lse.shape() != std::vector<int>{T} || grad_out.shape() != std::vector<int>{T}) {
    throw std::invalid_argument("kd_kl_topk_bwd: lse and grad_out must be (T,)");
  }
  return array(
      logits.shape(), logits.dtype(),
      std::make_shared<KdKlTopkBwd>(to_stream(s), invtemp, tail_mode),
      {contiguous(logits, false, s), contiguous(t_idx, false, s), contiguous(t_prob, false, s),
       contiguous(astype(lse, float32, s), false, s),
       contiguous(astype(grad_out, float32, s), false, s)});
}

void KdKlTopkFwd::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("KdKlTopkFwd has no CPU implementation.");
}

void KdKlTopkFwd::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& t_idx = inputs[1];
  auto& t_prob = inputs[2];
  auto& loss = outputs[0];
  auto& lse = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  loss.set_data(allocator::malloc_or_wait(loss.nbytes()));
  lse.set_data(allocator::malloc_or_wait(lse.nbytes()));
  const int rows = logits.shape(0);
  const int V = logits.shape(1);
  const int K = t_idx.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_kd_kl_topk_fwd(
      enc, logits, t_idx, t_prob, loss, lse, rows, V, K, invtemp_, tail_mode_,
      type_to_name(logits));
}

void KdKlTopkBwd::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("KdKlTopkBwd has no CPU implementation.");
}

void KdKlTopkBwd::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& t_idx = inputs[1];
  auto& t_prob = inputs[2];
  auto& lse = inputs[3];
  auto& grad_out = inputs[4];
  auto& grad = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  grad.set_data(allocator::malloc_or_wait(grad.nbytes()));
  const int rows = logits.shape(0);
  const int V = logits.shape(1);
  const int K = t_idx.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_kd_kl_topk_bwd(
      enc, logits, t_idx, t_prob, lse, grad_out, grad, rows, V, K, invtemp_, tail_mode_,
      type_to_name(logits));
}

#define TK_KD_NO_AUTODIFF(CLASS, LABEL)                                       \
  std::vector<array> CLASS::jvp(                                              \
      const std::vector<array>&, const std::vector<array>&,                   \
      const std::vector<int>&) {                                              \
    throw std::runtime_error(LABEL " has no jvp implementation.");            \
  }                                                                           \
  std::vector<array> CLASS::vjp(                                              \
      const std::vector<array>&, const std::vector<array>&,                   \
      const std::vector<int>&, const std::vector<array>&) {                   \
    throw std::runtime_error(LABEL " has no vjp implementation.");            \
  }                                                                           \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(                \
      const std::vector<array>&, const std::vector<int>&) {                   \
    throw std::runtime_error(LABEL " has no vmap implementation.");           \
  }

TK_KD_NO_AUTODIFF(KdKlTopkFwd, "KdKlTopkFwd")
TK_KD_NO_AUTODIFF(KdKlTopkBwd, "KdKlTopkBwd")

} // namespace mlx::core
