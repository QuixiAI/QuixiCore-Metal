#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "weight_quant_ternary/weight_quant_ternary.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {

bool wqt_is_float(Dtype d) {
  return d == float32 || d == float16 || d == bfloat16;
}

void wqt_check(const array& w, const char* name) {
  if (w.ndim() != 2 && w.ndim() != 3) {
    throw std::invalid_argument(std::string(name) + ": W must be (N,K) or (E,N,K)");
  }
  if (!wqt_is_float(w.dtype())) {
    throw std::invalid_argument(std::string(name) + ": W must be float32/float16/bfloat16");
  }
  if (w.shape(-1) % 32 != 0) {
    throw std::invalid_argument(std::string(name) + ": K must be a multiple of 32");
  }
}

std::vector<int> wqt_qshape(const array& w) {
  std::vector<int> sh(w.shape().begin(), w.shape().end() - 1);
  sh.push_back(w.shape(-1) / 32);
  sh.push_back(10);
  return sh;
}

int wqt_E(const array& w) {
  return w.ndim() == 3 ? w.shape(0) : 1;
}

int wqt_N(const array& w) {
  return w.shape(-2);
}

int wqt_K(const array& w) {
  return w.shape(-1);
}

} // namespace

std::vector<array> weight_quant_ternary(const array& w, int group_k, StreamOrDevice s) {
  wqt_check(w, "weight_quant_ternary");
  const int K = wqt_K(w);
  if (group_k < 32 || group_k % 32 != 0 || K % group_k != 0) {
    throw std::invalid_argument(
        "weight_quant_ternary: group_k must be a multiple of 32 that divides K");
  }
  return array::make_arrays(
      {wqt_qshape(w), w.shape()},
      {uint8, bfloat16},
      std::make_shared<WeightQuantTernary>(to_stream(s), group_k),
      {contiguous(w, false, s)});
}

std::vector<array> weight_quant_ternary_pt(const array& w, StreamOrDevice s) {
  wqt_check(w, "weight_quant_ternary_pt");
  const int E = wqt_E(w);
  return array::make_arrays(
      {wqt_qshape(w), w.shape(), {E}},
      {uint8, bfloat16, float32},
      std::make_shared<WeightQuantTernaryPt>(to_stream(s)),
      {contiguous(w, false, s)});
}

void WeightQuantTernary::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("WeightQuantTernary has no CPU implementation.");
}

void WeightQuantTernary::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& w = inputs[0];
  auto& wq = outputs[0];
  auto& w_deq = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  wq.set_data(allocator::malloc_or_wait(wq.nbytes()));
  w_deq.set_data(allocator::malloc_or_wait(w_deq.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_weight_quant_ternary(
      enc, w, wq, w_deq, wqt_E(w), wqt_N(w), wqt_K(w), group_k_, type_to_name(w));
}

void WeightQuantTernaryPt::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("WeightQuantTernaryPt has no CPU implementation.");
}

void WeightQuantTernaryPt::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& w = inputs[0];
  auto& wq = outputs[0];
  auto& w_deq = outputs[1];
  auto& abssum = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  wq.set_data(allocator::malloc_or_wait(wq.nbytes()));
  w_deq.set_data(allocator::malloc_or_wait(w_deq.nbytes()));
  abssum.set_data(allocator::malloc_or_wait(abssum.nbytes()));
  const int E = wqt_E(w);
  const int N = wqt_N(w);
  const int K = wqt_K(w);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_weight_quant_zero_float(enc, abssum, E);
  tk::launch_weight_quant_ternary_abssum(enc, w, abssum, E, N * K, type_to_name(w));
  tk::launch_weight_quant_ternary_pt_encode(enc, w, abssum, wq, w_deq, E, N, K, type_to_name(w));
}

#define TK_WQT_NO_AUTODIFF(CLASS, LABEL)                                      \
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

TK_WQT_NO_AUTODIFF(WeightQuantTernary, "WeightQuantTernary")
TK_WQT_NO_AUTODIFF(WeightQuantTernaryPt, "WeightQuantTernaryPt")

} // namespace mlx::core
