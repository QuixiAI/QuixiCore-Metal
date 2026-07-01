// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "moe/moe.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

std::vector<array> moe_route_topk(const array& logits, int k, StreamOrDevice s /* = {} */) {
  if (logits.ndim() != 2) {
    throw std::invalid_argument("moe_route_topk: logits must have shape (num_tokens, num_experts)");
  }
  if (!(logits.dtype() == float32 || logits.dtype() == float16 || logits.dtype() == bfloat16)) {
    throw std::invalid_argument("moe_route_topk: logits must be float32, float16, or bfloat16");
  }
  const int E = logits.shape(1);
  if (k <= 0 || k > 16 || k > E) {
    throw std::invalid_argument("moe_route_topk: require 1 <= k <= min(16, num_experts)");
  }
  const int T = logits.shape(0);
  auto x = contiguous(logits, false, s);
  return array::make_arrays(
      {{T, k}, {T, k}},
      {int32, float32},
      std::make_shared<MoeRouteTopk>(to_stream(s), k),
      {x});
}

void MoeRouteTopk::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MoeRouteTopk has no CPU implementation.");
}

void MoeRouteTopk::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& ids = outputs[0];
  auto& weights = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  ids.set_data(allocator::malloc_or_wait(ids.nbytes()));
  weights.set_data(allocator::malloc_or_wait(weights.nbytes()));
  const int T = logits.shape(0);
  const int E = logits.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_moe_route_topk(enc, logits, ids, weights, T, E, k_, type_to_name(logits));
}

std::vector<array> MoeRouteTopk::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("MoeRouteTopk has no jvp implementation.");
}
std::vector<array> MoeRouteTopk::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("MoeRouteTopk has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> MoeRouteTopk::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("MoeRouteTopk has no vmap implementation.");
}

// ----------------------------- moe_permute -----------------------------

std::vector<array> moe_permute(const array& topk_ids, int num_experts, StreamOrDevice s /* = {} */) {
  if (topk_ids.ndim() != 2) {
    throw std::invalid_argument("moe_permute: topk_ids must have shape (num_tokens, k)");
  }
  if (num_experts <= 0) {
    throw std::invalid_argument("moe_permute: num_experts must be positive");
  }
  const int T = topk_ids.shape(0);
  const int K = topk_ids.shape(1);
  const int TK = T * K;
  auto ids = contiguous(astype(topk_ids, int32, s), false, s);
  return array::make_arrays(
      {{TK}, {num_experts + 1}, {TK}, {num_experts}, {num_experts}},
      {int32, int32, int32, int32, int32},
      std::make_shared<MoePermute>(to_stream(s), num_experts),
      {ids});
}

void MoePermute::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MoePermute has no CPU implementation.");
}

void MoePermute::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& ids = inputs[0];
  auto& sorted_row_idx = outputs[0];
  auto& offsets = outputs[1];
  auto& inv_idx = outputs[2];
  auto& counts = outputs[3];
  auto& cursor = outputs[4];

  auto& s = stream();
  auto& d = metal::device(s.device);
  for (auto* o : {&sorted_row_idx, &offsets, &inv_idx, &counts, &cursor}) {
    o->set_data(allocator::malloc_or_wait(o->nbytes()));
  }
  const int TK = static_cast<int>(ids.size());

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_moe_zero_i32(enc, counts, num_experts_);
  tk::launch_moe_histogram(enc, ids, counts, TK);
  tk::launch_moe_scan_offsets(enc, counts, offsets, cursor, num_experts_);
  tk::launch_moe_scatter(enc, ids, cursor, sorted_row_idx, inv_idx, TK);
}

// ----------------------------- moe_finalize -----------------------------

array moe_finalize(
    const array& expert_out, const array& inv_idx, const array& topk_weights, int k,
    StreamOrDevice s /* = {} */) {
  if (expert_out.ndim() != 2) {
    throw std::invalid_argument("moe_finalize: expert_out must have shape (T*k, Hdim)");
  }
  if (topk_weights.ndim() != 2 || topk_weights.shape(1) != k) {
    throw std::invalid_argument("moe_finalize: topk_weights must have shape (num_tokens, k)");
  }
  if (!(expert_out.dtype() == float32 || expert_out.dtype() == float16 ||
        expert_out.dtype() == bfloat16)) {
    throw std::invalid_argument("moe_finalize: expert_out must be float");
  }
  const int T = topk_weights.shape(0);
  const int Hdim = expert_out.shape(1);
  auto eo = contiguous(expert_out, false, s);
  auto inv = contiguous(astype(inv_idx, int32, s), false, s);
  auto w = contiguous(astype(topk_weights, float32, s), false, s);
  return array(
      {T, Hdim}, expert_out.dtype(), std::make_shared<MoeFinalize>(to_stream(s), k), {eo, inv, w});
}

void MoeFinalize::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MoeFinalize has no CPU implementation.");
}

void MoeFinalize::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& expert_out = inputs[0];
  auto& inv_idx = inputs[1];
  auto& weights = inputs[2];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int T = out.shape(0);
  const int Hdim = out.shape(1);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_moe_finalize(enc, expert_out, inv_idx, weights, out, T, k_, Hdim, type_to_name(out));
}

// ----------------------------- moe_grouped_gemm -----------------------------

array moe_grouped_gemm(
    const array& permuted_input, const array& W, const array& expert_of_tile, StreamOrDevice s) {
  if (permuted_input.ndim() != 2) {
    throw std::invalid_argument("moe_grouped_gemm: permuted_input must be (total_rows, H)");
  }
  if (W.ndim() != 3) {
    throw std::invalid_argument("moe_grouped_gemm: W must be (num_experts, H, H)");
  }
  const int total_rows = permuted_input.shape(0);
  const int H = permuted_input.shape(1);
  if (total_rows % 32 != 0 || H % 32 != 0) {
    throw std::invalid_argument("moe_grouped_gemm: total_rows and H must be multiples of 32");
  }
  if (W.shape(1) != H || W.shape(2) != H) {
    throw std::invalid_argument("moe_grouped_gemm: W must be (E, H, H)");
  }
  if (expert_of_tile.ndim() != 1 || expert_of_tile.shape(0) != total_rows / 32) {
    throw std::invalid_argument("moe_grouped_gemm: expert_of_tile must be (total_rows/32,)");
  }
  auto dtype = permuted_input.dtype();
  if (!(dtype == float32 || dtype == bfloat16)) {
    throw std::invalid_argument("moe_grouped_gemm: dtype must be float32 or bfloat16");
  }
  auto a = contiguous(permuted_input, false, s);
  auto w = contiguous(astype(W, dtype, s), false, s);
  auto eot = contiguous(astype(expert_of_tile, int32, s), false, s);
  return array(
      {total_rows, H}, dtype, std::make_shared<MoeGroupedGemm>(to_stream(s)), {a, w, eot});
}

void MoeGroupedGemm::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MoeGroupedGemm has no CPU implementation.");
}

void MoeGroupedGemm::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& a = inputs[0];
  auto& w = inputs[1];
  auto& eot = inputs[2];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int total_rows = a.shape(0);
  const int H = a.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_moe_grouped_gemm(enc, out, a, w, eot, total_rows, H, type_to_name(a));
}

#define TK_MOE_NO_AUTODIFF(CLASS, LABEL)                                     \
  std::vector<array> CLASS::jvp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&) {                                             \
    throw std::runtime_error(LABEL " has no jvp implementation.");           \
  }                                                                          \
  std::vector<array> CLASS::vjp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&, const std::vector<array>&) {                  \
    throw std::runtime_error(LABEL " has no vjp implementation.");           \
  }                                                                          \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(               \
      const std::vector<array>&, const std::vector<int>&) {                  \
    throw std::runtime_error(LABEL " has no vmap implementation.");          \
  }

TK_MOE_NO_AUTODIFF(MoePermute, "MoePermute")
TK_MOE_NO_AUTODIFF(MoeFinalize, "MoeFinalize")
TK_MOE_NO_AUTODIFF(MoeGroupedGemm, "MoeGroupedGemm")

} // namespace mlx::core
