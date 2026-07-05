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

// ----------------------------- moe_pad_schedule / moe_gather -----------------------------

std::vector<array> moe_pad_schedule(
    const array& sorted_row_idx, const array& offsets, int k, StreamOrDevice s /* = {} */) {
  if (sorted_row_idx.ndim() != 1 || offsets.ndim() != 1) {
    throw std::invalid_argument("moe_pad_schedule: sorted_row_idx (TK,), offsets (E+1,)");
  }
  if (k <= 0) {
    throw std::invalid_argument("moe_pad_schedule: k must be positive");
  }
  const int TK = sorted_row_idx.shape(0);
  const int E = offsets.shape(0) - 1;
  // worst case: every expert pads by up to 31 rows; ceil32 gives a whole tile count
  const int total_pad_max = ((TK + 31 * E + 31) / 32) * 32;
  const int max_tiles = total_pad_max / 32;
  auto sorted_c = contiguous(astype(sorted_row_idx, int32, s), false, s);
  auto offsets_c = contiguous(astype(offsets, int32, s), false, s);
  return array::make_arrays(
      {{max_tiles}, {total_pad_max}, {TK}, {E + 1}},
      {int32, int32, int32, int32},
      std::make_shared<MoePadSchedule>(to_stream(s), E, k),
      {sorted_c, offsets_c});
}

void MoePadSchedule::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MoePadSchedule has no CPU implementation.");
}

void MoePadSchedule::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& sorted_row_idx = inputs[0];
  auto& offsets = inputs[1];
  auto& expert_of_tile = outputs[0];
  auto& gather_idx = outputs[1];
  auto& inv_pad = outputs[2];
  auto& off_pad = outputs[3];

  auto& s = stream();
  auto& d = metal::device(s.device);
  for (auto* o : {&expert_of_tile, &gather_idx, &inv_pad, &off_pad}) {
    o->set_data(allocator::malloc_or_wait(o->nbytes()));
  }
  const int TK = sorted_row_idx.shape(0);
  const int max_tiles = expert_of_tile.shape(0);
  const int total_pad_max = gather_idx.shape(0);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_moe_pad_offsets(enc, offsets, off_pad, expert_of_tile, gather_idx,
                             num_experts_, max_tiles, total_pad_max);
  tk::launch_moe_pad_scatter(enc, sorted_row_idx, offsets, off_pad, gather_idx, inv_pad,
                             TK, num_experts_, k_);
}

std::vector<array> MoePadSchedule::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("MoePadSchedule has no jvp implementation.");
}
std::vector<array> MoePadSchedule::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("MoePadSchedule has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> MoePadSchedule::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("MoePadSchedule has no vmap implementation.");
}

array moe_gather(const array& x, const array& gather_idx, StreamOrDevice s /* = {} */) {
  if (x.ndim() != 2 || gather_idx.ndim() != 1) {
    throw std::invalid_argument("moe_gather: x (T, H), gather_idx (total_pad,)");
  }
  if (x.dtype() != float32 && x.dtype() != bfloat16) {
    throw std::invalid_argument("moe_gather: x must be float32 or bfloat16");
  }
  auto idx_c = contiguous(astype(gather_idx, int32, s), false, s);
  return array(
      {gather_idx.shape(0), x.shape(1)}, x.dtype(),
      std::make_shared<MoeGather>(to_stream(s)),
      {contiguous(x, false, s), idx_c});
}

void MoeGather::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MoeGather has no CPU implementation.");
}

void MoeGather::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& gather_idx = inputs[1];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int H = x.shape(1);
  const int total_pad_max = gather_idx.shape(0);
  const std::string tn = x.dtype() == float32 ? "float32" : "bfloat16";

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_moe_gather(enc, x, gather_idx, out, H, total_pad_max, tn);
}

std::vector<array> MoeGather::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("MoeGather has no jvp implementation.");
}
std::vector<array> MoeGather::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("MoeGather has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> MoeGather::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("MoeGather has no vmap implementation.");
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

// --------------------- moe_grouped_gemm_rect / _swiglu ---------------------

array moe_grouped_gemm_rect(
    const array& A, const array& W, const array& expert_of_tile, StreamOrDevice s) {
  if (A.ndim() != 2 || W.ndim() != 3) {
    throw std::invalid_argument("moe_grouped_gemm_rect: A must be (total_rows,K_dim), W (E,K_dim,N_out)");
  }
  const int total_rows = A.shape(0), K_dim = A.shape(1), N_out = W.shape(2);
  if (total_rows % 32 != 0 || K_dim % 16 != 0 || N_out % 32 != 0) {
    throw std::invalid_argument("moe_grouped_gemm_rect: total_rows%32, K_dim%16, N_out%32 required");
  }
  if (W.shape(1) != K_dim) {
    throw std::invalid_argument("moe_grouped_gemm_rect: W must be (E, K_dim, N_out)");
  }
  if (expert_of_tile.ndim() != 1 || expert_of_tile.shape(0) != total_rows / 32) {
    throw std::invalid_argument("moe_grouped_gemm_rect: expert_of_tile must be (total_rows/32,)");
  }
  auto dtype = A.dtype();
  if (!(dtype == float32 || dtype == bfloat16)) {
    throw std::invalid_argument("moe_grouped_gemm_rect: dtype must be float32 or bfloat16");
  }
  auto a = contiguous(A, false, s);
  auto w = contiguous(astype(W, dtype, s), false, s);
  auto eot = contiguous(astype(expert_of_tile, int32, s), false, s);
  return array({total_rows, N_out}, dtype, std::make_shared<MoeGroupedGemmRect>(to_stream(s)),
               {a, w, eot});
}

array moe_grouped_gemm_swiglu(
    const array& A, const array& W1, const array& expert_of_tile, StreamOrDevice s) {
  if (A.ndim() != 2 || W1.ndim() != 3) {
    throw std::invalid_argument("moe_grouped_gemm_swiglu: A (total_rows,H), W1 (E,H,2*inter)");
  }
  const int total_rows = A.shape(0), H = A.shape(1);
  if (W1.shape(2) % 2 != 0) {
    throw std::invalid_argument("moe_grouped_gemm_swiglu: W1 last dim must be 2*inter (even)");
  }
  const int inter = W1.shape(2) / 2;
  if (total_rows % 32 != 0 || H % 16 != 0 || inter % 32 != 0) {
    throw std::invalid_argument("moe_grouped_gemm_swiglu: total_rows%32, H%16, inter%32 required");
  }
  if (W1.shape(1) != H) {
    throw std::invalid_argument("moe_grouped_gemm_swiglu: W1 must be (E, H, 2*inter)");
  }
  if (expert_of_tile.ndim() != 1 || expert_of_tile.shape(0) != total_rows / 32) {
    throw std::invalid_argument("moe_grouped_gemm_swiglu: expert_of_tile must be (total_rows/32,)");
  }
  auto dtype = A.dtype();
  if (!(dtype == float32 || dtype == bfloat16)) {
    throw std::invalid_argument("moe_grouped_gemm_swiglu: dtype must be float32 or bfloat16");
  }
  auto a = contiguous(A, false, s);
  auto w = contiguous(astype(W1, dtype, s), false, s);
  auto eot = contiguous(astype(expert_of_tile, int32, s), false, s);
  return array({total_rows, inter}, dtype, std::make_shared<MoeGroupedGemmSwiglu>(to_stream(s)),
               {a, w, eot});
}

// --------------------- quantized grouped GEMMs ---------------------

namespace {
// Instantiated [[host_name]] format variants and their quant-group sizes (contraction axis).
// Must stay in sync with instantiate_moe_gemm_q in moe.metal and the structs in dequant.metal.
int tk_moe_q_block_k(const std::string& f) {
  if (f == "mxfp4" || f == "fp8_e4m3" || f == "q8_0") return 32;
  if (f == "nvfp4") return 16;
  if (f == "kU4") return 128;
  if (f == "q4_K") return 256;
  return -1;
}
} // namespace

array moe_grouped_gemm_rect_q(
    const array& A, const array& Wq, const array& expert_of_tile, const array& bias,
    bool has_bias, int K_dim, int N_out, const std::string& format, StreamOrDevice s) {
  const int block_k = tk_moe_q_block_k(format);
  if (block_k < 0) {
    throw std::invalid_argument("moe_grouped_gemm_rect_q: unsupported format " + format);
  }
  if (A.ndim() != 2 || Wq.ndim() != 3) {
    throw std::invalid_argument(
        "moe_grouped_gemm_rect_q: A must be (total_rows,K_dim), Wq (E,N_out,row_bytes)");
  }
  const int total_rows = A.shape(0);
  if (A.shape(1) != K_dim || Wq.shape(1) != N_out) {
    throw std::invalid_argument("moe_grouped_gemm_rect_q: K_dim/N_out mismatch with A/Wq shapes");
  }
  if (total_rows % 32 != 0 || K_dim % 32 != 0 || K_dim % block_k != 0 || N_out % 32 != 0) {
    throw std::invalid_argument(
        "moe_grouped_gemm_rect_q: total_rows%32, K_dim%32, K_dim%block_k, N_out%32 required");
  }
  if (A.dtype() != bfloat16 || Wq.dtype() != uint8) {
    throw std::invalid_argument("moe_grouped_gemm_rect_q: A must be bfloat16, Wq uint8");
  }
  if (expert_of_tile.ndim() != 1 || expert_of_tile.shape(0) != total_rows / 32) {
    throw std::invalid_argument("moe_grouped_gemm_rect_q: expert_of_tile must be (total_rows/32,)");
  }
  if (has_bias && (bias.ndim() != 2 || bias.shape(0) != Wq.shape(0) || bias.shape(1) != N_out)) {
    throw std::invalid_argument("moe_grouped_gemm_rect_q: bias must be (E, N_out)");
  }
  auto a = contiguous(A, false, s);
  auto wq = contiguous(Wq, false, s);
  auto eot = contiguous(astype(expert_of_tile, int32, s), false, s);
  auto b = contiguous(astype(bias, bfloat16, s), false, s);
  return array({total_rows, N_out}, bfloat16,
               std::make_shared<MoeGroupedGemmRectQ>(to_stream(s), format, has_bias, K_dim, N_out),
               {a, wq, eot, b});
}

array moe_grouped_gemm_swiglu_q(
    const array& A, const array& W1q, const array& expert_of_tile, const array& bias,
    bool has_bias, int inter, int act_mode, float alpha, float limit,
    const std::string& format, StreamOrDevice s) {
  const int block_k = tk_moe_q_block_k(format);
  if (block_k < 0) {
    throw std::invalid_argument("moe_grouped_gemm_swiglu_q: unsupported format " + format);
  }
  if (A.ndim() != 2 || W1q.ndim() != 3) {
    throw std::invalid_argument(
        "moe_grouped_gemm_swiglu_q: A must be (total_rows,H), W1q (E,2*inter,row_bytes)");
  }
  const int total_rows = A.shape(0), H = A.shape(1);
  if (W1q.shape(1) != 2 * inter) {
    throw std::invalid_argument("moe_grouped_gemm_swiglu_q: W1q must be (E, 2*inter, row_bytes)");
  }
  if (total_rows % 32 != 0 || H % 32 != 0 || H % block_k != 0 || inter % 32 != 0) {
    throw std::invalid_argument(
        "moe_grouped_gemm_swiglu_q: total_rows%32, H%32, H%block_k, inter%32 required");
  }
  if (A.dtype() != bfloat16 || W1q.dtype() != uint8) {
    throw std::invalid_argument("moe_grouped_gemm_swiglu_q: A must be bfloat16, W1q uint8");
  }
  if (expert_of_tile.ndim() != 1 || expert_of_tile.shape(0) != total_rows / 32) {
    throw std::invalid_argument("moe_grouped_gemm_swiglu_q: expert_of_tile must be (total_rows/32,)");
  }
  if (has_bias &&
      (bias.ndim() != 2 || bias.shape(0) != W1q.shape(0) || bias.shape(1) != 2 * inter)) {
    throw std::invalid_argument("moe_grouped_gemm_swiglu_q: bias must be (E, 2*inter)");
  }
  if (act_mode != 0 && act_mode != 1) {
    throw std::invalid_argument("moe_grouped_gemm_swiglu_q: act_mode must be 0 or 1");
  }
  auto a = contiguous(A, false, s);
  auto wq = contiguous(W1q, false, s);
  auto eot = contiguous(astype(expert_of_tile, int32, s), false, s);
  auto b = contiguous(astype(bias, bfloat16, s), false, s);
  return array({total_rows, inter}, bfloat16,
               std::make_shared<MoeGroupedGemmSwigluQ>(
                   to_stream(s), format, has_bias, inter, act_mode, alpha, limit),
               {a, wq, eot, b});
}

void MoeGroupedGemmRectQ::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MoeGroupedGemmRectQ has no CPU implementation.");
}
void MoeGroupedGemmRectQ::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& a = inputs[0];
  auto& wq = inputs[1];
  auto& eot = inputs[2];
  auto& bias = inputs[3];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_moe_grouped_gemm_rect_q(enc, out, a, wq, eot, bias, a.shape(0), K_dim_, N_out_,
                                     has_bias_ ? 1 : 0, format_);
}

void MoeGroupedGemmSwigluQ::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MoeGroupedGemmSwigluQ has no CPU implementation.");
}
void MoeGroupedGemmSwigluQ::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& a = inputs[0];
  auto& wq = inputs[1];
  auto& eot = inputs[2];
  auto& bias = inputs[3];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_moe_grouped_gemm_swiglu_q(enc, out, a, wq, eot, bias, a.shape(0), a.shape(1),
                                       inter_, has_bias_ ? 1 : 0, act_mode_, alpha_, limit_,
                                       format_);
}

void MoeGroupedGemmRect::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MoeGroupedGemmRect has no CPU implementation.");
}
void MoeGroupedGemmRect::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& a = inputs[0];
  auto& w = inputs[1];
  auto& eot = inputs[2];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_moe_grouped_gemm_rect(enc, out, a, w, eot, a.shape(0), a.shape(1), w.shape(2),
                                   type_to_name(a));
}

void MoeGroupedGemmSwiglu::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MoeGroupedGemmSwiglu has no CPU implementation.");
}
void MoeGroupedGemmSwiglu::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& a = inputs[0];
  auto& w = inputs[1];
  auto& eot = inputs[2];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_moe_grouped_gemm_swiglu(enc, out, a, w, eot, a.shape(0), a.shape(1), w.shape(2) / 2,
                                     type_to_name(a));
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
TK_MOE_NO_AUTODIFF(MoeGroupedGemmRect, "MoeGroupedGemmRect")
TK_MOE_NO_AUTODIFF(MoeGroupedGemmSwiglu, "MoeGroupedGemmSwiglu")
TK_MOE_NO_AUTODIFF(MoeGroupedGemmRectQ, "MoeGroupedGemmRectQ")
TK_MOE_NO_AUTODIFF(MoeGroupedGemmSwigluQ, "MoeGroupedGemmSwigluQ")

} // namespace mlx::core
