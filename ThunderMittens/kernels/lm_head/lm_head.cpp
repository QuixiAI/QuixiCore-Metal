// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "lm_head/lm_head.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static constexpr int LMH_TILE_V = 256;

array lm_head_sample(
    const array& h,
    const array& W,
    const array& bias,
    int mode,
    int k,
    float temperature,
    uint32_t seed,
    StreamOrDevice s /* = {} */) {
  if (h.ndim() != 2 || W.ndim() != 2 || h.shape(1) != W.shape(1)) {
    throw std::invalid_argument("lm_head_sample: h (T,K) and W (V,K) must share K");
  }
  if (!(h.dtype() == float32 || h.dtype() == float16 || h.dtype() == bfloat16)) {
    throw std::invalid_argument("lm_head_sample: h/W must be float32, float16, or bfloat16");
  }
  if (mode < 0 || mode > 2) {
    throw std::invalid_argument("lm_head_sample: mode must be 0 (argmax), 1 (categorical), 2 (topk)");
  }
  const int T = h.shape(0);
  const int V = W.shape(0);
  const int num_vtiles = (V + LMH_TILE_V - 1) / LMH_TILE_V;
  const float invtemp = 1.0f / temperature;
  const int use_bias = bias.size() > 1 ? 1 : 0;

  auto dtype = h.dtype();
  auto h_c = contiguous(astype(h, dtype, s), false, s);
  auto W_c = contiguous(astype(W, dtype, s), false, s);
  auto bias_c = use_bias ? contiguous(astype(bias, float32, s), false, s) : zeros({1}, float32, s);

  if (mode == 2) {
    if (k < 1 || k > 64) {
      throw std::invalid_argument("lm_head_sample: topk k must be in [1, 64]");
    }
    if (k > LMH_TILE_V) {
      throw std::invalid_argument("lm_head_sample: topk k must be <= TILE_V (256)");
    }
    auto parts = array::make_arrays(
        {{T, num_vtiles, k}, {T, num_vtiles, k}}, {float32, int32},
        std::make_shared<LmHeadTopkPartials>(to_stream(s), k, use_bias, LMH_TILE_V),
        {h_c, W_c, bias_c});
    return array({T}, int32,
                 std::make_shared<LmHeadTopkReduce>(to_stream(s), k, invtemp, seed),
                 {parts[0], parts[1]});
  }

  const int use_gumbel = (mode == 1) ? 1 : 0;
  auto parts = array::make_arrays(
      {{T, num_vtiles}, {T, num_vtiles}}, {float32, int32},
      std::make_shared<LmHeadArgcatPartials>(to_stream(s), use_gumbel, invtemp, seed, use_bias,
                                             LMH_TILE_V),
      {h_c, W_c, bias_c});
  return array({T}, int32, std::make_shared<LmHeadArgcatReduce>(to_stream(s)),
               {parts[0], parts[1]});
}

array lm_head_sample_q(
    const array& h,
    const array& Wq,
    const array& bias,
    int V,
    int K,
    const std::string& fmt,
    int mode,
    float temperature,
    uint32_t seed,
    StreamOrDevice s /* = {} */) {
  if (h.ndim() != 2 || h.shape(1) != K) {
    throw std::invalid_argument("lm_head_sample_q: h must be (T, K)");
  }
  if (!(h.dtype() == float32 || h.dtype() == float16 || h.dtype() == bfloat16)) {
    throw std::invalid_argument("lm_head_sample_q: h must be float32/float16/bfloat16");
  }
  if (mode != 0 && mode != 1) {
    throw std::invalid_argument("lm_head_sample_q: only argmax (0) / categorical (1) supported");
  }
  const int T = h.shape(0);
  const int num_vtiles = (V + LMH_TILE_V - 1) / LMH_TILE_V;
  const int use_bias = bias.size() > 1 ? 1 : 0;
  const int use_gumbel = (mode == 1) ? 1 : 0;
  auto h_c = contiguous(h, false, s);
  auto Wq_c = contiguous(astype(Wq, uint8, s), false, s);
  auto bias_c = use_bias ? contiguous(astype(bias, float32, s), false, s) : zeros({1}, float32, s);
  auto parts = array::make_arrays(
      {{T, num_vtiles}, {T, num_vtiles}}, {float32, int32},
      std::make_shared<LmHeadArgcatPartialsQ>(to_stream(s), use_gumbel, 1.0f / temperature, seed,
                                              use_bias, LMH_TILE_V, V, K, fmt),
      {h_c, Wq_c, bias_c});
  return array({T}, int32, std::make_shared<LmHeadArgcatReduce>(to_stream(s)),
               {parts[0], parts[1]});
}

void LmHeadArgcatPartialsQ::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadArgcatPartialsQ has no CPU implementation.");
}
void LmHeadArgcatPartialsQ::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& h = inputs[0];
  auto& Wq = inputs[1];
  auto& bias = inputs[2];
  auto& part_val = outputs[0];
  auto& part_id = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  part_val.set_data(allocator::malloc_or_wait(part_val.nbytes()));
  part_id.set_data(allocator::malloc_or_wait(part_id.nbytes()));
  const int T = h.shape(0);
  const int num_vtiles = part_val.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lm_head_argcat_partials_q(enc, h, Wq, part_val, part_id, bias, V_, K_, tile_v_,
                                       num_vtiles, invtemp_, seed_, use_gumbel_, use_bias_, T, fmt_,
                                       type_to_name(h));
}

// ---------------- argcat partials ----------------
void LmHeadArgcatPartials::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadArgcatPartials has no CPU implementation.");
}
void LmHeadArgcatPartials::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& h = inputs[0];
  auto& W = inputs[1];
  auto& bias = inputs[2];
  auto& part_val = outputs[0];
  auto& part_id = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  part_val.set_data(allocator::malloc_or_wait(part_val.nbytes()));
  part_id.set_data(allocator::malloc_or_wait(part_id.nbytes()));

  const int T = h.shape(0);
  const int K = h.shape(1);
  const int V = W.shape(0);
  const int num_vtiles = part_val.shape(1);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lm_head_argcat_partials(
      enc, h, W, part_val, part_id, bias, V, K, tile_v_, num_vtiles, invtemp_, seed_,
      use_gumbel_, use_bias_, T, type_to_name(h));
}

// ---------------- argcat reduce ----------------
void LmHeadArgcatReduce::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadArgcatReduce has no CPU implementation.");
}
void LmHeadArgcatReduce::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& part_val = inputs[0];
  auto& part_id = inputs[1];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int T = part_val.shape(0);
  const int num_vtiles = part_val.shape(1);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lm_head_argcat_reduce(enc, part_val, part_id, out, num_vtiles, T);
}

// ---------------- topk partials ----------------
void LmHeadTopkPartials::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadTopkPartials has no CPU implementation.");
}
void LmHeadTopkPartials::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& h = inputs[0];
  auto& W = inputs[1];
  auto& bias = inputs[2];
  auto& part_val = outputs[0];
  auto& part_id = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  part_val.set_data(allocator::malloc_or_wait(part_val.nbytes()));
  part_id.set_data(allocator::malloc_or_wait(part_id.nbytes()));

  const int T = h.shape(0);
  const int K = h.shape(1);
  const int V = W.shape(0);
  const int num_vtiles = part_val.shape(1);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lm_head_topk_partials(
      enc, h, W, part_val, part_id, bias, V, K, tile_v_, num_vtiles, topk_, use_bias_, T,
      type_to_name(h));
}

// ---------------- topk reduce ----------------
void LmHeadTopkReduce::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadTopkReduce has no CPU implementation.");
}
void LmHeadTopkReduce::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& part_val = inputs[0];
  auto& part_id = inputs[1];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int T = part_val.shape(0);
  const int num_vtiles = part_val.shape(1);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lm_head_topk_reduce(enc, part_val, part_id, out, num_vtiles, topk_, seed_, invtemp_, T);
}

#define TK_LMH_NO_AUTODIFF(CLASS, LABEL)                                     \
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

TK_LMH_NO_AUTODIFF(LmHeadArgcatPartials, "LmHeadArgcatPartials")
TK_LMH_NO_AUTODIFF(LmHeadArgcatPartialsQ, "LmHeadArgcatPartialsQ")
TK_LMH_NO_AUTODIFF(LmHeadArgcatReduce, "LmHeadArgcatReduce")
TK_LMH_NO_AUTODIFF(LmHeadTopkPartials, "LmHeadTopkPartials")
TK_LMH_NO_AUTODIFF(LmHeadTopkReduce, "LmHeadTopkReduce")

} // namespace mlx::core
