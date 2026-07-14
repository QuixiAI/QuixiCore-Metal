// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "lm_head/lm_head.h"
#include "sampling/sampling.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static constexpr int LMH_TILE_V = 256;
static constexpr int LMH_MASKED_TILE_V_SMALL = 128;
static constexpr int LMH_MASKED_TILE_V_BATCHED = 256;

namespace {
bool lmh_float(Dtype dtype) {
  return dtype == float32 || dtype == float16 || dtype == bfloat16;
}

int check_masked_weight(
    const array& h, const array& W, const std::string& format,
    const char* name) {
  if (h.ndim() != 2 || h.shape(0) <= 0 || h.shape(1) <= 0 ||
      !lmh_float(h.dtype())) {
    throw std::invalid_argument(std::string(name) +
        ": h must be non-empty (T,K) fp32/fp16/bf16");
  }
  const int K = h.shape(1);
  if (format.empty()) {
    if (W.ndim() != 2 || W.shape(0) <= 0 || W.shape(1) != K ||
        W.dtype() != h.dtype()) {
      throw std::invalid_argument(std::string(name) +
          ": dense W must be (V,K) with h dtype");
    }
    return W.shape(0);
  }
  int block_k = 0, block_bytes = 0;
  if (format == "q4_0") { block_k = 32; block_bytes = 18; }
  else if (format == "q8_0") { block_k = 32; block_bytes = 34; }
  else if (format == "q6_K") { block_k = 256; block_bytes = 210; }
  else if (format == "nvfp4") { block_k = 16; block_bytes = 9; }
  else {
    throw std::invalid_argument(std::string(name) +
        ": format must be empty/dense, q4_0, q8_0, q6_K, or nvfp4");
  }
  if (W.dtype() != uint8 || W.ndim() != 3 || W.shape(0) <= 0 ||
      K % block_k != 0 || W.shape(1) != K / block_k ||
      W.shape(2) != block_bytes) {
    throw std::invalid_argument(std::string(name) +
        ": packed W must be uint8 (V,K/block_k,block_bytes)");
  }
  return W.shape(0);
}

array lmh_bias(
    const array& bias, int vocab, bool use_bias, StreamOrDevice s,
    const char* name) {
  if (use_bias && (bias.ndim() != 1 || bias.shape(0) != vocab)) {
    throw std::invalid_argument(std::string(name) +
        ": bias must be (V,) or a scalar placeholder");
  }
  return use_bias ? contiguous(astype(bias, float32, s), false, s)
                  : zeros({1}, float32, s);
}
}

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
    if (k > V) {
      throw std::invalid_argument("lm_head_sample: topk k must be <= vocab size V");
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
    int k,
    float temperature,
    uint32_t seed,
    float top_p /* = 0.0f */,
    StreamOrDevice s /* = {} */) {
  if (h.ndim() != 2 || h.shape(1) != K) {
    throw std::invalid_argument("lm_head_sample_q: h must be (T, K)");
  }
  if (!(h.dtype() == float32 || h.dtype() == float16 || h.dtype() == bfloat16)) {
    throw std::invalid_argument("lm_head_sample_q: h must be float32/float16/bfloat16");
  }
  if (mode < 0 || mode > 3) {
    throw std::invalid_argument(
        "lm_head_sample_q: mode must be 0 (argmax), 1 (categorical), 2 (topk), 3 (topp)");
  }
  if (!(temperature > 0.0f)) {
    throw std::invalid_argument("lm_head_sample_q: temperature must be positive");
  }
  int block_k = 0;
  int block_bytes = 0;
  if (fmt == "q8_0") {
    block_k = 32; block_bytes = 34;
  } else if (fmt == "q4_0") {
    block_k = 32; block_bytes = 18;
  } else if (fmt == "nvfp4") {
    block_k = 16; block_bytes = 9;
  } else if (fmt == "q6_K") {
    block_k = 256; block_bytes = 210;
    if (h.dtype() != float32 || mode > 1) {
      throw std::invalid_argument(
          "lm_head_sample_q: q6_K requires fp32 h and supports argmax/categorical only");
    }
  } else {
    throw std::invalid_argument(
        "lm_head_sample_q: format must be q8_0, q4_0, q6_K, or nvfp4");
  }
  if (K % block_k != 0 || Wq.dtype() != uint8 || Wq.ndim() != 3 ||
      Wq.shape(0) != V || Wq.shape(1) != K / block_k || Wq.shape(2) != block_bytes) {
    throw std::invalid_argument("lm_head_sample_q: packed weight shape does not match V/K/format");
  }
  const int T = h.shape(0);
  if (T <= 0 || V <= 0 || K <= 0) {
    throw std::invalid_argument("lm_head_sample_q: T, V, and K must be positive");
  }
  const int num_vtiles = (V + LMH_TILE_V - 1) / LMH_TILE_V;
  const int use_bias = bias.size() > 1 ? 1 : 0;
  auto h_c = contiguous(h, false, s);
  auto Wq_c = contiguous(astype(Wq, uint8, s), false, s);
  auto bias_c = use_bias ? contiguous(astype(bias, float32, s), false, s) : zeros({1}, float32, s);

  if (mode == 2 || mode == 3) {
    if (k < 1 || k > 64 || k > LMH_TILE_V) {
      throw std::invalid_argument("lm_head_sample_q: k (topk / topp candidate cap) in [1, min(64, TILE_V)]");
    }
    if (k > V) {
      throw std::invalid_argument("lm_head_sample_q: k must be <= vocab size V");
    }
    if (mode == 3 && !(top_p > 0.0f && top_p <= 1.0f)) {
      throw std::invalid_argument("lm_head_sample_q: topp requires top_p in (0, 1]");
    }
    if (mode == 2) {
      auto parts = array::make_arrays(
          {{T, num_vtiles, k}, {T, num_vtiles, k}}, {float32, int32},
          std::make_shared<LmHeadTopkPartialsQ>(to_stream(s), k, use_bias, LMH_TILE_V, V, K, fmt),
          {h_c, Wq_c, bias_c});
      return array({T}, int32,
                   std::make_shared<LmHeadTopkReduce>(to_stream(s), k, 1.0f / temperature, seed),
                   {parts[0], parts[1]});
    }
    // top-p: the dedicated partials also emit the per-tile logsumexp for the true full-vocab Z
    const float invtemp = 1.0f / temperature;
    auto parts = array::make_arrays(
        {{T, num_vtiles, k}, {T, num_vtiles, k}, {T, num_vtiles}}, {float32, int32, float32},
        std::make_shared<LmHeadToppPartialsQ>(to_stream(s), k, use_bias, LMH_TILE_V, V, K, invtemp,
                                              fmt),
        {h_c, Wq_c, bias_c});
    return array({T}, int32,
                 std::make_shared<LmHeadToppReduce>(to_stream(s), k, top_p, invtemp, seed),
                 {parts[0], parts[1], parts[2]});
  }

  const int use_gumbel = (mode == 1) ? 1 : 0;
  auto parts = array::make_arrays(
      {{T, num_vtiles}, {T, num_vtiles}}, {float32, int32},
      std::make_shared<LmHeadArgcatPartialsQ>(to_stream(s), use_gumbel, 1.0f / temperature, seed,
                                              use_bias, LMH_TILE_V, V, K, fmt),
      {h_c, Wq_c, bias_c});
  return array({T}, int32, std::make_shared<LmHeadArgcatReduce>(to_stream(s)),
               {parts[0], parts[1]});
}

std::vector<array> lm_head_beam_advance(
    const array& h,
    const array& Wq,
    const array& bias,
    const array& cum_log_probs,
    int beam_width,
    const std::string& format,
    StreamOrDevice s) {
  if (h.ndim() != 2 || h.shape(0) <= 0 || h.shape(1) <= 0 ||
      !lmh_float(h.dtype())) {
    throw std::invalid_argument(
        "lm_head_beam_advance: h must be non-empty (B*BM,K) fp32/fp16/bf16");
  }
  int block_k = 0, block_bytes = 0;
  if (format == "q4_0") { block_k = 32; block_bytes = 18; }
  else if (format == "q8_0") { block_k = 32; block_bytes = 34; }
  else if (format == "nvfp4") { block_k = 16; block_bytes = 9; }
  else {
    throw std::invalid_argument(
        "lm_head_beam_advance: format must be q4_0, q8_0, or nvfp4");
  }
  const int rows = h.shape(0);
  const int hidden = h.shape(1);
  if (hidden % block_k != 0 || Wq.dtype() != uint8 || Wq.ndim() != 3 ||
      Wq.shape(0) <= 0 || Wq.shape(1) != hidden / block_k ||
      Wq.shape(2) != block_bytes) {
    throw std::invalid_argument(
        "lm_head_beam_advance: packed W has invalid shape for its format");
  }
  if (beam_width < 1 || beam_width > 16) {
    throw std::invalid_argument(
        "lm_head_beam_advance: beam_width must be in [1,16]");
  }
  if (cum_log_probs.ndim() != 2 ||
      cum_log_probs.shape(1) != beam_width ||
      rows != cum_log_probs.shape(0) * beam_width) {
    throw std::invalid_argument(
        "lm_head_beam_advance: cum_log_probs must be (B,BM) and h rows B*BM");
  }
  const int vocab = Wq.shape(0);
  const int two_bm = 2 * beam_width;
  if (vocab < two_bm) {
    throw std::invalid_argument(
        "lm_head_beam_advance: vocab size must be >= 2*beam_width");
  }
  const bool use_bias = bias.size() > 1;
  auto bias_c = lmh_bias(bias, vocab, use_bias, s, "lm_head_beam_advance");
  auto h_c = contiguous(h, false, s);
  auto Wq_c = contiguous(astype(Wq, uint8, s), false, s);
  auto cum_c = contiguous(
      astype(reshape(cum_log_probs, {rows}, s), float32, s), false, s);
  const int num_vtiles = (vocab + LMH_TILE_V - 1) / LMH_TILE_V;

  // The quantized tile pass emits both top-2BM candidates and the exact tile
  // logsumexp. The first reduce merges those into per-beam cumulative scores;
  // the established BeamSelect then finds the global BM children per batch.
  auto partials = array::make_arrays(
      {{rows, num_vtiles, two_bm}, {rows, num_vtiles, two_bm},
       {rows, num_vtiles}},
      {float32, int32, float32},
      std::make_shared<LmHeadToppPartialsQ>(
          to_stream(s), two_bm, use_bias, LMH_TILE_V, vocab, hidden, 1.0f,
          format, 4),
      {h_c, Wq_c, bias_c});
  auto candidates = array::make_arrays(
      {{rows, two_bm}, {rows, two_bm}}, {float32, int32},
      std::make_shared<LmHeadBeamReduce>(to_stream(s), two_bm),
      {partials[0], partials[1], partials[2], cum_c});
  const int batches = cum_log_probs.shape(0);
  return array::make_arrays(
      {{batches, beam_width}, {batches, beam_width},
       {batches, beam_width}},
      {int32, int32, float32},
      std::make_shared<BeamSelect>(to_stream(s), beam_width, two_bm),
      {candidates[0], candidates[1]});
}

std::vector<array> lm_head_constrained(
    const array& h,
    const array& W,
    const array& bias,
    const array& forbidden,
    const array& previous,
    int eos_id,
    bool forbid_eos,
    StreamOrDevice s) {
  if (h.ndim() != 2 || W.ndim() != 2 || h.shape(1) != W.shape(1) ||
      h.dtype() != W.dtype() ||
      !(h.dtype() == float32 || h.dtype() == float16 || h.dtype() == bfloat16)) {
    throw std::invalid_argument(
        "lm_head_constrained: h (T,K) and W (V,K) must share a floating dtype");
  }
  const int tokens = h.shape(0), vocab = W.shape(0), hidden = h.shape(1);
  if (tokens <= 0 || vocab <= 0 || hidden <= 0) {
    throw std::invalid_argument("lm_head_constrained: T, V, and K must be positive");
  }
  if (forbidden.dtype() != uint8 || forbidden.ndim() != 2 ||
      forbidden.shape(0) != vocab || forbidden.shape(1) != vocab) {
    throw std::invalid_argument("lm_head_constrained: forbidden must be uint8 (V,V)");
  }
  if (previous.size() != tokens) {
    throw std::invalid_argument("lm_head_constrained: previous must contain one id per h row");
  }
  if (forbid_eos && (eos_id < 0 || eos_id >= vocab)) {
    throw std::invalid_argument("lm_head_constrained: eos_id must be in range when forbidden");
  }
  const int use_bias = bias.size() > 1 ? 1 : 0;
  if (use_bias && (bias.ndim() != 1 || bias.shape(0) != vocab)) {
    throw std::invalid_argument("lm_head_constrained: bias must be (V,) or a scalar placeholder");
  }
  const int num_vtiles = (vocab + LMH_TILE_V - 1) / LMH_TILE_V;
  auto bias_c = use_bias ? contiguous(astype(bias, float32, s), false, s)
                         : zeros({1}, float32, s);
  auto partials = array::make_arrays(
      {{tokens, num_vtiles}, {tokens, num_vtiles}, {tokens, num_vtiles},
       {tokens, num_vtiles}},
      {float32, float32, float32, int32},
      std::make_shared<LmHeadConstrainedPartials>(
          to_stream(s), vocab, hidden, use_bias, eos_id, forbid_eos),
      {contiguous(h, false, s), contiguous(W, false, s), bias_c,
       contiguous(forbidden, false, s), contiguous(astype(previous, int32, s), false, s)});
  return array::make_arrays(
      {{tokens}, {tokens}}, {int32, float32},
      std::make_shared<LmHeadConstrainedReduce>(to_stream(s)), partials);
}

std::vector<array> lm_head_masked(
    const array& h, const array& W, const array& bias,
    const array& allow_mask, const std::string& format, int topk,
    bool normalize_allowed, StreamOrDevice s) {
  const int vocab = check_masked_weight(h, W, format, "lm_head_masked");
  const int tokens = h.shape(0), hidden = h.shape(1);
  if (topk < 1 || topk > 8 || topk > vocab) {
    throw std::invalid_argument("lm_head_masked: topk must be in [1, min(8,V)]");
  }
  const int mask_words = (vocab + 31) / 32;
  if (allow_mask.ndim() != 2 || allow_mask.shape(0) != tokens ||
      allow_mask.shape(1) != mask_words) {
    throw std::invalid_argument(
        "lm_head_masked: allow_mask must be (T,ceil(V/32)) packed words");
  }
  const bool use_bias = bias.size() > 1;
  auto bias_c = lmh_bias(bias, vocab, use_bias, s, "lm_head_masked");
  const int tile_v = tokens >= 4
      ? LMH_MASKED_TILE_V_BATCHED
      : LMH_MASKED_TILE_V_SMALL;
  const int num_vtiles = (vocab + tile_v - 1) / tile_v;
  auto partials = array::make_arrays(
      {{tokens, num_vtiles, topk}, {tokens, num_vtiles, topk},
       {tokens, num_vtiles}, {tokens, num_vtiles}},
      {float32, int32, float32, float32},
      std::make_shared<LmHeadMaskedPartials>(
          to_stream(s), format, vocab, hidden, tile_v, topk, use_bias,
          normalize_allowed),
      {contiguous(h, false, s), contiguous(W, false, s), bias_c,
       contiguous(astype(allow_mask, uint32, s), false, s)});
  return array::make_arrays(
      {{tokens, topk}, {tokens, topk}}, {int32, float32},
      std::make_shared<LmHeadMaskedReduce>(to_stream(s), topk), partials);
}

std::vector<array> lm_head_candidates(
    const array& h, const array& W, const array& bias,
    const array& candidate_ids, const array& offsets,
    const std::string& format, int topk, StreamOrDevice s) {
  const int vocab = check_masked_weight(h, W, format, "lm_head_candidates");
  const int tokens = h.shape(0), hidden = h.shape(1);
  if (topk < 1 || topk > 8 || topk > vocab) {
    throw std::invalid_argument("lm_head_candidates: topk must be in [1, min(8,V)]");
  }
  if (candidate_ids.ndim() != 1 || offsets.ndim() != 1 ||
      offsets.shape(0) != tokens + 1) {
    throw std::invalid_argument(
        "lm_head_candidates: candidate_ids must be flat and offsets must have T+1 entries");
  }
  const bool use_bias = bias.size() > 1;
  auto bias_c = lmh_bias(bias, vocab, use_bias, s, "lm_head_candidates");
  return array::make_arrays(
      {{tokens, topk}, {tokens, topk}}, {int32, float32},
      std::make_shared<LmHeadCandidates>(
          to_stream(s), format, vocab, hidden, topk, use_bias),
      {contiguous(h, false, s), contiguous(W, false, s),
       contiguous(astype(candidate_ids, int32, s), false, s),
       contiguous(astype(offsets, int32, s), false, s), bias_c});
}

void LmHeadMaskedPartials::eval_cpu(
    const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadMaskedPartials has no CPU implementation.");
}
void LmHeadMaskedPartials::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  for (auto& output : outputs) {
    output.set_data(allocator::malloc_or_wait(output.nbytes()));
  }
  auto& s = stream(); auto& d = metal::device(s.device);
  auto& ce = d.get_command_encoder(s.index); MLXEncoder enc(d, ce);
  const int tokens = inputs[0].shape(0);
  const int num_vtiles = outputs[2].shape(1);
  tk::launch_lm_head_masked_partials(
      enc, inputs[0], inputs[1], inputs[2], inputs[3],
      outputs[0], outputs[1], outputs[2], outputs[3],
      vocab_, hidden_, tile_v_, num_vtiles, topk_, use_bias_,
      normalize_allowed_, inputs[3].shape(1), tokens, format_,
      type_to_name(inputs[0]));
}
bool LmHeadMaskedPartials::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const LmHeadMaskedPartials&>(other);
  return format_ == o.format_ && vocab_ == o.vocab_ && hidden_ == o.hidden_ &&
      tile_v_ == o.tile_v_ && topk_ == o.topk_ && use_bias_ == o.use_bias_ &&
      normalize_allowed_ == o.normalize_allowed_;
}

void LmHeadMaskedReduce::eval_cpu(
    const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadMaskedReduce has no CPU implementation.");
}
void LmHeadMaskedReduce::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  for (auto& output : outputs) {
    output.set_data(allocator::malloc_or_wait(output.nbytes()));
  }
  auto& s = stream(); auto& d = metal::device(s.device);
  auto& ce = d.get_command_encoder(s.index); MLXEncoder enc(d, ce);
  tk::launch_lm_head_masked_reduce(
      enc, inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], outputs[1],
      inputs[2].shape(1), topk_, inputs[2].shape(0));
}
bool LmHeadMaskedReduce::is_equivalent(const Primitive& other) const {
  return typeid(*this) == typeid(other) &&
      topk_ == static_cast<const LmHeadMaskedReduce&>(other).topk_;
}

void LmHeadCandidates::eval_cpu(
    const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadCandidates has no CPU implementation.");
}
void LmHeadCandidates::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  for (auto& output : outputs) {
    output.set_data(allocator::malloc_or_wait(output.nbytes()));
  }
  auto& s = stream(); auto& d = metal::device(s.device);
  auto& ce = d.get_command_encoder(s.index); MLXEncoder enc(d, ce);
  tk::launch_lm_head_candidates(
      enc, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
      outputs[0], outputs[1], vocab_, hidden_, topk_, use_bias_,
      inputs[0].shape(0), format_, type_to_name(inputs[0]));
}
bool LmHeadCandidates::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const LmHeadCandidates&>(other);
  return format_ == o.format_ && vocab_ == o.vocab_ && hidden_ == o.hidden_ &&
      topk_ == o.topk_ && use_bias_ == o.use_bias_;
}

void LmHeadConstrainedPartials::eval_cpu(
    const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadConstrainedPartials has no CPU implementation.");
}
void LmHeadConstrainedPartials::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  for (auto& output : outputs) {
    output.set_data(allocator::malloc_or_wait(output.nbytes()));
  }
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  const int tokens = inputs[0].shape(0);
  const int num_vtiles = outputs[0].shape(1);
  tk::launch_lm_head_constrained_partials(
      enc, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], outputs[0],
      outputs[1], outputs[2], outputs[3], vocab_, hidden_, LMH_TILE_V,
      num_vtiles, use_bias_, eos_id_, forbid_eos_ ? 1 : 0, tokens,
      type_to_name(inputs[0]));
}
bool LmHeadConstrainedPartials::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const LmHeadConstrainedPartials&>(other);
  return vocab_ == o.vocab_ && hidden_ == o.hidden_ && use_bias_ == o.use_bias_ &&
      eos_id_ == o.eos_id_ && forbid_eos_ == o.forbid_eos_;
}

void LmHeadConstrainedReduce::eval_cpu(
    const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadConstrainedReduce has no CPU implementation.");
}
void LmHeadConstrainedReduce::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  outputs[0].set_data(allocator::malloc_or_wait(outputs[0].nbytes()));
  outputs[1].set_data(allocator::malloc_or_wait(outputs[1].nbytes()));
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lm_head_constrained_reduce(
      enc, inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], outputs[1],
      inputs[0].shape(1), inputs[0].shape(0));
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

void LmHeadTopkPartialsQ::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadTopkPartialsQ has no CPU implementation.");
}
void LmHeadTopkPartialsQ::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
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
  tk::launch_lm_head_topk_partials_q(enc, h, Wq, part_val, part_id, bias, V_, K_, tile_v_,
                                     num_vtiles, topk_, use_bias_, T, fmt_, type_to_name(h));
}

// ---------------- topp partials (quant) ----------------
void LmHeadToppPartialsQ::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadToppPartialsQ has no CPU implementation.");
}
void LmHeadToppPartialsQ::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& h = inputs[0];
  auto& Wq = inputs[1];
  auto& bias = inputs[2];
  auto& part_val = outputs[0];
  auto& part_id = outputs[1];
  auto& part_lse = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  part_val.set_data(allocator::malloc_or_wait(part_val.nbytes()));
  part_id.set_data(allocator::malloc_or_wait(part_id.nbytes()));
  part_lse.set_data(allocator::malloc_or_wait(part_lse.nbytes()));
  const int T = h.shape(0);
  const int num_vtiles = part_val.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lm_head_topp_partials_q(enc, h, Wq, part_val, part_id, bias, V_, K_, tile_v_,
                                     num_vtiles, topk_, use_bias_, invtemp_, part_lse, T, fmt_,
                                     type_to_name(h), rows_per_tg_);
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

// ---------------- topp reduce ----------------
void LmHeadToppReduce::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadToppReduce has no CPU implementation.");
}
void LmHeadToppReduce::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& part_val = inputs[0];
  auto& part_id = inputs[1];
  auto& part_lse = inputs[2];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int T = part_val.shape(0);
  const int num_vtiles = part_val.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lm_head_topp_reduce(enc, part_val, part_id, out, num_vtiles, topk_, p_, seed_,
                                 invtemp_, part_lse, T);
}

// ---------------- quantized LM-head beam reduce ----------------
void LmHeadBeamReduce::eval_cpu(
    const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LmHeadBeamReduce has no CPU implementation.");
}
void LmHeadBeamReduce::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  for (auto& output : outputs) {
    output.set_data(allocator::malloc_or_wait(output.nbytes()));
  }
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lm_head_beam_reduce(
      enc, inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], outputs[1],
      inputs[2].shape(1), two_bm_, inputs[2].shape(0));
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
TK_LMH_NO_AUTODIFF(LmHeadTopkPartialsQ, "LmHeadTopkPartialsQ")
TK_LMH_NO_AUTODIFF(LmHeadToppPartialsQ, "LmHeadToppPartialsQ")
TK_LMH_NO_AUTODIFF(LmHeadTopkReduce, "LmHeadTopkReduce")
TK_LMH_NO_AUTODIFF(LmHeadToppReduce, "LmHeadToppReduce")
TK_LMH_NO_AUTODIFF(LmHeadBeamReduce, "LmHeadBeamReduce")
TK_LMH_NO_AUTODIFF(LmHeadConstrainedPartials, "LmHeadConstrainedPartials")
TK_LMH_NO_AUTODIFF(LmHeadConstrainedReduce, "LmHeadConstrainedReduce")
TK_LMH_NO_AUTODIFF(LmHeadMaskedPartials, "LmHeadMaskedPartials")
TK_LMH_NO_AUTODIFF(LmHeadMaskedReduce, "LmHeadMaskedReduce")
TK_LMH_NO_AUTODIFF(LmHeadCandidates, "LmHeadCandidates")

} // namespace mlx::core
