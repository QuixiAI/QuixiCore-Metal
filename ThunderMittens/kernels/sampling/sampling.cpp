// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "sampling/sampling.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array argmax_sample(const array& logits, StreamOrDevice s /* = {} */) {
  if (logits.ndim() < 1) {
    throw std::invalid_argument("argmax_sample: logits must have at least 1 dimension");
  }
  if (!(logits.dtype() == float32 || logits.dtype() == float16 || logits.dtype() == bfloat16)) {
    throw std::invalid_argument("argmax_sample: logits must be float32, float16, or bfloat16");
  }
  auto x = contiguous(logits, false, s);
  std::vector<int> out_shape(logits.shape().begin(), logits.shape().end() - 1);
  if (out_shape.empty()) {
    out_shape.push_back(1);
  }
  return array(out_shape, int32, std::make_shared<ArgmaxSample>(to_stream(s)), {x});
}

void ArgmaxSample::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("ArgmaxSample has no CPU implementation.");
}

void ArgmaxSample::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int V = logits.shape(-1);
  const int rows = static_cast<int>(logits.size() / V);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_argmax(enc, logits, out, rows, V, type_to_name(logits));
}

std::vector<array> ArgmaxSample::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("ArgmaxSample has no jvp implementation.");
}
std::vector<array> ArgmaxSample::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("ArgmaxSample has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> ArgmaxSample::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("ArgmaxSample has no vmap implementation.");
}

array sample_categorical(
    const array& logits, float temperature /* = 1.0f */, uint32_t seed /* = 0 */,
    StreamOrDevice s /* = {} */) {
  if (logits.ndim() < 1) {
    throw std::invalid_argument("sample_categorical: logits must have at least 1 dimension");
  }
  if (!(logits.dtype() == float32 || logits.dtype() == float16 || logits.dtype() == bfloat16)) {
    throw std::invalid_argument("sample_categorical: logits must be float32, float16, or bfloat16");
  }
  if (temperature <= 0.0f) {
    throw std::invalid_argument("sample_categorical: temperature must be > 0");
  }
  auto x = contiguous(logits, false, s);
  std::vector<int> out_shape(logits.shape().begin(), logits.shape().end() - 1);
  if (out_shape.empty()) {
    out_shape.push_back(1);
  }
  return array(
      out_shape, int32,
      std::make_shared<SampleCategorical>(to_stream(s), 1.0f / temperature, seed),
      {x});
}

void SampleCategorical::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("SampleCategorical has no CPU implementation.");
}

void SampleCategorical::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int V = logits.shape(-1);
  const int rows = static_cast<int>(logits.size() / V);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_sample_categorical(enc, logits, out, rows, V, seed_, invtemp_, type_to_name(logits));
}

std::vector<array> SampleCategorical::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SampleCategorical has no jvp implementation.");
}
std::vector<array> SampleCategorical::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("SampleCategorical has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> SampleCategorical::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SampleCategorical has no vmap implementation.");
}

array top_k_sample(
    const array& logits, int k, float temperature /* = 1.0f */, uint32_t seed /* = 0 */,
    StreamOrDevice s /* = {} */) {
  if (logits.ndim() < 1) {
    throw std::invalid_argument("top_k_sample: logits must have at least 1 dimension");
  }
  if (!(logits.dtype() == float32 || logits.dtype() == float16 || logits.dtype() == bfloat16)) {
    throw std::invalid_argument("top_k_sample: logits must be float32, float16, or bfloat16");
  }
  if (temperature <= 0.0f) {
    throw std::invalid_argument("top_k_sample: temperature must be > 0");
  }
  const int V = logits.shape(-1);
  if (k <= 0 || k > 64 || k > V) {
    throw std::invalid_argument("top_k_sample: require 1 <= k <= min(64, vocab)");
  }
  auto x = contiguous(logits, false, s);
  std::vector<int> out_shape(logits.shape().begin(), logits.shape().end() - 1);
  if (out_shape.empty()) {
    out_shape.push_back(1);
  }
  return array(
      out_shape, int32,
      std::make_shared<TopKSample>(to_stream(s), k, 1.0f / temperature, seed),
      {x});
}

void TopKSample::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("TopKSample has no CPU implementation.");
}

void TopKSample::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int V = logits.shape(-1);
  const int rows = static_cast<int>(logits.size() / V);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_top_k_sample(enc, logits, out, rows, V, k_, seed_, invtemp_, type_to_name(logits));
}

std::vector<array> TopKSample::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("TopKSample has no jvp implementation.");
}
std::vector<array> TopKSample::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("TopKSample has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> TopKSample::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("TopKSample has no vmap implementation.");
}

array top_p_sample(
    const array& logits, float p, float temperature /* = 1.0f */, uint32_t seed /* = 0 */,
    StreamOrDevice s /* = {} */) {
  if (logits.ndim() < 1) {
    throw std::invalid_argument("top_p_sample: logits must have at least 1 dimension");
  }
  if (!(logits.dtype() == float32 || logits.dtype() == float16 || logits.dtype() == bfloat16)) {
    throw std::invalid_argument("top_p_sample: logits must be float32, float16, or bfloat16");
  }
  if (temperature <= 0.0f) {
    throw std::invalid_argument("top_p_sample: temperature must be > 0");
  }
  if (!(p > 0.0f && p <= 1.0f)) {
    throw std::invalid_argument("top_p_sample: p must be in (0, 1]");
  }
  auto x = contiguous(logits, false, s);
  std::vector<int> out_shape(logits.shape().begin(), logits.shape().end() - 1);
  if (out_shape.empty()) {
    out_shape.push_back(1);
  }
  return array(
      out_shape, int32,
      std::make_shared<TopPSample>(to_stream(s), p, 1.0f / temperature, seed),
      {x});
}

void TopPSample::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("TopPSample has no CPU implementation.");
}

void TopPSample::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int V = logits.shape(-1);
  const int rows = static_cast<int>(logits.size() / V);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_top_p_sample(enc, logits, out, rows, V, p_, seed_, invtemp_, type_to_name(logits));
}

std::vector<array> TopPSample::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("TopPSample has no jvp implementation.");
}
std::vector<array> TopPSample::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("TopPSample has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> TopPSample::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("TopPSample has no vmap implementation.");
}

std::vector<array> apply_penalty(
    const array& logits, const array& prev_tokens, const array& bias, const array& parent_ids,
    float temperature /* = 1.0f */,
    float repetition_penalty /* = 1.0f */, float presence_penalty /* = 0.0f */,
    float frequency_penalty /* = 0.0f */, int eos_id /* = -1 */, int min_length /* = 0 */,
    int gen_len /* = 0 */, StreamOrDevice s /* = {} */) {
  if (logits.ndim() != 2) {
    throw std::invalid_argument("apply_penalty: logits must have shape (num_tokens, vocab)");
  }
  if (!(logits.dtype() == float32 || logits.dtype() == float16 || logits.dtype() == bfloat16)) {
    throw std::invalid_argument("apply_penalty: logits must be float32, float16, or bfloat16");
  }
  if (prev_tokens.ndim() != 2 || prev_tokens.shape(0) != logits.shape(0)) {
    throw std::invalid_argument("apply_penalty: prev_tokens must have shape (num_tokens, history_len)");
  }
  if (temperature <= 0.0f) {
    throw std::invalid_argument("apply_penalty: temperature must be > 0");
  }
  const int T = logits.shape(0);
  const int V = logits.shape(1);
  if (bias.ndim() != 1 || bias.shape(0) != V) {
    throw std::invalid_argument("apply_penalty: bias must have shape (vocab,)");
  }
  if (parent_ids.ndim() != 1 || parent_ids.shape(0) != T) {
    throw std::invalid_argument("apply_penalty: parent_ids must have shape (num_tokens,)");
  }
  auto x = contiguous(logits, false, s);
  auto prev = contiguous(astype(prev_tokens, int32, s), false, s);
  auto bias_c = contiguous(astype(bias, float32, s), false, s);
  auto parent_c = contiguous(astype(parent_ids, int32, s), false, s);
  return array::make_arrays(
      {{T, V}, {T, V}},
      {logits.dtype(), int32},
      std::make_shared<ApplyPenalty>(
          to_stream(s), 1.0f / temperature, repetition_penalty, presence_penalty, frequency_penalty,
          eos_id, min_length, gen_len),
      {x, prev, bias_c, parent_c});
}

void ApplyPenalty::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("ApplyPenalty has no CPU implementation.");
}

void ApplyPenalty::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& prev = inputs[1];
  auto& bias = inputs[2];
  auto& parent_ids = inputs[3];
  auto& out = outputs[0];
  auto& counts = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  counts.set_data(allocator::malloc_or_wait(counts.nbytes()));

  const int T = logits.shape(0);
  const int V = logits.shape(1);
  const int L = prev.shape(1);

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_moe_zero_i32(enc, counts, T * V);
  tk::launch_penalty_histogram(enc, prev, counts, V, L, T * L, parent_ids);
  tk::launch_apply_penalty(
      enc, logits, counts, out, bias, T, V, invtemp_, rep_, presence_, freq_,
      eos_id_, min_length_, gen_len_, type_to_name(logits));
}

std::vector<array> ApplyPenalty::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("ApplyPenalty has no jvp implementation.");
}
std::vector<array> ApplyPenalty::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("ApplyPenalty has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> ApplyPenalty::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("ApplyPenalty has no vmap implementation.");
}

// ----------------------------- beam_advance -----------------------------

std::vector<array> beam_advance(
    const array& logits,
    const array& cum_log_probs,
    int beam_width,
    StreamOrDevice s /* = {} */) {
  if (logits.ndim() != 2) {
    throw std::invalid_argument("beam_advance: logits must be (B*BM, V)");
  }
  if (cum_log_probs.ndim() != 2 || cum_log_probs.shape(1) != beam_width) {
    throw std::invalid_argument("beam_advance: cum_log_probs must be (B, beam_width)");
  }
  if (beam_width < 1 || beam_width > 16) {
    throw std::invalid_argument("beam_advance: beam_width must be in [1, 16]");
  }
  const int B = cum_log_probs.shape(0);
  const int BR = logits.shape(0);
  if (BR != B * beam_width) {
    throw std::invalid_argument("beam_advance: logits rows must equal B * beam_width");
  }
  const int two_bm = 2 * beam_width;
  auto logits_c = contiguous(logits, false, s);
  auto cum_c = contiguous(astype(reshape(cum_log_probs, {BR}, s), float32, s), false, s);

  auto cands = array::make_arrays(
      {{BR, two_bm}, {BR, two_bm}}, {float32, int32},
      std::make_shared<BeamTopkPartials>(to_stream(s), two_bm),
      {logits_c, cum_c});

  return array::make_arrays(
      {{B, beam_width}, {B, beam_width}, {B, beam_width}}, {int32, int32, float32},
      std::make_shared<BeamSelect>(to_stream(s), beam_width, two_bm),
      {cands[0], cands[1]});
}

void BeamTopkPartials::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("BeamTopkPartials has no CPU implementation.");
}
void BeamTopkPartials::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& cum = inputs[1];
  auto& cand_score = outputs[0];
  auto& cand_token = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  cand_score.set_data(allocator::malloc_or_wait(cand_score.nbytes()));
  cand_token.set_data(allocator::malloc_or_wait(cand_token.nbytes()));

  const int BR = logits.shape(0);
  const int V = logits.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_beam_topk_partials(enc, logits, cum, cand_score, cand_token, BR, V, two_bm_,
                                type_to_name(logits));
}

void BeamSelect::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("BeamSelect has no CPU implementation.");
}
void BeamSelect::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& cand_score = inputs[0];
  auto& cand_token = inputs[1];
  auto& next_token = outputs[0];
  auto& parent_beam = outputs[1];
  auto& new_cum = outputs[2];

  auto& s = stream();
  auto& d = metal::device(s.device);
  next_token.set_data(allocator::malloc_or_wait(next_token.nbytes()));
  parent_beam.set_data(allocator::malloc_or_wait(parent_beam.nbytes()));
  new_cum.set_data(allocator::malloc_or_wait(new_cum.nbytes()));

  const int B = next_token.shape(0);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_beam_select(enc, cand_score, cand_token, next_token, parent_beam, new_cum, B,
                         beam_width_, two_bm_);
}

#define TK_BEAM_NO_AUTODIFF(CLASS, LABEL)                                    \
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

TK_BEAM_NO_AUTODIFF(BeamTopkPartials, "BeamTopkPartials")
TK_BEAM_NO_AUTODIFF(BeamSelect, "BeamSelect")

// ----------------------------- min_p_sample / apply_token_bitmask -----------------------------

array min_p_sample(
    const array& logits, float min_p, float temperature /* = 1.0f */, uint32_t seed /* = 0 */,
    StreamOrDevice s /* = {} */) {
  if (logits.ndim() < 1) {
    throw std::invalid_argument("min_p_sample: logits must have at least 1 dimension");
  }
  if (!(logits.dtype() == float32 || logits.dtype() == float16 || logits.dtype() == bfloat16)) {
    throw std::invalid_argument("min_p_sample: logits must be float32, float16, or bfloat16");
  }
  if (temperature <= 0.0f) {
    throw std::invalid_argument("min_p_sample: temperature must be > 0");
  }
  if (!(min_p > 0.0f && min_p <= 1.0f)) {
    throw std::invalid_argument("min_p_sample: min_p must be in (0, 1]");
  }
  auto x = contiguous(logits, false, s);
  std::vector<int> out_shape(logits.shape().begin(), logits.shape().end() - 1);
  if (out_shape.empty()) {
    out_shape.push_back(1);
  }
  return array(
      out_shape, int32,
      std::make_shared<MinPSample>(to_stream(s), min_p, 1.0f / temperature, seed),
      {x});
}

void MinPSample::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MinPSample has no CPU implementation.");
}
void MinPSample::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int V = logits.shape(-1);
  const int rows = static_cast<int>(logits.size() / V);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_min_p_sample(enc, logits, out, rows, V, min_p_, seed_, invtemp_, type_to_name(logits));
}

array typical_p_sample(
    const array& logits, float typical_p, float temperature /* = 1.0f */, uint32_t seed /* = 0 */,
    StreamOrDevice s /* = {} */) {
  if (logits.ndim() < 1) {
    throw std::invalid_argument("typical_p_sample: logits must have at least 1 dimension");
  }
  if (!(logits.dtype() == float32 || logits.dtype() == float16 || logits.dtype() == bfloat16)) {
    throw std::invalid_argument("typical_p_sample: logits must be float32, float16, or bfloat16");
  }
  if (temperature <= 0.0f) {
    throw std::invalid_argument("typical_p_sample: temperature must be > 0");
  }
  if (!(typical_p > 0.0f && typical_p <= 1.0f)) {
    throw std::invalid_argument("typical_p_sample: typical_p must be in (0, 1]");
  }
  auto x = contiguous(logits, false, s);
  std::vector<int> out_shape(logits.shape().begin(), logits.shape().end() - 1);
  if (out_shape.empty()) {
    out_shape.push_back(1);
  }
  return array(
      out_shape, int32,
      std::make_shared<TypicalPSample>(to_stream(s), typical_p, 1.0f / temperature, seed),
      {x});
}

void TypicalPSample::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("TypicalPSample has no CPU implementation.");
}
void TypicalPSample::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int V = logits.shape(-1);
  const int rows = static_cast<int>(logits.size() / V);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_typical_p_sample(enc, logits, out, rows, V, typ_p_, seed_, invtemp_,
                              type_to_name(logits));
}

array apply_token_bitmask(const array& logits, const array& bitmask, StreamOrDevice s /* = {} */) {
  if (logits.ndim() < 1) {
    throw std::invalid_argument("apply_token_bitmask: logits must have at least 1 dimension");
  }
  if (!(logits.dtype() == float32 || logits.dtype() == float16 || logits.dtype() == bfloat16)) {
    throw std::invalid_argument("apply_token_bitmask: logits must be float32, float16, or bfloat16");
  }
  const int V = logits.shape(-1);
  const int num_words = (V + 31) / 32;
  if (bitmask.shape(-1) != num_words) {
    throw std::invalid_argument("apply_token_bitmask: bitmask last dim must be ceil(V/32)");
  }
  auto x = contiguous(logits, false, s);
  // packed as int32 words on both backends; the kernel reads the raw bytes as uint (bit-exact).
  auto m = contiguous(astype(bitmask, int32, s), false, s);
  return array(
      logits.shape(), logits.dtype(),
      std::make_shared<ApplyTokenBitmask>(to_stream(s)),
      {x, m});
}

void ApplyTokenBitmask::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("ApplyTokenBitmask has no CPU implementation.");
}
void ApplyTokenBitmask::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& bitmask = inputs[1];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int V = logits.shape(-1);
  const int num_words = (V + 31) / 32;
  const int rows = static_cast<int>(logits.size() / V);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_apply_token_bitmask(enc, logits, bitmask, out, rows, V, num_words,
                                 type_to_name(logits));
}

array apply_bad_words(const array& logits, const array& bad_ids, const array& bad_lens,
                      StreamOrDevice s /* = {} */) {
  if (logits.ndim() < 1) {
    throw std::invalid_argument("apply_bad_words: logits must have at least 1 dimension");
  }
  if (!(logits.dtype() == float32 || logits.dtype() == float16 || logits.dtype() == bfloat16)) {
    throw std::invalid_argument("apply_bad_words: logits must be float32, float16, or bfloat16");
  }
  const int V = logits.shape(-1);
  const int rows = static_cast<int>(logits.size() / V);
  if (bad_ids.ndim() != 2 || bad_ids.shape(0) != rows) {
    throw std::invalid_argument("apply_bad_words: bad_ids must be (num_tokens, maxbad)");
  }
  if (bad_lens.ndim() != 1 || bad_lens.shape(0) != rows) {
    throw std::invalid_argument("apply_bad_words: bad_lens must be (num_tokens,)");
  }
  auto x = contiguous(logits, false, s);
  auto ids = contiguous(astype(bad_ids, int32, s), false, s);
  auto lens = contiguous(astype(bad_lens, int32, s), false, s);
  return array(logits.shape(), logits.dtype(),
               std::make_shared<ApplyBadWords>(to_stream(s)), {x, ids, lens});
}

void ApplyBadWords::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("ApplyBadWords has no CPU implementation.");
}
void ApplyBadWords::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& bad_ids = inputs[1];
  auto& bad_lens = inputs[2];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int V = logits.shape(-1);
  const int rows = static_cast<int>(logits.size() / V);
  const int maxbad = bad_ids.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_apply_bad_words(enc, logits, bad_ids, bad_lens, out, rows, V, maxbad,
                             type_to_name(logits));
}

TK_BEAM_NO_AUTODIFF(MinPSample, "MinPSample")
TK_BEAM_NO_AUTODIFF(TypicalPSample, "TypicalPSample")
TK_BEAM_NO_AUTODIFF(ApplyTokenBitmask, "ApplyTokenBitmask")
TK_BEAM_NO_AUTODIFF(ApplyBadWords, "ApplyBadWords")

// ----------------------------- spec_verify_linear -----------------------------

std::vector<array> spec_verify_linear(
    const array& draft_tokens,
    const array& draft_probs,
    const array& target_probs,
    const array& bonus_tokens,
    const array& accept_u,
    uint32_t seed,
    StreamOrDevice s /* = {} */) {
  if (draft_tokens.ndim() != 2) {
    throw std::invalid_argument("spec_verify_linear: draft_tokens must be (B, S)");
  }
  if (draft_probs.ndim() != 3 || target_probs.ndim() != 3) {
    throw std::invalid_argument("spec_verify_linear: draft_probs (B,S,V) / target_probs (B,S+1,V)");
  }
  const int B = draft_tokens.shape(0);
  const int S = draft_tokens.shape(1);
  const int V = draft_probs.shape(2);
  if (target_probs.shape(1) != S + 1 || target_probs.shape(2) != V || draft_probs.shape(0) != B ||
      draft_probs.shape(1) != S) {
    throw std::invalid_argument(
        "spec_verify_linear: shapes must be draft_probs (B,S,V), target_probs (B,S+1,V)");
  }
  auto dt_c = contiguous(astype(draft_tokens, int32, s), false, s);
  auto dp_c = contiguous(astype(draft_probs, float32, s), false, s);
  auto tp_c = contiguous(astype(target_probs, float32, s), false, s);
  auto bt_c = contiguous(astype(bonus_tokens, int32, s), false, s);
  auto au_c = contiguous(astype(accept_u, float32, s), false, s);
  return array::make_arrays(
      {{B, S + 1}, {B}}, {int32, int32},
      std::make_shared<SpecVerifyLinear>(to_stream(s), seed),
      {dt_c, dp_c, tp_c, bt_c, au_c});
}

void SpecVerifyLinear::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("SpecVerifyLinear has no CPU implementation.");
}
void SpecVerifyLinear::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& draft_tokens = inputs[0];
  auto& draft_probs = inputs[1];
  auto& target_probs = inputs[2];
  auto& bonus_tokens = inputs[3];
  auto& accept_u = inputs[4];
  auto& out_tokens = outputs[0];
  auto& accepted_cnt = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);
  out_tokens.set_data(allocator::malloc_or_wait(out_tokens.nbytes()));
  accepted_cnt.set_data(allocator::malloc_or_wait(accepted_cnt.nbytes()));

  const int B = draft_tokens.shape(0);
  const int S = draft_tokens.shape(1);
  const int V = draft_probs.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_spec_verify_linear(enc, draft_tokens, draft_probs, target_probs, bonus_tokens,
                                accept_u, out_tokens, accepted_cnt, B, S, V, seed_);
}
TK_BEAM_NO_AUTODIFF(SpecVerifyLinear, "SpecVerifyLinear")

std::vector<array> spec_compact(
    const array& out_tokens, const array& accepted_cnt, const array& seq_lens,
    StreamOrDevice s /* = {} */) {
  if (out_tokens.ndim() != 2) {
    throw std::invalid_argument("spec_compact: out_tokens must be (B, S+1)");
  }
  const int B = out_tokens.shape(0);
  const int Sp1 = out_tokens.shape(1);
  if (B > 256) {
    throw std::invalid_argument("spec_compact: B must be <= 256 (single-threadgroup scan)");
  }
  if (accepted_cnt.ndim() != 1 || accepted_cnt.shape(0) != B || seq_lens.shape(0) != B) {
    throw std::invalid_argument("spec_compact: accepted_cnt / seq_lens must be (B,)");
  }
  auto ot_c = contiguous(astype(out_tokens, int32, s), false, s);
  auto ac_c = contiguous(astype(accepted_cnt, int32, s), false, s);
  auto sl_c = contiguous(astype(seq_lens, int32, s), false, s);
  return array::make_arrays(
      {{B * Sp1}, {B * Sp1}, {B + 1}}, {int32, int32, int32},
      std::make_shared<SpecCompact>(to_stream(s)), {ot_c, ac_c, sl_c});
}

void SpecCompact::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("SpecCompact has no CPU implementation.");
}
void SpecCompact::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& out_tokens = inputs[0];
  auto& accepted_cnt = inputs[1];
  auto& seq_lens = inputs[2];
  auto& packed_tokens = outputs[0];
  auto& packed_pos = outputs[1];
  auto& cu_accepted = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  packed_tokens.set_data(allocator::malloc_or_wait(packed_tokens.nbytes()));
  packed_pos.set_data(allocator::malloc_or_wait(packed_pos.nbytes()));
  cu_accepted.set_data(allocator::malloc_or_wait(cu_accepted.nbytes()));
  const int B = out_tokens.shape(0);
  const int Sp1 = out_tokens.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_spec_compact(enc, out_tokens, accepted_cnt, seq_lens, packed_tokens, packed_pos,
                          cu_accepted, B, Sp1);
}
TK_BEAM_NO_AUTODIFF(SpecCompact, "SpecCompact")

array spec_update_kv_meta(const array& seq_lens, const array& accepted_cnt,
                          StreamOrDevice s /* = {} */) {
  if (seq_lens.ndim() != 1 || accepted_cnt.shape(0) != seq_lens.shape(0)) {
    throw std::invalid_argument("spec_update_kv_meta: seq_lens / accepted_cnt must be (B,)");
  }
  const int B = seq_lens.shape(0);
  auto sl_c = contiguous(astype(seq_lens, int32, s), false, s);
  auto ac_c = contiguous(astype(accepted_cnt, int32, s), false, s);
  return array({B}, int32, std::make_shared<SpecUpdateKvMeta>(to_stream(s)), {sl_c, ac_c});
}

void SpecUpdateKvMeta::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("SpecUpdateKvMeta has no CPU implementation.");
}
void SpecUpdateKvMeta::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& seq_lens = inputs[0];
  auto& accepted_cnt = inputs[1];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = seq_lens.shape(0);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_spec_update_kv_meta(enc, seq_lens, accepted_cnt, out, B);
}
TK_BEAM_NO_AUTODIFF(SpecUpdateKvMeta, "SpecUpdateKvMeta")

} // namespace mlx::core
