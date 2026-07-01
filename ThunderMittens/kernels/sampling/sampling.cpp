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
    const array& logits, const array& prev_tokens, const array& bias, float temperature /* = 1.0f */,
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
  auto x = contiguous(logits, false, s);
  auto prev = contiguous(astype(prev_tokens, int32, s), false, s);
  auto bias_c = contiguous(astype(bias, float32, s), false, s);
  return array::make_arrays(
      {{T, V}, {T, V}},
      {logits.dtype(), int32},
      std::make_shared<ApplyPenalty>(
          to_stream(s), 1.0f / temperature, repetition_penalty, presence_penalty, frequency_penalty,
          eos_id, min_length, gen_len),
      {x, prev, bias_c});
}

void ApplyPenalty::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("ApplyPenalty has no CPU implementation.");
}

void ApplyPenalty::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& logits = inputs[0];
  auto& prev = inputs[1];
  auto& bias = inputs[2];
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
  tk::launch_penalty_histogram(enc, prev, counts, V, L, T * L);
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

} // namespace mlx::core
