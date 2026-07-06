// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <cmath>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "sampling/transforms.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
void st_check(const array& x, const char* name) {
  if (x.ndim() != 2) {
    throw std::invalid_argument(std::string(name) + ": input must be (rows, V)");
  }
  if (!(x.dtype() == float32 || x.dtype() == float16 || x.dtype() == bfloat16)) {
    throw std::invalid_argument(std::string(name) + ": dtype must be f32/f16/bf16");
  }
}
float st_invtemp(float temperature) {
  return 1.0f / std::max(temperature, 1e-6f);
}
array st_make(const array& x, int kind, float f0, float f1 = 0.0f, float f2 = 0.0f,
              float f3 = 0.0f, int i0 = 0, int i1 = 0, int i2 = 0, int i3 = 0,
              uint32_t seed = 0, std::vector<array> extra = {}, StreamOrDevice s = {}) {
  std::vector<array> inputs = {contiguous(x, false, s)};
  for (auto& e : extra) inputs.push_back(e);
  return array(x.shape(), x.dtype(),
               std::make_shared<SamplerTransform>(to_stream(s), kind, f0, f1, f2, f3,
                                                  i0, i1, i2, i3, seed),
               inputs);
}
} // namespace

array quadratic_transform(const array& logits, float factor, float curve, float temperature,
                          StreamOrDevice s) {
  st_check(logits, "quadratic_transform");
  return st_make(logits, 0, factor, curve, st_invtemp(temperature), 0, 0, 0, 0, 0, 0, {}, s);
}

array top_nsigma_mask(const array& logits, float nsigma, float temperature, StreamOrDevice s) {
  st_check(logits, "top_nsigma_mask");
  return st_make(logits, 1, nsigma, st_invtemp(temperature), 0, 0, 0, 0, 0, 0, 0, {}, s);
}

array top_a_mask(const array& logits, float top_a, float temperature, StreamOrDevice s) {
  st_check(logits, "top_a_mask");
  return st_make(logits, 2, top_a, st_invtemp(temperature), 0, 0, 0, 0, 0, 0, 0, {}, s);
}

array epsilon_cutoff_mask(const array& logits, float epsilon, float temperature,
                          StreamOrDevice s) {
  st_check(logits, "epsilon_cutoff_mask");
  return st_make(logits, 3, epsilon, st_invtemp(temperature), 0, 0, 0, 0, 0, 0, 0, {}, s);
}

array eta_cutoff_mask(const array& logits, float eta, float temperature, StreamOrDevice s) {
  st_check(logits, "eta_cutoff_mask");
  return st_make(logits, 4, eta, st_invtemp(temperature), 0, 0, 0, 0, 0, 0, 0, {}, s);
}

array xtc_mask(const array& logits, float threshold, float probability, int seed,
               float temperature, StreamOrDevice s) {
  st_check(logits, "xtc_mask");
  return st_make(logits, 5, threshold, probability, st_invtemp(temperature), 0, 0, 0, 0, 0,
                 static_cast<uint32_t>(seed), {}, s);
}

array skew_transform(const array& probs, float skew, StreamOrDevice s) {
  st_check(probs, "skew_transform");
  return st_make(probs, 6, skew, 0, 0, 0, 0, 0, 0, 0, 0, {}, s);
}

array top_k_renorm(const array& probs, int k, StreamOrDevice s) {
  st_check(probs, "top_k_renorm");
  if (k <= 0 || k > 64) {
    throw std::invalid_argument("top_k_renorm: require 1 <= k <= 64");
  }
  return st_make(probs, 7, 0, 0, 0, 0, k, 0, 0, 0, 0, {}, s);
}

array top_p_renorm(const array& probs, float p, StreamOrDevice s) {
  st_check(probs, "top_p_renorm");
  return st_make(probs, 8, p, 0, 0, 0, 0, 0, 0, 0, 0, {}, s);
}

array no_repeat_ngram_mask(const array& logits, const array& prev_tokens, const array& lens,
                           int ngram_size, float temperature, StreamOrDevice s) {
  st_check(logits, "no_repeat_ngram_mask");
  if (prev_tokens.ndim() != 2 || prev_tokens.shape(0) != logits.shape(0) ||
      lens.ndim() != 1 || lens.shape(0) != logits.shape(0)) {
    throw std::invalid_argument(
        "no_repeat_ngram_mask: prev_tokens must be (rows, L), lens (rows,)");
  }
  if (ngram_size < 2) {
    throw std::invalid_argument("no_repeat_ngram_mask: ngram_size must be >= 2");
  }
  return st_make(logits, 9, st_invtemp(temperature), 0, 0, 0, ngram_size,
                 prev_tokens.shape(1), 0, 0, 0,
                 {contiguous(astype(prev_tokens, int32, s), false, s),
                  contiguous(astype(lens, int32, s), false, s)}, s);
}

array dry_penalty(const array& logits, const array& prev_tokens, const array& lens,
                  const array& breakers, float multiplier, float base, int allowed_length,
                  int range, int max_ngram, int max_occurrences, int early_exit_match_len,
                  float temperature, StreamOrDevice s) {
  st_check(logits, "dry_penalty");
  if (prev_tokens.ndim() != 2 || prev_tokens.shape(0) != logits.shape(0) ||
      lens.ndim() != 1 || lens.shape(0) != logits.shape(0) || breakers.ndim() != 1) {
    throw std::invalid_argument(
        "dry_penalty: prev_tokens (rows, L), lens (rows,), breakers (NB,)");
  }
  // pack the int params: i0 = L, i1 = NB, i2 = allowed | range<<16? No — use extra floats.
  // f0 multiplier, f1 base, f2 invtemp; i0 allowed, i1 range, i2 max_ngram,
  // i3 = max_occurrences | (early_exit << 16) would overflow; keep early_exit in seed_.
  return st_make(logits, 10, multiplier, base, st_invtemp(temperature), 0,
                 allowed_length, range, max_ngram, max_occurrences,
                 static_cast<uint32_t>(early_exit_match_len),
                 {contiguous(astype(prev_tokens, int32, s), false, s),
                  contiguous(astype(lens, int32, s), false, s),
                  contiguous(astype(breakers, int32, s), false, s)}, s);
}

void SamplerTransform::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("SamplerTransform has no CPU implementation.");
}

void SamplerTransform::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int rows = x.shape(0);
  const int V = x.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  const std::string tn = type_to_name(x);
  switch (kind_) {
    case 0:
      tk::launch_quadratic_transform(enc, x, out, rows, V, f0_, f1_, f2_, tn);
      break;
    case 1:
      tk::launch_logit_mask1(enc, "top_nsigma_mask_" + tn, x, out, rows, V, f0_, f1_);
      break;
    case 2:
      tk::launch_logit_mask1(enc, "top_a_mask_" + tn, x, out, rows, V, f0_, f1_);
      break;
    case 3:
      tk::launch_logit_mask1(enc, "epsilon_cutoff_mask_" + tn, x, out, rows, V, f0_, f1_);
      break;
    case 4:
      tk::launch_logit_mask1(enc, "eta_cutoff_mask_" + tn, x, out, rows, V, f0_, f1_);
      break;
    case 5:
      tk::launch_xtc_mask(enc, x, out, rows, V, f0_, f1_, f2_, seed_, tn);
      break;
    case 6:
      tk::launch_prob_transform1(enc, "skew_transform_" + tn, x, out, rows, V, f0_);
      break;
    case 7:
      tk::launch_top_k_renorm(enc, x, out, rows, V, i0_, tn);
      break;
    case 8:
      tk::launch_prob_transform1(enc, "top_p_renorm_probs_" + tn, x, out, rows, V, f0_);
      break;
    case 9:
      tk::launch_no_repeat_ngram_mask(enc, x, inputs[1], inputs[2], out, rows, V, i1_, i0_,
                                      f0_, tn);
      break;
    case 10:
      tk::launch_dry_penalty(enc, x, inputs[1], inputs[2], inputs[3], out, rows, V,
                             inputs[1].shape(1), inputs[3].shape(0), f0_, f1_, i0_, i1_, i2_,
                             i3_, static_cast<int>(seed_), f2_, tn);
      break;
    default:
      throw std::runtime_error("SamplerTransform: unknown kind");
  }
}

std::vector<array> SamplerTransform::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SamplerTransform has no jvp implementation.");
}
std::vector<array> SamplerTransform::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("SamplerTransform has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> SamplerTransform::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SamplerTransform has no vmap implementation.");
}

} // namespace mlx::core
