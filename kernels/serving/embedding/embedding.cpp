// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "embedding/embedding.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static bool emb_is_float(Dtype dt) {
  return dt == float32 || dt == float16 || dt == bfloat16;
}

array embedding_lookup(
    const array& token_ids,
    const array& table,
    const array& pos_table,
    float scale,
    StreamOrDevice s /* = {} */) {
  if (token_ids.ndim() != 1) {
    throw std::invalid_argument("embedding_lookup: token_ids must be (num_tok,)");
  }
  if (table.ndim() != 2 || !emb_is_float(table.dtype())) {
    throw std::invalid_argument("embedding_lookup: table must be (vocab, D) float");
  }
  const int n_tok = token_ids.shape(0);
  const int D = table.shape(1);
  const bool use_pos = pos_table.size() > 1;
  if (use_pos && (pos_table.ndim() != 2 || pos_table.shape(0) != n_tok || pos_table.shape(1) != D)) {
    throw std::invalid_argument("embedding_lookup: pos_table must be (num_tok, D)");
  }
  auto tok_c = contiguous(astype(token_ids, int32, s), false, s);
  auto tab_c = contiguous(table, false, s);
  auto pos_c = use_pos ? contiguous(astype(pos_table, table.dtype(), s), false, s)
                       : zeros({1}, table.dtype(), s);
  return array(
      {n_tok, D}, table.dtype(),
      std::make_shared<EmbeddingLookup>(to_stream(s), scale, use_pos),
      {tok_c, tab_c, pos_c});
}

void EmbeddingLookup::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("EmbeddingLookup has no CPU implementation.");
}
void EmbeddingLookup::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& token_ids = inputs[0];
  auto& table = inputs[1];
  auto& pos_table = inputs[2];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int n_tok = token_ids.shape(0);
  const int vocab = table.shape(0);
  const int D = table.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_embedding_lookup(enc, token_ids, table, pos_table, out, D, vocab, n_tok, scale_,
                              use_pos_ ? 1 : 0, type_to_name(table));
}

array embedding_lookup_types(
    const array& token_ids, const array& type_ids,
    const array& token_table, const array& type_table,
    float token_scale, StreamOrDevice s) {
  if (token_ids.ndim() != 1 || type_ids.shape() != token_ids.shape()) {
    throw std::invalid_argument("embedding_lookup_types: token_ids/type_ids must be equal 1-D arrays");
  }
  if (token_table.ndim() != 2 || type_table.ndim() != 2 ||
      !emb_is_float(token_table.dtype()) || type_table.dtype() != token_table.dtype() ||
      type_table.shape(1) != token_table.shape(1)) {
    throw std::invalid_argument("embedding_lookup_types: tables must be same-dtype (vocab,D) float tensors");
  }
  const int n_tok = token_ids.shape(0), D = token_table.shape(1);
  return array(
      {n_tok, D}, token_table.dtype(),
      std::make_shared<EmbeddingLookupTypes>(to_stream(s), token_scale),
      {contiguous(astype(token_ids, int32, s), false, s),
       contiguous(astype(type_ids, int32, s), false, s),
       contiguous(token_table, false, s), contiguous(type_table, false, s)});
}

void EmbeddingLookupTypes::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("EmbeddingLookupTypes has no CPU implementation.");
}
void EmbeddingLookupTypes::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& token_ids = inputs[0]; auto& type_ids = inputs[1];
  auto& token_table = inputs[2]; auto& type_table = inputs[3]; auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index); MLXEncoder enc(d, ce);
  tk::launch_embedding_lookup_types(
      enc, token_ids, type_ids, token_table, type_table, out,
      token_table.shape(1), token_table.shape(0), type_table.shape(0),
      token_ids.shape(0), token_scale_, type_to_name(token_table));
}

array embedding_backward(
    const array& token_ids,
    const array& dY,
    int vocab,
    float scale,
    StreamOrDevice s /* = {} */) {
  if (token_ids.ndim() != 1) {
    throw std::invalid_argument("embedding_backward: token_ids must be (num_tok,)");
  }
  if (dY.ndim() != 2 || !emb_is_float(dY.dtype()) || dY.shape(0) != token_ids.shape(0)) {
    throw std::invalid_argument("embedding_backward: dY must be (num_tok, D) float, num_tok matching");
  }
  const int D = dY.shape(1);
  auto tok_c = contiguous(astype(token_ids, int32, s), false, s);
  auto dY_c = contiguous(dY, false, s);
  return array(
      {vocab, D}, float32,
      std::make_shared<EmbeddingBackward>(to_stream(s), vocab, scale),
      {tok_c, dY_c});
}

void EmbeddingBackward::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("EmbeddingBackward has no CPU implementation.");
}
void EmbeddingBackward::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& token_ids = inputs[0];
  auto& dY = inputs[1];
  auto& dtable = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  dtable.set_data(allocator::malloc_or_wait(dtable.nbytes()));
  const int n_tok = token_ids.shape(0);
  const int D = dY.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_embedding_zero_f32(enc, dtable, vocab_ * D);         // zero the accumulator first
  tk::launch_embedding_backward(enc, token_ids, dY, dtable, D, vocab_, n_tok, scale_,
                                type_to_name(dY));
}

array embedding_backward_sorted(
    const array& sorted_ids,
    const array& perm,
    const array& dY,
    int vocab,
    float scale,
    StreamOrDevice s /* = {} */) {
  if (sorted_ids.ndim() != 1 || perm.ndim() != 1 || perm.shape(0) != sorted_ids.shape(0)) {
    throw std::invalid_argument("embedding_backward_sorted: sorted_ids / perm must be (num_tok,)");
  }
  if (dY.ndim() != 2 || !emb_is_float(dY.dtype()) || dY.shape(0) != sorted_ids.shape(0)) {
    throw std::invalid_argument(
        "embedding_backward_sorted: dY must be (num_tok, D) float, num_tok matching");
  }
  const int D = dY.shape(1);
  auto sid_c = contiguous(astype(sorted_ids, int32, s), false, s);
  auto perm_c = contiguous(astype(perm, int32, s), false, s);
  auto dY_c = contiguous(dY, false, s);
  return array(
      {vocab, D}, float32,
      std::make_shared<EmbeddingBackwardSorted>(to_stream(s), vocab, scale),
      {sid_c, perm_c, dY_c});
}

void EmbeddingBackwardSorted::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("EmbeddingBackwardSorted has no CPU implementation.");
}
void EmbeddingBackwardSorted::eval_gpu(const std::vector<array>& inputs,
                                       std::vector<array>& outputs) {
  auto& sorted_ids = inputs[0];
  auto& perm = inputs[1];
  auto& dY = inputs[2];
  auto& dtable = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  dtable.set_data(allocator::malloc_or_wait(dtable.nbytes()));
  const int n_tok = sorted_ids.shape(0);
  const int D = dY.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_embedding_zero_f32(enc, dtable, vocab_ * D);         // zero first (absent ids stay 0)
  tk::launch_embedding_backward_sorted(enc, sorted_ids, perm, dY, dtable, D, vocab_, n_tok, scale_,
                                       type_to_name(dY));
}

array build_multimodal_src(
    const array& span_offsets, const array& span_lengths, const array& modal_starts, int num_tok,
    StreamOrDevice s /* = {} */) {
  if (span_offsets.ndim() != 1 || span_lengths.shape() != span_offsets.shape() ||
      modal_starts.shape() != span_offsets.shape()) {
    throw std::invalid_argument("build_multimodal_src: span_* must be equal-length 1-D");
  }
  auto so = contiguous(astype(span_offsets, int32, s), false, s);
  auto sl = contiguous(astype(span_lengths, int32, s), false, s);
  auto ms = contiguous(astype(modal_starts, int32, s), false, s);
  return array({num_tok}, int32, std::make_shared<BuildMultimodalSrc>(to_stream(s), num_tok),
               {so, sl, ms});
}

void BuildMultimodalSrc::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("BuildMultimodalSrc has no CPU implementation.");
}
void BuildMultimodalSrc::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& so = inputs[0];
  auto& sl = inputs[1];
  auto& ms = inputs[2];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int num_spans = so.shape(0);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_build_multimodal_src(enc, so, sl, ms, out, num_spans, num_tok_);
}

array merge_multimodal_spans(
    const array& text,
    const array& modal,
    const array& src,
    StreamOrDevice s /* = {} */) {
  if (text.ndim() != 2 || !emb_is_float(text.dtype())) {
    throw std::invalid_argument("merge_multimodal_spans: text must be (num_tok, D) float");
  }
  if (modal.ndim() != 2 || modal.shape(1) != text.shape(1) || modal.dtype() != text.dtype()) {
    throw std::invalid_argument("merge_multimodal_spans: modal must be (num_modal, D), same D/dtype");
  }
  if (src.ndim() != 1 || src.shape(0) != text.shape(0)) {
    throw std::invalid_argument("merge_multimodal_spans: src must be (num_tok,)");
  }
  auto text_c = contiguous(text, false, s);
  auto modal_c = contiguous(modal, false, s);
  auto src_c = contiguous(astype(src, int32, s), false, s);
  return array(
      text.shape(), text.dtype(),
      std::make_shared<MergeMultimodalSpans>(to_stream(s)),
      {text_c, modal_c, src_c});
}

void MergeMultimodalSpans::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MergeMultimodalSpans has no CPU implementation.");
}
void MergeMultimodalSpans::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& text = inputs[0];
  auto& modal = inputs[1];
  auto& src = inputs[2];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int n_tok = text.shape(0);
  const int D = text.shape(1);
  const int n_modal = modal.shape(0);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_merge_multimodal_spans(enc, text, modal, src, out, D, n_tok, n_modal,
                                    type_to_name(text));
}

#define TK_EMB_NO_AUTODIFF(CLASS, LABEL)                                     \
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

TK_EMB_NO_AUTODIFF(EmbeddingLookup, "EmbeddingLookup")
TK_EMB_NO_AUTODIFF(EmbeddingLookupTypes, "EmbeddingLookupTypes")
TK_EMB_NO_AUTODIFF(EmbeddingBackward, "EmbeddingBackward")
TK_EMB_NO_AUTODIFF(EmbeddingBackwardSorted, "EmbeddingBackwardSorted")
TK_EMB_NO_AUTODIFF(BuildMultimodalSrc, "BuildMultimodalSrc")
TK_EMB_NO_AUTODIFF(MergeMultimodalSpans, "MergeMultimodalSpans")

} // namespace mlx::core
