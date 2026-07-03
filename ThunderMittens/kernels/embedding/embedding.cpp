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
TK_EMB_NO_AUTODIFF(MergeMultimodalSpans, "MergeMultimodalSpans")

} // namespace mlx::core
