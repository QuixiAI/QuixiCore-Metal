// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Token embedding lookup: out[t] = scale * table[token_ids[t]] (+ pos_table[t] if provided).
 *  token_ids (num_tok,) int; table (vocab, D); optional pos_table (num_tok, D) same dtype as table.
 *  A negative / out-of-range token id emits zeros. Returns (num_tok, D), table's dtype.
 **/
array embedding_lookup(
    const array& token_ids,
    const array& table,
    const array& pos_table,   // (num_tok, D) or a size-1 placeholder (use_pos=false)
    float scale,
    StreamOrDevice s = {});

/**
 *  Multimodal span merge: out[t] = src[t] >= 0 ? modal[src[t]] : text[t]. text (num_tok, D),
 *  modal (num_modal, D) same dtype, src (num_tok,) int (-1 = keep text, >=0 = modal row). Returns
 *  (num_tok, D). The src map is the flattened placeholder->modal index list (built host-side).
 **/
array merge_multimodal_spans(
    const array& text,
    const array& modal,
    const array& src,
    StreamOrDevice s = {});

class EmbeddingLookup : public Primitive {
 public:
  EmbeddingLookup(Stream stream, float scale, bool use_pos)
      : Primitive(stream), scale_(scale), use_pos_(use_pos) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "EmbeddingLookup"; }
  void print(std::ostream& os) override { os << "EmbeddingLookup"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const EmbeddingLookup&>(other);
    return scale_ == o.scale_ && use_pos_ == o.use_pos_;
  }

 private:
  float scale_;
  bool use_pos_;
};

class MergeMultimodalSpans : public Primitive {
 public:
  explicit MergeMultimodalSpans(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MergeMultimodalSpans"; }
  void print(std::ostream& os) override { os << "MergeMultimodalSpans"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

} // namespace mlx::core
