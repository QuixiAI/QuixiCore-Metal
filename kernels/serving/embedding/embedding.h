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

/**
 *  Embedding backward: scatter-add the upstream grad dY (num_tok, D) into a (vocab, D) fp32 gradient
 *  table by token id (out[token_ids[t]] += scale * dY[t]). token_ids (num_tok,) int; dY float. A
 *  negative / out-of-range id contributes nothing. Returns (vocab, D) float32.
 **/
array embedding_backward(
    const array& token_ids,
    const array& dY,
    int vocab,
    float scale,
    StreamOrDevice s = {});

class EmbeddingBackward : public Primitive {
 public:
  EmbeddingBackward(Stream stream, int vocab, float scale)
      : Primitive(stream), vocab_(vocab), scale_(scale) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "EmbeddingBackward"; }
  void print(std::ostream& os) override { os << "EmbeddingBackward"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const EmbeddingBackward&>(other);
    return vocab_ == o.vocab_ && scale_ == o.scale_;
  }

 private:
  int vocab_;
  float scale_;
};

/**
 *  Embedding backward (sorted-segment, atomic-free): the tokens are pre-sorted by id on the host
 *  (perm = argsort(token_ids), sorted_ids = token_ids[perm]); each id's gradient is accumulated by a
 *  single threadgroup, so no atomics. sorted_ids / perm (num_tok,) int; dY (num_tok, D) float.
 *  Returns (vocab, D) float32. Same result as embedding_backward; wins under heavy id duplication.
 **/
array embedding_backward_sorted(
    const array& sorted_ids,
    const array& perm,
    const array& dY,
    int vocab,
    float scale,
    StreamOrDevice s = {});

class EmbeddingBackwardSorted : public Primitive {
 public:
  EmbeddingBackwardSorted(Stream stream, int vocab, float scale)
      : Primitive(stream), vocab_(vocab), scale_(scale) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "EmbeddingBackwardSorted"; }
  void print(std::ostream& os) override { os << "EmbeddingBackwardSorted"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const EmbeddingBackwardSorted&>(other);
    return vocab_ == o.vocab_ && scale_ == o.scale_;
  }

 private:
  int vocab_;
  float scale_;
};

/**
 *  Build the multimodal `src` map on-device (input to merge_multimodal_spans). span_offsets /
 *  span_lengths / modal_starts (num_spans,) int describe each modal span; returns src (num_tok,)
 *  int32 with src[t] = modal_starts[k]+offset for a token inside span k, else -1.
 **/
array build_multimodal_src(
    const array& span_offsets,
    const array& span_lengths,
    const array& modal_starts,
    int num_tok,
    StreamOrDevice s = {});

class BuildMultimodalSrc : public Primitive {
 public:
  BuildMultimodalSrc(Stream stream, int num_tok) : Primitive(stream), num_tok_(num_tok) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "BuildMultimodalSrc"; }
  void print(std::ostream& os) override { os << "BuildMultimodalSrc"; }
  bool is_equivalent(const Primitive& other) const override {
    return num_tok_ == static_cast<const BuildMultimodalSrc&>(other).num_tok_;
  }

 private:
  int num_tok_;
};

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
