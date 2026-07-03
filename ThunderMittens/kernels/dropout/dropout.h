#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Inverted dropout (training). out = keep ? x/(1-p) : 0, keep_i = rng_uniform(seed, i) >= p.
 *  The mask is a pure function of (seed, index) — the backward recomputes it, no mask is stored.
 *  x float (fp16/bf16/fp32); p in [0,1). Returns x's shape/dtype.
 **/
array dropout(const array& x, float p, uint32_t seed, StreamOrDevice s = {});

/** Dropout backward: dx = keep ? dy/(1-p) : 0, same mask recomputed from (seed, index). */
array dropout_backward(const array& dy, float p, uint32_t seed, StreamOrDevice s = {});

#define TK_DROPOUT_PRIM(CLASS)                                                     \
  class CLASS : public Primitive {                                                 \
   public:                                                                         \
    CLASS(Stream stream, float p, uint32_t seed)                                   \
        : Primitive(stream), p_(p), seed_(seed) {}                                 \
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override;        \
    void eval_gpu(const std::vector<array>&, std::vector<array>&) override;        \
    std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,   \
                           const std::vector<int>&) override;                      \
    std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,   \
                           const std::vector<int>&, const std::vector<array>&) override; \
    std::pair<std::vector<array>, std::vector<int>> vmap(                          \
        const std::vector<array>&, const std::vector<int>&) override;              \
    const char* name() const { return #CLASS; }                                   \
    void print(std::ostream& os) override { os << #CLASS; }                        \
    bool is_equivalent(const Primitive& other) const override {                    \
      auto& o = static_cast<const CLASS&>(other);                                  \
      return p_ == o.p_ && seed_ == o.seed_;                                       \
    }                                                                              \
   private:                                                                        \
    float p_;                                                                      \
    uint32_t seed_;                                                                \
  };

TK_DROPOUT_PRIM(Dropout)
TK_DROPOUT_PRIM(DropoutBwd)
#undef TK_DROPOUT_PRIM

} // namespace mlx::core
