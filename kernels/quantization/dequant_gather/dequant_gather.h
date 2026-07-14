#pragma once

#include <string>
#include <utility>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array dequant_gather(
    const array& table,
    const array& ids,
    const std::string& format,
    float scale = 1.0f,
    StreamOrDevice s = {});

array quantized_embedding(
    const array& table,
    const array& ids,
    const array& add,
    const std::string& format,
    float scale = 1.0f,
    bool use_add = false,
    const std::string& output_dtype = "float16",
    StreamOrDevice s = {});

array quantized_embedding_bag(
    const array& table,
    const array& ids,
    const array& offsets,
    const array& sample_weights,
    const std::string& format,
    float scale = 1.0f,
    bool use_weights = false,
    bool mean_mode = false,
    const std::string& output_dtype = "float16",
    StreamOrDevice s = {});

class DequantGather : public Primitive {
 public:
  DequantGather(Stream stream, std::string format, int rows, int columns, float scale)
      : Primitive(stream), format_(std::move(format)), rows_(rows), columns_(columns),
        scale_(scale) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "DequantGather"; }
  void print(std::ostream& os) override { os << "DequantGather[" << format_ << "]"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::string format_;
  int rows_;
  int columns_;
  float scale_;
};

class QuantizedEmbedding : public Primitive {
 public:
  QuantizedEmbedding(Stream stream, std::string format, int rows, int columns,
                     float scale, bool use_add, std::string output_dtype)
      : Primitive(stream), format_(std::move(format)), rows_(rows), columns_(columns),
        scale_(scale), use_add_(use_add), output_dtype_(std::move(output_dtype)) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QuantizedEmbedding"; }
  void print(std::ostream& os) override { os << "QuantizedEmbedding[" << format_ << "]"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::string format_;
  int rows_;
  int columns_;
  float scale_;
  bool use_add_;
  std::string output_dtype_;
};

class QuantizedEmbeddingBag : public Primitive {
 public:
  QuantizedEmbeddingBag(Stream stream, std::string format, int rows, int columns,
                        float scale, bool use_weights, bool mean_mode,
                        std::string output_dtype)
      : Primitive(stream), format_(std::move(format)), rows_(rows), columns_(columns),
        scale_(scale), use_weights_(use_weights), mean_mode_(mean_mode),
        output_dtype_(std::move(output_dtype)) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QuantizedEmbeddingBag"; }
  void print(std::ostream& os) override {
    os << "QuantizedEmbeddingBag[" << format_ << (mean_mode_ ? ",mean]" : ",sum]");
  }
  bool is_equivalent(const Primitive& other) const override;

 private:
  std::string format_;
  int rows_;
  int columns_;
  float scale_;
  bool use_weights_;
  bool mean_mode_;
  std::string output_dtype_;
};

} // namespace mlx::core
