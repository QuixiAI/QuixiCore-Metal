#pragma once

#include <optional>
#include <string>

#include "base_q_descriptor.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array base_qdequant(
    const array& codes, const array& scales,
    const std::optional<array>& biases, int bits, int group_size,
    const std::string& scale_dtype = "bf16", bool symmetric = false,
    const std::string& layout = "metal",
    const std::string& output_dtype = "float16", StreamOrDevice s = {});

array base_qgemv(
    const array& codes, const array& scales,
    const std::optional<array>& biases, const array& x, int bits,
    int group_size, const std::string& scale_dtype = "bf16",
    bool symmetric = false, const std::string& layout = "metal",
    StreamOrDevice s = {});

array base_qgemm(
    const array& codes, const array& scales,
    const std::optional<array>& biases, const array& x, int bits,
    int group_size, const std::string& scale_dtype = "bf16",
    bool symmetric = false, const std::string& layout = "metal",
    StreamOrDevice s = {});

std::vector<array> base_qgemv_qkv(
    const array& q_codes, const array& q_scales,
    const std::optional<array>& q_biases, const array& k_codes,
    const array& k_scales, const std::optional<array>& k_biases,
    const array& v_codes, const array& v_scales,
    const std::optional<array>& v_biases, const array& x, int bits,
    int group_size, const std::string& scale_dtype = "bf16",
    bool symmetric = false, const std::string& layout = "metal",
    StreamOrDevice s = {});

array base_qgemv_swiglu(
    const array& gate_codes, const array& gate_scales,
    const std::optional<array>& gate_biases, const array& up_codes,
    const array& up_scales, const std::optional<array>& up_biases,
    const array& x, int bits, int group_size,
    const std::string& scale_dtype = "bf16", bool symmetric = false,
    const std::string& layout = "metal", StreamOrDevice s = {});

array base_qembedding(
    const array& codes, const array& scales,
    const std::optional<array>& biases, const array& ids, int bits,
    int group_size, const std::string& scale_dtype = "bf16",
    bool symmetric = false, const std::string& layout = "metal",
    const std::string& output_dtype = "float16", StreamOrDevice s = {});

array base_qmoe_gemm(
    const array& codes, const array& scales,
    const std::optional<array>& biases, const array& input,
    const array& expert_of_tile, int bits, int group_size,
    const std::string& scale_dtype = "bf16", bool symmetric = false,
    const std::string& layout = "metal", StreamOrDevice s = {});

array base_qmoe_swiglu(
    const array& codes, const array& scales,
    const std::optional<array>& biases, const array& input,
    const array& expert_of_tile, int bits, int group_size,
    const std::string& scale_dtype = "bf16", bool symmetric = false,
    const std::string& layout = "metal", StreamOrDevice s = {});

class BaseQDequant : public Primitive {
 public:
  BaseQDequant(Stream stream, tk::BaseQDescriptor descriptor, int rows, int columns)
      : Primitive(stream), descriptor_(descriptor), rows_(rows), columns_(columns) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "BaseQDequant"; }
  void print(std::ostream& os) override { os << "BaseQDequant[q" << descriptor_.bits << "]"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  tk::BaseQDescriptor descriptor_;
  int rows_;
  int columns_;
};

class BaseQMatmul : public Primitive {
 public:
  BaseQMatmul(Stream stream, tk::BaseQDescriptor descriptor, int rows,
              int inner, int columns)
      : Primitive(stream), descriptor_(descriptor), rows_(rows), inner_(inner),
        columns_(columns) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "BaseQMatmul"; }
  void print(std::ostream& os) override { os << "BaseQMatmul[q" << descriptor_.bits << "]"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  tk::BaseQDescriptor descriptor_;
  int rows_;
  int inner_;
  int columns_;
};

class BaseQEmbedding : public Primitive {
 public:
  BaseQEmbedding(Stream stream, tk::BaseQDescriptor descriptor, int rows,
                 int columns, int tokens)
      : Primitive(stream), descriptor_(descriptor), rows_(rows), columns_(columns),
        tokens_(tokens) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "BaseQEmbedding"; }
  void print(std::ostream& os) override { os << "BaseQEmbedding[q" << descriptor_.bits << "]"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  tk::BaseQDescriptor descriptor_;
  int rows_;
  int columns_;
  int tokens_;
};

class BaseQGemvQKV : public Primitive {
 public:
  BaseQGemvQKV(Stream stream, tk::BaseQDescriptor descriptor, int q_rows,
               int k_rows, int v_rows, int inner)
      : Primitive(stream), descriptor_(descriptor), q_rows_(q_rows),
        k_rows_(k_rows), v_rows_(v_rows), inner_(inner) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "BaseQGemvQKV"; }
  void print(std::ostream& os) override { os << "BaseQGemvQKV[q" << descriptor_.bits << "]"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  tk::BaseQDescriptor descriptor_;
  int q_rows_;
  int k_rows_;
  int v_rows_;
  int inner_;
};

class BaseQGemvSwiGLU : public Primitive {
 public:
  BaseQGemvSwiGLU(Stream stream, tk::BaseQDescriptor descriptor, int rows,
                  int inner)
      : Primitive(stream), descriptor_(descriptor), rows_(rows), inner_(inner) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "BaseQGemvSwiGLU"; }
  void print(std::ostream& os) override { os << "BaseQGemvSwiGLU[q" << descriptor_.bits << "]"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  tk::BaseQDescriptor descriptor_;
  int rows_;
  int inner_;
};

class BaseQMoeGemm : public Primitive {
 public:
  BaseQMoeGemm(Stream stream, tk::BaseQDescriptor descriptor, int total_rows,
               int inner, int output_rows)
      : Primitive(stream), descriptor_(descriptor), total_rows_(total_rows),
        inner_(inner), output_rows_(output_rows) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "BaseQMoeGemm"; }
  void print(std::ostream& os) override { os << "BaseQMoeGemm[q" << descriptor_.bits << "]"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  tk::BaseQDescriptor descriptor_;
  int total_rows_;
  int inner_;
  int output_rows_;
};

class BaseQMoeSwiGLU : public Primitive {
 public:
  BaseQMoeSwiGLU(Stream stream, tk::BaseQDescriptor descriptor, int total_rows,
                 int inner, int intermediate)
      : Primitive(stream), descriptor_(descriptor), total_rows_(total_rows),
        inner_(inner), intermediate_(intermediate) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "BaseQMoeSwiGLU"; }
  void print(std::ostream& os) override { os << "BaseQMoeSwiGLU[q" << descriptor_.bits << "]"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  tk::BaseQDescriptor descriptor_;
  int total_rows_;
  int inner_;
  int intermediate_;
};

} // namespace mlx::core
