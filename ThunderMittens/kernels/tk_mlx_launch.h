// MLX encoder adapter for the shared tk::launch_<name>() functions (tk_launch.h).
//
// Binding goes through MLX's set_input_array/set_output_array so MLX's residency and
// scheduling bookkeeping is preserved — the MLX Primitive `eval_gpu` builds an
// MLXEncoder over its (device, command_encoder) and calls the same launch function the
// PyTorch backend uses, so the host ABI cannot drift between the two backends.

#pragma once

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

#include "tk_launch.h"

namespace mlx::core {

struct MLXEncoder {
  using in_t = const array&;
  using out_t = array&;

  metal::Device& d;
  metal::CommandEncoder& ce;

  MLXEncoder(metal::Device& device, metal::CommandEncoder& encoder)
      : d(device), ce(encoder) {}

  void pipeline(const std::string& name) {
    d.register_library("mlx_ext");
    ce.set_compute_pipeline_state(d.get_kernel(name, "mlx_ext"));
  }
  void in(const array& a, int i) { ce.set_input_array(a, i); }
  void out(array& a, int i) { ce.set_output_array(a, i); }
  template <class T>
  void bytes(const T& v, int i) {
    ce.set_bytes(v, i);
  }
  void dispatch(int gx, int gy, int gz, int tx, int ty, int tz) {
    ce.dispatch_threadgroups(MTL::Size(gx, gy, gz), MTL::Size(tx, ty, tz));
  }
};

} // namespace mlx::core
