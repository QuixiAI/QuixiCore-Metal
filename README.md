# ThunderMittens

ThunderMittens is an Apple Metal Shading Language (MSL) port for the [ThunderKittens](github.com/HazyResearch/ThunderKittens/) framework.

<div align="center" >
    <img src="assets/mittens.jpeg" height=350 alt="ThunderKittens logo" style="margin-bottom:px"/> 
</div>

<br>
<br>

## Prerequisites (all paths)

- Apple Silicon Mac.
- **Xcode with the Metal Toolchain installed.** Recent Xcode ships the Metal compiler as a
  separate component — `xcrun --find metal` resolving a path is not enough. Install it once with:
  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```
  Without this, both the MLX build and the PyTorch metallib build fail with
  `cannot execute tool 'metal' due to missing Metal Toolchain`.

## Project Structure

The same framework-agnostic `.metal` kernels (under `ThunderMittens/include` + `ThunderMittens/kernels`)
power three use cases:

### 1. MSL Kernel Development

For writing Metal Shading Language (MSL) kernels:
- Open the project in Xcode
- Xcode will handle all build processes
- Primitive unit tests live in `ThunderMittens/tests/unit` (gated by `ENABLE_TESTS` in
  `tests/unit/testing_commons/testing_flags.hpp`); build/run the `ThunderMittens` scheme.

### 2. MLX Kernel Integration with Python

For using ThunderKittens kernels within MLX in Python:

#### Prerequisites
- Python 3.8+
- CMake
- Xcode Command Line Tools

#### Installation Steps

1. Navigate to ThunderMittens/mlx directory
2. Install MLX with parallel build:
   ```bash
   CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e ".[dev]"
   ```

3. Navigate to ThunderMittens/kernels directory
4. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

5. Build kernels and bindings:
   ```bash
   python setup.py build_ext -j8 --inplace
   ```

> Build artifacts are written to a repo-root `build/` dir (via `kernels/setup.cfg`) so they stay
> out of the Xcode-synchronized source tree. Validate with `python -m pytest */correctness/`.

### 3. PyTorch (Apple MPS) Integration with Python

For using ThunderKittens kernels from PyTorch on the `mps` device. This path is **independent of
MLX** — it needs only PyTorch and the Metal toolchain (no CMake/nanobind, no MLX build). The same
`.metal` kernels are compiled into a standalone metallib with `xcrun metal` and dispatched onto
PyTorch's MPS stream.

#### Prerequisites
- Python 3.9+
- PyTorch with MPS (`torch>=2.1`; nightlies work). MPS custom-kernel support uses
  `torch::mps::get_command_buffer()`.
- The Metal toolchain (see [Prerequisites](#prerequisites-all-paths)).

#### Installation Steps

1. Navigate to the PyTorch backend and install it (pulls in PyTorch, leaves an existing install
   such as a nightly untouched):
   ```bash
   pip install -e ThunderMittens/kernels/tk_torch
   ```
2. Use it — the metallib and the ObjC++ extension build automatically on first import:
   ```python
   import torch
   import tk_torch

   x = torch.randn(2, 128, 1024, dtype=torch.bfloat16, device="mps")
   w = torch.randn(1024, dtype=torch.bfloat16, device="mps")
   b = torch.randn(1024, dtype=torch.bfloat16, device="mps")
   y = tk_torch.layernorm(x, w, b)   # matches torch.nn.functional.layer_norm
   ```

## Testing

Run from `ThunderMittens/kernels/`:

```bash
# MLX correctness (each kernel vs an MLX oracle, e.g. mx.fast.layer_norm)
python -m pytest */correctness/

# PyTorch MPS correctness (each kernel vs a torch reference, e.g. F.layer_norm)
python -m pytest tk_torch/tests/

# Cross-backend parity: the SAME metallib kernel on MLX vs MPS must agree for
# identical inputs (catches host-ABI drift between the two backends)
python -m pytest tests_parity/

# ...or everything at once (each suite skips cleanly if its framework is absent)
python -m pytest */correctness/ tk_torch/tests/ tests_parity/
```

Primitive-level MSL unit tests (register/shared tile ops) build and run through Xcode — see
[1. MSL Kernel Development](#1-msl-kernel-development).

## References

Please see [our blog post](https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx) to learn more about this work. Please checkout [our paper](https://arxiv.org/abs/2410.20399) to learn more about the ThunderKittens project. 



