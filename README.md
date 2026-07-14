# QuixiCore Metal

QuixiCore Metal is the Apple Silicon backend for the
[QuixiCore](https://github.com/QuixiAI/QuixiCore) kernel library. It provides
native Metal kernels with Python integrations for MLX and PyTorch MPS.

The backend follows the shared QuixiCore contract: common operation names,
correctness expectations, quantization metadata, and benchmark conventions,
implemented natively for Apple GPUs.

## What Is Included

- Metal Shading Language kernels under `kernels/` and `include/metal/`.
- MLX Python bindings exposed as `tk`.
- PyTorch MPS bindings exposed as `tk_torch`.
- Xcode project support through `QuixiCoreMetal.xcodeproj`.
- Correctness, parity, and benchmark harnesses for the supported integrations.

Kernel coverage includes normalization, activation, attention, linear attention,
state-space, matmul, quantization, vision, MoE, sampling, serving, optimizer, and
utility operations. The exact supported surface is tracked in
[`.quixicore/kernels.yaml`](.quixicore/kernels.yaml).

## Requirements

- Apple Silicon Mac.
- Xcode with the Metal Toolchain installed.
- Python virtual environment for Python integrations.

Install the Metal Toolchain once if `xcrun metal` is present but kernel builds
fail with a missing toolchain error:

```bash
xcodebuild -downloadComponent MetalToolchain
```

The MLX binding currently targets the MLX 0.21 C++ extension API. Use Python
3.12 for the MLX path unless you are intentionally porting the extension to a
newer MLX C++ API.

## Build

Run commands from the repository root.

```bash
scripts/configure
scripts/build xcode -configuration Debug
```

For the MLX-backed Python package:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r bindings/python/requirements.txt
PYTHON=.venv/bin/python scripts/build python
```

For the PyTorch MPS package:

```bash
python3 -m venv .venv-torch
. .venv-torch/bin/activate
python -m pip install torch
PYTHON=.venv-torch/bin/python scripts/build pytorch_mps
```

The PyTorch package builds its Objective-C++ extension and Metal library on
first import.

## Use

MLX:

```python
import mlx.core as mx
import tk

x = mx.random.normal((4096, 1024)).astype(mx.bfloat16)
w = mx.ones((1024,), dtype=mx.bfloat16)
y = tk.rms_norm(x, w)
mx.eval(y)
```

PyTorch MPS:

```python
import torch
import tk_torch

x = torch.randn(2, 128, 1024, dtype=torch.bfloat16, device="mps")
w = torch.ones(1024, dtype=torch.bfloat16, device="mps")
b = torch.zeros(1024, dtype=torch.bfloat16, device="mps")
y = tk_torch.layernorm(x, w, b)
torch.mps.synchronize()
```

### Specialized composed operations

The public `tk` module also exposes pure tensor operations that combine common
decode, sparse-projection, embedding, and vision stages without introducing an
application-level runtime:

| Operation | Public API and contract |
| --- | --- |
| Packed embeddings | `quantized_embedding` and `quantized_embedding_bag` gather or reduce GGUF/MX/FP rows directly from packed tables. |
| Decode projections | `decode_linear_epilogue` and `decode_swiglu` support dense, q4_0, q8_0, and q6_K weights, fused activations/bias/residuals, and optional output quantization. |
| Sparse output projection | `lm_head_masked` consumes packed allow masks; `lm_head_candidates` consumes CSR candidate lists. Both return deterministic top-k ids and log-probabilities without materializing full logits. |
| Spatial projection | `space_to_depth_norm_linear` composes block-2/block-4 space-to-depth, LayerNorm, and projection with odd-edge padding. |
| Functional cache decode | `decode_cache_attention` composes optional Q/K RMSNorm, split-half RoPE, functional cache append, and GQA attention. |

MLX arrays and PyTorch MPS tensors use the same top-level functions. Operations
with measured crossover points auto-route between direct Metal and framework
composition; `use_kernel=True` or `False` selects a path explicitly where the
API exposes that option.

## Test

```bash
# Xcode build-for-testing
scripts/test xcode

# MLX correctness
PYTHON=.venv/bin/python scripts/test correctness

# PyTorch MPS correctness
PYTHON=.venv-torch/bin/python scripts/test mps

# Cross-backend parity; install torch in the MLX venv first if needed.
PYTHON=.venv/bin/python scripts/test parity
```

MPS and parity targets exit cleanly with a skip message when Torch or MPS support
is unavailable.

## Benchmark

Use `perf/bench_kernels.py` from the repository root:

```bash
PYTHON=.venv/bin/python perf/bench_kernels.py --backend mlx --preset smoke --kernel all
PYTHON=.venv-torch/bin/python perf/bench_kernels.py --backend torch --preset smoke --kernel all
```

Benchmark results are hardware-, OS-, framework-, and shape-dependent. See
[`perf/perf.md`](perf/perf.md) for methodology and benchmark conventions.

## Repository Layout

```text
.quixicore/              Backend and kernel metadata
include/metal/           Shared Metal tile substrate and headers
include/quixicore/metal/ Public QuixiCore Metal headers
kernels/                 Operation implementations by family
bindings/python/         MLX-backed Python package
bindings/pytorch_mps/    PyTorch MPS package
bindings/mlx/            MLX source integration
tests/                   Correctness, parity, integration, and unit tests
perf/                    Benchmark harnesses, configs, results, and baselines
scripts/                 Common build, test, bench, and clean entry points
```

More detail is in [`docs/repository-structure.md`](docs/repository-structure.md).

## Metadata And Docs

- Backend metadata: [`.quixicore/backend.yaml`](.quixicore/backend.yaml)
- Kernel coverage: [`.quixicore/kernels.yaml`](.quixicore/kernels.yaml)
- Quant formats: [`.quixicore/quant-formats.yaml`](.quixicore/quant-formats.yaml)
- Contribution guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security policy: [`SECURITY.md`](SECURITY.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)

## Credits

QuixiCore Metal builds on the ThunderKittens-style tiled GPU programming model
originally published by HazyResearch and adapted for Apple Metal by QuixiAI.
Some model-serving kernels were informed by reference implementations from
[metal-forge](https://github.com/AlpinDale) by
[AlpinDale](https://github.com/AlpinDale).

## License

MIT. See [`LICENSE`](LICENSE).
