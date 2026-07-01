#!/usr/bin/env python3
"""Benchmark ThunderMittens serving kernels and write a Markdown + JSONL summary.

Self-contained (numpy + whichever framework you have). Times each case with a warmup
then the median of N iterations, syncing the GPU each iteration. Head-to-head pairs
(e.g. paged_attention vs paged_attention_staged) are reported side by side with a ratio
so item-6/item-2 style tradeoffs can be read straight off the table.

    # from the repo root
    .venv/bin/python perf/bench_kernels.py                 # MLX, default preset
    .venv/bin/python perf/bench_kernels.py --backend torch # PyTorch MPS
    .venv/bin/python perf/bench_kernels.py --preset quick --iters 50

Cases self-skip (recorded, not fatal) when a kernel/framework is unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNELS_DIR = REPO_ROOT / "ThunderMittens" / "kernels"
if str(KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(KERNELS_DIR))

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------- backends
class Backend:
    """Thin adapter so cases are written once and run on MLX or PyTorch-MPS."""

    def __init__(self, name):
        self.name = name
        if name == "mlx":
            import mlx.core as mx
            self.mx = mx
            self._dtypes = {"f32": mx.float32, "f16": mx.float16, "bf16": mx.bfloat16}
        elif name == "torch":
            import torch
            self.torch = torch
            if not torch.backends.mps.is_available():
                raise RuntimeError("torch MPS not available")
            self._dtypes = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
        else:
            raise ValueError(name)

    def array(self, np_arr, dtype="f32"):
        if self.name == "mlx":
            return self.mx.array(np_arr).astype(self._dtypes[dtype])
        return self.torch.from_numpy(np_arr).to(self._dtypes[dtype]).to("mps")

    def int_array(self, np_arr):
        if self.name == "mlx":
            return self.mx.array(np_arr)
        return self.torch.from_numpy(np_arr).to("mps")

    def sync(self, *vals):
        if self.name == "mlx":
            self.mx.eval(*[v for v in vals if v is not None])
        else:
            self.torch.mps.synchronize()

    def tk(self):
        import tk
        return tk


# --------------------------------------------------------------------------- timing
def time_call(fn, backend, warmup, iters):
    """Return the median per-call latency in ms (each call fully synced)."""
    for _ in range(warmup):
        backend.sync(fn())
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        backend.sync(out)
        samples.append(1e3 * (time.perf_counter() - t0))
    return statistics.median(samples)


# --------------------------------------------------------------------------- cases
# Each case factory returns (label, thunk) or raises to self-skip. Grouped so head-to-head
# variants sit next to each other in the output.
def paged_decode_cases(be, shp):
    """paged_attention (one threadgroup per query head) vs the KV-reuse staged variant."""
    tk = be.tk()
    rng = np.random.default_rng(0)
    B, H, H_KV, D = shp["B"], shp["H"], shp["H_KV"], shp["D"]
    ctx, block_size = shp["ctx"], 16
    num_blocks = B * ((ctx + block_size - 1) // block_size)
    max_blocks = (ctx + block_size - 1) // block_size
    q = (0.1 * rng.standard_normal((B, H, D))).astype(np.float32)
    kc = (0.1 * rng.standard_normal((num_blocks, block_size, H_KV, D))).astype(np.float32)
    vc = (0.1 * rng.standard_normal((num_blocks, block_size, H_KV, D))).astype(np.float32)
    bt = np.arange(B * max_blocks, dtype=np.int32).reshape(B, max_blocks)
    cl = np.full((B,), ctx, dtype=np.int32)
    q_d, kc_d, vc_d = be.array(q, "bf16"), be.array(kc, "bf16"), be.array(vc, "bf16")
    bt_d, cl_d = be.int_array(bt), be.int_array(cl)
    yield ("paged_attention", lambda: tk.paged_attention(q_d, kc_d, vc_d, bt_d, cl_d))
    yield ("paged_attention_staged", lambda: tk.paged_attention_staged(q_d, kc_d, vc_d, bt_d, cl_d))
    yield ("paged_attention_v2", lambda: tk.paged_attention_v2(q_d, kc_d, vc_d, bt_d, cl_d,
                                                               partition_size=256))


def norm_cases(be, shp):
    tk = be.tk()
    rng = np.random.default_rng(1)
    N, Dm = shp["N"], shp["Dm"]
    x = (rng.standard_normal((N, Dm))).astype(np.float32)
    w = (rng.standard_normal((Dm,))).astype(np.float32)
    b = (rng.standard_normal((Dm,))).astype(np.float32)
    x_d, w_d, b_d = be.array(x, "bf16"), be.array(w, "bf16"), be.array(b, "bf16")
    yield ("layernorm", lambda: tk.layernorm(x_d, w_d, b_d))


def quant_cases(be, shp):
    tk = be.tk()
    rng = np.random.default_rng(2)
    N, Dm = shp["N"], shp["Dm"]
    x = (rng.standard_normal((N, Dm)) * 2.0).astype(np.float32)
    x_d = be.array(x, "f16")
    yield ("quantize_per_tensor_fp8", lambda: tk.quantize_per_tensor_fp8(x_d))
    yield ("quantize_per_token_fp8", lambda: tk.quantize_per_token_fp8(x_d))


def moe_cases(be, shp):
    tk = be.tk()
    rng = np.random.default_rng(3)
    E, Hd = shp["E"], shp["Hd"]
    rows = shp["moe_rows"]
    # one 32-padded segment per expert (grouped_gemm contract): expert_of_tile per 32-row tile.
    tiles = rows // 32
    x = (0.1 * rng.standard_normal((rows, Hd))).astype(np.float32)
    W = (0.1 * rng.standard_normal((E, Hd, Hd))).astype(np.float32)
    eot = (np.arange(tiles, dtype=np.int32) * E // max(tiles, 1)).astype(np.int32)
    x_d, W_d, eot_d = be.array(x, "bf16"), be.array(W, "bf16"), be.int_array(eot)
    yield ("moe_grouped_gemm", lambda: tk.moe_grouped_gemm(x_d, W_d, eot_d))


CASE_GROUPS = [paged_decode_cases, norm_cases, quant_cases, moe_cases]

# Note: the layernorm kernel is instantiated for D in {256,512,768,1024}, so Dm is pinned to
# 1024 (its max). MLX hard-aborts on an unsupported kernel load, so shapes must stay in range.
PRESETS = {
    "smoke": dict(B=4, H=16, H_KV=4, D=128, ctx=512, N=1024, Dm=1024,
                  E=8, Hd=1024, moe_rows=512),
    "quick": dict(B=8, H=32, H_KV=8, D=128, ctx=2048, N=4096, Dm=1024,
                  E=8, Hd=2048, moe_rows=2048),
    "serving": dict(B=16, H=32, H_KV=8, D=128, ctx=4096, N=8192, Dm=1024,
                    E=16, Hd=4096, moe_rows=4096),
}


# --------------------------------------------------------------------------- runner
def run(backend_name, preset, warmup, iters):
    try:
        be = Backend(backend_name)
    except Exception as e:  # noqa: BLE001
        print(f"backend {backend_name} unavailable: {e}")
        return []
    shp = PRESETS[preset]
    rows = []
    for group in CASE_GROUPS:
        try:
            cases = list(group(be, shp))
        except Exception as e:  # noqa: BLE001
            print(f"  skip group {group.__name__}: {type(e).__name__}: {e}")
            continue
        for label, thunk in cases:
            try:
                ms = time_call(thunk, be, warmup, iters)
                rows.append({"backend": backend_name, "kernel": label, "median_ms": ms})
                print(f"  {label:32s} {ms:8.4f} ms")
            except Exception as e:  # noqa: BLE001
                rows.append({"backend": backend_name, "kernel": label, "skipped": str(e)})
                print(f"  {label:32s} SKIP ({type(e).__name__}: {e})")
    return rows


def add_ratios(rows):
    """Attach paired-variant ratios (staged / baseline) for the decode head-to-head."""
    by = {r["kernel"]: r.get("median_ms") for r in rows if "median_ms" in r}
    base, staged = by.get("paged_attention"), by.get("paged_attention_staged")
    if base and staged:
        return {"paged_staged_vs_base": round(staged / base, 3)}
    return {}


def to_markdown(rows, meta, ratios):
    lines = ["# ThunderMittens kernel benchmarks", ""]
    lines.append(f"- backend: `{meta['backend']}`  preset: `{meta['preset']}`  "
                 f"warmup/iters: {meta['warmup']}/{meta['iters']}")
    lines.append(f"- shapes: `{meta['shapes']}`")
    lines.append("")
    lines.append("| kernel | median (ms) |")
    lines.append("|---|---:|")
    for r in rows:
        if "median_ms" in r:
            lines.append(f"| {r['kernel']} | {r['median_ms']:.4f} |")
        else:
            lines.append(f"| {r['kernel']} | _skip_ |")
    if ratios:
        lines.append("")
        lines.append("## Head-to-head")
        for k, v in ratios.items():
            note = "staged faster" if v < 1 else "baseline faster"
            lines.append(f"- `{k}` = {v}  ({note})")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--preset", choices=list(PRESETS), default="quick")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default=None, help="markdown output path (default perf/results/<backend>_<preset>.md)")
    args = ap.parse_args()

    print(f"== {args.backend} / {args.preset} (warmup={args.warmup} iters={args.iters}) ==")
    rows = run(args.backend, args.preset, args.warmup, args.iters)
    if not rows:
        return 1
    ratios = add_ratios(rows)
    meta = {"backend": args.backend, "preset": args.preset, "warmup": args.warmup,
            "iters": args.iters, "shapes": PRESETS[args.preset]}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = Path(args.out) if args.out else RESULTS_DIR / f"{args.backend}_{args.preset}.md"
    md_path.write_text(to_markdown(rows, meta, ratios))
    jsonl_path = RESULTS_DIR / f"{args.backend}_{args.preset}.jsonl"
    with jsonl_path.open("w") as f:
        for r in rows:
            f.write(json.dumps({**r, "preset": args.preset}) + "\n")
    print(f"\nwrote {md_path}")
    if ratios:
        print("ratios:", ratios)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
