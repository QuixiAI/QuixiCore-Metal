import time

import mlx.core as mx

from tk import layernorm


def bench(fn, itt):
    for _ in range(itt):
        mx.eval(fn())
    toi = time.perf_counter()
    for _ in range(itt):
        mx.eval(fn())
    toc = time.perf_counter()
    return 1e3 * (toc - toi) / itt  # ms / iter


def bandwidth_gbps(x, w, b, ms):
    # bytes moved: read x + read w + read b + write o (bf16 = 2 bytes)
    nbytes = (x.size + w.size + b.size + x.size) * 2
    return nbytes / (ms / 1e3) / 1e9


for D in [256, 512, 768, 1024]:
    for rows in [4096, 16384, 65536]:
        x = mx.random.normal((rows, D)).astype(mx.bfloat16)
        w = mx.random.normal((D,)).astype(mx.bfloat16)
        b = mx.random.normal((D,)).astype(mx.bfloat16)
        eps = 1e-5
        itt = 50
        mx.metal.clear_cache()

        mlx_ms = bench(lambda: mx.fast.layer_norm(x, w, b, eps), itt)
        tk_ms = bench(lambda: layernorm(x, w, b, eps=eps), itt)
        print(
            f"({rows} x {D})  mlx {mlx_ms:.4f}ms ({bandwidth_gbps(x, w, b, mlx_ms):.1f} GB/s)"
            f"   tk {tk_ms:.4f}ms ({bandwidth_gbps(x, w, b, tk_ms):.1f} GB/s)"
            f"   ratio {mlx_ms / tk_ms * 100:.0f}%"
        )
    print("-" * 80)
