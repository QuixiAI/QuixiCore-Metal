# ThunderMittens — performance status

Baselines and tuning conclusions from `perf/bench_kernels.py`. Numbers are median per-call
latency (ms), Apple Silicon, one warm GPU. Regenerate with:

```
.venv/bin/python perf/bench_kernels.py --backend mlx   --preset quick
.venv/bin/python perf/bench_kernels.py --backend torch --preset quick
```

## Serving-kernel latencies (`quick` preset: B=8, H=32, H_KV=8, D=128, ctx=2048)

| kernel | MLX (ms) | torch-MPS (ms) |
|---|---:|---:|
| paged_attention (v1 decode) | 1.13 | 1.78 |
| paged_attention_staged (GQA KV-reuse) | 1.90 | 1.93 |
| paged_attention_v2 (partition/reduce) | 0.51 | 0.50 |
| layernorm (N=4096, D=1024) | 0.17 | 0.17 |
| quantize_per_tensor_fp8 | 0.45 | 0.45 |
| quantize_per_token_fp8 | 0.22 | 0.21 |
| moe_grouped_gemm (E=8, H=2048, 2048 rows) | 1.32 | 1.33 |

## Tuning conclusions

### Long-context decode: **use `paged_attention_v2`**, not v1 or staged
`paged_attention_v2` is ~2.2× faster than the v1 single-threadgroup decode at ctx=2048 and the
gap widens with context, because it exposes the KV-sequence partitions as an extra grid axis
(`grid.z = num_partitions`) — far more threadgroups in flight than v1's `num_heads × batch`.

### GQA KV-reuse staging (item 6): **measured slower on Apple — keep v1/v2 as default**
`paged_attention_gqa_staged` is **1.7× slower** than plain `paged_attention` on MLX (1.08× on
torch-MPS) despite staging each KV vector once and reusing it across the group. Two Apple-specific
reasons: (1) it collapses the grid from `num_heads × batch` to `num_kv_heads × batch` threadgroups,
cutting the parallelism the GPU needs to hide memory latency; (2) the per-token `threadgroup_barrier`
pair serializes the group's simdgroups. The bandwidth saved (group_size× fewer KV reads) does not
recover that on this hardware — the caches are small enough that occupancy, not KV bandwidth, is the
bottleneck at decode batch sizes. This mirrors the earlier `gemm_staged.metal` finding that bigger
multi-warp tiles were slower on Apple. **The kernel is kept** (bit-equivalent, selectable) for
hardware/shapes where KV bandwidth dominates, but it is not the default.

### MoE grouped GEMM (item 2): one segmented launch replaces the per-expert host loop
`moe_grouped_gemm` runs all experts in a single dispatch (32×32 tiles, `expert_of_tile` lookup),
avoiding E separate encoder round-trips. It scales with total padded rows × H, as expected.

## Open items
- Sweep `paged_attention_v2` `partition_size` (256 used here) per context length.
- Benchmark the fp8 KV read paths (`paged_attention_fp8`, `paged_attention_v2_fp8`) vs the fp
  caches to quantify the dequant-on-read cost.
- MoE grouped GEMM vs the retained per-expert-dispatch fallback at small E / few rows per expert.
