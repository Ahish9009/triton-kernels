"""Benchmark the Triton flash-decoding kernel against PyTorch SDPA.

Decode step: a single query attends over a KV cache of length ``N_kv``
(``softmax(Q·Kᵀ / √d)·V`` with ``Q`` shape ``(1, d)``). This is a
memory-bound regime, so throughput is reported as GB/s of K+V read.
The head dimension ``d`` is fixed; the cache length ``N_kv`` is swept.

Run with::

    python benchmarks/bench_flash_decoding.py
"""

import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.flash_decoding import flash_decoding

D = 32  # head dimension


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N_kv"],
        x_vals=[2**i for i in range(8, 17)],  # KV-cache length: 256 .. 65536
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch (SDPA)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="flash-decoding-performance",
        args={},
    )
)
def benchmark(N_kv, provider):
    Q = torch.randn((1, D), device="cuda", dtype=torch.float32)
    K = torch.randn((N_kv, D), device="cuda", dtype=torch.float32)
    V = torch.randn((N_kv, D), device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        q, k, v = Q[None, None], K[None, None], V[None, None]
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
    else:
        fn = lambda: flash_decoding(Q, K, V)
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # memory-bound: reads K and V (2 * N_kv * d elements).
    gbps = lambda ms: 2 * N_kv * D * K.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmark_outputs",
    )
    os.makedirs(out_dir, exist_ok=True)
    benchmark.run(print_data=True, show_plots=False, save_path=out_dir)
    print(f"\nSaved results (.png + .csv) to {out_dir}")
