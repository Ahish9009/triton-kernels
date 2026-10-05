"""Benchmark the Triton flash-attention kernel against PyTorch SDPA.

Single-head, non-causal attention ``softmax(Q·Kᵀ / √d)·V`` on ``(N, d)`` inputs.
The head dimension ``d`` is fixed; the sequence length ``N`` is swept.

Run with::

    python benchmarks/bench_flash_attention.py
"""

import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.flash_attention import flash_attention

D = 64  # head dimension


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(7, 13)],  # sequence length: 128 .. 4096
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch (SDPA)"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="TFLOP/s",
        plot_name="flash-attention-performance",
        args={},
    )
)
def benchmark(N, provider):
    Q = torch.randn((N, D), device="cuda", dtype=torch.float32)
    K = torch.randn((N, D), device="cuda", dtype=torch.float32)
    V = torch.randn((N, D), device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        q, k, v = Q[None, None], K[None, None], V[None, None]
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
    else:
        fn = lambda: flash_attention(Q, K, V)
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # FLOPs: QKᵀ (2·N²·d) + softmax·V (2·N²·d) = 4·N²·d.
    flops = 4 * N * N * D
    tflops = lambda ms: flops * 1e-12 / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmark_outputs",
    )
    os.makedirs(out_dir, exist_ok=True)
    benchmark.run(print_data=True, show_plots=False, save_path=out_dir)
    print(f"\nSaved results (.png + .csv) to {out_dir}")
