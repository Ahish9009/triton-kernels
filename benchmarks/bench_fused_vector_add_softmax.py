"""Benchmark the fused vector-add + softmax kernel against PyTorch.

Row-wise ``softmax(x + y)`` over the last axis. Rows are fixed; the feature
dimension ``N`` (columns) is swept.

Run with::

    python benchmarks/bench_fused_vector_add_softmax.py
"""

import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.fused_vector_add_softmax import fused_vector_add_softmax

ROWS = 4096


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(8, 15)],  # columns: 256 .. 16384
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="fused-vector-add-softmax-performance",
        args={},
    )
)
def benchmark(N, provider):
    x = torch.rand((ROWS, N), device="cuda", dtype=torch.float32)
    y = torch.rand((ROWS, N), device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        fn = lambda: torch.softmax(x + y, dim=-1)
    else:
        fn = lambda: fused_vector_add_softmax(x, y)
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    # reads x and y, writes out -> 3 * numel * bytes moved.
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmark_outputs",
    )
    os.makedirs(out_dir, exist_ok=True)
    benchmark.run(print_data=True, show_plots=False, save_path=out_dir)
    print(f"\nSaved results (.png + .csv) to {out_dir}")
