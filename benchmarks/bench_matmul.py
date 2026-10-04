"""Benchmark the Triton matmul kernel against native PyTorch.
"""

import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels import matmul
torch.backends.cuda.matmul.allow_tf32 = True 


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(1, 13, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand((size,size), device="cuda", dtype=torch.float32)
    y = torch.rand((size,size), device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x @ y, quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(x, y), quantiles=quantiles
        )
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
