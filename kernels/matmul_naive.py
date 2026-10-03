"""Matrix multiplication in Triton (naive).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_naive_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_stride,
    y_stride,
    x_max,
    y_max,
    o_max,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    final = 0.
    n_seen = 0
    i = 0
    while n_seen < x_stride:
        offsets_x = pid_x * x_stride + tl.arange(0, BLOCK_SIZE_X) + i * BLOCK_SIZE_X
        offsets_y = pid_y + y_stride * (i*BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y))
        mask_x = offsets_x < x_max
        mask_y = offsets_y < y_max

        x = tl.load(x_ptr + offsets_x, mask=mask_x)
        y = tl.load(y_ptr + offsets_y, mask=mask_y)
        p = x*y
        final += p.sum(axis=0)
        
        n_seen += BLOCK_SIZE_X
        i += 1
        
    out_offset = pid_x * y_stride + pid_y
    mask_out = out_offset < o_max
    tl.store(out_ptr + out_offset, final, mask=mask_out)


def matmul_naive(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return ``x @ y`` computed on the GPU with a Triton kernel."""
    a,b = x.shape
    p,q = y.shape
    out = torch.empty((a,q), device="cuda")
    assert b == p, "inputs must have the inner dimension"
    assert x.is_cuda and y.is_cuda, "inputs must be on a CUDA device"
    n_x, n_y = out.shape

    grid = lambda meta: (a,q)
    _matmul_naive_kernel[grid](x, y, out, b, q, a*b, p*q, a*q, BLOCK_SIZE_X=256, BLOCK_SIZE_Y=256)
    return out
