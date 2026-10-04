"""Fused vector add and softmax in Triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_vector_add_softmax_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n,
    col_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    mx = -float('inf')
    st = False
    s = 0.
    for i in range(col_size//BLOCK_SIZE +1):
        col_offset = i*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offsets = pid*col_size + col_offset
        x = tl.load(x_ptr+offsets, mask=col_offset<col_size, other=-float('inf'))
        y = tl.load(y_ptr+offsets, mask=col_offset<col_size, other=-float('inf'))
        z = x+y
        new_mx = tl.maximum(z.max(axis=0), mx)
        s = tl.exp(mx-new_mx)*s + tl.exp(z-new_mx).sum(axis=0)
        mx = new_mx
    for i in range(col_size//BLOCK_SIZE +1):
        col_offset = i*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offsets = pid*col_size + col_offset
        x = tl.load(x_ptr+offsets, mask=col_offset<col_size, other=-float('inf'))
        y = tl.load(y_ptr+offsets, mask=col_offset<col_size, other=-float('inf'))
        exps = tl.exp(x+y-mx)/s
        tl.store(out_ptr+offsets, exps, mask=col_offset<col_size)

def fused_vector_add_softmax(x: torch.Tensor, y: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Return ``softmax(x+y)`` computed on the GPU with a Triton kernel."""
    assert x.shape == y.shape, "inputs must have the same shape"
    assert x.is_cuda and y.is_cuda, "inputs must be on a CUDA device"

    r, c = x.shape
    n = x.numel();
    out = torch.empty_like(x)
    grid = lambda meta: (r,)
    _fused_vector_add_softmax_kernel[grid](x, y, out, n, c, BLOCK_SIZE=1024)
    return out
