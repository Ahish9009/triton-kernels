"""Element-wise vector subtraction in Triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _subtract_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x - y, mask=mask)


def vector_subtract(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    "Returns x - y computed on the GPU."
    out = torch.empty_like(x)
    assert x.shape == y.shape, "inputs need to have the same shape"
    assert x.is_cuda and y.is_cuda, "inputs must be on a CUDA device"
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _subtract_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out
