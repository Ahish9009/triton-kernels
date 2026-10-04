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
    xstr, xstc,
    ystr, ystc,
    outstr, outstc,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    mx = -float('inf')
    st = False
    s = 0.
    for i in range(col_size//BLOCK_SIZE +1):
        col_offset = i*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        xoffsets = pid*xstr + col_offset*xstc
        yoffsets = pid*ystr + col_offset*ystc
        x = tl.load(x_ptr+xoffsets, mask=col_offset<col_size, other=-float('inf'))
        y = tl.load(y_ptr+yoffsets, mask=col_offset<col_size, other=-float('inf'))
        z = x+y
        new_mx = tl.maximum(z.max(axis=0), mx)
        s = tl.exp(mx-new_mx)*s + tl.exp(z-new_mx).sum(axis=0)
        mx = new_mx
    for i in range(col_size//BLOCK_SIZE +1):
        col_offset = i*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        xoffsets = pid*xstr + col_offset*xstc
        yoffsets = pid*ystr + col_offset*ystc
        outoffsets = pid*outstr + col_offset*outstc
        x = tl.load(x_ptr+xoffsets, mask=col_offset<col_size, other=-float('inf'))
        y = tl.load(y_ptr+yoffsets, mask=col_offset<col_size, other=-float('inf'))
        exps = tl.exp(x+y-mx)/s
        tl.store(out_ptr+outoffsets, exps, mask=col_offset<col_size)

def fused_vector_add_softmax(x: torch.Tensor, y: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Return ``softmax(x+y)`` computed on the GPU with a Triton kernel."""
    assert x.shape == y.shape, "inputs must have the same shape"
    assert x.is_cuda and y.is_cuda, "inputs must be on a CUDA device"
    
    # Change the shape for the kernel
    dims = x.shape
    changed_dims = False
    changed_shape = False
    if axis != len(dims)-1 or axis != -1:
        changed_dims = True
        x = x.transpose(axis, -1)
        y = y.transpose(axis, -1)
    if len(dims) > 2:
        changed_shape = True
        old_shape = x.shape
        x = x.reshape(-1,x.shape[-1])
        y = y.reshape(-1,y.shape[-1])

    r, c = x.shape
    n = x.numel();
    out = torch.empty_like(x)
    grid = lambda meta: (r,)
    _fused_vector_add_softmax_kernel[grid](
            x, 
            y, 
            out,
            n,
            c,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE=1024)

    if changed_shape:
        out = out.reshape(old_shape)
    if changed_dims:
        out = out.transpose(axis, -1)

    return out
