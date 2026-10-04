"""Matrix multiplication in Triton (hopefully not naive).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    A_stride_rows,
    A_stride_cols,
    B_stride_rows,
    B_stride_cols,
    C_stride_rows,
    C_stride_cols,
    a, b, q,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_r = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    out = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    for k in range(0,b,BLOCK_SIZE_K):
        A_cols = tl.arange(0, BLOCK_SIZE_K) + k
        A_rows = tl.arange(0, BLOCK_SIZE_Y) + pid_r*BLOCK_SIZE_Y
        mask_A_rows = (A_rows < a)[:,None]
        mask_A_cols = (A_cols < b)[None,:]
        A_cols = A_cols[None,:]
        A_rows = A_rows[:,None]
        A_offsets = A_rows*A_stride_rows + A_cols*A_stride_cols

        A = tl.load(A_ptr + A_offsets, mask=mask_A_rows&mask_A_cols)

        B_cols = tl.arange(0, BLOCK_SIZE_X) + pid_c*BLOCK_SIZE_X
        B_rows = tl.arange(0, BLOCK_SIZE_K) + k
        mask_B_rows = (B_rows < b)[:,None]
        mask_B_cols = (B_cols < q)[None,:]
        B_cols = B_cols[None,:]
        B_rows = B_rows[:,None]
        B_offsets = B_rows*B_stride_rows + B_cols*B_stride_cols

        B = tl.load(B_ptr + B_offsets, mask=mask_B_rows&mask_B_cols)
        out += tl.dot(A, B)

    out_rows_offset = tl.arange(0, BLOCK_SIZE_Y) + pid_r*BLOCK_SIZE_Y
    out_cols_offset = tl.arange(0, BLOCK_SIZE_X) + pid_c*BLOCK_SIZE_X
    mask_rows = out_rows_offset < a
    mask_cols = out_cols_offset < q
    out_rows_offset = out_rows_offset[:,None]
    out_cols_offset = out_cols_offset[None,:]
    mask = mask_rows&mask_cols
    out_offsets = out_rows_offset*C_stride_rows + out_cols_offset
    tl.store(C_ptr + out_offsets, out, mask=mask)
    

def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Return ``x @ y`` computed on the GPU with a Triton kernel."""
    a,b = A.shape
    p,q = B.shape
    C = torch.zeros((a,q), device="cuda")
    assert b == p, "inputs must have the inner dimension"
    assert A.is_cuda and B.is_cuda, "inputs must be on a CUDA device"

    grid = lambda meta: (triton.cdiv(a, meta["BLOCK_SIZE_Y"]),
                         triton.cdiv(q, meta["BLOCK_SIZE_X"]))
    _matmul_kernel[grid](
            A, 
            B,
            C,
            A.stride(0), A.stride(1), 
            B.stride(0), B.stride(1), 
            C.stride(0), C.stride(1), 
            a, b, q,
            BLOCK_SIZE_X=128,
            BLOCK_SIZE_Y=128,
            BLOCK_SIZE_K=32,
            num_stages=5,
            num_warps=8
    )
    return C

