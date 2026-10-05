"""Flash attention in Triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    out_ptr,
    N,
    Q_str_x, Q_str_y,
    K_str_x, K_str_y,
    V_str_x, V_str_y,
    out_str_x, out_str_y,
    d: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Load the row tile from Q
    Q_row_start = pid*BLOCK_SIZE_Y
    Q_col_offsets = tl.arange(0, d)[None,:] 
    Q_row_offsets = (Q_row_start + tl.arange(0, BLOCK_SIZE_Y))[:,None]
    Q_offsets = Q_row_offsets*Q_str_x + Q_col_offsets*Q_str_y
    Q_mask = Q_offsets < N*d
    Q = tl.load(Q_ptr + Q_offsets, mask=Q_mask, other=0.0) # (BLOCK_SIZE_Y, d)

    root_d = d**0.5
    tmp = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    tmp2 = tl.zeros((BLOCK_SIZE_Y, d), dtype=tl.float32)
    row_mxs = tl.zeros((BLOCK_SIZE_Y, 1), dtype=tl.float32) - float('inf')
    s = tl.zeros((BLOCK_SIZE_Y, 1), dtype=tl.float32)

    for i in range(tl.cdiv(N,BLOCK_SIZE_X)):
        K_row_start = i*BLOCK_SIZE_X
        K_col_offsets = tl.arange(0, d)[None,:]
        K_row_offsets = (K_row_start + tl.arange(0, BLOCK_SIZE_X))[:,None]
        K_offsets = K_row_offsets*K_str_x + K_col_offsets*K_str_y
        K_mask = K_offsets < N*d
        K = tl.load(K_ptr + K_offsets, mask=K_mask, other=0.0) # (BLOCK_SIZE_X, d)

        tmp = tl.dot(Q, tl.trans(K))/root_d # (BLOCK_SIZE_Y, BLOCK_SIZE_X)
        key_idx = i * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)   # (BLOCK_X,)
        tmp = tl.where((key_idx < N)[None, :], tmp, -float('inf'))
        new_mxs = tmp.max(axis=1, keep_dims=True)
        s = tl.exp(row_mxs - new_mxs)*s + tl.exp(tmp-new_mxs).sum(axis=1, keep_dims=True)
        
        V_row_start = i*BLOCK_SIZE_X
        V_col_offsets = tl.arange(0, d)[None,:]
        V_row_offsets = (V_row_start + tl.arange(0, BLOCK_SIZE_X))[:,None]
        V_offsets = V_row_offsets*V_str_x + V_col_offsets*V_str_y
        V_mask = V_offsets < N*d
        V = tl.load(V_ptr + V_offsets, mask=V_mask, other=0.0) # (BLOCK_SIZE_X, d)

        tmp2 = tl.exp(row_mxs - new_mxs)*tmp2 + tl.dot(tl.exp(tmp-new_mxs), V)
        row_mxs = new_mxs
    tmp2 /= s
    
    out_row_offsets = pid*BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)[:,None]
    out_col_offsets = (tl.arange(0, d))[None,:]
    out_offsets = out_row_offsets*out_str_x + out_col_offsets*out_str_y
    out_mask = (out_row_offsets < N) & (out_col_offsets < d)
    tl.store(out_ptr + out_offsets, tmp2, mask=out_mask)


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Return ``flash attention`` computed on the GPU with a Triton kernel."""
    assert Q.shape == K.shape, "inputs must have the same shape"
    assert K.shape == V.shape, "inputs must have the same shape"
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "inputs must be on a CUDA device"
    
    N, d = Q.shape
    out = torch.empty_like(V)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_Y"]),)
    _flash_attention_kernel[grid](
            Q, K, V,
            out,
            N,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1),
            V.stride(0), V.stride(1),
            out.stride(0), out.stride(1),
            d=32,
            BLOCK_SIZE_X=32,
            BLOCK_SIZE_Y=32
        )

    return out
