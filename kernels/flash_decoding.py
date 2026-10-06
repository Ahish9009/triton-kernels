"""Flash decoding in Triton.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _reduce_kernel(
    out_ptr,
    maxs_ptr,
    sums_ptr,
    res_ptr,
    N_kv,
    out_str_x, out_str_y,
    res_str_x, res_str_y,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):

    mx = -float('inf')
    s = 0.
    final = tl.zeros((1, d), dtype=tl.float32)
    for i in range(tl.cdiv(N_kv, BLOCK_SIZE)):
        cmax = tl.load(maxs_ptr + i, mask=i<N_kv, other=0.0)
        cs = tl.load(sums_ptr + i, mask=i<N_kv, other=0.0)

        row_offsets = i
        row_mask = row_offsets < N_kv
        col_offsets = tl.arange(0, d)
        offsets = row_offsets*out_str_x + col_offsets*out_str_y
        partial_o = tl.load(out_ptr + offsets, mask=row_mask, other=0.0)

        new_max = tl.maximum(mx, cmax)
        final = tl.exp(mx-new_max)*final + tl.exp(cmax-new_max)*partial_o
        s += cs*tl.exp(cmax-new_max)

    final /= s
    row_offsets = tl.arange(0,1)
    col_offsets = tl.arange(0,d)
    offsets = row_offsets[:,None]*res_str_x + col_offsets[None,:]*res_str_y
    mask = col_offsets < d
    tl.store(res_ptr + offsets, final, mask=mask)


@triton.jit
def _flash_decoding_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    out_ptr, maxs_ptr, sums_ptr,
    N_kv,
    Q_str_x, Q_str_y,
    K_str_x, K_str_y,
    V_str_x, V_str_y,
    out_str_x, out_str_y,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    K_row_start = pid*BLOCK_SIZE

    Q_row_offsets = tl.arange(0, 1)[:,None]
    Q_col_offsets = tl.arange(0, d)[None,:]
    Q_offsets = Q_row_offsets*Q_str_x + Q_col_offsets*Q_str_y
    Q_mask = Q_col_offsets < d
    Q = tl.load(Q_ptr + Q_offsets, mask=Q_mask, other=0.0)

    K_row_offsets = tl.arange(0, BLOCK_SIZE) + K_row_start
    K_col_offsets = tl.arange(0, d)
    K_mask = K_row_offsets < N_kv
    K_offsets = K_row_offsets*K_str_x + K_col_offsets*K_str_y
    K = tl.load(K_ptr + K_offsets, mask=K_mask, other=-float('inf'))

    root_d = d**0.5
    tmp = tl.dot(Q, tl.trans(K))/root_d # 1xBLOCK_SIZE

    k_idx = K_row_start + tl.arange(0, BLOCK_SIZE)
    tmp = tl.where(k_idx[None, :] < N_kv, tmp, -float('inf'))

    V_row_offsets = K_row_start + tl.arange(0, BLOCK_SIZE)
    V_col_offsets = tl.arange(0, d)
    V_mask = V_row_offsets < N_kv
    V_offsets = V_row_offsets*V_str_x + V_col_offsets*V_str_y
    V = tl.load(V_ptr + V_offsets, mask=V_mask, other=0.0)

    mx = tmp.max(axis=1)
    tmp = tl.exp(tmp-mx)
    s = tmp.sum(axis=1)
    partial_o = tl.dot(tmp, V) # (1xd)
    
    out_row_offsets = pid
    out_col_offsets = tl.arange(0, d)[None,:]
    out_mask = out_row_offsets < N_kv
    out_offsets = out_row_offsets[:,None]*out_str_x + out_col_offsets
    tl.store(out_ptr + out_offsets, partial_o, mask=out_mask)
    tl.store(maxs_ptr + pid, mx, mask=(pid < N_kv))
    tl.store(sums_ptr + pid, s, mask=(pid < N_kv))


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Return ``flash attention`` computed on the GPU with a Triton kernel."""
    assert K.shape == V.shape, "inputs must have the same shape"
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "inputs must be on a CUDA device"
    
    N_kv, d = K.shape
    res = torch.zeros((1,d), device="cuda")
    out = torch.zeros((N_kv,d), device="cuda")
    maxs = torch.zeros((N_kv,), device="cuda")
    sums = torch.zeros((N_kv,), device="cuda")
    grid = lambda meta: (triton.cdiv(N_kv, meta["BLOCK_SIZE"]),)
    _flash_decoding_kernel[grid](
            Q, K, V,
            out, maxs, sums,
            N_kv,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1),
            V.stride(0), V.stride(1),
            out.stride(0), out.stride(1),
            d=32,
            BLOCK_SIZE=32
        )
    _reduce_kernel[(1,)](
            out,
            maxs,
            sums,
            res,
            N_kv,
            out.stride(0), out.stride(1),
            res.stride(0), res.stride(1),
            d=32,
            BLOCK_SIZE=32
    )

    return res
