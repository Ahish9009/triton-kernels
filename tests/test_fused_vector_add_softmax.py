import pytest
import torch

from kernels.fused_vector_add_softmax import fused_vector_add_softmax

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA device"
)

RTOL, ATOL = 1e-3, 1e-5


# --- 2D, softmax over the last axis (incl. > BLOCK_SIZE and non-multiples) ---
@pytest.mark.parametrize("rows", [1, 8, 128])
@pytest.mark.parametrize("cols", [1, 16, 64, 100, 128, 1000, 1024, 2048, 5000])
def test_2d_last_axis(rows, cols):
    x = torch.rand((rows, cols), device="cuda")
    y = torch.rand((rows, cols), device="cuda")
    out = fused_vector_add_softmax(x, y)
    ref = torch.softmax(x + y, dim=-1)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


# --- arbitrary rank, softmax over the last axis ---
@pytest.mark.parametrize(
    "shape",
    [(4, 32), (2, 3, 64), (2, 3, 4, 50), (8, 2048), (3, 5, 1000)],
)
def test_multidim_last_axis(shape):
    x = torch.rand(shape, device="cuda")
    y = torch.rand(shape, device="cuda")
    out = fused_vector_add_softmax(x, y)
    ref = torch.softmax(x + y, dim=-1)
    assert out.shape == x.shape
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


# --- softmax over an arbitrary axis ---
@pytest.mark.parametrize(
    "shape,axis",
    [
        ((8, 16), 0),
        ((16, 32), 1),
        ((16, 32), -1),
        ((4, 5, 6), 0),
        ((4, 5, 6), 1),
        ((4, 5, 6), 2),
        ((2, 3, 4, 5), 2),
    ],
)
def test_arbitrary_axis(shape, axis):
    x = torch.rand(shape, device="cuda")
    y = torch.rand(shape, device="cuda")
    out = fused_vector_add_softmax(x, y, axis=axis)
    ref = torch.softmax(x + y, dim=axis)
    assert out.shape == x.shape
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


# --- the reduced axis must sum to 1 ---
@pytest.mark.parametrize("cols", [16, 100, 1024, 2048])
def test_rows_sum_to_one(cols):
    x = torch.rand((32, cols), device="cuda")
    y = torch.rand((32, cols), device="cuda")
    out = fused_vector_add_softmax(x, y)
    sums = out.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-4, atol=1e-4)


# --- non-contiguous inputs (kernel receives explicit strides) ---
def test_non_contiguous_inputs():
    base_x = torch.rand((64, 128), device="cuda")
    base_y = torch.rand((64, 128), device="cuda")
    x = base_x.t()  # (128, 64), non-contiguous
    y = base_y.t()
    assert not x.is_contiguous()
    out = fused_vector_add_softmax(x, y)
    ref = torch.softmax(x + y, dim=-1)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


# --- numerical stability: large magnitudes must not overflow ---
def test_numerically_stable_for_large_values():
    x = torch.full((4, 2048), 50.0, device="cuda")
    y = torch.full((4, 2048), 50.0, device="cuda")
    out = fused_vector_add_softmax(x, y)
    ref = torch.softmax(x + y, dim=-1)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
