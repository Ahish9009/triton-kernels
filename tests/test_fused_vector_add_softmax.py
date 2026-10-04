import pytest
import torch

from kernels.fused_vector_add_softmax import fused_vector_add_softmax

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA device"
)


@pytest.mark.parametrize("rows", [1, 8, 128])
@pytest.mark.parametrize("cols", [1, 16, 64, 100, 128, 500, 1024])
def test_matches_torch_softmax(rows, cols):
    x = torch.rand((rows, cols), device="cuda")
    y = torch.rand((rows, cols), device="cuda")
    out = fused_vector_add_softmax(x, y)
    ref = torch.softmax(x + y, dim=-1)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("cols", [16, 64, 100, 1024])
def test_rows_sum_to_one(cols):
    # softmax over the last axis: every row must sum to 1.
    x = torch.rand((32, cols), device="cuda")
    y = torch.rand((32, cols), device="cuda")
    out = fused_vector_add_softmax(x, y)
    sums = out.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-4, atol=1e-4)


def test_numerically_stable_for_large_values():
    # x + y large should not overflow (exercises the max-subtraction path).
    x = torch.full((4, 256), 50.0, device="cuda")
    y = torch.full((4, 256), 50.0, device="cuda")
    out = fused_vector_add_softmax(x, y)
    ref = torch.softmax(x + y, dim=-1)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)
