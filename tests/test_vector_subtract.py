import pytest
import torch

from kernels import vector_subtract

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA device"
)


@pytest.mark.parametrize("n", [1, 128, 1024, 100000])
def test_vector_subtract_matches_torch(n):
    x = torch.rand(n, device="cuda")
    y = torch.rand(n, device="cuda")
    out = vector_subtract(x, y)
    torch.testing.assert_close(out, x - y)
