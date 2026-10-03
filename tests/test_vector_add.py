import pytest
import torch

from kernels import vector_add

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA device"
)


@pytest.mark.parametrize("n", [1, 128, 1024, 100_000])
def test_vector_add_matches_torch(n):
    x = torch.rand(n, device="cuda")
    y = torch.rand(n, device="cuda")
    out = vector_add(x, y)
    torch.testing.assert_close(out, x + y)
