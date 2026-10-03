import pytest
import torch

from kernels import matmul_naive

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA device"
)


@pytest.mark.parametrize("n", [1, 64, 128, 256, 512, 1024])
def test_vector_add_matches_torch(n):
    x = torch.rand((2*n,n), device="cuda")
    y = torch.rand((n,n), device="cuda")
    out = matmul_naive(x, y)
    print(x, y)
    print(out)
    print(x @ y)
    torch.testing.assert_close(out, x @ y)
