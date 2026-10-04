import pytest
import torch

from kernels import matmul

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA device"
)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048])
def test_vector_add_matches_torch(n):
    x = torch.rand((2*n,n), device="cuda")
    y = torch.rand((n,n), device="cuda")
    out = matmul(x, y)

    print(x)
    print(y)
    print("---")
    print(out)
    print(x @ y)
    torch.testing.assert_close(out, x @ y)
