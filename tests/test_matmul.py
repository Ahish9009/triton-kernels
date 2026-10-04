import pytest
import torch

from kernels import matmul

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA device"
)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048])
def test_matmul_matches_torch(n):
    x = torch.rand((2*n,n), device="cuda")
    y = torch.rand((n,n), device="cuda")
    out = matmul(x, y)

    torch.backends.cuda.matmul.allow_tf32 = True 
    ref = torch.matmul(x, y).to(out.dtype)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
