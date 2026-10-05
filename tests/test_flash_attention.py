import pytest
import torch

from kernels.flash_attention import flash_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA device"
)

# tl.dot defaults to TF32, so compare against a TF32-level tolerance.
RTOL, ATOL = 2e-2, 1e-2


def reference(Q, K, V):
    scores = (Q @ K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
    return torch.softmax(scores, dim=-1) @ V


@pytest.mark.parametrize("N", [16, 32, 64, 100, 128, 512, 1000])
@pytest.mark.parametrize("d", [16, 32, 64, 128])
def test_matches_reference(N, d):
    Q = torch.randn((N, d), device="cuda")
    K = torch.randn((N, d), device="cuda")
    V = torch.randn((N, d), device="cuda")
    out = flash_attention(Q, K, V)
    ref = reference(Q, K, V)
    assert out.shape == (N, d)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_matches_torch_sdpa():
    N, d = 256, 64
    Q = torch.randn((N, d), device="cuda")
    K = torch.randn((N, d), device="cuda")
    V = torch.randn((N, d), device="cuda")
    out = flash_attention(Q, K, V)
    # SDPA wants (..., L, E); add batch+head dims then squeeze back.
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q[None, None], K[None, None], V[None, None]
    )[0, 0]
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_rows_of_attention_weights_normalized():
    # out should be a convex combination of V rows: within the range of V.
    N, d = 128, 32
    Q = torch.randn((N, d), device="cuda")
    K = torch.randn((N, d), device="cuda")
    V = torch.rand((N, d), device="cuda")  # all positive in [0, 1)
    out = flash_attention(Q, K, V)
    assert torch.isfinite(out).all()
    assert (out >= -ATOL).all() and (out <= 1 + ATOL).all()
