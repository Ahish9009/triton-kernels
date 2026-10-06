import pytest
import torch

from kernels.flash_decoding import flash_decoding

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA device"
)

# d is a compile-time constant in the kernel (currently specialized to 32).
D = 32
# tl.dot defaults to TF32, so compare against a TF32-level tolerance.
RTOL, ATOL = 2e-2, 1e-2


def reference(Q, K, V):
    # Q: (1, d), K/V: (N_kv, d) -> (1, d)
    scores = (Q @ K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
    return torch.softmax(scores, dim=-1) @ V


@pytest.mark.parametrize("N_kv", [1, 16, 32, 64, 100, 128, 512, 1000, 4096])
def test_matches_reference(N_kv):
    Q = torch.randn((1, D), device="cuda")
    K = torch.randn((N_kv, D), device="cuda")
    V = torch.randn((N_kv, D), device="cuda")
    out = flash_decoding(Q, K, V)
    ref = reference(Q, K, V)
    assert out.shape == (1, D)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_matches_torch_sdpa():
    N_kv = 1024
    Q = torch.randn((1, D), device="cuda")
    K = torch.randn((N_kv, D), device="cuda")
    V = torch.randn((N_kv, D), device="cuda")
    out = flash_decoding(Q, K, V)
    # SDPA wants (..., L, E); one query token attending over N_kv keys.
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q[None, None], K[None, None], V[None, None]
    )[0, 0]
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_output_is_convex_combination_of_V():
    # softmax weights sum to 1, so the output must lie within the range of V.
    N_kv = 512
    Q = torch.randn((1, D), device="cuda")
    K = torch.randn((N_kv, D), device="cuda")
    V = torch.rand((N_kv, D), device="cuda")  # all in [0, 1)
    out = flash_decoding(Q, K, V)
    assert torch.isfinite(out).all()
    assert (out >= -ATOL).all() and (out <= 1 + ATOL).all()


def test_single_key():
    # with one key, softmax is trivially 1 -> output equals that V row.
    Q = torch.randn((1, D), device="cuda")
    K = torch.randn((1, D), device="cuda")
    V = torch.randn((1, D), device="cuda")
    out = flash_decoding(Q, K, V)
    torch.testing.assert_close(out, V, rtol=RTOL, atol=ATOL)
