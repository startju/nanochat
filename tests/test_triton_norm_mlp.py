"""Parity tests for the fused Triton MLP block.

Math: y = x + relu²(norm(x) @ W_fc.T) @ W_proj.T
"""

import pytest
import torch
import torch.nn.functional as F

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("triton kernels require CUDA", allow_module_level=True)

from nanoops.triton_fused_mlp import fused_mlp


def _reference(x, W_fc, W_proj, eps=1e-6):
    """Eager PyTorch reference using plain RMSNorm."""
    x_hat = F.rms_norm(x, (x.shape[-1],), eps=eps)
    z = x_hat @ W_fc.t()
    r = z.clamp(min=0).square()
    return x + r @ W_proj.t()


@pytest.mark.parametrize("dtype,atol", [
    (torch.float32, 1e-3),
    # bf16 atol raised from 1e-1 → 1.5e-1: fused fwd's `summed * summed`
    # now runs in bf16 before the fp32 accumulator (`tl.sum(..., dtype=fp32)`),
    # whereas F.rms_norm promotes bf16 → fp32 *before* the squaring. The
    # truncated-mantissa squared products plus a long sum over D push max
    # diff slightly past the old 1e-1 tolerance on adversarial seeds. The
    # change is intentional (one less `summed_f32` intermediate); end-to-end
    # training shows no loss/MFU regression.
    (torch.bfloat16, 1.5e-1),
])
def test_forward_parity(dtype, atol):
    torch.manual_seed(0)
    B, T, K, N_fc = 2, 32, 128, 256
    x = torch.randn(B, T, K, dtype=dtype, device="cuda")
    W_fc = torch.randn(N_fc, K, dtype=dtype, device="cuda") * 0.1
    W_proj = torch.randn(K, N_fc, dtype=dtype, device="cuda") * 0.1

    y_ref = _reference(x, W_fc, W_proj)
    y_triton = fused_mlp(x, W_fc, W_proj)
    assert torch.allclose(y_ref, y_triton, atol=atol), \
        f"forward mismatch (max {(y_ref - y_triton).abs().max().item():.4f}, dtype={dtype})"


def test_backward_parity():
    torch.manual_seed(0)
    B, T, K, N_fc = 2, 32, 128, 256
    dtype = torch.float32  # IEEE path for bit-tight parity

    x0 = torch.randn(B, T, K, dtype=dtype, device="cuda")
    W_fc0 = torch.randn(N_fc, K, dtype=dtype, device="cuda") * 0.1
    W_proj0 = torch.randn(K, N_fc, dtype=dtype, device="cuda") * 0.1
    g = torch.randn(B, T, K, dtype=dtype, device="cuda")

    def _grads(use_triton):
        x = x0.clone().requires_grad_(True)
        Wfc = W_fc0.clone().requires_grad_(True)
        Wproj = W_proj0.clone().requires_grad_(True)
        y = (fused_mlp if use_triton else _reference)(x, Wfc, Wproj)
        y.backward(g)
        return x.grad, Wfc.grad, Wproj.grad

    ref = _grads(use_triton=False)
    got = _grads(use_triton=True)
    atol = 5e-3
    for name, r, g_ in zip(("x", "W_fc", "W_proj"), ref, got):
        max_diff = (r - g_).abs().max().item()
        assert torch.allclose(r, g_, atol=atol), \
            f"{name}.grad mismatch (max {max_diff:.4e})"


def test_fused_mlp_rejects_noncontiguous_inputs():
    B, T, K, N_fc = 2, 4, 16, 32
    x = torch.randn(B, T, K, dtype=torch.float32, device="cuda")
    W_fc = torch.randn(K, N_fc, dtype=torch.float32, device="cuda").t()
    W_proj = torch.randn(N_fc, K, dtype=torch.float32, device="cuda").t()

    assert not W_fc.is_contiguous()
    assert not W_proj.is_contiguous()

    with pytest.raises(AssertionError):
        fused_mlp(x, W_fc, W_proj.contiguous())

    with pytest.raises(AssertionError):
        fused_mlp(x, W_fc.contiguous(), W_proj)
