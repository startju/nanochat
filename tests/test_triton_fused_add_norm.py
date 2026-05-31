"""Parity tests for fused_add_norm: y = norm(x + residual)."""

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("triton kernels require CUDA", allow_module_level=True)

from nanoops.triton_fused_add_norm import _pick_tile_config, fused_add_norm


def _reference(x, residual, nw, eps=1e-6):
    summed = x + residual
    rms_inv = torch.rsqrt((summed.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    normed = summed.float() * rms_inv
    if nw is not None:
        normed = normed * nw.float()
    y = normed.to(x.dtype)
    return y, summed


# M chosen to cover: exact BLOCK_M=32 multiple, off-by-one (33),
# mid-block (50), and several blocks with tail (100 → 4 blocks, last has 4 rows).
# has_nw covers the affine vs no-affine RMSNorm path.
@pytest.mark.parametrize("M", [32, 33, 50, 100])
@pytest.mark.parametrize("has_nw", [True, False])
def test_fused_add_norm_forward(M, has_nw):
    torch.manual_seed(0)
    D = 128
    x = torch.randn(M, D, dtype=torch.float32, device="cuda")
    res = torch.randn(M, D, dtype=torch.float32, device="cuda")
    nw = torch.randn(D, dtype=torch.float32, device="cuda") if has_nw else None

    y_ref, s_ref = _reference(x, res, nw)
    y_triton, s_triton = fused_add_norm(x, res, nw)
    assert torch.allclose(y_ref, y_triton, atol=1e-3), \
        f"y max diff {(y_ref - y_triton).abs().max():.4e}"
    assert torch.allclose(s_ref, s_triton, atol=1e-6), \
        f"summed max diff {(s_ref - s_triton).abs().max():.4e}"


@pytest.mark.parametrize("M", [32, 33, 50, 100])
@pytest.mark.parametrize("has_nw", [True, False])
def test_fused_add_norm_backward(M, has_nw):
    torch.manual_seed(0)
    D = 128
    x0 = torch.randn(M, D, dtype=torch.float32, device="cuda")
    r0 = torch.randn(M, D, dtype=torch.float32, device="cuda")
    nw0 = torch.randn(D, dtype=torch.float32, device="cuda") if has_nw else None
    gy = torch.randn(M, D, dtype=torch.float32, device="cuda")
    gs = torch.randn(M, D, dtype=torch.float32, device="cuda")

    # Reference
    x1, r1 = (t.clone().requires_grad_() for t in (x0, r0))
    nw1 = nw0.clone().requires_grad_() if has_nw else None
    y_ref, s_ref = _reference(x1, r1, nw1)
    (y_ref * gy + s_ref * gs).sum().backward()

    # Triton
    x2, r2 = (t.clone().requires_grad_() for t in (x0, r0))
    nw2 = nw0.clone().requires_grad_() if has_nw else None
    y_triton, s_triton = fused_add_norm(x2, r2, nw2)
    (y_triton * gy + s_triton * gs).sum().backward()

    atol = 5e-3
    checks = [
        ("x.grad", x1.grad, x2.grad),
        ("res.grad", r1.grad, r2.grad),
    ]
    if has_nw:
        checks.append(("nw.grad", nw1.grad, nw2.grad))
    for name, ref, got in checks:
        assert torch.allclose(ref, got, atol=atol), \
            f"{name} max diff {(ref - got).abs().max():.4e}"


def test_fused_add_norm_backward_large_d_fallback():
    """D=32768 with affine weight exceeds the inline reg budget and uses fallback."""
    torch.manual_seed(0)
    M, D = 8, 32768
    BLOCK_D = triton.next_power_of_2(D)
    assert not _pick_tile_config(M, BLOCK_D, n_live_tiles=5).fits_reg_budget
    x0 = torch.randn(M, D, dtype=torch.float32, device="cuda")
    r0 = torch.randn(M, D, dtype=torch.float32, device="cuda")
    nw0 = torch.randn(D, dtype=torch.float32, device="cuda")
    gy = torch.randn(M, D, dtype=torch.float32, device="cuda")
    gs = torch.randn(M, D, dtype=torch.float32, device="cuda")

    x1, r1 = (t.clone().requires_grad_() for t in (x0, r0))
    nw1 = nw0.clone().requires_grad_()
    y_ref, s_ref = _reference(x1, r1, nw1)
    (y_ref * gy + s_ref * gs).sum().backward()

    x2, r2 = (t.clone().requires_grad_() for t in (x0, r0))
    nw2 = nw0.clone().requires_grad_()
    y_triton, s_triton = fused_add_norm(x2, r2, nw2)
    (y_triton * gy + s_triton * gs).sum().backward()

    for name, ref, got in (
        ("x.grad", x1.grad, x2.grad),
        ("res.grad", r1.grad, r2.grad),
        ("nw.grad", nw1.grad, nw2.grad),
    ):
        assert torch.allclose(ref, got, atol=5e-2), \
            f"{name} max diff {(ref - got).abs().max():.4e}"


def test_fused_add_norm_rejects_noncontiguous_norm_weight():
    D = 128
    x = torch.randn(4, D, dtype=torch.float32, device="cuda")
    res = torch.randn(4, D, dtype=torch.float32, device="cuda")
    nw_base = torch.randn(D, 2, dtype=torch.float32, device="cuda")
    nw = nw_base[:, 0]
    assert not nw.is_contiguous()

    with pytest.raises(AssertionError):
        fused_add_norm(x, res, nw)
