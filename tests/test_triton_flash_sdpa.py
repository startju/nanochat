"""Parity tests for Flash-style sliding-causal SDPA Triton kernels.

Reference: nanoops's SlidingWindowSDPA (Python chunked, math-equivalent).
"""

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("triton kernels require CUDA", allow_module_level=True)

from nanoops.functional import sliding_window_sdpa
from nanoops.triton_fused_attn_spda import _fused_attn_spda_bwd_op, _fused_attn_spda_fwd_op


def _fused_attn_spda_internal(q, k, v, window_size):
    out, _lse = _fused_attn_spda_fwd_op(q, k, v, window_size)
    return out


def _fused_attn_spda_internal_backward(q, k, v, do, window_size):
    out, lse = _fused_attn_spda_fwd_op(q, k, v, window_size)
    delta = torch.sum(out.float() * do.float(), dim=-1)
    return _fused_attn_spda_bwd_op(do, q, k, v, lse, delta, window_size)


@pytest.mark.parametrize("B,H,L,D,W", [
    (1, 2, 32, 16, 8),
    (2, 4, 64, 32, 16),
    (1, 8, 128, 64, 32),
])
def test_forward_parity_fp32(B, H, L, D, W):
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, dtype=torch.float32, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.float32, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.float32, device="cuda")

    o_ref = sliding_window_sdpa(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        W,
    ).transpose(1, 2)
    o_triton = _fused_attn_spda_internal(q, k, v, W)
    max_diff = (o_ref - o_triton).abs().max().item()
    assert torch.allclose(o_ref, o_triton, atol=3e-3), \
        f"forward mismatch (max {max_diff:.4e}, B={B} H={H} L={L} D={D} W={W})"


@pytest.mark.parametrize("B,H,L,D,W", [
    (1, 2, 32, 16, 8),
    (2, 4, 64, 32, 16),
])
def test_backward_parity_fp32(B, H, L, D, W):
    torch.manual_seed(0)
    q0 = torch.randn(B, L, H, D, dtype=torch.float32, device="cuda")
    k0 = torch.randn(B, L, H, D, dtype=torch.float32, device="cuda")
    v0 = torch.randn(B, L, H, D, dtype=torch.float32, device="cuda")
    g = torch.randn(B, L, H, D, dtype=torch.float32, device="cuda")

    # Reference
    q1, k1, v1 = q0.clone().requires_grad_(True), k0.clone().requires_grad_(True), v0.clone().requires_grad_(True)
    sliding_window_sdpa(
        q1.transpose(1, 2),
        k1.transpose(1, 2),
        v1.transpose(1, 2),
        W,
    ).transpose(1, 2).backward(g)

    # Triton backward-from-delta path.
    q2, k2, v2 = q0.clone(), k0.clone(), v0.clone()
    dq, dk, dv = _fused_attn_spda_internal_backward(q2, k2, v2, g, W)

    atol = 8e-3
    for name, ref, got in [
        ("q.grad", q1.grad, dq),
        ("k.grad", k1.grad, dk),
        ("v.grad", v1.grad, dv),
    ]:
        max_diff = (ref - got).abs().max().item()
        assert torch.allclose(ref, got, atol=atol), \
            f"{name} mismatch (max {max_diff:.4e}, B={B} H={H} L={L} D={D} W={W})"


@pytest.mark.parametrize("B,Hq,Hkv,L,D,W", [
    (1, 4, 2, 32, 16, 8),
    (2, 8, 2, 64, 32, 16),
])
def test_forward_parity_gqa_fp32(B, Hq, Hkv, L, D, W):
    torch.manual_seed(0)
    q = torch.randn(B, L, Hq, D, dtype=torch.float32, device="cuda")
    k = torch.randn(B, L, Hkv, D, dtype=torch.float32, device="cuda")
    v = torch.randn(B, L, Hkv, D, dtype=torch.float32, device="cuda")

    o_ref = sliding_window_sdpa(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        W,
        enable_gqa=True,
    ).transpose(1, 2)
    o_triton = _fused_attn_spda_internal(q, k, v, W)
    max_diff = (o_ref - o_triton).abs().max().item()
    assert torch.allclose(o_ref, o_triton, atol=3e-3), \
        f"forward GQA mismatch (max {max_diff:.4e}, B={B} Hq={Hq} Hkv={Hkv} L={L} D={D} W={W})"


@pytest.mark.parametrize("B,Hq,Hkv,L,D,W", [
    (1, 4, 2, 32, 16, 8),
    (2, 8, 2, 64, 32, 16),
])
def test_backward_parity_gqa_fp32(B, Hq, Hkv, L, D, W):
    torch.manual_seed(0)
    q0 = torch.randn(B, L, Hq, D, dtype=torch.float32, device="cuda")
    k0 = torch.randn(B, L, Hkv, D, dtype=torch.float32, device="cuda")
    v0 = torch.randn(B, L, Hkv, D, dtype=torch.float32, device="cuda")
    g = torch.randn(B, L, Hq, D, dtype=torch.float32, device="cuda")

    q1, k1, v1 = q0.clone().requires_grad_(True), k0.clone().requires_grad_(True), v0.clone().requires_grad_(True)
    sliding_window_sdpa(
        q1.transpose(1, 2),
        k1.transpose(1, 2),
        v1.transpose(1, 2),
        W,
        enable_gqa=True,
    ).transpose(1, 2).backward(g)

    q2, k2, v2 = q0.clone(), k0.clone(), v0.clone()
    dq, dk, dv = _fused_attn_spda_internal_backward(q2, k2, v2, g, W)

    atol = 8e-3
    for name, ref, got in [
        ("q.grad", q1.grad, dq),
        ("k.grad", k1.grad, dk),
        ("v.grad", v1.grad, dv),
    ]:
        max_diff = (ref - got).abs().max().item()
        assert torch.allclose(ref, got, atol=atol), \
            f"{name} GQA mismatch (max {max_diff:.4e}, B={B} Hq={Hq} Hkv={Hkv} L={L} D={D} W={W})"
