"""Parity tests for the small Triton kernels covering the
remaining elementwise pieces of nanchat's CausalSelfAttention forward.
"""

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("triton kernels require CUDA", allow_module_level=True)

from nanoops.triton_fused_attn_qkv import (
    _fused_attn_qkv_projection_fwd_op,
    _fused_attn_qkv_projection_bwd_impl,
    fused_attn_qkv_projection as _fused_attn_qkv_projection,
)
from nanoops.triton_fused_attn_spda_and_output import fused_attn_spda_and_output
from nanoops.functional import sliding_window_sdpa


# ─────────────────────────────────────────────────────────────────────
# fused_attn_qkv_projection:
#   RMSNorm(x) @ W_qkv.T, with Q/K rotary + QK RMSNorm + scale fused before writeback
# ─────────────────────────────────────────────────────────────────────

def _fused_attn_qkv_projection_ref(
    x,
    ve_ids,
    ve_weight,
    ve_gate_channels,
    ve_gate_weight,
    q_weight,
    k_weight,
    v_weight,
    cos,
    sin,
    n_head,
    n_kv_head,
    head_dim,
    scale,
    eps,
):
    prefix_shape = x.shape[:-1]
    C = x.shape[-1]
    M = x.numel() // C
    x_2d = x.reshape(M, C)
    x_rms_inv = torch.rsqrt((x_2d.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    x_hat = x_2d * x_rms_inv.to(x.dtype)
    q = (x_hat @ q_weight.t()).view(-1, n_head, head_dim)
    k = (x_hat @ k_weight.t()).view(-1, n_kv_head, head_dim)
    v = (x_hat @ v_weight.t()).view(-1, n_kv_head, head_dim)
    if ve_ids is not None or ve_weight is not None:
        assert ve_gate_weight is not None
        assert ve_ids is not None and ve_weight is not None
        ve = ve_weight[ve_ids.reshape(-1)].reshape(M, n_kv_head, head_dim)
        gate = 3.0 * torch.sigmoid(
            x_hat[:, :ve_gate_channels].float() @ ve_gate_weight.float().t()
        )
        v = v + gate.to(v.dtype).view(M, n_kv_head, 1) * ve

    half = head_dim // 2
    assert cos.ndim == 4 and sin.ndim == 4
    assert cos.shape == sin.shape
    assert cos.shape[0] == 1 and cos.shape[2] == 1 and cos.shape[-1] == half
    T = cos.shape[1]
    assert M % T == 0
    B = M // T
    cos_b = cos.expand(B, T, 1, half).reshape(M, 1, half)
    sin_b = sin.expand(B, T, 1, half).reshape(M, 1, half)

    def _rot_norm_scale(qk):
        lo, hi = qk[..., :half], qk[..., half:]
        rot_lo = lo * cos_b + hi * sin_b
        rot_hi = -lo * sin_b + hi * cos_b
        rotated = torch.cat([rot_lo, rot_hi], dim=-1)
        qk_rms_inv = torch.rsqrt((rotated.float() ** 2).mean(dim=-1, keepdim=True) + eps)
        return rotated * qk_rms_inv.to(rotated.dtype) * scale

    return (
        _rot_norm_scale(q).view(*prefix_shape, n_head, head_dim),
        _rot_norm_scale(k).view(*prefix_shape, n_kv_head, head_dim),
        v.view(*prefix_shape, n_kv_head, head_dim),
    )


def fused_attn_qkv_projection_identity_mix(
    x,
    ve_ids,
    ve_weight,
    ve_gate_channels,
    ve_gate_weight,
    q_weight,
    k_weight,
    v_weight,
    cos,
    sin,
    n_head,
    n_kv_head,
    head_dim,
    scale,
    eps,
):
    """Exercise the residual-mix-only Triton path with identity mixing."""
    x0 = torch.zeros_like(x)
    resid_scale = torch.ones((), dtype=x.dtype, device=x.device)
    x0_scale = torch.zeros((), dtype=x.dtype, device=x.device)
    q, k, v, _x_mix = _fused_attn_qkv_projection(
        x,
        x0,
        resid_scale,
        x0_scale,
        ve_ids,
        ve_weight,
        ve_gate_channels,
        ve_gate_weight,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        scale,
        eps,
    )
    return q, k, v


def test_fused_attn_qkv_projection_forward():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 16, 128, 4, 2, 128
    x = torch.randn(B, T, K, dtype=torch.float32, device="cuda")
    q_weight = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    k_weight = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    v_weight = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    q_ref, k_ref, v_ref = _fused_attn_qkv_projection_ref(
        x,
        None,
        None,
        1,
        None,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )
    q_tri, k_tri, v_tri = fused_attn_qkv_projection_identity_mix(
        x,
        None,
        None,
        1,
        None,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )

    for name, ref, got in [
        ("q", q_ref, q_tri),
        ("k", k_ref, k_tri),
        ("v", v_ref, v_tri),
    ]:
        assert torch.allclose(ref, got, atol=5e-3), \
            f"{name} max diff {(ref - got).abs().max():.4e}"


def test_fused_attn_qkv_projection_broadcast_rotary_table():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 16, 128, 4, 2, 128
    M = B * T
    x = torch.randn(B, T, K, dtype=torch.float32, device="cuda")
    q_weight = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    k_weight = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    v_weight = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    q_ref, k_ref, v_ref = _fused_attn_qkv_projection_ref(
        x,
        None,
        None,
        1,
        None,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )
    q_tri, k_tri, v_tri = fused_attn_qkv_projection_identity_mix(
        x,
        None,
        None,
        1,
        None,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )

    for name, ref, got in [
        ("q", q_ref, q_tri),
        ("k", k_ref, k_tri),
        ("v", v_ref, v_tri),
    ]:
        assert torch.allclose(ref, got, atol=5e-3), \
            f"{name} max diff {(ref - got).abs().max():.4e}"


def test_fused_attn_qkv_projection_value_embedding_lookup_forward_backward():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 8, 128, 2, 1, 128
    vocab_size = 11
    ve_gate_channels = 12
    x0 = torch.randn(B, T, K, dtype=torch.float32, device="cuda")
    qw0 = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    kw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    vw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    ve_w0 = torch.randn(vocab_size, n_kv_head * head_dim, dtype=torch.float32, device="cuda")
    gw0 = torch.randn(n_kv_head, ve_gate_channels, dtype=torch.float32, device="cuda") * 0.1
    ve_ids = torch.randint(0, vocab_size, (B, T), dtype=torch.long, device="cuda")
    ve_ids[0, :4] = ve_ids[1, :4]
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    qg = torch.randn(B, T, n_head, head_dim, dtype=torch.float32, device="cuda")
    kg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")
    vg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")

    x1, qw1, kw1, vw1, ve_w1, gw1 = (
        t.clone().requires_grad_() for t in (x0, qw0, kw0, vw0, ve_w0, gw0)
    )
    q1, k1, v1 = _fused_attn_qkv_projection_ref(
        x1,
        ve_ids,
        ve_w1,
        ve_gate_channels,
        gw1,
        qw1,
        kw1,
        vw1,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )
    torch.autograd.backward((q1, k1, v1), (qg, kg, vg))

    x2, qw2, kw2, vw2, ve_w2, gw2 = (
        t.clone().requires_grad_() for t in (x0, qw0, kw0, vw0, ve_w0, gw0)
    )
    q2, k2, v2 = fused_attn_qkv_projection_identity_mix(
        x2,
        ve_ids,
        ve_w2,
        ve_gate_channels,
        gw2,
        qw2,
        kw2,
        vw2,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )
    torch.autograd.backward((q2, k2, v2), (qg, kg, vg))

    for name, ref, got, atol in [
        ("q", q1, q2, 5e-3),
        ("k", k1, k2, 5e-3),
        ("v", v1, v2, 5e-3),
        ("x.grad", x1.grad, x2.grad, 2e-2),
        ("q_weight.grad", qw1.grad, qw2.grad, 4e-2),
        ("k_weight.grad", kw1.grad, kw2.grad, 4e-2),
        ("v_weight.grad", vw1.grad, vw2.grad, 4e-2),
        ("ve_weight.grad", ve_w1.grad, ve_w2.grad, 1e-2),
        ("ve_gate_weight.grad", gw1.grad, gw2.grad, 1e-2),
    ]:
        assert torch.allclose(ref, got, atol=atol), \
            f"{name} max diff {(ref - got).abs().max():.4e}"


def test_fused_attn_qkv_projection_backward():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 8, 128, 2, 1, 128
    x0 = torch.randn(B, T, K, dtype=torch.float32, device="cuda")
    qw0 = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    kw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    vw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    qg = torch.randn(B, T, n_head, head_dim, dtype=torch.float32, device="cuda")
    kg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")
    vg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")

    x1, qw1, kw1, vw1 = (t.clone().requires_grad_() for t in (x0, qw0, kw0, vw0))
    q1, k1, v1 = _fused_attn_qkv_projection_ref(
        x1, None, None, 1, None, qw1, kw1, vw1, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q1, k1, v1), (qg, kg, vg))

    x2, qw2, kw2, vw2 = (t.clone().requires_grad_() for t in (x0, qw0, kw0, vw0))
    q2, k2, v2 = fused_attn_qkv_projection_identity_mix(
        x2, None, None, 1, None, qw2, kw2, vw2, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q2, k2, v2), (qg, kg, vg))

    for name, ref, got in [
        ("x", x1.grad, x2.grad),
        ("q_weight", qw1.grad, qw2.grad),
        ("k_weight", kw1.grad, kw2.grad),
        ("v_weight", vw1.grad, vw2.grad),
    ]:
        assert torch.allclose(ref, got, atol=4e-2), \
            f"{name}.grad max diff {(ref - got).abs().max():.4e}"


def test_fused_attn_qkv_projection_backward_broadcast_rotary_table():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 8, 128, 2, 1, 128
    M = B * T
    x0 = torch.randn(B, T, K, dtype=torch.float32, device="cuda")
    qw0 = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    kw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    vw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    qg = torch.randn(B, T, n_head, head_dim, dtype=torch.float32, device="cuda")
    kg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")
    vg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")

    x1, qw1, kw1, vw1 = (t.clone().requires_grad_() for t in (x0, qw0, kw0, vw0))
    q1, k1, v1 = _fused_attn_qkv_projection_ref(
        x1, None, None, 1, None, qw1, kw1, vw1, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q1, k1, v1), (qg, kg, vg))

    x2, qw2, kw2, vw2 = (t.clone().requires_grad_() for t in (x0, qw0, kw0, vw0))
    q2, k2, v2 = fused_attn_qkv_projection_identity_mix(
        x2, None, None, 1, None, qw2, kw2, vw2, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q2, k2, v2), (qg, kg, vg))

    for name, ref, got in [
        ("x", x1.grad, x2.grad),
        ("q_weight", qw1.grad, qw2.grad),
        ("k_weight", kw1.grad, kw2.grad),
        ("v_weight", vw1.grad, vw2.grad),
    ]:
        assert torch.allclose(ref, got, atol=4e-2), \
            f"{name}.grad max diff {(ref - got).abs().max():.4e}"


def test_fused_attn_qkv_projection_backward_bf16_smoke():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 8, 128, 2, 1, 128
    x = torch.randn(B, T, K, dtype=torch.bfloat16, device="cuda").requires_grad_()
    q_weight = (
        torch.randn(n_head * head_dim, K, dtype=torch.bfloat16, device="cuda") * 0.1
    ).requires_grad_()
    k_weight = (
        torch.randn(n_kv_head * head_dim, K, dtype=torch.bfloat16, device="cuda") * 0.1
    ).requires_grad_()
    v_weight = (
        torch.randn(n_kv_head * head_dim, K, dtype=torch.bfloat16, device="cuda") * 0.1
    ).requires_grad_()
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.bfloat16, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.bfloat16, device="cuda")

    q, k, v = fused_attn_qkv_projection_identity_mix(
        x,
        None,
        None,
        1,
        None,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
    )
    (q.float().sum() + k.float().sum() + v.float().sum()).backward()

    for name, grad in [
        ("x", x.grad),
        ("q_weight", q_weight.grad),
        ("k_weight", k_weight.grad),
        ("v_weight", v_weight.grad),
    ]:
        assert torch.isfinite(grad.float()).all(), f"{name}.grad contains non-finite values"


def test_fused_attn_qkv_projection_head64_backward():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 8, 128, 2, 1, 64
    x0 = torch.randn(B, T, K, dtype=torch.float32, device="cuda")
    qw0 = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    kw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    vw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    qg = torch.randn(B, T, n_head, head_dim, dtype=torch.float32, device="cuda")
    kg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")
    vg = torch.randn(B, T, n_kv_head, head_dim, dtype=torch.float32, device="cuda")

    x1, qw1, kw1, vw1 = (t.clone().requires_grad_() for t in (x0, qw0, kw0, vw0))
    q1, k1, v1 = _fused_attn_qkv_projection_ref(
        x1, None, None, 1, None, qw1, kw1, vw1, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q1, k1, v1), (qg, kg, vg))

    x2, qw2, kw2, vw2 = (t.clone().requires_grad_() for t in (x0, qw0, kw0, vw0))
    q2, k2, v2 = fused_attn_qkv_projection_identity_mix(
        x2, None, None, 1, None, qw2, kw2, vw2, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q2, k2, v2), (qg, kg, vg))

    for name, ref, got in [
        ("x", x1.grad, x2.grad),
        ("q_weight", qw1.grad, qw2.grad),
        ("k_weight", kw1.grad, kw2.grad),
        ("v_weight", vw1.grad, vw2.grad),
    ]:
        assert torch.allclose(ref, got, atol=4e-2), \
            f"{name}.grad max diff {(ref - got).abs().max():.4e}"


def test_fused_attn_qkv_projection_backward_formula():
    torch.manual_seed(0)
    M, K, n_head, n_kv_head, head_dim = 16, 128, 2, 1, 128
    x0 = torch.randn(M, K, dtype=torch.float32, device="cuda")
    qw0 = torch.randn(n_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    kw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    vw0 = torch.randn(n_kv_head * head_dim, K, dtype=torch.float32, device="cuda") * 0.1
    cos = torch.randn(1, M, 1, head_dim // 2, dtype=torch.float32, device="cuda")
    sin = torch.randn(1, M, 1, head_dim // 2, dtype=torch.float32, device="cuda")

    qg = torch.randn(M, n_head, head_dim, dtype=torch.float32, device="cuda")
    kg = torch.randn(M, n_kv_head, head_dim, dtype=torch.float32, device="cuda")
    vg = torch.randn(M, n_kv_head, head_dim, dtype=torch.float32, device="cuda")

    x1, qw1, kw1, vw1 = (t.clone().requires_grad_() for t in (x0, qw0, kw0, vw0))
    q1, k1, v1 = _fused_attn_qkv_projection_ref(
        x1, None, None, 1, None, qw1, kw1, vw1, cos, sin, n_head, n_kv_head, head_dim, 1.2, 1e-6
    )
    torch.autograd.backward((q1, k1, v1), (qg, kg, vg))

    ident_x0 = torch.zeros_like(x0).view(1, M, K)
    resid_scale = torch.ones((), dtype=x0.dtype, device=x0.device)
    x0_scale = torch.zeros((), dtype=x0.dtype, device=x0.device)
    q_saved, k_saved, _v_saved, _x_mix, rms_inv, qk_rms_inv = (
        _fused_attn_qkv_projection_fwd_op(
            x0.view(1, M, K),
            ident_x0,
            resid_scale,
            x0_scale,
            None,
            None,
            1,
            None,
            qw0,
            kw0,
            vw0,
            cos,
            sin,
            n_head,
            n_kv_head,
            head_dim,
            1.2,
            1e-6,
        )
    )
    (
        dx,
        _dx0,
        _d_resid_scale,
        _d_x0_scale,
        d_ve_weight,
        d_ve_gate_weight,
        d_q_weight,
        d_k_weight,
        d_v_weight,
    ) = _fused_attn_qkv_projection_bwd_impl(
        qg.view(1, M, n_head, head_dim),
        kg.view(1, M, n_kv_head, head_dim),
        vg.view(1, M, n_kv_head, head_dim),
        None,
        None,
        1,
        None,
        qw0,
        kw0,
        vw0,
        cos,
        sin,
        rms_inv,
        q_saved,
        k_saved,
        qk_rms_inv,
        n_head,
        n_kv_head,
        head_dim,
        1.2,
        1e-6,
        grad_x_mix=torch.zeros_like(x0).view(1, M, K),
        x_base=x0.view(1, M, K),
        x0=ident_x0,
        resid_scale=resid_scale,
        x0_scale=x0_scale,
    )
    assert d_ve_weight is None
    assert d_ve_gate_weight is None

    for name, ref, got in [
        ("x", x1.grad, dx.view(M, K)),
        ("q_weight", qw1.grad, d_q_weight),
        ("k_weight", kw1.grad, d_k_weight),
        ("v_weight", vw1.grad, d_v_weight),
    ]:
        assert torch.allclose(ref, got, atol=4e-2), \
            f"{name}.grad max diff {(ref - got).abs().max():.4e}"


def test_fused_attn_qkv_projection_residual_mix_backward():
    torch.manual_seed(0)
    B, T, K, n_head, n_kv_head, head_dim = 2, 8, 128, 2, 1, 64
    dtype = torch.float32
    x0_base = torch.randn(B, T, K, dtype=dtype, device="cuda")
    x_base = torch.randn(B, T, K, dtype=dtype, device="cuda")
    qw_base = torch.randn(n_head * head_dim, K, dtype=dtype, device="cuda") * 0.1
    kw_base = torch.randn(n_kv_head * head_dim, K, dtype=dtype, device="cuda") * 0.1
    vw_base = torch.randn(n_kv_head * head_dim, K, dtype=dtype, device="cuda") * 0.1
    resid_scale_base = torch.tensor(1.1, dtype=dtype, device="cuda")
    x0_scale_base = torch.tensor(0.2, dtype=dtype, device="cuda")
    cos = torch.randn(1, T, 1, head_dim // 2, dtype=dtype, device="cuda")
    sin = torch.randn(1, T, 1, head_dim // 2, dtype=dtype, device="cuda")
    qg = torch.randn(B, T, n_head, head_dim, dtype=dtype, device="cuda")
    kg = torch.randn(B, T, n_kv_head, head_dim, dtype=dtype, device="cuda")
    vg = torch.randn(B, T, n_kv_head, head_dim, dtype=dtype, device="cuda")
    xmix_g = torch.randn(B, T, K, dtype=dtype, device="cuda")

    def _run(use_triton):
        x = x_base.clone().requires_grad_()
        x0 = x0_base.clone().requires_grad_()
        qw = qw_base.clone().requires_grad_()
        kw = kw_base.clone().requires_grad_()
        vw = vw_base.clone().requires_grad_()
        resid_scale = resid_scale_base.clone().requires_grad_()
        x0_scale = x0_scale_base.clone().requires_grad_()
        if use_triton:
            q, k, v, x_mix = _fused_attn_qkv_projection(
                x,
                x0,
                resid_scale,
                x0_scale,
                None,
                None,
                1,
                None,
                qw,
                kw,
                vw,
                cos,
                sin,
                n_head,
                n_kv_head,
                head_dim,
                1.2,
                1e-6,
            )
        else:
            x_mix = resid_scale * x + x0_scale * x0
            q, k, v = _fused_attn_qkv_projection_ref(
                x_mix,
                None,
                None,
                1,
                None,
                qw,
                kw,
                vw,
                cos,
                sin,
                n_head,
                n_kv_head,
                head_dim,
                1.2,
                1e-6,
            )
        torch.autograd.backward((q, k, v, x_mix), (qg, kg, vg, xmix_g))
        return (
            x.grad,
            x0.grad,
            resid_scale.grad,
            x0_scale.grad,
            qw.grad,
            kw.grad,
            vw.grad,
        )

    ref = _run(use_triton=False)
    got = _run(use_triton=True)
    for name, ref_grad, got_grad in zip(
        ("x", "x0", "resid_scale", "x0_scale", "q_weight", "k_weight", "v_weight"),
        ref,
        got,
    ):
        max_diff = (ref_grad - got_grad).abs().max().item()
        assert torch.allclose(ref_grad, got_grad, atol=6e-2), \
            f"{name}.grad max diff {max_diff:.4e}"


def test_fused_attn_spda_and_output_backward():
    torch.manual_seed(0)
    B, T, H, D, C, W = 1, 32, 4, 16, 64, 8
    q0 = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    k0 = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    v0 = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    w0 = (torch.randn(C, H * D, dtype=torch.bfloat16, device="cuda") * 0.1).to(torch.bfloat16)
    r0 = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(B, T, C, dtype=torch.bfloat16, device="cuda")

    q1, k1, v1 = q0.clone().requires_grad_(), k0.clone().requires_grad_(), v0.clone().requires_grad_()
    w1, r1 = w0.clone().requires_grad_(), r0.clone().requires_grad_()
    attn_ref = sliding_window_sdpa(
        q1.transpose(1, 2),
        k1.transpose(1, 2),
        v1.transpose(1, 2),
        W,
    ).transpose(1, 2)
    y_ref = torch.addmm(
        r1.view(B * T, C),
        attn_ref.contiguous().view(B * T, H * D),
        w1.t(),
    ).view(B, T, C)
    y_ref.backward(g)

    q2, k2, v2 = q0.clone().requires_grad_(), k0.clone().requires_grad_(), v0.clone().requires_grad_()
    w2, r2 = w0.clone().requires_grad_(), r0.clone().requires_grad_()
    fused_attn_spda_and_output(q2, k2, v2, w2, r2, W).backward(g)

    for name, ref, got, atol in [
        ("q", q1.grad, q2.grad, 6e-2),
        ("k", k1.grad, k2.grad, 6e-2),
        ("v", v1.grad, v2.grad, 6e-2),
        ("w", w1.grad, w2.grad, 1.5e-1),
        ("r", r1.grad, r2.grad, 0.0),
    ]:
        max_diff = (ref - got).abs().max().item()
        assert torch.allclose(ref, got, atol=atol), \
            f"{name}.grad max diff {max_diff:.4e}"
