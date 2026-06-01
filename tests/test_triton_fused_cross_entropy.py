"""Parity tests for fused lm-head + softcap + cross entropy."""

import pytest
import torch
import torch.nn.functional as F

from nanoops.triton_fused_cross_entropy import fused_cross_entropy


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    vocab_size: int,
    softcap: float,
    reduction: str,
    *,
    x_backout: torch.Tensor | None = None,
    backout_scale: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1])
    if x_backout is not None:
        assert backout_scale is not None
        x_2d = x_2d - backout_scale.to(dtype=x.dtype) * x_backout.reshape(-1, x.shape[-1])
        x_2d = F.rms_norm(x_2d, (x.shape[-1],), eps=eps)
    logits = x_2d @ weight[:vocab_size].to(dtype=x.dtype).t()
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)
    loss = F.cross_entropy(
        logits,
        target.reshape(-1),
        ignore_index=-1,
        reduction=reduction,
    )
    return loss.view_as(target) if reduction == "none" else loss


def test_fused_cross_entropy_cpu_fallback():
    torch.manual_seed(0)
    B, T, K, V, V_PAD = 2, 3, 8, 11, 16
    x = torch.randn(B, T, K, dtype=torch.float32).contiguous()
    weight = torch.randn(V_PAD, K, dtype=torch.float32).contiguous() * 0.1
    target = torch.tensor([[0, 3, -1], [10, 2, 4]], dtype=torch.long).contiguous()

    ref = _reference(x, weight, target, V, 15.0, "none")
    got = fused_cross_entropy(x, weight, target, V, reduction="none")
    assert torch.allclose(ref, got)


def test_fused_cross_entropy_final_tail_cpu_fallback():
    torch.manual_seed(0)
    B, T, K, V, V_PAD = 2, 3, 8, 11, 16
    x = torch.randn(B, T, K, dtype=torch.float32).contiguous()
    x_backout = torch.randn(B, T, K, dtype=torch.float32).contiguous()
    backout_scale = torch.tensor([0.2], dtype=torch.float32)
    weight = torch.randn(V_PAD, K, dtype=torch.float32).contiguous() * 0.1
    target = torch.tensor([[0, 3, -1], [10, 2, 4]], dtype=torch.long).contiguous()

    ref = _reference(
        x,
        weight,
        target,
        V,
        15.0,
        "none",
        x_backout=x_backout,
        backout_scale=backout_scale,
    )
    got = fused_cross_entropy(
        x,
        weight,
        target,
        V,
        reduction="none",
        x_backout=x_backout,
        backout_scale=backout_scale,
    )
    assert torch.allclose(ref, got)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernels require CUDA")
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_fused_cross_entropy_cuda_forward(reduction):
    torch.manual_seed(0)
    B, T, K, V, V_PAD = 2, 5, 32, 67, 80
    x = torch.randn(B, T, K, dtype=torch.bfloat16, device="cuda").contiguous()
    weight = (torch.randn(V_PAD, K, dtype=torch.float32, device="cuda") * 0.02).contiguous()
    target = torch.randint(0, V, (B, T), dtype=torch.long, device="cuda").contiguous()
    target[0, 1] = -1

    ref = _reference(x, weight, target, V, 15.0, reduction)
    got = fused_cross_entropy(x, weight, target, V, reduction=reduction)
    assert torch.allclose(ref, got, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernels require CUDA")
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_fused_cross_entropy_final_tail_cuda_forward(reduction):
    torch.manual_seed(0)
    B, T, K, V, V_PAD = 2, 5, 32, 67, 80
    x = torch.randn(B, T, K, dtype=torch.bfloat16, device="cuda").contiguous()
    x_backout = torch.randn(B, T, K, dtype=torch.bfloat16, device="cuda").contiguous()
    backout_scale = torch.tensor([0.2], dtype=torch.float32, device="cuda")
    weight = (torch.randn(V_PAD, K, dtype=torch.float32, device="cuda") * 0.02).contiguous()
    target = torch.randint(0, V, (B, T), dtype=torch.long, device="cuda").contiguous()
    target[0, 1] = -1

    ref = _reference(
        x,
        weight,
        target,
        V,
        15.0,
        reduction,
        x_backout=x_backout,
        backout_scale=backout_scale,
    )
    got = fused_cross_entropy(
        x,
        weight,
        target,
        V,
        reduction=reduction,
        x_backout=x_backout,
        backout_scale=backout_scale,
    )
    assert torch.allclose(ref, got, atol=4e-2, rtol=4e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernels require CUDA")
def test_fused_cross_entropy_cuda_backward():
    torch.manual_seed(0)
    B, T, K, V, V_PAD = 2, 5, 32, 67, 80
    x0 = torch.randn(B, T, K, dtype=torch.bfloat16, device="cuda")
    weight0 = torch.randn(V_PAD, K, dtype=torch.float32, device="cuda") * 0.02
    target = torch.randint(0, V, (B, T), dtype=torch.long, device="cuda").contiguous()
    target[0, 1] = -1

    def _grads(use_triton: bool):
        x = x0.clone().contiguous().requires_grad_(True)
        weight = weight0.clone().contiguous().requires_grad_(True)
        if use_triton:
            loss = fused_cross_entropy(x, weight, target, V, reduction="mean")
        else:
            loss = _reference(x, weight, target, V, 15.0, "mean")
        loss.backward()
        return loss.detach(), x.grad, weight.grad

    loss_ref, dx_ref, dw_ref = _grads(use_triton=False)
    loss_tri, dx_tri, dw_tri = _grads(use_triton=True)
    assert torch.allclose(loss_ref, loss_tri, atol=3e-2, rtol=3e-2)
    assert torch.allclose(dx_ref, dx_tri, atol=4e-2, rtol=4e-2)
    assert torch.allclose(dw_ref[:V], dw_tri[:V], atol=5e-2, rtol=5e-2)
    assert torch.count_nonzero(dw_tri[V:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernels require CUDA")
def test_fused_cross_entropy_final_tail_cuda_backward():
    torch.manual_seed(0)
    B, T, K, V, V_PAD = 2, 5, 32, 67, 80
    x0 = torch.randn(B, T, K, dtype=torch.bfloat16, device="cuda")
    x_backout0 = torch.randn(B, T, K, dtype=torch.bfloat16, device="cuda")
    backout_scale0 = torch.tensor([0.2], dtype=torch.float32, device="cuda")
    weight0 = torch.randn(V_PAD, K, dtype=torch.float32, device="cuda") * 0.02
    target = torch.randint(0, V, (B, T), dtype=torch.long, device="cuda").contiguous()
    target[0, 1] = -1

    def _grads(use_triton: bool):
        x = x0.clone().contiguous().requires_grad_(True)
        x_backout = x_backout0.clone().contiguous().requires_grad_(True)
        backout_scale = backout_scale0.clone().contiguous().requires_grad_(True)
        weight = weight0.clone().contiguous().requires_grad_(True)
        if use_triton:
            loss = fused_cross_entropy(
                x,
                weight,
                target,
                V,
                reduction="mean",
                x_backout=x_backout,
                backout_scale=backout_scale,
            )
        else:
            loss = _reference(
                x,
                weight,
                target,
                V,
                15.0,
                "mean",
                x_backout=x_backout,
                backout_scale=backout_scale,
            )
        loss.backward()
        return loss.detach(), x.grad, x_backout.grad, backout_scale.grad, weight.grad

    loss_ref, dx_ref, dx_backout_ref, dscale_ref, dw_ref = _grads(use_triton=False)
    loss_tri, dx_tri, dx_backout_tri, dscale_tri, dw_tri = _grads(use_triton=True)
    assert torch.allclose(loss_ref, loss_tri, atol=4e-2, rtol=4e-2)
    assert torch.allclose(dx_ref, dx_tri, atol=6e-2, rtol=6e-2)
    assert torch.allclose(dx_backout_ref, dx_backout_tri, atol=6e-2, rtol=6e-2)
    assert torch.allclose(dscale_ref, dscale_tri, atol=8e-2, rtol=8e-2)
    assert torch.allclose(dw_ref[:V], dw_tri[:V], atol=6e-2, rtol=6e-2)
    assert torch.count_nonzero(dw_tri[V:]) == 0
