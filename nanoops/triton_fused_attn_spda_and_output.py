"""Attention output-side Triton kernels for nanoops.

Public entry: `fused_attn_spda_and_output`, the fused training attention tail:

    attn_out = sdpa(q, k, v)
    y        = residual + attn_out @ proj_weight.T

Public tensors use `(B, T, H, D)` for Q/K/V and `(B, T, C)` for residual.
Backward computes `d_attn_out = dy @ proj_weight`, `d_proj_weight =
dy.T @ attn_out`, residual's direct gradient `dy`, and the SDPA softmax
`delta = sum(attn_out * d_attn_out)` in the same pass that materializes
`d_attn_out`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.library import wrap_triton

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ─────────────────────────────────────────────────────────────────────
# Attention SDPA + output projection + residual.
#
# Forward math for flattened row m and output channel o:
#   y[m, o] = residual[m, o] + Σ_i attn_out[m, i] * proj_weight[o, i]
#
# Backward math:
#   d_attn_out[m, i]    = Σ_o dy[m, o] * proj_weight[o, i]
#   d_proj_weight[o, i] = Σ_m dy[m, o] * attn_out[m, i]
#   d_residual[m, o]    = dy[m, o]
#
# d_attn_out/delta and d_proj_weight use separate kernels because their
# reductions are over different axes. d_proj_weight is owned by `(D_out, D_in)`
# tiles and reduces over M internally, so it does not need atomics.
# ─────────────────────────────────────────────────────────────────────

if _HAS_TRITON:
    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_FWD_BLOCK_M = 128
    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_FWD_BLOCK_DOUT = 128
    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_FWD_BLOCK_DIN = 32
    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_FWD_NUM_WARPS = 8
    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_FWD_NUM_STAGES = 3

    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DATTN_BLOCK_M = 64
    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DATTN_BLOCK_DOUT = 32
    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DATTN_BLOCK_DIN = 128
    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DATTN_NUM_WARPS = 8
    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DATTN_NUM_STAGES = 3

    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DWEIGHT_BLOCK_M = 64
    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DWEIGHT_BLOCK_DOUT = 128
    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DWEIGHT_BLOCK_DIN = 64
    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DWEIGHT_NUM_WARPS = 8
    _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DWEIGHT_NUM_STAGES = 1

    @triton.jit
    def _fused_attn_spda_and_output_proj_fwd_kernel(
        attn_out_ptr,  # (M, D_in), activation dtype — in: attention output
        proj_w_ptr,  # (D_out, D_in), weight dtype — in: output projection weight
        residual_ptr,  # (M, D_out), activation dtype — in: residual stream
        y_ptr,  # (M, D_out), activation dtype — out
        M,  # int — row count after flattening leading dims
        D_OUT: tl.constexpr,  # projection output width
        D_IN: tl.constexpr,  # projection input width
        BLOCK_M: tl.constexpr,
        BLOCK_DOUT: tl.constexpr,
        BLOCK_DIN: tl.constexpr,
    ):
        """Fused `y = residual + attn_out @ proj_weight.T`.

        Standard tiled matmul; projection weight is cast to the activation
        dtype on load for tensor-core matmuls when the master weight is fp32.
        Residual is added after casting the accumulator back to output dtype.
        """
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_d * BLOCK_DOUT + tl.arange(0, BLOCK_DOUT)
        row_mask = rows < M
        col_mask = cols < D_OUT
        out_mask = row_mask[:, None] & col_mask[None, :]

        # Matmul-accumulate
        acc = tl.zeros((BLOCK_M, BLOCK_DOUT), dtype=tl.float32)
        for k_start in range(0, D_IN, BLOCK_DIN):
            ks = k_start + tl.arange(0, BLOCK_DIN)
            k_mask = ks < D_IN
            a_ptrs = attn_out_ptr + rows[:, None] * D_IN + ks[None, :]
            a = tl.load(
                a_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            pw_ptrs = proj_w_ptr + cols[:, None] * D_IN + ks[None, :]
            pw = tl.load(
                pw_ptrs,
                mask=col_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(a.dtype)
            acc += tl.dot(a, tl.trans(pw))

        # Add residual at the end in native dtype (saves a bf16→fp32
        # conversion on residual load + skips the final store cast).
        res_ptrs = residual_ptr + rows[:, None] * D_OUT + cols[None, :]
        residual = tl.load(res_ptrs, mask=out_mask, other=0.0)
        y = acc.to(y_ptr.dtype.element_ty) + residual

        y_ptrs = y_ptr + rows[:, None] * D_OUT + cols[None, :]
        tl.store(y_ptrs, y, mask=out_mask)

    @triton.jit
    def _fused_attn_spda_and_output_proj_dattn_delta_bwd_kernel(
        attn_out_ptr,  # (M, H_Q*D_HEAD), activation dtype — in: SDPA output
        proj_w_ptr,  # (D_out, H_Q*D_HEAD), weight dtype — in
        dy_ptr,  # (M, D_out), activation dtype — in
        d_attn_out_ptr,  # (M, H_Q*D_HEAD), activation dtype — out
        delta_ptr,  # (M, H_Q) fp32 — out: sum_d(attn_out * d_attn_out)
        M,  # int — flattened B*T rows
        H_Q: tl.constexpr,  # query/output head count
        D_HEAD: tl.constexpr,  # attention head width
        D_OUT: tl.constexpr,  # projection output width
        BLOCK_M: tl.constexpr,
        BLOCK_DOUT: tl.constexpr,
    ):
        """Compute `d_attn_out` for one attention head and its SDPA.

        This is the fused hand-off from output-projection backward to SDPA
        backward:

          d_attn_out[m, h, d] = sum_o dy[m, o] * proj_weight[o, h, d]
          delta[m, h]         = sum_d attn_out[m, h, d] * d_attn_out[m, h, d]

        The standalone SDPA backward normally launches a separate delta
        kernel. The combined attention op uses this kernel to avoid that pass.
        """
        pid_m = tl.program_id(0)
        hid = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D_HEAD)
        row_mask = rows < M
        head_mask = hid < H_Q
        din_cols = hid * D_HEAD + offs_d

        d_attn_acc = tl.zeros((BLOCK_M, D_HEAD), dtype=tl.float32)
        for dout_start in range(0, D_OUT, BLOCK_DOUT):
            dout_cols = dout_start + tl.arange(0, BLOCK_DOUT)
            dout_mask = dout_cols < D_OUT
            dy = tl.load(
                dy_ptr + rows[:, None] * D_OUT + dout_cols[None, :],
                mask=row_mask[:, None] & dout_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                proj_w_ptr + dout_cols[:, None] * (H_Q * D_HEAD) + din_cols[None, :],
                mask=dout_mask[:, None] & head_mask,
                other=0.0,
            ).to(dy.dtype)
            d_attn_acc += tl.dot(dy, w)

        d_attn = d_attn_acc.to(d_attn_out_ptr.dtype.element_ty)
        out_ptrs = d_attn_out_ptr + rows[:, None] * (H_Q * D_HEAD) + din_cols[None, :]
        tl.store(out_ptrs, d_attn, mask=row_mask[:, None] & head_mask)

        attn = tl.load(
            attn_out_ptr + rows[:, None] * (H_Q * D_HEAD) + din_cols[None, :],
            mask=row_mask[:, None] & head_mask,
            other=0.0,
        )
        delta = tl.sum(attn * d_attn, axis=1, dtype=tl.float32)
        tl.store(delta_ptr + rows * H_Q + hid, delta, mask=row_mask & head_mask)

    @triton.jit
    def _fused_attn_spda_and_output_proj_dweight_bwd_kernel(
        attn_out_ptr,  # (M, D_in), activation dtype — in: forward attention output
        dy_ptr,  # (M, D_out), activation dtype — in: output gradient
        d_proj_w_ptr,  # (D_out, D_in), proj_weight dtype — out
        M,  # int — row count after flattening leading dims
        D_OUT: tl.constexpr,  # projection output width
        D_IN: tl.constexpr,  # projection input width
        BLOCK_M: tl.constexpr,
        BLOCK_DOUT: tl.constexpr,
        BLOCK_DIN: tl.constexpr,
    ):
        """Compute `d_proj_weight = dy.T @ attn_out` without atomics."""
        pid_dout = tl.program_id(0)
        pid_din = tl.program_id(1)
        dout_cols = pid_dout * BLOCK_DOUT + tl.arange(0, BLOCK_DOUT)
        din_cols = pid_din * BLOCK_DIN + tl.arange(0, BLOCK_DIN)
        dout_mask = dout_cols < D_OUT
        din_mask = din_cols < D_IN

        d_w_acc = tl.zeros((BLOCK_DOUT, BLOCK_DIN), dtype=tl.float32)
        for m_start in range(0, M, BLOCK_M):
            rows = m_start + tl.arange(0, BLOCK_M)
            row_mask = rows < M
            dy = tl.load(
                dy_ptr + rows[:, None] * D_OUT + dout_cols[None, :],
                mask=row_mask[:, None] & dout_mask[None, :],
                other=0.0,
            )
            attn = tl.load(
                attn_out_ptr + rows[:, None] * D_IN + din_cols[None, :],
                mask=row_mask[:, None] & din_mask[None, :],
                other=0.0,
            ).to(dy.dtype)
            d_w_acc += tl.dot(tl.trans(dy), attn)

        tl.store(
            d_proj_w_ptr + dout_cols[:, None] * D_IN + din_cols[None, :],
            d_w_acc.to(d_proj_w_ptr.dtype.element_ty),
            mask=dout_mask[:, None] & din_mask[None, :],
        )


def _fused_attn_spda_and_output_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj_weight: torch.Tensor,
    residual: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run SDPA + output projection + residual forward.

    Returns `(y, attn_out, lse)`. The public wrapper returns only `y`; the
    auxiliary tensors are saved so backward can fuse output-projection delta
    production with SDPA backward.
    """
    from .triton_fused_attn_spda import _fused_attn_spda_fwd_op

    assert q.ndim == k.ndim == v.ndim == 4
    assert residual.ndim == 3
    B, T, n_head, head_dim = q.shape
    attn_out, lse = _fused_attn_spda_fwd_op(q, k, v, window_size)
    attn_flat = attn_out.contiguous().view(B * T, n_head * head_dim)
    residual_flat = residual.contiguous().view(B * T, -1)
    assert attn_flat.is_cuda and proj_weight.is_cuda and residual_flat.is_cuda
    assert attn_flat.is_contiguous() and proj_weight.is_contiguous()
    assert residual_flat.is_contiguous()
    M, d_in = attn_flat.shape
    M_res, d_out = residual_flat.shape
    d_out_w, d_in_w = proj_weight.shape
    assert M == M_res and d_in == d_in_w and d_out == d_out_w

    y_flat = torch.empty((M, d_out), dtype=attn_flat.dtype, device=attn_flat.device)
    proj_block_m = _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_FWD_BLOCK_M
    proj_block_dout = _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_FWD_BLOCK_DOUT
    proj_block_din = _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_FWD_BLOCK_DIN
    proj_grid = (triton.cdiv(M, proj_block_m), triton.cdiv(d_out, proj_block_dout))
    wrap_triton(_fused_attn_spda_and_output_proj_fwd_kernel)[proj_grid](
        attn_flat,
        proj_weight,
        residual_flat,
        y_flat,
        M,
        d_out,
        d_in,
        BLOCK_M=proj_block_m,
        BLOCK_DOUT=proj_block_dout,
        BLOCK_DIN=proj_block_din,
        num_warps=_FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_FWD_NUM_WARPS,
        num_stages=_FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_FWD_NUM_STAGES,
    )
    return y_flat.view(B, T, -1), attn_out, lse


def _fused_attn_spda_and_output_bwd_impl(
    dy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_out: torch.Tensor,
    lse: torch.Tensor,
    proj_weight: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run backward for SDPA + output projection + residual.

    Returns gradients for `(q, k, v, proj_weight)`. The residual gradient is
    the direct `dy` passthrough and is returned by the autograd callback.
    """
    from .triton_fused_attn_spda import _fused_attn_spda_bwd_op

    B, T, n_head, head_dim = q.shape
    dy_flat = dy.contiguous().view(B * T, -1)
    attn_flat = attn_out.contiguous().view(B * T, n_head * head_dim)
    assert dy_flat.is_cuda and attn_flat.is_cuda and proj_weight.is_cuda
    assert dy_flat.is_contiguous() and attn_flat.is_contiguous()
    assert proj_weight.is_contiguous()
    M, d_in = attn_flat.shape
    M_dy, d_out = dy_flat.shape
    d_out_w, d_in_w = proj_weight.shape
    assert M == M_dy and d_in == d_in_w and d_out == d_out_w
    assert d_in == n_head * head_dim

    d_attn_flat = torch.empty_like(attn_flat)
    delta_flat = torch.empty((M, n_head), dtype=torch.float32, device=attn_flat.device)
    dattn_block_m = _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DATTN_BLOCK_M
    dattn_block_dout = _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DATTN_BLOCK_DOUT
    dattn_grid = (triton.cdiv(M, dattn_block_m), n_head)
    wrap_triton(_fused_attn_spda_and_output_proj_dattn_delta_bwd_kernel)[dattn_grid](
        attn_flat,
        proj_weight,
        dy_flat,
        d_attn_flat,
        delta_flat,
        M,
        n_head,
        head_dim,
        D_OUT=d_out,
        BLOCK_M=dattn_block_m,
        BLOCK_DOUT=dattn_block_dout,
        num_warps=_FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DATTN_NUM_WARPS,
        num_stages=_FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DATTN_NUM_STAGES,
    )

    d_proj_weight = torch.empty_like(proj_weight)
    dweight_block_m = _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DWEIGHT_BLOCK_M
    dweight_block_dout = _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DWEIGHT_BLOCK_DOUT
    dweight_block_din = _FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DWEIGHT_BLOCK_DIN
    dweight_grid = (
        triton.cdiv(d_out, dweight_block_dout),
        triton.cdiv(d_in, dweight_block_din),
    )
    wrap_triton(_fused_attn_spda_and_output_proj_dweight_bwd_kernel)[dweight_grid](
        attn_flat,
        dy_flat,
        d_proj_weight,
        M,
        d_out,
        d_in,
        BLOCK_M=dweight_block_m,
        BLOCK_DOUT=dweight_block_dout,
        BLOCK_DIN=dweight_block_din,
        num_warps=_FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DWEIGHT_NUM_WARPS,
        num_stages=_FUSED_ATTN_SPDA_AND_OUTPUT_PROJ_DWEIGHT_NUM_STAGES,
    )
    d_attn = d_attn_flat.view(B, T, n_head, head_dim)
    delta = delta_flat.view(B, T, n_head)
    dq, dk, dv = _fused_attn_spda_bwd_op(
        d_attn,
        q,
        k,
        v,
        lse,
        delta,
        window_size,
    )
    return dq, dk, dv, d_proj_weight


# ── torch.library.triton_op wrapping — visible to torch.compile ──


@torch.library.triton_op(
    "nanoops::fused_attn_spda_and_output_fwd",
    mutates_args=(),
)
def _fused_attn_spda_and_output_fwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj_weight: torch.Tensor,
    residual: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-op forward wrapper returning `(y, attn_out, lse)`."""
    return _fused_attn_spda_and_output_fwd_impl(
        q,
        k,
        v,
        proj_weight,
        residual,
        window_size,
    )


@torch.library.triton_op(
    "nanoops::fused_attn_spda_and_output_bwd",
    mutates_args=(),
)
def _fused_attn_spda_and_output_bwd_op(
    dy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_out: torch.Tensor,
    lse: torch.Tensor,
    proj_weight: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-op backward wrapper returning `(dq, dk, dv, dW)`."""
    return _fused_attn_spda_and_output_bwd_impl(
        dy,
        q,
        k,
        v,
        attn_out,
        lse,
        proj_weight,
        window_size,
    )


def _fused_attn_spda_and_output_setup_context(
    ctx: Any,
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ],
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Save tensors needed by the combined SDPA/output backward."""
    q, k, v, proj_weight, _residual, window_size = inputs
    _y, attn_out, lse = output
    ctx.save_for_backward(q, k, v, attn_out, lse, proj_weight)
    ctx.window_size = window_size


def _fused_attn_spda_and_output_autograd_backward(
    ctx: Any,
    dy: torch.Tensor,
    _d_attn_out_aux: torch.Tensor,
    _d_lse_aux: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """Backward for combined SDPA + output projection."""
    q, k, v, attn_out, lse, proj_weight = ctx.saved_tensors
    dq, dk, dv, d_proj_weight = _fused_attn_spda_and_output_bwd_op(
        dy,
        q,
        k,
        v,
        attn_out,
        lse,
        proj_weight,
        ctx.window_size,
    )
    return dq, dk, dv, d_proj_weight, dy, None


_fused_attn_spda_and_output_fwd_op.register_autograd(
    _fused_attn_spda_and_output_autograd_backward,
    setup_context=_fused_attn_spda_and_output_setup_context,
)


def fused_attn_spda_and_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj_weight: torch.Tensor,
    residual: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Combined training attention tail: SDPA, output projection, residual.

    Args:
      q: `(B, T, H_q, D)` contiguous CUDA query tensor.
      k: `(B, T, H_kv, D)` contiguous CUDA key tensor.
      v: `(B, T, H_kv, D)` contiguous CUDA value tensor.
      proj_weight: `(C, H_q * D)` contiguous output projection weight.
      residual: `(B, T, C)` residual stream to add after projection.
      window_size: total visible keys per query.

    Returns:
      `(B, T, C)` tensor. Backward fuses output-projection `d_attn_out`
      with SDPA `delta = sum(attn_out * d_attn_out)`.
    """
    assert q.ndim == k.ndim == v.ndim == 4
    assert residual.ndim == 3
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    proj_weight = proj_weight.contiguous()
    residual = residual.contiguous()
    y, _attn_out, _lse = _fused_attn_spda_and_output_fwd_op(
        q,
        k,
        v,
        proj_weight,
        residual,
        window_size,
    )
    return y
