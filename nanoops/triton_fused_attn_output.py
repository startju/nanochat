"""Attention-output Triton kernels for nanoops.

Contains `attn_output_proj_residual`, the output-projection side of attention:

    y = residual + attn_out @ proj_weight.T

Public tensors use flattened `(M, D)` layout, where `M = B*T` for the
training path:

  - `attn_out`: `(M, D_in)`, CUDA contiguous, dtype is activation dtype.
  - `proj_weight`: `(D_out, D_in)`, CUDA contiguous, usually fp32 master
    weights or activation dtype.
  - `residual`: `(M, D_out)`, CUDA contiguous, activation dtype.
  - return: `(M, D_out)`, dtype=`attn_out.dtype`.

Backward returns `d_attn_out = dy @ proj_weight` and
`d_proj_weight = dy.T @ attn_out`; residual's gradient is the direct `dy`.
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
# Attention output projection + residual.
#
# Forward math for flattened row m and output channel o:
#   y[m, o] = residual[m, o] + Σ_i attn_out[m, i] * proj_weight[o, i]
#
# Backward math:
#   d_attn_out[m, i]    = Σ_o dy[m, o] * proj_weight[o, i]
#   d_proj_weight[o, i] = Σ_m dy[m, o] * attn_out[m, i]
#   d_residual[m, o]    = dy[m, o]
#
# d_attn_out and d_proj_weight use separate kernels because their reductions
# are over different axes. d_proj_weight is owned by `(D_out, D_in)` tiles and
# reduces over M internally, so it does not need atomics.
# ─────────────────────────────────────────────────────────────────────

if _HAS_TRITON:
    _ATTN_OUTPUT_PROJ_FWD_BLOCK_M = 64
    _ATTN_OUTPUT_PROJ_FWD_BLOCK_DOUT = 128
    _ATTN_OUTPUT_PROJ_FWD_BLOCK_DIN = 32
    _ATTN_OUTPUT_PROJ_FWD_NUM_WARPS = 8
    _ATTN_OUTPUT_PROJ_FWD_NUM_STAGES = 3

    _ATTN_OUTPUT_PROJ_DATTN_BLOCK_M = 64
    _ATTN_OUTPUT_PROJ_DATTN_BLOCK_DOUT = 32
    _ATTN_OUTPUT_PROJ_DATTN_BLOCK_DIN = 128
    _ATTN_OUTPUT_PROJ_DATTN_NUM_WARPS = 8
    _ATTN_OUTPUT_PROJ_DATTN_NUM_STAGES = 3

    _ATTN_OUTPUT_PROJ_DWEIGHT_BLOCK_M = 64
    _ATTN_OUTPUT_PROJ_DWEIGHT_BLOCK_DOUT = 128
    _ATTN_OUTPUT_PROJ_DWEIGHT_BLOCK_DIN = 64
    _ATTN_OUTPUT_PROJ_DWEIGHT_NUM_WARPS = 8
    _ATTN_OUTPUT_PROJ_DWEIGHT_NUM_STAGES = 1

    @triton.jit
    def _attn_output_proj_residual_fwd_kernel(
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
    def _attn_output_proj_residual_dattn_bwd_kernel(
        proj_w_ptr,  # (D_out, D_in), weight dtype — in: projection weight
        dy_ptr,  # (M, D_out), activation dtype — in: output gradient
        d_attn_out_ptr,  # (M, D_in), activation dtype — out
        M,  # int — row count after flattening leading dims
        D_OUT: tl.constexpr,  # projection output width
        D_IN: tl.constexpr,  # projection input width
        BLOCK_M: tl.constexpr,
        BLOCK_DOUT: tl.constexpr,
        BLOCK_DIN: tl.constexpr,
    ):
        """Compute `d_attn_out = dy @ proj_weight`."""
        pid_m = tl.program_id(0)
        pid_din = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        din_cols = pid_din * BLOCK_DIN + tl.arange(0, BLOCK_DIN)
        row_mask = rows < M
        din_mask = din_cols < D_IN

        d_attn_acc = tl.zeros((BLOCK_M, BLOCK_DIN), dtype=tl.float32)

        for dout_start in range(0, D_OUT, BLOCK_DOUT):
            dout_cols = dout_start + tl.arange(0, BLOCK_DOUT)
            dout_mask = dout_cols < D_OUT

            dy_ptrs = dy_ptr + rows[:, None] * D_OUT + dout_cols[None, :]
            dy = tl.load(
                dy_ptrs,
                mask=row_mask[:, None] & dout_mask[None, :],
                other=0.0,
            )
            w_ptrs = proj_w_ptr + dout_cols[:, None] * D_IN + din_cols[None, :]
            w = tl.load(
                w_ptrs,
                mask=dout_mask[:, None] & din_mask[None, :],
                other=0.0,
            ).to(dy.dtype)

            d_attn_acc += tl.dot(dy, w)

        d_attn_ptrs = d_attn_out_ptr + rows[:, None] * D_IN + din_cols[None, :]
        tl.store(
            d_attn_ptrs,
            d_attn_acc.to(d_attn_out_ptr.dtype.element_ty),
            mask=row_mask[:, None] & din_mask[None, :],
        )

    @triton.jit
    def _attn_output_proj_residual_dweight_bwd_kernel(
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


def _attn_output_proj_residual_fwd_impl(
    attn_out: torch.Tensor,
    proj_weight: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Launch the attention-output projection forward kernel.

    Args:
      attn_out: `(M, D_in)` contiguous CUDA tensor, activation dtype.
      proj_weight: `(D_out, D_in)` contiguous CUDA tensor, weight dtype.
      residual: `(M, D_out)` contiguous CUDA tensor, activation dtype.

    Returns:
      `y`: `(M, D_out)` tensor, dtype=`attn_out.dtype`.
    """
    if not _HAS_TRITON:
        raise RuntimeError("attn_output_proj_residual requires triton")
    assert attn_out.is_cuda and proj_weight.is_cuda and residual.is_cuda
    assert attn_out.is_contiguous() and proj_weight.is_contiguous() and residual.is_contiguous()
    assert attn_out.ndim == proj_weight.ndim == residual.ndim == 2
    M, D_in = attn_out.shape
    M_res, D_out = residual.shape
    D_out_w, D_in_w = proj_weight.shape
    assert M == M_res and D_in == D_in_w and D_out == D_out_w

    y = torch.empty((M, D_out), dtype=attn_out.dtype, device=attn_out.device)
    BLOCK_M = _ATTN_OUTPUT_PROJ_FWD_BLOCK_M
    BLOCK_DOUT = _ATTN_OUTPUT_PROJ_FWD_BLOCK_DOUT
    BLOCK_DIN = _ATTN_OUTPUT_PROJ_FWD_BLOCK_DIN
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D_out, BLOCK_DOUT))
    wrap_triton(_attn_output_proj_residual_fwd_kernel)[grid](
        attn_out,
        proj_weight,
        residual,
        y,
        M,
        D_out,
        D_in,
        BLOCK_M=BLOCK_M,
        BLOCK_DOUT=BLOCK_DOUT,
        BLOCK_DIN=BLOCK_DIN,
        num_warps=_ATTN_OUTPUT_PROJ_FWD_NUM_WARPS,
        num_stages=_ATTN_OUTPUT_PROJ_FWD_NUM_STAGES,
    )
    return y


def _attn_output_proj_residual_bwd_impl(
    dy: torch.Tensor,
    attn_out: torch.Tensor,
    proj_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch backward kernels for `y = residual + attn_out @ proj_weight.T`.

    Args:
      dy: `(M, D_out)` gradient of output. Made contiguous before launch.
      attn_out: `(M, D_in)` saved forward attention output.
      proj_weight: `(D_out, D_in)` saved projection weight.

    Returns:
      d_attn_out: `(M, D_in)`, dtype=`attn_out.dtype`.
      d_proj_weight: `(D_out, D_in)`, dtype=`proj_weight.dtype`.
    """
    if not _HAS_TRITON:
        raise RuntimeError("attn_output_proj_residual backward requires triton")
    dy = dy.contiguous()
    assert dy.is_cuda and attn_out.is_cuda and proj_weight.is_cuda
    assert dy.is_contiguous() and attn_out.is_contiguous() and proj_weight.is_contiguous()
    M, D_in = attn_out.shape
    M_dy, D_out = dy.shape
    D_out_w, D_in_w = proj_weight.shape
    assert M == M_dy and D_in == D_in_w and D_out == D_out_w

    d_attn_out = torch.empty_like(attn_out)
    d_proj_weight = torch.empty_like(proj_weight)
    BLOCK_M = _ATTN_OUTPUT_PROJ_DATTN_BLOCK_M
    BLOCK_DOUT = _ATTN_OUTPUT_PROJ_DATTN_BLOCK_DOUT
    BLOCK_DIN = _ATTN_OUTPUT_PROJ_DATTN_BLOCK_DIN
    dattn_grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D_in, BLOCK_DIN))
    wrap_triton(_attn_output_proj_residual_dattn_bwd_kernel)[dattn_grid](
        proj_weight,
        dy,
        d_attn_out,
        M,
        D_out,
        D_in,
        BLOCK_M=BLOCK_M,
        BLOCK_DOUT=BLOCK_DOUT,
        BLOCK_DIN=BLOCK_DIN,
        num_warps=_ATTN_OUTPUT_PROJ_DATTN_NUM_WARPS,
        num_stages=_ATTN_OUTPUT_PROJ_DATTN_NUM_STAGES,
    )
    BLOCK_M = _ATTN_OUTPUT_PROJ_DWEIGHT_BLOCK_M
    BLOCK_DOUT = _ATTN_OUTPUT_PROJ_DWEIGHT_BLOCK_DOUT
    BLOCK_DIN = _ATTN_OUTPUT_PROJ_DWEIGHT_BLOCK_DIN
    dweight_grid = (triton.cdiv(D_out, BLOCK_DOUT), triton.cdiv(D_in, BLOCK_DIN))
    wrap_triton(_attn_output_proj_residual_dweight_bwd_kernel)[dweight_grid](
        attn_out,
        dy,
        d_proj_weight,
        M,
        D_out,
        D_in,
        BLOCK_M=BLOCK_M,
        BLOCK_DOUT=BLOCK_DOUT,
        BLOCK_DIN=BLOCK_DIN,
        num_warps=_ATTN_OUTPUT_PROJ_DWEIGHT_NUM_WARPS,
        num_stages=_ATTN_OUTPUT_PROJ_DWEIGHT_NUM_STAGES,
    )
    return d_attn_out, d_proj_weight


@torch.library.triton_op(
    "nanoops::attn_output_proj_residual_fwd",
    mutates_args=(),
)
def _attn_output_proj_residual_fwd_op(
    attn_out: torch.Tensor,
    proj_weight: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Triton-op forward wrapper.

    Args:
      attn_out: `(M, D_in)` contiguous CUDA tensor.
      proj_weight: `(D_out, D_in)` contiguous CUDA tensor.
      residual: `(M, D_out)` contiguous CUDA tensor.

    Returns:
      `(M, D_out)` tensor, dtype=`attn_out.dtype`.
    """
    return _attn_output_proj_residual_fwd_impl(attn_out, proj_weight, residual)


@torch.library.triton_op(
    "nanoops::attn_output_proj_residual_bwd",
    mutates_args=(),
)
def _attn_output_proj_residual_bwd_op(
    dy: torch.Tensor,
    attn_out: torch.Tensor,
    proj_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-op backward wrapper.

    Args:
      dy: `(M, D_out)` output gradient.
      attn_out: `(M, D_in)` saved forward attention output.
      proj_weight: `(D_out, D_in)` saved projection weight.

    Returns:
      `(d_attn_out, d_proj_weight)` with shapes `(M, D_in)` and
      `(D_out, D_in)`.
    """
    return _attn_output_proj_residual_bwd_impl(dy, attn_out, proj_weight)


def _attn_output_proj_residual_setup_context(
    ctx: Any,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output: torch.Tensor,
) -> None:
    """Save forward tensors needed by the Triton-op autograd callback."""
    attn_out, proj_weight, _residual = inputs
    ctx.save_for_backward(attn_out, proj_weight)


def _attn_output_proj_residual_autograd_backward(
    ctx: Any,
    grad_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autograd callback for `nanoops::attn_output_proj_residual_fwd`.

    Args:
      grad_y: `(M, D_out)` gradient of public output.

    Returns:
      Gradients for `(attn_out, proj_weight, residual)`. The residual
      gradient is the direct passthrough `grad_y`.
    """
    attn_out, proj_weight = ctx.saved_tensors
    d_attn_out, d_proj_weight = _attn_output_proj_residual_bwd_op(
        grad_y,
        attn_out,
        proj_weight,
    )
    return d_attn_out, d_proj_weight, grad_y


_attn_output_proj_residual_fwd_op.register_autograd(
    _attn_output_proj_residual_autograd_backward,
    setup_context=_attn_output_proj_residual_setup_context,
)


def attn_output_proj_residual(
    attn_out: torch.Tensor,
    proj_weight: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Fused `y = residual + attn_out @ proj_weight.T`.

    Args:
      attn_out: `(M, D_in)` CUDA tensor, activation dtype.
      proj_weight: `(D_out, D_in)` projection weight tensor.
      residual: `(M, D_out)` residual stream tensor.

    Returns:
      `(M, D_out)` projected residual output, dtype=`attn_out.dtype`.
    """
    assert attn_out.ndim == proj_weight.ndim == residual.ndim == 2
    attn_out = attn_out.contiguous()
    proj_weight = proj_weight.contiguous()
    residual = residual.contiguous()
    return _attn_output_proj_residual_fwd_op(attn_out, proj_weight, residual)
