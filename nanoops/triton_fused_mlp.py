"""Fused MLP Triton kernels for nanoops.

Contains `fused_mlp`, the transformer MLP side with pre-norm and outer
residual fused around the two linear projections:

    x_hat = RMSNorm(x)
    z     = x_hat @ fc_weight.T
    r     = relu(z)^2
    y     = x + r @ proj_weight.T

Public tensors use `(B, T, K)` for activations and standard row-major weight
layout:

  - `x`: `(B, T, K)`, CUDA contiguous activation tensor.
  - `fc_weight`: `(N_fc, K)`, CUDA contiguous c_fc weight.
  - `proj_weight`: `(K, N_fc)`, CUDA contiguous c_proj weight.
  - return: `(B, T, K)`, dtype=`x.dtype`.

Internally the public wrapper flattens `B*T -> M`; all Triton kernels operate
on contiguous `(M, *)` views. Forward uses three kernels and backward uses four
kernels so each matmul can fuse its neighboring elementwise/reduction work.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.library import wrap_triton

from .triton_fused_add_norm import _pick_fused_add_norm_tile_config

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# d24 compile-path tuning constants. Benchmark harnesses can monkeypatch these
# before torch.compile; defaults preserve the checked-in path.
_MLP_FWD_FC_BLOCK_M = 128
_MLP_FWD_FC_BLOCK_N = 64
_MLP_FWD_FC_BLOCK_K = 32
_MLP_FWD_FC_NUM_WARPS = 4
_MLP_FWD_FC_NUM_STAGES = 3

_MLP_FWD_PROJ_BLOCK_M = 128
_MLP_FWD_PROJ_BLOCK_K_OUT = 64
_MLP_FWD_PROJ_BLOCK_N = 32
_MLP_FWD_PROJ_NUM_WARPS = 4
_MLP_FWD_PROJ_NUM_STAGES = 3

_MLP_BWD_DZ_BLOCK_M = 128
_MLP_BWD_DZ_BLOCK_N = 64
_MLP_BWD_DZ_BLOCK_K_OUT = 32
_MLP_BWD_DZ_NUM_WARPS = 4
_MLP_BWD_DZ_NUM_STAGES = 3

_MLP_BWD_DW_PROJ_BLOCK_K_OUT = 64
_MLP_BWD_DW_PROJ_BLOCK_N = 64
_MLP_BWD_DW_PROJ_BLOCK_M = 32
_MLP_BWD_DW_PROJ_NUM_WARPS = 4
_MLP_BWD_DW_PROJ_NUM_STAGES = 2

_MLP_BWD_DW_FC_BLOCK_M = 32
_MLP_BWD_DW_FC_BLOCK_N = 128
_MLP_BWD_DW_FC_BLOCK_K = 64
_MLP_BWD_DW_FC_NUM_WARPS = 4
_MLP_BWD_DW_FC_NUM_STAGES = 2

_MLP_BWD_DX_BLOCK_M = 128
_MLP_BWD_DX_BF16_BLOCK_K = 64
_MLP_BWD_DX_BF16_BLOCK_N = 64
_MLP_BWD_DX_BF16_NUM_WARPS = 4
_MLP_BWD_DX_BF16_NUM_STAGES = 2


# ─────────────────────────────────────────────────────────────────────
# Fused MLP + outer residual — standard transformer mlp side:
#     y = x + relu(RMSNorm(x) @ W_fc.T)² @ W_proj.T
#
# Public Python API keeps the transformer shape `(B, T, K)`. The public wrapper
# flattens to `M = B*T` before calling the Triton op; all kernels below use the
# row-major `(M, *)` view.
#
# Per flattened row m, hidden channel k, fc channel n:
#     rms_inv[m]   = rsqrt((1/K) * Σ_k x[m, k]^2 + eps)
#     x_hat[m, k]  = x[m, k] * rms_inv[m]
#     z[m, n]      = Σ_k x_hat[m, k] * fc_weight[n, k]
#     r[m, n]      = relu(z[m, n])^2
#     y[m, k]      = x[m, k] + Σ_n r[m, n] * proj_weight[k, n]
#
# Backward saves only `rms_inv` and `z` from forward. It rematerializes x_hat
# from `(x, rms_inv)` and computes a side-output
# `inner[m] = (1/K) * Σ_n dz[m, n] * z[m, n]` in the dz kernel so the final dx
# kernel can apply RMSNorm backward without doing a K-wide reduction.
#
# See `fused_mlp` and the impl helper docstrings for the kernel
# breakdown (3-step fwd, 4-step Triton bwd) and saved-state details.
# ─────────────────────────────────────────────────────────────────────


if _HAS_TRITON:

    @triton.jit
    def _fused_mlp_rms_norm_fwd_kernel(
        x_ptr,  # (M, K), activation dtype — in
        x_hat_ptr,  # (M, K), activation dtype — out: RMSNorm(x)
        rms_inv_ptr,  # (M,) fp32 — out
        M,  # int — row count
        K,  # int — hidden width
        eps,  # float
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Plain no-affine RMSNorm forward."""
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks = tl.arange(0, BLOCK_K)
        row_mask = rows < M
        k_mask = ks < K
        mask = row_mask[:, None] & k_mask[None, :]
        x = tl.load(
            x_ptr + rows[:, None] * K + ks[None, :],
            mask=mask,
            other=0.0,
        )
        rms_inv = tl.rsqrt(tl.sum(x * x, axis=1, dtype=tl.float32) / K + eps)
        tl.store(rms_inv_ptr + rows, rms_inv, mask=row_mask)
        tl.store(
            x_hat_ptr + rows[:, None] * K + ks[None, :],
            (x * rms_inv[:, None]).to(x_hat_ptr.dtype.element_ty),
            mask=mask,
        )

    # c_fc matmul with inline weight cast: z = x @ W_fc.T, but W_fc is
    # loaded in its native dtype (fp32 master) and cast to x's dtype on
    # load — avoids materializing a cast weight tile in HBM. Replaces
    # `torch.matmul(x_hat, fc_weight.t())` in fwd step 1.
    # Trade: lose cuBLAS's tensor-core efficiency (~70% peak) for
    # Triton's (~60% peak), gain 1 launch + ~75 μs HBM round-trip
    # (36 MB write+read at d24).
    # d24 config locked: (BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, nw=4, st=3).
    @triton.jit
    def _fused_mlp_fc_matmul_fwd_kernel(
        x_ptr,  # (M, K), activation dtype — in: x_hat
        w_ptr,  # (N, K), weight dtype — in: fc_weight
        z_ptr,  # (M, N), activation dtype — out: c_fc pre-activation
        M,  # int — row count
        N,  # int — output width / fc hidden width
        K,  # int — input width / residual width
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        ms = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = ms < M
        n_mask = ns < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            x_tile = tl.load(
                x_ptr + ms[:, None] * K + ks[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w_tile = tl.load(
                w_ptr + ns[:, None] * K + ks[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(x_tile, tl.trans(w_tile.to(x_tile.dtype)))

        tl.store(
            z_ptr + ms[:, None] * N + ns[None, :],
            acc.to(z_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    # relu² + c_proj + residual_add in one Triton pass — r stays in
    # registers (saves M·N HBM round-trip). Caller passes fixed config
    # (autotune dispatch isn't CUDA Graph capture-friendly); d24 winner
    # locked: (BLOCK_M=128, BLOCK_K_OUT=64, BLOCK_N=32, nw=4, st=3).
    @triton.jit
    def _fused_mlp_proj_residual_fwd_kernel(
        z_ptr,  # (M, N), activation dtype — in: c_fc pre-activation
        proj_w_ptr,  # (K_out, N), weight dtype — in: c_proj weight
        residual_ptr,  # (M, K_out), activation dtype — in: residual stream x
        y_ptr,  # (M, K_out), activation dtype — out
        M,  # int — row count
        N,  # int — c_fc output width / c_proj input width
        K_out,  # int — c_proj output width, equals residual width K
        BLOCK_M: tl.constexpr,
        BLOCK_K_OUT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused c_proj matmul + residual add.

        Matmul inputs stay in activation dtype where possible; fp32 master
        weights are cast to the activation dtype on load.
        """
        pid_m = tl.program_id(0)
        pid_k_out = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        k_outs = pid_k_out * BLOCK_K_OUT + tl.arange(0, BLOCK_K_OUT)
        row_mask = rows < M
        k_out_mask = k_outs < K_out

        # c_proj matmul, inner loop over N.
        acc = tl.zeros((BLOCK_M, BLOCK_K_OUT), dtype=tl.float32)
        for n_start in range(0, N, BLOCK_N):
            n_cols = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_cols < N

            z = tl.load(
                z_ptr + rows[:, None] * N + n_cols[None, :],
                mask=row_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            relu_z = tl.where(z > 0.0, z, 0.0)
            r = relu_z * relu_z
            # Cast weight to z's dtype on load — handles fp32 master +
            # bf16 activation. `.to(z.dtype)` is a no-op when matched.
            proj_w = tl.load(
                proj_w_ptr + k_outs[:, None] * N + n_cols[None, :],
                mask=k_out_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(r, tl.trans(proj_w.to(z.dtype)))

        # Add residual, write y. Keep residual in native dtype; cast the
        # matmul acc to output dtype first, then add — saves a bf16→fp32
        # conversion on the load + skips the final store cast.
        offs = rows[:, None] * K_out + k_outs[None, :]
        mask_2d = row_mask[:, None] & k_out_mask[None, :]
        residual = tl.load(residual_ptr + offs, mask=mask_2d, other=0.0)
        y = acc.to(y_ptr.dtype.element_ty) + residual
        tl.store(y_ptr + offs, y, mask=mask_2d)

    # dz bwd for the c_proj + relu² + residual_add fwd op
    # (`_fused_mlp_proj_residual_fwd_kernel`). The other bwd outputs of
    # that fwd are split off:
    #   dW_proj          → `_fused_mlp_dproj_weight_bwd_kernel` (different reduction axis)
    #   d_residual = dy  → folded into dx by `_fused_mlp_dx_bwd_kernel`'s outer-residual pass
    # This kernel produces only dz, plus the inner_buf side-output for D.
    #
    # Math:
    #   dr = dy @ W_proj           (matmul; dr kept in registers, never to HBM,
    #                               saves ~50 MB round-trip at d24)
    #   dz = 2·relu(z) · dr        (relu² bwd, applied inline)
    #
    # Side-output: inner_buf[m] += Σ_n(dz·z)/norm_dim via atomic_add (= inner).
    # D uses it via the identity Σ_k(dx_hat·x_hat) = Σ_n(dz·z) to skip its
    # per-row reduction over norm_dim. Division folded in here (we already
    # have K_out, which equals norm_dim in MLP by construction — asserted
    # in forward as K_proj_out == K). Free — dz and z are both already in
    # registers.
    #
    # Why atomic_add (and not a scratchpad + downstream reduce)?
    # Our grid here is (M/BM, N/BN), so the N-axis reduction needed for
    # inner is split across multiple programs per m_tile. None of the
    # downstream kernels (B, C, D) has a free natural N-axis reduction
    # we could piggyback on — D reduces over N for its dx_hat matmul,
    # but its (M/BM, K/BK) grid duplicates each m_tile K/BK times, which
    # would either duplicate the inner work or require inter-program sync.
    # A scratchpad-then-`torch.sum` route works (benched ~equal) but adds
    # an extra (num_n_tiles, M) buffer + a separate reduce launch. Using
    # atomic_add here keeps everything self-contained in this kernel.
    # d24 config locked: (BLOCK_M=128, BLOCK_N=64, BLOCK_K_OUT=32, nw=4, st=3).
    @triton.jit
    def _fused_mlp_dz_bwd_kernel(
        dy_ptr,  # (M, K_out), activation dtype — in: gradient w.r.t. y
        z_ptr,  # (M, N), activation dtype — in: saved c_fc pre-activation
        proj_w_ptr,  # (K_out, N), weight dtype — in: proj_weight
        dz_ptr,  # (M, N), activation dtype — out: gradient w.r.t. z
        inner_buf_ptr,  # (M,) fp32 — side-output Σ_n(dz·z)/norm_dim; caller zero-inits
        M,  # int — row count
        N,  # int — c_fc output width / c_proj input width
        K_out,  # int — c_proj output width / norm dimension
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K_OUT: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < M
        n_mask = n_cols < N

        # dr matmul: dr[m, n] = sum_{kp} dy[m, kp] * W_proj[kp, n]
        dr = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kp_start in range(0, K_out, BLOCK_K_OUT):
            kps = kp_start + tl.arange(0, BLOCK_K_OUT)
            kp_mask = kps < K_out
            dy = tl.load(
                dy_ptr + rows[:, None] * K_out + kps[None, :],
                mask=row_mask[:, None] & kp_mask[None, :],
                other=0.0,
            )
            # Cast fp32 master weight to dy's dtype on load (see Step 2 fwd).
            proj_w = tl.load(
                proj_w_ptr + kps[:, None] * N + n_cols[None, :],
                mask=kp_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            dr += tl.dot(dy, proj_w.to(dy.dtype))

        # relu² bwd applied to dr in registers — dr never materialized to HBM.
        z = tl.load(
            z_ptr + rows[:, None] * N + n_cols[None, :],
            mask=row_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        relu_z = tl.where(z > 0.0, z, 0.0)
        dz = dr.to(dz_ptr.dtype.element_ty) * 2 * relu_z

        tl.store(
            dz_ptr + rows[:, None] * N + n_cols[None, :],
            dz,
            mask=row_mask[:, None] & n_mask[None, :],
        )

        # Side-output: partial inner = Σ_n(dz·z) / norm_dim (see kernel header).
        # Force fp32 accumulator — dz and z are bf16 in the bf16 path, and
        # summing BLOCK_N bf16 products in bf16 would lose precision.
        # K_out == norm_dim in MLP, so dividing here saves D a div-on-load.
        inner_partial = tl.sum(dz * z, axis=1, dtype=tl.float32) / K_out
        tl.atomic_add(inner_buf_ptr + rows, inner_partial, mask=row_mask)

    # dW_proj = dy.T @ relu²(z), r recomputed inline (no materialization).
    # Pairs with `_fused_mlp_dz_bwd_kernel` to cover both bwd outputs of the
    # fwd kernel `_fused_mlp_proj_residual_fwd_kernel`; reduction axis
    # differs (M here vs K_out there) so they're separate kernels.
    # d24 config locked: (BLOCK_K_OUT=64, BLOCK_N=64, BLOCK_M=32, nw=4, st=2).
    @triton.jit
    def _fused_mlp_dproj_weight_bwd_kernel(
        dy_ptr,  # (M, K_out), activation dtype — in
        z_ptr,  # (M, N), activation dtype — in: saved c_fc pre-activation
        dW_proj_ptr,  # (K_out, N), proj_weight dtype — out
        M,  # int — row count / reduction dimension
        N,  # int — c_proj input width
        K_out,  # int — c_proj output width
        BLOCK_K_OUT: tl.constexpr,  # output tile along K_out (c_proj's output dim)
        BLOCK_N: tl.constexpr,  # output tile along N (= N_fc)
        BLOCK_M: tl.constexpr,  # reduction tile along M
    ):
        pid_k_out = tl.program_id(0)
        pid_n = tl.program_id(1)

        k_outs = pid_k_out * BLOCK_K_OUT + tl.arange(0, BLOCK_K_OUT)
        ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_out_mask = k_outs < K_out
        n_mask = ns < N

        acc = tl.zeros((BLOCK_K_OUT, BLOCK_N), dtype=tl.float32)
        for m_start in range(0, M, BLOCK_M):
            ms = m_start + tl.arange(0, BLOCK_M)
            m_mask = ms < M

            # dy[ms, k_outs] — needs transpose for dW_proj = dy.T @ r
            dy = tl.load(
                dy_ptr + ms[:, None] * K_out + k_outs[None, :],
                mask=m_mask[:, None] & k_out_mask[None, :],
                other=0.0,
            )  # (BLOCK_M, BLOCK_K_OUT)

            # Load z and compute r = relu²(z) inline — never to HBM.
            z = tl.load(
                z_ptr + ms[:, None] * N + ns[None, :],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            relu_z = tl.where(z > 0.0, z, 0.0)
            r = relu_z * relu_z

            # acc[k_out, n] += sum_m dy[m, k_out] * r[m, n]
            acc += tl.dot(tl.trans(dy), r)

        tl.store(
            dW_proj_ptr + k_outs[:, None] * N + ns[None, :],
            acc.to(dW_proj_ptr.dtype.element_ty),
            mask=k_out_mask[:, None] & n_mask[None, :],
        )

    # dW_fc = dz.T @ x_hat with x_hat reconstructed from (x, rms_inv, nw)
    # inside the GEMM inner loop — no x_hat materialization. Saves
    # M·K HBM write+read + one launch vs the eager (cuBLAS) chain.
    # d24 config locked: (BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, nw=4, st=2).
    # Other shapes can prefer (BLOCK_N=64, BLOCK_K=128), within ~5% on 3090.
    @triton.jit
    def _fused_mlp_dfc_weight_bwd_kernel(
        dz_ptr,  # (M, N_fc), activation dtype — in
        x_ptr,  # (M, K), activation dtype — in: fwd input for x_hat recompute
        rms_inv_ptr,  # (M,) fp32
        dW_fc_ptr,  # (N_fc, K), fc_weight dtype — out
        M,  # int — row count / reduction dimension
        N_fc,  # int — c_fc output width
        K,  # int — residual width / norm dimension / c_fc input width
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        n_mask = ns < N_fc
        k_mask = ks < K

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for m_start in range(0, M, BLOCK_M):
            ms = m_start + tl.arange(0, BLOCK_M)
            m_mask = ms < M

            # Reconstruct x_hat tile (BLOCK_M, BLOCK_K) in registers.
            x = tl.load(
                x_ptr + ms[:, None] * K + ks[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            rms_inv = tl.load(rms_inv_ptr + ms, mask=m_mask, other=0.0)
            x_hat = x * rms_inv[:, None]

            # dz tile (BLOCK_M, BLOCK_N)
            dz_tile = tl.load(
                dz_ptr + ms[:, None] * N_fc + ns[None, :],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            # dW_fc[n, k] += sum_m(dz[m, n] * x_hat[m, k])
            # => acc[n, k] += dz.T[n, m] @ x_hat[m, k]
            acc += tl.dot(tl.trans(dz_tile), x_hat.to(dz_tile.dtype))

        tl.store(
            dW_fc_ptr + ns[:, None] * K + ks[None, :],
            acc.to(dW_fc_ptr.dtype.element_ty),
            mask=n_mask[:, None] & k_mask[None, :],
        )

    # Fused (c_fc bwd dx_hat) + (RMSNorm bwd) — dx_hat = dz @ W_fc computed
    # in-kernel then immediately consumed by the dx formula, never to HBM.
    # (Safe because dW_fc uses x_hat, not dx_hat.) Step A's pre-computed
    # `inner` makes dx purely elementwise — unlocks big tensor-core tiles
    # vs the BLOCK_M=4 a K-reduction would force.
    #
    # Math (per element; K below = norm_dim, the kernel's `K` param):
    #   y_norm   = x · rms_inv
    #   g_eff    = dx_hat
    #   inner    = inner_buf                    (A already divided by norm_dim)
    #   dx       = rms_inv · (g_eff - y_norm · inner) + dy  (outer residual passthrough)
    @triton.jit
    def _fused_mlp_dx_bwd_kernel(
        dz_ptr,  # (M, N_fc), activation dtype — in
        W_fc_ptr,  # (N_fc, K), weight dtype — in: fc_weight
        x_ptr,  # (M, K), activation dtype — in: fwd input
        rms_inv_ptr,  # (M,) fp32
        dy_ptr,  # (M, K), activation dtype — in: outer residual passthrough
        inner_buf_ptr,  # (M,) fp32 — Σ_n(dz·z) / norm_dim from A (= inner)
        dx_ptr,  # (M, K), activation dtype — out
        M,  # int — row count
        N_fc,  # int — c_fc output width / dx_hat reduction dimension
        K,  # int — residual width / norm dimension
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = rows < M
        k_mask = ks < K

        # Matmul: dx_hat_tile = dz @ W_fc[:, k_tile], reduce over N_fc.
        dx_hat = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for n_start in range(0, N_fc, BLOCK_N):
            ns = n_start + tl.arange(0, BLOCK_N)
            n_mask = ns < N_fc
            dz_tile = tl.load(
                dz_ptr + rows[:, None] * N_fc + ns[None, :],
                mask=row_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            # Cast fp32 master weight to dz's dtype on load.
            W_fc_tile = tl.load(
                W_fc_ptr + ns[:, None] * K + ks[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            dx_hat += tl.dot(dz_tile, W_fc_tile.to(dz_tile.dtype))

        # Apply RMSNorm bwd inline using pre-computed inner; fold outer
        # residual gradient (dy) into dx in-kernel.
        offs = rows[:, None] * K + ks[None, :]
        mask_2d = row_mask[:, None] & k_mask[None, :]
        rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0)
        # inner = (1/norm_dim) · Σ_n(dz·z); A already did the division.
        inner = tl.load(inner_buf_ptr + rows, mask=row_mask, other=0.0)
        x = tl.load(x_ptr + offs, mask=mask_2d, other=0.0)
        # dy kept in native dtype — cast the fp32 norm-path result to
        # output dtype first, then add dy. Saves a bf16→fp32 conversion
        # on dy load + skips the final cast on store (small but free).
        dy = tl.load(dy_ptr + offs, mask=mask_2d, other=0.0)

        y_norm = x * rms_inv[:, None]
        g_eff = dx_hat
        dx = (rms_inv[:, None] * (g_eff - y_norm * inner[:, None])).to(
            dx_ptr.dtype.element_ty
        ) + dy

        tl.store(dx_ptr + offs, dx, mask=mask_2d)


def _fused_mlp_fwd_impl(
    x: torch.Tensor,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the three-kernel fused MLP forward.

    Args:
      x: `(M, K)` contiguous CUDA activation tensor.
      fc_weight: `(N_fc, K)` contiguous CUDA c_fc weight. Usually fp32 master
        weight; cast inline to activation dtype before tensor-core matmul.
      proj_weight: `(K, N_fc)` contiguous CUDA c_proj weight. Its output width
        must equal K so the outer residual add is valid.
      eps: RMSNorm epsilon.

    Returns:
      y: `(M, K)`, dtype=`x.dtype`.
      rms_inv: `(M,)`, fp32, saved for RMSNorm backward.
      z: `(M, N_fc)`, dtype=`x.dtype`, saved for relu² backward.

    Standard transformer mlp sub-block + outer residual:
        x_hat = norm(x)                            (RMSNorm)
        z     = x_hat @ W_fc.T                      (Linear: c_fc)
        r     = relu(z)²                            (ReluSquare)
        mlp   = r @ W_proj.T                        (Linear: c_proj)
        y     = x + mlp                             (Residual add)

    Three steps:
      0. `_fused_mlp_rms_norm_fwd_kernel` — produces x_hat + rms_inv.
      1. `_fused_mlp_fc_matmul_fwd_kernel` — z = x_hat @ W_fc.T with the fp32→x.dtype
         weight cast fused into the matmul load (no bf16 weight tile to
         HBM).
      2. `_fused_mlp_proj_residual_fwd_kernel` — relu² + c_proj +
         residual_add(x) → y in one Triton pass."""
    if not _HAS_TRITON:
        raise RuntimeError("fused_mlp requires triton to be installed")
    assert x.is_cuda and x.is_contiguous()
    assert fc_weight.is_cuda and fc_weight.is_contiguous()
    assert proj_weight.is_cuda and proj_weight.is_contiguous()
    M, K = x.shape
    N_fc, K_w = fc_weight.shape
    K_proj_out, N_proj_in = proj_weight.shape
    assert K == K_w, f"x last dim {K} != fc_weight in dim {K_w}"
    assert N_fc == N_proj_in, f"fc out dim {N_fc} != proj in dim {N_proj_in}"
    assert K_proj_out == K, (
        f"c_proj out dim {K_proj_out} must equal x's K {K} (residual stream width)"
    )
    # Step 0: no-affine RMSNorm. Step 2 / bwd consume x directly as the
    # residual stream.
    BLOCK_D_NORM = triton.next_power_of_2(K)
    norm_cfg = _pick_fused_add_norm_tile_config(M, BLOCK_D_NORM, n_live_tiles=2)
    x_hat = torch.empty_like(x)
    rms_inv = torch.empty((M,), dtype=torch.float32, device=x.device)
    wrap_triton(_fused_mlp_rms_norm_fwd_kernel)[(triton.cdiv(M, norm_cfg.block_m),)](
        x,
        x_hat,
        rms_inv,
        M,
        K,
        eps,
        BLOCK_M=norm_cfg.block_m,
        BLOCK_K=BLOCK_D_NORM,
        num_warps=norm_cfg.num_warps,
    )

    # Step 1: c_fc via `_fused_mlp_fc_matmul_fwd_kernel` — Triton matmul that
    # loads fc_weight (fp32 master typical) in native dtype and casts
    # inline. Replaces `torch.matmul(x_hat, fc_weight.t())` to avoid
    # materializing a bf16 cast weight tile.
    # d24 compile-path sweep winner (bf16): (BM=128, BN=64, BK=32, nw=4,
    # st=3). This beats the older (256,64,32,nw=8,st=2) by ~8% on the
    # isolated kernel and avoids over-subscribing registers.
    BLOCK_M_S1 = _MLP_FWD_FC_BLOCK_M
    BLOCK_N_S1 = _MLP_FWD_FC_BLOCK_N
    BLOCK_K_S1 = _MLP_FWD_FC_BLOCK_K
    z = torch.empty((M, N_fc), dtype=x.dtype, device=x.device)
    grid_s1 = (triton.cdiv(M, BLOCK_M_S1), triton.cdiv(N_fc, BLOCK_N_S1))
    wrap_triton(_fused_mlp_fc_matmul_fwd_kernel)[grid_s1](
        x_hat,
        fc_weight,
        z,
        M,
        N_fc,
        K,
        BLOCK_M=BLOCK_M_S1,
        BLOCK_N=BLOCK_N_S1,
        BLOCK_K=BLOCK_K_S1,
        num_warps=_MLP_FWD_FC_NUM_WARPS,
        num_stages=_MLP_FWD_FC_NUM_STAGES,
    )

    # Step 2: Triton kernel for relu² + c_proj + outer residual (= x).
    # d24 compile-path sweep winner: (BM=128, BKO=64, BN=32, nw=4, st=3),
    # about 10% faster than the older nw=8/st=2 setting.
    BLOCK_M_FWD = _MLP_FWD_PROJ_BLOCK_M
    BLOCK_K_OUT_FWD = _MLP_FWD_PROJ_BLOCK_K_OUT
    BLOCK_N_FWD = _MLP_FWD_PROJ_BLOCK_N
    y = torch.empty((M, K_proj_out), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, BLOCK_M_FWD), triton.cdiv(K_proj_out, BLOCK_K_OUT_FWD))
    wrap_triton(_fused_mlp_proj_residual_fwd_kernel)[grid](
        z,
        proj_weight,
        x,
        y,
        M,
        N_fc,
        K_proj_out,
        BLOCK_M=BLOCK_M_FWD,
        BLOCK_K_OUT=BLOCK_K_OUT_FWD,
        BLOCK_N=BLOCK_N_FWD,
        num_warps=_MLP_FWD_PROJ_NUM_WARPS,
        num_stages=_MLP_FWD_PROJ_NUM_STAGES,
    )
    return y, rms_inv, z


def _fused_mlp_bwd_impl(
    dy: torch.Tensor,
    x: torch.Tensor,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    rms_inv: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the four-kernel fused MLP backward.

    Args:
      dy: `(M, K)` gradient w.r.t. public output y. Made contiguous.
      x: `(M, K)` saved forward input / residual stream.
      fc_weight: `(N_fc, K)` saved c_fc weight.
      proj_weight: `(K, N_fc)` saved c_proj weight.
      rms_inv: `(M,)` fp32 inverse RMS saved from forward.
      z: `(M, N_fc)` c_fc pre-activation saved from forward.

    Returns:
      dx: `(M, K)`, dtype=`x.dtype`.
      dW_fc: `(N_fc, K)`, dtype=`fc_weight.dtype`.
      dW_proj: `(K, N_fc)`, dtype=`proj_weight.dtype`.

    Four Triton kernels:
      A. `_fused_mlp_dz_bwd_kernel` — dz = 2·relu(z) · (dy @ W_proj). Side-output
         `inner_buf[m] = Σ_n(dz·z) / norm_dim` via atomic_add (free —
         dz/z in registers; division folded in since K_out == norm_dim in
         MLP). Step D uses it via the identity Σ_k(dx_hat·x_hat) =
         Σ_n(dz·z) to skip its K-reduction.
      B. `_fused_mlp_dproj_weight_bwd_kernel` — dW_proj = dy.T @ relu²(z) (r
         recomputed inline; output dtype = W_proj.dtype master).
      C. `_fused_mlp_dfc_weight_bwd_kernel` — dW_fc = dz.T @ x_hat (x_hat
         reconstructed from x and rms_inv inline; output dtype =
         W_fc.dtype master).
      D. `_fused_mlp_dx_bwd_kernel` — dx in one fused pass.
         x has TWO gradient paths in `y = x + mlp(norm(x))`:
           (i)  outer-residual: dx ← dy   (direct passthrough)
           (ii) norm path:      dx ← RMSNorm_bwd(dx_hat),
                                dx_hat ← dz @ W_fc
         RMSNorm_bwd math (per element):
             g_eff = dx_hat
             dx_path_ii = rms_inv · (g_eff - y_norm · inner)
         where inner is the one A pre-divided by norm_dim. dx_hat is
         never written to HBM; (i)'s `+ dy` is folded into the store.
         Safe because kernel C uses x_hat, not dx_hat."""
    if not _HAS_TRITON:
        raise RuntimeError("fused_mlp backward requires triton to be installed")
    M, K = x.shape
    N_fc = fc_weight.shape[0]
    K_out = proj_weight.shape[0]
    dy = dy.contiguous()

    # A: dz + inner_buf side-output via atomic_add.
    # d24 sweep winner: (BM=128, BN=64, BKO=32, nw=4, st=3).
    BLOCK_M_A = _MLP_BWD_DZ_BLOCK_M
    BLOCK_N_A = _MLP_BWD_DZ_BLOCK_N
    BLOCK_K_OUT_A = _MLP_BWD_DZ_BLOCK_K_OUT
    dz = torch.empty_like(z)
    inner_buf = torch.zeros((M,), dtype=torch.float32, device=z.device)
    grid_a = (triton.cdiv(M, BLOCK_M_A), triton.cdiv(N_fc, BLOCK_N_A))
    wrap_triton(_fused_mlp_dz_bwd_kernel)[grid_a](
        dy,
        z,
        proj_weight,
        dz,
        inner_buf,
        M,
        N_fc,
        K_out,
        BLOCK_M=BLOCK_M_A,
        BLOCK_N=BLOCK_N_A,
        BLOCK_K_OUT=BLOCK_K_OUT_A,
        num_warps=_MLP_BWD_DZ_NUM_WARPS,
        num_stages=_MLP_BWD_DZ_NUM_STAGES,
    )

    # B: dW_proj = dy.T @ relu²(z), r recomputed inline.
    # Output dtype = proj_weight.dtype (fp32 master typical), so the
    # gradient lands directly on the master weight without a downstream
    # .to() cast. d24 sweep winner: (BKO=64, BN=64, BM=32, nw=4, st=2).
    BLOCK_K_OUT_B = _MLP_BWD_DW_PROJ_BLOCK_K_OUT
    BLOCK_N_B = _MLP_BWD_DW_PROJ_BLOCK_N
    BLOCK_M_B = _MLP_BWD_DW_PROJ_BLOCK_M
    dW_proj = torch.empty((K_out, N_fc), dtype=proj_weight.dtype, device=z.device)
    grid_b = (triton.cdiv(K_out, BLOCK_K_OUT_B), triton.cdiv(N_fc, BLOCK_N_B))
    wrap_triton(_fused_mlp_dproj_weight_bwd_kernel)[grid_b](
        dy,
        z,
        dW_proj,
        M,
        N_fc,
        K_out,
        BLOCK_K_OUT=BLOCK_K_OUT_B,
        BLOCK_N=BLOCK_N_B,
        BLOCK_M=BLOCK_M_B,
        num_warps=_MLP_BWD_DW_PROJ_NUM_WARPS,
        num_stages=_MLP_BWD_DW_PROJ_NUM_STAGES,
    )

    # C: dW_fc = dz.T @ x_hat, x_hat reconstructed in registers from
    # (x, rms_inv) — no x_hat HBM materialization. Output dtype
    # = fc_weight.dtype (fp32 master typical).
    BLOCK_M_C = _MLP_BWD_DW_FC_BLOCK_M
    BLOCK_N_C = _MLP_BWD_DW_FC_BLOCK_N
    BLOCK_K_C = _MLP_BWD_DW_FC_BLOCK_K
    dW_fc = torch.empty((N_fc, K), dtype=fc_weight.dtype, device=x.device)
    grid_c = (triton.cdiv(N_fc, BLOCK_N_C), triton.cdiv(K, BLOCK_K_C))
    wrap_triton(_fused_mlp_dfc_weight_bwd_kernel)[grid_c](
        dz,
        x,
        rms_inv,
        dW_fc,
        M,
        N_fc,
        K,
        BLOCK_M=BLOCK_M_C,
        BLOCK_N=BLOCK_N_C,
        BLOCK_K=BLOCK_K_C,
        num_warps=_MLP_BWD_DW_FC_NUM_WARPS,
        num_stages=_MLP_BWD_DW_FC_NUM_STAGES,
    )

    # D: dx_hat matmul + RMSNorm bwd + outer-residual fold in one kernel.
    # d24 bf16 sweep winner: (BM=128, BK=64, BN=64, nw=4, st=2).
    BLOCK_M_D = _MLP_BWD_DX_BLOCK_M
    BLOCK_K_D = _MLP_BWD_DX_BF16_BLOCK_K
    BLOCK_N_D = _MLP_BWD_DX_BF16_BLOCK_N
    NW_D = _MLP_BWD_DX_BF16_NUM_WARPS
    ST_D = _MLP_BWD_DX_BF16_NUM_STAGES
    num_m_tiles = triton.cdiv(M, BLOCK_M_D)
    dx = torch.empty_like(x)
    grid_d = (num_m_tiles, triton.cdiv(K, BLOCK_K_D))
    wrap_triton(_fused_mlp_dx_bwd_kernel)[grid_d](
        dz,
        fc_weight,
        x,
        rms_inv,
        dy,
        inner_buf,
        dx,
        M,
        N_fc,
        K,
        BLOCK_M=BLOCK_M_D,
        BLOCK_K=BLOCK_K_D,
        BLOCK_N=BLOCK_N_D,
        num_warps=NW_D,
        num_stages=ST_D,
    )
    return dx, dW_fc, dW_proj


# ── torch.library.triton_op wrapping — visible to torch.compile ──
# triton_op + wrap_triton lets Dynamo/AOTAutograd decompose the op into its
# inner Triton launches during compile, avoiding the opaque op boundary
# while still giving the public API a single autograd-registered op.


@torch.library.triton_op(
    "nanoops::fused_mlp_fwd",
    mutates_args=(),
)
def _fused_mlp_fwd_op(
    x: torch.Tensor,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-op forward wrapper.

    Args:
      x: `(M, K)` contiguous CUDA activation tensor.
      fc_weight: `(N_fc, K)` contiguous CUDA c_fc weight.
      proj_weight: `(K, N_fc)` contiguous CUDA c_proj weight.
      eps: RMSNorm epsilon.

    Returns:
      `(y, rms_inv, z)` with shapes `(M, K)`, `(M,)`, `(M, N_fc)`.
    """
    return _fused_mlp_fwd_impl(x, fc_weight, proj_weight, eps)


@torch.library.triton_op(
    "nanoops::fused_mlp_bwd",
    mutates_args=(),
)
def _fused_mlp_bwd_op(
    dy: torch.Tensor,
    x: torch.Tensor,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    rms_inv: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-op backward wrapper.

    Args:
      dy: `(M, K)` output gradient.
      x: `(M, K)` saved forward input.
      fc_weight: `(N_fc, K)` saved c_fc weight.
      proj_weight: `(K, N_fc)` saved c_proj weight.
      rms_inv: `(M,)` fp32 saved inverse RMS.
      z: `(M, N_fc)` saved c_fc pre-activation.

    Returns:
      `(dx, dW_fc, dW_proj)` with shapes `(M, K)`, `(N_fc, K)`, `(K, N_fc)`.
    """
    return _fused_mlp_bwd_impl(dy, x, fc_weight, proj_weight, rms_inv, z)


def _fused_mlp_setup_context(
    ctx: Any,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float],
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Save forward inputs/outputs needed by Triton-op autograd.

    `inputs` is (x, fc_weight, proj_weight, eps).
    `output` is (y, rms_inv, z); y has no backward-only use, while rms_inv
    and z are the saved tensors that avoid recomputing the forward.
    """
    x, fc_weight, proj_weight, _eps = inputs
    _y, rms_inv, z = output
    ctx.save_for_backward(fc_weight, proj_weight, x, rms_inv, z)


def _fused_mlp_autograd_backward(
    ctx: Any,
    grad_y: torch.Tensor,
    grad_rms_inv: torch.Tensor,
    grad_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """Autograd callback for `nanoops::fused_mlp_fwd`.

    Receives gradients for the three forward outputs:
      grad_y (M, K), grad_rms_inv (M,), grad_z (M, N_fc).
    Only grad_y is meaningful to users; rms_inv/z are hidden saved-state
    outputs, so their incoming grads are ignored.

    Returns one gradient per forward input:
      dx (M, K), dW_fc (N_fc, K), dW_proj (K, N_fc), None for eps.
    """
    # grad_rms_inv / grad_z are zeros: rms_inv and z exist only to
    # plumb fwd→bwd state inside this op, no downstream consumer.
    W_fc, W_proj, x, rms_inv, z = ctx.saved_tensors
    dx, dW_fc, dW_proj = _fused_mlp_bwd_op(grad_y, x, W_fc, W_proj, rms_inv, z)
    return dx, dW_fc, dW_proj, None


_fused_mlp_fwd_op.register_autograd(
    _fused_mlp_autograd_backward,
    setup_context=_fused_mlp_setup_context,
)


def fused_mlp(
    x: torch.Tensor,
    fc_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Full MLP sub-block + residual add with pre-norm.

    Math:
      `y = x + relu²(RMSNorm(x) @ fc_weight.T) @ proj_weight.T`

    Args:
      x: `(B, T, K)` contiguous CUDA activation tensor. The wrapper flattens
        to `(M, K)`, `M=B*T`, for internal kernels.
      fc_weight: `(N_fc, K)` c_fc weight tensor.
      proj_weight: `(K, N_fc)` c_proj weight tensor. Its output dim must match
        x's K so the outer residual add is valid.
      eps: RMSNorm epsilon.

    Returns:
      `y`: `(B, T, K)` tensor with dtype=`x.dtype`.

    Standard transformer mlp side: `y = x + mlp(norm(x))`. If the caller
    needs to pre-sum with an attention residual, do it outside.
    fc_weight / proj_weight are loaded in their native dtype inside the
    fwd/bwd Triton kernels and cast inline to the activation dtype before
    each tensor-core matmul (handles the fp32-master + bf16-activation
    case typical in nanchat). dW_fc / dW_proj are allocated with the
    master weight's dtype, so the gradient lands directly on the master
    weight — no wrapper-level `.to()` and no autograd routing needed.

    Implemented as a `torch.library.triton_op` with `wrap_triton` kernel
    launches so torch.compile can see the inner Triton kernels instead of
    treating the block as opaque. See the impl helpers for the actual kernel
    call sequences."""
    assert x.ndim == 3, f"x must be 3D (B, T, K), got {x.ndim}D"
    assert x.is_contiguous()
    B, T, K = x.shape
    x_2d = x.view(B * T, K)

    # triton_op returns internal-M-view (y, rms_inv, z); rms_inv/z are
    # saved-for-backward only, so we drop them here and return just y in the
    # public `(B, T, K)` view.
    y, _rms_inv, _z = _fused_mlp_fwd_op(x_2d, fc_weight, proj_weight, eps)
    return y.view(B, T, K)
