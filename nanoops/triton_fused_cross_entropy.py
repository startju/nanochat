"""Memory-efficient fused lm-head + softcap + cross entropy.

Public entry: `fused_cross_entropy`.

Training GPT tail math:

    raw_logits[m, v] = x[m] @ weight[v].T
    logits[m, v]     = softcap * tanh(raw_logits[m, v] / softcap)
    loss[m]          = logsumexp_v(logits[m, v]) - logits[m, target[m]]

When `x_backout` and `backout_scale` are provided, the public entry also
absorbs the model tail that normally runs before lm-head projection:

    x_mix[m, k]  = x[m, k] - backout_scale * x_backout[m, k]
    rms_inv[m]   = rsqrt(mean_k(x_mix[m, k]^2) + eps)
    x_norm[m, k] = x_mix[m, k] * rms_inv[m]

and the lm-head projection consumes `x_norm`.

The key memory property is that `(M, V)` logits are never materialized. Triton
kernels stream vocab tiles, save only one fp32 `lse` value per token row, and
recompute logits in backward for `dx` and `d_weight`. The final-tail path saves
one additional fp32 `rms_inv` value per token row.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.library import wrap_triton

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


_FUSED_CROSS_ENTROPY_FWD_BLOCK_M = 16
_FUSED_CROSS_ENTROPY_FWD_BLOCK_V = 64
_FUSED_CROSS_ENTROPY_FWD_BLOCK_K = 32
_FUSED_CROSS_ENTROPY_FWD_NUM_WARPS = 4
_FUSED_CROSS_ENTROPY_FWD_NUM_STAGES = 3

_FUSED_CROSS_ENTROPY_DX_BLOCK_M = 16
_FUSED_CROSS_ENTROPY_DX_BLOCK_K = 32
_FUSED_CROSS_ENTROPY_DX_BLOCK_V = 64
_FUSED_CROSS_ENTROPY_DX_NUM_WARPS = 4
_FUSED_CROSS_ENTROPY_DX_NUM_STAGES = 3

_FUSED_CROSS_ENTROPY_DW_BLOCK_V = 64
_FUSED_CROSS_ENTROPY_DW_BLOCK_K = 32
_FUSED_CROSS_ENTROPY_DW_BLOCK_M = 32
_FUSED_CROSS_ENTROPY_DW_NUM_WARPS = 4
_FUSED_CROSS_ENTROPY_DW_NUM_STAGES = 3

_FUSED_CROSS_ENTROPY_NORM_BWD_BLOCK_M = 16
_FUSED_CROSS_ENTROPY_NORM_BWD_BLOCK_K = 32
_FUSED_CROSS_ENTROPY_NORM_BWD_NUM_WARPS = 4


if _HAS_TRITON:
    @triton.jit
    def _fused_cross_entropy_fwd_kernel(
        x_ptr,  # (M, K), activation dtype - in
        weight_ptr,  # (V_PAD, K), fp32/bf16 - in: lm_head weight
        target_ptr,  # (M,), int64 - in
        loss_ptr,  # (M,), fp32 - out
        lse_ptr,  # (M,), fp32 - out
        M,  # int - row count
        V_PAD: tl.constexpr,  # physical weight rows
        K: tl.constexpr,  # hidden width
        V: tl.constexpr,  # logical vocab size
        SOFTCAP: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Forward streaming logsumexp over vocab tiles.

        One program owns `BLOCK_M` token rows and scans all vocab tiles. It
        stores per-token loss and LSE only; no `(M,V)` logits tensor exists.
        """
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        target = tl.load(target_ptr + rows, mask=row_mask, other=IGNORE_INDEX)
        valid = row_mask & (target != IGNORE_INDEX) & (target >= 0) & (target < V)

        row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        target_logit = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for v_start in range(0, V, BLOCK_V):
            vocab = v_start + tl.arange(0, BLOCK_V)
            vocab_mask = vocab < V
            acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                ks = k_start + tl.arange(0, BLOCK_K)
                k_mask = ks < K
                x = tl.load(
                    x_ptr + rows[:, None] * K + ks[None, :],
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                w = tl.load(
                    weight_ptr + vocab[:, None] * K + ks[None, :],
                    mask=vocab_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(x.dtype)
                acc += tl.dot(x, tl.trans(w))

            tanh_raw = 2.0 * tl.sigmoid((2.0 / SOFTCAP) * acc) - 1.0
            logits = SOFTCAP * tanh_raw
            logits = tl.where(vocab_mask[None, :], logits, -float("inf"))

            tile_max = tl.max(logits, axis=1)
            new_max = tl.maximum(row_max, tile_max)
            row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(
                tl.exp(logits - new_max[:, None]),
                axis=1,
            )
            row_max = new_max

            is_target = vocab[None, :] == target[:, None]
            target_logit += tl.sum(tl.where(is_target, logits, 0.0), axis=1)

        lse = row_max + tl.log(row_sum)
        loss = tl.where(valid, lse - target_logit, 0.0)
        tl.store(loss_ptr + rows, loss, mask=row_mask)
        tl.store(lse_ptr + rows, lse, mask=row_mask)

    @triton.jit
    def _fused_cross_entropy_dx_bwd_kernel(
        x_ptr,  # (M, K), activation dtype - in
        weight_ptr,  # (V_PAD, K), weight dtype - in
        target_ptr,  # (M,), int64 - in
        lse_ptr,  # (M,), fp32 - in
        grad_loss_ptr,  # (M,), fp32 - in
        dx_ptr,  # (M, K), activation dtype - out
        M,  # int
        V_PAD: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        SOFTCAP: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Row-owned `dx = d_raw_logits @ weight` without materializing logits."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks_out = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = rows < M
        k_out_mask = ks_out < K

        target = tl.load(target_ptr + rows, mask=row_mask, other=IGNORE_INDEX)
        valid = row_mask & (target != IGNORE_INDEX) & (target >= 0) & (target < V)
        lse = tl.load(lse_ptr + rows, mask=row_mask, other=0.0)
        grad_loss = tl.load(grad_loss_ptr + rows, mask=row_mask, other=0.0)

        dx_acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for v_start in range(0, V, BLOCK_V):
            vocab = v_start + tl.arange(0, BLOCK_V)
            vocab_mask = vocab < V
            acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                ks = k_start + tl.arange(0, BLOCK_K)
                k_mask = ks < K
                x = tl.load(
                    x_ptr + rows[:, None] * K + ks[None, :],
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                w = tl.load(
                    weight_ptr + vocab[:, None] * K + ks[None, :],
                    mask=vocab_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(x.dtype)
                acc += tl.dot(x, tl.trans(w))

            tanh_raw = 2.0 * tl.sigmoid((2.0 / SOFTCAP) * acc) - 1.0
            logits = SOFTCAP * tanh_raw
            prob = tl.exp(logits - lse[:, None])
            one_hot = vocab[None, :] == target[:, None]
            dlogits = (prob - tl.where(one_hot, 1.0, 0.0)) * grad_loss[:, None]
            dlogits = tl.where(valid[:, None] & vocab_mask[None, :], dlogits, 0.0)
            draw = dlogits * (1.0 - tanh_raw * tanh_raw)

            w_out = tl.load(
                weight_ptr + vocab[:, None] * K + ks_out[None, :],
                mask=vocab_mask[:, None] & k_out_mask[None, :],
                other=0.0,
            ).to(x_ptr.dtype.element_ty)
            dx_acc += tl.dot(draw.to(x_ptr.dtype.element_ty), w_out)

        tl.store(
            dx_ptr + rows[:, None] * K + ks_out[None, :],
            dx_acc.to(dx_ptr.dtype.element_ty),
            mask=row_mask[:, None] & k_out_mask[None, :],
        )

    @triton.jit
    def _fused_cross_entropy_dweight_bwd_kernel(
        x_ptr,  # (M, K), activation dtype - in
        weight_ptr,  # (V_PAD, K), weight dtype - in
        target_ptr,  # (M,), int64 - in
        lse_ptr,  # (M,), fp32 - in
        grad_loss_ptr,  # (M,), fp32 - in
        dweight_ptr,  # (V_PAD, K), weight dtype - out
        M,  # int
        V_PAD: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        SOFTCAP: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Vocab-owned `d_weight = d_raw_logits.T @ x` without atomics."""
        pid_v = tl.program_id(0)
        pid_k = tl.program_id(1)
        vocab = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        ks_out = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        vocab_mask = vocab < V
        k_out_mask = ks_out < K

        dw_acc = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)
        for m_start in range(0, M, BLOCK_M):
            rows = m_start + tl.arange(0, BLOCK_M)
            row_mask = rows < M
            target = tl.load(target_ptr + rows, mask=row_mask, other=IGNORE_INDEX)
            valid = row_mask & (target != IGNORE_INDEX) & (target >= 0) & (target < V)
            lse = tl.load(lse_ptr + rows, mask=row_mask, other=0.0)
            grad_loss = tl.load(grad_loss_ptr + rows, mask=row_mask, other=0.0)

            acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                ks = k_start + tl.arange(0, BLOCK_K)
                k_mask = ks < K
                x = tl.load(
                    x_ptr + rows[:, None] * K + ks[None, :],
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                w = tl.load(
                    weight_ptr + vocab[:, None] * K + ks[None, :],
                    mask=vocab_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(x.dtype)
                acc += tl.dot(x, tl.trans(w))

            tanh_raw = 2.0 * tl.sigmoid((2.0 / SOFTCAP) * acc) - 1.0
            logits = SOFTCAP * tanh_raw
            prob = tl.exp(logits - lse[:, None])
            one_hot = vocab[None, :] == target[:, None]
            dlogits = (prob - tl.where(one_hot, 1.0, 0.0)) * grad_loss[:, None]
            dlogits = tl.where(valid[:, None] & vocab_mask[None, :], dlogits, 0.0)
            draw = dlogits * (1.0 - tanh_raw * tanh_raw)

            x_out = tl.load(
                x_ptr + rows[:, None] * K + ks_out[None, :],
                mask=row_mask[:, None] & k_out_mask[None, :],
                other=0.0,
            )
            dw_acc += tl.dot(tl.trans(draw.to(x_out.dtype)), x_out)

        tl.store(
            dweight_ptr + vocab[:, None] * K + ks_out[None, :],
            dw_acc.to(dweight_ptr.dtype.element_ty),
            mask=vocab_mask[:, None] & k_out_mask[None, :],
        )

    @triton.jit
    def _fused_cross_entropy_norm_fwd_kernel(
        x_ptr,  # (M, K), activation dtype - in: residual stream
        x_backout_ptr,  # (M, K), activation dtype - in: cached mid-layer stream
        backout_scale_ptr,  # (1,), scalar tensor - in
        weight_ptr,  # (V_PAD, K), fp32/bf16 - in: lm_head weight
        target_ptr,  # (M,), int64 - in
        loss_ptr,  # (M,), fp32 - out
        lse_ptr,  # (M,), fp32 - out
        rms_inv_ptr,  # (M,), fp32 - out
        M,  # int - row count
        V_PAD: tl.constexpr,  # physical weight rows
        K: tl.constexpr,  # hidden width
        V: tl.constexpr,  # logical vocab size
        SOFTCAP: tl.constexpr,
        EPS: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Forward fused backout mix + final RMSNorm + streaming CE."""
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        target = tl.load(target_ptr + rows, mask=row_mask, other=IGNORE_INDEX)
        valid = row_mask & (target != IGNORE_INDEX) & (target >= 0) & (target < V)
        backout_scale = tl.load(backout_scale_ptr).to(x_ptr.dtype.element_ty).to(tl.float32)

        sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            mask = row_mask[:, None] & k_mask[None, :]
            x = tl.load(
                x_ptr + rows[:, None] * K + ks[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            x_backout = tl.load(
                x_backout_ptr + rows[:, None] * K + ks[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            x_mix = x - backout_scale * x_backout
            sum_sq += tl.sum(x_mix * x_mix, axis=1)
        rms_inv = tl.rsqrt(sum_sq / K + EPS)
        tl.store(rms_inv_ptr + rows, rms_inv, mask=row_mask)

        row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        target_logit = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for v_start in range(0, V, BLOCK_V):
            vocab = v_start + tl.arange(0, BLOCK_V)
            vocab_mask = vocab < V
            acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                ks = k_start + tl.arange(0, BLOCK_K)
                k_mask = ks < K
                mask = row_mask[:, None] & k_mask[None, :]
                x = tl.load(
                    x_ptr + rows[:, None] * K + ks[None, :],
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                x_backout = tl.load(
                    x_backout_ptr + rows[:, None] * K + ks[None, :],
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                x_norm = ((x - backout_scale * x_backout) * rms_inv[:, None]).to(
                    x_ptr.dtype.element_ty
                )
                w = tl.load(
                    weight_ptr + vocab[:, None] * K + ks[None, :],
                    mask=vocab_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(x_ptr.dtype.element_ty)
                acc += tl.dot(x_norm, tl.trans(w))

            tanh_raw = 2.0 * tl.sigmoid((2.0 / SOFTCAP) * acc) - 1.0
            logits = SOFTCAP * tanh_raw
            logits = tl.where(vocab_mask[None, :], logits, -float("inf"))

            tile_max = tl.max(logits, axis=1)
            new_max = tl.maximum(row_max, tile_max)
            row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(
                tl.exp(logits - new_max[:, None]),
                axis=1,
            )
            row_max = new_max

            is_target = vocab[None, :] == target[:, None]
            target_logit += tl.sum(tl.where(is_target, logits, 0.0), axis=1)

        lse = row_max + tl.log(row_sum)
        loss = tl.where(valid, lse - target_logit, 0.0)
        tl.store(loss_ptr + rows, loss, mask=row_mask)
        tl.store(lse_ptr + rows, lse, mask=row_mask)

    @triton.jit
    def _fused_cross_entropy_norm_dx_bwd_kernel(
        x_ptr,  # (M, K), activation dtype - in
        x_backout_ptr,  # (M, K), activation dtype - in
        backout_scale_ptr,  # (1,), scalar tensor - in
        weight_ptr,  # (V_PAD, K), weight dtype - in
        target_ptr,  # (M,), int64 - in
        lse_ptr,  # (M,), fp32 - in
        rms_inv_ptr,  # (M,), fp32 - in
        grad_loss_ptr,  # (M,), fp32 - in
        d_x_norm_ptr,  # (M, K), activation dtype - out
        rms_row_inner_ptr,  # (M,), fp32 - in/out
        M,  # int
        V_PAD: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        SOFTCAP: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Compute d(final_norm_output) and its RMSNorm row inner."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks_out = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = rows < M
        k_out_mask = ks_out < K
        backout_scale = tl.load(backout_scale_ptr).to(x_ptr.dtype.element_ty).to(tl.float32)

        target = tl.load(target_ptr + rows, mask=row_mask, other=IGNORE_INDEX)
        valid = row_mask & (target != IGNORE_INDEX) & (target >= 0) & (target < V)
        lse = tl.load(lse_ptr + rows, mask=row_mask, other=0.0)
        rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0)
        grad_loss = tl.load(grad_loss_ptr + rows, mask=row_mask, other=0.0)

        dx_norm_acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for v_start in range(0, V, BLOCK_V):
            vocab = v_start + tl.arange(0, BLOCK_V)
            vocab_mask = vocab < V
            acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                ks = k_start + tl.arange(0, BLOCK_K)
                k_mask = ks < K
                mask = row_mask[:, None] & k_mask[None, :]
                x = tl.load(
                    x_ptr + rows[:, None] * K + ks[None, :],
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                x_backout = tl.load(
                    x_backout_ptr + rows[:, None] * K + ks[None, :],
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                x_norm = ((x - backout_scale * x_backout) * rms_inv[:, None]).to(
                    x_ptr.dtype.element_ty
                )
                w = tl.load(
                    weight_ptr + vocab[:, None] * K + ks[None, :],
                    mask=vocab_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(x_ptr.dtype.element_ty)
                acc += tl.dot(x_norm, tl.trans(w))

            tanh_raw = 2.0 * tl.sigmoid((2.0 / SOFTCAP) * acc) - 1.0
            logits = SOFTCAP * tanh_raw
            prob = tl.exp(logits - lse[:, None])
            one_hot = vocab[None, :] == target[:, None]
            dlogits = (prob - tl.where(one_hot, 1.0, 0.0)) * grad_loss[:, None]
            dlogits = tl.where(valid[:, None] & vocab_mask[None, :], dlogits, 0.0)
            draw = dlogits * (1.0 - tanh_raw * tanh_raw)

            w_out = tl.load(
                weight_ptr + vocab[:, None] * K + ks_out[None, :],
                mask=vocab_mask[:, None] & k_out_mask[None, :],
                other=0.0,
            ).to(x_ptr.dtype.element_ty)
            dx_norm_acc += tl.dot(draw.to(x_ptr.dtype.element_ty), w_out)

        d_x_norm = dx_norm_acc.to(d_x_norm_ptr.dtype.element_ty)
        tl.store(
            d_x_norm_ptr + rows[:, None] * K + ks_out[None, :],
            d_x_norm,
            mask=row_mask[:, None] & k_out_mask[None, :],
        )

        x = tl.load(
            x_ptr + rows[:, None] * K + ks_out[None, :],
            mask=row_mask[:, None] & k_out_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x_backout = tl.load(
            x_backout_ptr + rows[:, None] * K + ks_out[None, :],
            mask=row_mask[:, None] & k_out_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x_norm_out = (x - backout_scale * x_backout) * rms_inv[:, None]
        row_inner_partial = (
            tl.sum(d_x_norm.to(tl.float32) * x_norm_out, axis=1, dtype=tl.float32)
            / K
        )
        tl.atomic_add(
            rms_row_inner_ptr + rows,
            row_inner_partial,
            sem="relaxed",
            mask=row_mask,
        )

    @triton.jit
    def _fused_cross_entropy_norm_dweight_bwd_kernel(
        x_ptr,  # (M, K), activation dtype - in
        x_backout_ptr,  # (M, K), activation dtype - in
        backout_scale_ptr,  # (1,), scalar tensor - in
        weight_ptr,  # (V_PAD, K), weight dtype - in
        target_ptr,  # (M,), int64 - in
        lse_ptr,  # (M,), fp32 - in
        rms_inv_ptr,  # (M,), fp32 - in
        grad_loss_ptr,  # (M,), fp32 - in
        dweight_ptr,  # (V_PAD, K), weight dtype - out
        M,  # int
        V_PAD: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        SOFTCAP: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Vocab-owned d_weight for the final-norm fused CE path."""
        pid_v = tl.program_id(0)
        pid_k = tl.program_id(1)
        vocab = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        ks_out = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        vocab_mask = vocab < V
        k_out_mask = ks_out < K
        backout_scale = tl.load(backout_scale_ptr).to(x_ptr.dtype.element_ty).to(tl.float32)

        dw_acc = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)
        for m_start in range(0, M, BLOCK_M):
            rows = m_start + tl.arange(0, BLOCK_M)
            row_mask = rows < M
            target = tl.load(target_ptr + rows, mask=row_mask, other=IGNORE_INDEX)
            valid = row_mask & (target != IGNORE_INDEX) & (target >= 0) & (target < V)
            lse = tl.load(lse_ptr + rows, mask=row_mask, other=0.0)
            rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0)
            grad_loss = tl.load(grad_loss_ptr + rows, mask=row_mask, other=0.0)

            acc = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                ks = k_start + tl.arange(0, BLOCK_K)
                k_mask = ks < K
                mask = row_mask[:, None] & k_mask[None, :]
                x = tl.load(
                    x_ptr + rows[:, None] * K + ks[None, :],
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                x_backout = tl.load(
                    x_backout_ptr + rows[:, None] * K + ks[None, :],
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                x_norm = ((x - backout_scale * x_backout) * rms_inv[:, None]).to(
                    x_ptr.dtype.element_ty
                )
                w = tl.load(
                    weight_ptr + vocab[:, None] * K + ks[None, :],
                    mask=vocab_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(x_ptr.dtype.element_ty)
                acc += tl.dot(x_norm, tl.trans(w))

            tanh_raw = 2.0 * tl.sigmoid((2.0 / SOFTCAP) * acc) - 1.0
            logits = SOFTCAP * tanh_raw
            prob = tl.exp(logits - lse[:, None])
            one_hot = vocab[None, :] == target[:, None]
            dlogits = (prob - tl.where(one_hot, 1.0, 0.0)) * grad_loss[:, None]
            dlogits = tl.where(valid[:, None] & vocab_mask[None, :], dlogits, 0.0)
            draw = dlogits * (1.0 - tanh_raw * tanh_raw)

            x = tl.load(
                x_ptr + rows[:, None] * K + ks_out[None, :],
                mask=row_mask[:, None] & k_out_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            x_backout = tl.load(
                x_backout_ptr + rows[:, None] * K + ks_out[None, :],
                mask=row_mask[:, None] & k_out_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            x_norm_out = ((x - backout_scale * x_backout) * rms_inv[:, None]).to(
                x_ptr.dtype.element_ty
            )
            dw_acc += tl.dot(tl.trans(draw.to(x_ptr.dtype.element_ty)), x_norm_out)

        tl.store(
            dweight_ptr + vocab[:, None] * K + ks_out[None, :],
            dw_acc.to(dweight_ptr.dtype.element_ty),
            mask=vocab_mask[:, None] & k_out_mask[None, :],
        )

    @triton.jit
    def _fused_cross_entropy_norm_mix_bwd_kernel(
        d_x_norm_ptr,  # (M, K), activation dtype - in
        x_ptr,  # (M, K), activation dtype - in
        x_backout_ptr,  # (M, K), activation dtype - in
        backout_scale_ptr,  # (1,), scalar tensor - in
        rms_inv_ptr,  # (M,), fp32 - in
        rms_row_inner_ptr,  # (M,), fp32 - in
        dx_ptr,  # (M, K), activation dtype - out
        dx_backout_ptr,  # (M, K), activation dtype - out
        d_backout_scale_ptr,  # (1,), fp32/param dtype - out
        M,  # int
        K,  # int
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Final RMSNorm backward followed by x - scale*x_backout backward."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = rows < M
        k_mask = ks < K
        mask = row_mask[:, None] & k_mask[None, :]
        offs = rows[:, None] * K + ks[None, :]

        backout_scale = tl.load(backout_scale_ptr).to(x_ptr.dtype.element_ty).to(tl.float32)
        rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0)
        row_inner = tl.load(rms_row_inner_ptr + rows, mask=row_mask, other=0.0)
        d_x_norm = tl.load(d_x_norm_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x_backout = tl.load(x_backout_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x_norm = (x - backout_scale * x_backout) * rms_inv[:, None]
        dx_mix = rms_inv[:, None] * (d_x_norm - x_norm * row_inner[:, None])

        tl.store(dx_ptr + offs, dx_mix.to(dx_ptr.dtype.element_ty), mask=mask)
        tl.store(
            dx_backout_ptr + offs,
            (-backout_scale * dx_mix).to(dx_backout_ptr.dtype.element_ty),
            mask=mask,
        )
        d_scale_tile = tl.sum(dx_mix * (-x_backout), axis=1)
        tl.atomic_add(
            d_backout_scale_ptr,
            tl.sum(d_scale_tile, axis=0),
            sem="relaxed",
        )


def _fused_cross_entropy_fwd_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    vocab_size: int,
    softcap: float,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token loss and saved LSE for flattened `(M,K)` inputs."""
    if not _HAS_TRITON:
        raise RuntimeError("fused_cross_entropy requires triton")
    assert x.is_cuda and weight.is_cuda and target.is_cuda
    assert x.is_contiguous() and weight.is_contiguous() and target.is_contiguous()
    assert x.ndim == 2 and weight.ndim == 2 and target.ndim == 1
    M, K = x.shape
    v_pad, K_w = weight.shape
    assert K == K_w
    assert 0 < vocab_size <= v_pad
    assert target.shape == (M,)

    loss = torch.empty((M,), dtype=torch.float32, device=x.device)
    lse = torch.empty((M,), dtype=torch.float32, device=x.device)
    block_m = _FUSED_CROSS_ENTROPY_FWD_BLOCK_M
    grid = (triton.cdiv(M, block_m),)
    wrap_triton(_fused_cross_entropy_fwd_kernel)[grid](
        x,
        weight,
        target,
        loss,
        lse,
        M,
        V_PAD=v_pad,
        K=K,
        V=vocab_size,
        SOFTCAP=softcap,
        IGNORE_INDEX=ignore_index,
        BLOCK_M=block_m,
        BLOCK_V=_FUSED_CROSS_ENTROPY_FWD_BLOCK_V,
        BLOCK_K=_FUSED_CROSS_ENTROPY_FWD_BLOCK_K,
        num_warps=_FUSED_CROSS_ENTROPY_FWD_NUM_WARPS,
        num_stages=_FUSED_CROSS_ENTROPY_FWD_NUM_STAGES,
    )
    return loss, lse


def _fused_cross_entropy_bwd_impl(
    grad_loss: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    vocab_size: int,
    softcap: float,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(dx, d_weight)` for flattened `(M,K)` inputs."""
    if not _HAS_TRITON:
        raise RuntimeError("fused_cross_entropy backward requires triton")
    grad_loss = grad_loss.contiguous()
    assert grad_loss.shape == target.shape
    M, K = x.shape
    v_pad, _K_w = weight.shape

    dx = torch.empty_like(x)
    # Zero padding rows when `weight` has more rows than the logical vocab.
    dweight = torch.zeros_like(weight)

    dx_block_m = _FUSED_CROSS_ENTROPY_DX_BLOCK_M
    dx_block_k = _FUSED_CROSS_ENTROPY_DX_BLOCK_K
    dx_grid = (triton.cdiv(M, dx_block_m), triton.cdiv(K, dx_block_k))
    wrap_triton(_fused_cross_entropy_dx_bwd_kernel)[dx_grid](
        x,
        weight,
        target,
        lse,
        grad_loss,
        dx,
        M,
        V_PAD=v_pad,
        K=K,
        V=vocab_size,
        SOFTCAP=softcap,
        IGNORE_INDEX=ignore_index,
        BLOCK_M=dx_block_m,
        BLOCK_K=dx_block_k,
        BLOCK_V=_FUSED_CROSS_ENTROPY_DX_BLOCK_V,
        num_warps=_FUSED_CROSS_ENTROPY_DX_NUM_WARPS,
        num_stages=_FUSED_CROSS_ENTROPY_DX_NUM_STAGES,
    )

    dw_block_v = _FUSED_CROSS_ENTROPY_DW_BLOCK_V
    dw_block_k = _FUSED_CROSS_ENTROPY_DW_BLOCK_K
    dw_grid = (triton.cdiv(vocab_size, dw_block_v), triton.cdiv(K, dw_block_k))
    wrap_triton(_fused_cross_entropy_dweight_bwd_kernel)[dw_grid](
        x,
        weight,
        target,
        lse,
        grad_loss,
        dweight,
        M,
        V_PAD=v_pad,
        K=K,
        V=vocab_size,
        SOFTCAP=softcap,
        IGNORE_INDEX=ignore_index,
        BLOCK_V=dw_block_v,
        BLOCK_K=dw_block_k,
        BLOCK_M=_FUSED_CROSS_ENTROPY_DW_BLOCK_M,
        num_warps=_FUSED_CROSS_ENTROPY_DW_NUM_WARPS,
        num_stages=_FUSED_CROSS_ENTROPY_DW_NUM_STAGES,
    )
    return dx, dweight


def _fused_cross_entropy_norm_fwd_impl(
    x: torch.Tensor,
    x_backout: torch.Tensor,
    backout_scale: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    vocab_size: int,
    softcap: float,
    eps: float,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(loss_per_token, lse, rms_inv)` for the fused final-tail path.

    Inputs are flattened to `(M,K)` before this helper is called. `x` is the
    residual stream before final backout/RMSNorm, and `x_backout` is the saved
    mid-layer stream subtracted with scalar `backout_scale`.
    """
    if not _HAS_TRITON:
        raise RuntimeError("fused_cross_entropy final-tail path requires triton")
    assert x.is_cuda and x_backout.is_cuda and backout_scale.is_cuda
    assert weight.is_cuda and target.is_cuda
    assert x.is_contiguous() and x_backout.is_contiguous()
    assert backout_scale.is_contiguous()
    assert weight.is_contiguous() and target.is_contiguous()
    assert x.ndim == 2 and x_backout.ndim == 2 and weight.ndim == 2 and target.ndim == 1
    assert backout_scale.numel() == 1
    M, K = x.shape
    v_pad, K_w = weight.shape
    assert x_backout.shape == x.shape
    assert K == K_w
    assert 0 < vocab_size <= v_pad
    assert target.shape == (M,)

    loss = torch.empty((M,), dtype=torch.float32, device=x.device)
    lse = torch.empty((M,), dtype=torch.float32, device=x.device)
    rms_inv = torch.empty((M,), dtype=torch.float32, device=x.device)
    block_m = _FUSED_CROSS_ENTROPY_FWD_BLOCK_M
    grid = (triton.cdiv(M, block_m),)
    wrap_triton(_fused_cross_entropy_norm_fwd_kernel)[grid](
        x,
        x_backout,
        backout_scale,
        weight,
        target,
        loss,
        lse,
        rms_inv,
        M,
        V_PAD=v_pad,
        K=K,
        V=vocab_size,
        SOFTCAP=softcap,
        EPS=eps,
        IGNORE_INDEX=ignore_index,
        BLOCK_M=block_m,
        BLOCK_V=_FUSED_CROSS_ENTROPY_FWD_BLOCK_V,
        BLOCK_K=_FUSED_CROSS_ENTROPY_FWD_BLOCK_K,
        num_warps=_FUSED_CROSS_ENTROPY_FWD_NUM_WARPS,
        num_stages=_FUSED_CROSS_ENTROPY_FWD_NUM_STAGES,
    )
    return loss, lse, rms_inv


def _fused_cross_entropy_norm_bwd_impl(
    grad_loss: torch.Tensor,
    x: torch.Tensor,
    x_backout: torch.Tensor,
    backout_scale: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    rms_inv: torch.Tensor,
    vocab_size: int,
    softcap: float,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(dx, dx_backout, d_backout_scale, d_weight)` for final-tail CE."""
    if not _HAS_TRITON:
        raise RuntimeError("fused_cross_entropy final-tail backward requires triton")
    grad_loss = grad_loss.contiguous()
    assert grad_loss.shape == target.shape
    assert x.shape == x_backout.shape
    assert backout_scale.numel() == 1
    M, K = x.shape
    v_pad, _K_w = weight.shape

    d_x_norm = torch.empty_like(x)
    rms_row_inner = torch.zeros((M,), dtype=torch.float32, device=x.device)
    dx = torch.empty_like(x)
    dx_backout = torch.empty_like(x_backout)
    d_backout_scale = torch.zeros_like(backout_scale)
    # Zero padding rows when `weight` has more rows than the logical vocab.
    dweight = torch.zeros_like(weight)

    dx_block_m = _FUSED_CROSS_ENTROPY_DX_BLOCK_M
    dx_block_k = _FUSED_CROSS_ENTROPY_DX_BLOCK_K
    dx_grid = (triton.cdiv(M, dx_block_m), triton.cdiv(K, dx_block_k))
    wrap_triton(_fused_cross_entropy_norm_dx_bwd_kernel)[dx_grid](
        x,
        x_backout,
        backout_scale,
        weight,
        target,
        lse,
        rms_inv,
        grad_loss,
        d_x_norm,
        rms_row_inner,
        M,
        V_PAD=v_pad,
        K=K,
        V=vocab_size,
        SOFTCAP=softcap,
        IGNORE_INDEX=ignore_index,
        BLOCK_M=dx_block_m,
        BLOCK_K=dx_block_k,
        BLOCK_V=_FUSED_CROSS_ENTROPY_DX_BLOCK_V,
        num_warps=_FUSED_CROSS_ENTROPY_DX_NUM_WARPS,
        num_stages=_FUSED_CROSS_ENTROPY_DX_NUM_STAGES,
    )

    dw_block_v = _FUSED_CROSS_ENTROPY_DW_BLOCK_V
    dw_block_k = _FUSED_CROSS_ENTROPY_DW_BLOCK_K
    dw_grid = (triton.cdiv(vocab_size, dw_block_v), triton.cdiv(K, dw_block_k))
    wrap_triton(_fused_cross_entropy_norm_dweight_bwd_kernel)[dw_grid](
        x,
        x_backout,
        backout_scale,
        weight,
        target,
        lse,
        rms_inv,
        grad_loss,
        dweight,
        M,
        V_PAD=v_pad,
        K=K,
        V=vocab_size,
        SOFTCAP=softcap,
        IGNORE_INDEX=ignore_index,
        BLOCK_V=dw_block_v,
        BLOCK_K=dw_block_k,
        BLOCK_M=_FUSED_CROSS_ENTROPY_DW_BLOCK_M,
        num_warps=_FUSED_CROSS_ENTROPY_DW_NUM_WARPS,
        num_stages=_FUSED_CROSS_ENTROPY_DW_NUM_STAGES,
    )

    mix_block_m = _FUSED_CROSS_ENTROPY_NORM_BWD_BLOCK_M
    mix_block_k = _FUSED_CROSS_ENTROPY_NORM_BWD_BLOCK_K
    mix_grid = (triton.cdiv(M, mix_block_m), triton.cdiv(K, mix_block_k))
    wrap_triton(_fused_cross_entropy_norm_mix_bwd_kernel)[mix_grid](
        d_x_norm,
        x,
        x_backout,
        backout_scale,
        rms_inv,
        rms_row_inner,
        dx,
        dx_backout,
        d_backout_scale,
        M,
        K,
        BLOCK_M=mix_block_m,
        BLOCK_K=mix_block_k,
        num_warps=_FUSED_CROSS_ENTROPY_NORM_BWD_NUM_WARPS,
    )
    return dx, dx_backout, d_backout_scale, dweight


@torch.library.triton_op(
    "nanoops::fused_cross_entropy_fwd",
    mutates_args=(),
)
def _fused_cross_entropy_fwd_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    vocab_size: int,
    softcap: float,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-op forward wrapper returning `(loss_per_token, lse)`."""
    return _fused_cross_entropy_fwd_impl(
        x,
        weight,
        target,
        vocab_size,
        softcap,
        ignore_index,
    )


@torch.library.triton_op(
    "nanoops::fused_cross_entropy_bwd",
    mutates_args=(),
)
def _fused_cross_entropy_bwd_op(
    grad_loss: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    vocab_size: int,
    softcap: float,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-op backward wrapper returning `(dx, d_weight)`."""
    return _fused_cross_entropy_bwd_impl(
        grad_loss,
        x,
        weight,
        target,
        lse,
        vocab_size,
        softcap,
        ignore_index,
    )


def _fused_cross_entropy_setup_context(
    ctx: Any,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, int],
    output: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Save inputs and LSE for backward recomputation."""
    x, weight, target, vocab_size, softcap, ignore_index = inputs
    _loss, lse = output
    ctx.save_for_backward(x, weight, target, lse)
    ctx.vocab_size = vocab_size
    ctx.softcap = softcap
    ctx.ignore_index = ignore_index


def _fused_cross_entropy_autograd_backward(
    ctx: Any,
    grad_loss: torch.Tensor,
    _grad_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None]:
    """Autograd callback for `nanoops::fused_cross_entropy_fwd`."""
    x, weight, target, lse = ctx.saved_tensors
    dx, dweight = _fused_cross_entropy_bwd_op(
        grad_loss,
        x,
        weight,
        target,
        lse,
        ctx.vocab_size,
        ctx.softcap,
        ctx.ignore_index,
    )
    return dx, dweight, None, None, None, None


_fused_cross_entropy_fwd_op.register_autograd(
    _fused_cross_entropy_autograd_backward,
    setup_context=_fused_cross_entropy_setup_context,
)


@torch.library.triton_op(
    "nanoops::fused_cross_entropy_norm_fwd",
    mutates_args=(),
)
def _fused_cross_entropy_norm_fwd_op(
    x: torch.Tensor,
    x_backout: torch.Tensor,
    backout_scale: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    vocab_size: int,
    softcap: float,
    eps: float,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-op final-tail forward returning `(loss_per_token, lse, rms_inv)`."""
    return _fused_cross_entropy_norm_fwd_impl(
        x,
        x_backout,
        backout_scale,
        weight,
        target,
        vocab_size,
        softcap,
        eps,
        ignore_index,
    )


@torch.library.triton_op(
    "nanoops::fused_cross_entropy_norm_bwd",
    mutates_args=(),
)
def _fused_cross_entropy_norm_bwd_op(
    grad_loss: torch.Tensor,
    x: torch.Tensor,
    x_backout: torch.Tensor,
    backout_scale: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    rms_inv: torch.Tensor,
    vocab_size: int,
    softcap: float,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-op final-tail backward returning `(dx, dx_backout, d_scale, d_weight)`."""
    return _fused_cross_entropy_norm_bwd_impl(
        grad_loss,
        x,
        x_backout,
        backout_scale,
        weight,
        target,
        lse,
        rms_inv,
        vocab_size,
        softcap,
        ignore_index,
    )


def _fused_cross_entropy_norm_setup_context(
    ctx: Any,
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        float,
        float,
        int,
    ],
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Save final-tail inputs plus LSE/RMS inverse for backward recomputation."""
    x, x_backout, backout_scale, weight, target, vocab_size, softcap, _eps, ignore_index = inputs
    _loss, lse, rms_inv = output
    ctx.save_for_backward(x, x_backout, backout_scale, weight, target, lse, rms_inv)
    ctx.vocab_size = vocab_size
    ctx.softcap = softcap
    ctx.ignore_index = ignore_index


def _fused_cross_entropy_norm_autograd_backward(
    ctx: Any,
    grad_loss: torch.Tensor,
    _grad_lse: torch.Tensor,
    _grad_rms_inv: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    None,
    None,
    None,
    None,
    None,
]:
    """Autograd callback for `nanoops::fused_cross_entropy_norm_fwd`."""
    x, x_backout, backout_scale, weight, target, lse, rms_inv = ctx.saved_tensors
    dx, dx_backout, d_backout_scale, dweight = _fused_cross_entropy_norm_bwd_op(
        grad_loss,
        x,
        x_backout,
        backout_scale,
        weight,
        target,
        lse,
        rms_inv,
        ctx.vocab_size,
        ctx.softcap,
        ctx.ignore_index,
    )
    return dx, dx_backout, d_backout_scale, dweight, None, None, None, None, None


_fused_cross_entropy_norm_fwd_op.register_autograd(
    _fused_cross_entropy_norm_autograd_backward,
    setup_context=_fused_cross_entropy_norm_setup_context,
)


def fused_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    vocab_size: int,
    softcap: float = 15.0,
    ignore_index: int = -1,
    reduction: str = "mean",
    x_backout: torch.Tensor | None = None,
    backout_scale: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused GPT loss tail.

    Args:
      x: `(B, T, K)` or `(M, K)` contiguous activation tensor. Without
        `x_backout`, this is already final-normalized. With `x_backout`, this
        is the residual stream before final backout/RMSNorm.
      weight: `(V_pad, K)` contiguous lm_head weight. Only the first
        `vocab_size` rows participate; padding rows get zero grad.
      target: integer targets shaped like `x.shape[:-1]`.
      vocab_size: logical vocabulary size before padding.
      softcap: logits are capped as `softcap * tanh(raw / softcap)`.
      ignore_index: target value that contributes zero loss and grad.
      reduction: `"mean"`, `"sum"`, or `"none"`.
      x_backout: optional `(B,T,K)`/`(M,K)` tensor for final
        `x - backout_scale * x_backout`.
      backout_scale: optional scalar tensor used with `x_backout`.
      eps: RMSNorm epsilon for the final-tail path.

    Returns:
      Scalar loss for `"mean"`/`"sum"` or per-token loss with shape
      `target.shape` for `"none"`.
    """
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"unknown reduction: {reduction!r}")
    assert x.is_contiguous() and weight.is_contiguous()
    assert target.is_contiguous()
    assert x.shape[:-1] == target.shape
    if (x_backout is None) != (backout_scale is None):
        raise ValueError("x_backout and backout_scale must be provided together")
    if x_backout is not None:
        assert x_backout.is_contiguous()
        assert x_backout.shape == x.shape
        assert backout_scale is not None and backout_scale.is_contiguous()
        assert backout_scale.numel() == 1
    K = x.shape[-1]
    x_2d = x.view(-1, K)
    target_1d = target.view(-1)
    x_backout_2d = x_backout.view(-1, K) if x_backout is not None else None

    if not (_HAS_TRITON and x.is_cuda and weight.is_cuda and target.is_cuda):
        if x_backout_2d is not None:
            assert backout_scale is not None
            x_2d = x_2d - backout_scale.to(dtype=x.dtype) * x_backout_2d
            x_2d = F.rms_norm(x_2d, (K,), eps=eps)
        logits = x_2d @ weight[:vocab_size].to(dtype=x.dtype).t()
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        loss = F.cross_entropy(
            logits,
            target_1d,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        return loss.view_as(target) if reduction == "none" else loss

    if x_backout_2d is not None:
        assert backout_scale is not None
        # Keep this materialization outside the streaming CE kernels. Pulling
        # it into the vocab-tiled kernels repeats x/x_backout loads and RMSNorm
        # work once per vocab tile (512x for d24), which dominates the step.
        x_2d = x_2d - backout_scale.to(dtype=x.dtype) * x_backout_2d
        x_2d = F.rms_norm(x_2d, (K,), eps=eps).contiguous()

    per_token, _lse = _fused_cross_entropy_fwd_op(
        x_2d,
        weight,
        target_1d,
        vocab_size,
        softcap,
        ignore_index,
    )
    if reduction == "none":
        return per_token.view_as(target)
    if reduction == "sum":
        return per_token.sum()
    valid_count = (target_1d != ignore_index).sum().to(per_token.dtype)
    return per_token.sum() / valid_count
