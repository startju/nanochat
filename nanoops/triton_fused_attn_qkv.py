"""Attention QKV-side Triton kernels for nanoops.

Contains:
  - `fused_attn_qkv_projection`: fused residual/x0 blend,
    no-affine outer RMSNorm, Q/K/V projection, Q/K rotary + QK RMSNorm,
    and optional gated value-embedding lookup. Backward is implemented with
    Triton kernels, including optional value-embedding gate gradients.
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


# d24 compile-path tuning constants. Kept module-local so benchmark harnesses
# can monkeypatch them before torch.compile without editing kernel code.
_QKV_FWD_VE_BLOCK_M = 64
_QKV_FWD_VE_BLOCK_K = 16
_QKV_FWD_VE_NUM_WARPS = 4
_QKV_FWD_VE_NUM_STAGES = 3
_QKV_FWD_NO_VE_BLOCK_M = 128
_QKV_FWD_NO_VE_BLOCK_K = 32
_QKV_FWD_NO_VE_NUM_WARPS = 4
_QKV_FWD_NO_VE_NUM_STAGES = 1

_QKV_BWD_QK_PRE_VE_BLOCK_M = 8
_QKV_BWD_QK_PRE_VE_NUM_WARPS = 8
_QKV_BWD_QK_PRE_VE_NUM_STAGES = 1
_QKV_BWD_QK_PRE_NO_VE_BLOCK_M = 32
_QKV_BWD_QK_PRE_NO_VE_NUM_WARPS = 4
_QKV_BWD_QK_PRE_NO_VE_NUM_STAGES = 1

_QKV_BWD_DX_HAT_VE_BLOCK_M = 128
_QKV_BWD_DX_HAT_VE_BLOCK_K = 64
_QKV_BWD_DX_HAT_VE_HEAD_SPLIT = 4
_QKV_BWD_DX_HAT_VE_NUM_WARPS = 4
_QKV_BWD_DX_HAT_VE_NUM_STAGES = 1
_QKV_BWD_DX_HAT_NO_VE_BLOCK_M = 128
_QKV_BWD_DX_HAT_NO_VE_BLOCK_K = 64
_QKV_BWD_DX_HAT_NO_VE_HEAD_SPLIT = 4
_QKV_BWD_DX_HAT_NO_VE_NUM_WARPS = 4
_QKV_BWD_DX_HAT_NO_VE_NUM_STAGES = 1

_QKV_BWD_X_NORM_BLOCK_M = 128
_QKV_BWD_X_NORM_BLOCK_K = 64
_QKV_BWD_X_NORM_NUM_WARPS = 4
_QKV_BWD_OUTER_RMS_BLOCK_M = 128
_QKV_BWD_OUTER_RMS_BLOCK_K = 64
_QKV_BWD_OUTER_RMS_NUM_WARPS = 4
_QKV_BWD_WEIGHT_GRAD_BLOCK_N = 64
_QKV_BWD_WEIGHT_GRAD_BLOCK_K = 128
_QKV_BWD_WEIGHT_GRAD_BLOCK_M = 32
_QKV_BWD_WEIGHT_GRAD_NUM_WARPS = 4
_QKV_BWD_WEIGHT_GRAD_NUM_STAGES = 2


# ─────────────────────────────────────────────────────────────────────
# Fused residual mix + RMSNorm + QKV projection.
#
# nanochat's attention forward looks like:
#     x = resid_lambdas[i] * x + x0_lambdas[i] * x0
#     x_norm = norm(x)                       (RMSNorm at Block level)
#     q = c_q(x_norm); k = c_k(x_norm); v = c_v(x_norm)        ← FUSED HERE
#     q, k = apply_rotary(q, k, cos_sin)
#     q, k = norm(q), norm(k)
#     q, k = q * 1.2, k * 1.2
#     y = sliding_window_sdpa(q, k, v, window_size)
#     y = c_proj(y)
#
# This path materializes the OUTER RMSNorm output once, then shares it
# across the Q/K/V projection kernels. Q/K immediately apply rotary +
# QK RMSNorm + scale before being written.
# ─────────────────────────────────────────────────────────────────────


if _HAS_TRITON:
    @triton.jit
    def _fused_attn_qkv_projection_norm_fwd_kernel(
        x_ptr,  # (M, K), dtype=x.dtype — in: previous layer residual stream
        x0_ptr,  # (M, K), dtype=x.dtype — in: initial embedding stream
        resid_scale_ptr,  # scalar tensor — in: resid_lambdas[i]
        x0_scale_ptr,  # scalar tensor — in: x0_lambdas[i]
        x_mix_ptr,  # (M, K), dtype=x.dtype — out: resid*x + x0_scale*x0
        x_hat_ptr,  # (M, K), dtype=x.dtype — out: RMSNorm(x_mix)
        rms_inv_ptr,  # (M,) fp32 — out: outer RMSNorm inverse
        M,  # int — row count after flattening B*T
        K,  # int — hidden width
        eps,  # float — RMSNorm epsilon
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused residual/x0 blend + RMSNorm for the QKV input."""
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_K)
        row_mask = rows < M
        col_mask = cols < K
        mask = row_mask[:, None] & col_mask[None, :]
        offs = rows[:, None] * K + cols[None, :]

        resid_scale = tl.load(resid_scale_ptr).to(x_ptr.dtype.element_ty)
        x0_scale = tl.load(x0_scale_ptr).to(x_ptr.dtype.element_ty)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x0 = tl.load(x0_ptr + offs, mask=mask, other=0.0)
        x_mix = x * resid_scale + x0 * x0_scale
        tl.store(x_mix_ptr + offs, x_mix, mask=mask)

        sum_sq = tl.sum(x_mix * x_mix, axis=1, dtype=tl.float32)
        rms_inv = tl.rsqrt(sum_sq / K + eps)
        tl.store(rms_inv_ptr + rows, rms_inv, mask=row_mask)

        x_hat = x_mix * rms_inv[:, None]
        tl.store(x_hat_ptr + offs, x_hat.to(x_hat_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _fused_attn_qkv_projection_qkv_fwd_kernel(
        x_ptr,  # (M, K), dtype=x.dtype — in: materialized outer RMSNorm output
        q_w_ptr,  # (n_head * D, K), dtype=q_weight.dtype — in: Q projection weight
        k_w_ptr,  # (n_kv_head * D, K), dtype=k_weight.dtype — in: K projection weight
        v_w_ptr,  # (n_kv_head * D, K), dtype=v_weight.dtype — in: V projection weight
        ve_ids_ptr,  # (M,), int64 — optional in: token ids for value embedding lookup
        ve_w_ptr,  # (vocab, n_kv_head * D), dtype=ve_weight.dtype — optional VE table
        ve_gate_w_ptr,  # (n_kv_head, VE_GATE_CH), dtype=ve_gate_weight.dtype — optional gate
        cos_ptr,  # (1, T, 1, D/2), dtype=cos.dtype — in: rotary cosine table
        sin_ptr,  # (1, T, 1, D/2), dtype=sin.dtype — in: rotary sine table
        q_ptr,  # (M, n_head, D), dtype=x.dtype — out: Q after rotary + QK norm + scale
        k_ptr,  # (M, n_kv_head, D), dtype=x.dtype — out: K after rotary + QK norm + scale
        v_ptr,  # (M, n_kv_head, D), dtype=x.dtype — out: projected V plus optional VE
        qk_rms_inv_ptr,  # (M, n_head + n_kv_head), fp32 — out: per Q/K-head RMS inverse
        M,  # int — row count after flattening batch/time
        K,  # int — input hidden width
        D,  # int — per-head width
        n_head,  # int — number of Q heads
        n_kv_head,  # int — number of K/V heads
        rotary_seq_len,  # int — rotary table sequence length before batch broadcast
        eps,  # float — outer and QK RMSNorm epsilon
        scale,  # float — post-QK-norm scalar multiplier
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_VALUE_EMBEDDING: tl.constexpr,
        VE_GATE_CH: tl.constexpr,
        VE_GATE_BLOCK: tl.constexpr,
    ):
        """Project one Q/K/V head tile selected by the second grid axis.

        A part is one projected head: first Q heads, then K heads, then V
        heads. The input is already outer-RMS-normalized; V exits before
        rotary and QK RMSNorm, after optionally adding the ResFormer value
        embedding gate.
        """
        pid_m = tl.program_id(0)
        part = tl.program_id(1)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        half_cols = tl.arange(0, BLOCK_D // 2)
        row_mask = rows < M

        is_q = part < n_head
        is_k = (part >= n_head) & (part < n_head + n_kv_head)
        is_v = part >= n_head + n_kv_head
        head = tl.where(
            is_q,
            part,
            tl.where(is_k, part - n_head, part - n_head - n_kv_head),
        )
        weight_rows = head * D + cols

        # Project one full head with a single dot in ordinary column order:
        # [lo0, lo1, ..., hi0, hi1, ...]. Q/K split the accumulator into
        # rotary halves in registers; V can store the same order directly.
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            x_ptrs = x_ptr + rows[:, None] * K + ks[None, :]
            x = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            if is_q:
                w_ptrs = q_w_ptr + weight_rows[:, None] * K + ks[None, :]
            elif is_k:
                w_ptrs = k_w_ptr + weight_rows[:, None] * K + ks[None, :]
            else:
                w_ptrs = v_w_ptr + weight_rows[:, None] * K + ks[None, :]
            w = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0).to(x_ptr.dtype.element_ty)
            acc += tl.dot(x, tl.trans(w))

        if is_v:
            v_out = acc.to(x_ptr.dtype.element_ty)
            if HAS_VALUE_EMBEDDING:
                gate_cols = tl.arange(0, VE_GATE_BLOCK)
                gate_mask = gate_cols < VE_GATE_CH
                gate_x = tl.load(
                    x_ptr + rows[:, None] * K + gate_cols[None, :],
                    mask=row_mask[:, None] & gate_mask[None, :],
                    other=0.0,
                )
                gate_w = tl.load(
                    ve_gate_w_ptr + head * VE_GATE_CH + gate_cols,
                    mask=gate_mask,
                    other=0.0,
                ).to(x_ptr.dtype.element_ty)
                gate_logits = tl.sum(gate_x * gate_w[None, :], axis=1, dtype=tl.float32)
                gate = 3 * tl.sigmoid(gate_logits).to(x_ptr.dtype.element_ty)
                token_ids = tl.load(ve_ids_ptr + rows, mask=row_mask, other=0)
                ve_ptrs = (
                    ve_w_ptr
                    + token_ids[:, None] * n_kv_head * D
                    + head * D
                    + cols[None, :]
                )
                ve = tl.load(
                    ve_ptrs,
                    mask=row_mask[:, None],
                    other=0.0,
                )
                v_out += gate[:, None] * ve
            v_ptrs = (
                v_ptr
                + rows[:, None] * n_kv_head * D
                + head * D
                + cols[None, :]
            )
            tl.store(
                v_ptrs,
                v_out,
                mask=row_mask[:, None],
            )
            return

        acc_halves = tl.reshape(acc.to(x_ptr.dtype.element_ty), (BLOCK_M, 2, BLOCK_D // 2))
        acc_lo, acc_hi = tl.split(tl.trans(acc_halves, 0, 2, 1))
        rotary_rows = rows % rotary_seq_len

        cos = tl.load(
            cos_ptr + rotary_rows[:, None] * (BLOCK_D // 2) + half_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        sin = tl.load(
            sin_ptr + rotary_rows[:, None] * (BLOCK_D // 2) + half_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        rot_lo = acc_lo * cos + acc_hi * sin
        rot_hi = -acc_lo * sin + acc_hi * cos
        rot_pair = tl.join(rot_lo, rot_hi)
        qk_halves = tl.trans(rot_pair, 0, 2, 1)
        qk = tl.reshape(qk_halves, (BLOCK_M, BLOCK_D))
        qk_sum_sq = tl.sum(qk * qk, axis=1, dtype=tl.float32)
        qk_rms_inv = tl.rsqrt(qk_sum_sq / D + eps)
        norm_scale = qk_rms_inv * scale
        qk = qk * norm_scale[:, None]

        if is_q:
            q_ptrs = (
                q_ptr
                + rows[:, None] * n_head * D
                + head * D
                + cols[None, :]
            )
            tl.store(
                qk_rms_inv_ptr + rows * (n_head + n_kv_head) + head,
                qk_rms_inv,
                mask=row_mask,
            )
            tl.store(q_ptrs, qk.to(q_ptr.dtype.element_ty), mask=row_mask[:, None])
        else:
            k_ptrs = (
                k_ptr
                + rows[:, None] * n_kv_head * D
                + head * D
                + cols[None, :]
            )
            tl.store(
                qk_rms_inv_ptr + rows * (n_head + n_kv_head) + n_head + head,
                qk_rms_inv,
                mask=row_mask,
            )
            tl.store(k_ptrs, qk.to(k_ptr.dtype.element_ty), mask=row_mask[:, None])

    # Backward kernels follow the materialized-pregrad path:
    #   1. rematerialize RMSNorm(x_mix) once as x_norm,
    #   2. recover d_q0/d_k0 and VE side gradients,
    #   3. compute/materialize d_x_hat and outer RMS row inner products,
    #   4. finish outer RMSNorm d_x,
    #   5. compute Q/K/V projection weight gradients from the materialized grads.

    @triton.jit
    def _fused_attn_qkv_projection_x_norm_bwd_kernel(
        x_base_ptr,  # (M, K) — in: residual stream before mix
        x0_ptr,  # (M, K) — in: initial embedding stream
        resid_scale_ptr,  # scalar — in
        x0_scale_ptr,  # scalar — in
        rms_inv_ptr,  # (M,) fp32 — in: outer RMSNorm inverse
        x_norm_ptr,  # (M, K), dtype=x.dtype — out: materialized RMSNorm(x_mix)
        M,  # int
        K,  # int
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Materialize x_norm = (resid_scale*x + x0_scale*x0) * rms_inv."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = rows < M
        k_mask = ks < K
        mask = row_mask[:, None] & k_mask[None, :]
        x_base = tl.load(
            x_base_ptr + rows[:, None] * K + ks[None, :],
            mask=mask,
            other=0.0,
        )
        x0 = tl.load(
            x0_ptr + rows[:, None] * K + ks[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        resid_scale = tl.load(resid_scale_ptr).to(x_base.dtype)
        x0_scale = tl.load(x0_scale_ptr).to(x_base.dtype)
        x_mix = x_base * resid_scale + x0 * x0_scale
        x_rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0)
        tl.store(
            x_norm_ptr + rows[:, None] * K + ks[None, :],
            (x_mix * x_rms_inv[:, None]).to(x_norm_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _fused_attn_qkv_projection_qk_pre_ve_bwd_kernel(
        q_ptr,  # (M, n_head, D), dtype=x.dtype — in: final Q output
        k_ptr,  # (M, n_kv_head, D), dtype=x.dtype — in: final K output
        qk_rms_inv_ptr,  # (M, n_head + n_kv_head), fp32 — in
        cos_ptr,  # (1, rotary_seq_len, 1, D/2) — in
        sin_ptr,  # (1, rotary_seq_len, 1, D/2) — in
        d_q_ptr,  # (M, n_head, D) — in
        d_k_ptr,  # (M, n_kv_head, D) — in
        d_v_ptr,  # (M, n_kv_head, D) — in
        x_norm_ptr,  # (M, K), dtype=x.dtype — in: materialized RMSNorm(x)
        ve_ids_ptr,  # (M,), int64 — optional in: VE token ids
        ve_w_ptr,  # (vocab, n_kv_head * D) — optional in: VE table
        ve_gate_w_ptr,  # (n_kv_head, VE_GATE_CH) — optional in: gate weight
        d_q_pre_ptr,  # (M, n_head, D), dtype=x.dtype — out: grad before Q rotary
        d_k_pre_ptr,  # (M, n_kv_head, D), dtype=x.dtype — out: grad before K rotary
        dx_hat_ve_ptr,  # (M, VE_GATE_BLOCK), fp32 — optional out: VE d_x_hat slice
        d_ve_w_ptr,  # (vocab, n_kv_head * D), dtype=ve_weight.dtype — optional out
        d_ve_gate_w_ptr,  # (n_kv_head, VE_GATE_CH), dtype=ve_gate_weight.dtype — optional out
        M,  # int
        K,  # int
        D,  # int
        rotary_seq_len,  # int
        scale,  # float
        inv_scale,  # float
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        N_HEAD: tl.constexpr,
        N_KV_HEAD: tl.constexpr,
        HAS_VALUE_EMBEDDING: tl.constexpr,
        VE_GATE_CH: tl.constexpr,
        VE_GATE_BLOCK: tl.constexpr,
    ):
        """Materialize d_q0/d_k0 and optionally compute VE backward once."""
        pid_m = tl.program_id(0)
        part = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_D)
        half_cols = tl.arange(0, BLOCK_D // 2)
        row_mask = rows < M

        is_q = part < N_HEAD
        is_v = part >= N_HEAD + N_KV_HEAD

        if HAS_VALUE_EMBEDDING and is_v:
            head = part - N_HEAD - N_KV_HEAD
            gate_cols = tl.arange(0, VE_GATE_BLOCK)
            gate_mask = gate_cols < VE_GATE_CH
            token_ids = tl.load(ve_ids_ptr + rows, mask=row_mask, other=0)
            x_hat_gate = tl.load(
                x_norm_ptr + rows[:, None] * K + gate_cols[None, :],
                mask=row_mask[:, None] & gate_mask[None, :],
                other=0.0,
            ).to(x_norm_ptr.dtype.element_ty)

            d_v = tl.load(
                d_v_ptr
                + rows[:, None] * N_KV_HEAD * D
                + head * D
                + cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ).to(x_norm_ptr.dtype.element_ty)
            gate_w = tl.load(
                ve_gate_w_ptr + head * VE_GATE_CH + gate_cols,
                mask=gate_mask,
                other=0.0,
            ).to(x_norm_ptr.dtype.element_ty)
            gate_logits = tl.sum(
                x_hat_gate * gate_w[None, :],
                axis=1,
                dtype=tl.float32,
            )
            sigmoid = tl.sigmoid(gate_logits)
            gate = (3 * sigmoid).to(x_norm_ptr.dtype.element_ty)
            ve = tl.load(
                ve_w_ptr
                + token_ids[:, None] * N_KV_HEAD * D
                + head * D
                + cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            d_gate = tl.sum(d_v * ve, axis=1, dtype=tl.float32)
            d_gate_logits = (3 * d_gate * sigmoid * (1.0 - sigmoid)).to(
                x_norm_ptr.dtype.element_ty
            )

            tl.atomic_add(
                d_ve_w_ptr
                + token_ids[:, None] * N_KV_HEAD * D
                + head * D
                + cols[None, :],
                d_v * gate[:, None],
                sem="relaxed",
                mask=row_mask[:, None],
            )
            tl.atomic_add(
                d_ve_gate_w_ptr + head * VE_GATE_CH + gate_cols,
                tl.sum(d_gate_logits[:, None] * x_hat_gate, axis=0, dtype=tl.float32),
                sem="relaxed",
                mask=gate_mask,
            )
            tl.atomic_add(
                dx_hat_ve_ptr + rows[:, None] * VE_GATE_BLOCK + gate_cols[None, :],
                d_gate_logits[:, None] * gate_w[None, :],
                sem="relaxed",
                mask=row_mask[:, None] & gate_mask[None, :],
            )
        else:
            if is_q:
                head = part
                out_base = q_ptr + rows[:, None] * N_HEAD * D + head * D
                grad_base = d_q_ptr + rows[:, None] * N_HEAD * D + head * D
            else:
                head = part - N_HEAD
                out_base = k_ptr + rows[:, None] * N_KV_HEAD * D + head * D
                grad_base = d_k_ptr + rows[:, None] * N_KV_HEAD * D + head * D

            y0 = tl.load(
                out_base + cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            y0 = (y0 * inv_scale).to(d_q_pre_ptr.dtype.element_ty)
            g = tl.load(
                grad_base + cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            g = (g * scale).to(d_q_pre_ptr.dtype.element_ty)
            qk_rms_inv = tl.load(
                qk_rms_inv_ptr + rows * (N_HEAD + N_KV_HEAD) + part,
                mask=row_mask,
                other=0.0,
            )
            qk_inner = tl.sum(g * y0, axis=1, dtype=tl.float32) / D
            d_rot = (
                qk_rms_inv[:, None] * (g - y0 * qk_inner[:, None])
            ).to(d_q_pre_ptr.dtype.element_ty)
            d_rot_halves = tl.reshape(d_rot, (BLOCK_M, 2, BLOCK_D // 2))
            d_rot_lo, d_rot_hi = tl.split(tl.trans(d_rot_halves, 0, 2, 1))

            rotary_rows = rows % rotary_seq_len
            cos = tl.load(
                cos_ptr + rotary_rows[:, None] * (BLOCK_D // 2) + half_cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            sin = tl.load(
                sin_ptr + rotary_rows[:, None] * (BLOCK_D // 2) + half_cols[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            d_pre_lo = d_rot_lo * cos - d_rot_hi * sin
            d_pre_hi = d_rot_lo * sin + d_rot_hi * cos
            d_pre_halves = tl.trans(tl.join(d_pre_lo, d_pre_hi), 0, 2, 1)
            d_pre = tl.reshape(d_pre_halves, (BLOCK_M, BLOCK_D))

            if is_q:
                pre_ptrs = (
                    d_q_pre_ptr
                    + rows[:, None] * N_HEAD * D
                    + head * D
                    + cols[None, :]
                )
            else:
                pre_ptrs = (
                    d_k_pre_ptr
                    + rows[:, None] * N_KV_HEAD * D
                    + head * D
                    + cols[None, :]
                )
            tl.store(
                pre_ptrs,
                d_pre.to(d_q_pre_ptr.dtype.element_ty),
                mask=row_mask[:, None],
            )

    @triton.jit
    def _fused_attn_qkv_projection_dx_hat_bwd_kernel(
        d_q_pre_ptr,  # (M, n_head, D), dtype=x.dtype — in
        d_k_pre_ptr,  # (M, n_kv_head, D), dtype=x.dtype — in
        d_v_ptr,  # (M, n_kv_head, D) — in
        dx_hat_ve_ptr,  # (M, VE_GATE_BLOCK), fp32 — optional in: VE d_x_hat slice
        q_w_ptr,  # (n_head * D, K) — in
        k_w_ptr,  # (n_kv_head * D, K) — in
        v_w_ptr,  # (n_kv_head * D, K) — in
        dx_hat_ptr,  # (M, K), dtype=x.dtype — out: materialized d_x_hat
        x_norm_ptr,  # (M, K), dtype=x.dtype — in: materialized RMSNorm(x)
        outer_rms_row_inner_ptr,  # (M,) fp32 — in/out: outer RMSNorm row-inner
        M,  # int
        K,  # int
        D,  # int
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        N_HEAD: tl.constexpr,
        N_KV_HEAD: tl.constexpr,
        HEAD_SPLIT: tl.constexpr,
        HAS_VALUE_EMBEDDING: tl.constexpr,
        VE_GATE_CH: tl.constexpr,
        VE_GATE_BLOCK: tl.constexpr,
    ):
        """Compute d_x_hat from materialized Q/K grads and optional VE slice."""
        if HAS_VALUE_EMBEDDING:
            tl.static_assert(
                VE_GATE_CH <= BLOCK_K,
                "VE gate channels must fit in the first K tile",
            )

        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        cols = tl.arange(0, BLOCK_D)
        row_mask = rows < M
        k_mask = ks < K

        dx_hat_tile = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for head in range(0, N_HEAD):
            for split in range(0, HEAD_SPLIT):
                head_cols = split * BLOCK_D + cols
                d_part = tl.load(
                    d_q_pre_ptr
                    + rows[:, None] * N_HEAD * D
                    + head * D
                    + head_cols[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(x_norm_ptr.dtype.element_ty)
                w = tl.load(
                    q_w_ptr + (head * D + head_cols)[:, None] * K + ks[None, :],
                    mask=k_mask[None, :],
                    other=0.0,
                ).to(x_norm_ptr.dtype.element_ty)
                dx_hat_tile += tl.dot(d_part, w)

        for head in range(0, N_KV_HEAD):
            for split in range(0, HEAD_SPLIT):
                head_cols = split * BLOCK_D + cols
                d_part = tl.load(
                    d_k_pre_ptr
                    + rows[:, None] * N_KV_HEAD * D
                    + head * D
                    + head_cols[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(x_norm_ptr.dtype.element_ty)
                w = tl.load(
                    k_w_ptr + (head * D + head_cols)[:, None] * K + ks[None, :],
                    mask=k_mask[None, :],
                    other=0.0,
                ).to(x_norm_ptr.dtype.element_ty)
                dx_hat_tile += tl.dot(d_part, w)

        for head in range(0, N_KV_HEAD):
            for split in range(0, HEAD_SPLIT):
                head_cols = split * BLOCK_D + cols
                d_part = tl.load(
                    d_v_ptr
                    + rows[:, None] * N_KV_HEAD * D
                    + head * D
                    + head_cols[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(x_norm_ptr.dtype.element_ty)
                w = tl.load(
                    v_w_ptr + (head * D + head_cols)[:, None] * K + ks[None, :],
                    mask=k_mask[None, :],
                    other=0.0,
                ).to(x_norm_ptr.dtype.element_ty)
                dx_hat_tile += tl.dot(d_part, w)

        if HAS_VALUE_EMBEDDING:
            if pid_k == 0:
                tile_gate_mask = ks < VE_GATE_CH
                dx_hat_ve = tl.load(
                    dx_hat_ve_ptr + rows[:, None] * VE_GATE_BLOCK + ks[None, :],
                    mask=row_mask[:, None] & tile_gate_mask[None, :],
                    other=0.0,
                )
                dx_hat_tile += dx_hat_ve

        dx_hat_store = dx_hat_tile.to(dx_hat_ptr.dtype.element_ty)
        tl.store(
            dx_hat_ptr + rows[:, None] * K + ks[None, :],
            dx_hat_store,
            mask=row_mask[:, None] & k_mask[None, :],
        )
        x_norm = tl.load(
            x_norm_ptr + rows[:, None] * K + ks[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        outer_rms_row_inner_partial = (
            tl.sum(dx_hat_store * x_norm, axis=1, dtype=tl.float32) / K
        )
        tl.atomic_add(
            outer_rms_row_inner_ptr + rows,
            outer_rms_row_inner_partial,
            sem="relaxed",
            mask=row_mask,
        )

    @triton.jit
    def _fused_attn_qkv_projection_weight_section_bwd_kernel(
        d_q_pre_ptr,  # (M, n_head, D), dtype=x.dtype — in
        d_k_pre_ptr,  # (M, n_kv_head, D), dtype=x.dtype — in
        d_v_ptr,  # (M, n_kv_head, D) — in
        x_norm_ptr,  # (M, K), dtype=x.dtype — in: materialized RMSNorm(x)
        d_q_w_ptr,  # (n_head * D, K) — out
        d_k_w_ptr,  # (n_kv_head * D, K) — out
        d_v_w_ptr,  # (n_kv_head * D, K) — out
        M,  # int
        K,  # int
        D,  # int
        n_head,  # int
        n_kv_head,  # int
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Compute Q/K/V weight grads by section, allowing N tiles across heads."""
        section = tl.program_id(0)  # 0=q, 1=k, 2=v
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = ks < K
        n_total = tl.where(section == 0, n_head * D, n_kv_head * D)
        n_mask = ns < n_total

        d_weight_tile = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for m_start in range(0, M, BLOCK_M):
            rows = m_start + tl.arange(0, BLOCK_M)
            row_mask = rows < M
            x_hat = tl.load(
                x_norm_ptr + rows[:, None] * K + ks[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            if section == 0:
                d_part = tl.load(
                    d_q_pre_ptr + rows[:, None] * n_head * D + ns[None, :],
                    mask=row_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
            elif section == 1:
                d_part = tl.load(
                    d_k_pre_ptr + rows[:, None] * n_kv_head * D + ns[None, :],
                    mask=row_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
            else:
                d_part = tl.load(
                    d_v_ptr + rows[:, None] * n_kv_head * D + ns[None, :],
                    mask=row_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )

            d_weight_tile += tl.dot(tl.trans(d_part), x_hat)

        if section == 0:
            tl.store(
                d_q_w_ptr + ns[:, None] * K + ks[None, :],
                d_weight_tile,
                mask=n_mask[:, None] & k_mask[None, :],
            )
        elif section == 1:
            tl.store(
                d_k_w_ptr + ns[:, None] * K + ks[None, :],
                d_weight_tile,
                mask=n_mask[:, None] & k_mask[None, :],
            )
        else:
            tl.store(
                d_v_w_ptr + ns[:, None] * K + ks[None, :],
                d_weight_tile,
                mask=n_mask[:, None] & k_mask[None, :],
            )

    @triton.jit
    def _fused_attn_qkv_projection_outer_rms_dx_bwd_kernel(
        x_norm_ptr,  # (M, K), dtype=x.dtype — in: materialized RMSNorm(x_mix)
        rms_inv_ptr,  # (M,) fp32 — in
        dx_hat_ptr,  # (M, K), dtype=x.dtype — in: materialized d_x_hat
        outer_rms_row_inner_ptr,  # (M,) fp32 — in
        grad_x_mix_ptr,  # (M, K) — in: direct grad wrt x_mix output
        x_base_ptr,  # (M, K) — in: residual stream before mix
        x0_ptr,  # (M, K) — in: initial embedding stream
        resid_scale_ptr,  # scalar tensor — in
        x0_scale_ptr,  # scalar tensor — in
        dx_ptr,  # (M, K) — out: grad wrt residual stream before mix
        dx0_ptr,  # (M, K) — out: grad wrt initial embedding stream
        d_resid_scale_ptr,  # scalar tensor — out, zero-initialized
        d_x0_scale_ptr,  # scalar tensor — out, zero-initialized
        M,  # int
        K,  # int
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute d_x_mix and distribute through residual mixing."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = rows < M
        k_mask = ks < K
        mask = row_mask[:, None] & k_mask[None, :]
        offs = rows[:, None] * K + ks[None, :]

        x_rms_inv = tl.load(rms_inv_ptr + rows, mask=row_mask, other=0.0)
        outer_rms_row_inner = tl.load(
            outer_rms_row_inner_ptr + rows,
            mask=row_mask,
            other=0.0,
        )
        x_norm = tl.load(
            x_norm_ptr + offs,
            mask=mask,
            other=0.0,
        )
        dx_hat = tl.load(
            dx_hat_ptr + offs,
            mask=mask,
            other=0.0,
        )
        dx_mix = x_rms_inv[:, None] * (
            dx_hat - x_norm * outer_rms_row_inner[:, None]
        )
        grad_x_mix = tl.load(grad_x_mix_ptr + offs, mask=mask, other=0.0)
        dx_mix += grad_x_mix

        resid_scale = tl.load(resid_scale_ptr).to(dx_mix.dtype)
        x0_scale = tl.load(x0_scale_ptr).to(dx_mix.dtype)
        tl.store(
            dx_ptr + offs,
            (dx_mix * resid_scale).to(dx_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            dx0_ptr + offs,
            (dx_mix * x0_scale).to(dx0_ptr.dtype.element_ty),
            mask=mask,
        )

        x_base = tl.load(x_base_ptr + offs, mask=mask, other=0.0)
        x0 = tl.load(x0_ptr + offs, mask=mask, other=0.0)
        dx_mix_f32 = dx_mix.to(tl.float32)
        d_resid_rows = tl.sum(dx_mix_f32 * x_base.to(tl.float32), axis=1)
        d_x0_rows = tl.sum(dx_mix_f32 * x0.to(tl.float32), axis=1)
        tl.atomic_add(
            d_resid_scale_ptr,
            tl.sum(d_resid_rows, axis=0),
            sem="relaxed",
        )
        tl.atomic_add(
            d_x0_scale_ptr,
            tl.sum(d_x0_rows, axis=0),
            sem="relaxed",
        )

def _validate_rotary_table_4d(
    table: torch.Tensor,
    T: int,
    head_dim: int,
) -> None:
    """Validate a 4D broadcast rotary table."""
    half = head_dim // 2
    assert table.is_cuda and table.is_contiguous()
    assert table.ndim == 4, (
        f"rotary table must be 4D (1, T, 1, D/2), got {table.ndim}D"
    )
    assert table.shape[0] == 1 and table.shape[2] == 1
    assert T > 0
    assert table.shape[1] == T, (
        f"rotary table sequence length must be {T}, got {table.shape[1]}"
    )
    assert table.shape[-1] == half, (
        f"rotary table last dim must be {half}, got {table.shape[-1]}"
    )


def _fused_attn_qkv_projection_bwd_impl(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    d_v: torch.Tensor,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_gate_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_inv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    qk_rms_inv: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
    grad_x_mix: torch.Tensor,
    x_base: torch.Tensor,
    x0: torch.Tensor,
    resid_scale: torch.Tensor,
    x0_scale: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Backward implementation for `nanoops::fused_attn_qkv_projection_bwd`.

    This function launches Triton kernels for the fp32 algebra used by the
    Triton-op autograd callback. Rotary cos/sin are treated as constants.
    Q/K RMSNorm backward uses saved `q`/`k`/`qk_rms_inv` from the forward
    Triton op, recovering `y0 = q_or_k / scale` without recomputing Q/K
    projection outputs. Optional VE gradients are None when VE is disabled.
    Tensor inputs are expected to be CUDA contiguous tensors. The public API
    returns `(B, T, head, head_dim)`, but the public wrapper is just a view over
    internal `(M, head, head_dim)` tensors where `M = B*T`. Autograd therefore
    enters this implementation with M-shaped Q/K/V gradients and saved Q/K
    state. All Triton kernels below use `(M, head, head_dim)` for Q/K/V-like
    tensors and `(M, K)` for activations.

    Args:
      d_q: (M, n_head, head_dim), CUDA, grad of q output.
      d_k: (M, n_kv_head, head_dim), CUDA, grad of k output.
      d_v: (M, n_kv_head, head_dim), CUDA, grad of v output.
      grad_x_mix: public (B, T, K), direct grad from the returned `x_mix`.
      x_base: public (B, T, K), residual stream before the forward blend.
      x0: public (B, T, K), initial embedding stream used by the blend.
      resid_scale/x0_scale: scalar tensors used by
        `x_mix = resid_scale*x_base + x0_scale*x0`.
      ve_ids: optional (B, T), CUDA integer token ids.
      ve_weight: optional (vocab, n_kv_head * head_dim), value embedding table.
      ve_gate_channels: int, leading normalized input channels used by the VE gate.
      ve_gate_weight: optional (n_kv_head, ve_gate_channels), gate weight.
      q_weight: (n_head * head_dim, K), dtype=q_weight.dtype, Q projection weight.
      k_weight: (n_kv_head * head_dim, K), dtype=k_weight.dtype, K projection weight.
      v_weight: (n_kv_head * head_dim, K), dtype=v_weight.dtype, V projection weight.
      cos: (1, T, 1, head_dim/2), rotary cosine table.
      sin: (1, T, 1, head_dim/2), rotary sine table.
      rms_inv: (M,), fp32 outer RMSNorm inverse from forward.
      q: (M, n_head, head_dim), dtype=x.dtype, final Q output.
      k: (M, n_kv_head, head_dim), dtype=x.dtype, final K output.
      qk_rms_inv: (M, n_head + n_kv_head), fp32 Q/K RMS inverse.
      n_head: int, number of query heads.
      n_kv_head: int, number of key/value heads.
      head_dim: int, per-head width; must be even.
      scale: float, post-Q/K RMSNorm scalar multiplier.
      eps: float, kept to mirror the forward Triton-op signature; saved
        inverses already include epsilon.

    Returns:
      dx: public (B, T, K), grad wrt `x_base`.
      dx0: public (B, T, K), grad wrt `x0`.
      d_resid_scale/d_x0_scale: scalar gradients for the residual blend.
      d_ve_weight: optional (vocab, n_kv_head * head_dim), dtype=ve_weight.dtype.
      d_ve_gate_weight: optional (n_kv_head, ve_gate_channels),
        dtype=ve_gate_weight.dtype.
      d_q_weight: (n_head * head_dim, K), dtype=q_weight.dtype.
      d_k_weight: (n_kv_head * head_dim, K), dtype=k_weight.dtype.
      d_v_weight: (n_kv_head * head_dim, K), dtype=v_weight.dtype.

    Forward math:
      Residual mix + outer RMSNorm:
        x_mix = resid_scale*x_base + x0_scale*x0
        s_x   = rsqrt(mean_k(x_mix^2) + eps)
        x_n   = x_mix * s_x
        x_hat = x_n

      Projections:
        q0 = x_hat @ q_weight.T
        k0 = x_hat @ k_weight.T
        v  = x_hat @ v_weight.T
        z  = concat(q0, k0, v) conceptually; it is not materialized here.

      Optional value embedding:
        x_g       = x_hat[:, :ve_gate_channels]            # (M, ch)
        gate      = 3 * sigmoid(x_g @ ve_gate_weight.T)    # (M, n_kv_head)
        ve        = ve_weight[ve_ids].view(M, n_kv_head, head_dim)
        v        += gate[..., None] * ve

      Q/K rotary + QK RMSNorm + scale:
        lo, hi = split(q0 or k0)
        r_lo   = lo * cos + hi * sin
        r_hi   = -lo * sin + hi * cos
        r      = concat(r_lo, r_hi)
        s_qk   = rsqrt(mean_d(r^2) + eps)
        y0     = r * s_qk
        q or k = scale * y0

      Saved for Triton-op backward:
        x, optional VE tensors, q/k/v weights, and cos/sin are saved from
        the forward inputs. The forward outputs used only by backward are:
          rms_inv     = s_x                         # (M,), fp32
          q, k        = final scaled Q/K outputs    # x.dtype
          qk_rms_inv  = s_qk per Q/K head           # (M, n_head+n_kv_head)

        Not saved: x_mix, x_hat, q0, k0, v, q/k rotary pre-norm
        intermediates, or dz. Backward rematerializes
        x_norm = (resid_scale*x_base + x0_scale*x0) * rms_inv, then
        materializes Q/K projection-output grads from saved final q/k plus
        qk_rms_inv, and computes d_z conceptually without storing the full
        concatenated buffer.

    Backward math:
      Q/K RMSNorm backward receives g_y = d_q or d_k:
        y0 = q_or_k / scale
        g0 = g_y * scale
        dr = s_qk * (g0 - y0 * mean_d(g0 * y0))

      y0 comes from saved final q/k and s_qk comes from saved qk_rms_inv.

      Inverse rotary:
        d_lo = d_r_lo * cos - d_r_hi * sin
        d_hi = d_r_lo * sin + d_r_hi * cos
        d_q0 or d_k0 = concat(d_lo, d_hi)

      Projection backward:
        d_z     = concat(d_q0, d_k0, d_v) conceptually
        d_x_hat = d_q0 @ q_weight + d_k0 @ k_weight + d_v @ v_weight
        dW_q    = d_q0.T @ x_hat
        dW_k    = d_k0.T @ x_hat
        dW_v    = d_v.T @ x_hat

      In code, `d_z` is not stored. A Q/K/V-part grad kernel materializes
      d_q0/d_k0 in x.dtype for the Q/K parts; when VE is enabled, its V parts
      also compute VE table/gate gradients and a small
      d_x_hat[:, :ve_gate_channels] contribution buffer. The dx_hat kernel
      then runs separate full-head Q/K/V projection loops, adds that VE
      contribution on the first K tile, materializes d_x_hat in x.dtype, and
      accumulates the outer RMSNorm inner contribution from that materialized
      value.

      Optional value-embedding backward, computed in the V parts of the grad
      kernel before outer RMSNorm backward because the gate also consumes x_hat:
        x_g                 = x_hat[:, :ve_gate_channels]
        ve                  = ve_weight[ve_ids].view(M, n_kv_head, head_dim)
        d_gate              = sum_d(d_v * ve)
        d_gate_logits       = 3 * d_gate * sigmoid * (1 - sigmoid)
        d_x_g               = d_gate_logits @ ve_gate_weight
        d_x_hat[:, :ve_gate_channels] += d_x_g
        d_ve_weight[token]  += d_v * gate
        d_ve_gate_weight    += d_gate_logits.T @ x_g

      Outer RMSNorm backward:
        d_x_mix = s_x * (d_x_hat - x_n * mean_k(d_x_hat * x_n))

      Residual mix backward, fused into the final outer-RMS dx kernel when
      enabled:
        d_x_mix += grad_x_mix
        d_x            = d_x_mix * resid_scale
        d_x0           = d_x_mix * x0_scale
        d_resid_scale  = sum(d_x_mix * x)
        d_x0_scale     = sum(d_x_mix * x0)
    """
    if not _HAS_TRITON:
        raise RuntimeError("fused_attn_qkv_projection backward requires triton")
    assert q_weight.is_cuda and k_weight.is_cuda and v_weight.is_cuda
    assert x_base.is_cuda and x0.is_cuda
    assert x_base.is_contiguous() and x0.is_contiguous()
    assert x_base.ndim == 3 and x0.shape == x_base.shape
    B, T, K = x_base.shape
    M = B * T
    assert resid_scale.is_cuda and x0_scale.is_cuda
    assert resid_scale.numel() == 1 and x0_scale.numel() == 1
    assert grad_x_mix.is_cuda and grad_x_mix.numel() == M * K
    x_base_flat = x_base.view(M, K)
    x0_flat = x0.view(M, K)
    grad_x_mix_flat = grad_x_mix.contiguous().view(M, K)
    assert (
        q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and v_weight.is_contiguous()
    )
    assert head_dim % 2 == 0

    if d_q.ndim == 4:
        d_q = d_q.contiguous().reshape(M, n_head, head_dim)
    else:
        assert d_q.shape == (M, n_head, head_dim)
        d_q = d_q.contiguous()
    if d_k.ndim == 4:
        d_k = d_k.contiguous().reshape(M, n_kv_head, head_dim)
    else:
        assert d_k.shape == (M, n_kv_head, head_dim)
        d_k = d_k.contiguous()
    if d_v.ndim == 4:
        d_v = d_v.contiguous().reshape(M, n_kv_head, head_dim)
    else:
        assert d_v.shape == (M, n_kv_head, head_dim)
        d_v = d_v.contiguous()
    if q.ndim == 4:
        saved_q = q.view(M, n_head, head_dim)
    else:
        assert q.shape == (M, n_head, head_dim)
        saved_q = q
    if k.ndim == 4:
        saved_k = k.view(M, n_kv_head, head_dim)
    else:
        assert k.shape == (M, n_kv_head, head_dim)
        saved_k = k
    if qk_rms_inv.ndim == 3:
        saved_qk_rms_inv = qk_rms_inv.view(M, n_head + n_kv_head)
    else:
        assert qk_rms_inv.shape == (M, n_head + n_kv_head)
        saved_qk_rms_inv = qk_rms_inv
    q_n = n_head * head_dim
    kv_n = n_kv_head * head_dim
    assert d_q.shape == (M, n_head, head_dim)
    assert d_k.shape == (M, n_kv_head, head_dim)
    assert d_v.shape == (M, n_kv_head, head_dim)
    assert q_weight.shape == (q_n, K)
    assert k_weight.shape == (kv_n, K)
    assert v_weight.shape == (kv_n, K)
    assert saved_q.is_cuda and saved_k.is_cuda and saved_qk_rms_inv.is_cuda
    assert saved_q.is_contiguous() and saved_k.is_contiguous()
    assert saved_qk_rms_inv.is_contiguous()
    assert saved_q.shape == (M, n_head, head_dim)
    assert saved_k.shape == (M, n_kv_head, head_dim)
    assert saved_qk_rms_inv.shape == (M, n_head + n_kv_head)
    assert saved_qk_rms_inv.dtype == torch.float32
    has_value_embedding = (
        ve_ids is not None or ve_weight is not None or ve_gate_weight is not None
    )
    if has_value_embedding:
        assert ve_ids is not None and ve_weight is not None and ve_gate_weight is not None
        assert ve_ids.is_cuda and ve_weight.is_cuda and ve_gate_weight.is_cuda
        assert ve_weight.is_contiguous() and ve_gate_weight.is_contiguous()
        assert ve_ids.numel() == M
        assert ve_weight.ndim == 2 and ve_weight.shape[1] == kv_n
        assert 0 < ve_gate_channels <= K
        assert ve_gate_weight.shape == (n_kv_head, ve_gate_channels)
    rotary_seq_len = cos.shape[1]
    _validate_rotary_table_4d(cos, rotary_seq_len, head_dim)
    _validate_rotary_table_4d(sin, rotary_seq_len, head_dim)
    assert M % rotary_seq_len == 0, (
        f"M={M} is not divisible by rotary T={rotary_seq_len}"
    )

    assert rms_inv.is_cuda and rms_inv.is_contiguous()
    assert rms_inv.shape == (M,) and rms_inv.dtype == torch.float32
    x_norm = torch.empty_like(x_base_flat)
    d_q_pre = torch.empty_like(saved_q)
    d_k_pre = torch.empty_like(saved_k)
    dx_hat = torch.empty_like(x_base_flat)
    outer_rms_row_inner = torch.zeros((M,), dtype=torch.float32, device=x_base.device)
    dx = torch.empty_like(x_base_flat)
    dx0 = torch.empty_like(x0_flat)
    d_resid_scale = torch.zeros_like(resid_scale)
    d_x0_scale = torch.zeros_like(x0_scale)
    d_ve_weight = None
    d_ve_gate_weight = None
    ve_gate_block = triton.next_power_of_2(ve_gate_channels)

    if has_value_embedding:
        assert ve_ids is not None and ve_weight is not None and ve_gate_weight is not None
        ve_ids_for_kernel = ve_ids.reshape(M).contiguous()
        ve_weight_for_kernel = ve_weight
        ve_gate_weight_for_kernel = ve_gate_weight
        dx_hat_ve_for_kernel = torch.zeros(
            (M, ve_gate_block),
            dtype=torch.float32,
            device=x_base.device,
        )
        d_ve_weight = torch.zeros_like(ve_weight)
        d_ve_gate_weight = torch.zeros_like(ve_gate_weight)
        d_ve_weight_for_kernel = d_ve_weight
        d_ve_gate_weight_for_kernel = d_ve_gate_weight
    else:
        ve_ids_for_kernel = x_base_flat
        ve_weight_for_kernel = x_base_flat
        ve_gate_weight_for_kernel = x_base_flat
        dx_hat_ve_for_kernel = x_base_flat
        d_ve_weight_for_kernel = x_base_flat
        d_ve_gate_weight_for_kernel = x_base_flat

    if has_value_embedding:
        QK_PRE_BLOCK_M = _QKV_BWD_QK_PRE_VE_BLOCK_M
        QK_PRE_NUM_WARPS = _QKV_BWD_QK_PRE_VE_NUM_WARPS
        QK_PRE_NUM_STAGES = _QKV_BWD_QK_PRE_VE_NUM_STAGES
        DX_HAT_BLOCK_M = _QKV_BWD_DX_HAT_VE_BLOCK_M
        DX_HAT_BLOCK_K = _QKV_BWD_DX_HAT_VE_BLOCK_K
        DX_HAT_HEAD_SPLIT = _QKV_BWD_DX_HAT_VE_HEAD_SPLIT
        DX_HAT_NUM_WARPS = _QKV_BWD_DX_HAT_VE_NUM_WARPS
        DX_HAT_NUM_STAGES = _QKV_BWD_DX_HAT_VE_NUM_STAGES
    else:
        QK_PRE_BLOCK_M = _QKV_BWD_QK_PRE_NO_VE_BLOCK_M
        QK_PRE_NUM_WARPS = _QKV_BWD_QK_PRE_NO_VE_NUM_WARPS
        QK_PRE_NUM_STAGES = _QKV_BWD_QK_PRE_NO_VE_NUM_STAGES
        DX_HAT_BLOCK_M = _QKV_BWD_DX_HAT_NO_VE_BLOCK_M
        DX_HAT_BLOCK_K = _QKV_BWD_DX_HAT_NO_VE_BLOCK_K
        DX_HAT_HEAD_SPLIT = _QKV_BWD_DX_HAT_NO_VE_HEAD_SPLIT
        DX_HAT_NUM_WARPS = _QKV_BWD_DX_HAT_NO_VE_NUM_WARPS
        DX_HAT_NUM_STAGES = _QKV_BWD_DX_HAT_NO_VE_NUM_STAGES
    X_NORM_BLOCK_M = _QKV_BWD_X_NORM_BLOCK_M
    X_NORM_BLOCK_K = _QKV_BWD_X_NORM_BLOCK_K
    X_NORM_NUM_WARPS = _QKV_BWD_X_NORM_NUM_WARPS
    OUTER_RMS_BLOCK_M = _QKV_BWD_OUTER_RMS_BLOCK_M
    OUTER_RMS_BLOCK_K = _QKV_BWD_OUTER_RMS_BLOCK_K
    OUTER_RMS_NUM_WARPS = _QKV_BWD_OUTER_RMS_NUM_WARPS
    WEIGHT_GRAD_BLOCK_N = _QKV_BWD_WEIGHT_GRAD_BLOCK_N
    WEIGHT_GRAD_BLOCK_K = _QKV_BWD_WEIGHT_GRAD_BLOCK_K
    WEIGHT_GRAD_BLOCK_M = _QKV_BWD_WEIGHT_GRAD_BLOCK_M
    WEIGHT_GRAD_NUM_WARPS = _QKV_BWD_WEIGHT_GRAD_NUM_WARPS
    WEIGHT_GRAD_NUM_STAGES = _QKV_BWD_WEIGHT_GRAD_NUM_STAGES
    if has_value_embedding:
        assert ve_gate_channels <= DX_HAT_BLOCK_K, (
            f"ve_gate_channels={ve_gate_channels} must fit in "
            f"DX_HAT_BLOCK_K={DX_HAT_BLOCK_K}"
        )
    assert head_dim % DX_HAT_HEAD_SPLIT == 0

    # Phase 1: rematerialize x_norm for backward reuse. We intentionally do
    # not save or materialize raw x_mix in backward; only RMSNorm(x_mix) is
    # reused by later kernels.
    wrap_triton(_fused_attn_qkv_projection_x_norm_bwd_kernel)[
        (triton.cdiv(M, X_NORM_BLOCK_M), triton.cdiv(K, X_NORM_BLOCK_K))
    ](
        x_base_flat,
        x0_flat,
        resid_scale,
        x0_scale,
        rms_inv,
        x_norm,
        M,
        K,
        BLOCK_M=X_NORM_BLOCK_M,
        BLOCK_K=X_NORM_BLOCK_K,
        num_warps=X_NORM_NUM_WARPS,
    )
    # Phase 2: recover Q/K projection-output grads and optional VE gradients.
    wrap_triton(_fused_attn_qkv_projection_qk_pre_ve_bwd_kernel)[
        (
            triton.cdiv(M, QK_PRE_BLOCK_M),
            n_head + n_kv_head + (n_kv_head if has_value_embedding else 0),
        )
    ](
        saved_q,
        saved_k,
        saved_qk_rms_inv,
        cos,
        sin,
        d_q,
        d_k,
        d_v,
        x_norm,
        ve_ids_for_kernel,
        ve_weight_for_kernel,
        ve_gate_weight_for_kernel,
        d_q_pre,
        d_k_pre,
        dx_hat_ve_for_kernel,
        d_ve_weight_for_kernel,
        d_ve_gate_weight_for_kernel,
        M,
        K,
        head_dim,
        rotary_seq_len,
        scale,
        1.0 / scale,
        BLOCK_M=QK_PRE_BLOCK_M,
        BLOCK_D=head_dim,
        N_HEAD=n_head,
        N_KV_HEAD=n_kv_head,
        HAS_VALUE_EMBEDDING=has_value_embedding,
        VE_GATE_CH=ve_gate_channels,
        VE_GATE_BLOCK=ve_gate_block,
        num_warps=QK_PRE_NUM_WARPS,
        num_stages=QK_PRE_NUM_STAGES,
    )
    # Phase 3: project Q/K/V grads back to x_hat and accumulate RMS row inner.
    # d24 uses half-head tiles and loads projection weights in their native
    # dtype; the Triton kernel casts tiles to activation dtype where needed.
    wrap_triton(_fused_attn_qkv_projection_dx_hat_bwd_kernel)[
        (triton.cdiv(M, DX_HAT_BLOCK_M), triton.cdiv(K, DX_HAT_BLOCK_K))
    ](
        d_q_pre,
        d_k_pre,
        d_v,
        dx_hat_ve_for_kernel,
        q_weight,
        k_weight,
        v_weight,
        dx_hat,
        x_norm,
        outer_rms_row_inner,
        M,
        K,
        head_dim,
        BLOCK_M=DX_HAT_BLOCK_M,
        BLOCK_K=DX_HAT_BLOCK_K,
        BLOCK_D=head_dim // DX_HAT_HEAD_SPLIT,
        N_HEAD=n_head,
        N_KV_HEAD=n_kv_head,
        HEAD_SPLIT=DX_HAT_HEAD_SPLIT,
        HAS_VALUE_EMBEDDING=has_value_embedding,
        VE_GATE_CH=ve_gate_channels,
        VE_GATE_BLOCK=ve_gate_block,
        num_warps=DX_HAT_NUM_WARPS,
        num_stages=DX_HAT_NUM_STAGES,
    )
    # Phase 4: finish outer RMSNorm input gradient.
    wrap_triton(_fused_attn_qkv_projection_outer_rms_dx_bwd_kernel)[
        (triton.cdiv(M, OUTER_RMS_BLOCK_M), triton.cdiv(K, OUTER_RMS_BLOCK_K))
    ](
        x_norm,
        rms_inv,
        dx_hat,
        outer_rms_row_inner,
        grad_x_mix_flat,
        x_base_flat,
        x0_flat,
        resid_scale,
        x0_scale,
        dx,
        dx0,
        d_resid_scale,
        d_x0_scale,
        M,
        K,
        BLOCK_M=OUTER_RMS_BLOCK_M,
        BLOCK_K=OUTER_RMS_BLOCK_K,
        num_warps=OUTER_RMS_NUM_WARPS,
    )
    del dx_hat, outer_rms_row_inner

    # Phase 5: compute projection weight gradients.
    d_q_weight = torch.empty_like(q_weight)
    d_k_weight = torch.empty_like(k_weight)
    d_v_weight = torch.empty_like(v_weight)
    wrap_triton(_fused_attn_qkv_projection_weight_section_bwd_kernel)[
        (
            3,
            triton.cdiv(max(q_n, kv_n), WEIGHT_GRAD_BLOCK_N),
            triton.cdiv(K, WEIGHT_GRAD_BLOCK_K),
        )
    ](
        d_q_pre,
        d_k_pre,
        d_v,
        x_norm,
        d_q_weight,
        d_k_weight,
        d_v_weight,
        M,
        K,
        head_dim,
        n_head,
        n_kv_head,
        BLOCK_N=WEIGHT_GRAD_BLOCK_N,
        BLOCK_K=WEIGHT_GRAD_BLOCK_K,
        BLOCK_M=WEIGHT_GRAD_BLOCK_M,
        num_warps=WEIGHT_GRAD_NUM_WARPS,
        num_stages=WEIGHT_GRAD_NUM_STAGES,
    )
    return (
        dx.reshape_as(x_base),
        dx0.reshape_as(x0),
        d_resid_scale,
        d_x0_scale,
        d_ve_weight,
        d_ve_gate_weight,
        d_q_weight,
        d_k_weight,
        d_v_weight,
    )


def _fused_attn_qkv_projection_fwd_impl(
    x: torch.Tensor,
    x0: torch.Tensor,
    resid_scale: torch.Tensor,
    x0_scale: torch.Tensor,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_gate_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Forward with fused `x_mix = resid_scale*x + x0_scale*x0`.

    Public inputs `x`/`x0` are `(B, T, K)`. The fused residual mix, RMSNorm,
    and QKV projection all run on the internal flattened `M=B*T` rows. Returns
    internal `(M, *)` outputs: q/k/v, x_mix, rms_inv, qk_rms_inv.
    """
    if not _HAS_TRITON:
        raise RuntimeError("fused_attn_qkv_projection residual-mix path requires triton")
    assert x.is_cuda and x.ndim == 3 and x.is_contiguous()
    assert x0.is_cuda and x0.shape == x.shape and x0.is_contiguous()
    assert resid_scale.is_cuda and x0_scale.is_cuda
    assert resid_scale.ndim == 0 and x0_scale.ndim == 0
    B, T, K = x.size()
    M = B * T
    x_2d = x.view(M, K)
    x0_2d = x0.view(M, K)
    assert q_weight.is_cuda and k_weight.is_cuda and v_weight.is_cuda
    assert (
        q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and v_weight.is_contiguous()
    )
    assert head_dim % 2 == 0
    q_n = n_head * head_dim
    kv_n = n_kv_head * head_dim
    assert q_weight.shape == (q_n, K)
    assert k_weight.shape == (kv_n, K)
    assert v_weight.shape == (kv_n, K)
    _validate_rotary_table_4d(cos, T, head_dim)
    _validate_rotary_table_4d(sin, T, head_dim)

    has_value_embedding = ve_ids is not None or ve_weight is not None
    if has_value_embedding:
        assert ve_ids is not None and ve_weight is not None
        assert ve_ids.is_cuda and ve_weight.is_cuda and ve_weight.is_contiguous()
        assert ve_ids.shape == (B, T)
        assert ve_weight.ndim == 2 and ve_weight.shape[1] == kv_n
        ve_ids_for_kernel = ve_ids.reshape(M).contiguous()
        ve_weight_for_kernel = ve_weight
        assert ve_gate_weight is not None
        assert ve_gate_weight.is_cuda and ve_gate_weight.is_contiguous()
        assert 0 < ve_gate_channels <= K
        assert ve_gate_weight.shape == (n_kv_head, ve_gate_channels)
    else:
        assert ve_gate_weight is None
        ve_ids_for_kernel = x_2d
        ve_weight_for_kernel = x_2d
        ve_gate_weight = x_2d
        ve_gate_channels = 1

    norm_block_d = triton.next_power_of_2(K)
    norm_cfg = _pick_fused_add_norm_tile_config(M, norm_block_d, n_live_tiles=3)
    x_mix = torch.empty_like(x_2d)
    x_hat = torch.empty_like(x_2d)
    rms_inv = torch.empty((M,), dtype=torch.float32, device=x.device)
    wrap_triton(_fused_attn_qkv_projection_norm_fwd_kernel)[(triton.cdiv(M, norm_cfg.block_m),)](
        x_2d,
        x0_2d,
        resid_scale,
        x0_scale,
        x_mix,
        x_hat,
        rms_inv,
        M,
        K,
        eps,
        BLOCK_M=norm_cfg.block_m,
        BLOCK_K=norm_block_d,
        num_warps=norm_cfg.num_warps,
    )

    q = torch.empty((M, n_head, head_dim), dtype=x_hat.dtype, device=x_hat.device)
    k = torch.empty((M, n_kv_head, head_dim), dtype=x_hat.dtype, device=x_hat.device)
    v = torch.empty((M, n_kv_head, head_dim), dtype=x_hat.dtype, device=x_hat.device)
    qk_rms_inv = torch.empty((M, n_head + n_kv_head), dtype=torch.float32, device=x.device)
    if has_value_embedding:
        qkv_block_m = _QKV_FWD_VE_BLOCK_M
        qkv_block_k = _QKV_FWD_VE_BLOCK_K
        qkv_num_warps = _QKV_FWD_VE_NUM_WARPS
        qkv_num_stages = _QKV_FWD_VE_NUM_STAGES
    else:
        qkv_block_m = _QKV_FWD_NO_VE_BLOCK_M
        qkv_block_k = _QKV_FWD_NO_VE_BLOCK_K
        qkv_num_warps = _QKV_FWD_NO_VE_NUM_WARPS
        qkv_num_stages = _QKV_FWD_NO_VE_NUM_STAGES
    qkv_grid = (triton.cdiv(M, qkv_block_m), n_head + 2 * n_kv_head)
    wrap_triton(_fused_attn_qkv_projection_qkv_fwd_kernel)[qkv_grid](
        x_hat,
        q_weight,
        k_weight,
        v_weight,
        ve_ids_for_kernel,
        ve_weight_for_kernel,
        ve_gate_weight,
        cos,
        sin,
        q,
        k,
        v,
        qk_rms_inv,
        M,
        K,
        head_dim,
        n_head,
        n_kv_head,
        T,
        eps,
        scale,
        BLOCK_M=qkv_block_m,
        BLOCK_D=head_dim,
        BLOCK_K=qkv_block_k,
        HAS_VALUE_EMBEDDING=has_value_embedding,
        VE_GATE_CH=ve_gate_channels,
        VE_GATE_BLOCK=triton.next_power_of_2(ve_gate_channels),
        num_warps=qkv_num_warps,
        num_stages=qkv_num_stages,
    )
    return q, k, v, x_mix, rms_inv, qk_rms_inv


@torch.library.triton_op(
    "nanoops::fused_attn_qkv_projection_fwd",
    mutates_args=(),
)
def _fused_attn_qkv_projection_fwd_op(
    x: torch.Tensor,
    x0: torch.Tensor,
    resid_scale: torch.Tensor,
    x0_scale: torch.Tensor,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_gate_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Triton-op forward wrapper for fused residual/x0 mix + QKV projection.

    Returns internal M-view `(q, k, v, x_mix, rms_inv, qk_rms_inv)`.
    The public wrapper reshapes q/k/v/x_mix back to `(B, T, *)`.
    """
    return _fused_attn_qkv_projection_fwd_impl(
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


@torch.library.triton_op(
    "nanoops::fused_attn_qkv_projection_bwd",
    mutates_args=(),
)
def _fused_attn_qkv_projection_bwd_op(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    d_v: torch.Tensor,
    grad_x_mix: torch.Tensor,
    x: torch.Tensor,
    x0: torch.Tensor,
    resid_scale: torch.Tensor,
    x0_scale: torch.Tensor,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_gate_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_inv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    qk_rms_inv: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Triton-op backward wrapper.

    Inputs mirror `_fused_attn_qkv_projection_bwd_impl`. This op returns tensors only;
    optional gradients are represented by 1-element placeholders because
    `torch.library.triton_op` return values cannot be Optional.

    Returns:
      dx: same shape/dtype as x.
      dx0: same shape/dtype as x0.
      d_resid_scale/d_x0_scale: scalar grads.
      d_ve_weight: same shape/dtype as ve_weight, or a 1-element placeholder.
      d_ve_gate_weight: same shape/dtype as ve_gate_weight, or a 1-element placeholder.
      d_q_weight: same shape/dtype as q_weight.
      d_k_weight: same shape/dtype as k_weight.
      d_v_weight: same shape/dtype as v_weight.
    """
    (
        dx,
        dx0,
        d_resid_scale,
        d_x0_scale,
        d_ve_weight,
        d_ve_gate_weight,
        d_q_weight,
        d_k_weight,
        d_v_weight,
    ) = _fused_attn_qkv_projection_bwd_impl(
        d_q,
        d_k,
        d_v,
        ve_ids,
        ve_weight,
        ve_gate_channels,
        ve_gate_weight,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        rms_inv,
        q,
        k,
        qk_rms_inv,
        n_head,
        n_kv_head,
        head_dim,
        scale,
        eps,
        grad_x_mix=grad_x_mix,
        x_base=x,
        x0=x0,
        resid_scale=resid_scale,
        x0_scale=x0_scale,
    )
    if d_ve_weight is None:
        d_ve_weight = torch.empty(1, dtype=x.dtype, device=x.device)
    if d_ve_gate_weight is None:
        d_ve_gate_weight = torch.empty(1, dtype=x.dtype, device=x.device)
    return (
        dx,
        dx0,
        d_resid_scale,
        d_x0_scale,
        d_ve_weight,
        d_ve_gate_weight,
        d_q_weight,
        d_k_weight,
        d_v_weight,
    )


def _fused_attn_qkv_projection_setup_context(
    ctx: Any,
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        int,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        int,
        float,
        float,
    ],
    output: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> None:
    """Save tensors for fused residual/x0 mix + QKV projection backward."""
    (
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
    ) = inputs
    q, k, _v, _x_mix, rms_inv, qk_rms_inv = output
    ctx.save_for_backward(
        x,
        x0,
        resid_scale,
        x0_scale,
        ve_ids,
        ve_weight,
        ve_gate_weight,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        rms_inv,
        q,
        k,
        qk_rms_inv,
    )
    ctx.n_head = n_head
    ctx.n_kv_head = n_kv_head
    ctx.head_dim = head_dim
    ctx.scale = scale
    ctx.eps = eps
    ctx.ve_gate_channels = ve_gate_channels
    ctx.has_value_embedding = ve_ids is not None or ve_weight is not None


def _fused_attn_qkv_projection_autograd_backward(
    ctx: Any,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    grad_x_mix: torch.Tensor | None,
    _grad_rms_inv: torch.Tensor,
    _grad_qk_rms_inv: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    None,
    torch.Tensor | None,
    None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
]:
    """Backward for fused residual/x0 mix + QKV projection."""
    (
        x,
        x0,
        resid_scale,
        x0_scale,
        ve_ids,
        ve_weight,
        ve_gate_weight,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        rms_inv,
        q,
        k,
        qk_rms_inv,
    ) = ctx.saved_tensors
    if grad_x_mix is None:
        grad_x_mix = torch.zeros_like(x)
    (
        dx,
        dx0,
        d_resid_scale,
        d_x0_scale,
        d_ve_weight,
        d_ve_gate_weight,
        d_q_weight,
        d_k_weight,
        d_v_weight,
    ) = _fused_attn_qkv_projection_bwd_op(
        grad_q,
        grad_k,
        grad_v,
        grad_x_mix,
        x,
        x0,
        resid_scale,
        x0_scale,
        ve_ids,
        ve_weight,
        ctx.ve_gate_channels,
        ve_gate_weight,
        q_weight,
        k_weight,
        v_weight,
        cos,
        sin,
        rms_inv,
        q,
        k,
        qk_rms_inv,
        ctx.n_head,
        ctx.n_kv_head,
        ctx.head_dim,
        ctx.scale,
        ctx.eps,
    )
    if not ctx.has_value_embedding:
        d_ve_weight = None
        d_ve_gate_weight = None
    return (
        dx,
        dx0,
        d_resid_scale,
        d_x0_scale,
        None,
        d_ve_weight,
        None,
        d_ve_gate_weight,
        d_q_weight,
        d_k_weight,
        d_v_weight,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_fused_attn_qkv_projection_fwd_op.register_autograd(
    _fused_attn_qkv_projection_autograd_backward,
    setup_context=_fused_attn_qkv_projection_setup_context,
)


def fused_attn_qkv_projection(
    x: torch.Tensor,
    x0: torch.Tensor,
    resid_scale: torch.Tensor,
    x0_scale: torch.Tensor,
    ve_ids: torch.Tensor | None,
    ve_weight: torch.Tensor | None,
    ve_gate_channels: int,
    ve_gate_weight: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float = 1.2,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused residual/x0 blend + RMSNorm + QKV projection.

    Public GPT-layer math:
      x_mix = resid_scale * x + x0_scale * x0
      q, k, v = RMSNorm(x_mix) projected through Q/K/V, with Q/K rotary
                and QK RMSNorm applied before writeback.

    Public inputs/outputs keep `(B, T, *)`; the Triton op uses the internal
    flattened `M=B*T` view. `x_mix` is returned because the attention residual
    path needs to compute `x_mix + attn_out`.

    Returns:
      q: (B, T, n_head, head_dim), dtype=x.dtype.
      k: (B, T, n_kv_head, head_dim), dtype=x.dtype.
      v: (B, T, n_kv_head, head_dim), dtype=x.dtype.
      x_mix: (B, T, K), dtype=x.dtype.
    """
    B, T, K = x.shape
    q, k, v, x_mix, _rms_inv, _qk_rms_inv = _fused_attn_qkv_projection_fwd_op(
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
    return (
        q.view(B, T, n_head, head_dim),
        k.view(B, T, n_kv_head, head_dim),
        v.view(B, T, n_kv_head, head_dim),
        x_mix.view(B, T, K),
    )
