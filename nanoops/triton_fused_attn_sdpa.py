"""Attention SDPA-side Triton kernels for nanoops.

Contains `flash_sdpa`: Flash-style sliding-causal SDPA with a split
backward. Re-exported through `nanoops.triton_kernels`.
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


def _pick_sdpa_fwd_tile_config(
    head_dim: int,
) -> tuple[int, int, int, int]:
    """Return `(block_m, block_n, num_warps, num_stages)` for SDPA forward."""
    if head_dim >= 128:
        # d24 forward sweep winner (B=1, T=2048, H=12, D=128, bf16).
        return 128, 64, 8, 1
    return 64, 64, 4, 1


def _pick_sdpa_bwd_tile_config(
    head_dim: int,
) -> tuple[int, int, int, int]:
    """Return `(block_m, block_n, num_warps, num_stages)` for split SDPA backward."""
    if head_dim >= 128:
        # d24 split-bwd sweep winner. Larger forward tiles OOR in dKV.
        return 32, 32, 4, 1
    return 64, 64, 4, 1


def _pick_gqa_block_m(block_m: int, gqa_group: int) -> int:
    """Keep total `(query rows × grouped heads)` similar to the baseline tile."""
    return max(8, block_m // gqa_group)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


# ─────────────────────────────────────────────────────────────────────
# Flash-style sliding-window SDPA in Triton.
#
# Standard Flash Attention pattern, adapted for nanchat's sliding-causal mask.
#
# Forward math for one (batch, head), with i indexing query rows and j key rows:
#   visible(i, j) = max(0, i - WINDOW + 1) <= j <= i
#   S_ij          = sm_scale * dot(Q_i, K_j)          if visible(i, j)
#                 = -inf                             otherwise
#   P_ij          = exp(S_ij - LSE_i)
#   LSE_i         = log(sum_j exp(S_ij))
#   O_i           = sum_j P_ij * V_j
#
# The kernel never materializes S or P as (L, L). It tiles Q by BLOCK_M rows and
# streams K/V by BLOCK_N rows. For each Q row it maintains an online-softmax
# triple `(m, l, acc)` where:
#   m   = running max score
#   l   = running sum exp(score - m)
#   acc = running sum exp(score - m) * V
# For a new score tile `s`:
#   m_new   = max(m, max_j s_j)
#   alpha   = exp(m - m_new)
#   p_hat_j = exp(s_j - m_new)
#   l_new   = alpha * l + sum_j p_hat_j
#   acc_new = alpha * acc + p_hat @ V_tile
# Final output is `O = acc / l`; saved backward state is
# `LSE = m + log(l)` per query row.
#
# Backward math, given G = dO, is split into three Triton passes:
#
#   Preprocess, Q-row owned:
#     Delta_i = sum_d O_id * G_id
#
#   dQ pass, Q-tile owned:
#     P_ij  = exp(S_ij - LSE_i)              # recomputed from Q/K/LSE
#     dP_ij = dot(G_i, V_j)
#     dS_ij = P_ij * (dP_ij - Delta_i) * sm_scale
#     dQ_i += sum_j dS_ij * K_j
#
#   dK/dV pass, K/V-tile owned:
#     P_ij  = exp(S_ij - LSE_i)
#     dV_j  = sum_i P_ij * G_i
#     dP_ij = dot(G_i, V_j)
#     dS_ij = P_ij * (dP_ij - Delta_i) * sm_scale
#     dK_j  = sum_i dS_ij * Q_i
#
# This is the FlashAttention-style split backward: dQ and dK/dV are written by
# their owning tiles, so dK/dV no longer need atomic accumulation.
#
# Sliding-window mask: per-tile lower-bound `j ≥ i - W + 1`. Combined
# with causal `j ≤ i`, this lets us skip entire K/V tiles whose j range
# is outside [i_min - W + 1, i_max].
#
# Scope of v1:
#   - Supports GQA when H_q is an integer multiple of H_kv. Each query head
#     maps to `kv_head = query_head // (H_q / H_kv)`.
#   - The GQA path groups all query heads that share one K/V head into one
#     program. Non-GQA is the same kernel with GQA_GROUP=1.
#   - No FA-3-style asynchronous TMA / split-k. Single-stage tiling.
#   - bf16 inputs OK (matmul in fp32 accumulator); fp16/fp32 also work.
# ─────────────────────────────────────────────────────────────────────


if _HAS_TRITON:

    @triton.jit
    def _flash_attn_fwd_gqa_kernel(
        Q,  # (B, M, H_Q, D) — in: query
        K,  # (B, N, H_KV, D) — in: key
        V,  # (B, N, H_KV, D) — in: value
        sm_scale,  # float — attention scale, usually D**-0.5
        LSE,  # (B, M, H_Q) fp32 — out: row log-sum-exp for bwd
        OUT,  # (B, M, H_Q, D) — out: attention output
        B,  # int — batch size
        H_Q,  # int — number of query/output heads
        H_KV,  # int — number of key/value heads
        M,  # int — query sequence length
        N,  # int — key/value sequence length
        WINDOW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HM: tl.constexpr,
        M_TILES: tl.constexpr,
        GQA_GROUP: tl.constexpr,
        D: tl.constexpr,
    ):
        """GQA forward: one program per (batch × Q-tile × K/V head).

        `BLOCK_HM` flattens `(GQA_GROUP, BLOCK_M)`, so one K/V tile feeds all
        query heads that share the same K/V head. With non-GQA, GQA_GROUP=1.
        """
        pid_bm = tl.program_id(0)
        kv_hid = tl.program_id(1)
        bid = pid_bm // M_TILES
        pid_m = pid_bm - bid * M_TILES

        kv_tile_start = tl.maximum(0, pid_m * BLOCK_M - WINDOW + 1) // BLOCK_N
        kv_high = tl.minimum(N, pid_m * BLOCK_M + BLOCK_M)
        kv_tile_end = (kv_high + BLOCK_N - 1) // BLOCK_N
        batch_q_base = bid * M * H_Q * D
        batch_k_base = bid * N * H_KV * D
        batch_lse_base = bid * M * H_Q
        offs_hm = tl.arange(0, BLOCK_HM)
        head_off = offs_hm // BLOCK_M
        row_in_tile = offs_hm - head_off * BLOCK_M
        offs_m = pid_m * BLOCK_M + row_in_tile
        offs_d = tl.arange(0, D)

        hid = kv_hid * GQA_GROUP + head_off
        hm_mask = (
            (offs_m < M)
            & (head_off < GQA_GROUP)
            & (kv_hid < H_KV)
            & (hid < H_Q)
        )

        q_ptrs = (
            Q
            + batch_q_base
            + offs_m[:, None] * H_Q * D
            + hid[:, None] * D
            + offs_d[None, :]
        )
        q = tl.load(q_ptrs, mask=hm_mask[:, None], other=0.0)

        m_i = tl.full((BLOCK_HM,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_HM,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_HM, D), dtype=tl.float32)

        offs_n_base = tl.arange(0, BLOCK_N)
        for kv_idx in range(kv_tile_start, kv_tile_end):
            offs_n = kv_idx * BLOCK_N + offs_n_base
            n_mask = (offs_n < N) & (kv_hid < H_KV)

            k_ptrs = (
                K
                + batch_k_base
                + offs_n[:, None] * H_KV * D
                + kv_hid * D
                + offs_d[None, :]
            )
            v_ptrs = (
                V
                + batch_k_base
                + offs_n[:, None] * H_KV * D
                + kv_hid * D
                + offs_d[None, :]
            )
            k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

            s = tl.dot(q, tl.trans(k_tile)) * sm_scale
            j = offs_n[None, :]
            i = offs_m[:, None]
            mask_keep = (
                (j <= i)
                & (j >= i - WINDOW + 1)
                & hm_mask[:, None]
                & n_mask[None, :]
            )
            s = tl.where(mask_keep, s, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            all_masked = m_new == -float("inf")
            alpha = tl.where(all_masked, 1.0, tl.exp(m_i - m_new))
            p_unscaled = tl.where(
                all_masked[:, None],
                0.0,
                tl.exp(s - m_new[:, None]),
            )
            l_i = l_i * alpha + tl.sum(p_unscaled, axis=1)
            acc = acc * alpha[:, None] + tl.dot(
                p_unscaled.to(v_tile.dtype),
                v_tile,
            )
            m_i = m_new

        acc = acc / l_i[:, None]
        lse = m_i + tl.log(l_i)

        o_ptrs = (
            OUT
            + batch_q_base
            + offs_m[:, None] * H_Q * D
            + hid[:, None] * D
            + offs_d[None, :]
        )
        tl.store(o_ptrs, acc.to(OUT.dtype.element_ty), mask=hm_mask[:, None])

        lse_ptrs = LSE + batch_lse_base + offs_m * H_Q + hid
        tl.store(lse_ptrs, lse, mask=hm_mask)

    @triton.jit
    def _flash_attn_bwd_preprocess_kernel(
        OUT,  # (B, M, H_Q, D) — in: forward output
        dO,  # (B, M, H_Q, D) — in: gradient of output
        DELTA,  # (B, M, H_Q) fp32 — out: row dot(O, dO)
        B,  # int — batch size
        H_Q,  # int — number of query/output heads
        M,  # int — query sequence length
        BLOCK_M: tl.constexpr,
        M_TILES: tl.constexpr,
        D: tl.constexpr,
    ):
        """Precompute Delta[i] = sum_j o[i, j] * dO[i, j] — used in bwd to skip
        the inner softmax-bwd reduction. This is the classic Flash trick."""
        pid_bm = tl.program_id(0)
        hid = tl.program_id(1)
        bid = pid_bm // M_TILES
        pid_m = pid_bm - bid * M_TILES

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        m_mask = offs_m < M
        batch_q_base = bid * M * H_Q * D
        batch_lse_base = bid * M * H_Q

        head_mask = hid < H_Q
        o_base = batch_q_base + hid * D
        o_ptrs = OUT + o_base + offs_m[:, None] * H_Q * D + offs_d[None, :]
        do_ptrs = dO + o_base + offs_m[:, None] * H_Q * D + offs_d[None, :]
        mask = m_mask[:, None] & head_mask
        o = tl.load(o_ptrs, mask=mask, other=0.0)
        do = tl.load(do_ptrs, mask=mask, other=0.0)
        d_row = tl.sum(o * do, axis=1, dtype=tl.float32)
        d_ptrs = DELTA + batch_lse_base + offs_m * H_Q + hid
        tl.store(d_ptrs, d_row, mask=m_mask & head_mask)

    @triton.jit
    def _flash_attn_bwd_dq_gqa_kernel(
        Q,  # (B, M, H_Q, D) — in: query
        K,  # (B, N, H_KV, D) — in: key
        V,  # (B, N, H_KV, D) — in: value
        sm_scale,  # float — attention scale
        LSE,  # (B, M, H_Q) fp32 — in: saved row log-sum-exp
        DELTA,  # (B, M, H_Q) fp32 — in: row dot(O, dO)
        dO,  # (B, M, H_Q, D) — in: grad wrt output
        dQ,  # (B, M, H_Q, D) — out: grad wrt query
        B,  # int — batch size
        H_Q,  # int — number of query/output heads
        H_KV,  # int — number of key/value heads
        M,  # int — query sequence length
        N,  # int — key/value sequence length
        WINDOW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HM: tl.constexpr,
        M_TILES: tl.constexpr,
        GQA_GROUP: tl.constexpr,
        D: tl.constexpr,
    ):
        """Compute dQ for one `(batch, Q-tile, K/V head)`.

        Like the forward path, this flattens `(GQA_GROUP, BLOCK_M)` into the
        Q-row tile so one K/V tile feeds all Q heads sharing it. dK/dV are
        computed in a separate K/V-owned pass to avoid atomics.
        """
        pid_bm = tl.program_id(0)
        kv_hid = tl.program_id(1)
        bid = pid_bm // M_TILES
        pid_m = pid_bm - bid * M_TILES

        offs_hm = tl.arange(0, BLOCK_HM)
        head_off = offs_hm // BLOCK_M
        row_in_tile = offs_hm - head_off * BLOCK_M
        offs_m = pid_m * BLOCK_M + row_in_tile
        offs_d = tl.arange(0, D)

        kv_tile_start = tl.maximum(0, pid_m * BLOCK_M - WINDOW + 1) // BLOCK_N
        kv_high = tl.minimum(N, pid_m * BLOCK_M + BLOCK_M)
        kv_tile_end = (kv_high + BLOCK_N - 1) // BLOCK_N
        batch_q_base = bid * M * H_Q * D
        batch_k_base = bid * N * H_KV * D
        batch_lse_base = bid * M * H_Q

        hid = kv_hid * GQA_GROUP + head_off
        hm_mask = (
            (offs_m < M)
            & (head_off < GQA_GROUP)
            & (kv_hid < H_KV)
            & (hid < H_Q)
        )

        q_ptrs = (
            Q
            + batch_q_base
            + offs_m[:, None] * H_Q * D
            + hid[:, None] * D
            + offs_d[None, :]
        )
        do_ptrs = (
            dO
            + batch_q_base
            + offs_m[:, None] * H_Q * D
            + hid[:, None] * D
            + offs_d[None, :]
        )
        lse_ptrs = LSE + batch_lse_base + offs_m * H_Q + hid
        d_ptrs = DELTA + batch_lse_base + offs_m * H_Q + hid
        q = tl.load(q_ptrs, mask=hm_mask[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=hm_mask[:, None], other=0.0)
        lse = tl.load(lse_ptrs, mask=hm_mask, other=0.0)
        d_row = tl.load(d_ptrs, mask=hm_mask, other=0.0)
        dq_acc = tl.zeros((BLOCK_HM, D), dtype=tl.float32)

        offs_n_base = tl.arange(0, BLOCK_N)
        for kv_idx in range(kv_tile_start, kv_tile_end):
            offs_n = kv_idx * BLOCK_N + offs_n_base
            n_mask = (offs_n < N) & (kv_hid < H_KV)
            k_ptrs = (
                K
                + batch_k_base
                + offs_n[:, None] * H_KV * D
                + kv_hid * D
                + offs_d[None, :]
            )
            v_ptrs = (
                V
                + batch_k_base
                + offs_n[:, None] * H_KV * D
                + kv_hid * D
                + offs_d[None, :]
            )
            k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

            s = tl.dot(q, tl.trans(k_tile)) * sm_scale
            j = offs_n[None, :]
            i = offs_m[:, None]
            mask_keep = (
                (j <= i)
                & (j >= i - WINDOW + 1)
                & hm_mask[:, None]
                & n_mask[None, :]
            )
            s = tl.where(mask_keep, s, -float("inf"))
            p = tl.exp(s - lse[:, None])

            dp = tl.dot(do, tl.trans(v_tile))
            ds = p * (dp - d_row[:, None]) * sm_scale
            dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile)

        dq_ptrs = (
            dQ
            + batch_q_base
            + offs_m[:, None] * H_Q * D
            + hid[:, None] * D
            + offs_d[None, :]
        )
        tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=hm_mask[:, None])

    @triton.jit
    def _flash_attn_bwd_dkv_gqa_kernel(
        Q,  # (B, M, H_Q, D) — in: query
        K,  # (B, N, H_KV, D) — in: key
        V,  # (B, N, H_KV, D) — in: value
        sm_scale,  # float — attention scale
        LSE,  # (B, M, H_Q) fp32 — in: saved row log-sum-exp
        DELTA,  # (B, M, H_Q) fp32 — in: row dot(O, dO)
        dO,  # (B, M, H_Q, D) — in: grad wrt output
        dK,  # (B, N, H_KV, D) — out: grad wrt key
        dV,  # (B, N, H_KV, D) — out: grad wrt value
        B,  # int — batch size
        H_Q,  # int — number of query/output heads
        H_KV,  # int — number of key/value heads
        M,  # int — query sequence length
        N,  # int — key/value sequence length
        WINDOW: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HM: tl.constexpr,
        N_TILES: tl.constexpr,
        GQA_GROUP: tl.constexpr,
        D: tl.constexpr,
    ):
        """Compute dK/dV for one `(batch, K/V-tile, K/V head)`.

        This is the FlashAttention-style K/V-owned backward pass: every
        dK/dV tile has a single writer. It loops over the Q tiles that can
        attend to this K/V tile under the sliding-causal mask, so no atomic
        accumulation is needed.
        """
        pid_bn = tl.program_id(0)
        kv_hid = tl.program_id(1)
        bid = pid_bn // N_TILES
        pid_n = pid_bn - bid * N_TILES

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        n_mask = (offs_n < N) & (kv_hid < H_KV)

        batch_q_base = bid * M * H_Q * D
        batch_k_base = bid * N * H_KV * D
        batch_lse_base = bid * M * H_Q

        k_ptrs = (
            K
            + batch_k_base
            + offs_n[:, None] * H_KV * D
            + kv_hid * D
            + offs_d[None, :]
        )
        v_ptrs = (
            V
            + batch_k_base
            + offs_n[:, None] * H_KV * D
            + kv_hid * D
            + offs_d[None, :]
        )
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        dk_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
        dv_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)

        kv_start = pid_n * BLOCK_N
        q_tile_start = kv_start // BLOCK_M
        q_high = tl.minimum(M, kv_start + BLOCK_N + WINDOW - 1)
        q_tile_end = (q_high + BLOCK_M - 1) // BLOCK_M

        offs_hm = tl.arange(0, BLOCK_HM)
        head_off = offs_hm // BLOCK_M
        row_in_tile = offs_hm - head_off * BLOCK_M
        hid = kv_hid * GQA_GROUP + head_off

        for q_idx in range(q_tile_start, q_tile_end):
            offs_m = q_idx * BLOCK_M + row_in_tile
            hm_mask = (
                (offs_m < M)
                & (head_off < GQA_GROUP)
                & (kv_hid < H_KV)
                & (hid < H_Q)
            )

            q_ptrs = (
                Q
                + batch_q_base
                + offs_m[:, None] * H_Q * D
                + hid[:, None] * D
                + offs_d[None, :]
            )
            do_ptrs = (
                dO
                + batch_q_base
                + offs_m[:, None] * H_Q * D
                + hid[:, None] * D
                + offs_d[None, :]
            )
            lse_ptrs = LSE + batch_lse_base + offs_m * H_Q + hid
            d_ptrs = DELTA + batch_lse_base + offs_m * H_Q + hid
            q = tl.load(q_ptrs, mask=hm_mask[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=hm_mask[:, None], other=0.0)
            lse = tl.load(lse_ptrs, mask=hm_mask, other=0.0)
            d_row = tl.load(d_ptrs, mask=hm_mask, other=0.0)

            s = tl.dot(q, tl.trans(k_tile)) * sm_scale
            j = offs_n[None, :]
            i = offs_m[:, None]
            mask_keep = (
                (j <= i)
                & (j >= i - WINDOW + 1)
                & hm_mask[:, None]
                & n_mask[None, :]
            )
            s = tl.where(mask_keep, s, -float("inf"))
            p = tl.exp(s - lse[:, None])

            dv_acc += tl.dot(tl.trans(p).to(do.dtype), do)
            dp = tl.dot(do, tl.trans(v_tile))
            ds = p * (dp - d_row[:, None]) * sm_scale
            dk_acc += tl.dot(tl.trans(ds).to(q.dtype), q)

        dk_ptrs = (
            dK
            + batch_k_base
            + offs_n[:, None] * H_KV * D
            + kv_hid * D
            + offs_d[None, :]
        )
        dv_ptrs = (
            dV
            + batch_k_base
            + offs_n[:, None] * H_KV * D
            + kv_hid * D
            + offs_d[None, :]
        )
        tl.store(dk_ptrs, dk_acc.to(dK.dtype.element_ty), mask=n_mask[:, None])
        tl.store(dv_ptrs, dv_acc.to(dV.dtype.element_ty), mask=n_mask[:, None])


def _flash_sdpa_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Flash-style sliding-causal SDPA and return `(out, lse)`.

    Args:
      q: (B, M, H_q, D) contiguous CUDA query tensor.
      k: (B, N, H_kv, D) contiguous CUDA key tensor; v1 requires N == M.
      v: (B, N, H_kv, D) contiguous CUDA value tensor.
      window_size: total visible keys per query.

    Returns:
      out: (B, M, H_q, D), dtype=q.dtype.
      lse: (B, M, H_q), fp32 row log-sum-exp for backward.
    """
    if not _HAS_TRITON:
        raise RuntimeError("flash_sdpa requires triton")
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.ndim == k.ndim == v.ndim == 4
    B, M, h_q, D = q.shape
    B_k, N, h_kv, D_k = k.shape
    assert v.shape == (B_k, N, h_kv, D_k)
    assert B == B_k and D == D_k, f"q{k.shape=} / {q.shape=} / {v.shape=}"
    assert M == N, f"flash_sdpa v1 requires same query/key length, got M={M}, N={N}"
    assert h_q % h_kv == 0, f"H_q={h_q} must be divisible by H_kv={h_kv}"
    gqa_group = h_q // h_kv
    sm_scale = D**-0.5

    out = torch.empty_like(q)
    lse = torch.empty((B, M, h_q), dtype=torch.float32, device=q.device)

    # Keep tensor layout as `(B, T, H, D)` into Triton. The launch grid
    # prioritizes batch/row tiles on axis 0 and uses axis 1 for heads.
    block_m, block_n, num_warps, num_stages = _pick_sdpa_fwd_tile_config(D)
    segment_block_m = _pick_gqa_block_m(block_m, gqa_group)
    segment_block_hm = _next_power_of_2(segment_block_m * gqa_group)
    m_tiles = triton.cdiv(M, segment_block_m)
    grid = (B * m_tiles, h_kv)
    wrap_triton(_flash_attn_fwd_gqa_kernel)[grid](
        q,
        k,
        v,
        sm_scale,
        lse,
        out,
        B,
        h_q,
        h_kv,
        M,
        N,
        WINDOW=window_size,
        BLOCK_M=segment_block_m,
        BLOCK_N=block_n,
        BLOCK_HM=segment_block_hm,
        M_TILES=m_tiles,
        GQA_GROUP=gqa_group,
        D=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, lse


def _flash_sdpa_bwd_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backprop for Flash-style sliding-causal SDPA.

    Args:
      do: (B, M, H_q, D), grad wrt forward output.
      q/out: (B, M, H_q, D), saved forward query/output.
      k/v: (B, M, H_kv, D), saved forward key/value.
      lse: (B, M, H_q), fp32 saved forward row log-sum-exp.
      window_size: total visible keys per query.

    Returns:
      dq, dk, dv with the same shapes/dtypes as q, k, v.
    """
    if not _HAS_TRITON:
        raise RuntimeError("flash_sdpa backward requires triton")
    do = do.contiguous()
    B, M, h_q, D = q.shape
    B_k, N, h_kv, D_k = k.shape
    assert v.shape == (B_k, N, h_kv, D_k)
    assert do.shape == q.shape and out.shape == q.shape
    assert lse.shape == (B, M, h_q)
    assert B == B_k and D == D_k and M == N
    assert h_q % h_kv == 0
    gqa_group = h_q // h_kv
    sm_scale = D**-0.5
    block_m, block_n, num_warps, num_stages = _pick_sdpa_bwd_tile_config(D)

    # Delta[i] = sum_j out[i, j] * dO[i, j].
    delta = torch.empty((B, M, h_q), dtype=torch.float32, device=q.device)
    block_m_pre = block_m
    m_tiles_pre = triton.cdiv(M, block_m_pre)
    grid_pre = (B * m_tiles_pre, h_q)
    wrap_triton(_flash_attn_bwd_preprocess_kernel)[grid_pre](
        out,
        do,
        delta,
        B,
        h_q,
        M,
        BLOCK_M=block_m_pre,
        M_TILES=m_tiles_pre,
        D=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Split FlashAttention-style backward:
    #   1. Q-owned pass writes dQ.
    #   2. K/V-owned pass writes dK/dV directly, avoiding atomics.
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    segment_block_m = _pick_gqa_block_m(block_m, gqa_group)
    segment_block_hm = _next_power_of_2(segment_block_m * gqa_group)
    m_tiles = triton.cdiv(M, segment_block_m)
    grid_dq = (B * m_tiles, h_kv)
    wrap_triton(_flash_attn_bwd_dq_gqa_kernel)[grid_dq](
        q,
        k,
        v,
        sm_scale,
        lse,
        delta,
        do,
        dq,
        B,
        h_q,
        h_kv,
        M,
        N,
        WINDOW=window_size,
        BLOCK_M=segment_block_m,
        BLOCK_N=block_n,
        BLOCK_HM=segment_block_hm,
        M_TILES=m_tiles,
        GQA_GROUP=gqa_group,
        D=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    n_tiles = triton.cdiv(N, block_n)
    grid_dkv = (B * n_tiles, h_kv)
    wrap_triton(_flash_attn_bwd_dkv_gqa_kernel)[grid_dkv](
        q,
        k,
        v,
        sm_scale,
        lse,
        delta,
        do,
        dk,
        dv,
        B,
        h_q,
        h_kv,
        M,
        N,
        WINDOW=window_size,
        BLOCK_M=segment_block_m,
        BLOCK_N=block_n,
        BLOCK_HM=segment_block_hm,
        N_TILES=n_tiles,
        GQA_GROUP=gqa_group,
        D=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dq, dk, dv


@torch.library.triton_op(
    "nanoops::flash_sdpa_fwd",
    mutates_args=(),
)
def _flash_sdpa_fwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-op forward wrapper returning `(out, lse)`."""
    return _flash_sdpa_fwd_impl(q, k, v, window_size)


@torch.library.triton_op(
    "nanoops::flash_sdpa_bwd",
    mutates_args=(),
)
def _flash_sdpa_bwd_op(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-op backward wrapper returning `(dq, dk, dv)`."""
    return _flash_sdpa_bwd_impl(do, q, k, v, out, lse, window_size)


def _flash_sdpa_setup_context(
    ctx: Any,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    output: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Save tensors for `nanoops::flash_sdpa_fwd` backward."""
    q, k, v, window_size = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.window_size = window_size


def _flash_sdpa_autograd_backward(
    ctx: Any,
    do: torch.Tensor,
    _d_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """Autograd callback for Flash-style SDPA."""
    q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = _flash_sdpa_bwd_op(do, q, k, v, out, lse, ctx.window_size)
    return dq, dk, dv, None


_flash_sdpa_fwd_op.register_autograd(
    _flash_sdpa_autograd_backward,
    setup_context=_flash_sdpa_setup_context,
)


def flash_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Flash-style sliding-causal SDPA. q: (B, L, H_q, D), k/v: (B, L, H_kv, D).

    window_size: total keys each query attends to (= nanchat's window+1).

    Args:
      q: (B, L, H_q, D) contiguous CUDA query tensor.
      k: (B, L, H_kv, D) contiguous CUDA key tensor.
      v: (B, L, H_kv, D) contiguous CUDA value tensor.
      window_size: total visible keys per query.

    Returns:
      (B, L, H_q, D) attention output.
    """
    out, _lse = _flash_sdpa_fwd_op(q, k, v, window_size)
    return out
