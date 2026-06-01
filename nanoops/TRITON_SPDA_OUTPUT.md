# Chapter 5 — `fused_attn_spda_and_output`: Flash-style SDPA + output projection

> Part of [nanoops Triton Kernels](TRITON.md). Chinese version: [TRITON_SPDA_OUTPUT_zh.md](TRITON_SPDA_OUTPUT_zh.md).

This chapter covers `nanoops/triton_fused_attn_spda.py` and `nanoops/triton_fused_attn_spda_and_output.py`.

---

## 5.1 Scope

The public training op is `fused_attn_spda_and_output`:

```python
attn_out = sliding_window_sdpa(q, k, v)
y        = residual + attn_out @ proj_weight.T
```

The SDPA side is FlashAttention-style: it never materializes the score matrix `S` or probability matrix `P`. The output side is fused with SDPA backward so the output-projection gradient path produces the FlashAttention `delta` buffer directly.

## 5.2 Public API and Tensor Contract

```python
def fused_attn_spda_and_output(
    q: torch.Tensor,            # (B, T, H_q, D), contiguous CUDA
    k: torch.Tensor,            # (B, T, H_kv, D), contiguous CUDA
    v: torch.Tensor,            # (B, T, H_kv, D), contiguous CUDA
    proj_weight: torch.Tensor,  # (C, H_q * D), contiguous CUDA
    residual: torch.Tensor,     # (B, T, C), contiguous CUDA
    window_size: int,
) -> torch.Tensor               # (B, T, C)
```

Constraints in the current implementation:

- `T_q == T_kv`; varlen is not implemented.
- GQA is supported when `H_q % H_kv == 0`; `GQA_GROUP = H_q // H_kv`.
- The full-context path is selected by `window_size >= T`; otherwise the sliding-causal mask is used.

## 5.3 SDPA Forward Math

For one batch and one query head, with query row `i`, key row `j`, and head channel `d`:

```text
visible(i, j) = max(0, i - window_size + 1) <= j <= i
S_ij          = (1/sqrt(D)) * sum_d Q_i,d * K_j,d    if visible(i, j)
              = -inf                                otherwise
P_ij          = exp(S_ij - LSE_i)
LSE_i         = log(sum_j exp(S_ij))
O_i,d         = sum_j P_ij * V_j,d
```

The forward kernel streams K/V tiles while holding a Q tile. Each row tracks the online-softmax triple `(m, l, acc)`:

```text
m_new   = max(m, max_j s_j)
alpha   = exp(m - m_new)
p_hat_j = exp(s_j - m_new)
l_new   = alpha * l + sum_j p_hat_j
acc_new = alpha * acc + p_hat @ V_tile
O       = acc / l
LSE     = m + log(l)
```

## 5.4 GQA Schedule

Input tensors stay in `(B, T, H, D)` layout. The SDPA grid is:

```text
grid = (B * M_tiles, H_kv)
```

A program owns one `(batch, row-tile, kv-head)` tuple. For GQA it handles all query heads that share that K/V head in the same program. The flattened Q tile has `GQA_GROUP * BLOCK_M` logical rows, but it is row-major expanded:

```text
row_in_tile = offs_hm // GQA_GROUP
head_off    = offs_hm - row_in_tile * GQA_GROUP
offs_m      = pid_m * BLOCK_M + row_in_tile
hid         = kv_hid * GQA_GROUP + head_off
```

This lets one K/V tile feed the whole query-head group instead of loading the same K/V tile once per query head. Non-GQA is the same path with `GQA_GROUP = 1`.

## 5.5 Forward Pipeline

Forward uses two logical stages:

| Stage | Kernel/op | Output |
|---|---|---|
| SDPA | `_fused_attn_spda_fwd_kernel` via `_fused_attn_spda_fwd_op` | `attn_out`, `lse` |
| Output projection | `_fused_attn_spda_and_output_proj_fwd_kernel` | `y = residual + attn_out @ proj_weight.T` |

The projection kernel flattens `attn_out` to `(B*T, H_q*D)` and `residual` to `(B*T, C)`. `proj_weight` can be a fp32 master weight; it is cast to activation dtype on load before `tl.dot`.

## 5.6 Backward Pipeline

The combined backward is arranged around the FlashAttention `delta` term:

```text
delta_i = sum_d O_i,d * dO_i,d
```

Output projection backward first materializes `d_attn_out` and `delta` together:

```text
d_attn_out[m, h, d] = sum_o dy[m, o] * proj_weight[o, h, d]
delta[m, h]         = sum_d attn_out[m, h, d] * d_attn_out[m, h, d]
d_proj_weight[o, i] = sum_m dy[m, o] * attn_out[m, i]
d_residual          = dy
```

Then SDPA backward consumes caller-provided `delta`:

| Stage | Kernel | Ownership |
|---|---|---|
| dQ | `_fused_attn_spda_dq_bwd_kernel` | Q-row tile owned, writes `dq` directly |
| dK/dV | `_fused_attn_spda_dkv_bwd_kernel` | K/V tile owned, writes `dk`/`dv` directly |

For each recomputed score tile:

```text
P_ij  = exp(S_ij - LSE_i)
dP_ij = sum_d dO_i,d * V_j,d
dS_ij = P_ij * (dP_ij - delta_i) / sqrt(D)
dQ_i += sum_j dS_ij * K_j
dV_j += sum_i P_ij * dO_i
dK_j += sum_i dS_ij * Q_i
```

Because dQ and dK/dV are owned by their natural output tiles, SDPA backward does not need atomics for `dk` or `dv`.

## 5.7 Boundary and Mask Handling

Sliding-window attention skips K/V tiles that are fully outside the visible interval. The streamed K/V loop distinguishes:

- left boundary tiles, which need elementwise causal/window masks;
- full-valid tiles, where every row in the Q tile can see every key in the K tile;
- right boundary tiles, which need elementwise masks again.

Full-context causal attention uses a constexpr fast path that drops the sliding-window lower-bound check. This is still simpler than industrial FA-3/TMA kernels: there is no varlen path, no persistent scheduler, and no asynchronous TMA pipeline.

## 5.8 Saved State

The autograd wrapper saves:

| Saved | Why |
|---|---|
| `q`, `k`, `v` | recompute SDPA scores in backward |
| `attn_out` | output projection backward and `delta` |
| `lse` | softmax backward normalization |
| `proj_weight` | output projection backward |

Not saved: attention scores `S`, probabilities `P`, or `d_attn_out`. `d_attn_out` is materialized during backward because SDPA backward needs it as `dO`.

## 5.9 Performance Notes

The important fusion is not the forward projection itself; it is the backward hand-off. A standalone output projection backward would compute `d_attn_out`, and standalone SDPA backward would launch another pass to compute `delta = sum(attn_out * d_attn_out)`. The fused op computes both in `_fused_attn_spda_and_output_proj_dattn_delta_bwd_kernel`, removing one pass over `(B*T, H_q, D)`.

The current schedule is deliberately d24-focused: `D=128`, contiguous `(B, T, H, D)` tensors, GQA-aware K/V reuse, and direct bf16 tensor-core matmuls with fp32 accumulators.
