# 第 5 章 —— `fused_attn_spda_and_output`：Flash-style SDPA + output projection

> 属于 [nanoops Triton Kernels](TRITON_zh.md)。English version: [TRITON_SPDA_OUTPUT.md](TRITON_SPDA_OUTPUT.md)。

本文档覆盖 `nanoops/triton_fused_attn_spda.py` 和 `nanoops/triton_fused_attn_spda_and_output.py`。

---

## 5.1 范围

公开训练 op 是 `fused_attn_spda_and_output`：

```python
attn_out = sliding_window_sdpa(q, k, v)
y        = residual + attn_out @ proj_weight.T
```

SDPA 侧是 FlashAttention-style：不物化 score 矩阵 `S`，也不物化 probability 矩阵 `P`。output 侧和 SDPA backward 的交界被 fuse 了：output-projection backward 在产生 `d_attn_out` 的同时直接产生 FlashAttention backward 需要的 `delta` buffer。

## 5.2 公开 API 和张量约定

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

当前实现约束：

- `T_q == T_kv`；还没有 varlen。
- 支持 GQA，要求 `H_q % H_kv == 0`；`GQA_GROUP = H_q // H_kv`。
- `window_size >= T` 时走 full-context causal fast path；否则走 sliding-causal mask。

## 5.3 SDPA forward 数学

对一个 batch、一个 query head，query row 为 `i`，key row 为 `j`，head channel 为 `d`：

```text
visible(i, j) = max(0, i - window_size + 1) <= j <= i
S_ij          = (1/sqrt(D)) * sum_d Q_i,d * K_j,d    if visible(i, j)
              = -inf                                otherwise
P_ij          = exp(S_ij - LSE_i)
LSE_i         = log(sum_j exp(S_ij))
O_i,d         = sum_j P_ij * V_j,d
```

forward kernel 固定一个 Q tile，stream K/V tile。每个 Q row 维护 online-softmax 三元组 `(m, l, acc)`：

```text
m_new   = max(m, max_j s_j)
alpha   = exp(m - m_new)
p_hat_j = exp(s_j - m_new)
l_new   = alpha * l + sum_j p_hat_j
acc_new = alpha * acc + p_hat @ V_tile
O       = acc / l
LSE     = m + log(l)
```

## 5.4 GQA schedule

输入保持 `(B, T, H, D)` layout。SDPA grid 是：

```text
grid = (B * M_tiles, H_kv)
```

一个 program 负责一个 `(batch, row-tile, kv-head)`。GQA 场景下，一个 K/V head 对应的一组 query heads 会在同一个 program 里处理。flatten 后 Q tile 有 `GQA_GROUP * BLOCK_M` 个逻辑 row，但展开顺序是 row-major：

```text
row_in_tile = offs_hm // GQA_GROUP
head_off    = offs_hm - row_in_tile * GQA_GROUP
offs_m      = pid_m * BLOCK_M + row_in_tile
hid         = kv_hid * GQA_GROUP + head_off
```

这样同一个 K/V tile 可以喂给整组 query head，而不是每个 query head 重新 load 一遍 K/V。非 GQA 就是 `GQA_GROUP = 1` 的同一条路径。

## 5.5 Forward pipeline

forward 有两个逻辑阶段：

| 阶段 | Kernel/op | 输出 |
|---|---|---|
| SDPA | `_fused_attn_spda_fwd_kernel` via `_fused_attn_spda_fwd_op` | `attn_out`, `lse` |
| Output projection | `_fused_attn_spda_and_output_proj_fwd_kernel` | `y = residual + attn_out @ proj_weight.T` |

projection kernel 把 `attn_out` flatten 成 `(B*T, H_q*D)`，把 `residual` flatten 成 `(B*T, C)`。`proj_weight` 可以是 fp32 master weight；load 之后先 cast 到 activation dtype 再做 `tl.dot`。

## 5.6 Backward pipeline

combined backward 围绕 FlashAttention 的 `delta` 项组织：

```text
delta_i = sum_d O_i,d * dO_i,d
```

output projection backward 先同时物化 `d_attn_out` 和 `delta`：

```text
d_attn_out[m, h, d] = sum_o dy[m, o] * proj_weight[o, h, d]
delta[m, h]         = sum_d attn_out[m, h, d] * d_attn_out[m, h, d]
d_proj_weight[o, i] = sum_m dy[m, o] * attn_out[m, i]
d_residual          = dy
```

然后 SDPA backward 消费外部传入的 `delta`：

| 阶段 | Kernel | ownership |
|---|---|---|
| dQ | `_fused_attn_spda_dq_bwd_kernel` | Q-row tile owned，直接写 `dq` |
| dK/dV | `_fused_attn_spda_dkv_bwd_kernel` | K/V tile owned，直接写 `dk`/`dv` |

对每个重算的 score tile：

```text
P_ij  = exp(S_ij - LSE_i)
dP_ij = sum_d dO_i,d * V_j,d
dS_ij = P_ij * (dP_ij - delta_i) / sqrt(D)
dQ_i += sum_j dS_ij * K_j
dV_j += sum_i P_ij * dO_i
dK_j += sum_i dS_ij * Q_i
```

因为 dQ 和 dK/dV 都由自己的输出 tile owner 写回，SDPA backward 不需要对 `dk`/`dv` 做 atomic。

## 5.7 边界和 mask 处理

sliding-window attention 会跳过完全不可见的 K/V tile。stream K/V loop 里区分：

- left boundary tile：需要 elementwise causal/window mask；
- full-valid tile：Q tile 内每一行都能看到这个 K tile 的所有 key；
- right boundary tile：再次需要 elementwise mask。

full-context causal attention 用 constexpr fast path 去掉 sliding-window lower-bound check。这仍然比工业 FA-3/TMA 简化：没有 varlen path、没有 persistent scheduler、没有 async TMA pipeline。

## 5.8 Saved state

autograd wrapper 保存：

| 保存项 | 用途 |
|---|---|
| `q`, `k`, `v` | backward 重算 SDPA score |
| `attn_out` | output projection backward 和 `delta` |
| `lse` | softmax backward normalization |
| `proj_weight` | output projection backward |

不保存 attention score `S`、probability `P`、`d_attn_out`。`d_attn_out` 在 backward 中物化，因为 SDPA backward 需要它作为 `dO`。

## 5.9 性能取舍

这里最重要的 fusion 不是 forward projection 本身，而是 backward hand-off。单独 output projection backward 会算 `d_attn_out`；单独 SDPA backward 还要另起一个 pass 算 `delta = sum(attn_out * d_attn_out)`。combined op 在 `_fused_attn_spda_and_output_proj_dattn_delta_bwd_kernel` 里把两者一起算掉，省一次对 `(B*T, H_q, D)` 的读写。

当前 schedule 明确针对 d24：`D=128`、contiguous `(B, T, H, D)` tensor、GQA-aware K/V reuse、bf16 tensor-core matmul + fp32 accumulator。
