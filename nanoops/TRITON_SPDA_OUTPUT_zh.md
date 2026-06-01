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
dS_ij = P_ij * (dP_ij - delta_i)
dQ_i += (1/sqrt(D)) * sum_j dS_ij * K_j
dV_j += sum_i P_ij * dO_i
dK_j += (1/sqrt(D)) * sum_i dS_ij * Q_i
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

## 5.10 Kernel 布局总览

第 5 章横跨两个文件，因为 public op 覆盖两个不同领域：

- `triton_fused_attn_spda.py`：Flash-style SDPA forward/backward；
- `triton_fused_attn_spda_and_output.py`：output projection、residual add，以及
  backward 里产生 `delta` 的桥接 kernel。

当前路径包含：

| 顺序 | Kernel/op | Owner | 主要输出 | 为什么在这里切开 |
| ---- | --------- | ----- | -------- | ---------------- |
| F0 | `_fused_attn_spda_fwd_kernel` | `(batch, Q-row tile, K/V head)` | `attn_out`, `lse` | stream K/V tile，online softmax，不存 score/probability。 |
| F1 | `_fused_attn_spda_and_output_proj_fwd_kernel` | `(M tile, C tile)` | `y = residual + attn_out @ W_o.T` | output projection 是标准 GEMM + residual add。 |
| B1 | `_fused_attn_spda_and_output_proj_dattn_delta_bwd_kernel` | `(M tile, Q head)` | `d_attn_out`, `delta` | 生成 `dO` 时顺手算 FlashAttention 的 `delta`。 |
| B2 | `_fused_attn_spda_and_output_proj_dweight_bwd_kernel` | `(C tile, D_in tile)` | `d_proj_weight` | weight grad 对 M 归约，需要 weight-tile ownership。 |
| B3 | `_fused_attn_spda_dq_bwd_kernel` | `(batch, Q-row tile, K/V head)` | `dQ` | Q-owned pass，每个 dQ 元素只写一次。 |
| B4 | `_fused_attn_spda_dkv_bwd_kernel` | `(batch, K/V tile, K/V head)` | `dK`, `dV` | K/V-owned pass，每个 dK/dV 元素只写一次，避免 atomic。 |

这和第 3/4 章同一个原则：不要为了减少 launch 数跨过 reduction ownership。
真正有价值的 fusion 是 `d_attn_out + delta` handoff；把 dQ 和 dKV 硬塞在一起
要么重复工作，要么引入 atomic。

## 5.11 SDPA Forward 细节

### 5.11.1 Tile ownership

forward launch grid 是：

```text
grid = (B * M_tiles, H_kv)
```

每个 program 固定：

```text
bid    = pid_bm // M_TILES
pid_m  = pid_bm - bid * M_TILES
kv_hid = program_id(1)
```

GQA 下，一个 K/V head 服务 `GQA_GROUP = H_q // H_kv` 个 query head。
program 读取逻辑形状 `(BLOCK_M, GQA_GROUP, D)` 的 Q tile，只是在 `tl.dot`
前 flatten：

```text
offs_hm     = arange(0, GQA_GROUP * BLOCK_M)
row_in_tile = offs_hm // GQA_GROUP
head_off    = offs_hm - row_in_tile * GQA_GROUP
offs_m      = pid_m * BLOCK_M + row_in_tile
hid         = kv_hid * GQA_GROUP + head_off
```

这个 row-major 展开保持 row locality，同时让一个 K/V tile 供给整个 query-head
group。非 GQA 就是 `GQA_GROUP = 1` 的同一路径。

### 5.11.2 Online softmax recurrence

每个 streamed K/V tile 先算：

```text
s = (Q_tile @ K_tile.T) * sm_scale
```

mask 掉的位置置为 `-inf`。每一行维护：

```text
m_i   = running row max
l_i   = shifted basis 下的 running exp-sum
acc_i = P @ V 的 running numerator
```

更新公式：

```text
m_new   = max(m_i, max(s))
alpha   = exp(m_i - m_new)
p       = exp(s - m_new)
l_new   = alpha * l_i + sum(p)
acc_new = alpha * acc_i + p @ V_tile
```

最后：

```text
out = acc / l
lse = m + log(l)
```

`lse` 用 fp32 保存，因为 backward 要靠 `exp(score - lse)` 重建 probability。

### 5.11.3 full tile 和 boundary tile

kernel 把循环分成三段：

1. left boundary：需要 elementwise sliding/causal mask；
2. full-valid middle：Q tile 里每一行都能看到 K tile 里每个 key；
3. right boundary：再次需要 elementwise mask。

`IS_FULL_CONTEXT` 下 sliding-window lower-bound check 会被 constexpr 编译掉。
这还没有工业 FlashAttention 那种 full causal/local/boundary/varlen 多 kernel
特化极致，但已经能避免 middle loop 带最重的 mask 判断。

## 5.12 Output Projection Forward

`attn_out` flatten 成 `(M, D_in)`：

```text
M    = B * T
D_in = H_q * D
C    = residual width
```

projection kernel 计算：

```text
y[m, o] = residual[m, o] + sum_i attn_out[m, i] * proj_weight[o, i]
```

和 QKV kernel 一样，output projection 加载权重后 cast 到 activation dtype 再
进入 `tl.dot`，保持 d24 bf16 tensor-core path，输出按 activation dtype 存。

forward 仍然保存 `attn_out`。这看起来占内存，但 SDPA backward 和
output-projection backward 都需要它：

- `d_proj_weight = dy.T @ attn_out`;
- `delta = sum(attn_out * d_attn_out)`。

为了少保存 `attn_out` 而重算完整 SDPA output，在当前 d24 B=2 下不划算。

## 5.13 Output Projection Backward 和 Delta Handoff

本章最关键的 fusion 是 B1：

```text
d_attn_out[m, i] = sum_o dy[m, o] * proj_weight[o, i]
delta[m, h]      = sum_d attn_out[m, h, d] * d_attn_out[m, h, d]
```

如果 output projection 和 SDPA backward 分开，系统会：

1. 计算 `d_attn_out = dy @ W_o`；
2. 再起一个 kernel 算 `delta = row_dot(attn_out, d_attn_out)`；
3. 再跑 SDPA backward。

现在 B1 在同一个 program 里完成 1 和 2。它已经拥有 `(M tile, head)` 的
`d_attn_out` slice，也有对应 `attn_out` slice，row-dot side output 几乎是白捡。

`d_proj_weight` 保持独立 B2 kernel，因为它的 reduction axis 是 M：

```text
dW_o[o, i] = sum_m dy[m, o] * attn_out[m, i]
```

如果在 B1 里顺手产 `dW_o`，要么对 weight tile 做 atomic，要么物化很大的
partial buffer。单独 weight-owned kernel 对当前 d24 shape 更简单也更快。

## 5.14 SDPA Backward From Delta

backward 按 tile 重算 score：

```text
S_ij = sm_scale * dot(Q_i, K_j)
P_ij = exp(S_ij - LSE_i)
```

给定 `dO = d_attn_out` 和 `delta_i = sum_d O_i,d * dO_i,d`，softmax backward：

```text
dP_ij = dot(dO_i, V_j)
dS_ij = P_ij * (dP_ij - delta_i)
```

scale 乘在 matmul gradient 上：

```text
dQ_i += sm_scale * sum_j dS_ij * K_j
dK_j += sm_scale * sum_i dS_ij * Q_i
dV_j +=            sum_i P_ij  * dO_i
```

实现用两个 kernel：

- `_fused_attn_spda_dq_bwd_kernel`：Q-owned，循环 visible K/V tile，直接写 `dQ`；
- `_fused_attn_spda_dkv_bwd_kernel`：K/V-owned，循环 visible Q tile，直接写 `dK`/`dV`。

这是 FlashAttention-style backward 的结构性拆分。Q-owned backward 里直接
atomic-add dK/dV 更容易写，但 `(B, T, H_kv, D)` 上的 atomic 对训练路径太贵。

## 5.15 Sliding Window 和 Boundary 数学

对 query row `i`，可见 key 是：

```text
max(0, i - window_size + 1) <= j <= i
```

对 Q tile `[q_first, q_last]`，forward kernel 从：

```text
kv_tile_start = max(0, q_first - WINDOW + 1) // BLOCK_N
kv_tile_end   = ceil(min(N, q_first + BLOCK_M) / BLOCK_N)
```

开始枚举 K/V tile。middle full-valid 区间表示 Q tile 内每一行都能看到该 K tile
全部 key；boundary tile 保留 elementwise mask：

```text
mask = (offs_n <= offs_m) & (offs_n >= offs_m - WINDOW + 1)
```

dKV pass 反过来枚举。一个 K/V tile 只需要扫描可能 attend 到它的 Q tile：

```text
q_tile_start = kv_start // BLOCK_M
q_tile_end   = ceil(min(M, kv_start + BLOCK_N + WINDOW - 1) / BLOCK_M)
```

这样避免每个 K/V tile 都扫完整 `T/BLOCK_M` 个 Q tile。

## 5.16 保存 / 重算账本

| Tensor | 保存吗 | 原因 |
| ------ | ------ | ---- |
| `q`, `k`, `v` | 保存 | SDPA backward 重算 score/probability。 |
| `attn_out` | 保存 | `d_proj_weight` 和 `delta` 都需要。 |
| `lse` | 保存 | 稳定重建 `P = exp(S - LSE)`。 |
| `proj_weight` | 保存 | 计算 `d_attn_out`。 |
| scores `S` | 不保存 | 按 tile 重算。 |
| probabilities `P` | 不保存 | 按 tile 重算。 |
| `d_attn_out` | backward 临时 | SDPA backward 需要它作为 `dO`。 |
| `delta` | backward 临时 | B1 物化，SDPA backward 消费。 |

主要内存取舍是保存 `attn_out`。不保存它意味着 output-projection backward 之前要
重算 SDPA output；对当前 d24 sequence length/head count 不划算。

## 5.17 数值精度路径

d24 路径使用 bf16 Q/K/V 和 projection activation：

```text
QK dot          -> fp32 accumulator
score/max/lse   -> fp32
P @ V acc       -> fp32 accumulator
OUT store       -> activation dtype
projection dot  -> fp32 accumulator
projection store-> activation dtype
delta           -> fp32
```

backward 同理。softmax probability 和 `dS` 是 fp32；大的输出 tensor
（`dq`、`dk`、`dv`、`d_attn_out`）按 activation dtype 存。这和 native bf16
训练行为一致，也让显存能压进 24 GiB。

## 5.18 预期收益账本

### Forward

| Native work | Fused result |
| ----------- | ------------ |
| materialized score/probability matrix | 不物化，online softmax stream |
| GQA head 重复加载 K/V | 一个 K/V tile 供给整个 query-head group |
| output projection + residual add | 一个 projection kernel，residual fold-in |

forward 仍然是两个逻辑阶段，因为 SDPA 和 output projection dataflow 不同。
把它们塞进一个 kernel 会让 SDPA program 同时负责 C 维输出 channel，破坏干净的
online-softmax tile shape。

### Backward

| Native work | Fused result |
| ----------- | ------------ |
| `d_attn_out = dy @ W_o` | B1 |
| 独立 `delta = sum(O * dO)` pass | 折进 B1 |
| `dW_o = dy.T @ O` | B2 weight-owned kernel |
| SDPA dQ | B3 Q-owned kernel |
| SDPA dK/dV atomic | B4 K/V-owned kernel，无 atomic |

省掉对 `(B*T, H_q, D)` 的 `delta` pass 是 combined op 最确定的收益。

## 5.19 性能现实和工业差距

当前 kernel 比工业 FlashAttention/FA-3/vLLM kernel 简化很多：

- 没有 varlen packed sequence layout；
- 没有 persistent CTA scheduler；
- 没有 TMA 或 async producer/consumer pipeline；
- checked-in 版本没有 split-K / split-Q reduction path；
- 没有针对 full-causal/local/boundary 生成多套 kernel；
- 没有 FP8 path。

但它具备 d24 需要的骨架：

- contiguous `(B, T, H, D)` tensor；
- GQA-aware K/V reuse；
- full-context vs sliding-window constexpr branch；
- K/V loop 内区分 boundary/full tile；
- 不物化 score/probability；
- dK/dV 无 atomic；
- output-projection delta handoff。

训练实测里，fused SDPA/output path 的价值在于减少中间流量，并让 attention tail
对 `torch.compile` 可见。它不声称所有 shape 都打赢成熟 vendor FlashAttention
kernel；这个章节的目标是用可读代码展示工业路径的基本骨架。

## 5.20 End-to-end 落地

public op：

```text
fused_attn_spda_and_output
  -> nanoops::fused_attn_spda_and_output_fwd
       -> nanoops::fused_attn_spda_fwd
       -> output projection/residual kernel
  -> nanoops::fused_attn_spda_and_output_bwd
       -> d_attn_out + delta
       -> d_proj_weight
       -> nanoops::fused_attn_spda_bwd
```

`nanoops.integration` 会在 `fused_attn_qkv_projection` 之后立刻调用它。
attention path 的 public tensor 保持 `(B, T, H, D)`，只有 output projection GEMM
前 flatten 成 `(B*T, *)`。模型 API 仍然接近 PyTorch，Triton 内部则保持连续密集
访存。

## 5.21 Takeaway

`fused_attn_spda_and_output` 不是“所有东西塞成一个 kernel”。它是一个
Flash-style SDPA 实现，加上一个选得很准的 output projection backward fusion。
这个边界省掉 redundant delta pass，保持 dQ/dKV owner 干净，也给 nanochat 一个
能被 `torch.compile` 看见、又足够小到能读懂的 attention tail。
