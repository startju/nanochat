# 第 4 章 —— `fused_attn_qkv_projection`：attention input fusion

> 属于 [nanoops Triton Kernels](TRITON_zh.md)。English version: [TRITON_QKV.md](TRITON_QKV.md)。

本文档对应 `nanoops/triton_fused_attn_qkv.py`，也就是 nanochat 集成路径里的 attention 输入侧 fused op。

---

## 4.1 范围

`fused_attn_qkv_projection` 替换 nanochat block 前半段：

```python
x_mix = resid_scale * x + x0_scale * x0
x_hat = rmsnorm_no_weight(x_mix)
q0 = x_hat @ q_weight.T
k0 = x_hat @ k_weight.T
v  = x_hat @ v_weight.T
q, k = apply_rotary(q0, k0, cos, sin)
q, k = rmsnorm_no_weight(q) * scale, rmsnorm_no_weight(k) * scale
```

如果打开 value embedding，V 还会多一段：

```python
x_g  = x_hat[:, :ve_gate_channels]              # (M, ch)
gate = 3 * sigmoid(x_g @ ve_gate_weight.T)      # (M, n_kv_head)
ve   = ve_weight[ve_ids].view(M, n_kv_head, D)
v   += gate[..., None] * ve
```

Python 对外保持 `(B, T, *)` 形状；Triton 内部把 token 维 flatten 成 `M = B*T`，activation 看成 `(M, K)`，head tensor 看成 `(M, H, D)`。

## 4.2 公开 API 和张量约定

```python
def fused_attn_qkv_projection(
    x: torch.Tensor,                 # (B, T, K), contiguous CUDA activation
    x0: torch.Tensor,                # (B, T, K), contiguous CUDA initial stream
    resid_scale: torch.Tensor,       # scalar CUDA tensor
    x0_scale: torch.Tensor,          # scalar CUDA tensor
    ve_ids: torch.Tensor | None,     # optional (B, T) int token ids
    ve_weight: torch.Tensor | None,  # optional (vocab, n_kv_head * D)
    ve_gate_channels: int,
    ve_gate_weight: torch.Tensor | None,  # optional (n_kv_head, ve_gate_channels)
    q_weight: torch.Tensor,          # (n_head * D, K)
    k_weight: torch.Tensor,          # (n_kv_head * D, K)
    v_weight: torch.Tensor,          # (n_kv_head * D, K)
    cos: torch.Tensor,               # (1, T, 1, D/2)
    sin: torch.Tensor,               # (1, T, 1, D/2)
    n_head: int,
    n_kv_head: int,
    head_dim: int,
    scale: float = 1.2,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

返回值：

| Tensor | 形状 | 含义 |
|---|---|---|
| `q` | `(B, T, n_head, head_dim)` | projection + rotary + QK RMSNorm + scale 后的 Q |
| `k` | `(B, T, n_kv_head, head_dim)` | projection + rotary + QK RMSNorm + scale 后的 K |
| `v` | `(B, T, n_kv_head, head_dim)` | projection 后的 V，可选加 value embedding |
| `x_mix` | `(B, T, K)` | residual mix 后的 stream，后续 attention residual 要用 |

nanochat 主路径是无 affine RMSNorm，所以这个 op 没有 `norm_weight` 参数。

## 4.3 Forward pipeline

forward 两次 Triton launch：

| 阶段 | Kernel | Grid | 输出 |
|---|---|---|---|
| Fwd 0 | `_fused_attn_qkv_projection_norm_fwd_kernel` | 1D over `M` | `x_mix`, `x_hat`, `rms_inv` |
| Fwd 1 | `_fused_attn_qkv_projection_qkv_fwd_kernel` | `(M tile, n_head + 2*n_kv_head)` | `q`, `k`, `v`, `qk_rms_inv` |

第一个 kernel 做 residual/x0 blend 和无 affine RMSNorm，并物化 `x_hat`。原因是 Q/K/V 所有 projection head 共享同一个 normalized activation；如果塞进每个 head 里重算，会重复整行 RMSNorm reduction。

QKV kernel 的 grid 第二维用 `part` 表示一个完整 projected head：

```text
part in [0, n_head)                            -> Q head
part in [n_head, n_head+n_kv_head)             -> K head
part in [n_head+n_kv_head, n_head+2*n_kv_head) -> V head
```

Q/K 在 projection 后继续在 register 里做 rotary、inner QK RMSNorm 和 scale，然后写回。V 不做 rotary/RMSNorm，直接 early return；如果有 value embedding，就在写回前加 gate。projection weight 可以是 fp32 master tensor，kernel load weight tile 后 inline cast 到 activation dtype 再 `tl.dot`，避免单独物化 bf16 weight copy。

## 4.4 Saved state

Triton op 保存输入以及 backward 必需的最少 forward 产物：

| 保存项 | 用途 |
|---|---|
| `x`, `x0`, `resid_scale`, `x0_scale` | backward 里重物化 `x_mix`/`x_hat` |
| optional `ve_ids`, `ve_weight`, `ve_gate_weight` | value-embedding 梯度 |
| `q_weight`, `k_weight`, `v_weight` | projection backward |
| `cos`, `sin` | inverse rotary |
| `rms_inv` | outer RMSNorm backward |
| `q`, `k`, `qk_rms_inv` | 不重算 Q/K projection 的前提下做 Q/K RMSNorm backward |

不保存：`x_hat`、`q0`、`k0`、raw rotary output、VE 前的 V projection output、拼起来的 `dz` buffer。

## 4.5 Backward pipeline

backward 走 materialized-pregrad 路径：

| Phase | Kernel | 作用 |
|---|---|---|
| 1 | `_fused_attn_qkv_projection_x_norm_bwd_kernel` | 从保存的 `rms_inv` 重物化 `x_norm = RMSNorm(x_mix)` |
| 2 | `_fused_attn_qkv_projection_qk_pre_ve_bwd_kernel` | 还原 `d_q0`/`d_k0`；可选计算 VE table/gate 梯度 |
| 3 | `_fused_attn_qkv_projection_dx_hat_bwd_kernel` | 计算 `d_x_hat = d_q0@W_q + d_k0@W_k + d_v@W_v`，同时累计 RMS inner |
| 4 | `_fused_attn_qkv_projection_outer_rms_dx_bwd_kernel` | 完成 outer RMSNorm backward 和 residual/x0 scale 梯度 |
| 5 | `_fused_attn_qkv_projection_weight_section_bwd_kernel` | 用物化 pregrad 计算 `dW_q`, `dW_k`, `dW_v` |

Q/K RMSNorm backward 用保存的最终 `q`/`k` 和 scale 反推：

```text
y0 = q_or_k / scale
g0 = grad_q_or_k * scale
dr = s_qk * (g0 - y0 * mean_d(g0 * y0))
```

然后 inverse rotary 把 `dr` 变回 `d_q0` 或 `d_k0`。V 没有 QK RMSNorm/rotary，所以 `d_v` 直接进入 projection backward 和可选 VE backward。

outer RMSNorm 和 residual mix 在拿到完整 `d_x_hat` 后完成：

```text
d_x_mix = s_x * (d_x_hat - x_norm * mean_k(d_x_hat * x_norm))
d_x_mix += grad_x_mix
d_x        = d_x_mix * resid_scale
d_x0       = d_x_mix * x0_scale
d_resid_s  = sum(d_x_mix * x)
d_x0_s     = sum(d_x_mix * x0)
```

## 4.6 性能取舍

这里没有硬塞成一个超大 kernel。行 RMSNorm reduction 被所有 head 共享，所以先物化一次；projection 是主要算力部分，按 head 切；Q/K 可以把 rotary 和 inner RMSNorm 留在 register 里，V 则跳过这些分支。backward 同理：把昂贵的 pregrad 和 `d_x_hat` 放在合适位置物化，避免重复 projection reduction。

当前 d24 参数对 VE/no-VE 分别锁配置，因为 gate 路径会改变 register pressure 和最佳 `BLOCK_M/BLOCK_K`。op 通过 `torch.library.triton_op` + `wrap_triton` 包装，`torch.compile` 看到的是稳定 opaque Triton node，不会被 Python autograd wrapper graph-break。

## 4.7 Kernel 布局总览

第 3 章的标准是：少量比较胖的 kernel，每个 kernel 负责一个清晰的代数边界，
并把紧贴 matmul 的 elementwise/reduction 融进去。QKV 也是同一个规则。
它没有写成一个超级 kernel，因为这里有三个天然 owner 不同的 reduction：

1. outer RMSNorm 是 row-owned，而且被所有 Q/K/V head 共享；
2. Q/K rotary 后 RMSNorm 是 head-row-owned；
3. projection weight grad 是 weight-tile-owned。

当前路径包含：

| 顺序 | Kernel | Owner | 主要输出 | 为什么在这里切开 |
| ---- | ------ | ----- | -------- | ---------------- |
| F0 | `_fused_attn_qkv_projection_norm_fwd_kernel` | `(M tile)` | `x_mix`, `x_hat`, `rms_inv` | 一次 row RMS reduction 被所有 Q/K/V head 复用。 |
| F1 | `_fused_attn_qkv_projection_qkv_fwd_kernel` | `(M tile, part)`，`part` 是 Q/K/V head | `q`, `k`, `v`, `qk_rms_inv` | projection 是主要算力，天然按 head 切；Q/K 把 rotary+RMSNorm 留在寄存器里。 |
| B1 | `_fused_attn_qkv_projection_x_norm_bwd_kernel` | `(M tile, K tile)` | `x_norm` | 用保存的 `rms_inv` 重建 normalized input。 |
| B2 | `_fused_attn_qkv_projection_qk_pre_ve_bwd_kernel` | `(M tile, Q/K/V part)` | `d_q_pre`, `d_k_pre`, 可选 VE grads | 在保存的 Q/K 附近反推 scale+RMSNorm+rotary；V part 负责 VE gate/table 梯度。 |
| B3 | `_fused_attn_qkv_projection_dx_hat_bwd_kernel` | `(M tile, K tile)` | `d_x_hat`, `outer_rms_row_inner` | 汇总 Q/K/V projection backward 对 activation 的贡献。 |
| B4 | `_fused_attn_qkv_projection_outer_rms_dx_bwd_kernel` | `(M tile, K tile)` | `dx`, `dx0`, scale grads | 完成 outer RMSNorm 和 residual/x0 mix backward。 |
| B5 | `_fused_attn_qkv_projection_weight_section_bwd_kernel` | `(section, output-row tile, K tile)` | `dW_q`, `dW_k`, `dW_v` | weight grad 对 M 归约，所以需要 weight-tile ownership。 |

这和第 3 章一样：不要为了减少 launch 数量跨过 incompatible reduction
axis。多几个 launch 换来更干净的 matmul 形状，也避免大 activation/weight grad
上使用 atomic。

## 4.8 Forward 细节

### 4.8.1 Step F0 - residual mix + outer RMSNorm

第一个 forward kernel 计算：

```text
x_mix[m, k] = resid_scale * x[m, k] + x0_scale * x0[m, k]
sum_sq[m]   = sum_k x_mix[m, k]^2
s_x[m]      = rsqrt(sum_sq[m] / K + eps)
x_hat[m, k] = x_mix[m, k] * s_x[m]
```

输出：

- `x_mix`：返回给 attention residual path。这是真正的 public output。
- `x_hat`：物化下来，因为每个 Q/K/V head 都会读取同一个 normalized row。
- `rms_inv`：fp32 `(M,)`，保存给 backward。

为什么要物化 `x_hat`？d24 下 `K=1536`，QKV 一共有
`n_head + 2*n_kv_head = 36` 个 projected head part。如果每个 head program
都重新做一次 row RMSNorm，就会重复几十次 1536-wide reduction。写/读一次
`x_hat` 比重复 reduction 更划算。

### 4.8.2 Step F1 - 每个 projected head part 一个 program

`_fused_attn_qkv_projection_qkv_fwd_kernel` 用 grid axis 1 表示 logical part：

```text
part < n_head                         -> Q head
n_head <= part < n_head + n_kv_head   -> K head
otherwise                             -> V head
```

每个 program 计算一个完整 head tile：

```text
acc[m, d] = sum_k x_hat[m, k] * W_part[head*D + d, k]
```

权重在 optimizer 路径里可能是 fp32 master weight。kernel 加载 weight tile 后
cast 成 activation dtype 再进入 `tl.dot`，这样保持 bf16 tensor-core path，
不需要单独起一个“整块 weight cast” kernel。

### 4.8.3 Q/K rotary + inner RMSNorm

Q/K part 会立刻处理 projection output：

```text
lo, hi = split_half(z)        # z 是 q0 或 k0，D 必须是偶数
r_lo   = lo * cos + hi * sin
r_hi   = -lo * sin + hi * cos
r      = concat(r_lo, r_hi)
s_qk   = rsqrt(mean_d(r^2) + eps)
out    = scale * r * s_qk
```

这段留在同一个 program 里，避免存 raw `q0`/`k0`、rotary output 或 Q/K
pre-scale output。backward 只额外保存 `qk_rms_inv[m, head]`。

这里使用 nanochat 的 split-half rotary 约定。public `cos/sin` 表是
`(1, T, 1, D/2)`，batch/head 维靠广播；进入 fused op 时会验证当前 `T`。

### 4.8.4 V 和可选 value embedding

V 不做 rotary，也不做 QK RMSNorm。它跳过 Q/K 分支，然后可选加 value
embedding：

```text
x_g       = x_hat[:, :ve_gate_channels]
logits    = x_g @ ve_gate_weight.T
gate      = 3 * sigmoid(logits)
ve        = ve_weight[ve_ids].view(M, n_kv_head, D)
v        += gate[..., None] * ve
```

`gate` 是 token/head 级别的权重。`ve_weight[token]` 存的是所有 K/V head
打平后的 `(n_kv_head * D)`；Triton 里直接索引当前 head slice，不需要真的
创建 view。

VE 和 no-VE 使用不同 compile-time tile 常量。gate/table 路径会带来额外的
`x_g`、`gate_w`、sigmoid、table load 和 backward state，和 no-VE 共用同一个
tile shape 在 d24 sweep 中更慢。

## 4.9 Backward 细节

### 4.9.1 Step B1 - 重物化 `x_norm`

Backward 不保存 `x_mix` 或 `x_hat`。它保存原始 `x`、`x0`、scale 和
`rms_inv`，然后重建：

```text
x_mix  = resid_scale * x + x0_scale * x0
x_norm = x_mix * rms_inv[:, None]
```

`x_norm` 会物化，因为 weight grad 和 VE gate grad 都需要 normalized input。
这和第 3 章一样：只保存能明显减少下游 reduction/matmul 成本的东西。

### 4.9.2 Step B2 - Q/K pregrad 和 VE backward

保存的最终 Q/K 已经包含 scale：

```text
q = scale * y0_q
k = scale * y0_k
```

所以 Q/K RMSNorm backward 使用：

```text
y0 = q_or_k / scale
g0 = d_q_or_d_k * scale
dr = s_qk * (g0 - y0 * mean_d(g0 * y0))
```

再用 inverse rotary 回到 projection-output grad：

```text
d_lo = d_r_lo * cos - d_r_hi * sin
d_hi = d_r_lo * sin + d_r_hi * cos
d_q_pre or d_k_pre = concat(d_lo, d_hi)
```

V 没有 QK norm/rotary。no-VE 路径下它就是 `d_v`；VE 路径下，V-owned
program 还会计算：

```text
d_gate         = sum_d(d_v * ve)
d_logits       = 3 * d_gate * sigmoid * (1 - sigmoid)
d_x_g         += d_logits @ ve_gate_weight
d_ve_weight   += d_v * gate
d_gate_weight += d_logits.T @ x_g
```

`d_ve_weight` 必须用 atomic，因为同一个 batch 里多个 token 可能有相同
`ve_id`。`d_ve_gate_weight` 也需要跨 row tile 累加，但形状很小：
`(n_kv_head, ve_gate_channels)`。

### 4.9.3 Step B3 - `d_x_hat`

projection input grad 概念上是：

```text
d_x_hat = d_q_pre @ q_weight + d_k_pre @ k_weight + d_v @ v_weight
```

kernel 循环 Q、K、V head section，把结果累到一个 `(BLOCK_M, BLOCK_K)` tile。
它把 `d_x_hat` 按 activation dtype 物化，并顺手累计 outer RMSNorm 需要的
row inner：

```text
inner[m] += sum_k d_x_hat[m, k] * x_norm[m, k]
```

这个 side output 和第 3 章 MLP dx kernel 的思路一样：既然 tile 里已经同时有
两个 operand，就把 row inner reduction 一起做掉。

当前 kernel 支持 `HEAD_SPLIT`，d24 下已调参固定。拆 head 对 forward 没有太大
收益，但对 backward `d_x_hat` 这种重复读权重的大 matmul 更有用。

### 4.9.4 Step B4 - outer RMSNorm 和 residual mix

outer RMSNorm backward 使用物化的 `d_x_hat` 和 B3 算好的 row inner：

```text
d_x_mix_norm = rms_inv * (d_x_hat - x_norm * mean_k(d_x_hat * x_norm))
```

返回的 `x_mix` 还会从 attention residual tail 收到直接梯度。最后一个 kernel
先把它加进去，再拆 residual mix：

```text
d_x_mix = d_x_mix_norm + grad_x_mix
dx      = d_x_mix * resid_scale
dx0     = d_x_mix * x0_scale
d_resid_scale = sum(d_x_mix * x)
d_x0_scale    = sum(d_x_mix * x0)
```

scale gradient 是 scalar atomic，相对 Q/K/V matmul 很小。

### 4.9.5 Step B5 - projection weight gradients

weight grad 对 row 做归约：

```text
dW_q = d_q_pre.T @ x_norm
dW_k = d_k_pre.T @ x_norm
dW_v = d_v.T     @ x_norm
```

`_fused_attn_qkv_projection_weight_section_bwd_kernel` 使用 `section` grid axis：

```text
section = 0 -> Q weight
section = 1 -> K weight
section = 2 -> V weight
```

这样三个 weight-gradient matmul 共用同一个 kernel definition，但输出仍然是三块
独立权重。输出 tensor dtype 跟 weight dtype 一致，所以 fp32 master weight
路径下梯度直接落到 fp32 tensor。

## 4.10 保存 / 重算账本

| Tensor | 保存吗 | 原因 |
| ------ | ------ | ---- |
| `x`, `x0`, residual scales | 保存 | 重建 `x_mix`，并计算 residual-scale grad。 |
| `rms_inv` | 保存 | 避免 backward 再做一次 outer RMS reduction。 |
| `x_mix` | public output | 返回给 attention residual path；backward 仍重建 normalized input。 |
| `x_hat` / `x_norm` | 不保存 | B1 重建一次；保存会多占 `B*T*K` activation memory。 |
| `q`, `k` | 保存 | Q/K RMSNorm backward 不用重算 Q/K projection。 |
| `v` | public output | grad 直接走；VE 前 raw V 不保存。 |
| `qk_rms_inv` | 保存 | Q/K RMSNorm backward 必需。 |
| `q0`, `k0`, rotary intermediates | 不保存 | 从最终 Q/K 通过 inverse scale/RMSNorm/rotary 反推。 |
| `d_z` concat buffer | 不保存 | Q/K pregrad 和 V grad 被定向 kernel 消费。 |
| `d_x_hat` | backward 临时 | 只在 B3-B4 之间物化，简化 outer RMS path。 |

这是“全保存”和“全重算”之间的折中：Q/K projection 足够贵，所以保存最终 Q/K；
`x_hat` 足够大，所以只重算一次。

## 4.11 数值精度路径

d24 训练路径是 bf16 activation + fp32 master weight。QKV kernel 策略是：

```text
load activation -> activation dtype
load weight     -> cast to activation dtype before tl.dot
tl.dot          -> fp32 accumulator
elementwise RMS -> fp32 reductions
store Q/K/V     -> activation dtype
save rms_inv    -> fp32
```

为什么不是所有中间量都保持 fp32？因为 projection matmul 主要吃 tensor core。
加载 weight tile 后 cast 到 bf16 能走和 native compiled GEMM 一样的快路径。
RMS reduction 仍然用 fp32 保数值稳定，最终 normalized vector 再 cast 回
activation dtype 存储。

backward 同理：reduction 和 row inner 用 fp32，大的 activation-gradient buffer
用 activation dtype。这样才能在 24 GiB 上撑住 d24 B=2，同时保留关键 reduction
的数值稳定性。

## 4.12 预期收益账本

### Forward

| Native work | Fused result |
| ----------- | ------------ |
| residual mix kernel | 折进 F0 |
| RMSNorm kernel | 折进 F0 |
| 三个 projection call 和 reshape | 一个按 head 切的 Triton projection kernel |
| rotary Q/K kernel | 折进 F1 的 Q/K part |
| Q/K RMSNorm kernels | 折进 F1 的 Q/K part |
| Q/K scale kernels | 折进 Q/K store |
| 可选 VE gate/table add | 折进 F1 的 V part |

主要收益不只是 launch 数量，而是少存 `q0`、`k0`、rotary output、Q/K
pre-scale output 和 VE add 等大中间张量。

### Backward

| Native work | Fused result |
| ----------- | ------------ |
| 独立 Q/K norm backward | B2，贴着 inverse rotary |
| 独立 rotary backward | B2 |
| 可选 VE backward PyTorch ops | V-owned B2 parts |
| 三个独立 matmul 生成 `d_x_hat` | B3 汇总所有来源 |
| outer RMSNorm backward + residual mix backward | B4 |
| 三个 weight-gradient call site | 一个 sectioned B5 kernel |

剩下的大头仍然是 GEMM-like 的 `d_x_hat` 和 `dW`。所以实现里单独调它们的 tile
shape，而不是追求一个全能 fused kernel。

## 4.13 性能现实

当前 d24 compile path 最后固定在：

- forward no-VE：更大的 `BLOCK_M/BLOCK_K`，因为每行标量逻辑少；
- forward VE：更小的 `BLOCK_K`，因为 gate/table 逻辑增加 register pressure；
- backward `d_x_hat`：`BLOCK_M=128`、`BLOCK_K=64`、head splitting；
- weight gradients：sectioned kernel，row/output/K tile 都按 d24 调过。

调参中最重要的经验是：microbench 小赢不一定能带到 compiled training。完整图会改变
live range 和 peak memory。比如物化 `d_x_hat` 对 QKV backward 有帮助，最终胜过
重算路径；但把 `x_hat` 保存给 backward 就不划算。

## 4.14 End-to-end 落地

public op 是注册了 autograd 的 `torch.library.triton_op`：

```text
fused_attn_qkv_projection
  -> nanoops::fused_attn_qkv_projection_fwd
  -> nanoops::fused_attn_qkv_projection_bwd
```

`nanoops.integration` 在 `NANOOPS_FUSED_ATTN=1` 时把它接进 `GPT.forward`。
public tensor 维持 `(B, T, *)`，kernel 内部 flatten 成 `B*T -> M`。这样模型代码
仍然可读，Triton kernel 则吃连续的 `(M, K)` / `(M, H, D)` 布局。

当前限制：

- 主 d24 路径没有 affine RMSNorm weight；
- rotary table 必须是 4D broadcast table `(1, T, 1, D/2)`；
- `head_dim` 必须是偶数；
- VE gate channel 数需要 fit 住调好的 backward `DX_HAT_BLOCK_K`；
- 调参目标是 d24 形状，不是通用 autotuned library。

## 4.15 Takeaway

`fused_attn_qkv_projection` 是第 3 章 MLP block 在 attention prefix 上的对应物：
把能贴着 matmul 白捡的 elementwise/reduction 融进去，但当 owner 改变时就停下。
最终不是一个巨型 kernel，而是一个短 pipeline：边界分别对应 RMS row reduction、
head-local Q/K transform、activation-gradient matmul 和 weight-gradient reduction。
