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
