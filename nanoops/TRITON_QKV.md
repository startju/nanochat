# Chapter 4 — `fused_attn_qkv_projection`: attention input fusion

> Part of [nanoops Triton Kernels](TRITON.md). Chinese version: [TRITON_QKV_zh.md](TRITON_QKV_zh.md).

This chapter documents `nanoops/triton_fused_attn_qkv.py`, the attention input-side fused op used by the nanochat integration path.

---

## 4.1 Scope

`fused_attn_qkv_projection` replaces this nanochat block prefix:

```python
x_mix = resid_scale * x + x0_scale * x0
x_hat = rmsnorm_no_weight(x_mix)
q0 = x_hat @ q_weight.T
k0 = x_hat @ k_weight.T
v  = x_hat @ v_weight.T
q, k = apply_rotary(q0, k0, cos, sin)
q, k = rmsnorm_no_weight(q) * scale, rmsnorm_no_weight(k) * scale
```

If value embedding is enabled, V also gets:

```python
x_g  = x_hat[:, :ve_gate_channels]              # (M, ch)
gate = 3 * sigmoid(x_g @ ve_gate_weight.T)      # (M, n_kv_head)
ve   = ve_weight[ve_ids].view(M, n_kv_head, D)
v   += gate[..., None] * ve
```

The public API keeps `(B, T, *)` shapes. Triton kernels flatten the token axis to `M = B*T` and operate on `(M, K)` activations and `(M, H, D)` head tensors.

## 4.2 Public API and Tensor Contract

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

Returns:

| Tensor | Shape | Meaning |
|---|---|---|
| `q` | `(B, T, n_head, head_dim)` | final Q after projection, rotary, QK RMSNorm, scale |
| `k` | `(B, T, n_kv_head, head_dim)` | final K after projection, rotary, QK RMSNorm, scale |
| `v` | `(B, T, n_kv_head, head_dim)` | projected V, plus optional value embedding |
| `x_mix` | `(B, T, K)` | residual-mixed stream consumed by the attention residual path |

The main nanochat path is no-affine RMSNorm; there is no `norm_weight` argument in this op.

## 4.3 Forward Pipeline

Forward is two Triton launches:

| Stage | Kernel | Grid | Output |
|---|---|---|---|
| Fwd 0 | `_fused_attn_qkv_projection_norm_fwd_kernel` | 1D over `M` | `x_mix`, `x_hat`, `rms_inv` |
| Fwd 1 | `_fused_attn_qkv_projection_qkv_fwd_kernel` | `(M tile, n_head + 2*n_kv_head)` | `q`, `k`, `v`, `qk_rms_inv` |

The first kernel computes the residual/x0 blend and no-affine RMSNorm once. It materializes `x_hat` because all Q/K/V projection heads reuse the same normalized activation; recomputing it per head would repeat the row reduction.

The QKV kernel assigns one full projected head to each program on grid axis 1:

```text
part in [0, n_head)                         -> Q head
part in [n_head, n_head+n_kv_head)          -> K head
part in [n_head+n_kv_head, n_head+2*n_kv_head) -> V head
```

Q and K run projection, rotary, inner QK RMSNorm, and final scale before store. V exits before rotary/RMSNorm and optionally applies the value-embedding gate. Projection weights may be fp32 master tensors; each kernel loads weight tiles and casts to activation dtype before `tl.dot`, avoiding a standalone fp32-to-bf16 weight copy.

## 4.4 Saved State

The Triton op saves inputs plus the minimal forward products needed by backward:

| Saved | Why |
|---|---|
| `x`, `x0`, `resid_scale`, `x0_scale` | rematerialize `x_mix`/`x_hat` for backward |
| optional `ve_ids`, `ve_weight`, `ve_gate_weight` | value-embedding gradients |
| `q_weight`, `k_weight`, `v_weight` | projection backward |
| `cos`, `sin` | inverse rotary |
| `rms_inv` | outer RMSNorm backward |
| `q`, `k`, `qk_rms_inv` | Q/K RMSNorm backward without recomputing Q/K projection outputs |

Not saved: `x_hat`, `q0`, `k0`, raw rotary outputs, V projection output before VE, or a concatenated `dz` buffer.

## 4.5 Backward Pipeline

Backward is the materialized-pregrad path:

| Phase | Kernel | Role |
|---|---|---|
| 1 | `_fused_attn_qkv_projection_x_norm_bwd_kernel` | rematerialize `x_norm = RMSNorm(x_mix)` from saved `rms_inv` |
| 2 | `_fused_attn_qkv_projection_qk_pre_ve_bwd_kernel` | recover `d_q0`/`d_k0`; compute optional VE table/gate grads |
| 3 | `_fused_attn_qkv_projection_dx_hat_bwd_kernel` | compute `d_x_hat = d_q0@W_q + d_k0@W_k + d_v@W_v`; accumulate RMS inner |
| 4 | `_fused_attn_qkv_projection_outer_rms_dx_bwd_kernel` | finish outer RMSNorm backward and residual/x0 scale gradients |
| 5 | `_fused_attn_qkv_projection_weight_section_bwd_kernel` | compute `dW_q`, `dW_k`, `dW_v` from materialized pregrads |

Q/K RMSNorm backward uses saved final `q`/`k` and inverse scale:

```text
y0 = q_or_k / scale
g0 = grad_q_or_k * scale
dr = s_qk * (g0 - y0 * mean_d(g0 * y0))
```

Then inverse rotary maps `dr` back to `d_q0` or `d_k0`. V has no QK RMSNorm or rotary, so `d_v` directly participates in projection backward and optional VE backward.

Outer RMSNorm and residual mix are finished after the projection gradients have produced the full `d_x_hat`:

```text
d_x_mix = s_x * (d_x_hat - x_norm * mean_k(d_x_hat * x_norm))
d_x_mix += grad_x_mix
d_x        = d_x_mix * resid_scale
d_x0       = d_x_mix * x0_scale
d_resid_s  = sum(d_x_mix * x)
d_x0_s     = sum(d_x_mix * x0)
```

## 4.6 Performance Notes

The split is intentionally not a single giant kernel. The row RMSNorm reduction is shared by all heads, so it is materialized once. The projection work is compute-heavy and head-partitioned; Q/K can keep rotary and inner RMSNorm in registers, while V can skip those branches. Backward similarly materializes the expensive pregrads and `d_x_hat` where that avoids repeated projection reductions.

Current d24 tuning keeps separate VE and no-VE constants because the gate path changes register pressure and the best `BLOCK_M/BLOCK_K` pair. The op is wrapped with `torch.library.triton_op` plus `wrap_triton`, so `torch.compile` sees stable opaque Triton nodes instead of graph-breaking through Python autograd wrappers.
