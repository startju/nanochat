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

## 4.7 Kernel Layout Overview

Chapter 3 uses a "small number of fat kernels" style: each kernel owns one
clear algebraic boundary and fuses the elementwise/reduction work that sits
next to its matmul. QKV follows the same rule. It is not one monolithic
attention-prefix kernel because three reductions have different natural owners:

1. outer RMSNorm is row-owned and shared by every Q/K/V head;
2. Q/K post-rotary RMSNorm is head-row-owned;
3. projection weight gradients are weight-tile-owned.

The checked-in path uses:

| Order | Kernel | Owner | Main output | Why this boundary exists |
| ----- | ------ | ----- | ----------- | ------------------------ |
| F0 | `_fused_attn_qkv_projection_norm_fwd_kernel` | `(M tile)` | `x_mix`, `x_hat`, `rms_inv` | One row RMS reduction is shared by all Q/K/V heads. |
| F1 | `_fused_attn_qkv_projection_qkv_fwd_kernel` | `(M tile, part)` where `part` is Q/K/V head | `q`, `k`, `v`, `qk_rms_inv` | Projection is compute-heavy and naturally head-tiled; Q/K keep rotary+RMSNorm in registers. |
| B1 | `_fused_attn_qkv_projection_x_norm_bwd_kernel` | `(M tile, K tile)` | `x_norm` | Rebuild normalized input once from saved `rms_inv`. |
| B2 | `_fused_attn_qkv_projection_qk_pre_ve_bwd_kernel` | `(M tile, Q/K/V part)` | `d_q_pre`, `d_k_pre`, optional VE grads | Invert Q/K scale+RMSNorm+rotary near the saved Q/K tensors; V part owns VE gate/table updates. |
| B3 | `_fused_attn_qkv_projection_dx_hat_bwd_kernel` | `(M tile, K tile)` | `d_x_hat`, `outer_rms_row_inner` | Sums Q/K/V projection-backward contributions into one activation gradient. |
| B4 | `_fused_attn_qkv_projection_outer_rms_dx_bwd_kernel` | `(M tile, K tile)` | `dx`, `dx0`, residual-scale grads | Finishes outer RMSNorm and residual/x0 mix backward. |
| B5 | `_fused_attn_qkv_projection_weight_section_bwd_kernel` | `(section, output-row tile, K tile)` | `dW_q`, `dW_k`, `dW_v` | Weight gradients reduce over `M`, so they need weight-tile ownership. |

This is the same "do not fuse across incompatible reduction axes" rule used in
Chapter 3. The extra launches buy simpler matmul shapes and avoid atomics on
large activation/weight gradients.

## 4.8 Forward Details

### 4.8.1 Step F0 - residual mix + outer RMSNorm

The first forward kernel computes:

```text
x_mix[m, k] = resid_scale * x[m, k] + x0_scale * x0[m, k]
sum_sq[m]   = sum_k x_mix[m, k]^2
s_x[m]      = rsqrt(sum_sq[m] / K + eps)
x_hat[m, k] = x_mix[m, k] * s_x[m]
```

Outputs:

- `x_mix`: returned to the attention residual path. This is a real public
  output, not just backward scratch.
- `x_hat`: materialized because every Q/K/V head reads the same normalized row.
- `rms_inv`: fp32 `(M,)`, saved for backward.

Why materialize `x_hat`? With d24, `K=1536` and the QKV side has
`n_head + 2*n_kv_head = 36` projected head parts. Recomputing the row RMSNorm
inside each head program would repeat a 1536-wide reduction dozens of times.
One HBM write/read of `x_hat` is cheaper than repeating that reduction.

### 4.8.2 Step F1 - one program per projected head part

`_fused_attn_qkv_projection_qkv_fwd_kernel` uses grid axis 1 as a logical part:

```text
part < n_head                         -> Q head
n_head <= part < n_head + n_kv_head   -> K head
otherwise                             -> V head
```

Each program computes one full head tile:

```text
acc[m, d] = sum_k x_hat[m, k] * W_part[head*D + d, k]
```

Weights may be fp32 master weights in the optimizer path. The kernel casts the
loaded weight tile to the activation dtype before `tl.dot`; this preserves the
bf16 tensor-core path without a standalone "cast whole weight matrix" kernel.

### 4.8.3 Q/K rotary + inner RMSNorm

For Q/K parts, projection output `q0` or `k0` is immediately transformed:

```text
lo, hi = split_half(z)        # z is q0 or k0, D must be even
r_lo   = lo * cos + hi * sin
r_hi   = -lo * sin + hi * cos
r      = concat(r_lo, r_hi)
s_qk   = rsqrt(mean_d(r^2) + eps)
out    = scale * r * s_qk
```

The implementation keeps this work in the same program as the head projection.
That avoids storing raw `q0`/`k0` or rotary pre-norm tensors. The only extra
state saved for backward is `qk_rms_inv[m, head]`.

The code uses the "split-half" rotary convention used by nanochat's `cos/sin`
cache. The public tables have shape `(1, T, 1, D/2)`. Batch is broadcasted;
sequence length is validated against the current `T`.

### 4.8.4 V and optional value embedding

V does not use rotary or QK RMSNorm. It exits the Q/K branch and optionally
adds value embedding:

```text
x_g       = x_hat[:, :ve_gate_channels]
logits    = x_g @ ve_gate_weight.T
gate      = 3 * sigmoid(logits)
ve        = ve_weight[ve_ids].view(M, n_kv_head, D)
v        += gate[..., None] * ve
```

The gate is per token and per K/V head. `ve_weight[token]` stores all K/V heads
flattened as `(n_kv_head * D)`; the kernel indexes the head slice directly, so
there is no runtime view materialization in Triton.

VE and no-VE use different compile-time tile constants. The gate path carries
extra `x_g`, `gate_w`, sigmoid, table load, and extra backward state; using the
same tile shape for both was slower in d24 sweeps.

## 4.9 Backward Details

### 4.9.1 Step B1 - rematerialize `x_norm`

Backward does not save `x_mix` or `x_hat`. It saves the original `x`, `x0`,
scales, and `rms_inv`, then rebuilds:

```text
x_mix = resid_scale * x + x0_scale * x0
x_norm = x_mix * rms_inv[:, None]
```

`x_norm` is materialized because both weight gradients and VE gate gradients
need the normalized input. This is the QKV equivalent of Chapter 3's careful
"save only what makes downstream reductions cheaper" rule.

### 4.9.2 Step B2 - Q/K pregrads and VE backward

Saved final Q/K already include scale:

```text
q = scale * y0_q
k = scale * y0_k
```

So Q/K RMSNorm backward uses:

```text
y0 = q_or_k / scale
g0 = d_q_or_d_k * scale
dr = s_qk * (g0 - y0 * mean_d(g0 * y0))
```

Then inverse rotary maps `dr` back to pre-rotary projection-output grads:

```text
d_lo = d_r_lo * cos - d_r_hi * sin
d_hi = d_r_lo * sin + d_r_hi * cos
d_q_pre or d_k_pre = concat(d_lo, d_hi)
```

V has no QK norm or rotary. In the no-VE path, its projection-output grad is
just `d_v`. In the VE path, V-owned programs also compute:

```text
d_gate        = sum_d(d_v * ve)
d_logits      = 3 * d_gate * sigmoid * (1 - sigmoid)
d_x_g        += d_logits @ ve_gate_weight
d_ve_weight  += d_v * gate
d_gate_weight += d_logits.T @ x_g
```

`d_ve_weight` uses atomics because multiple tokens in the batch can have the
same `ve_id`. `d_ve_gate_weight` also uses accumulation over row tiles, but its
shape is tiny: `(n_kv_head, ve_gate_channels)`.

### 4.9.3 Step B3 - `d_x_hat`

Projection input gradient is conceptually:

```text
d_x_hat = d_q_pre @ q_weight + d_k_pre @ k_weight + d_v @ v_weight
```

The kernel loops over Q, K, and V head sections and accumulates into one
`(BLOCK_M, BLOCK_K)` tile. It materializes `d_x_hat` in activation dtype and
also accumulates the outer-RMSNorm row inner:

```text
inner[m] += sum_k d_x_hat[m, k] * x_norm[m, k]
```

That side output is the same trick as Chapter 3's MLP dx kernel: move the row
inner reduction next to the tile that already has both operands hot.

The kernel supports `HEAD_SPLIT`, currently tuned for d24. Splitting a head
lets the matmul tile use smaller `D` chunks while still covering all Q/K/V
heads. This was more useful for QKV backward than for forward because the
backward `d_x_hat` kernel is a large repeated weight-read matmul.

### 4.9.4 Step B4 - outer RMSNorm and residual mix

Outer RMSNorm backward uses the materialized `d_x_hat` and precomputed row
inner:

```text
d_x_mix_norm = rms_inv * (d_x_hat - x_norm * mean_k(d_x_hat * x_norm))
```

The returned `x_mix` also has a direct gradient from the attention residual
tail. The final kernel folds that in before splitting the residual mix:

```text
d_x_mix = d_x_mix_norm + grad_x_mix
dx      = d_x_mix * resid_scale
dx0     = d_x_mix * x0_scale
d_resid_scale = sum(d_x_mix * x)
d_x0_scale    = sum(d_x_mix * x0)
```

Scale gradients are scalar atomics, which are cheap compared with the Q/K/V
matmuls.

### 4.9.5 Step B5 - projection weight gradients

Weight gradients reduce over rows:

```text
dW_q = d_q_pre.T @ x_norm
dW_k = d_k_pre.T @ x_norm
dW_v = d_v.T     @ x_norm
```

`_fused_attn_qkv_projection_weight_section_bwd_kernel` uses a `section` grid
axis:

```text
section = 0 -> Q weight
section = 1 -> K weight
section = 2 -> V weight
```

This makes the three weight-gradient matmuls share one kernel definition while
keeping the outputs separate. The output tensor dtype is the weight dtype, so
gradients land directly in the fp32 master weight path when weights are fp32.

## 4.10 Saved and Recomputed Ledger

| Tensor | Save? | Reason |
| ------ | ----- | ------ |
| `x`, `x0`, residual scales | Yes | Needed to rebuild `x_mix` and produce residual-scale grads. |
| `rms_inv` | Yes | Saves one full outer RMS reduction in backward. |
| `x_mix` | Public output only | Returned for the attention residual path; backward still rematerializes normalized input. |
| `x_hat` / `x_norm` | No | Rebuilt once in B1; saving it would cost `B*T*K` activation memory. |
| `q`, `k` | Yes | Lets Q/K RMSNorm backward avoid recomputing Q/K projections. |
| `v` | Public output only | Its grad path is direct; raw V-before-VE is not saved. |
| `qk_rms_inv` | Yes | Needed for Q/K RMSNorm backward. |
| `q0`, `k0`, rotary intermediates | No | Recovered by inverse scale/RMSNorm/rotary from saved final Q/K. |
| `d_z` concat buffer | No | Q/K pregrads and V grad are consumed by targeted kernels. |
| `d_x_hat` | Backward temp | Materialized only between B3 and B4 to keep the outer RMS path simple. |

This is a deliberate midpoint between "save everything" and "recompute every
matmul". Q/K projections are expensive enough that final Q/K are saved; `x_hat`
is large enough that it is recomputed once instead.

## 4.11 Numerical Precision Path

The d24 training path is bf16 activations with fp32 master weights. The QKV
kernel policy is:

```text
load activation -> activation dtype
load weight     -> cast to activation dtype before tl.dot
tl.dot          -> fp32 accumulator
elementwise RMS -> fp32 reductions
store Q/K/V     -> activation dtype
save rms_inv    -> fp32
```

Why not keep every intermediate fp32? Because the projection matmuls are
tensor-core dominated. Casting loaded weight tiles to bf16 lets `tl.dot` use
the same fast path as the native compiled GEMM. RMS reductions still use fp32
for stability; the final normalized vectors are cast back to activation dtype
before storage.

Backward follows the same rule: reductions and row inners are fp32, but the
large materialized activation-gradient buffers use activation dtype. That keeps
peak memory low enough for B=2 d24 while preserving the numerically important
row reductions.

## 4.12 Expected Savings Ledger

### Forward

Compared with a native unfused attention prefix:

| Native work | Fused result |
| ----------- | ------------ |
| residual mix kernel | folded into F0 |
| RMSNorm kernel | folded into F0 |
| three projection calls with separate reshapes | one head-partitioned Triton projection kernel |
| rotary Q/K kernel | folded into Q/K parts of F1 |
| Q/K RMSNorm kernels | folded into Q/K parts of F1 |
| Q/K scale kernels | folded into Q/K stores |
| optional VE gate and table add | folded into V parts of F1 |

The key win is not just launch count. It is avoiding large intermediate tensors
for `q0`, `k0`, rotary output, Q/K pre-scale output, and value-embedding add.

### Backward

Backward saves:

| Native work | Fused result |
| ----------- | ------------ |
| separate Q/K norm backward | B2, next to inverse rotary |
| separate rotary backward | B2 |
| optional VE backward Python/PyTorch ops | V-owned B2 parts |
| `d_x_hat` from three independent matmuls | B3 accumulates all sources |
| outer RMSNorm backward plus residual mix backward | B4 |
| three weight-gradient call sites | one sectioned B5 kernel |

The remaining large costs are still GEMM-like: `d_x_hat` and `dW`. This is why
the implementation tunes their tile shapes separately instead of chasing one
all-purpose fused kernel.

## 4.13 Performance Reality

The measured d24 compile path settled on:

- forward no-VE: larger `BLOCK_M/BLOCK_K` because there is less per-row scalar
  work;
- forward VE: smaller `BLOCK_K` because gate/table logic increases register
  pressure;
- backward `d_x_hat`: `BLOCK_M=128`, `BLOCK_K=64`, and head splitting;
- weight gradients: sectioned kernel with d24-tuned row/output/K tiles.

The important lesson from tuning was that microbench wins did not always carry
into compiled training. The full graph changes live ranges and peak memory. In
particular, materializing `d_x_hat` helped the QKV backward enough to beat the
recompute-heavy version, but materializing `x_hat` for backward did not pay for
itself.

## 4.14 End-to-End Landing

The public op is a `torch.library.triton_op` with registered autograd:

```text
fused_attn_qkv_projection
  -> nanoops::fused_attn_qkv_projection_fwd
  -> nanoops::fused_attn_qkv_projection_bwd
```

`nanoops.integration` wires it into `GPT.forward` when
`NANOOPS_FUSED_ATTN=1`. Public tensors stay in `(B, T, *)` layout, while the
kernel implementation flattens `B*T -> M`. That makes the model code readable
and keeps Triton kernels on dense row-major `(M, K)` / `(M, H, D)` layouts.

Current limitations:

- no affine RMSNorm weight on the main d24 path;
- rotary tables must be 4D broadcast tables `(1, T, 1, D/2)`;
- `head_dim` must be even;
- VE gate channel count must fit the tuned backward `DX_HAT_BLOCK_K`;
- d24 shapes are the tuning target, not a universal autotuned library.

## 4.15 Takeaway

`fused_attn_qkv_projection` is the attention-prefix analogue of Chapter 3's
MLP block: use Triton where the surrounding elementwise and reduction work can
ride for free next to matmuls, but stop fusing when ownership changes. The
result is not one giant kernel. It is a short pipeline whose boundaries match
RMS row reductions, head-local Q/K transforms, activation-gradient matmuls, and
weight-gradient reductions.
