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
dS_ij = P_ij * (dP_ij - delta_i)
dQ_i += (1/sqrt(D)) * sum_j dS_ij * K_j
dV_j += sum_i P_ij * dO_i
dK_j += (1/sqrt(D)) * sum_i dS_ij * Q_i
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

## 5.10 Kernel Layout Overview

Chapter 5 is split across two files because the public op spans two different
domains:

- `triton_fused_attn_spda.py`: Flash-style SDPA forward/backward;
- `triton_fused_attn_spda_and_output.py`: output projection, residual add, and
  the backward bridge that produces `delta`.

The checked-in path uses:

| Order | Kernel/op | Owner | Main output | Why this boundary exists |
| ----- | --------- | ----- | ----------- | ------------------------ |
| F0 | `_fused_attn_spda_fwd_kernel` | `(batch, Q-row tile, K/V head)` | `attn_out`, `lse` | Streams K/V tiles with online softmax; never stores scores/probs. |
| F1 | `_fused_attn_spda_and_output_proj_fwd_kernel` | `(M tile, C tile)` | `y = residual + attn_out @ W_o.T` | Output projection is a standard GEMM plus residual add. |
| B1 | `_fused_attn_spda_and_output_proj_dattn_delta_bwd_kernel` | `(M tile, Q head)` | `d_attn_out`, `delta` | Computes the FlashAttention `delta` while producing `dO`. |
| B2 | `_fused_attn_spda_and_output_proj_dweight_bwd_kernel` | `(C tile, D_in tile)` | `d_proj_weight` | Weight gradient reduces over M, so it needs weight-tile ownership. |
| B3 | `_fused_attn_spda_dq_bwd_kernel` | `(batch, Q-row tile, K/V head)` | `dQ` | Q-owned pass writes each dQ element once. |
| B4 | `_fused_attn_spda_dkv_bwd_kernel` | `(batch, K/V tile, K/V head)` | `dK`, `dV` | K/V-owned pass writes each dK/dV element once, avoiding atomics. |

This follows the same rule as Chapters 3 and 4: do not cross reduction-axis
ownership boundaries just to reduce launch count. The useful fusion is the
`d_attn_out + delta` handoff; fusing dQ and dKV into one kernel would either
duplicate work or require atomics.

## 5.11 SDPA Forward in Detail

### 5.11.1 Tile ownership

The forward launch grid is:

```text
grid = (B * M_tiles, H_kv)
```

Each program fixes:

```text
bid    = pid_bm // M_TILES
pid_m  = pid_bm - bid * M_TILES
kv_hid = program_id(1)
```

For GQA, one K/V head serves `GQA_GROUP = H_q // H_kv` query heads. The program
loads one logical Q tile shaped `(BLOCK_M, GQA_GROUP, D)` and flattens it only
for tensor-core dot:

```text
offs_hm     = arange(0, GQA_GROUP * BLOCK_M)
row_in_tile = offs_hm // GQA_GROUP
head_off    = offs_hm - row_in_tile * GQA_GROUP
offs_m      = pid_m * BLOCK_M + row_in_tile
hid         = kv_hid * GQA_GROUP + head_off
```

This row-major expansion preserves row locality while letting one K/V tile feed
all query heads in the group. Non-GQA is the same path with `GQA_GROUP = 1`.

### 5.11.2 Online softmax recurrence

For each streamed K/V tile, the kernel computes scores:

```text
s = (Q_tile @ K_tile.T) * sm_scale
```

Masked positions receive `-inf`. Each row tracks:

```text
m_i   = running row max
l_i   = running row exp-sum in the shifted basis
acc_i = running numerator for P @ V
```

The update is:

```text
m_new   = max(m_i, max(s))
alpha   = exp(m_i - m_new)
p       = exp(s - m_new)
l_new   = alpha * l_i + sum(p)
acc_new = alpha * acc_i + p @ V_tile
```

After the last visible K/V tile:

```text
out = acc / l
lse = m + log(l)
```

`lse` is saved in fp32 because backward recomputes probabilities from
`exp(score - lse)`.

### 5.11.3 Full tiles vs boundary tiles

The kernel computes three loop regions:

1. left boundary: needs elementwise sliding/causal mask;
2. full-valid middle: every Q row in the tile can see every K in the tile;
3. right boundary: needs elementwise mask.

For `IS_FULL_CONTEXT`, the lower sliding-window bound is compiled away. This
keeps the main path simpler while still sharing one source kernel. It is not as
specialized as industrial FlashAttention kernels that generate separate varlen,
full causal, local, and boundary kernels, but it avoids carrying the most
expensive mask checks in the middle loop.

## 5.12 Output Projection Forward

`attn_out` is flattened to `(M, D_in)` where:

```text
M    = B * T
D_in = H_q * D
C    = residual width
```

The projection kernel computes:

```text
y[m, o] = residual[m, o] + sum_i attn_out[m, i] * proj_weight[o, i]
```

Like the QKV kernels, output projection loads weights in their stored dtype and
casts tiles to the activation dtype before `tl.dot`. This keeps bf16 tensor-core
matmuls on the d24 path while storing the result in activation dtype.

Forward still saves `attn_out`. That looks expensive, but SDPA backward and
output-projection backward both need it:

- `d_proj_weight = dy.T @ attn_out`;
- `delta = sum(attn_out * d_attn_out)`.

Recomputing full SDPA output just to avoid saving `attn_out` would be more
expensive than the memory saved for current d24 B=2.

## 5.13 Output Projection Backward and Delta Handoff

The key fusion in this chapter is B1:

```text
d_attn_out[m, i] = sum_o dy[m, o] * proj_weight[o, i]
delta[m, h]      = sum_d attn_out[m, h, d] * d_attn_out[m, h, d]
```

If output projection and SDPA backward were separate, the system would:

1. run `d_attn_out = dy @ W_o`;
2. launch another kernel to compute `delta = row_dot(attn_out, d_attn_out)`;
3. run SDPA backward.

The fused bridge does 1 and 2 in the same program. It already owns an
`(M tile, head)` slice of `d_attn_out`, and it has the matching `attn_out`
slice hot, so the row-dot side output is cheap.

`d_proj_weight` stays a separate B2 kernel because its reduction axis is M:

```text
dW_o[o, i] = sum_m dy[m, o] * attn_out[m, i]
```

Trying to produce `dW_o` in B1 would either use atomics over weight tiles or
materialize a large partial buffer. The separate weight-owned kernel is simpler
and faster for the checked-in d24 shapes.

## 5.14 SDPA Backward From Delta

Backward recomputes scores tile-by-tile:

```text
S_ij = sm_scale * dot(Q_i, K_j)
P_ij = exp(S_ij - LSE_i)
```

Given `dO = d_attn_out` and `delta_i = sum_d O_i,d * dO_i,d`, the softmax
backward is:

```text
dP_ij = dot(dO_i, V_j)
dS_ij = P_ij * (dP_ij - delta_i)
```

The scale belongs on the matmul gradients:

```text
dQ_i += sm_scale * sum_j dS_ij * K_j
dK_j += sm_scale * sum_i dS_ij * Q_i
dV_j +=            sum_i P_ij  * dO_i
```

The implementation uses two kernels:

- `_fused_attn_spda_dq_bwd_kernel`: Q-owned, loops over visible K/V tiles and
  writes `dQ` directly;
- `_fused_attn_spda_dkv_bwd_kernel`: K/V-owned, loops over visible Q tiles and
  writes `dK`/`dV` directly.

This is the same structural split used by FlashAttention-style backward. The
alternative "one Q-owned backward that atomic-adds dK/dV" is much simpler to
write, but atomics on `(B, T, H_kv, D)` are too expensive and noisy for the
training path.

## 5.15 Sliding Window and Boundary Math

For query row `i`, visible keys are:

```text
max(0, i - window_size + 1) <= j <= i
```

For a Q tile `[q_first, q_last]`, the forward kernel starts with:

```text
kv_tile_start = max(0, q_first - WINDOW + 1) // BLOCK_N
kv_tile_end   = ceil(min(N, q_first + BLOCK_M) / BLOCK_N)
```

The middle fully-valid region is bounded by rows for which every Q in the tile
can see every K in the K tile. Boundary tiles keep elementwise masks:

```text
mask = (offs_n <= offs_m) & (offs_n >= offs_m - WINDOW + 1)
```

The dKV pass inverts this relation. A K/V tile only needs Q tiles whose rows
could attend to that K/V tile:

```text
q_tile_start = kv_start // BLOCK_M
q_tile_end   = ceil(min(M, kv_start + BLOCK_N + WINDOW - 1) / BLOCK_M)
```

This avoids scanning all `T/BLOCK_M` Q tiles for each K/V tile.

## 5.16 Saved and Recomputed Ledger

| Tensor | Save? | Reason |
| ------ | ----- | ------ |
| `q`, `k`, `v` | Yes | SDPA backward recomputes scores and probabilities. |
| `attn_out` | Yes | Needed for `d_proj_weight` and `delta`. |
| `lse` | Yes | Needed to reconstruct `P = exp(S - LSE)` stably. |
| `proj_weight` | Yes | Needed for `d_attn_out`. |
| scores `S` | No | Recomputed tile-by-tile. |
| probabilities `P` | No | Recomputed tile-by-tile. |
| `d_attn_out` | Backward temp | Materialized because SDPA backward consumes it as `dO`. |
| `delta` | Backward temp | Materialized by B1 and consumed by SDPA backward. |

The main memory trade-off is saving `attn_out`. Not saving it would require
recomputing SDPA output before output-projection backward, which is not a win
for the current d24 sequence length and head count.

## 5.17 Numerical Precision Path

The d24 path uses bf16 Q/K/V and projection activations:

```text
QK dot          -> fp32 accumulator
score/max/lse   -> fp32
P @ V acc       -> fp32 accumulator
OUT store       -> activation dtype
projection dot  -> fp32 accumulator
projection store-> activation dtype
delta           -> fp32
```

Backward follows the same pattern. Softmax probabilities and `dS` are fp32;
the large output tensors (`dq`, `dk`, `dv`, `d_attn_out`) are stored in
activation dtype. This is the practical compromise that matches native bf16
training behavior and keeps memory within 24 GiB.

## 5.18 Expected Savings Ledger

### Forward

| Native work | Fused result |
| ----------- | ------------ |
| materialized score/probability matrix | never materialized; streamed online softmax |
| repeated K/V loads for GQA heads | one K/V tile feeds a query-head group |
| output projection plus residual add | one projection kernel with residual fold-in |

Forward still has two logical stages because SDPA and output projection have
different dataflow. Fusing them into one kernel would force the SDPA program to
also own C output channels, which breaks the clean online-softmax tile shape.

### Backward

| Native work | Fused result |
| ----------- | ------------ |
| `d_attn_out = dy @ W_o` | B1 |
| separate `delta = sum(O * dO)` pass | folded into B1 |
| `dW_o = dy.T @ O` | B2 weight-owned kernel |
| SDPA dQ | B3 Q-owned kernel |
| SDPA dK/dV with atomics | B4 K/V-owned kernel, no atomics |

The saved pass over `(B*T, H_q, D)` for `delta` is the "free" win that made the
combined op worth keeping even when output projection by itself was not enough
to move end-to-end training.

## 5.19 Performance Reality and Industrial Gap

The current kernels are intentionally simple compared with industrial
FlashAttention/FA-3/vLLM kernels:

- no varlen packed sequence layout;
- no persistent CTA scheduler;
- no TMA or async producer/consumer pipeline;
- no split-K or split-Q reduction path in the checked-in version;
- no separate generated kernels for every full-causal/local/boundary case;
- no FP8 path.

What they do have is the set needed by d24:

- contiguous `(B, T, H, D)` tensors;
- GQA-aware K/V reuse;
- full-context vs sliding-window constexpr branch;
- boundary/full-tile split inside the K/V loops;
- no score/probability materialization;
- no atomics for dK/dV;
- output-projection delta handoff.

In training measurements, the fused SDPA/output path is useful because it
reduces intermediate traffic and makes the attention tail compile-visible. It
does not claim to beat a mature vendor FlashAttention kernel on every shape.
The point of this chapter is to show the industrial design skeleton in code
small enough to inspect.

## 5.20 End-to-End Landing

The public op is:

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

`nanoops.integration` calls it immediately after
`fused_attn_qkv_projection`. Public tensors stay `(B, T, H, D)` through the
attention path and only flatten to `(B*T, *)` for the output projection GEMM.
This keeps the model API close to PyTorch while preserving dense Triton memory
access.

## 5.21 Takeaway

`fused_attn_spda_and_output` is not a "merge everything" kernel. It is a
Flash-style SDPA implementation plus one carefully chosen fusion at the output
projection backward boundary. That boundary removes the redundant delta pass,
keeps dQ and dKV ownership clean, and gives nanochat a compile-visible
attention tail that is understandable without vendor-kernel machinery.
