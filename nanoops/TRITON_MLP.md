# Chapter 3 — `fused_mlp`: production-level, fwd+bwd all Triton

> Part of [nanoops Triton Kernels](TRITON.md). Chinese version: [TRITON_MLP_zh.md](TRITON_MLP_zh.md).

---

Chapter 2's `FusedAddNorm` was a teaching artifact. This chapter is
nanchat's actual mlp-side fusion target — the standard transformer
mlp block (pre-norm + linear + relu² + linear + outer residual),
seven ops end-to-end, compressed into **3 fwd Triton kernels +
4 bwd Triton kernels**, no cuBLAS in the call chain. nanchat's
fc_weight / proj_weight are fp32 master tensors; every matmul
inline-casts to the activation dtype (bf16) at load time, so the
bf16 weight tile never gets materialized in HBM.

Math:
```
y = x + relu²(RMSNorm(x) @ W_fc.T) @ W_proj.T
```

API:
```python
def fused_mlp(x, fc_weight, proj_weight, eps=1e-6) -> y
#   x, fc_weight, and proj_weight must be contiguous CUDA tensors
#   RMSNorm is plain no-affine; there is no Python norm_weight argument.
```

The caller pre-sums any outer residual if needed; this block does the
standard `y = x + mlp(norm(x))` pattern.

What makes this worth reading after ch2: **ch2 is pure memory-bound
2-op fusion; ch3 is about fusing things into the bwd of a matmul**.
The matmul itself is compute-bound, but each bwd step
(dz / dW_proj / dW_fc / dx) carries an elementwise or reduction
"byproduct" — those byproducts are what we fuse.

### 3.1 Kernel layout overview

| Stage | Kernel | Grid | Role |
|---|---|---|---|
| Fwd 0 | `_fused_mlp_rms_norm_fwd_kernel` | 1D over M | Plain RMSNorm computes `x_hat` + side-output `rms_inv` |
| Fwd 1 | `_fused_mlp_fc_matmul_fwd_kernel` | 2D over (M, N_fc) | `z = x_hat @ W_fc.T`, W_fc cast inline at load |
| Fwd 2 | `_fused_mlp_proj_residual_fwd_kernel` | 2D over (M, K_out) | relu² + c_proj + outer residual add → `y` |
| Bwd A | `_fused_mlp_dz_bwd_kernel` | 2D over (M, N_fc) | `dz` + side-output `inner_buf` (D needs it) |
| Bwd B | `_fused_mlp_dproj_weight_bwd_kernel` | 2D over (K_out, N_fc) | `dW_proj` (fp32 master output) |
| Bwd C | `_fused_mlp_dfc_weight_bwd_kernel` | 2D over (N_fc, K) | `dW_fc` (fp32 master output) |
| Bwd D | `_fused_mlp_dx_bwd_kernel` | 2D over (M, K) | `dx_hat` matmul + no-affine RMSNorm bwd + outer residual fold → `dx` |

**All-Triton fwd/bwd is intentional**. At d24 shape (M=2048,
N_fc=6144, K=1536), every matmul can fuse one HBM round-trip or one
launch with an adjacent elementwise / weight cast / reduction; that
saving beats Triton's ~10-15% efficiency gap vs cuBLAS. Step 1 looks
like an isolated big matmul, but the fp32 master → bf16 activation
`.to()` cast on its own (one launch + 36 MB HBM write/read at d24)
eats exactly that cuBLAS edge — so Step 1 also goes Triton with the
cast folded into the load. See §3.4.

### 3.2 Forward

#### 3.2.1 Step 0 — no-affine RMSNorm

The fwd runs a small local `_fused_mlp_rms_norm_fwd_kernel`:

```python
# Step 0 caller (_fused_mlp_fwd_impl)
_fused_mlp_rms_norm_fwd_kernel[...](
    x,
    x_hat,
    rms_inv,
    M, K, eps,
    BLOCK_M=norm_cfg.block_m, BLOCK_D=BLOCK_D_NORM,
    num_warps=norm_cfg.num_warps,
)
```

Inside the kernel:
```python
x = tl.load(x_ptr + rows[:, None] * K + cols[None, :], ...)
rms_inv = rsqrt(sum(x * x) / K + eps)
x_hat = x * rms_inv[:, None]
tl.store(x_hat_ptr + ..., x_hat.to(x_hat_ptr.dtype.element_ty), ...)
tl.store(rms_inv_ptr + rows, rms_inv, ...)
```

This is deliberately no-affine: nanchat's RMSNorm has no learnable
per-channel scale on the hot path, and the Python API no longer accepts
one.

#### 3.2.2 Step 1 — `_fused_mlp_fc_matmul_fwd_kernel`: c_fc + inline weight cast

c_fc is an isolated large matmul on its own — no neighboring
elementwise byproduct to fuse into the matmul's register stage.
But fc_weight is **fp32 master** in nanchat while the activation
x_hat is bf16, so something has to cast before the matmul:

```python
# Naive version (replaced)
fc_w_bf16 = fc_weight.to(x_hat.dtype)        # standalone launch + 36 MB HBM write
z = torch.matmul(x_hat, fc_w_bf16.t())       # cuBLAS bf16 matmul, ~70% peak
```

That `.to()` is a standalone kernel: 36 MB of HBM write, then the
next kernel reads it back. At d24 that round-trip is ~75 μs, which
is exactly the size of cuBLAS's ~10-15% lead over Triton. So Step 1
goes Triton too:

```python
@triton.jit
def _fused_mlp_fc_matmul_fwd_kernel(x_ptr, w_ptr, z_ptr, M, N, K, ...):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        x_tile = tl.load(x_ptr + ...)                  # bf16
        w_tile = tl.load(w_ptr + ...)                  # fp32 native
        acc += tl.dot(x_tile, tl.trans(w_tile.to(x_tile.dtype)))
        #                                  ↑ cast after load, before dot
        #                                    bf16 tile lives only in registers
    tl.store(z_ptr + ..., acc.to(z_ptr.dtype.element_ty), ...)
```

The key pattern: **`w_tile.to(x_tile.dtype)` happens in registers**.
The bf16 weight tile never reaches HBM — saves 36 MB write/read and
one launch.

d24 manual sweep locked `(BLOCK_M=256, BLOCK_N=64, BLOCK_K=32, nw=8,
st=2)` — single kernel ~639 μs, slightly beating cuBLAS+cast at
~654 μs and 2× faster than a naïve Triton `(64,64,64,nw=4,st=3)` at
~1300 μs. Per-stage shared mem is `(256·32 + 64·32)·4 = 40 KB`, ×2 stages
= 80 KB, within the 100 KB SM budget on 3090.

#### 3.2.3 Step 2 — `_fused_mlp_proj_residual_fwd_kernel`

Three ops — `relu²(z) @ W_proj.T + x` — packed into one Triton
kernel:

```python
acc = tl.zeros((BLOCK_M, BLOCK_K_OUT), dtype=tl.float32)
for n_start in range(0, N, BLOCK_N):
    z = tl.load(...)                                # bf16 native (from fwd Step 1)
    relu_z = tl.where(z > 0.0, z, 0.0)              # bf16; tl.where preserves x's dtype
    r = relu_z * relu_z                             # bf16 * bf16 = bf16
    proj_w = tl.load(...)                           # fp32 native (master weight)
    acc += tl.dot(r, tl.trans(proj_w).to(z.dtype))  # cast folded before dot
    #                                ↑ bf16 weight tile only in registers

# Residual fold-in: cast acc back to bf16 first, then add native-dtype residual
residual = tl.load(residual_ptr + offs, ...)         # bf16
y = acc.to(y_ptr.dtype.element_ty) + residual         # bf16
tl.store(y_ptr + offs, y, ...)
```

Patterns to notice:
- **bf16 throughout + fp32 acc**: z/r are bf16 feeding the tensor
  cores, proj_w is fp32 master cast to bf16 in registers before the
  dot, and the accumulator stays in fp32 for safety.
  `tl.where(z > 0.0, z, 0.0)` coerces the literal `0.0` to z's dtype
  (unlike `tl.maximum(z, 0.0)` which would promote z to fp32) —
  critical for keeping the bf16 pipeline alive.
- **Inline weight cast same as Step 1**: fp32 master cast to z's
  dtype on load. The bf16 weight tile never leaves registers; the
  caller doesn't have to pre-cast proj_weight.
- **Residual cast deferred**: cast `acc.to(bf16)` first, then `+ residual(bf16)`,
  rather than promoting residual to fp32 first. Saves one bf16→fp32
  conversion on the load + skips a final cast on store. The cost is
  the last add happens in bf16 instead of fp32 — precision loss
  ~1e-3/element, within atol.
- **d24 locked**: `(BLOCK_M=128, BLOCK_K_OUT=64, BLOCK_N=32, nw=8, st=2)`.

### 3.3 Backward — 4 Triton kernels do it all

bwd produces 4 gradient tensors: `dz, dW_proj, dW_fc, dx`. The four
reduction axes are mutually orthogonal (A reduces K_out, B reduces M,
C reduces M, D reduces N_fc), so **packing them into a single kernel
isn't possible**. But each step fuses one HBM round-trip with an
adjacent elementwise op.

#### 3.3.1 Step A — `_fused_mlp_dz_bwd_kernel`: matmul + relu² bwd + side-output

Math:
```
dr = dy @ W_proj                # matmul, reduce K_out
dz = 2·relu(z) · dr             # elementwise (relu² bwd)
inner_partial = Σ_n(dz·z) / norm_dim  ← side-output for D
```

Condensed kernel body:
```python
dr = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for kp_start in range(0, K_out, BLOCK_K_OUT):
    dy = tl.load(...)                            # bf16
    proj_w = tl.load(...)                        # fp32 master
    dr += tl.dot(dy, proj_w.to(dy.dtype))        # cast folded before dot

z = tl.load(...)                                              # bf16 native
relu_z = tl.where(z > 0.0, z, 0.0)
dz = dr.to(dz_ptr.dtype.element_ty) * 2 * relu_z              # bf16 throughout
tl.store(dz_ptr + ..., dz, ...)

# Side-output: per-tile partial inner, atomic_add into (M,) fp32 buffer
inner_partial = tl.sum(dz * z, axis=1, dtype=tl.float32) / K_out
tl.atomic_add(inner_buf_ptr + rows, inner_partial, mask=row_mask)
```

d24 locked: `(BLOCK_M=128, BLOCK_N=128, BLOCK_K_OUT=32, nw=8, st=2)`.

Three patterns worth calling out:

**1. dr never leaves HBM** — the matmul accumulator gets consumed by
`* 2·relu(z)` into dz immediately. If PyTorch ran this path you'd
see `dr = dy @ W_proj` (writes 25 MB to HBM at d24) then
`dz = 2·relu(z)·dr` (reads it back). The fused version stays in
registers throughout.

**2. dz is bf16 throughout** — `dr.to(bf16) * 2 * relu_z`. The `2` is
intentionally an integer literal (not `2.0`) so bf16 isn't promoted
to fp32. If it were promoted, the downstream `tl.sum` would receive
fp32 and the store would need an extra cast.

**3. inner_partial side-output** — `tl.sum(dz * z, axis=1, dtype=tl.float32)`,
then atomic_add into `inner_buf[rows]`. Three details:

- **`dtype=tl.float32` forces the accumulator**: dz and z are both
  bf16, and summing N=6144 bf16 products in bf16 would overflow
  precision (8-bit mantissa). `dtype=` makes the sum's accumulator
  fp32, promoting each bf16 product to fp32 before adding — same as
  PyTorch's internal promotion.
- **Divide by `K_out` here, not in D**: in MLP, `K_out == norm_dim`
  (forward asserts `K_proj_out == K`), so A can use its own K_out
  parameter to divide. D then loads `inner_buf` directly without
  dividing. Dividing before the atomic_add also means the
  accumulated values have smaller magnitude — better fp32 rounding.
- **Why atomic_add instead of scratchpad+reduce**: see below.

##### Key algebraic identity

D's RMSNorm bwd formula needs:
```
inner[m] = (1/norm_dim) · Σ_k(g_eff[m,k] · y_norm[m,k])
```
where `g_eff = dx_hat · nw`, `y_norm = x · rms_inv`, `x_hat = y_norm · nw`.

If D computed this directly, each (m, k_tile) program would need a
full K-reduction with `dx_hat` spread across a tiny BLOCK_M=4 tile
(to fit full-K in registers) — tensor cores would be unused, ~5×
slower.

But the forward has `z[m,n] = Σ_k x_hat[m,k] · W_fc[n,k]`, and bwd
has `dx_hat[m,k] = Σ_n dz[m,n] · W_fc[n,k]`. Substituting both into
the inner inner product:

```
Σ_k(dx_hat[m,k] · x_hat[m,k])
  = Σ_k (Σ_n dz[m,n] · W[n,k]) · x_hat[m,k]
  = Σ_n dz[m,n] · (Σ_k W[n,k] · x_hat[m,k])
  = Σ_n dz[m,n] · z[m,n]
```

— the adjoint property `⟨L*v, u⟩ = ⟨v, Lu⟩` of a linear operator;
c_fc's transpose lets us compute the same inner over the N
dimension. Since A already has dz and z live in registers, one extra
`tl.sum(dz * z)` is essentially free.

##### Why atomic_add (not scratchpad)

A's grid is `(M/BM, N/BN)` — the same m_tile is split across
N/BN = 48 programs. Computing `inner` needs to sum those 48 partials
along N. Options:

| Option | Cost at d24 |
|---|---|
| **atomic_add (current)** | ~10 μs; hardware atomic, inner_buf (8 KB) lives in L2 |
| Scratchpad `(num_n_tiles, M)` + `torch.sum(dim=0)` | ~15 μs; extra buffer + one reduce launch (benched flat) |
| Compute it inside D | ~25-50 μs; D's grid is `(M/BM, K/BK)`, m_tiles get K-duplicated 24× — either redundant compute or inter-program sync; z isn't a D input, so we'd also have to add an HBM read |

atomic_add wins because dz/z are already in registers, the target
buffer (M,) lives in L2, no extra buffer, no extra launch.
**Self-contained inside the kernel**.

#### 3.3.2 Step B — `_fused_mlp_dproj_weight_bwd_kernel`: dy.T @ relu²(z)

```python
acc = tl.zeros((BLOCK_K_OUT, BLOCK_N), dtype=tl.float32)
for m_start in range(0, M, BLOCK_M):
    dy = tl.load(...)                                 # bf16
    z = tl.load(...)                                  # bf16
    relu_z = tl.where(z > 0.0, z, 0.0)
    r = relu_z * relu_z                               # r recomputed inline, not read from HBM
    acc += tl.dot(tl.trans(dy), r)                    # bf16 @ bf16
tl.store(dW_proj_ptr + ..., acc.to(dW_proj_ptr.dtype.element_ty), ...)
#                              ↑ caller allocates with W_proj.dtype, so store lands directly on the fp32 master
```

B and A are "two bwd outputs of the same fwd op", but their
reduction axes differ (B reduces M, A reduces K_out), so they're
split into two kernels.

Two fusions happen at once:
- **r recomputed in registers** — A already wrote dz out to HBM, but
  r itself was never saved (fwd doesn't save it either; only z is
  saved). B reconstructs `relu²(z)` on the fly, saving one M·N_fc
  HBM round-trip.
- **dW_proj lands directly on fp32 master** — the caller allocates
  dW_proj with `dtype=W_proj.dtype`, so the kernel's fp32 acc stores
  straight into the fp32 master buffer. The optimizer doesn't need
  any `.to()` to lift bf16 grads back to fp32 master.

d24 locked: `(BLOCK_K_OUT=64, BLOCK_N=128, BLOCK_M=64, nw=4, st=2)`.
Note B is the only matmul in this group that doesn't need an inline
weight cast — dy and z are both already bf16 (caller passes bf16,
not fp32 master), there's nothing to cast.

#### 3.3.3 Step C — `_fused_mlp_dfc_weight_bwd_kernel`: dz.T @ x_hat, x_hat recomputed

```python
acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
for m_start in range(0, M, BLOCK_M):
    x = tl.load(x_ptr + ..., ...)               # bf16
    rms_inv = tl.load(rms_inv_ptr + ms, ...)    # fp32
    x_hat = x * rms_inv[:, None]                # fp32 (auto-promote)

    dz_tile = tl.load(...)                                       # bf16
    acc += tl.dot(tl.trans(dz_tile), x_hat.to(dz_tile.dtype))    # bf16 @ bf16
tl.store(dW_fc_ptr + ..., acc.to(dW_fc_ptr.dtype.element_ty), ...)
#                            ↑ caller allocates with W_fc.dtype, dW_fc lands on fp32 master
```

**x_hat is reconstructed in the GEMM inner loop** — fwd doesn't have
to write x_hat to HBM for bwd's sake (forward's ctx saves only `x`
and `rms_inv`; x_hat is discarded).

Compared to the cuBLAS path:
```
[cuBLAS path]
x_hat = (x * rms_inv).contiguous()             # M·K HBM write
dW_fc = dz.T @ x_hat                            # M·K HBM read + cuBLAS matmul
```

vs the Triton fused version: the matmul reconstructs x_hat inline,
x_hat never leaves registers. **One M·K HBM write + read saved**.
The cost is Triton matmul being ~10-15% slower than cuBLAS; net win
at d24 is ~30 μs. dW_fc lands on fp32 master same as B.

d24 locked: `(BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, nw=4, st=2)`. Note
that C's matmul inputs are `dz_tile (bf16)` and `x_hat (fp32 register)`,
so the cast direction is `x_hat.to(bf16)`, not the usual weight cast
— but the effect is identical (bf16 tile feeds the tensor cores).

#### 3.3.4 Step D — `_fused_mlp_dx_bwd_kernel`: dx all-sources merge

x appears twice in the forward:
```
y = x + mlp(norm(x))
       ↑     ↑
     outer  norm path
```

So dx has two contributions:
- **outer-residual path**: `dx ← dy` (direct passthrough)
- **norm path**: `dx ← RMSNorm_bwd(dx_hat)`, `dx_hat ← dz @ W_fc`

D packs both into **one kernel**:

```python
dx_hat = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
for n_start in range(0, N_fc, BLOCK_N):
    dz_tile = tl.load(...)                                       # bf16
    W_fc_tile = tl.load(...)                                     # fp32 master
    dx_hat += tl.dot(dz_tile, W_fc_tile.to(dz_tile.dtype))       # cast folded before dot

# RMSNorm bwd inline, using A's pre-computed inner
rms_inv = tl.load(rms_inv_ptr + rows, ...)       # fp32
inner = tl.load(inner_buf_ptr + rows, ...)       # fp32 (A already divided by norm_dim)
x = tl.load(x_ptr + offs, ...)                   # bf16
dy = tl.load(dy_ptr + offs, ...)                 # bf16 native (passthrough)

y_norm = x * rms_inv[:, None]                    # fp32 (auto-promote)
g_eff = dx_hat

# One-line merge: RMSNorm bwd norm path → cast bf16 → + dy
dx = (rms_inv[:, None] * (g_eff - y_norm * inner[:, None])).to(bf16) + dy
tl.store(dx_ptr + offs, dx, ...)
```

Three fusions at once:

**1. dx_hat never leaves HBM** — matmul produces an fp32 register
tile that's immediately consumed by the RMSNorm bwd formula. Cf. the
cuBLAS path: `dx_hat = dz @ W_fc` (M·K HBM write) + a later kernel
reads it back. Fused stays in registers. **Saves ~13 μs of HBM
round-trip at d24**.

> Safety precondition: dW_fc (kernel C) uses x_hat, not dx_hat, so
> dx_hat doesn't need to be materialized for any other kernel.

**2. inner comes pre-computed from A** — D doesn't do a K-reduction,
the whole dx formula is pure elementwise. That unlocks tensor-core-
friendly tiles like BLOCK_M=64, BLOCK_K=64 (if D had to do its own
K-reduction, BLOCK_M would be pinned to 4 and tensor cores wouldn't
fire).

**3. outer-residual folded into the store** — `+ dy` is done in-kernel,
no external Python `dx_total = dx_norm + dy` step and no extra HBM
round-trip.

**D's locked config + shared-mem budget**: the d24 bf16 sweep winner is
`(BLOCK_K=64, BLOCK_N=64, nw=4, st=2)`, now used unconditionally. The old
fp32 IEEE branch was removed so the training-oriented tensor-core path is the
only implementation.

This was a trap discovered when we first tried `@triton.autotune`
across all 6 kernels — autotune ran every candidate, the
over-budget ones produced wrong results silently, and autotune
picked one that "looked fastest" but had wrong output. The fix was
to drop autotune dispatch entirely, switch to a manual sweep with
shared-mem filtering, and lock configs in the caller. Side benefit:
works with CUDA Graph capture (autotune dispatch doesn't). BLOCK_M=64
is fixed for the dx kernel.

##### Precision path in the dx formula

```python
dx = (rms_inv[:, None] * (g_eff - y_norm * inner[:, None])).to(bf16) + dy
#     [    fp32     *      (fp32 - fp32 * fp32)        ]    bf16 + bf16
```

The bracket computes in fp32 end-to-end, then **casts to bf16 before
adding dy**. Same residual-defer trick as §3.2.3 — dy doesn't have
to be promoted, the final add happens in bf16.

### 3.4 Design tradeoff summary

| Op | Owner | Why |
|---|---|---|
| Fwd c_fc matmul (z = x_hat @ W_fc) | Triton (`_fused_mlp_fc_matmul_fwd_kernel`) | fp32→bf16 weight cast folded into load, saves 36 MB HBM round-trip + 1 launch, beats cuBLAS's ~10-15% efficiency edge |
| Fwd relu² + c_proj + residual | Triton | three ops fused, r never to HBM; proj_w also cast inline |
| Fwd RMSNorm | Triton | local no-affine RMSNorm kernel writes x_hat + rms_inv |
| Bwd dz (A) | Triton | matmul + relu² bwd + atomic_add side-output, three in one; proj_w cast inline |
| Bwd dW_proj (B) | Triton | r recomputed and fused into matmul; output goes straight to fp32 master |
| Bwd dW_fc (C) | Triton | x_hat recomputed and fused into matmul; output goes straight to fp32 master |
| Bwd dx (D) | Triton | dx_hat matmul + RMSNorm bwd + outer residual fold, **three ops in one**; W_fc cast inline |

**The guiding principle**: whenever a matmul has a fusable byproduct
nearby (elementwise / small reduction / **dtype cast**), writing
Triton is worth absorbing the ~10-15% efficiency hit. Under the
fp32-master + bf16-activation combination, **inline weight cast is
itself a worth-fusing byproduct** — a standalone `.to()` is one
launch + one HBM round-trip, which at d24 is enough to give back
cuBLAS's lead.

dW outputs match the master weight's dtype (caller allocates with
`dtype=W.dtype`), so the optimizer doesn't need to `.to()` grads
from bf16 back to fp32 master; bwd lands directly on the master.

### 3.5 Numerical precision path

The entire bwd is **bf16 in registers / fp32 in accumulator / bf16
in HBM**:

| Operation | dtype | Why |
|---|---|---|
| HBM load: x, z, dy, dz | bf16 native | caller dtype |
| HBM load: W_fc, W_proj | fp32 native (master) | nanchat uses fp32 master weights |
| HBM load: rms_inv, inner_buf | fp32 native | precision-sensitive side-outputs |
| Weight cast `W_*.to(activation.dtype)` | bf16 (in register) | inline cast before dot, bf16 weight tile never leaves registers |
| matmul accumulator (`acc`, `dx_hat`, `dr`) | fp32 | tensor core's default fp32 acc |
| `inner_partial = tl.sum(dz·z, dtype=fp32)` | fp32 | explicit accumulator promotion guards against bf16 mantissa overflow |
| `bf16 * fp32` (e.g. `x * rms_inv`) | fp32 (auto-promote) | Triton's standard promotion rule |
| relu² bwd: `dr.to(bf16) * 2 * relu_z` | bf16 | int `2` doesn't trigger fp32 promote (vs `2.0` would) |
| RMSNorm bwd formula `rms_inv·(g_eff - y_norm·inner)` | fp32 | precision-critical path |
| store: `dx`, `dz`, `y` | bf16 | caller dtype |
| store: `dW_fc`, `dW_proj` | fp32 (master) | lands directly on fp32 master, optimizer doesn't need to lift |
| dx final `+ dy` | bf16 + bf16 | residual defer (accepts bf16 add precision loss) |

The bf16 activation route never detours through fp32: load into
registers is bf16, feed the tensor cores in bf16, store still bf16.
fp32 only shows up in: (1) matmul accumulator; (2) weights' native
HBM representation (cast to bf16 happens in registers, HBM never
holds a bf16 weight copy); (3) the RMSNorm formula and dW outputs
(precision-sensitive / consumed by optimizer).

### 3.6 Expected savings ledger

For d24 (M=2048, N_fc=6144, K=1536, bf16).

#### Forward

Native (5 standalone ops + fp32→bf16 weight cast):
```
RMSNorm:    read x (M·K) + write x_hat (M·K)               = 2·M·K
cast W_fc:  read W_fc fp32 (2·N·K) + write W_fc bf16 (N·K) = 3·N·K  ← standalone launch
matmul:     read x_hat (M·K) + W_fc bf16 (N·K) + write z   = 2·M·K + N·K + M·N
relu²:      read z (M·N) + write r (M·N)                   = 2·M·N
cast W_proj: same as above                                  = 3·N·K
matmul:     read r (M·N) + W_proj bf16 (K·N) + write mlp   = M·N + N·K + M·K
add:        read mlp (M·K) + x (M·K) + write y (M·K)       = 3·M·K
────────────────────────────────────────
Total HBM:  8·M·K + 5·M·N + 8·N·K
launches:   7 (5 ops + 2 casts)
```

> fp32 master weight is nanchat's actual configuration; if weights
> were already bf16, the two casts disappear and the ledger drops
> back to 8·M·K + 5·M·N + 2·N·K + 5 launches.

Fused:
```
Step 0 (Triton):  read x (M·K) + write x_hat (M·K)               = 2·M·K
Step 1 (Triton _cast_matmul):
                  read x_hat (M·K) + W_fc fp32 (2·N·K)
                  + write z (M·N)                                 = 2·M·K + 2·N·K + M·N
                  (bf16 weight tile stays in registers, never HBM)
Step 2 (Triton):  read z (M·N) + W_proj fp32 (2·K·N)
                  + x (M·K) + write y (M·K)                       = M·N + 2·N·K + 2·M·K
────────────────────────────────────────
Total HBM:  6·M·K + 2·M·N + 4·N·K
launches:   3
```

Net savings (fp32 master case):
- **HBM saved: 2·M·K + 3·M·N + 4·N·K** — r/mlp never to HBM; two
  bf16 weight copies erased
- d24: ≈ 6.3 MB + 75.5 MB + 75.5 MB ≈ **157 MB / 936 GB/s ≈ 168 μs of HBM time**
- launches: 7 → 3, **4 saved** (~40-120 μs)

(If weights were natively bf16, the two cast rows disappear, HBM
savings drop to 2·M·K + 3·M·N ≈ 82 MB ≈ 87 μs, launches 5→3 saves 2.)

#### Backward

Native (PyTorch's mlp bwd chain expanded, estimated from prod impl):
```
~8 kernel launches:
  - relu² bwd                                    M·N
  - dW_proj = dy.T @ r                           big matmul
  - dr = dy @ W_proj                             big matmul (dr → HBM)
  - dz = dr * 2·relu(z)                          M·N (rd dr, rd z, wr dz)
  - x_hat = x * rms_inv * nw                     M·K (rd x, rd rms, rd nw, wr x_hat)
  - dW_fc = dz.T @ x_hat                         big matmul
  - dx_hat = dz @ W_fc                           big matmul (dx_hat → HBM)
  - RMSNorm bwd: dx_norm = f(dx_hat, x, ...)     M·K (rd dx_hat, rd x, wr dx_norm)
  - dx = dx_norm + dy                            M·K (rd dx_norm, rd dy, wr dx)
Total HBM: large amount of intermediate buffer round-trips
launches: ~8
```

Fused (4 Triton kernels):
```
A:  rd dy + z + W_proj, wr dz, atomic inner_buf  (no dr to HBM)
B:  rd dy + z, wr dW_proj                        (no r to HBM)
C:  rd dz + x + rms_inv, wr dW_fc               (no x_hat to HBM)
D:  rd dz + W_fc + x + rms_inv + dy + inner_buf, wr dx
                                                  (no dx_hat to HBM directly)
Total:
  - dr (M·N), r (M·N), x_hat (M·K), dx_hat (M·K), dx_norm (M·K) — none of
    these intermediates go to HBM
  - dx = dx_norm + dy fold saved (D writes the total directly)
launches: 4
```

Net savings (rough estimate):
- **HBM saved: ~3·M·K + 2·M·N** (5 intermediates stay out of HBM)
- d24: 3·M·K + 2·M·N = 9.4 MB + 50 MB = ~60 MB / 936 GB/s ≈ **64 μs**
- launches: ~8 → 4, **~4 saved** (~40-120 μs)

bwd is much more complex than fwd; the ledger is correspondingly less
precise — see §3.7 for measured numbers.

### 3.7 Performance reality

d24 (M=2048, N_fc=6144, K=1536, bf16 activation, fp32 master weight)
on RTX 3090, single-op micro-bench:

| Measurement | fused | native | Ratio |
|---|---|---|---|
| Forward only | ~2.6 ms | ~2.9 ms | **fused 1.12×** |
| Forward + Backward | ~8.2 ms | ~8.9 ms | **fused 1.09×** |

> ↑ These numbers were taken **before** the cast fusion landed, when
> fwd Step 1 was still cuBLAS + a standalone `.to()`. Folding the
> cast into Step 1 should raise the fwd ratio another 5-10% (saves
> 36 MB HBM round-trip + 1 launch), but the bench wasn't re-run; the
> table above is a conservative lower bound.

Other shapes (fwd + bwd ratios, pre-cast-fusion):

| Shape | fwd | f+b |
|---|---|---|
| M=2048, N_fc=6144, K=1536 (d24) | 1.12× | 1.09× |
| M=4096, N_fc=6144, K=1536 | 1.15× | 1.07× |
| M=2048, N_fc=8192, K=2048 | 1.10× | 1.08× |
| M=2048, N_fc=3072, K=768 | 1.35× | 1.24× |
| M=1024, N_fc=16384, K=4096 | 1.05× | 1.03× |

Observations:
- **Small shapes win the most** (fwd 1.35×, bwd 1.24×) — HBM /
  launch overhead is a bigger share, so fusion savings amplify
- **Large shapes win the least** (fwd 1.05×, bwd 1.03×) — matmul
  compute dominates, fusion's HBM saving is relatively small, and
  the Triton-vs-cuBLAS efficiency gap starts to show
- **bwd ratio slightly below fwd ratio** — bwd has 4 Triton matmuls,
  so the cuBLAS gap stacks; but after cast fusion fwd is also
  all-Triton and the gap should narrow. As §3.6 shows, in the
  fp32-master case the cast savings land mostly on fwd.

### 3.8 End-to-end landing

`fused_mlp` wins ~9% single-op fwd+bwd at d24 (micro-bench; higher
after cast fusion). Landing into nanchat training is via the
`NANOOPS_FUSED_MLP=1` environment variable; `nanoops/integration.py`'s
`patch_nanchat()` monkey-patches `nanchat.gpt.Block.forward` on the
mlp side:

```python
def _patched_block_forward(self, x, ve, cos_sin, window_size, kv_cache):
    x = x + self.attn(_orig_norm(x), ve, cos_sin, window_size, kv_cache)
    if kv_cache is not None or not x.is_cuda:
        return x + self.mlp(_orig_norm(x))          # CPU / kv-cache fallback
    B, T, C = x.shape
    x_2d = x.reshape(B * T, C).contiguous()
    y_2d = _fused_mlp(x_2d, self.mlp.c_fc.weight, self.mlp.c_proj.weight)
    return y_2d.reshape(B, T, C)
```

The fused block always uses nanchat's no-affine RMSNorm; `_orig_norm`
is the original `Block.norm`, captured in a module global at patch time.

**Single-op gain doesn't translate 1:1 to end-to-end gain**, but
`fused_mlp` is now wrapped as `torch.library.custom_op` (paired
fwd/bwd, plus `register_fake` + `register_autograd`). torch.compile
treats it as an opaque FX node — **no graph break, no FakeTensor
tracing into the Triton kernels**, and Inductor keeps fusing on both
sides of the wrapper.

> Before that: `torch.autograd.Function` was a dynamo black box —
> `.apply()` triggered a graph break, dynamo fell back to eager
> dispatch. We tried `@allow_in_graph` but that made dynamo replay
> the wrapper with FakeTensors, hitting Triton kernels' `.data_ptr()`
> and crashing. `custom_op` is PyTorch's official path for wrapping
> third-party / custom kernels, and it's the right fix.

d24 + B=1 end-to-end measurements (5-step mean from the same
checkpoint resume, 2× 3090):

| Path | dt (ms) | tok/sec | bf16_mfu (%) | vs baseline |
|---|---:|---:|---:|---:|
| baseline (no FUSED_MLP) | 67,175 | 15,610 | 52.49 | — |
| FUSED + `autograd.Function` (old) | 65,452 | 16,021 | 53.88 | +2.63% |
| FUSED + `custom_op` (current) | **65,038** | **16,124** | **54.22** | **+3.29%** |

Loss matches baseline to ~1e-4 across all three variants (same
checkpoint + same lr, kernel parity verified). fullgraph compile
also works straight through — `y / dx / dW_fc / dW_proj` diffs all
0.0 (bit-exact).

The remaining gap between single-op gain and end-to-end gain is:
1. **MLP is only ~50-55% of step time** — a single-op 1.09× implies
   ~4.5% end-to-end upper bound. custom_op gets ~3.3%, about 73% of
   that bound.
2. **Other overhead**: DDP all-reduce, optimizer step (Muon + AdamW),
   data load, Python control flow — none of these are touched by mlp
   fusion or affected by Inductor.
3. **CUDA Graph capture not yet wired** — locked configs satisfy the
   prerequisite (no autotune dispatch). The old fp32 IEEE branch is gone;
   remaining capture work is outside this MLP precision path. Another +1-2%
   possible.

But op-level 1.09× (higher after cast fusion) + end-to-end +3.3% is
real savings. At this point in production-grade kernel work, the
marginal optimization space is nearly drained — going faster means
switching paths (fp8, structured sparsity), tackling the attention
side (30-35% of step time), or pushing B=1 → B=2/4 (GEMM utilization
from 53% up to 70+%).

### 3.9 Takeaway

**Core patterns, ranked by performance impact**:

1. **Find fusion along the reduction axis** — elementwise ops and
   small reductions next to a matmul can ride in the matmul's
   register stage, saving HBM round-trips. The matmul's own
   compute-bound efficiency loss is smaller than what fusion saves.

2. **Move byproducts to the right kernel** — cross-kernel shared
   intermediates like `inner` are 10× faster computed in the kernel
   that already has the raw materials in registers (A, with dz and z
   already there) than in the kernel that needs the result but
   doesn't have the inputs (D).

3. **Algebraic identities are the key to fusion** —
   `Σ_k(dx_hat·x_hat) = Σ_n(dz·z)` rewrites D's K-reduction as A's
   N-reduction, which unlocks tensor-core-friendly tile sizes.

4. **Spend time on the dtype path** — bf16 throughout + fp32 acc
   demands precise judgment about when to promote vs cast.
   `tl.where(x>0, x, 0.0)` vs `tl.maximum(x, 0.0)`, int `2` vs float
   `2.0`, `dtype=tl.float32` on `tl.sum` — these details decide
   whether the register and HBM data is fp32 or bf16.

5. **dtype cast is itself a fusable byproduct** — under fp32 master
   + bf16 activation, a standalone `.to()` is one launch + one HBM
   round-trip; at d24 that's 36 MB ≈ ~75 μs, enough to give back
   cuBLAS's ~10-15% lead over Triton. Step 1 looks like an isolated
   big matmul but with cast folded into the load it becomes Triton's
   territory. The 3 bwd matmuls with fp32 master weights (A's
   W_proj, D's W_fc) use the same `weight.to(activation.dtype)`
   inline-cast pattern.

6. **dW output lands directly on master dtype** — the caller
   allocates dW with `dtype=W.dtype`, so the kernel's fp32 acc
   stores into a fp32 master buffer. The optimizer doesn't need to
   lift bf16 grads back to fp32. Paired with #5: the master /
   activation dtype mismatch is fully absorbed inside the Triton
   kernels.

7. **Compute the shared-mem budget along the dtype path** — autotune dispatch
   can still try over-budget candidates and pick a "fast" wrong result. Manual
   sweeps plus caller-locked configs remain the safer path.

8. **atomic_add is free on small target buffers** — L2-friendly,
   hardware atomic units handle contention. Simpler than
   scratchpad+reduce and not slower.
