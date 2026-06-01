# 第 3 章 —— `fused_mlp`：production-level、fwd+bwd 全 Triton

> 属于 [nanoops Triton Kernels](TRITON_zh.md)。English version: [TRITON_MLP.md](TRITON_MLP.md)。

---

第 2 章的 `FusedAddNorm` 是教学样本。这章是 nanchat mlp side 的**实际目标
fusion**——standard transformer mlp 块（pre-norm + linear + relu² + linear
+ outer residual）端到端 7 个 op，在我们这里被压缩成 **3 步 fwd + 4 步 bwd，
全 Triton**，整条 fwd/bwd 链路无 cuBLAS。fc_weight / proj_weight 在 nanchat
里是 fp32 master，所有 matmul kernel 在 load 时 inline cast 到 activation
dtype（bf16），不在 HBM 里物化 bf16 权重副本。

数学：
```
y = x + relu²(RMSNorm(x) @ W_fc.T) @ W_proj.T
```

API 签名：

```python
def fused_mlp(x, fc_weight, proj_weight, eps=1e-6) -> y
#   x、fc_weight、proj_weight 都必须是 contiguous CUDA tensors
#   RMSNorm 固定是无 affine 版本；Python 入口不再接收 norm_weight。
```

caller 在外面做 outer residual 的预求和（如果有），这个 block 只做
`y = x + mlp(norm(x))` 的标准 pattern。

为什么这个比第 2 章值得读：**第 2 章是 add+norm 这种纯 memory-bound 的
2-op fusion；第 3 章的核心是 _在 matmul 的 bwd 反向链路里塞 fusion_**。
matmul 本身是 compute-bound，但 bwd 的 dz/dW_proj/dW_fc/dx 每一步都
带 elementwise 或 reduction 的"副产物"——这些副产物是我们 fuse 的对象。

### 3.1 Kernel 布局总览

| 阶段 | Kernel | Grid | 干什么 |
|---|---|---|---|
| Fwd 0 | `_fused_mlp_rms_norm_fwd_kernel` | 1D over M | 无 affine RMSNorm 算 `x_hat` + 副产物 `rms_inv` |
| Fwd 1 | `_fused_mlp_fc_matmul_fwd_kernel` | 2D over (M, N_fc) | `z = x_hat @ W_fc.T`，W_fc 在 load 里 inline cast |
| Fwd 2 | `_fused_mlp_proj_residual_fwd_kernel` | 2D over (M, K_out) | relu² + c_proj + outer residual add → `y` |
| Bwd A | `_fused_mlp_dz_bwd_kernel` | 2D over (M, N_fc) | `dz` + 副产物 `inner_buf`（D 要用）|
| Bwd B | `_fused_mlp_dproj_weight_bwd_kernel` | 2D over (K_out, N_fc) | `dW_proj`（fp32 master 输出）|
| Bwd C | `_fused_mlp_dfc_weight_bwd_kernel` | 2D over (N_fc, K) | `dW_fc`（fp32 master 输出）|
| Bwd D | `_fused_mlp_dx_bwd_kernel` | 2D over (M, K) | `dx_hat` matmul + 无 affine RMSNorm bwd + outer residual fold → `dx` |

**fwd/bwd 全 Triton 是故意的**——nanchat 训练时 d24 shape (M=2048, N_fc=6144,
K=1536) 上，每个 matmul 都能跟相邻的 elementwise / weight cast / reduction
fuse 掉一次 HBM round-trip 或一次 launch；这种 saving 大于 Triton 自家
matmul 相对 cuBLAS 的 10-15% 效率劣势。Step 1 看起来是孤立的大 matmul，
但 fp32 master → bf16 activation 的 `.to()` cast 单独走一次（一次 launch +
36 MB HBM 写回读）就把 cuBLAS 的效率优势吃掉了，所以也用 Triton 把 cast
折进 load。详见 §3.4。

### 3.2 Forward

#### 3.2.1 Step 0 —— 无 affine RMSNorm

fwd 使用本文件内的小 `_fused_mlp_rms_norm_fwd_kernel`：

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

kernel 体里：
```python
x = tl.load(x_ptr + rows[:, None] * K + cols[None, :], ...)
rms_inv = rsqrt(sum(x * x) / K + eps)
x_hat = x * rms_inv[:, None]
tl.store(x_hat_ptr + ..., x_hat.to(x_hat_ptr.dtype.element_ty), ...)
tl.store(rms_inv_ptr + rows, rms_inv, ...)
```

这里固定是无 affine：nanchat 热路径的 RMSNorm 没有 learnable per-channel
scale，Python API 也不再接收 `norm_weight`。

#### 3.2.2 Step 1 —— `_fused_mlp_fc_matmul_fwd_kernel`：c_fc + inline weight cast

c_fc 本身是个孤立的大 matmul，没有相邻 elementwise 副产物可以 fuse 进
matmul 的 register stage。但 fc_weight 在 nanchat 里是 **fp32 master**，
activation x_hat 是 bf16，喂 cuBLAS 之前必须先 cast：

```python
# 朴素版（被取代）
fc_w_bf16 = fc_weight.to(x_hat.dtype)        # 单独 launch + 36 MB HBM 写
z = torch.matmul(x_hat, fc_w_bf16.t())       # cuBLAS bf16 matmul，~70% peak
```

那个 `.to()` 是独立 kernel：写 36 MB 到 HBM，下一个 kernel 再读回来。
d24 上这一来回大约 75 μs，正好把 cuBLAS 相对 Triton ~10-15% 的效率优势
吃掉。所以 Step 1 也写成 Triton：

```python
@triton.jit
def _fused_mlp_fc_matmul_fwd_kernel(x_ptr, w_ptr, z_ptr, M, N, K, ...):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        x_tile = tl.load(x_ptr + ...)                  # bf16
        w_tile = tl.load(w_ptr + ...)                  # fp32 native
        acc += tl.dot(x_tile, tl.trans(w_tile.to(x_tile.dtype)))
        #                                  ↑ cast 在 load 后、dot 前
        #                                    bf16 tile 只活在 register
    tl.store(z_ptr + ..., acc.to(z_ptr.dtype.element_ty), ...)
```

关键 pattern：**`w_tile.to(x_tile.dtype)` 在 register 里发生**。bf16 weight
tile 从未物化到 HBM，省掉 36 MB 写回读 + 一次 launch。

d24 manual sweep 锁了 `(BLOCK_M=256, BLOCK_N=64, BLOCK_K=32, nw=8, st=2)`，
单 kernel ~639 μs，比 cuBLAS+cast 的 ~654 μs 略快、比朴素 Triton
`(64,64,64,nw=4,st=3)` 的 ~1300 μs 快 2×。per-stage shared mem
`(256·32 + 64·32)·4 = 40 KB`，×2 stages = 80 KB，在 3090 SM 的 100 KB
预算内。

#### 3.2.3 Step 2 —— `_fused_mlp_proj_residual_fwd_kernel`

把 `relu²(z) @ W_proj.T + x` 三个 op 塞一个 Triton kernel 里：

```python
acc = tl.zeros((BLOCK_M, BLOCK_K_OUT), dtype=tl.float32)
for n_start in range(0, N, BLOCK_N):
    z = tl.load(...)                                # bf16 native（fwd Step 1 写出来的）
    relu_z = tl.where(z > 0.0, z, 0.0)              # bf16；tl.where 保持 x 的 dtype
    r = relu_z * relu_z                             # bf16 * bf16 = bf16
    proj_w = tl.load(...)                           # fp32 native（master weight）
    acc += tl.dot(r, tl.trans(proj_w).to(z.dtype))  # cast 折进 dot 前
    #                                ↑ bf16 weight tile 只活在 register

# Residual fold-in：acc 先 cast 回 bf16，再加 native dtype 的 residual
residual = tl.load(residual_ptr + offs, ...)         # bf16
y = acc.to(y_ptr.dtype.element_ty) + residual         # bf16
tl.store(y_ptr + offs, y, ...)
```

注意 patterns：
- **bf16 全程 + fp32 acc**：z/r 全 bf16 喂 tensor core，proj_w 是 fp32
  master、在 register 里 cast 到 bf16 再 dot，accumulator fp32 兜底精度。
  `tl.where(z > 0.0, z, 0.0)` 的字面量 `0.0` 被强制 coerce 到 z 的 dtype
  （不像 `tl.maximum(z, 0.0)` 会把 z promote 成 fp32），这是保 bf16 路径
  不破的关键。
- **inline weight cast 同 Step 1**：fp32 master 在 load 里 cast 到 z 的
  dtype，bf16 weight tile 不出 register。caller 不需要在外面预 cast
  proj_weight。
- **residual cast 推迟**：先 `acc.to(bf16)`、再 `+ residual(bf16)`，不是
  先 `residual.to(fp32)` 再 fp32 加。省一次 bf16→fp32 conversion，最后
  store 也少一次 cast。代价是最后那次加法在 bf16 而非 fp32——精度损失
  ~1e-3 / 元素，atol 兜得住。
- **d24 locked**：`(BLOCK_M=128, BLOCK_K_OUT=64, BLOCK_N=32, nw=8, st=2)`。

### 3.3 Backward —— 4 个 Triton kernel 全包

bwd 出 4 个梯度 tensor：`dz, dW_proj, dW_fc, dx`。这 4 个 reduction 轴
互相正交（A reduce K_out、B reduce M、C reduce M、D reduce N_fc），所以
**不可能塞进单 kernel**。但每一步都跟相邻 elementwise fuse 掉了一次 HBM
round-trip。

#### 3.3.1 Step A —— `_fused_mlp_dz_bwd_kernel`：matmul + relu² bwd + side-output

数学：
```
dr = dy @ W_proj                # matmul, reduce K_out
dz = 2·relu(z) · dr             # elementwise (relu² bwd)
inner_partial = Σ_n(dz·z) / norm_dim  ← 副产物，给 D
```

kernel 体精简版：
```python
dr = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for kp_start in range(0, K_out, BLOCK_K_OUT):
    dy = tl.load(...)                            # bf16
    proj_w = tl.load(...)                        # fp32 master
    dr += tl.dot(dy, proj_w.to(dy.dtype))        # cast 折进 dot 前

z = tl.load(...)                                              # bf16 native
relu_z = tl.where(z > 0.0, z, 0.0)
dz = dr.to(dz_ptr.dtype.element_ty) * 2 * relu_z              # bf16 throughout
tl.store(dz_ptr + ..., dz, ...)

# Side-output: per-tile partial inner, atomic_add into (M,) fp32 buffer
inner_partial = tl.sum(dz * z, axis=1, dtype=tl.float32) / K_out
tl.atomic_add(inner_buf_ptr + rows, inner_partial, mask=row_mask)
```

d24 locked: `(BLOCK_M=128, BLOCK_N=128, BLOCK_K_OUT=32, nw=8, st=2)`。

3 个值得提的 pattern：

**1. dr 不出 HBM**——matmul accumulator 出来后直接被 `* 2·relu(z)` 消费成
dz。如果让 PyTorch 走，就是 `dr = dy @ W_proj`（写 25 MB 到 HBM at d24）+
`dz = 2·relu(z)·dr`（再读这 25 MB 回来）。fused 全程 register。

**2. dz 全程 bf16**——`dr.to(bf16) * 2 * relu_z`。`2` 故意写成 int 字面量
（不是 `2.0`）以避免把 bf16 promote 到 fp32。要是 promote 了，下游 `tl.sum`
会拿到 fp32 输入，store 时需要额外 cast。

**3. 副产物 inner_partial**——`tl.sum(dz * z, axis=1, dtype=tl.float32)`，
然后 atomic_add 累进 `inner_buf[rows]`。3 个细节：

- **dtype=tl.float32 强制 accumulator**：dz 和 z 都是 bf16，bf16 累加 N=6144
  个 product 会爆精度（8-bit mantissa）。`dtype=` 让 sum 的累加器升 fp32，
  每个 bf16 product 升 fp32 再加，等价于 PyTorch 内部 promote。
- **除以 `K_out` 在这里、不在 D**：MLP 结构 `K_out == norm_dim`（forward
  assert `K_proj_out == K`），所以 A 用自己已有的 K_out 参数除即可。D 直接
  load `inner_buf` 不再 divide。除法在 atomic_add 前做也意味着累加的是更
  小数量级的值，fp32 rounding 更精细。
- **为什么 atomic_add 而不是 scratchpad+reduce**：详见 §3.4。

##### 关键代数 identity

D 的 RMSNorm bwd 公式需要：
```
inner[m] = (1/norm_dim) · Σ_k(g_eff[m,k] · y_norm[m,k])
```
其中 `g_eff = dx_hat · nw`，`y_norm = x · rms_inv`，`x_hat = y_norm · nw`。

如果让 D 自己算，per (m, k_tile) program 要做完整的 K-reduction，把 dx_hat
摊在 BM=4 的小 tile 上（要装 full-K 进 register）——tensor core 用不上，
慢 5×。

但 forward 里 `z[m,n] = Σ_k x_hat[m,k] · W_fc[n,k]`，bwd 里
`dx_hat[m,k] = Σ_n dz[m,n] · W_fc[n,k]`，把这俩 substitute 进 inner 的内
积：

```
Σ_k(dx_hat[m,k] · x_hat[m,k])
  = Σ_k (Σ_n dz[m,n] · W[n,k]) · x_hat[m,k]
  = Σ_n dz[m,n] · (Σ_k W[n,k] · x_hat[m,k])
  = Σ_n dz[m,n] · z[m,n]
```

——线性算子伴随性质 `⟨L*v, u⟩ = ⟨v, Lu⟩`，c_fc 的 transpose 让我们能用
N 维度算同一个 inner。A 在算 dz 的时候 dz 和 z 都在 register 里，多一行
`tl.sum(dz * z)` 几乎 0 成本。

##### 为什么 atomic_add（不是 scratchpad）

A 的 grid 是 `(M/BM, N/BN)`——同一 m_tile 被 N/BN = 48 个 program 切。
inner 需要把这 48 个 partial 沿 N 加起来。可选方案：

| 方案 | 开销 at d24 |
|---|---|
| **atomic_add（当前）** | ~10 μs；硬件 atomic，inner_buf 8 KB 全在 L2 |
| Scratchpad `(num_n_tiles, M)` + `torch.sum(dim=0)` | ~15 μs；多一个 buffer + 一次 reduce launch（bench 实测打平） |
| 塞进 D 里顺路算 | ~25-50 μs；D grid 是 `(M/BM, K/BK)`，m_tile 被 K 维 24× 复制，要嘛重复算要嘛 sync；而且 z 不是 D 的输入，要加 HBM 读 |

atomic_add 优势：dz/z 已经在 register、目标 buffer (M,) 全在 L2、不要额外
buffer 也不要额外 launch。**self-contained 在 kernel 内**。

#### 3.3.2 Step B —— `_fused_mlp_dproj_weight_bwd_kernel`：dy.T @ relu²(z)

```python
acc = tl.zeros((BLOCK_K_OUT, BLOCK_N), dtype=tl.float32)
for m_start in range(0, M, BLOCK_M):
    dy = tl.load(...)                                 # bf16
    z = tl.load(...)                                  # bf16
    relu_z = tl.where(z > 0.0, z, 0.0)
    r = relu_z * relu_z                               # r 重算，不从 HBM 读
    acc += tl.dot(tl.trans(dy), r)                    # bf16 @ bf16
tl.store(dW_proj_ptr + ..., acc.to(dW_proj_ptr.dtype.element_ty), ...)
#                              ↑ caller 用 W_proj.dtype 分配，所以这里 store 直接落到 fp32 master
```

跟 Step A 是「同一个 fwd op 的两个 bwd 输出」，但 reduction 轴不同（B
reduce M、A reduce K_out），所以分两个 kernel 而不是一个。

两个 fusion 同时发生：
- **r 在 register 重算**——A 已经把 dz 写出 HBM 了，但 r 本身没存（fwd
  也没存，z 才存了）。B 这里现场算 `relu²(z)`，省一个 M·N_fc 的 HBM
  round-trip。
- **dW_proj 直接落 fp32 master**——caller 用 `dtype=W_proj.dtype` 分配
  dW_proj，kernel 内 fp32 acc 直接 store 到 fp32 buffer，optimizer 不需要
  额外 `.to()` 把 bf16 grad 升回 fp32 master。

d24 locked: `(BLOCK_K_OUT=64, BLOCK_N=128, BLOCK_M=64, nw=4, st=2)`。注意
B 是这一组里唯一不需要 inline weight cast 的 matmul——dy 和 z 都是 bf16
（caller 直接是 bf16，不是 fp32 master），没东西可 cast。

#### 3.3.3 Step C —— `_fused_mlp_dfc_weight_bwd_kernel`：dz.T @ x_hat，x_hat 重算

```python
acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
for m_start in range(0, M, BLOCK_M):
    x = tl.load(x_ptr + ..., ...)               # bf16
    rms_inv = tl.load(rms_inv_ptr + ms, ...)    # fp32
    x_hat = x * rms_inv[:, None]                # fp32（auto-promote）

    dz_tile = tl.load(...)                                       # bf16
    acc += tl.dot(tl.trans(dz_tile), x_hat.to(dz_tile.dtype))    # bf16 @ bf16
tl.store(dW_fc_ptr + ..., acc.to(dW_fc_ptr.dtype.element_ty), ...)
#                            ↑ caller 用 W_fc.dtype 分配，dW_fc 直接落 fp32 master
```

**x_hat 在 GEMM inner loop 里现场重算**——不需要在 fwd 时把 x_hat 写到
HBM 给 bwd 用（forward 里 ctx 只存 `x` 和 `rms_inv`，x_hat 丢弃）。

这相当于跟「cuBLAS 版」对比：
```
[cuBLAS path]
x_hat = (x * rms_inv).contiguous()             # M·K HBM 写
dW_fc = dz.T @ x_hat                            # M·K HBM 读 + cuBLAS matmul
```

vs Triton fused：matmul 里直接 reconstruct，x_hat 不出 register。**省一次
M·K HBM 写 + 读**。代价是 Triton matmul 比 cuBLAS 慢 ~10-15%。d24 测下来
fused 净赢 ~30 μs。dW_fc 跟 B 同样直接落 fp32 master。

d24 locked: `(BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, nw=4, st=2)`。注意 C
的 matmul 输入是 `dz_tile (bf16)` 和 `x_hat (fp32 register)`，所以 cast 方向
是 `x_hat.to(bf16)`，不是 weight cast——但效果一样（bf16 tile 喂 tensor
core）。

#### 3.3.4 Step D —— `_fused_mlp_dx_bwd_kernel`：dx 全部来源汇总

x 在 forward 中出现两次：
```
y = x + mlp(norm(x))
       ↑     ↑
     outer  norm path
```

所以 dx 有两条贡献：
- **outer-residual path**：`dx ← dy`（直接 passthrough）
- **norm path**：`dx ← RMSNorm_bwd(dx_hat)`，`dx_hat ← dz @ W_fc`

D 把这两条全部塞进**一个 kernel**：

```python
dx_hat = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
for n_start in range(0, N_fc, BLOCK_N):
    dz_tile = tl.load(...)                                       # bf16
    W_fc_tile = tl.load(...)                                     # fp32 master
    dx_hat += tl.dot(dz_tile, W_fc_tile.to(dz_tile.dtype))       # cast 折进 dot 前

# RMSNorm bwd inline，用 A 已经算好的 inner
rms_inv = tl.load(rms_inv_ptr + rows, ...)       # fp32
inner = tl.load(inner_buf_ptr + rows, ...)       # fp32（A 已经 /norm_dim）
x = tl.load(x_ptr + offs, ...)                   # bf16
dy = tl.load(dy_ptr + offs, ...)                 # bf16 native（passthrough）

y_norm = x * rms_inv[:, None]                    # fp32（auto-promote）
g_eff = dx_hat

# 一行汇总：RMSNorm bwd norm path → cast bf16 → + dy
dx = (rms_inv[:, None] * (g_eff - y_norm * inner[:, None])).to(bf16) + dy
tl.store(dx_ptr + offs, dx, ...)
```

3 个 fusion 同时发生：

**1. dx_hat 不出 HBM**——matmul 出 fp32 register，立刻进 RMSNorm bwd 公
式被消费。对照 cuBLAS 路径：`dx_hat = dz @ W_fc`（M·K HBM 写）+ 后续 kernel
读回来用。fused 全程 register。**省 ~13 μs HBM round-trip at d24**。

> 安全前提：dW_fc（kernel C）用的是 x_hat 不是 dx_hat，所以 dx_hat 没必要
> materialize 出来给别的 kernel 用。

**2. inner 是 A 准备好的**——D 不做 K-reduction，整个 dx 公式纯 elementwise。
unlock 了用 BLOCK_M=64、BLOCK_K=64 这种 tensor core 友好的 tile（如果 D 自
己要做 K-reduction，BLOCK_M 被压到 4，tensor core 失效）。

**3. outer-residual 折进 store**——`+ dy` 在 kernel 内完成，不需要外面的
Python `dx_total = dx_norm + dy` 那一步 + 一次额外 HBM round-trip。

**D 的锁定配置 + shared-mem 预算**：bf16 路径 sweep winner 是
`(BLOCK_K=64, BLOCK_N=64, nw=4, st=2)`，现在无条件使用这一条路径。旧的
fp32 IEEE 分支已经删掉，MLP 只保留训练导向的 tensor-core 路径。

这是 d24 sweep 时第一波 autotune 全 kernel 撞出来的坑——autotune 会试图
跑所有候选，shared-mem 超的那些不报错、出错误结果、autotune 选了一个看
起来"最快"但是答案错的配置。后续放弃 autotune dispatch，改成 manual sweep
+ 按预算筛配置 + caller 锁死的方案，顺带也跟 CUDA Graph capture 兼容（autotune
dispatch 不能在 graph capture mode 下用）。dx kernel 的 BLOCK_M 固定为 64。

##### dx 公式表达式里的精度路径

```python
dx = (rms_inv[:, None] * (g_eff - y_norm * inner[:, None])).to(bf16) + dy
#     [    fp32     *      (fp32 - fp32 * fp32)        ]    bf16 + bf16
```

括号里全 fp32 算完，**cast 到 bf16 后再 + dy**。又是 §3.2.3 的 residual
defer 招——dy 不需要升 fp32，最后那次加法在 bf16 里完成。

### 3.4 设计权衡总结

| Op | 谁负责 | 为什么 |
|---|---|---|
| Fwd c_fc matmul (z = x_hat @ W_fc) | Triton (`_fused_mlp_fc_matmul_fwd_kernel`) | fp32→bf16 weight cast 折进 load，省 36 MB HBM 往返 + 1 launch，盖过 cuBLAS ~10-15% 效率优势 |
| Fwd relu² + c_proj + residual | Triton | 三 op fused，r 不出 HBM；proj_w 同样 inline cast |
| Fwd RMSNorm | Triton | 本地无 affine RMSNorm kernel 写 x_hat + rms_inv |
| Bwd dz (A) | Triton | matmul + relu² bwd + atomic_add 副产物，三 in one；proj_w inline cast |
| Bwd dW_proj (B) | Triton | r 重算 fused 进 matmul；输出直接落 fp32 master |
| Bwd dW_fc (C) | Triton | x_hat 重算 fused 进 matmul；输出直接落 fp32 master |
| Bwd dx (D) | Triton | dx_hat matmul + RMSNorm bwd + outer residual fold，**三 op in one**；W_fc inline cast |

**核心准则**：matmul 旁边有可 fuse 的副产物（elementwise / 小 reduction /
**dtype cast**）时，写 Triton 抵消 ~10-15% 效率劣势是值得的。fp32 master +
bf16 activation 这一组合下，**inline weight cast 本身就是一个值得 fuse 的
副产物**——单独跑的 `.to()` 是一次 launch + 一次 HBM 往返，d24 上够 cuBLAS
吃饱的效率优势。

dW 输出 dtype 跟 master weight 对齐（caller 用 `dtype=W.dtype` 分配），所以
optimizer 不需要再 `.to()` 把 grad 从 bf16 升回 fp32 master，bwd 直接落到
master 上。

### 3.5 数值精度路径

整个 bwd 全程**bf16 in register / fp32 in accumulator / bf16 in HBM**：

| 操作 | dtype | 原因 |
|---|---|---|
| HBM load: x, z, dy, dz | bf16 native | caller dtype |
| HBM load: W_fc, W_proj | fp32 native (master) | nanchat 用 fp32 master weight |
| HBM load: rms_inv, inner_buf | fp32 native | 精度敏感的副产物 |
| Weight cast `W_*.to(activation.dtype)` | bf16 (in register) | inline cast 在 dot 前，bf16 weight tile 不出 register |
| matmul accumulator (`acc`, `dx_hat`, `dr`) | fp32 | tensor core 默认 fp32 acc |
| `inner_partial = tl.sum(dz·z, dtype=fp32)` | fp32 | 显式累加器升 fp32 防止 bf16 mantissa 累加溢精度 |
| `bf16 * fp32` 形如 `x * rms_inv` | fp32（auto-promote） | Triton 标准 promotion 规则 |
| relu² bwd: `dr.to(bf16) * 2 * relu_z` | bf16 | int `2` 不触发 fp32 promote（vs `2.0` 会） |
| RMSNorm bwd 公式 `rms_inv·(g_eff - y_norm·inner)` | fp32 | 精度关键路径 |
| store: `dx`、`dz`、`y` | bf16 | caller dtype |
| store: `dW_fc`、`dW_proj` | fp32 (master) | 直接落 fp32 master，optimizer 不需要再升 |
| dx 末尾 `+ dy` | bf16 + bf16 | residual defer（接受 bf16 加法精度损失） |

bf16 activation routes 全程不踩 fp32 中转：load 进 register 是 bf16，喂
tensor core 是 bf16，最后 store 还是 bf16。fp32 只出现在：（1）matmul
accumulator；（2）weight 在 HBM 里的 native 表示（cast 到 bf16 在 register
里发生，HBM 永远不写 bf16 weight）；（3）RMSNorm 公式和 dW 输出（精度
敏感 / 直接进 optimizer）。

### 3.6 预期收益账本

按 d24 (M=2048, N_fc=6144, K=1536, bf16) 算。

#### Forward

Native（5 个独立 op + fp32→bf16 weight cast）：
```
RMSNorm:    read x (M·K) + write x_hat (M·K)              = 2·M·K
cast W_fc:  read W_fc fp32 (2·N·K) + write W_fc bf16 (N·K) = 3·N·K  ← 单独 launch
matmul:     read x_hat (M·K) + W_fc bf16 (N·K) + write z   = 2·M·K + N·K + M·N
relu²:      read z (M·N) + write r (M·N)                   = 2·M·N
cast W_proj:类似上面                                        = 3·N·K
matmul:     read r (M·N) + W_proj bf16 (K·N) + write mlp   = M·N + N·K + M·K
add:        read mlp (M·K) + x (M·K) + write y (M·K)       = 3·M·K
────────────────────────────────────────
合计 HBM:   8·M·K + 5·M·N + 8·N·K
launches:   7（5 个 op + 2 个 cast）
```

> fp32 master weight 是 nanchat 的实际场景；如果 weight 本来就是 bf16，
> 那两个 cast 不存在，账本回退到原始的 8·M·K + 5·M·N + 2·N·K + 5 launches。

Fused：
```
Step 0 (Triton):  read x (M·K) + write x_hat (M·K)               = 2·M·K
Step 1 (Triton _cast_matmul):
                  read x_hat (M·K) + W_fc fp32 (2·N·K)
                  + write z (M·N)                                 = 2·M·K + 2·N·K + M·N
                  （bf16 weight tile 只在 register，不出 HBM）
Step 2 (Triton):  read z (M·N) + W_proj fp32 (2·K·N)
                  + x (M·K) + write y (M·K)                       = M·N + 2·N·K + 2·M·K
────────────────────────────────────────
合计 HBM:   6·M·K + 2·M·N + 4·N·K
launches:   3
```

净收益（fp32 master 场景下）：
- **HBM 省 2·M·K + 3·M·N + 4·N·K**——r/mlp 不出 HBM；两个 bf16 weight 副本
  从 HBM 删掉
- d24: ≈ 6.3 MB + 75.5 MB + 75.5 MB ≈ **157 MB / 936 GB/s ≈ 168 μs HBM 时间**
- launch 数: 7 → 3，**省 4 次**（~40-120 μs）

（如果 weight 本来就是 bf16，weight cast 那两条不存在，HBM 节省回退到
2·M·K + 3·M·N ≈ 82 MB ≈ 87 μs，launch 5→3 省 2 次。）

#### Backward

Native（PyTorch 的 mlp bwd 链路展开，按生产实现估算）：
```
约 8 次 kernel launch：
  - relu² bwd                                    M·N
  - dW_proj = dy.T @ r                           大 matmul
  - dr = dy @ W_proj                             大 matmul（dr → HBM）
  - dz = dr * 2·relu(z)                          M·N（rd dr, rd z, wr dz）
  - x_hat = x * rms_inv * nw                     M·K（rd x, rd rms, rd nw, wr x_hat）
  - dW_fc = dz.T @ x_hat                         大 matmul
  - dx_hat = dz @ W_fc                           大 matmul（dx_hat → HBM）
  - RMSNorm bwd: dx_norm = f(dx_hat, x, ...)     M·K（rd dx_hat, rd x, wr dx_norm）
  - dx = dx_norm + dy                            M·K（rd dx_norm, rd dy, wr dx）
合计 HBM: 大量中间 buffer round-trip
launches: ~8
```

Fused（4 个 Triton kernel）：
```
A:  rd dy + z + W_proj, wr dz, atomic inner_buf  (no dr to HBM)
B:  rd dy + z, wr dW_proj                        (no r to HBM)
C:  rd dz + x + rms_inv, wr dW_fc               (no x_hat to HBM)
D:  rd dz + W_fc + x + rms_inv + dy + inner_buf, wr dx
                                                  (no dx_hat to HBM directly)
合计:
  - 省了 dr (M·N)、r (M·N)、x_hat (M·K)、dx_hat (M·K)、dx_norm (M·K) 这些
    中间 buffer 的 HBM round-trip
  - 省了 dx = dx_norm + dy 的最后 fold（D 直接折进 dx store）
launches: 4
```

净收益（粗略估算）：
- **HBM 省 ~3·M·K + 2·M·N**（5 个中间 buffer 不出 HBM）
- d24: 3·M·K + 2·M·N = 9.4 MB + 50 MB = ~60 MB / 936 GB/s ≈ **64 μs**
- launch 数：~8 → 4，**省 ~4 次**（~40-120 μs）

bwd 比 fwd 复杂得多，账本也更不精确——具体看后面性能现实。

### 3.7 性能现实

d24 (M=2048, N_fc=6144, K=1536, bf16 activation, fp32 master weight)
在 RTX 3090 上，单 op micro-bench：

| 测量 | fused | native | 对比 |
|---|---|---|---|
| Forward only | ~2.6 ms | ~2.9 ms | **fused 1.12×** |
| Forward + Backward | ~8.2 ms | ~8.9 ms | **fused 1.09×** |

> ↑ 这两组数字是 cast fusion **之前**测的；当时 fwd Step 1 还是 cuBLAS +
> 独立 `.to()`。把 cast 折进 Step 1 之后，fwd ratio 大致再涨 5-10%（省掉
> 36 MB HBM 往返 + 1 launch），但没重测；上面的数字算 conservative
> 下限。

其他 shape（fwd + bwd ratio，pre-cast-fusion）：

| shape | fwd | f+b |
|---|---|---|
| M=2048, N_fc=6144, K=1536 (d24) | 1.12× | 1.09× |
| M=4096, N_fc=6144, K=1536 | 1.15× | 1.07× |
| M=2048, N_fc=8192, K=2048 | 1.10× | 1.08× |
| M=2048, N_fc=3072, K=768 | 1.35× | 1.24× |
| M=1024, N_fc=16384, K=4096 | 1.05× | 1.03× |

观察：
- **小 shape 收益最大**（fwd 1.35×, bwd 1.24×）——HBM/launch overhead 占
  比高，fusion 收益相对放大
- **大 shape 收益最小**（fwd 1.05×, bwd 1.03×）——matmul compute 主导，
  fusion 的 HBM 节省相对小；Triton vs cuBLAS 的效率差也开始显现
- **bwd ratio 略小于 fwd ratio**——bwd 4 个 matmul 都是 Triton，跟 cuBLAS
  的效率差是叠加的；但 cast fusion 之后 fwd 也全 Triton，这个 gap 会缩小。
  实际看 §3.6 的账本，cast 在 fp32 master 场景下省的 HBM 主要落在 fwd 上。

### 3.8 End-to-end 落地

`fused_mlp` 在 d24 上单 op fwd+bwd 净赢 ~9%（micro-bench；cast fusion 后
更高）。落地到 nanchat 训练靠 `NANOOPS_FUSED_MLP=1` 环境变量，由
`nanoops/integration.py` 在 `patch_nanchat()` 时 monkey-patch 掉
`nanchat.gpt.Block.forward` 的 mlp side：

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

fused block 固定使用 nanchat 的无 affine RMSNorm；`_orig_norm` 是原
`Block.norm`，被捕获在 module global 里。

**单 op 增益不完全等于 end-to-end 增益**，但 `fused_mlp` 现在
封装成 `torch.library.custom_op`（fwd / bwd 各一个，配 `register_fake`
+ `register_autograd`），torch.compile 能把它当成一个 opaque FX 节点，
**不再 graph-break、不再 trace 进 Triton kernel**，Inductor 继续在
wrapper 两侧做 cross-op fusion。

> 之前用 `torch.autograd.Function` 时是 dynamo 黑盒——`.apply()` 会
> 触发 graph-break，dynamo 退回 eager dispatch；试过 `@allow_in_graph`
> 但那会让 dynamo 用 FakeTensor 重放 wrapper，撞到 Triton kernel
> 的 `.data_ptr()` 直接挂掉。`custom_op` 是 PyTorch 给"包第三方 / 自
> 定义 kernel"准备的官方路径，正好对症。

d24 + B=1 end-to-end 实测（同 checkpoint resume 5 步均值，3090 ×2）：

| 路径 | dt (ms) | tok/sec | bf16_mfu (%) | vs baseline |
|---|---:|---:|---:|---:|
| baseline（不开 FUSED_MLP） | 67,175 | 15,610 | 52.49 | — |
| FUSED + `autograd.Function`（旧） | 65,452 | 16,021 | 53.88 | +2.63% |
| FUSED + `custom_op`（现） | **65,038** | **16,124** | **54.22** | **+3.29%** |

每个版本的 loss 跟 baseline 都在 ~1e-4 量级内对得上（同 checkpoint
+ 相同 lr，kernel parity 验证过）。fullgraph compile 也直接 OK，
`y / dx / dW_fc / dW_proj` 差全是 0.0（bit-exact）。

剩下的兑现差是因为：
1. **MLP 只占 step time ~50-55%**——单 op 1.09× 端到端理论上限 ~4.5%，
   custom_op 拿到 ~3.3%，已经接近这个上限的 73%。
2. **其他 overhead**：DDP all-reduce、optimizer step（Muon + AdamW）、
   data load、Python 控制流——这些跟 mlp fusion 无关，不会被 inductor
   消掉。
3. **CUDA Graph capture 还没接**——locked configs 已经具备前提（不用
   autotune dispatch）。旧的 fp32 IEEE 分支已经删掉，剩余 capture 工作
   不在这个 MLP 精度分支上。再 +1-2% 可能。

但 op-level 1.09×（cast fusion 后更高）+ end-to-end +3.3% 是真实节省。
production-grade kernel 写到这个程度，主要的边际优化空间已经枯竭——
再快需要换路（fp8、structured sparsity）、动 attention 部分（占 step time
30-35%）、或者把 B=1 → B=2/4（GEMM 利用率从 53% 推到 70+%）。

### 3.9 Takeaway

**核心 patterns 总结**（按对性能影响的大小）：

1. **算子合并要在 reduction 轴上下功夫**——matmul 周围的 elementwise
   和小 reduction 可以塞进 matmul 的 register stage 里，省 HBM round-trip。
   matmul 本身（compute-bound）的效率劣势小于 fusion 节省。

2. **副产物挪到合适的 kernel 里**——inner 这种 cross-kernel 共享的中间
   量，放到「已经有 dz 和 z 在 register 的 kernel」里算（A），比放到「要
   用 inner 但没原料的 kernel」里算（D）快 10×。

3. **代数 identity 是 fusion 的钥匙**——`Σ_k(dx_hat·x_hat) = Σ_n(dz·z)`
   把 D 里的 K-reduction 换成 A 里的 N-reduction，unlock 了 tensor core 友
   好的 tile size。

4. **dtype 路径要精打细算**——bf16 全程 + fp32 accumulator + 精确知道
   什么时候 promote / 什么时候 cast。`tl.where(x>0, x, 0.0)` vs
   `tl.maximum(x, 0.0)`、int `2` vs float `2.0`、`dtype=tl.float32` on
   `tl.sum`——这些细节决定 register / HBM 是 fp32 还是 bf16。

5. **dtype cast 也是值得 fuse 的副产物**——fp32 master + bf16 activation
   组合下，单独的 `.to()` 是一次 launch + 一次 HBM 往返，d24 上 36 MB =
   ~75 μs，足够吃掉 cuBLAS 比 Triton 快的那 10-15%。Step 1 看上去是孤立
   的大 matmul，把 cast 折进 load 就变成 Triton 反超的场景。bwd 那 3 个
   带 fp32 master weight 的 matmul（A 的 W_proj、D 的 W_fc）也都用同样
   的 `weight.to(activation.dtype)` inline cast 写法。

6. **dW 输出直接落 master dtype**——caller 用 `dtype=W.dtype` 分配 dW
   buffer，kernel 的 fp32 accumulator 直接 store 到 fp32 master buffer，
   optimizer 不需要再 `.to()` 把 bf16 grad 升回 fp32。和 #5 是一对：把
   master/activation dtype 不匹配的代价全部塞进 Triton kernel 内联。

7. **shared-mem 预算要按 dtype 路径算**——autotune dispatch 仍可能试到
   超预算候选并挑到"最快"但错误的结果，所以 manual sweep + caller 锁配置
   仍然是更稳的路径。

8. **atomic_add 在 small target buffer 上是免费的**——L2 友好，硬件
   atomic unit 处理 contention。比 scratchpad+reduce 简单且不输。
