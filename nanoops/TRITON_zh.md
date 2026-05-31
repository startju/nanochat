# nanoops Triton Kernels

面向 **Triton 初学者**——介绍在 **RTX 3090 架构下最优 fusion** 怎么做。
读完之后你能理解 `nanoops/triton_*.py` 里每个 kernel 的设计取舍：
block size 是怎么按 3090 真实预算选出来的、fusion 边界为什么落在那里、
什么留 register / 什么 save ctx / 什么 backward 重算——全部都是对着芯片
的 compute/bandwidth 比例算出来的，不是凭感觉。

文档从最底层的硬件数字（SM 数、shared mem、register file）出发，一层
层往上推到 fused kernel 设计。读完后回头看按功能拆分的 Triton 模块里
任何一个 kernel，你都能讲清楚每个选择背后的理由。

> English version: [TRITON.md](TRITON.md)

---

## 第 1 章 —— 目标硬件：RTX 3090 (Ampere SM_86 / GA102)

下面每一个 block size 选择、每一次"shared memory 装不装得下"判断、每一个
"为什么这是独立 kernel 不是一个大 fused kernel"的取舍——**全部都是对着
这些参数定的**。换其他 GPU（4090 / A100 / H100 / 消费级 Ada）跑大部分
kernel 还能用，但 tile size 多半已经不是最优了。

### 整 chip 总览

| 项                  | 数值                  |
| ------------------- | --------------------- |
| **SM 数**           | **82**                |
| 总 FP32 cores       | 82 × 128 = 10,496     |
| 总 Tensor cores     | 82 × 4 = 328          |
| 总 register file    | 82 × 64K = 5.4 M regs |
| 总 shared mem       | 82 × 100 KB = 8.2 MB  |
| **显存**            | **24 GB GDDR6X**      |
| HBM 带宽            | **936 GB/s**          |
| Compute capability  | **8.6**               |

### 每 SM 资源

| 资源                     | 数值              | 说明                                  |
| ------------------------ | ----------------- | ------------------------------------- |
| L1 / Shared memory      | 128 KB combined   | 零和分配：`shared_carveout + L1 = 128 KB`；sm_86 允许的 shared 值是 `{0, 8, 16, 32, 64, 100} KB`——选 shared 100 KB 时 L1 只剩 28 KB |
| Shared mem **每 SM 总量** | **100 KB**       | 最大 carveout = 共住 block 共享的整池 |
| Shared mem **每 block 上限** | **100 KB**     | 单个 block 能申请的硬上限——等于每 SM 池子总大小（所以单个 block 可以独占整池，但代价是 blocks/SM = 1） |
| Registers / SM          | 65,536 × 32-bit   | = 256 KB；per-thread 分配，SM 上所有活跃 thread 共享这个池 → **既限活跃 thread 数也限 blocks/SM**（通过 thread 换算） |
| Max threads / SM        | 1,536             | = 48 warps；卡 blocks/SM 为 `1536 / threads_per_block` |
| Max blocks / SM         | 16                | 不管资源够不够，硬件 block 共住上限 |
| Tensor cores            | 4 (3rd gen)       | bf16 / fp16 / tf32 / int8             |
| FP32 cores              | 128               |                                       |
| Warp size               | 32 threads        |                                       |

### 每 Block 限制

（NVIDIA 全名叫 "thread block"，但 CUDA API（`blockDim`、`blockIdx`、
`<<<grid, block>>>`）和本文其余地方都简称 "**block**"。两个是同一个概念。）

| 限制                | 数值       |
| ------------------- | ---------- |
| Max threads          | 1024 (32 warps) |
| Max shared memory    | **100 KB** |
| Max registers/thread | 255（超了 spill 到 local memory，慢）|

### Compile-time vs runtime：什么时候 frozen

理解本章关键的一条事实：**几乎所有影响 occupancy 的量都是编译期 frozen
的**，不是 launch 时定的。

| 量                          | 何时定           | 谁定                          |
| --------------------------- | ---------------- | ----------------------------- |
| `threads_per_block`         | **编译期**       | 用户（Triton 通过 `num_warps × 32`，CUDA C++ 通过 `<<<grid, block>>>`） |
| `reg/thread`                | **编译期**       | 编译器（Triton / NVCC）静态分析 kernel 后决定 |
| `shared_mem / block`        | **编译期**       | 编译器（静态算 `tl.load` buffer + 累加器 + `num_stages` pipeline 深度） |
| `grid_dim` (block 总数)      | runtime          | 用户 —— `kernel[grid](...)` |
| `blocks/SM`                 | runtime          | 硬件（GigaThread Engine）；精确公式见下文 |
| 每个 block 落哪个 SM         | runtime          | 硬件（GigaThread Engine） |

含义：

- 一个 Triton kernel 定义如果用 `triton.autotune` 扫 `num_warps` /
  `num_stages` / `BLOCK_*`，**会编译出 N 份独立 binary**——每份各有自己
  frozen 的 `reg/thread` 和 `shared/block`。
- `reg/thread` **不是用户直接挑的**——只能通过写更简单 / 复杂 / pipeline
  深的 kernel 间接影响。
- 编译完后，每个 kernel 的 occupancy **是确定的**——硬件不会 runtime 重新
  调资源分配。

GigaThread Engine 用的精确公式（以 warp 为粒度——**warp 才是真正的调度
单位，不是 thread**）：

```
warps_per_block = ⌈threads_per_block / 32⌉
regs_per_warp   = ⌈(32 · reg_per_thread) / 256⌉ · 256       # 256 对齐
regs_per_block  = warps_per_block · regs_per_warp

blocks/SM = min(
    16,                                                   # ① 硬件 block 上限
    ⌊48 / warps_per_block⌋,                               # ② 每 SM warp 上限（= 1536 thread）
    ⌊100 KB / shared_per_block⌋,                          # ③ Shared mem 池
    ⌊65,536 / regs_per_block⌋,                            # ④ Register file 池
)

resident_warps = blocks/SM · warps_per_block
occupancy      = resident_warps / 48
```

几个细节：
- **`⌈ / 32⌉`（warp 取整）** —— `threads_per_block` 不是 32 倍数时，最后
  一个 warp 仍占整 32-lane 资源（有 inactive lane）。
- **`⌈ / 256⌉ · 256`（register 分配粒度）** —— Ampere 上 register 按
  warp 分配，且每 warp 按 256 整数倍 round up。所以 32 thread 每个用 33
  register 不是 `32·33 = 1,056`，而是 `⌈1,056/256⌉·256 = 1,280` ——
  浪费 224 个 register slot。`reg_per_thread` 是 8 的倍数时正好对齐 256
  per-warp 边界，浪费为 0；编译器选了奇数 register 数才看得到。
- **`⌊ ⌋`（block count 整数）** —— SM 不能装 1.5 个 block。
- **② 本质是 warp 上限** —— 48-warp scheduler 是硬件硬约束；"1536 thread"
  只是 `48 × 32` 的换算。

### FMA 和 MMA —— 两种基础算子

下面的吞吐表都用 **FMA / MMA / FLOPs** 计数，先把这俩定义清楚。

**FMA —— Fused Multiply-Add（标量）**
- CUDA core 的基础 op：`d = a * b + c`
- "Fused" = 最后只做**一次 rounding**（而不是两次），数值更准
- 1 个 FMA = **2 FLOPs**（1 乘 + 1 加）
- 所有现代 CPU/GPU FPU 都 1 周期完成 FMA；分两步算 `a*b` 再 `+c` 速度
  一样但数值更糟

**MMA —— Matrix Multiply-Accumulate（矩阵）**
- Tensor core 的基础 op：`C = A @ B + C`，A/B/C 都是小矩阵
- Ampere 3rd-gen Tensor core 形状是 **16×16×16**（一次 MMA 做一个 16×16
  矩阵 × 16×16 矩阵，累加到 16×16 结果上）
- 这样 1 个 MMA = 16 × 16 × 16 = **4,096 个 multiply-add = 8,192 FLOPs**
- 通过一条 warp 级 `mma.sync` PTX 指令发射。这 4,096 个 multiply-add 在
  Tensor core 上 pipeline 跨多周期完成——**每 SM 持续吞吐 1,024 BF16
  FLOPs/cycle**（见下面峰值算力表），即每个 Tensor core 持续 256 BF16
  FLOPs/cycle，所以一条 16×16×16 MMA（8,192 FLOPs）在一个 Tensor core
  上摊销 **~32 周期**。

也就是说**一次 MMA 等于 4,096 次 FMA 的工作量**。这就是为什么架构要拆开
：少数几个 Tensor core 跑 MMA >> 多个 CUDA core 跑 FMA。

### RTX 3090 峰值算力

Boost clock 1695 MHz，82 个 SM。

| 精度 / 单元                    | 每 SM 吞吐                  | 整 chip 峰值        |
| ------------------------------ | --------------------------- | ------------------- |
| FP32 (CUDA cores)              | 128 cores × 2 FLOPs/cycle  | **35.6 TFLOPS**     |
| TF32 (Tensor cores)            | 512 FLOPs/cycle             | **71 TFLOPS**       |
| **FP16 / BF16 (Tensor cores)** | 1024 FLOPs/cycle            | **142 TFLOPS**      |
| INT8 (Tensor cores)            | 2048 ops/cycle              | 284 TOPS            |

**Memory 侧**：HBM 带宽 936 GB/s。bf16 算力 / 带宽 = 142 TFLOPS / 936
GB/s = **每 byte 152 FLOPs**。一个 op 的 arithmetic intensity（每 byte
FLOPs）**低于这个值就是带宽 bound，高于就是算力 bound**。

**Arithmetic intensity 怎么算**（matmul `C = A @ B`，shape
`(M, K) @ (K, N) → (M, N)`，全 bf16）：

- **FLOPs** = `2·M·N·K`——每个 output 元素需要 K 次 multiply-add，每个
  multiply-add = 2 FLOPs
- **Bytes** = `2·(M·K + K·N + M·N)`——读 A + 读 B + 写 C，bf16 每元素 2 字节
- **AI** = FLOPs / Bytes = `M·N·K / (M·K + K·N + M·N)`（2 互相消掉）

跟 152 FLOPs/byte 分水岭（= 142 TFLOPS / 936 GB/s）比，**高于**是算力
bound，**低于**是带宽 bound。

nanchat d24 训练时各 matmul (M=2048, K=1536, N 不同)：

| Matmul 位置                | Shape (M, K, N)       | AI (FLOPs/byte) |
| -------------------------- | --------------------- | --------------- |
| c_q / c_k / c_v / attn c_proj | (2048, 1536, 1536) | **558**         |
| MLP c_fc (D → 4D)          | (2048, 1536, 6144)    | **770**         |
| MLP c_proj (4D → D)        | (2048, 6144, 1536)    | **770**         |

全部远超 152 FLOPs/byte 分水岭 → **算力 bound**。

**RMSNorm 和其他 elementwise / reduction op 严重 bandwidth-bound** ——
比 break-even 差 100×。对一行 D 个元素：

- FLOPs ≈ `4·D`（`mean(x²)` ≈ 2D，再 `x · rms_inv · weight` = 2D）
- Bytes (bf16) ≈ `4·D`（**fused** kernel 读 x **一次** = 2D，写 y = 2D；
  weight 整 kernel 共享 `D`，摊薄忽略）。朴素 2-pass 实现要读 x 两遍 →
  6D bytes，AI 降到 0.67 FLOPs/byte。Triton/CUDA kernel 把整行 x 持
  register 跨两 pass，HBM 只读一次。
- **AI ≈ 1 FLOPs/byte** —— 比 152 break-even 差约 **100×**，重度带宽 bound

各 op AI 对比（训练 shape）：

| Op                          | AI (FLOPs/byte) | 性质        |
| --------------------------- | --------------- | ----------- |
| MLP / QKV matmul            | 558 – 770       | 算力 bound  |
| SDPA (B=1, L=2048)          | ~114 naive / ~1024 Flash（见下方 SDPA 段） | 带宽 → 算力 |
| **RMSNorm**                 | **~1**          | 带宽 bound  |
| Elementwise (add, relu)     | ~0.5            | 带宽 bound  |
| Memcpy H2D / D2H            | 0               | 带宽 bound  |

**这就是 fusion 设计的根源**：独立的 RMSNorm kernel 做 ~4·M·D 字节 HBM
流量，但算力收益接近 0。把 norm fuse 进邻接的 matmul kernel
（mlp 走 `fused_mlp`）能让归一化的中间值留在 register 里，
**省下 4·M·D 字节 HBM 流量**（norm 输出写回 + matmul 输入再读）——带宽
bound 的 op 上**纯赚**。attn 侧 `norm_qkv_projection` 目前先用共享
RMSNorm kernel 物化 `x_hat`，再做 Q/K/V projection；这个版本在 d24
bench 里略快。

**SDPA：为什么 Flash Attention 存在。** 同样的 AI 视角下，SDPA 有
**两个** AI 数字，取决于 `(L, L)` 的 P 矩阵是否物化到 HBM。

nanchat d24 (B=1, H=12, L=2048, D_head=128) forward attention
（两个 matmul：`Q@K^T` 和 `attn@V`，加 softmax）：

| 实现                                    | 总 FLOPs | 总 Bytes | AI                |
| --------------------------------------- | -------- | -------- | ----------------- |
| Naive SDPA（物化 P 到 HBM）             | 25.8 G   | ~226 MB  | **~114 FLOPs/byte** (偏带宽)  |
| Flash SDPA（P 留 register）             | 25.8 G   | ~25 MB   | **~1024 FLOPs/byte** (算力 bound) |

字节拆解：这个 scale 下 `Q + K + V + O ≈ 25 MB`（4 个 `B·H·L·D =
12·2048·128` bf16 tensor）。P 矩阵是 `B·H·L·L = 100 MB`，naive SDPA
**写 P 然后再从 HBM 读回**（~200 MB），所以 naive 合计 ~225 MB，而
Flash 只 ~25 MB。Flash 用 online softmax + tile streaming 让 P 留
register，把 SDPA 从带宽 bound (~114 < 152 break-even) **翻成算力
bound** (~1024 >> break-even)。

Flash Attention 比 naive SDPA 快 2-4× 的根源——bytes 少 ~9×，FLOPs 也
有变化但幅度小：

- **Forward FLOPs**：几乎相同（Flash 多 ~5%，是 online softmax 的 rescale
  bookkeeping）。
- **Backward FLOPs**：Flash **多 ~33%**——backward 时不存 P，而是 `Q@K^T`
  现算一遍重建 P。典型的 **FLOPs 换 memory** trade。

净效果：Flash backward 多算 ~30% 但 HBM 带宽省 ~10×，wall time 仍少 2-4×
（在带宽 bound 阶段，省带宽远比多算 FLOPs 重要）。Sliding window attention
把 bytes 和 FLOPs 都按比例缩小（band size `W` 代替全 `L`），但 Flash-vs-
naive 的比值不变。

### Tensor cores vs CUDA (FP32) cores

**完全不同的两套硬件单元**，做不同工作。理解切分方式决定你怎么算 kernel
吞吐。

| 单元              | 每 SM | 每周期做啥（持续吞吐）                       | 每 SM 总吞吐                |
| ----------------- | ----- | -------------------------------------------- | --------------------------- |
| **FP32 cores**    | 128   | 每核 1 FMA/cycle = 2 FLOPs/cycle             | **256 FP32 FLOPs/cycle**    |
| **Tensor cores**  | 4     | 每核持续 256 BF16 FLOPs/cycle（一条 16×16×16 MMA 摊销 ~32 周期） | **1024 BF16 FLOPs/cycle**   |

**4 个 Tensor cores 的 bf16 算力是 128 FP32 cores 的 fp32 算力的 4×**
（1024 / 256）。切 bf16/fp16 + 用 `tl.dot` 才能解锁那 4×。

**为什么每 SM 只有 4 个 Tensor cores：**

1. **硅面积代价大**。一个 Tensor core ≈ 几十个 FP32 core 的晶体管数。
   4 个已经吃掉 SM 很大一块面积。
2. **调度上 perfect 1:1**。Ampere SM 有 **4 个 warp scheduler**；
   每周期每个 scheduler 发 1 条 `mma.sync` 给 1 个 Tensor core。
   4 schedulers × 4 Tensor cores 完美对应——加第 5 个也没人发指令给它。
3. **数据带宽喂不饱更多**。peak 下（每 SM 1024 BF16 FLOPs/cycle = 512
   multiply-adds/cycle），即便算上 register 里 tile 复用，4 个 Tensor
   core 一起仍要从 shared memory 拉 ~1 KB/cycle 量级的 operand 数据
   ——已经吃掉每 SM SMEM port 带宽的不小一块。

### Warps、warp schedulers 和 cores 的关系

**warp** = 32 个线程，**GPU 调度的基本单位**。

```
                    SM（RTX 3090 上 82 个之一）
   ┌──────────────────────────────────────────────────────────┐
   │  4 warp schedulers（round-robin 调度）                    │
   │     │                                                     │
   │     ├──► 每周期发一条 warp 指令 ─►                        │
   │     │                                                     │
   │     │      ┌─────────────────────────────────────┐       │
   │     ├─────►│ 128 FP32 cores（= 4 组 × 32 lanes） │       │
   │     │      │ scalar 操作: fma, sin, add, ld, st… │       │
   │     │      └─────────────────────────────────────┘       │
   │     │                                                     │
   │     │      ┌─────────────────────────────────────┐       │
   │     └─────►│ 4 Tensor cores                      │       │
   │            │ `mma.sync` → 矩阵 multiply-accumulate│      │
   │            └─────────────────────────────────────┘       │
   │                                                           │
   │  Register file: 65,536 × 32-bit（全 warp 共享）           │
   │  Shared memory: 100 KB（每 thread block）                 │
   └──────────────────────────────────────────────────────────┘
```

**关键关系：**

- **1 个 warp 1 次发 1 条指令**。要么发给 FP32 cores（32 lane × 1 fp32
  op = 一周期 32 个 op，每 thread 1 个），要么发给 Tensor core（整 warp
  协同生成 1 个 matrix `mma`）。
- **`mma.sync` 指令是 warp 级**，不是 thread 级——warp 内 32 个 thread
  全部参与。各自的 per-thread register fragment 拼成 Tensor core 要读的
  (16, 16) 矩阵 tile。
- 每 SM 可以**最多容纳 48 个 warp**（= 1536 threads）。Scheduler 在它们
  之间挑：一个 warp 卡在 load 等内存时，调度器切到别的 warp 跑。这就是
  **latency hiding**——GPU 不需要 CPU 那种花哨的 out-of-order 执行。
- **Block size 影响 occupancy**。一个 block 256 threads = 8 warps。
  100 KB shared memory per block → 一个 SM 只装得下 1 block → 48 warp
  容量里只有 8 warp 活跃 → occupancy 低 → 能 hide latency 的 warp 少 →
  吞吐掉。把 tile 调小（少占 shared mem）让 2-3 个 block 共驻同 SM 能
  提升 occupancy。

**Triton 跟 CUDA C++ 在 warp 控制上的区别：**

- **CUDA C++**：手动调 `__syncwarp()`、管 `mma::sync`、选 fragment layout。
- **Triton**：把 warp 隐藏。用户只看到 `tl.program_id`（block 级）和
  `tl.arange`（编译时 vector）。Triton 编译器决定怎么把你写的 vector op
  映射到 32-thread warp、哪个 lane 读哪个元素。用户能调的只有：
  - `num_warps=4`（默认）—— 每 block 几个 warp。大 = 单 block 内并行高
    但能共驻的 block 少。
  - `num_stages=2` —— `tl.load` / `tl.dot` 的 pipelining 深度。

### 这些参数对我们 kernel 意味着什么

- **Shared memory 是最紧的瓶颈**。我们大部分 kernel 的 tile 尺寸（BLOCK_M,
  BLOCK_N, BLOCK_K）都是选成**合并工作集**（input tiles + accumulator）
  ≤ 60-80 KB per block，留点 headroom 给 runtime，**避免**一个 SM 只能跑
  一个 block 的低 occupancy。

- **3090 的 shared memory 大约是 A100 (164 KB) 的一半、H100 (228 KB) 的
  三分之一**。所以 Flash Attention 在 3090 上**必须用更小的 tile**——很多
  公开的 Flash kernel 数据是 H100 上跑出来的，那些 Q/K/V tile 直接搬到
  3090 上要么 spill 要么编译都过不了。

- **82 个 SM 是塞满 GPU 的最小 grid 大小**。launch 少于这个就有 SM 闲着。
  对于 per-row reduction 类 kernel（RMSNorm, softmax, fused_add_norm），
  我们沿 M (batch × seq) 维度拆 tile，因为训练时 M 通常 ≥ 2048——并行度
  绰绰有余。

- **没有 FA3 那套 TMA / async-copy 硬件**。Hopper 专属的技巧（TMA、
  distributed shared memory、`wgmma` warp specialization）在 Ampere 上
  用不了。我们的 Flash SDPA 是**经典 Triton 教程版**，不是 FA3 版。

- **把 elementwise fuse 进 `tl.dot` epilogue**。独立 elementwise
  kernel 是 **带宽 bound**（见上面 AI 表里 RMSNorm / add / relu，全都
  ≪ 152 break-even）——真正的开销是 HBM round-trip + kernel launch
  overhead，**不是 FLOPs**。`tl.dot` 主导的 kernel 里 FP32 cores 本来就
  闲着（即便单独跑 elementwise，也只有 BF16 Tensor peak 的 ~1/4，
  35.6 / 142 TFLOPS），所以把 elementwise 折进 matmul epilogue 是**白
  捡的 side-channel**：省掉 elementwise 自己的 HBM round-trip
  （matmul 的 HBM 开销不变）、不多一次 launch、也不抢 Tensor core 的
  FLOP/s。

---

## 第 2 章 —— FusedAddNorm：2-op fusion 的最小完整样本

本 repo 里最简单的 fused kernel。**纯学习用**——nanchat 生产 block
里 RMSNorm 已经直接 fold 进相邻的 matmul kernel（attn 走
`norm_qkv_projection`，mlp 走 `fused_mlp`），所以根本不存在
独立的 `add → norm` op 边界让这个 kernel 上 hot path。但这个 kernel
用到的每一个 pattern 都是更大 fused kernel 的基石，所以它是学这些
patterns 最干净的样本。

### 这个 kernel 算什么

数学：
```
summed = x + residual
y      = summed · rsqrt(mean(summed²) + eps) · weight    # weight 可选
```

API 返回两个张量：
- `y` → 流给下一个 block 的 matmul 输入
- `summed` → 下一个 block 的 residual stream（caller 不用再 recompute
  `x + residual`）

公开 API 的形状约定：
```python
def fused_add_norm(x, residual, norm_weight, eps=1e-6) -> (y, summed)
# x, residual: contiguous CUDA tensors, shape (M, D)
# norm_weight: 可选 contiguous CUDA tensor, shape (D,)
```

整条 autograd 链路由 4 个 Triton kernel 撑起 —— fwd 1 个，bwd 主路径
1 个 inline kernel，再加上 2 个 kernel 的 D-split fallback（在 inline
会 spill 的大 D 上兜底）：

| Kernel | Grid | 角色 |
|---|---|---|
| `_fused_add_norm_fwd_kernel` | 1D over M | fwd：写 `y` + `summed` + `rms_inv` |
| `_fused_add_norm_inline_bwd_kernel` | 1D over M | bwd **主路径**：整行单 tile、inner reduction in-register、写 `d_summed`（+ `dnw_partial`） |
| `_fused_add_norm_inner_bwd_kernel` | 1D over M | bwd fallback 阶段 1：预算 `inner[m]` |
| `_fused_add_norm_split_bwd_kernel` | 2D over (M, D) | bwd fallback 阶段 2：写 `d_summed` + `dnw_partial` |

### 2.1 Forward kernel

单 pass 设计——一个 program 处理一个 `(BLOCK_M, BLOCK_D)` tile，
load 完所有数据，把 reduction 和 per-element 输出全算完，写回。

```python
@triton.jit
def _fused_add_norm_fwd_kernel(x_ptr, res_ptr, nw_ptr,
                               y_ptr, summed_ptr, rms_inv_ptr,
                               M, D, eps,
                               BLOCK_M: tl.constexpr,
                               BLOCK_D: tl.constexpr,
                               HAS_NW: tl.constexpr,
                               HAS_RESIDUAL: tl.constexpr):
    pid_m = tl.program_id(0)
    rows  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols  = tl.arange(0, BLOCK_D)
    mask  = (rows < M)[:, None] & (cols < D)[None, :]
    offs  = rows[:, None] * D + cols[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    r = tl.load(res_ptr + offs, mask=mask, other=0.0)
    summed = x + r
    tl.store(summed_ptr + offs, summed, mask=mask)        # ← caller 的 residual stream + bwd 用

    # bf16 squared products feed an fp32 accumulator via `dtype=`. 比起
    # 显式 `summed.to(fp32)` 后再平方，省一个中间 register tile;
    # 代价是平方在 bf16 里做（精度损失 ~1e-1, atol 容许内）。
    sum_sq = tl.sum(summed * summed, axis=1, dtype=tl.float32)
    rms_inv = tl.rsqrt(sum_sq / D + eps)               # rsqrt.approx.f32 单指令
    tl.store(rms_inv_ptr + rows, rms_inv, mask=rows < M)  # ← bwd 用

    y = summed * rms_inv[:, None]                       # bf16 * fp32 → fp32 (auto-promote)
    if HAS_NW:
        nw = tl.load(nw_ptr + cols, mask=cols < D, other=0.0)
        y = y * nw[None, :]                             # fp32 * bf16 → fp32 (auto-promote)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)
```

注意的 patterns：
- **`BLOCK_D = next_power_of_2(D)`** 配 `col_mask` 处理非 2 幂 `D`
  （nanchat d24 `D=1536` → `BLOCK_D=2048`，512 lane 被 mask 掉）。
  Triton 的 `tl.arange` 要求长度是 2 的幂。
- **2D mask** = row mask × col mask，应用到所有 load/store。越界 load
  返回 `0`（**故意选 0**——`sum_sq` 的加法/乘法 identity）。
- **`HAS_NW: tl.constexpr`** 是编译期分支。无 affine weight 时直接
  跳过 `nw` 的 load + 乘法；`(HAS_NW=True, HAS_NW=False)` 两个值各
  编译出一份 binary。
- **store 时 `element_ty` cast** 让同一个 kernel 支持 caller 传任意
  dtype（bf16 / fp16 / fp32）无需修改。
- **`summed` 是 kernel 中途写 HBM 的，不是最后**。故意如此：caller
  的 residual stream 需要 bf16 版，下游计算 reuse 这同一个 bf16
  register tile（喂给 `tl.sum(dtype=fp32)` 做平方累加，以及
  `summed * rms_inv` 时 Triton auto-promote 到 fp32），不重新 load。

### 2.2 Backward：根据 register 预算选 kernel

bwd 有两套可行的 kernel 形状。具体跑哪套取决于 **inline kernel**
的单 program tile 在 Ampere 255 fp32 reg/thread 上限下能不能装得
下。dispatch 写在 `_fused_add_norm_bwd_impl` 里，由
`TileConfig.fits_reg_budget` 决定。

#### 主路径：1 个 kernel，整行 in-register

主路径是 `_fused_add_norm_inline_bwd_kernel`：1D grid over M，
`BLOCK_D = next_pow_of_2(D)` 让**整行**塞进单 tile。这样
per-row reduction `inner[m] = mean_d(g_eff · y_norm)` 直接在
register 里算——不需要 precompute kernel、不需要 inner HBM buffer、
就一次 launch。

之所以行得通是因为 nanchat 常见 shape 下 tile 并不大：

```
   ~5 个 fp32 tile  ×  BLOCK_M  ×  BLOCK_D   /   (num_warps × 32)
                                                 ≤ 255 reg/thread
```

`_pick_fused_add_norm_tile_config(M, BLOCK_D, n_live_tiles=N)` 求能装下的最大
BLOCK_M。HAS_NW=True 时 N=5（`y_norm, g_eff, dy_t, d_ext, d_summed`
峰值同时 alive），HAS_NW=False 时 N=4（`y_norm` 别名 `src`、`g_eff`
别名 `dy_t`，少 2 个 distinct tile，但 `d_ext` 和 `d_summed` 还在，
所以是 4 不是 3）。`D=1536, HAS_NW=True` 时选 `BLOCK_M=4, num_warps=8`
→ 160 reg/thread；`D=4096, HAS_NW=True` 时选 `BLOCK_M=1, num_warps=4`
→ 也是 160 reg/thread，都离 255 cap 留足余量。

#### 大炮兜底：切 D 的 2-kernel fallback

当 `BLOCK_D` 大到连 `BLOCK_M=1, num_warps=16`（封顶）都装不下，
inline 路径就崩。crossover 取决于 n_live_tiles：
- HAS_NW=True（5 tile）：`BLOCK_D > 16384`（即 D > 16K）——模型
  估算 ≥320 reg/thread，bench 实测 ~10× spill 灾难数字。
- HAS_NW=False（4 tile）：`BLOCK_D > 32768`（即 D > 32K）——任何
  实际模型基本都触发不到（nanchat 哪怕 depth=128 也才 D=8192）。

这种情况 dispatch 自动 fallback 到一对**切 D 维**的 kernel，把
per-program tile 拉回安全范围。同样一组 `(BLOCK_M=32, BLOCK_D=64)`
在 inline 下会 spill，在切 D 下却合适——每 program 只拿 64 列而
不是整行 1536+：

```
   5 tile × 32 × 64  ≈  10K fp32 / program
   10K  /  128 thread ≈ 80 reg/thread       ← OK
```

但切 D 引入新问题——bwd 数学里那个 per-row reduction
`inner[m] = mean_d(g_eff · y_norm)` 跨**整个** D 维。同一个 m_tile
的所有 d_tile program 都要这个标量。最朴素的做法：每个 program
自己 loop 整 D 算一遍 `inner[m]`。`D=1536, BLOCK_D=64` 下这是
**24× 冗余计算**。

学 Flash Attention 的「预算 reduction」pattern，fallback 拆成 2 个
kernel：

**Stage 1 —— `_fused_add_norm_inner_bwd_kernel`**（1D over M，每
program 处理 `BLOCK_M` 行，单 tile 涵盖整 D）
```python
inner[m] = mean_d(g_eff[m, *] * y_norm[m, *])
# 写到 (M,) fp32 buffer
```

**Stage 2 —— `_fused_add_norm_split_bwd_kernel`**（2D over M × D，每
program 处理一个 `(BLOCK_M, BLOCK_D)` 输出 tile）
```python
inner_m = tl.load(inner_ptr + rows)              # 每行一个标量，预备好了
dx[m, d] = rms_inv[m] * (g_eff[m, d] - y_norm[m, d] * inner_m[m]) + d_ext[m, d]
```

Stage 2 里原来跨整 D 的 pass 1 缩成「每行一个标量 load」。总 HBM
流量跟单 kernel 版本相同（`summed`/`dy` 在 inner 读一次、在 stage 2
读一次），但 compute 上不再有跨 program 的 per-row reduction 重复。

#### 为什么不无脑「切 D」

这个 kernel 的早期版本 bwd 路径**只有**切 D 的 2-kernel——当时
的论证是「1D over M 必 spill，必须切 D」。但这个论证有个 bug：
它假设了 `BLOCK_M` 固定 32。`D=1536, BLOCK_M=32, num_warps=4`
（128 thread）下：

```
   5 tile × BLOCK_M=32 × D=1536  ≈  245K fp32 / program
   245K / 128 thread             ≈  1900 fp32 reg / thread
```

**1900 vs 255 cap → 惨烈 spill**。BLOCK_M=32 + 整 D 确实 spill。
但 inline 路径意识到 `BLOCK_M` 也是个自由参数——
`_pick_fused_add_norm_tile_config(M, BLOCK_D, n_live_tiles=5)` 推导出
`BLOCK_M ≤ 1638·nw / BLOCK_D`（= 256 reg/thread 上限 / 5 tile，
再摊到 nw·32 thread 上），向上取整到 pow-of-2。D=1536 时
BLOCK_M 落到 4（regs ≈ 160），D=4096 时落到 1（也 ≈ 160），
舒舒服服塞进 cap，不 spill。

本质思路跟切 D fallback 一样（哪根 axis 缩了能让 tile 留 register
就缩哪根），只是 inline 缩的是 M 这根——每行独立，没有跨行
reduction 要操心。fallback 只在 BLOCK_M 缩到 1 *仍然*装不下时
触发——也就是 BLOCK_D 单独（= 整 D）就爆预算的情况。

### 2.3 d_summed_external 折进 bwd kernel

op 返回两个输出（`y`、`summed`），所以 backward 收到两份梯度
（`dy`、`d_summed_external`）。前者是 norm 输出梯度，后者来自 caller
在下游直接用了 `summed`。

朴素写法：bwd kernel 算 `d_summed_from_norm`，外面 Python 加一行
`d_summed_total = d_summed_from_norm + d_summed_external`。这是
**多一次 torch elementwise op**——单独的 kernel launch（~10-30 μs）
+ 4·M·D HBM round-trip（d24 shape 下 ~10-15 μs）。

直接把 `d_summed_external` 作为 bwd kernel 的额外输入，把这次加法
折进同一个 tile store：

```python
d_summed_tile = rms_inv * (g_eff - y_norm * inner) + d_ext   # ← 「+ d_ext」就是融
tl.store(d_summed_ptr + offs, d_summed_tile.to(...), mask=mask)
```

纯 register-level 加法，零额外 HBM 流量、零额外 kernel launch。

### 2.4 Sizing —— 套用第 1 章的预算公式

所有按 tile 切的 kernel（fwd、bwd inline、fallback 的 inner）共享同
一个 sizing 决策 helper：

```python
def _pick_fused_add_norm_tile_config(M, BLOCK_D, n_live_tiles) -> TileConfig:
    # Register-budget 模型（Ampere 255 fp32 reg/thread spill cap）：
    #     regs/thread ≈ n_live_tiles × (BLOCK_M × BLOCK_D) / (nw × 32)
    # 按 256-reg 上限算：tile ≤ (8192 / n_live_tiles) × nw
    tile_per_nw = 8192 // n_live_tiles                    # n=2 时 4096, n=5 时 1638
    base_nw = 4                                           # 初始 guess
    BLOCK_M = max(1, min(
        triton.next_power_of_2(max(1, M // 64)),          # M-saturation：grid ≳ 64
        triton.next_power_of_2(max(1, tile_per_nw * base_nw // BLOCK_D)),
    ))
    tile = BLOCK_M * BLOCK_D
    num_warps = max(4, min(16, triton.next_power_of_2(max(1, tile // tile_per_nw))))
    est_regs = n_live_tiles * tile // (num_warps * 32)
    return TileConfig(BLOCK_M, num_warps, est_regs)
```

`n_live_tiles` 是 per-kernel 的旋钮——hot path 同时 alive 的 fp32
tile 数：

| Kernel | n_live_tiles | 原因 |
|---|---|---|
| fwd | 2 | auto-promoted summed-as-fp32 + `y_f32` |
| inner 预算（fallback） | 2 | `y_norm` + `g_eff` 短暂 |
| bwd inline (HAS_NW=True) | 5 | `y_norm`、`g_eff`、`dy_t`、`d_ext`、`d_summed` |
| bwd inline (HAS_NW=False) | 4 | 同上，但 `y_norm`/`g_eff` 别名 `src`/`dy_t` |

`TileConfig.fits_reg_budget` 属性（`est_regs ≤ 256`）就是
`_fused_add_norm_bwd_impl` 用来选 inline vs 2-kernel fallback 的依据。

d24 shape (M=2048, D=1536, BLOCK_D=2048) 走一遍：

**fwd (n=2)**：`tile_per_nw = 4096`。BLOCK_M = min(next_pow2(32),
next_pow2(4096·4/2048)) = min(32, 8) = 8。tile = 16K。
nw = next_pow2(16K/4096) = 4。→ `BLOCK_M=8, nw=4`，~256 reg/thread。
Grid = cdiv(2048, 8) = 256 programs。

**bwd inline (n=5)**：`tile_per_nw = 1638`。BLOCK_M = min(32,
next_pow2(1638·4/2048)) = min(32, 4) = 4。tile = 8K。
nw = next_pow2(8K/1638) = 8。→ `BLOCK_M=4, nw=8`，~160 reg/thread。
Grid = cdiv(2048, 4) = 512。

**2-kernel bwd fallback**：用**固定 config** `BLOCK_M=32, BLOCK_D=64,
num_warps=4`，不走 `_pick_fused_add_norm_tile_config`。原因不是性能（autotune 选的
也差不多），而是 Triton autotune 的 dispatch 保留了一些不能 CUDA
Graph stream capture 的操作。写死 config 让 fallback 路径对 graph
友好。只有 inner 预算 kernel 用 `_pick_fused_add_norm_tile_config`（n_live=2）。

### 2.5 数值精度

这个 kernel **跟 `F.rms_norm` 在 bf16 输入上的行为近似但不 bit-tight**——
PyTorch 的 `F.rms_norm` 内部把 bf16 升 fp32 **再做平方**，我们图省一
个 register tile，让平方留在 bf16 里、只把 accumulator 升 fp32：

```python
# F.rms_norm  : summed.to(fp32) → (fp32 * fp32)² → fp32 sum
# 我们的 fwd  : (bf16 * bf16)² → fp32 sum  (via `tl.sum(..., dtype=fp32)`)
```

平方在 bf16 截了一次精度，长 D 累加下来 max forward diff ~1.5e-1（在
adversarial 测试 seeds 上；正常 seeds 远低于这个），低于 gradient
noise floor。`test_triton_norm_mlp.py` 的 bf16 atol 也相应放到 1.5e-1。

| 操作 | 在哪 | 为什么 |
|---|---|---|
| HBM load `x`、`r`、`nw` | bf16（caller dtype） | 省带宽 |
| `summed = x + r`、residual-stream store | register 里 bf16 | caller 要 bf16 |
| `summed * summed` 平方 | **register 里 bf16**（截精） | 省一个 fp32 register tile |
| `tl.sum(..., dtype=tl.float32)` | **accumulator 升 fp32** | 防 long-D 累加溢 mantissa |
| `summed * rms_inv` / `* nw` | **auto-promote 到 fp32** | rms_inv 是 fp32，Triton 自动升级 |
| 最终 `y` store | cast 回 bf16 | caller 要 bf16 |

**净 register 压力跟「全 fp32 internal」实现差不多**（fwd kernel：
tile=16384, nw=4 实测 n_regs≈255）——`summed * rms_inv` 的 auto-promote
反正会在 register 里物化一个 fp32 tile，省下的就是显式 `summed_f32`
中间变量那一份 SSA 名字。bf16 路径是为了 **HBM dtype 兼容**，不是
register 省钱。

> 想要 bit-tight 跟 `F.rms_norm` 对齐？显式 `summed.to(tl.float32)` 后
> 再平方即可（多一份 register tile，跟 fp32-internal 实现一致）。当前
> 实现选了精度 vs register 的 tradeoff，工程上更划算。

### 2.6 预期收益 —— HBM 和 launch 账本

实测之前先算账。一句话总结：**前向真省 HBM round-trip；后向 inline
（主路径）也省字节；后向 2-kernel（fallback）只省 launch 和中间
buffer**。

#### Forward

Naive native（两步走：`summed = x + r`，然后 `y = F.rms_norm(summed)`）：
```
torch.add (x + r → summed):
  read x        M·D
  read r        M·D
  write summed  M·D
F.rms_norm (summed → y):
  read summed   M·D                     ← fusion 省掉的就是这次
  write y       M·D
─────────────────────────────────────────
合计:           5·M·D
```

Fused（一个 kernel 全做完）：
```
read x          M·D
read r          M·D
write summed    M·D    (caller residual stream 还是要)
write y         M·D
─────────────────────────────────────────
合计:           4·M·D
```

净 forward 收益：
- **HBM：省 1·M·D 字节** —— `summed` 从 `x + r` 直接接到 norm
  reduction，全程留 register，不进 HBM。
- **Kernel launches：省 1 次**（两个 op 合成一个 Triton kernel）。
- **中间 buffer：省 1 个**（torch.add 不需要单独分配 summed 输出）。

对 d24（M=2048, D=1536, bf16）：1·M·D = 6.3 MB / 936 GB/s ≈ 6.7 μs
HBM 时间，再加上避免的 ~10-30 μs launch overhead。

#### Backward

bwd 有两条 kernel 路径（dispatch 见 §2.2）。常见情况是 **inline
单 pass kernel**；**2-kernel D-split fallback** 只在 inline tile
会超 255 reg/thread cap 时触发（HAS_NW=True at D > 16K；
HAS_NW=False at D > 32K，实际不可能）。下面两条都算一遍。

下面所有账本都按 `HAS_NW=False`（没 learnable affine weight，
对齐 nanchat 实际配置——见 `nanchat/gpt.py:9`）算。HAS_NW=True
两条 fused 路径都要多一个 `dnw_partials` 写（~M·D fp32，inline
里 BLOCK_M 小所以更大）+ `.sum(dim=0)` reduction（~M·D 读，~D 写），
native 那边对应多一个 per-channel dW kernel（~2·M·D 读，~D 写）
—— 大致互相抵消，所以相对收益保持不变。

PyTorch 优化版 native（单 `F.rms_norm.backward` kernel + Python
外加 external 梯度）。bwd 数学本身要两次 D 维 reduction
（`inner[m] = mean_d(dy · y_norm)` 然后
`dx = rms_inv · (dy − y_norm · inner)`），所以下面写「读 1 次」
是按 PyTorch 实际走的 **1-pass shared-memory 模式**算：一个 CUDA
block 处理一行，把 `summed` 和 `dy` 一次性 load 进 shared mem，
reduce 完再回 shared mem 算 per-element `dx`。D=1536 fp32 +
dy = 12 KB/行，3090 单 SM 100 KB shared mem 装得下。**字面意义
的 naive 2-pass 实现**（把 `inner` 写回 HBM、第二个 pass 重读
`summed`/`dy`）会让读字节数翻倍——如果你要跟那种 2-pass 实现
比，记得照这里的数往上加。

Native（1-pass shared-mem + Python 外加 external 梯度）：
```
F.rms_norm.backward (1-pass shared-mem):
  read summed              M·D
  read dy                  M·D
  write d_summed_from_norm M·D                              ─┐
                                                             │ subtotal 3·M·D
Python d_summed = d_summed_from_norm + d_summed_external:
  read d_summed_from_norm  M·D
  read d_summed_external   M·D
  write d_summed_total     M·D                              ─┘ subtotal 3·M·D
─────────────────────────────────────────
合计:                      6·M·D
```

##### 路径 A —— Inline（主）：字节 + launch + buffer 三杀

inline kernel 用 `BLOCK_D = next_pow_of_2(D)` 把整行塞进一个 tile，
`inner` 在 register 里算——没有预算 kernel、没有 inner HBM buffer。
`d_summed_external` 通过 register 加法折进 d_summed store，跟
2-kernel 路径同样的招：

```
_fused_add_norm_inline_bwd_kernel:
  read summed/y            M·D
  read dy                  M·D
  read d_summed_external   M·D
  write d_summed           M·D
─────────────────────────────────────────
合计:                      4·M·D
```

inline 路径相对 native 净收益：
- **HBM：省 2·M·D 字节** —— 没有 `d_summed_from_norm` 中间 buffer
  流过 HBM（kernel 在同一行的 register 里算 norm 梯度 + 折进 `d_ext`）。
- **Kernel launch：省 1 次**（1 个 Triton kernel vs PyTorch 的
  `rms_norm.backward` + Python add）。
- **中间 buffer：省 1 个**（没 `d_summed_from_norm` 分配）。

对 d24（M=2048, D=1536, bf16）：2·M·D = 12.6 MB / 936 GB/s ≈ 13 μs
HBM 时间省下来，再加上避免的 ~10-30 μs launch overhead。这就让
fwd 和 bwd 主路径有了对称的「三杀」收益。

##### 路径 B —— 2-kernel fallback：字节打平，launch 还省

inline 会 spill 时（HAS_NW=True at D > 16K，见 §2.2），dispatch
fallback 到 2-kernel 对。切 D 的结构**没法**用 1-pass shared-mem
招——没有 program 拿到完整一行—— 所以 `summed`/`dy` 要读两次
（inner 一次，bwd 一次）：

```
_fused_add_norm_inner_bwd_kernel:
  read summed/y            M·D
  read dy                  M·D
  write inner_buf          ~0  (M floats)                   ─┐ subtotal 2·M·D
                                                             │
_fused_add_norm_split_bwd_kernel:                                  │
  read summed/y            M·D   ← 重复读（L2 大概率 hit）
  read dy                  M·D   ← 重复读（L2 大概率 hit）
  read d_summed_external   M·D
  read inner_buf           ~0
  write d_summed           M·D                              ─┘ subtotal 4·M·D
─────────────────────────────────────────
合计:                      6·M·D    (= native)
```

**字节跟 native 打平** —— fused 读 `summed`/`dy` 各两次（2·M·D
each），native 各一次，但 native 单独的 `+` kernel 又加上了 3·M·D
的 `d_summed_from_norm` round-trip。净：持平。收益缩水成：
- **省 1 个中间 buffer**：没 `d_summed_from_norm`（跟 inline 一样）。
- **省 1 次 kernel launch**：拿 native 的 Python add 跟 inner 预算
  kernel 互换——净 launch 数跟 native 持平，但 fused 两次 launch
  靠得近（bwd kernel 的重复读 L2 复用更好）。

#### Takeaway

| 路径 | bytes | launches | buffer |
|---|---|---|---|
| Forward fused | ✓ 省 1·M·D | ✓ 省 1 | ✓ 省 1 |
| Backward inline（主） | ✓ 省 2·M·D | ✓ 省 1 | ✓ 省 1 |
| Backward 2-kernel（fallback） | — 打平 | ~ 打平（互换） | ✓ 省 1 |

教训：**只要能在 register 里复用上一步的值、而不是从 HBM 来回搬，
就是真实的带宽节省**。inline 路径在 fwd 和 bwd 上都做到了这点。
2-kernel fallback 被迫重读 `summed`/`dy` 是因为 per-program tile
装不下整行——但 fallback 只在替代方案（catastrophic register
spill）更糟时才触发。

### 2.7 性能现实

d24 fwd-only（M=2048, D=1536, bf16, HAS_NW=False）在 RTX 3090
上实测：

| 测量方式 | fused | native | 对比 |
|---|---|---|---|
| Kernel-only timing（直接 launch kernel，绕过 autograd） | **~72 μs** | ~88 μs | **fused 快 1.22×** |
| 普通 eager `fused_add_norm(...)` call（仅 fwd） | ~163 μs | ~88 μs | fused 慢 1.85× |
| 普通 eager `fused_add_norm(...) + backward` | ~1075 μs | ~618 μs | fused 慢 1.74× |

> ↑ 这两组 eager 数字是 `autograd.Function` 时代测的。fused_add_norm
> 现已改写成 `torch.library.custom_op`（fwd / bwd 各一个，配
> `register_fake` + `register_autograd`），eager 走的还是 Python
> dispatch，per-call overhead 量级差不多；但**核心变化**是
> `torch.compile(fullgraph=True)` 现在能直接编译进去（autograd.Function
> 那时是 graph-break 起手），跨 op 路径不再被这个 wrapper 截断。

eager 模式下慢 **不是 kernel 慢**——是 wrapper 的 per-call 固定开销
（每次 fwd ~90 μs：tensor 分配、`save_for_backward`、ctx 设置、kernel
launch arg 打包）。这些 overhead 在 CUDA Graph 捕获或 `torch.compile`
包大 pipeline 里会消失。

大 shape 反过来：`M=2048, D=4096`（fwd+bwd, HAS_NW=False）下 fused
端到端**赢 1.21×**。crossover 在 kernel 实际 GPU 工作量超过 ~90 μs
framework overhead 的地方。

核心教训：**kernel 这么短的时候，Python 框架开销很容易超过 kernel
本身的 GPU 工作量**。fusion 真正赚到只有两种情况：
1. op 被包在 `torch.compile(model)` 整 model 里，dispatcher 开销被摊掉
   —— `custom_op` 改造后的现在这条路才真正打通（以前 autograd.Function
   会 graph-break，dynamo 退回 eager dispatch）
2. 整 train step 用 `torch.cuda.CUDAGraph` capture，per-call Python
   工作只在 capture 时跑一次

这跟 production 级 transformer kernel **都越做越大**的原因是一致的
（Flash Attention 整段 `Q@K^T → softmax → @V` 全在一个 kernel 里，
不是分小步走）：kernel launch 和 Python dispatcher 的开销是每 op 的
常数，更长的 kernel 摊薄得多，哪怕 per-element 吞吐不比一串小 kernel
更高，wall time 上还是赢。

这也是为什么 nanchat 生产路径大多跳过这个 standalone op-boundary
kernel：`fused_mlp` 把 norm 直接 fold 进自己的 matmul 路径；
`norm_qkv_projection` 只复用其中的 RMSNorm 物化子 kernel，然后接
Q/K/V projection。真正让 patterns 体现价值的，是更大的生产 kernel。

---

## 第 3 章 —— `fused_mlp`：production-level、fwd+bwd 全 Triton

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

---

（第 4 章及后续 —— 待写）
