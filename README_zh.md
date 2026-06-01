# nanochat-3090: nanochat on RTX 3090 —— 教学 fork

> English version: [README.md](README.md)

**为什么 RTX 3090（而不是租 H100）**。初学者绝大部分时间花在**调试和
学习**上，**不是真正在训练**——读源码、用 debugger 单步走 backward、
换一个 in-place trick 对比效果、跟 PyTorch reference 对拍 loss 曲线、
跑 20 iter 探针、profile 内存。在这些阶段租 H100（spot 价 $2-4/h）每
小时**贵 10-20 倍**——你付的钱大部分是你**没在用**的 flops。

按学习目标，最便宜的配置：
  - **只学算子 + 训练内部原理** → **单张 RTX 3090**
    （spot ~$0.18/h，或者自己买一张 → 后续 $0/h）。本仓库所有内容包括
    d24 用 offload stack 都跑得动；没有分布式带来的 surprise 要 debug。
  - **学习目标加上 NCCL / DDP / collective 通信** → **双 RTX 3090**
    （spot ~$0.36/h）。这是**真正能跑 cross-device** `dist.all_reduce`
    / `dist.reduce_scatter` 的最小配置（单卡 torchrun 只是设了 env 变量
    但**没有真正的跨 rank 网络**），用来 profile NCCL 瓶颈、测 ZeRO
    分片策略、复现 DDP 特有的 bug。

H100 只有在 **wall-time per run 超过 debug 迭代时间**时才划算——通常
等你对代码已经有信心、只想刷吞吐的阶段。

---

本 fork 在 [karpathy/nanochat](https://github.com/karpathy/nanochat) 基础上，
做两件相互关联的事：

1. **`nanoops/` —— 补上 nanchat 教学链路里"PyTorch 算子内部"那一块缺失。**
   nanchat 把整条 LLM 训练流水线（tokenizer → 训练循环 → eval → chat UI）
   讲得很完整，但里面的 PyTorch 算子都是黑盒——`F.linear`、
   `F.scaled_dot_product_attention`、`F.cross_entropy` 等等都是直接拿来用。
   nanoops 把这些黑盒打开：nanchat 用到的每个 PyTorch 算子（`Mm` /
   `Linear` / `RMSNorm` / `Softmax` / `CrossEntropy` /
   `ScaledDotProductAttention` / `ApplyRotaryEmb` / 滑动窗口 attention …）
   都用自定义 `torch.autograd.Function` 重写过——显式 forward + backward、
   in-place / 内存敏感实现，并在
   [`nanoops/README_zh.md`](nanoops/README_zh.md) 的附录里附了完整
   数学推导（也有 [English 版](nanoops/README.md)）。从源码里能看到
   `softmax_backward` 怎么用 `addcmul_` 融合、ctx 取舍怎么算账、
   GQA 怎么靠 `repeat_interleave + unflatten/sum` 收尾、在线 softmax /
   chunked LSE 长什么样、embedding backward 怎么做分段求和——不是停留
   在白板上的推导。

2. **在 24 GiB 消费级 GPU 上同时优化 nanchat 训练的两个维度：速度 和
   模型大小。** nanoops 现在有两层执行路径：第一层是上面说的 Python
   `autograd.Function` 教学实现；第二层是训练主路径的 Triton full-fuse：
   `fused_mlp`、`fused_attn_qkv_projection`、`fused_attn_spda_and_output`。
   旧的 Triton fused CE tail 已删除；lm-head / softcap / CE 继续走标准
   nanoops functional 路径。配合 optimizer-state CPU offload 和
   expandable-segments allocator，消费级 24 GiB GPU 既能装下 d24，也能把
   编译后的训练 step 往前推。

### 实际效果

**`--depth=24` 是 nanchat 的参考模型尺寸——3090 这种消费级显卡（RTX
3090 / 4090 等 24 GiB 级别的卡）原本根本跑不起来。** 用 nanchat 原生代码
在 24 GiB 卡上训 d24，**任何 batch size 都 OOM**：1.5 B 参数 auto-config
加宽到 `n_embd=1536` × 24 层 + AdamW state + bf16 gradients + 每个 sliding
layer 的完整 `(L, L)` attention 概率矩阵——加起来就是装不下。nanchat 的
参考硬件是 8× H100 节点——**对在家学习或预算有限的人来说远超能力**。

本 fork 的显存栈（带状/sliding attention、可选 activation checkpoint、
优化器 state CPU offload、`expandable_segments` allocator）省够内存并抑制
allocator 碎片，让 d24 终于能在 24 GiB 消费级显卡上以
`--device-batch-size=1` **真的装下并跑起来**——不管是**一张**卡还是
**两张**。Triton full-fuse 路径继续优化训练 step 本身，并在双 3090 上默认
`--device-batch-size=2`：fused MLP、fused QKV/rotary/QK-norm/VE、
fused SDPA/output-projection handoff。双卡通过 DDP 数据并行把同一份
per-iter 工作量分担到两块 GPU，把 wall time 减半，但峰值显存跟单卡相同。
**这个项目的意义就是把 nanchat 的默认训练拉进
初学者硬件预算的范围**。

| 配置             | nanchat 原生 | nanoops, 1× 24 GiB 卡 | nanoops, 2× 24 GiB 卡 |
| ---------------- | ------------ | --------------------- | --------------------- |
| `--depth=20`, B=4 | OOM (无 FA3) | (同样配方装得下)     | **~30.5k tok/s**, ~31h |
| `--depth=24`, B=1/2 | 任意 B 都 OOM | **~8k tok/s**, ~200h | **~16k tok/s** checkpoint-heavy B=1；**~19.5k tok/s** full-fuse B=2 |

**算成钱**：3090 spot 租赁价 ~$0.18/卡/小时，单卡 ~$0.18/h（~$30/周）、
双卡 ~$0.36/h（~$60/周）。一次完整的 `--depth=24` 训练：当前 full-fuse
B=2 路径在双卡上约 **3.8-4.0 天，~$33-35**。其中纯训练 step 按实测
~54.2 s/step 约 **3.5 天**，额外 wall time 来自默认 validation、checkpoint、
CORE 和 sample 节奏。单卡显存优先路径仍大约是 **~8.3 天，~$36**。
`--depth=20` 双卡训练 ~31h，**不到 $12**。原本目标硬件是 8× H100 节点，
本 fork 让这个训练在一台桌面机（一或两张消费级 GPU）上可行。

**很适合初学者上手**。即便跑较重的 d24 训练，一周预算里还剩 ~$25 /
~3 天 GPU 时间正好用来"折腾"——读一下 `nanoops/functional.py` 里
某个算子的实现、把某个 in-place trick 改掉、往 `.backward()` 加个
print、跑个 20-iter 看 loss 曲线和 MFU 怎么变。整套代码量小到可以
拿调试器一步步走完，配套测试（`tests/test_nanoops_e2e.py`,
`tests/test_sdpa_parity.py` 等）会把每个算子跟 PyTorch reference
对拍——**永远有 ground truth 可以参照**。

### 当前 d24 性能（2× RTX 3090）

| 配置 | dt/step | tok/sec | MFU | wall time / 成本 |
| ---- | ------- | ------- | --- | ---------------- |
| Checkpoint-heavy 显存优先，B=1 | ~66.5 s | ~15.8k | ~53% | 训练 ETA 约 61h；显存余量更保守 |
| **Full-fuse，B=2（默认）** | **~54.2 s** | **~19.3k** | **~65%** | **纯训练 step 约 3.5 天；含默认 eval+checkpoint 节奏约 3.8-4.0 天 / $33-35** |

full-fuse 行来自当前 d24 训练 compile/warmup 后的实测：
`NANOOPS_FUSED=1`，fused MLP、fused QKV、fused SDPA/output，不开 activation
checkpoint，optimizer state CPU offload，`device-batch-size=2`。更看重显存
余量而不是单步耗时时，可以用 `NANOOPS_FUSED=0` 切回 checkpoint-heavy 路径。

### 实测加速过程（d20 base_train on RTX 3090，双卡数据）

| 配置                                  | tok/sec    | MFU       | Peak 显存    | vs baseline |
| ------------------------------------- | ---------- | --------- | ------------ | ----------- |
| PyTorch SDPA, B=2 (baseline)          | 22,725     | 46.2%     | 16.5 GiB     | —           |
| nanoops Lookup default, B=2           | 28,800     | 58.5%     | 19.7 GiB     | +27%        |
| + SlidingWindowSDPA, B=2              | 30,594     | 62.2%     | 17.6 GiB     | +35%        |
| + B=4 + expandable_segments           | 32,678     | 66.4%     | 22.7 GiB     | +44%        |
| **+ MLP_CHECKPOINT（当前默认）**      | **30,500** | **62.0%** | **19.0 GiB** | **+34%, 留余量给 d24** |

所有行 loss 曲线在 bf16 数值噪声范围内**完全一致**。完整 A/B 分析记录在
[`SlidingWindowSDPA` 的 docstring](nanoops/functional.py)。

### `NANOOPS_FUSED=1` 覆盖哪些能力

默认 fused 路径不是简单把 `F.linear` 换快一点，而是把 transformer block
热路径上的链式算子换成 torch.compile 可见的 Triton custom op：

- `fused_mlp`：pre-RMSNorm、`c_fc`、ReLU²、`c_proj`、residual add，以及对应
  backward。
- `fused_attn_qkv_projection`：residual/x0 混合、RMSNorm、独立 Q/K/V
  projection、rotary、Q/K RMSNorm+scale、可选 value embedding gate/lookup，
  以及对应 backward。
- `fused_attn_spda_and_output`：GQA/sliding-window SDPA 加 attention output
  projection/residual tail，包括原本分开的 backward delta/output-proj 计算。
- `fused_add_norm`：独立 add+RMSNorm Triton op，也被 block 级 fused kernel 复用。

lm-head / softcap / cross-entropy tail 刻意保留在标准 nanoops functional 路径；
实验性的 Triton CE tail 已删除，因为它能省显存，但相对 native GEMM 主路径
吞吐损失太大。

### 怎么跑

```bash
# speedrun.sh 中 base_train 步骤的 drop-in 替代版——
# 默认 --depth=24。默认路径是 full-fuse，device-batch-size=2：
# fused MLP + fused QKV + fused SDPA/output，
# 不开 activation checkpoint，optimizer state offload 到 CPU，并在 CUDA init 前
# 设置 expandable_segments。
bash nanoops/train.sh                       # full-fuse，用所有可见 GPU
NPROC=1 bash nanoops/train.sh               # 单卡——同样的配方依然装得下
NANOOPS_FUSED=0 bash nanoops/train.sh       # 显存优先的 checkpoint-heavy 路径

# 也可以覆盖默认值——比如双卡 3090 上吞吐最大的 setup：
bash nanoops/train.sh --depth=20 --device-batch-size=4

# 有效默认值 / 环境变量：
#   NANOOPS=1                                       启用 nanoops 集成
#   PYTORCH_ALLOC_CONF=expandable_segments:True     回收碎片化内存
#   NANOOPS_OFFLOAD_OPTIM=1                         Muon+AdamW state 移到 CPU pinned;
#                                                    d24+B=1 装下的必要条件
#   NANOOPS_FUSED_MLP=1                             fused Triton MLP block
#   NANOOPS_FUSED_ATTN=1                            fused QKV + SDPA/output
#   NANOOPS_DEVICE_BATCH_SIZE=2                     full-fuse 默认 micro-batch
#   NANOOPS_SAVE_EVERY=200                          checkpoint 间隔
#
# 模式开关：
#   NANOOPS_FUSED=1                                  默认：full-fuse 性能路径
#   NANOOPS_FUSED=0                                  显存优先 checkpoint 路径
#   NANOOPS_MLP_CHECKPOINT=1                         显存优先的 MLP checkpoint
#   NANOOPS_L_ATTN_CHECKPOINT=1                      显存优先的 L-attn checkpoint
#   NANOOPS_LOOKUP_SORTED=1                          segmented embedding backward
```

详见：
- [`nanoops/README_zh.md`](nanoops/README_zh.md)（中文）/ [`nanoops/README.md`](nanoops/README.md)（English）——按算子排的 TODO + 数学推导附录
- [`nanoops/integration.py`](nanoops/integration.py) —— 注入 nanchat 的 monkey-patch 怎么写（不动 upstream 模型代码）

### nanchat 上游

本 fork 完整保留了 nanchat 训练流水线（tokenization、pretraining、
finetuning、evaluation、inference、chat UI）。关于 nanchat 本身的介绍、
GPT-2 leaderboard、使用方法等，请参见 [README.md](README.md) 后半部
（保留了原版英文文档），或者直接看
[karpathy/nanochat](https://github.com/karpathy/nanochat) 上游仓库。
