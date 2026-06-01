# nanoops

从零手写 nanochat 用到的 PyTorch 算子。

**本目录下所有算子仅用于教学。** 实现风格优先可读性、把数学过程摆在明面上，而
不是追求性能、融合 kernel 或边界覆盖。生产训练请直接用 `torch.nn` /
`torch.nn.functional`——这个包的存在意义是让你能读源码、用调试器单步走、并
与 PyTorch 行为做对照。

> English version: [README.md](README.md)

## 目录结构

公共 API 与 PyTorch 对齐，nanochat 切换实现只需改 import。

| 文件 | 镜像 | 内容 |
| --- | --- | --- |
| `nn.py` | `torch.nn` | Module 风格算子（`Linear`, …） |
| `functional.py` | `torch.nn.functional` | 函数式算子 + `autograd.Function` 子类 |

## 约定

- Module 的初始化方案是经过推敲的选择，不必然与 `torch.nn` 一致。当 PyTorch
  的默认值有历史包袱（例如 `Linear` 的 `kaiming_uniform_(a=sqrt(5))`），
  nanoops 选择更有理论依据的值（`a=1`），并在类的 docstring 里说明差异。
- `autograd.Function` 子类（例如 `Mm`、`Add`、`Lookup`）刻意使用 legacy 的
  `forward(ctx, ...)` 签名——把数学和缓存张量写在一起更直观。
- shape 限制（`Mm` 只接 2D、`Lookup` 只接 1D）是有意为之，目的是让 autograd
  原语一屏读完。更高维的处理由调用方负责（见 `functional.py` 里的 `linear`
  和 `embedding`，在 2D/1D 核心算子外做 flatten + unflatten）。

## Parity 测试

`tests/test_nanoops.py` 对每个算子做与 `torch` 的前向 + 反向对拍，运行方式：

```
pytest tests/test_nanoops.py
```

新增算子时请在同一文件里补一条 parity 测试。

## TODO

**范围：只实现有意义反向传播的算子。** 优化器（AdamW/Muon）、参数初始化、
离散采样（`topk`/`argmax`/`multinomial`）、常量生成器（`arange`、rotary 的
cos/sin 表）、DDP、`torch.compile` 等**不可导**的部分一律走 PyTorch——
nanoops 的目的是教 backward，不是复刻所有工具函数。

按 nanochat 真实依赖排序。Tier 1 完成即可跑通核心 block 的前向 + 反向；
Tier 2 加入注意力；Tier 3 是可选的性能优化版本。

### Tier 1 —— 核心 block

- [x] `nn.Linear` / `F.linear`
- [x] `nn.Embedding` / `F.embedding`
- [x] `nn.RMSNorm` / `F.rms_norm`（模型里唯一用到的 normalization）
- [x] `relu_square` —— 融合 `relu(x)**2`，对应 nanchat 的 `F.relu(x).square()`
- [x] `F.softmax`
- [x] `F.cross_entropy`（带 `ignore_index`）
- [x] `torch.outer`
- [x] `torch.cat`
- [x] `torch.stack`
- [x] `torch.sigmoid` / `torch.tanh`（gate 与 logit softcap）

### Tier 2 —— 注意力

- [x] `apply_rotary_emb`（cos/sin 表继续用 PyTorch）
- [x] `F.scaled_dot_product_attention`（朴素 `softmax(QK/√d) V`；与 nanchat 全等：is_causal + attn_mask + enable_gqa）
- [x] `torch.where` / `torch.roll`（eval 与 loss masking 用）
- [x] `SlidingWindowSDPA`（一个 autograd.Function 内部做 chunked sliding window attention；支持 GQA；集成里默认开。**端到端 +44% vs PyTorch baseline，2× RTX 3090, d20**）

### Tier 3 —— 用 Triton 写的融合 kernel（可选）

- [x] **`norm_mlp_relu_square`**：`relu(RMSNorm(x) @ W_fc.T)² @ W_proj.T` ——
      整个 MLP 块。Forward 在一个 tiled Triton kernel 里同时做 RMSNorm +
      c_fc + ReluSquare（per-row rms 在寄存器里跨 matmul 复用）；c_proj 走
      cuBLAS。Backward 串联两次 cuBLAS 梯度 + 一个小的 relu²-bwd Triton
      kernel + 一个 RMSNorm-bwd Triton kernel。
- [x] **`fused_attn_qkv_projection`**：`RMSNorm(x) @ W_qkv.T`，其中 W_qkv 是把
      `c_q.weight, c_k.weight, c_v.weight` 在 dim 0 上拼接得到，并在写回前
      对 Q/K 做 rotary、QK RMSNorm 和 scale。它覆盖 SDPA 之前的 QKV-side
      setup。
- [x] **`fused_cross_entropy`**：融合 lm-head projection、logit softcap 和
      cross entropy。训练 loss 路径按 vocab tile 流式处理，只保存每个 token
      一条 LSE，不物化完整 `(B*T, vocab)` logits。

block 级 fused kernel 覆盖 transformer block 里最重的 "norm + linear 链"
入口；`fused_cross_entropy` 覆盖显存最重的训练 loss tail。

### 新增算子流程

1. 按 PyTorch 的 import 路径放到 `nn.py` 或 `functional.py`。
2. 在 `tests/test_nanoops.py` 中补一条 parity 测试，**前向和反向都要覆盖**。
3. 实现保持一屏可读——不要融合 kernel，shape 也不要做超出 nanochat 实际
   需要的泛化。

## 附录：反向推导

nanoops 大多数算子反向是平凡的（gradient routing、elementwise scaling）。
少数算子的 Jacobian 非平凡，化简后的形式对可读性和显存都有意义。这些推导
放在这里。

### RmsNorm

最后一维长度 $D$，一个 slice 内 forward（不带 `weight`）：

$$
s = \text{mean}(x^2) + \epsilon, \qquad n = \sqrt{s}, \qquad y_i = \frac{x_i}{n} \quad (i = 0, \dots, D-1)
$$

**记号说明.** $a \odot b$ 表示**按元素相乘**（Hadamard 积），**不是**内积/点积——
结果是和输入同形状的向量，$(a \odot b)_i = a_i b_i$。需要 reduce 成标量时会显式写出
（例如 $\text{mean}(g \odot y) = \tfrac{1}{D} \sum_i g_i y_i$）。

给定 upstream $g_i = \partial L / \partial y_i$，推 $\partial L / \partial x_j$。

**Jacobian.** 对 $y_i = x_i / n$ 套商法则。回忆一下：$f = u/v$ 时，

$$
\frac{\partial f}{\partial t} = \frac{(\partial u / \partial t) \cdot v - u \cdot (\partial v / \partial t)}{v^2}
$$

这里有两个关键点让推导不平凡：

- $\partial x_i / \partial x_j = \delta_{ij}$（Kronecker delta）—— 输入向量的每个元素是**独立变量**，所以 $x_5$ 对 $x_7$ 的偏导是 0，对自己的偏导是 1。
- $\partial n / \partial x_j \neq 0$ —— $n$ 通过 $s = \text{mean}(x^2)$ 依赖**所有** $x_k$，不能当常数。这正是 RmsNorm 反向"非对角"的根源。

代入：

$$
\frac{\partial y_i}{\partial x_j} = \frac{\delta_{ij} \cdot n - x_i \cdot (\partial n / \partial x_j)}{n^2} = \frac{\delta_{ij}}{n} - \frac{x_i}{n^2} \cdot \frac{\partial n}{\partial x_j}
$$

第一项是"假设 $n$ 是常数"的对角缩放。第二项是 $n$ 也会动带来的修正——改任意一个 $x_j$ 都会让 $n$ 变，而 $n$ 又出现在**每个** $y_i$ 的分母里（大家共享同一个分母）。第二项**稠密**——每一对 $(i, j)$ 都贡献非零值——这就是 Jacobian 不是对角矩阵的原因，也是所有 normalization-class 算子反向里都有"减去某种投影"的根本来源。

下面用链式法则算 $\partial n / \partial x_j$。$\sqrt{\,}$ 的导数来自幂法则
（对 $s^{1/2}$ 求导）：

$$
\frac{\partial n}{\partial s} = \frac{1}{2} s^{-1/2} = \frac{1}{2 \sqrt{s}} = \frac{1}{2n}
$$

（直觉：$s = n^2$ 在 $n$ 处的斜率是 $2n$；$n = \sqrt{s}$ 是它的反函数，所以
斜率取倒数 $1/(2n)$。）再配上 $\partial s / \partial x_j = 2 x_j / D$：

$$
\frac{\partial n}{\partial x_j} = \frac{\partial n}{\partial s} \cdot \frac{\partial s}{\partial x_j} = \frac{1}{2n} \cdot \frac{2 x_j}{D} = \frac{x_j}{D n}
$$

$\sqrt{\,}$ 带的 $\tfrac{1}{2}$ 正好和 $x^2$ 带的 $2$ 抵消——一个小巧合，
让 RmsNorm 的反向公式看起来干净。

代回：

$$
\frac{\partial y_i}{\partial x_j} = \frac{\delta_{ij}}{n} - \frac{x_i x_j}{D n^3}
$$

**链式法则展开 $\partial L / \partial x_j$：**

$$
\frac{\partial L}{\partial x_j} = \sum_i g_i \frac{\partial y_i}{\partial x_j} = \frac{g_j}{n} - \frac{x_j}{D n^3} \sum_i g_i x_i
$$

**用 $y = x/n$ 化简。** $x_i = y_i n$，所以 $\sum_i g_i x_i = n \sum_i g_i y_i$：

$$
\frac{\partial L}{\partial x_j} = \frac{g_j}{n} - \frac{y_j n}{D n^3} \cdot n \sum_i g_i y_i = \frac{1}{n} \left[ g_j - y_j \cdot \text{mean}(g \odot y) \right]
$$

**最终形式**（向量，per slice）：

$$
\boxed{\ \frac{\partial L}{\partial x} = \frac{1}{n} \left( g - y \cdot \text{mean}(g \odot y) \right)\ }
$$

**带 weight $w$.** Forward 变成 $z_i = y_i \cdot w_i$（输出是 $z$，不是 $y$），
upstream 现在是 $g_i = \partial L / \partial z_i$。需要算两个梯度：
$\partial L / \partial x$ 和 $\partial L / \partial w$。

**先算 $\partial L / \partial x_j$**：链式法则先穿过 $z$。因为 $w_i$ 不依赖 $x_j$，

$$
\frac{\partial z_i}{\partial x_j} = w_i \cdot \frac{\partial y_i}{\partial x_j}
$$

所以

$$
\frac{\partial L}{\partial x_j} = \sum_i g_i \cdot \frac{\partial z_i}{\partial x_j} = \sum_i (g_i w_i) \cdot \frac{\partial y_i}{\partial x_j}
$$

这就是无 weight 推导里把 $g$ 整体替换成 $g \odot w$ ——"替换 $g \rightarrow g \odot w$"
不是黑魔法，**就是链式法则的一步**。把替换代入之前的最终形式：

$$
\boxed{\ \frac{\partial L}{\partial x} = \frac{1}{n} \left( g \odot w - y \cdot \text{mean}(g \odot w \odot y) \right)\ }
$$

**再算 $\partial L / \partial w_k$**：从 $z_i = y_i \cdot w_i$ 可知，$z_i$ 只在
$i = k$ 时依赖 $w_k$，所以

$$
\frac{\partial z_i}{\partial w_k} = y_i \cdot \delta_{ik}
$$

per-slice 算：

$$
\frac{\partial L}{\partial w_k} = \sum_i g_i \cdot y_i \cdot \delta_{ik} = g_k \cdot y_k
$$

但 $w$ 的 shape 是 $(D,)$，在 forward 时被**广播**到每个 batch 位置——所有
$(B \cdot T)$ 个 slice 共用同一份 $w$。按反向广播规则（和 `Add` 里的
`unbroadcast` 同一套机制），梯度要在广播过的维度上 sum：

$$
\frac{\partial L}{\partial w} = \sum_{\text{batch}} g \odot y
$$

（具体来说：如果 $g, y$ 形状是 $(B, T, D)$，那 `dL/dw = (g * y).sum(dim=(0,1))` 得到一个 $(D,)$ 张量。）

**对 nanoops 的意义。** 反向**只需要 $y$ 和 $n$**（或者等价的 $\text{rsqrt} = 1/n$），
**不需要原始 $x$**。autograd 自动追踪的版本必须保存 $x$（因为底层每个
`mul`/`div` 都需要两个输入），而自定义 Function 每层 RmsNorm 省一份
$(\dots, D)$ 的 tensor——LLM 规模下是真金白银的显存。

**几何直觉。** $\text{mean}(g \odot y)$ 是 $g$ 在归一化方向 $y$ 上的投影系数。
RmsNorm 把这个方向"压平了"（任何沿 $y$ 方向的拉伸都被 $n$ 自动除掉），
所以这个方向上的 $g$ 不能传回去——必须先减掉再 scale。同样的"减去
归一化方向上的投影"结构在 softmax 和 LayerNorm 的反向里会再次出现。

### Softmax

最后一维长度 $D$，一个 slice 内 forward：

$$
y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}, \qquad \sum_i y_i = 1
$$

（实际实现里 $x$ 先减去 $\max(x)$ 再取 exp——纯数值稳定，不影响导数。）

**Jacobian.** 对 $y_i = e^{x_i} / Z$ 用商法则（$Z = \sum_k e^{x_k}$）：

$$
\frac{\partial y_i}{\partial x_j} = \frac{(\partial_j e^{x_i}) \cdot Z - e^{x_i} \cdot (\partial_j Z)}{Z^2}
$$

两个分量：

- $\partial_j e^{x_i} = \delta_{ij} \cdot e^{x_i}$（$e^{x_i}$ 只依赖 $x_i$）。
- $\partial_j Z = \partial_j \sum_k e^{x_k} = e^{x_j}$（求和导数挑出第 $j$ 项）。

代入：

$$
\frac{\partial y_i}{\partial x_j} = \frac{e^{x_i}}{Z} \delta_{ij} - \frac{e^{x_i} \cdot e^{x_j}}{Z^2} = y_i (\delta_{ij} - y_j)
$$

第一项是"按 $y_i$ 对角缩放"；第二项 $y_i y_j$ 是稠密的外积，**耦合每对
$(i, j)$**——和 RmsNorm 一样是非对角 Jacobian。

**链式法则展开 $\partial L / \partial x_j$：**

$$
\frac{\partial L}{\partial x_j} = \sum_i g_i \cdot y_i (\delta_{ij} - y_j) = y_j g_j - y_j \sum_i g_i y_i = y_j \left( g_j - \langle g, y \rangle \right)
$$

其中 $\langle g, y \rangle = \sum_i g_i y_i$ 是沿 softmax dim 的内积。

**最终形式**（向量，per slice）：

$$
\boxed{\ \frac{\partial L}{\partial x} = y \odot \left( g - \langle g, y \rangle \right)\ }
$$

代码上：`(g * y).sum(dim=dim, keepdim=True)` 算内积，然后 `y * (g - inner)`。

**对 nanoops 的意义。** 反向**只需要 $y$**（forward 输出），**不需要 $x$**——
和 RmsNorm 一样的省内存技巧。完整 $(D, D)$ Jacobian **从未被物化**：
化简后只剩一次 sum-reduction + 一次 elementwise mul 链。

**插一段：什么是 "null direction"？** 算子的 null direction 是这样一种输入
扰动方向——沿它移动输入，输出**完全不变**（数学定义：Jacobian 的零空间里
的向量）。因为 $L$ 通过 $y$ 依赖 $x$，沿 null direction 移动 $x$ 不改 $y$，
就不可能改 $L$，所以 $\partial L / \partial x$ 沿这个方向**必须严格为 0**。
backward 公式必须减掉 upstream grad 在 null direction 上的投影，**否则链式
法则会无中生有地编造梯度信号**——和"输出不变"这个事实矛盾。RmsNorm 的
`mean(g ⊙ y)` 和 Softmax 的 `⟨g, y⟩` 减法做的正是这个修正。

**和 ctx 内存的关联：什么时候 backward 能 "save y" 而不存 x？** 能不能只存
`y`（加上 `1/n` 这种标量元数据）取决于反向公式是否需要 forward $x \to y$
过程中**丢掉的信息**：

| Forward 形状 | Save `y` 可行？ | 原因 |
|---|---|---|
| Bijective（sigmoid, tanh） | ✓ | $y$ 唯一确定 $x$，没丢信息 |
| 有 null direction 且 grad 沿其为 0（RmsNorm, Softmax, ReLU²） | ✓ | 丢的信息在 null direction 上；backward 对它不变 |
| Backward 显式需要 $x$（如 Linear 的 $\partial L/\partial W = g \otimes x$） | ✗ | 必须存 $x$；只看 $y$ 推不出 backward 需要的量 |

**新 autograd Function 的设计配方**：先推 backward，然后检查公式里**还有没有 $x$**。
如果有，看能不能用 $y$ 表达（比如 sigmoid backward 把 $\sigma(x)(1-\sigma(x))$
改写成 $y(1-y)$）。如果实在改不掉，就必须存 $x$。

**和 RmsNorm 的对比。** 二者都是"减去归一化方向上的投影"结构：

| 算子 | scale factor | projection | reduction |
|---|---|---|---|
| RmsNorm | $1/n$（标量，per slice） | $y \cdot \text{mean}(g \odot y)$ | mean（除以 $D$） |
| Softmax | $y$（elementwise） | $\langle g, y \rangle$（标量，per slice） | sum（不除） |

scale 不同、reduction 不同，但**反向的 pattern 完全一致**：**沿算子 null
方向的梯度分量不传递；减掉后再 scale。**

Softmax 的 null 方向是常数向量 $\mathbf{1}$：所有 $x_i$ 同加一个常数，
$y$ 不变（分子分母里常数抵消）。$\langle g, y \rangle$ 这一步正好把
$g$ 在这个 null 方向上的投影减掉。

### Cross-entropy（融合 log-softmax + NLL）

单样本（类别维长度 $C$ 的一个 slice）：logits $x \in \mathbb{R}^C$，整数
target $t \in \{0, \dots, C-1\}$：

$$
L = -\log(\text{softmax}(x)_t) = -x_t + \log \sum_j e^{x_j}
$$

第二种形式是 **log-sum-exp 减去 target 的 logit**——这是 PyTorch
`F.cross_entropy` 用的融合视角，forward 里**根本不显式算 softmax**（LSE
里同样做 max-subtract 保证数值稳定）。

**Backward.** 逐项求导：

- $\partial(-x_t)/\partial x_j = -\delta_{jt}$（target 位置 −1，其余为 0）。
- $\partial(\log \sum_k e^{x_k})/\partial x_j = e^{x_j}/Z = y_j$（链式法则：$\log \to 1/Z$，求和导数挑出 $e^{x_j}$）。

两项相加：

$$
\boxed{\ \frac{\partial L}{\partial x} = \text{softmax}(x) - \text{one\_hot}(t)\ }
$$

完整的 $(C, C)$ softmax Jacobian **完全不需要物化**——$\log$ 贡献的 $1/Z$
和 softmax Jacobian **互相抵消**，整个反向化简成**一次 elementwise 减法**。
这是深度学习里最优雅的化简之一。

**为什么 fusion 抵消得这么干净。** 如果按 `nll(log(softmax(x)), t)` 朴素
组合：

- $\log$ backward：$g_y = -\frac{1}{y_t}\,\delta_{it}$（target 处有 $1/y_t$ 的奇异！）
- Softmax backward：$y \odot (g_y - \langle g_y, y \rangle)$

代入：$\langle g_y, y \rangle = (-1/y_t) \cdot y_t = -1$，于是
$g_x = y \odot g_y + y = y - \text{one\_hot}(t)$ —— target 位置的 $y_t$
正好抵消 $1/y_t$。结果完全相同，但朴素路径：

- 产生中间量 $-1/y_t$，**bf16 下 $y_t$ 很小时会下溢**；
- 物化 softmax Jacobian 的 $\langle g, y \rangle$ 内积；
- 走 3 个独立的 backward function（log + softmax + indexing）。

融合推导**预先看穿这个抵消**，不仅性能好，还避开了数值陷阱。

**内存和 ctx.** 只存 $y$（softmax 输出）和 $t$（target 索引）。整个反向是
**一次 elementwise 减法 + 在 target 位置做 scatter**：

```python
grad_x = y                  # 复制 softmax 输出
grad_x[range(N), t] -= 1    # 每个 target 位置减 1
```

**和前面算子的对比。**

| 算子 | Backward 化简 | "消失"的是什么 |
|---|---|---|
| RmsNorm | $(1/n)(g - y \cdot \text{mean}(g \odot y))$ | sqrt + 除法链 |
| Softmax | $y \odot (g - \langle g, y \rangle)$ | $(D, D)$ Jacobian 的物化 |
| ReLU² | $2 y g$ | mask op + 乘法链 |
| **Cross-entropy** | $y - \text{one\_hot}(t)$ | **softmax Jacobian 和 log 的 $1/y$ 一起抵消** |

Cross-entropy 是**最戏剧化的**：两个非平凡算子（$\log$ 和 $\text{softmax}$）
组合后化简成**一次减法**。这个抵消不是巧合——$\log \circ \text{softmax}$
正是 "log-likelihood" 的规范形式，它对原始 logits 的梯度**总是**"prediction
minus target"，对任何分类损失都成立。这就是 cross-entropy + softmax 成为
通用分类损失的根本原因。

**`ignore_index`**（nanchat 在 `gpt.py:477` 用 `ignore_index=-1`）：
$t = $ `ignore_index` 的位置对 loss 和 grad **贡献都为 0**。在 reduction
之前把对应行的 `grad_x` 归零即可。

**Reduction**（`'mean'` / `'sum'` / `'none'`）：`'mean'` 时把 `grad_x`
除以 $N_{\text{valid}}$（不算 `ignore_index` 位置），`'sum'` 不变，`'none'`
不缩放。

### Sigmoid

最简单的 "save y" 例子——bijective 的 elementwise 算子，反向公式可以**纯用
输出 y 表达**。

Forward:

$$
y = \sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Backward.** 直接对商形式求导。设 $u = 1 + e^{-x}$，则 $y = 1/u$：

$$
\frac{dy}{dx} = -u^{-2} \cdot \frac{du}{dx} = -\frac{1}{(1+e^{-x})^2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2}
$$

这是对的，但还用着 $x$。为了 ctx 省内存，**用 $y$ 替换 $x$**：由
$y = 1/(1+e^{-x})$ 得 $e^{-x} = (1-y)/y$，代回：

$$
\frac{dy}{dx} = \frac{(1-y)/y}{(1/y)^2} = (1-y) \cdot y = y(1-y)
$$

**最终形式：**

$$
\boxed{\ \frac{\partial L}{\partial x} = g \cdot y \cdot (1 - y)\ }
$$

**为什么是最干净的 "save y" 例子。** Sigmoid 是 **bijective**：给定
$y \in (0, 1)$，可以唯一解出 $x = \log(y/(1-y))$。所以 $y$ **完全包含**反向
所需的全部信息——再存 $x$ 是冗余。同样的内存（一份 $(\dots,)$ shape 的
tensor）存 $x$ 或 $y$ 都行，但**用 $y$ 写反向公式更简洁**。

对应到前面"Connection to ctx memory"那个三种情形表：sigmoid 是 **case A**
（bijective）的典型——没有 null direction、没有 cancellation 戏法，**$y$
代替 $x$ 直接就行**。

**Tanh** 同款模式。Forward $y = \tanh(x)$；反向通过 $(e^x - e^{-x})/(e^x + e^{-x})$
的商法则化简：

$$
\frac{d \tanh}{dx} = \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2} = 1 - \tanh^2(x) = 1 - y^2
$$

所以 $\partial L / \partial x = g \cdot (1 - y^2)$，也是纯用 $y$ 表达。
和 sigmoid 一样是 "bijective / save y" 的故事。

### Rotary positional embedding（旋转位置编码）

整个 nanoops 里**最对称的算子**：**反向就是 forward 把 $\sin$ 取反**——
来自旋转矩阵的正交性。

**Forward.** 4D 输入 $x \in \mathbb{R}^{B \times T \times H \times d}$，
最后一维劈成两半 $x_1 = x[\dots, :d/2]$ 和 $x_2 = x[\dots, d/2:]$。每对
$(x_1, x_2)$ 被角度 $\theta$（编码在 $(\cos, \sin)$ 里）旋转：

$$
\begin{pmatrix} y_1 & y_2 \end{pmatrix} = \begin{pmatrix} x_1 & x_2 \end{pmatrix} \cdot R(\theta), \qquad R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
$$

（PyTorch 约定：feature 维在最后，所以把 $(x_1, x_2)$ 当成**行向量**，
$R(\theta)$ 从右边乘上去。）

展开：

$$
y_1 = x_1 \cos\theta + x_2 \sin\theta, \qquad y_2 = -x_1 \sin\theta + x_2 \cos\theta
$$

输出沿最后一维 $\text{cat}(y_1, y_2)$。

**Backward.** 两条路推同一个结果——基础链式法则 和 矩阵代数捷径。

*基础路径。* 从 forward 公式直接读出四个偏导：

$$
\frac{\partial y_1}{\partial x_1} = \cos\theta, \quad
\frac{\partial y_1}{\partial x_2} = \sin\theta, \quad
\frac{\partial y_2}{\partial x_1} = -\sin\theta, \quad
\frac{\partial y_2}{\partial x_2} = \cos\theta
$$

链式法则（记 $g_1 = \partial L / \partial y_1$，$g_2 = \partial L / \partial y_2$）：

$$
\frac{\partial L}{\partial x_1} = g_1 \cdot \frac{\partial y_1}{\partial x_1} + g_2 \cdot \frac{\partial y_2}{\partial x_1} = g_1 \cos\theta + g_2 (-\sin\theta) = g_1 \cos\theta - g_2 \sin\theta
$$

$$
\frac{\partial L}{\partial x_2} = g_1 \cdot \frac{\partial y_1}{\partial x_2} + g_2 \cdot \frac{\partial y_2}{\partial x_2} = g_1 \sin\theta + g_2 \cos\theta
$$

*矩阵代数路径（同样结果，换个视角）。* Forward 是 $x$ 的**线性变换**：
$y = x \cdot R(\theta)$。行向量约定下链式法则给出
$\partial L / \partial x = g \cdot R(\theta)^T$。旋转矩阵正交：
$R(\theta)^T = R(\theta)^{-1} = R(-\theta)$，即

$$
R(\theta)^T = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}
$$

两条路给出同一个结果：

$$
\boxed{\ g_x^1 = g_1 \cos\theta - g_2 \sin\theta, \qquad g_x^2 = g_1 \sin\theta + g_2 \cos\theta\ }
$$

**这就是 forward 公式把 $\sin \to -\sin$**——同一段 kernel，反向旋转一次。
backward = "反向旋转"。

**为什么 $R^T = R^{-1}$。** 旋转矩阵是正交的，直接验证：

$$
R(\theta) R(\theta)^T = \begin{pmatrix} \cos & -\sin \\ \sin & \cos \end{pmatrix} \begin{pmatrix} \cos & \sin \\ -\sin & \cos \end{pmatrix} = \begin{pmatrix} \cos^2 + \sin^2 & \cos\sin - \sin\cos \\ \sin\cos - \cos\sin & \sin^2 + \cos^2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

$\cos^2 + \sin^2 = 1$ 这个恒等式做了所有的活。

**内存和 ctx。** 反向只依赖 $g$, $\cos$, $\sin$——**不依赖 $x$ 或 $y$**。
ctx 只存 $(\cos, \sin)$，这俩是预计算的查找表，整个 attention 调用里共用
（存的是引用，不是新分配）。`ApplyRotaryEmb.forward` **完全不存 $x$**，
是整个 nanoops 里 ctx 占用最小的算子之一。

$\cos$ 和 $\sin$ 由非可导的 `arange + outer + cos()/sin()` 预算出来（见
`nanchat/gpt.py:_precompute_rotary_embeddings`），所以 backward 对它们
返回 `None`。

**为什么比 RmsNorm / Softmax 干净这么多。** 那些算子的归一化**依赖数据**——
backward 需要 $y$（或 $x$）因为 Jacobian 取决于输入值。Rotary 的 Jacobian
就是 $R(\theta)$，**只取决于位置不取决于输入张量值**——所以**根本没必要
从 $x$ 里存什么**。

### 平凡反向算子：Where 和 Roll

为完整性补两个算子——它们的反向都是一行公式，但分别是前面 "Connection to
ctx memory" 表里 **case A** 和 **case B** 最干净的例子。

**Where**: $y = \text{where}(\text{cond}, a, b)$——`cond` 为真时取 $a$，
否则取 $b$。

反向按 mask 路由 upstream grad：

$$
\frac{\partial L}{\partial a} = g \odot \mathbf{1}[\text{cond}], \qquad
\frac{\partial L}{\partial b} = g \odot \mathbf{1}[\neg\text{cond}]
$$

梯度只流向被选中的那个操作数；另一个操作数在该位置接收 0。ctx 只存 `cond`
（bool，1 字节/元素）。

这是 ctx-memory 表里 **case B** 模式最纯粹的形式：存在 "null direction"
（在 cond=False 的位置扰动 $a$，或在 cond=True 的位置扰动 $b$，都不改 $y$），
而该方向上的梯度天然为零。bool mask 携带了 backward 需要的全部信息。

和 ReLU 的 `g * (x > 0)` 是同样的 "gradient gating" 思想，但**更通用**——
mask 可以来自任何条件（`x > threshold`、`mask_token != PAD`、attention 因果
mask 等），不限于 $x > 0$。

**Roll**: $y = \text{roll}(x, \text{shifts}, \text{dims})$——沿指定维度循环
移位。

反向应用逆排列：

$$
\frac{\partial L}{\partial x} = \text{roll}(g, -\text{shifts}, \text{dims})
$$

ctx **完全不存任何 tensor**——只存整数 `(shifts, dims)` 参数。Roll 是
bijective 的（排列必有逆），所以 $y$ 完全决定 $x$——但我们连 $y$ 都不需要，
因为反向公式只引用移位参数。这是 ctx-memory 表里 **case A** 加上额外亮点：
**ctx tensor 占用为零**。

`shifts` 和 `dims` 是 Python int/tuple，不可导；backward 对它们都返回 `None`。

### 模式：正交变换的反向是"免费"的

上面 Rotary 和 Roll 两节背后有一个结构同构值得点明。两者 forward 都是 $x$
的**线性变换**（乘某个矩阵 $M$，按 PyTorch 行向量约定，feature 维在最后）：

- Rotary: $y = x \cdot R(\theta)$，$R(\theta)$ 是 2×2 旋转矩阵
- Roll:   $y = x \cdot P$，$P$ 是置换矩阵

两个矩阵都是**正交矩阵**：$M M^T = I$，等价于 $M^T = M^{-1}$。$y = x M$ 的
链式法则是 $\partial L / \partial x = g \cdot M^T$，所以对正交 $M$：

$$
\frac{\partial L}{\partial x} = g \cdot M^{-1}
$$

一句话：**backward 就是 forward 的逆作用到 grad_output 上**。

| Op | Forward 矩阵 $M$ | $M^{-1}$ | 代码里的 Backward |
|---|---|---|---|
| Rotary | $R(\theta)$ | $R(-\theta)$ | forward 公式把 $\sin \to -\sin$ |
| Roll | $P_k$（移位 $k$） | $P_{-k}$（移位 $-k$） | forward 算子把 $k \to -k$ |

"免费"指的是什么：**不需要物化 Jacobian、不需要存 $x$ 或 $y$、不需要代数化简**。
**同一段 forward kernel 把唯一的变换参数取反，就是 backward**。

为什么这能成立：正交矩阵的转置**就是**它的逆——两者重合。所以链式法则要求的
"Jacobian 转置" = 我们熟悉的"逆操作"。对非正交线性算子（比如一般的矩阵乘，
$W$ 不方或不正交），$W^T \neq W^{-1}$，必须老老实实算转置矩阵乘——这就是
`Mm` 的反向里 $W^T$ 显式出现的原因（`grad_left = grad_output @ right.T`），
而那是**不能**化成"forward 把某个参数翻一下"的。

如果将来给 nanoops 加新的线性算子，发现它的矩阵是正交的——DCT、实数版 FFT
（差个 scale 因子）、Walsh-Hadamard 变换、有符号置换等——它的反向都套这个
模板。

### ScaledDotProductAttention

Tier 2 的**收官算子**——把两个 matmul + 一个 softmax 融合进一个
autograd.Function，反向是穿过三层的闭式 chain rule。和 `CrossEntropy`
（融合 logsoftmax + nll）是同一招，只是搬到 attention 上。

**Forward**（PyTorch 行向量约定，feature 在最后一维）：

$$
S = \frac{Q K^T}{\sqrt{d_k}} + M, \qquad P = \text{softmax}(S, \text{dim}=-1), \qquad O = P V
$$

Shape：$Q \in \mathbb{R}^{... \times L \times d_k}$，$K \in \mathbb{R}^{... \times S \times d_k}$，
$V \in \mathbb{R}^{... \times S \times d_v}$，$O \in \mathbb{R}^{... \times L \times d_v}$。

$M \in \{0, -\infty\}^{L \times S}$ 是**加性 mask**——保留位置为 $0$，屏蔽
位置为 $-\infty$。`is_causal=True`（下三角）和 `attn_mask`（bool $\to \{0, -\infty\}$，
或 float 直接用）在 softmax 前都统一成这种加性形式。凡是 $M_{ij} = -\infty$
的位置，对应的 $P_{ij} = e^{-\infty} / Z = 0$。

**为什么不用条件乘法 mask？** 你可能会想：既然 attention 分数有正有负，那让
$M$ 看 $S_{ij}$ 的符号，被掩位置取 $M_{ij} = -\text{sign}(S_{ij}) \cdot \infty$，
乘出来就永远是 $-\infty$，不也能拿到 masked $P_{ij} = 0$？数值上确实等效，但
在每个维度上都比加法严格更差：

- **$S_{ij} = 0$ 没有合法 $M$**：$0 \times \infty = $ NaN 跟符号无关。需要单独 if 分支或 epsilon 凑；加法版没这问题。
- **$M$ 失去独立性**：$M$ 现在必须先读 $S$ 才能构造，不再是 forward 前预算好的独立输入张量——pipeline 多一道依赖。
- **每元素多算子**：加法是一个 `+`；条件乘法是 `sign` + `where` + `*`。$L \times L$ 个 scores 上三倍 FLOPs，不是小钱。
- **反向多一条 $M(S, \text{mask})$ 路径**：除非用 `stop_grad` 屏蔽掉，否则梯度多绕一圈；用了 stop_grad 就等于在加法 mask 上面套了一层假装的"乘法"。
- **没有代数故事**：跟 log-sum-exp、cross-entropy 融合、FlashAttention LSE 都搭不上线。

$-\infty$ 对于任意 $S_{ij}$ 都是 $\max$ / softmax 的吸收元——没有符号分支、
没有 $S = 0$ 边界。继续用加法 mask。

加性方案的三个好处：

1. **概率正确**：$\text{softmax}(S + M)$ 在 $M = -\infty$ 处等价于
   "在未屏蔽位置上做 softmax"——剩下的位置 re-normalize 到行和 $= 1$。
   每一行仍是合法概率分布。
2. **数值精确**：$e^{-\infty} = 0$ 是精确的，不需要 `1e-9` 这种 fudge。
3. **梯度自动消**（下面 backward 推导的核心）：$P_{ij} = 0$ 通过 softmax-backward
   公式本身就传成 $\partial L / \partial S_{ij} = 0$——不需要在反向再做一次 mask。

更深的代数原因：在对数空间里
$\log\text{softmax}(S + M) = (S + M) - \log\sum e^{S + M}$ ——把 mask 加到
logit 上跟 log-sum-exp 归一化是**同一个操作**。同一招让 log-sum-exp 数值稳定、
让 cross-entropy 跟 logsoftmax 可以融合、让 FlashAttention 只存
$L = \log\sum e^S$ 就够 backward。

**Backward.** 给定上游 $g = \partial L / \partial O$，分三步穿过三个算子。
每一步要么是单次 matmul，要么是闭式 softmax-pullback，**不需要物化任何 Jacobian**。

*第一步：穿过 $O = PV$。* 跟 `Mm` 同模板：

$$
\frac{\partial L}{\partial V} = P^T g, \qquad \frac{\partial L}{\partial P} = g V^T
$$

（nanoops 的行向量约定下：$g \in \mathbb{R}^{L \times d_v}$，所以
$P^T g \in \mathbb{R}^{S \times d_v}$ 跟 $V$ 同 shape，$g V^T \in \mathbb{R}^{L \times S}$
跟 $P$ 同 shape。）

*第二步：穿过 $P = \text{softmax}(S)$。* 这就是附录前面 `Softmax` 一节推过的
softmax-backward 公式——nanoops 用「存 $y$ 不存 $x$」存 P 这个 trick。
按行：

$$
\frac{\partial L}{\partial S} = P \odot \left(\frac{\partial L}{\partial P} - \text{sum}\!\left(P \odot \frac{\partial L}{\partial P}, \text{dim}=-1, \text{keepdim}=\text{True}\right)\right)
$$

sum-correction 那一项是 softmax-backward「稠密」的原因——每个输出概率都
依赖所有输入分数（分母共享），所以梯度必须减掉一个按行共享的标量。

**mask 抵消就藏在这一行里。** 注意外面那个 $P$ 因子——它乘到
$\partial L / \partial S$ 的**每一个**位置上。在被掩位置（forward 时
$P_{ij} = 0$）：

$$
\left(\frac{\partial L}{\partial S}\right)_{ij} = \underbrace{P_{ij}}_{= 0} \cdot \bigl(\dots\bigr) = 0 \qquad \text{当 } M_{ij} = -\infty
$$

mask 的效果**自动**穿过 backward —— `P` 把零从前向带过来，再以零梯度发出去。
**不需要在反向再做一次 mask、不需要 `masked_fill`、不需要 if 分支**。这就是
softmax + mask 这一对儿干净的原因：$P$ 同时充当了前向输出**和**反向门控。

*第三步：穿过 $S = (QK^T / \sqrt{d_k}) + M$。* **$M$ 跟 $Q, K$ 无关**（它是
输入张量，不是它们的函数），所以 $\partial M / \partial Q = \partial M / \partial K = 0$。
这意味着 `+M` 这条链式法则**不带任何项**——mask 自动消失，剩下的就是两次带
scale 的 matmul-backward：

$$
\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S} \cdot K, \qquad \frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \left(\frac{\partial L}{\partial S}\right)^T \! Q
$$

这跟第二步的"抵消"是**两件不同的事**，要分清：

- **第二步**（$\partial L / \partial S_{ij} = 0$ 在被掩 $(i, j)$）：**因为 $P_{ij} = 0$**。阻止被掩位置的 $S$ 通过第三步的 matmul-backward 传梯度。
- **第三步**（`+M` 不带额外项）：**因为 $M$ 跟 $Q, K$ 无关**。阻止 mask 自身另开一条传回 $Q, K$ 的梯度路径。

（如果 $M$ 是 $Q, K$ 的函数，第三步的抵消就失败，$\partial L / \partial Q$ 会多
出一项；但第二步的抵消仍然成立，所以那个多出的项也只在未掩位置传梯度。两个
保证彼此独立。）

把两件事合起来：第二步保证 $\partial L / \partial S$ 在被掩位置为 0，第三步
保证 matmul-backward 是回到 $Q, K$ 的**唯一**路径（没有 `+M` 旁路），所以
被掩位置对 $\partial L / \partial Q$、$\partial L / \partial K$ 的贡献是
$0 \cdot \text{something} = 0$。

把闭式最终结果框起来：

$$
\boxed{
\begin{aligned}
\frac{\partial L}{\partial V} &= P^T g \\
\frac{\partial L}{\partial Q} &= \tfrac{1}{\sqrt{d_k}} \,\bigl[P \odot (g V^T - \text{sum}(P \odot g V^T, \text{dim}=-1, \text{keepdim}))\bigr] \cdot K \\
\frac{\partial L}{\partial K} &= \tfrac{1}{\sqrt{d_k}} \,\bigl[P \odot (g V^T - \text{sum}(P \odot g V^T, \text{dim}=-1, \text{keepdim}))\bigr]^T \! \cdot Q
\end{aligned}
}
$$

**内存和 ctx。** `ScaledDotProductAttention.forward` 存 $(Q, K, V, P)$ 四个
张量。最贵的是 $P$，shape $(B, H, L, S)$——这是**朴素**内存策略：
$O(B \cdot H \cdot L \cdot S)$，self-attention 时就是 $O(B \cdot H \cdot L^2)$。
FlashAttention 优化的就是这一项：只存每行的归一化 stats（log-sum-exp 和 max，
都是 $O(B \cdot H \cdot L)$），backward 一块一块地重算 $P$，ctx 直接降一个
$S$ 量级。nanoops Tier 3 计划用 Triton 写这版（见 TODO）。

**GQA (Grouped Query Attention).** 当 `enable_gqa=True` 且 $H_q > H_{kv}$ 时，
forward 沿 heads 维度用 `repeat_interleave` 把 $K, V$ expand 出 $G = H_q / H_{kv}$ 倍。
backward 再把扩展后的梯度 `unflatten + sum` 折回 $H_{kv}$ 个 heads——
`repeat_interleave` 的伴随就是 `unflatten + sum` 这一对。ctx 存的是**未 expand**
的 $K, V$（backward 里重新 expand 很便宜），免得多付一份 $G\times$ 的内存。

**为什么这个算子推起来这么"舒服"？** 每一步用的反向**附录前面都推过了**：
matmul (Mm)、softmax (Softmax)、按元素乘法 (Mul)。SDPA 不过是把三个串起来——
"融合"的意义在于它们坐在**一个** `autograd.Function` 里：PyTorch autograd
引擎看到的是一个节点而不是一串，ctx 我们手动选（只存 $Q, K, V, P$），
不让 autograd 把每一个中间张量都保留下来。
