# nanoops Triton Kernels

A walkthrough of writing fused GPU kernels in Triton, targeting the
RTX 3090. Written for someone new to Triton who wants to understand
**why the fusion choices in `nanoops/triton_*.py` look the way
they do** — block sizes are picked against actual 3090 budgets,
fusion boundaries follow from the chip's compute/bandwidth ratio,
and what to keep in registers vs save in ctx vs recompute is driven
by the same arithmetic-intensity tradeoffs.

The doc starts from raw hardware numbers (SM count, shared mem,
register file) and works up to fused-kernel design. By the end you
should be able to read any kernel in the feature-split Triton modules
and see why every choice was made.

> 中文版：[TRITON_zh.md](TRITON_zh.md)

---

## Chapter 1 — Target hardware: RTX 3090 (Ampere SM_86 / GA102)

Every block-size choice, every "fits in shared memory" check, every
"why is this a separate kernel and not one giant fused kernel" trade-off
below is keyed to these numbers. If you port to a different GPU
(RTX 4090 / A100 / H100 / consumer Ada), most kernels will run, but the
tile sizes are likely no longer optimal.

### Whole-chip totals

| Item                | Value                |
| ------------------- | -------------------- |
| **SMs**             | **82**               |
| Total FP32 cores    | 82 × 128 = 10,496    |
| Total Tensor cores  | 82 × 4 = 328         |
| Total register file | 82 × 64K = 5.4 M regs|
| Total shared mem    | 82 × 100 KB = 8.2 MB |
| **Device memory**   | **24 GB GDDR6X**     |
| HBM bandwidth       | **936 GB/s**         |
| Compute capability  | **8.6**              |

### Per-SM resources

| Resource                | Value           | Notes                                     |
| ----------------------- | --------------- | ----------------------------------------- |
| L1 / shared memory      | 128 KB combined | zero-sum split: `shared_carveout + L1 = 128 KB`; sm_86 allowed shared values are `{0, 8, 16, 32, 64, 100} KB` — selecting 100 KB shared leaves only 28 KB L1 |
| Shared mem **per SM**    | **100 KB**     | max carveout = total pool shared by all co-resident blocks |
| Shared mem **per block** | **100 KB**     | hardware cap on what a single block can request — equal to the per-SM pool size (so a single block can hog all of it if it wants, but that means blocks/SM = 1) |
| Registers per SM        | 65,536 × 32-bit | = 256 KB; per-thread allocation, all active threads on SM share this pool → caps both active-thread count AND blocks/SM (via thread count) |
| Max threads / SM        | 1,536           | = 48 warps; caps blocks/SM as `1536 / threads_per_block` |
| Max blocks / SM         | 16              | hardware cap on number of co-resident blocks regardless of resources |
| Tensor cores            | 4 (3rd gen)     | bf16 / fp16 / tf32 / int8                 |
| FP32 cores              | 128             |                                           |
| Warp size               | 32 threads      |                                           |

### Per-block limits

(NVIDIA's full term is "thread block"; "block" is the standard shorthand
used in CUDA APIs — `blockDim`, `blockIdx`, `<<<grid, block>>>` — and in
the rest of this doc.)

| Limit              | Value      |
| ------------------ | ---------- |
| Max threads        | 1024 (32 warps) |
| Max shared memory  | **100 KB** |
| Max registers/thread | 255 (more → spill to local memory, slow) |

### Compile-time vs runtime: what's frozen when

A central fact that makes the rest of this chapter make sense:
**almost everything that constrains occupancy is frozen at compile
time**, not at launch.

| Quantity                  | Decided when    | Decided by              |
| ------------------------- | --------------- | ----------------------- |
| `threads_per_block`       | **compile time** | user (via `num_warps` × 32 for Triton, or `<<<grid, block>>>` for CUDA C++) |
| `reg/thread`              | **compile time** | compiler (Triton / NVCC) after static analysis of the kernel body |
| `shared_mem / block`      | **compile time** | compiler (static sum of all `tl.load` buffers, accumulators, `num_stages` pipeline depth) |
| `grid_dim` (block count)  | runtime         | user — `kernel[grid](...)` |
| `blocks/SM`               | runtime         | hardware (GigaThread Engine); see exact formula below |
| Which SM each block lands on | runtime      | hardware (GigaThread Engine) |

What this implies:

- A single Triton kernel definition compiles to **N separate binaries**
  if you sweep `num_warps`, `num_stages`, or `BLOCK_*` via
  `triton.autotune` — each binary has its own frozen `reg/thread` and
  `shared/block`.
- `reg/thread` is *not* something the user can pick directly; you only
  influence it indirectly by writing a smaller / larger / more
  pipelined kernel.
- Once compiled, occupancy is **deterministic per kernel** — the
  hardware doesn't reshuffle resources at runtime.

The exact formula the GigaThread Engine uses, in warp terms (warp =
the actual scheduling unit, **NOT** thread):

```
warps_per_block = ⌈threads_per_block / 32⌉
regs_per_warp   = ⌈(32 · reg_per_thread) / 256⌉ · 256       # 256-align
regs_per_block  = warps_per_block · regs_per_warp

blocks/SM = min(
    16,                                                   # ① hardware block cap
    ⌊48 / warps_per_block⌋,                               # ② per-SM warp cap (= 1536 threads)
    ⌊100 KB / shared_per_block⌋,                          # ③ shared mem pool
    ⌊65,536 / regs_per_block⌋,                            # ④ register file pool
)

resident_warps = blocks/SM · warps_per_block
occupancy      = resident_warps / 48
```

Subtleties:
- **`⌈ / 32⌉` (warp rounding)** — if `threads_per_block` isn't a
  multiple of 32, the last warp still claims a full 32-lane slot
  (with inactive lanes).
- **`⌈ / 256⌉ · 256` (register allocation granularity)** — Ampere
  allocates registers per-warp in chunks of 256. So 32 threads each
  using 33 registers don't claim `32·33 = 1,056`; they claim
  `⌈1,056/256⌉·256 = 1,280` — 224 register slots wasted. The
  effect is invisible at well-tuned `reg_per_thread` values (multiples
  of 8 land cleanly on the 256-per-warp boundary), but real when the
  compiler picks an oddly sized count.
- **`⌊ ⌋` (block count integer)** — the SM can't host 1.5 blocks.
- **② is fundamentally a warp cap** — the 48-warp scheduler limit is
  the hardware constraint; the "1536 thread" framing is just
  `48 × 32`.

### FMA vs MMA — the two primitive ops

Throughput tables below count work in **FMA / MMA / FLOPs**. Worth
nailing down what each means first.

**FMA — Fused Multiply-Add (scalar)**
- The basic op of a CUDA core: `d = a * b + c`
- "Fused" = one rounding step at the end (not two), better numerics
- 1 FMA = **2 FLOPs** (one multiply + one add)
- Every modern CPU/GPU FPU does FMA in 1 cycle; doing `a*b` and `+c`
  separately is the same speed but worse numerics

**MMA — Matrix Multiply-Accumulate (matrix)**
- The basic op of a Tensor core: `C = A @ B + C` where A, B, C are
  small matrices
- Ampere 3rd-gen Tensor core: shape is **16×16×16** (one MMA does the
  product of a 16×16 matrix and a 16×16 matrix, accumulating into a
  16×16 result)
- 1 such MMA = 16 × 16 × 16 = **4,096 multiply-adds = 8,192 FLOPs**
- Issued as a single warp-level `mma.sync` PTX instruction. The 4,096
  multiply-adds are pipelined across multiple cycles per Tensor core —
  per-SM sustained throughput is **1,024 BF16 FLOPs/cycle** (see peak
  compute table below), i.e. each Tensor core does 256 BF16 FLOPs/cycle,
  so one 16×16×16 MMA's work (8,192 FLOPs) amortizes over **~32 cycles**
  per Tensor core.

So one MMA does **the same work as 4,096 FMAs**. That's the asymmetry
that justifies the architectural split: a few Tensor cores running
MMAs >> many CUDA cores running FMAs.

### RTX 3090 peak compute

Boost clock 1695 MHz, 82 SMs.

| Precision / unit                 | Per-SM throughput          | Whole-chip peak     |
| -------------------------------- | -------------------------- | ------------------- |
| FP32 (CUDA cores)                | 128 cores × 2 FLOPs/cycle  | **35.6 TFLOPS**     |
| TF32 (Tensor cores)              | 512 FLOPs/cycle            | **71 TFLOPS**       |
| **FP16 / BF16 (Tensor cores)**   | 1024 FLOPs/cycle           | **142 TFLOPS**      |
| INT8 (Tensor cores)              | 2048 ops/cycle             | 284 TOPS            |

**Memory side**: 936 GB/s HBM bandwidth. Compute-to-bandwidth ratio at
bf16 = 142 TFLOPS / 936 GB/s = **152 FLOPs per byte**. Any op below
that ratio is bandwidth-bound; above it is compute-bound.

**How arithmetic intensity is computed** (for a matmul `C = A @ B`,
shapes `(M, K) @ (K, N) → (M, N)`, all bf16):

- **FLOPs** = `2·M·N·K` — each output element needs K multiply-adds (each
  multiply-add = 2 FLOPs)
- **Bytes** = `2·(M·K + K·N + M·N)` — read A + read B + write C, 2 bytes
  per bf16 element
- **AI** = FLOPs / Bytes = `M·N·K / (M·K + K·N + M·N)` (the 2's cancel)

The break-even at 152 FLOPs/byte (= 142 TFLOPS / 936 GB/s) tells you
whether the matmul is bottlenecked by GPU compute (above) or HBM
bandwidth (below).

For nanchat-d24 matmuls at training shapes (M=2048, K=1536; N varies):

| Matmul site                | Shape (M, K, N)       | AI (FLOPs/byte) |
| -------------------------- | --------------------- | --------------- |
| c_q / c_k / c_v / attn c_proj | (2048, 1536, 1536) | **558**         |
| MLP c_fc (D → 4D)          | (2048, 1536, 6144)    | **770**         |
| MLP c_proj (4D → D)        | (2048, 6144, 1536)    | **770**         |

All comfortably above the 152 FLOPs/byte break-even → **compute-bound**.

**RMSNorm and other elementwise/reduction ops are heavily
bandwidth-bound** — about 100× below break-even. For one row of D
elements:

- FLOPs ≈ `4·D` (mean(x²) = ~2D, then `x · rms_inv · weight` = 2D)
- Bytes (bf16) ≈ `4·D` (**fused** kernel reads x once = 2D, writes y = 2D;
  weight read is `D` per kernel, amortized away). A naive 2-pass
  implementation would read x twice → 6D bytes, AI = 0.67 FLOPs/byte.
  Triton/CUDA kernels keep the row of x in registers across the two
  passes, so HBM only touches it once.
- **AI ≈ 1 FLOPs/byte** — vs 152 break-even, this is bandwidth-bound
  by ~100×

Summary table (training-shape scale):

| Op                          | AI (FLOPs/byte) | Bound      |
| --------------------------- | --------------- | ---------- |
| MLP / QKV matmul            | 558 – 770       | compute    |
| SDPA (B=1, L=2048)          | ~114 naive / ~1024 Flash (see SDPA section below) | bandwidth → compute |
| **RMSNorm**                 | **~1**          | bandwidth  |
| Elementwise (add, relu)     | ~0.5            | bandwidth  |
| Memcpy H2D / D2H            | 0               | bandwidth  |

**Why this motivates the fusion stack:** standalone RMSNorm kernel
does ~4·M·D bytes of HBM traffic for ~0 compute return. Fusing norm
into the adjacent matmul kernel (`fused_mlp` on the mlp side)
keeps the normalized intermediate in registers, saving **4·M·D bytes**
of HBM traffic (norm output write + matmul input re-read) — pure
bandwidth win on a bandwidth-bound op. The attn-side
`norm_qkv_projection` currently materializes `x_hat` with the shared
RMSNorm kernel before Q/K/V projection because that benchmarks slightly
faster at d24.

**SDPA: why Flash Attention exists.** Same AI lens shows the SDPA
case has *two* numbers depending on whether the `(L, L)` P matrix
gets materialized.

For nanchat d24 (B=1, H=12, L=2048, D_head=128), forward attention
(both matmuls: `Q@K^T` and `attn@V`, plus softmax):

| Implementation                          | Total FLOPs | Total bytes | AI                |
| --------------------------------------- | ----------- | ----------- | ----------------- |
| Naive SDPA (materialize P to HBM)       | 25.8 G      | ~226 MB     | **~114 FLOPs/byte** (bandwidth-ish) |
| Flash SDPA (P stays in registers)       | 25.8 G      | ~25 MB      | **~1024 FLOPs/byte** (compute-bound) |

Byte breakdown: at this scale `Q + K + V + O ≈ 25 MB` (4 tensors of
`B·H·L·D = 12·2048·128` bf16 each). The P matrix is `B·H·L·L = 100 MB`,
and naive SDPA writes P then reads it back from HBM (~200 MB), so the
naive total is ~225 MB vs Flash's ~25 MB. Flash's online softmax +
tile streaming keeps P in registers, flipping SDPA from bandwidth-
bound (~114 < 152 break-even) to compute-bound (~1024 >> break-even).

This is why Flash Attention is 2-4× faster than naive SDPA — bytes drop
~9× while FLOPs change only slightly:

- **Forward FLOPs**: nearly identical (~5% more on Flash for the
  online softmax's rescale bookkeeping).
- **Backward FLOPs**: ~33% MORE on Flash, because Flash recomputes
  the P matrix from a fresh `Q @ K^T` instead of saving P. Classic
  FLOPs-for-memory trade.

Net effect: even though Flash backward does ~30% more arithmetic,
the ~10× HBM bandwidth saving dominates wall time (the workload was
bandwidth-bound, so saving bandwidth trumps spending FLOPs). Sliding-
window attention shrinks both bytes and FLOPs proportionally (band
size `W` instead of full `L`), but the Flash-vs-naive ratio stays
the same.

### Tensor cores vs CUDA (FP32) cores

These are **different physical units**, optimized for different work
patterns. Understanding the split changes how you reason about kernel
throughput.

| Unit              | Per SM | What each does (sustained)              | SM throughput               |
| ----------------- | ------ | --------------------------------------- | --------------------------- |
| **FP32 cores**    | 128    | 1 scalar FMA/cycle each = 2 FLOPs/cycle each | **256 FP32 FLOPs/cycle**    |
| **Tensor cores**  | 4      | sustained 256 BF16 FLOPs/cycle each (one 16×16×16 MMA amortized over ~32 cycles) | **1024 BF16 FLOPs/cycle**   |

So 4 Tensor cores deliver ~4× the bf16 throughput that 128 FP32 cores
deliver in fp32 (1024 / 256). Going bf16/fp16 and using `tl.dot`
unlocks that 4×.

**Why only 4 Tensor cores per SM:**

1. **Silicon area.** One Tensor core is ~tens of FP32-cores worth of
   transistors. Four already eats a big slice of the SM die.
2. **Scheduling matches.** An Ampere SM has **4 warp schedulers**;
   each cycle, each scheduler issues one `mma.sync` to one Tensor
   core. 4 schedulers × 4 Tensor cores = perfect 1:1 — adding a fifth
   Tensor core would just sit idle.
3. **Data bandwidth.** At peak (1024 BF16 FLOPs/cycle = 512
   multiply-adds/cycle per SM), even with full tile reuse from
   registers the 4 Tensor cores together still need operand data
   on the order of ~1 KB/cycle out of shared memory — already a
   substantial fraction of the per-SM SMEM port bandwidth.

### Warps, warp schedulers, and which cores actually run

A **warp** = 32 threads. The fundamental scheduling unit.

```
                    SM (one of 82 on RTX 3090)
   ┌──────────────────────────────────────────────────────────┐
   │  4 warp schedulers (round-robin)                          │
   │     │                                                     │
   │     ├──► dispatch warp instruction every cycle ─►         │
   │     │                                                     │
   │     │      ┌─────────────────────────────────────┐       │
   │     ├─────►│ 128 FP32 cores (= 4 lanes of 32)    │       │
   │     │      │ scalar ops: fma, sin, add, ld, st…  │       │
   │     │      └─────────────────────────────────────┘       │
   │     │                                                     │
   │     │      ┌─────────────────────────────────────┐       │
   │     └─────►│ 4 Tensor cores                      │       │
   │            │ `mma.sync` → matrix multiply-acc    │       │
   │            └─────────────────────────────────────┘       │
   │                                                           │
   │  Register file: 65,536 × 32-bit (shared by all warps)    │
   │  Shared memory: 100 KB (per thread block)                 │
   └──────────────────────────────────────────────────────────┘
```

**Key relationships:**

- **1 warp issues 1 instruction** at a time. Either to FP32 cores
  (32 lanes × 1 fp32 op = 32 ops in one cycle, one per thread) or to
  a Tensor core (the whole warp cooperates to produce one matrix `mma`).
- A **`mma.sync` instruction is warp-level**, not thread-level — all
  32 threads in the warp participate. Their per-thread register
  fragments concatenate into the (16, 16) matrix tiles the Tensor
  core reads.
- Up to **48 warps can be resident** per SM (= 1536 threads). The
  scheduler picks among them: when one warp stalls on a load, others
  fill the gap. This is **latency hiding** — the reason GPUs don't
  need fancy out-of-order execution like CPUs.
- **Block size matters for occupancy.** A block with 256 threads =
  8 warps. 100 KB shared memory per block → only 1 block fits per SM
  → 8 warps active out of 48 possible → low occupancy → fewer warps
  to hide latency with → throughput drops. Tighter tile sizes (less
  shared mem) let 2-3 blocks co-reside, raising occupancy.

**Triton vs CUDA C++ for warp control:**

- CUDA C++: you explicitly call `__syncwarp()`, manage `mma::sync`,
  pick fragment layouts.
- Triton: hides warps. You only see `tl.program_id` (block-level)
  and `tl.arange` (compile-time vector). Triton's compiler decides
  how to map your vector ops onto 32-thread warps and which lane
  reads which element. You only control:
  - `num_warps=4` (default) — how many warps per block (4 warps =
    128 threads). Bigger = more parallelism per block but fewer
    blocks can co-reside.
  - `num_stages=2` — pipelining depth for `tl.load`/`tl.dot`.

### What this means for our kernels

- **Shared memory budget is the binding constraint.** Most of our
  kernel tile sizes (BLOCK_M, BLOCK_N, BLOCK_K) were chosen so the
  combined working set (input tiles + accumulator) fits in ≤ 60–80 KB
  per block, leaving headroom for the runtime and not forcing a
  one-block-per-SM occupancy collapse.

- **3090's shared memory is ~half of A100 (164 KB) and ~third of
  H100 (228 KB).** This is why Flash Attention on 3090 needs smaller
  tile sizes than the canonical Flash kernels published with H100
  numbers. Larger Q/K/V tiles spill or refuse to compile.

- **82 SMs is the rough minimum grid size to saturate the GPU.**
  Anything that launches fewer programs leaves SMs idle. For
  per-row-reduction kernels (RMSNorm, softmax, fused_add_norm),
  we tile along the M (batch × seq) dim because M is typically
  ≥ 2048 in training — comfortable parallelism.

- **No FA3-style TMA / async-copy hardware.** Hopper-specific
  techniques (TMA, distributed shared memory, warp-specialization
  via `wgmma`) aren't usable on Ampere. Our Flash SDPA kernel is the
  classic Triton tutorial pattern, not the FA3 pattern.

- **Fuse elementwise into `tl.dot` epilogues.** Standalone elementwise
  kernels are bandwidth-bound (see RMSNorm / add / relu rows in the AI
  table, all ≪ 152 break-even) — the real cost is HBM round-trip plus
  kernel-launch overhead, not FLOPs. The FP32 cores are mostly idle
  during a `tl.dot`-dominated kernel anyway (a separate elementwise
  pass on them would still cap at ~1/4 of BF16 Tensor peak, 35.6 / 142
  TFLOPS), so folding the elementwise op into the matmul epilogue is a
  free side-channel: it saves the elementwise's HBM round-trip (matmul's
  HBM cost is unchanged), no extra launch, no FLOP/s contention with
  the Tensor cores.

---

## Chapter 2 — FusedAddNorm: a worked example of 2-op fusion

The simplest fused kernel in this repo. It exists purely as a learning
artifact — nanchat's production blocks fold the RMSNorm directly into
the adjacent matmul (see `norm_qkv_projection` on the attn side,
`fused_mlp` on the mlp side), so a standalone `add → norm`
op boundary doesn't actually appear in the hot path. But every
pattern this kernel uses is a building block of those bigger fused
kernels, so it's the cleanest place to learn them.

### What it computes

Mathematically:
```
summed = x + residual
y      = summed · rsqrt(mean(summed²) + eps) · weight    # weight optional
```

API returns both `y` and `summed` to the caller:
- `y` → flows to the next block's matmul input
- `summed` → the next block's residual stream (caller doesn't need to
  recompute `x + residual` later)

Public API shape contract:
```python
def fused_add_norm(x, residual, norm_weight, eps=1e-6) -> (y, summed)
# x, residual: contiguous CUDA tensors shaped (M, D)
# norm_weight: optional contiguous CUDA tensor shaped (D,)
```

Four Triton kernels make this work end-to-end through autograd —
one for fwd, plus a 3-kernel backward setup (primary inline kernel
+ a 2-kernel D-split fallback for large D where inline would spill):

| Kernel | Grid | Role |
|---|---|---|
| `_fused_add_norm_fwd_kernel` | 1D over M | fwd: writes `y` + `summed` + `rms_inv` |
| `_fused_add_norm_inline_bwd_kernel` | 1D over M | bwd **primary**: full row per tile, inline inner reduction, writes `d_summed` (+ `dnw_partial`) |
| `_fused_add_norm_inner_bwd_kernel` | 1D over M | bwd fallback stage 1: pre-computes `inner[m]` |
| `_fused_add_norm_split_bwd_kernel` | 2D over (M, D) | bwd fallback stage 2: writes `d_summed` + `dnw_partial` |

### 2.1 Forward kernel

Single-pass design — one program processes one `(BLOCK_M, BLOCK_D)`
tile, loads everything, computes both the reduction and the
per-element output, writes back.

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
    tl.store(summed_ptr + offs, summed, mask=mask)        # ← for caller's residual stream + bwd

    # bf16 squared products feed an fp32 accumulator via `dtype=`. Saves
    # an intermediate fp32 register tile vs `summed.to(fp32)` and squaring
    # in fp32, at the cost of doing the square in bf16 (precision loss
    # ~1.5e-1, within test atol).
    sum_sq = tl.sum(summed * summed, axis=1, dtype=tl.float32)
    rms_inv = tl.rsqrt(sum_sq / D + eps)               # rsqrt.approx.f32 (one PTX instruction)
    tl.store(rms_inv_ptr + rows, rms_inv, mask=rows < M)  # ← needed by bwd

    y = summed * rms_inv[:, None]                       # bf16 * fp32 → fp32 (auto-promote)
    if HAS_NW:
        nw = tl.load(nw_ptr + cols, mask=cols < D, other=0.0)
        y = y * nw[None, :]                             # fp32 * bf16 → fp32 (auto-promote)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)
```

Patterns to notice:
- **`BLOCK_D = next_power_of_2(D)`** with a `col_mask` to handle
  non-pow2 `D` (nanchat d24 has `D=1536` → `BLOCK_D=2048`, 512 lanes
  masked). Triton's `tl.arange` requires a pow-of-2 length.
- **2D mask** = row mask × col mask, applied to every load/store. Out-of-bounds
  loads return `0` (the additive / multiplicative identity for the
  downstream `sum_sq` — chosen deliberately).
- **`HAS_NW: tl.constexpr`** branches at compile time. The
  no-affine-weight path simply skips the `nw` load + multiply; one binary
  is compiled per `(HAS_NW=True, HAS_NW=False)` value.
- **`element_ty` cast at store** lets the same kernel handle any
  caller dtype (bf16 / fp16 / fp32) without modification.
- **`summed` is written to HBM mid-kernel**, not at the end. This is
  intentional: the caller's residual stream needs the bf16 value, and
  the downstream compute reuses that same bf16 register tile (feeding
  the `tl.sum(dtype=fp32)` reduction and the `summed * rms_inv` step,
  where Triton auto-promotes to fp32) without re-loading from HBM.

### 2.2 Backward path: register budget chooses the kernel

There are two viable backward kernel shapes. Which one runs depends
on whether the **inline** kernel's per-program tile fits the Ampere
255 fp32 reg/thread spill cap. The dispatch lives in
`_fused_add_norm_bwd_impl` and is driven by `TileConfig.fits_reg_budget`.

#### The simple thing first: one kernel, full row per program

The primary path is `_fused_add_norm_inline_bwd_kernel`: a 1D grid
over M with `BLOCK_D = next_pow_of_2(D)` so the **full row** lives in
one tile. The per-row reduction `inner[m] = mean_d(g_eff · y_norm)`
is then computed in registers — no precompute kernel, no inner HBM
buffer, one kernel launch.

This works because at typical nanchat shapes the tile is small:

```
   ~5 fp32 tiles  ×  BLOCK_M  ×  BLOCK_D   /   (num_warps × 32)
                                                    ≤ 255 regs/thread
```

`_pick_fused_add_norm_tile_config(M, BLOCK_D, n_live_tiles=N)` solves for the largest
BLOCK_M that fits. `N=5` for HAS_NW=True (`y_norm, g_eff, dy_t,
d_ext, d_summed` alive at peak), `N=4` for HAS_NW=False (`y_norm`
aliases `src` and `g_eff` aliases `dy_t` when there's no per-channel
weight). At `D=1536, HAS_NW=True` it picks `BLOCK_M=4, num_warps=8` →
160 regs/thread. At `D=4096, HAS_NW=True` it picks `BLOCK_M=1,
num_warps=4` → also 160 regs/thread. Comfortably under the cap.

#### The bigger gun: 2-kernel D-split fallback

The inline path breaks once `BLOCK_D` is large enough that even
`BLOCK_M=1, num_warps=16` (the cap) overflows. The crossover depends
on n_live_tiles:
- HAS_NW=True (5 live tiles): `BLOCK_D > 16384` (i.e. `D > 16K`) —
  model estimates ≥320 regs/thread, bench confirms ~10× slowdown
  from local-memory spill.
- HAS_NW=False (4 live tiles): `BLOCK_D > 32768` (i.e. `D > 32K`) —
  basically never triggers in any realistic model (nanchat tops out
  at D=8192 even at depth=128).

For those cases the dispatch falls back to a 2-kernel pair that
**splits along D**, dropping the per-program tile back to manageable
size. The same shape choice (`BLOCK_M=32, BLOCK_D=64`) that would
spill at full-D works here because each program only owns a 64-column
slice instead of the full 1536+ row:

```
   5 tiles × 32 × 64  ≈  10K fp32 / program
   10K  /  128 threads ≈ 80 regs/thread       ← fits
```

But splitting along D introduces a new problem. The per-row reduction
`inner[m] = mean_d(g_eff · y_norm)` spans the *whole* D axis — every
d_tile program touching the same m_tile would need this same scalar.
Naïvely each program would loop over all D in pass 1 to compute its
own copy: at `D=1536, BLOCK_D=64` that's a 24× redundant computation.

Following Flash Attention's pre-compute-reduction pattern, the
fallback is two kernels:

**Stage 1 — `_fused_add_norm_inner_bwd_kernel`** (1D over M, one
program per `BLOCK_M` rows, processes full D in one tile)
```python
inner[m] = mean_d(g_eff[m, *] * y_norm[m, *])
# writes (M,) fp32 buffer
```

**Stage 2 — `_fused_add_norm_split_bwd_kernel`** (2D over M × D, each
program handles one `(BLOCK_M, BLOCK_D)` output tile)
```python
inner_m = tl.load(inner_ptr + rows)              # single scalar per row, prebuilt
dx[m, d] = rms_inv[m] * (g_eff[m, d] - y_norm[m, d] * inner_m[m]) + d_ext[m, d]
```

Pass 1 in stage 2 collapses from a full-D loop to a single scalar
load per row. Same total HBM traffic as if we'd done it all in one
kernel (we read `summed`/`dy` once in inner, once in stage 2), but
no redundant per-row reduction.

#### Why we don't just "split D" unconditionally

The 2-kernel fallback was the *only* bwd path in an earlier version
of this kernel — the reasoning at the time was "1D over M will spill,
must split D". That reasoning had a bug: it assumed `BLOCK_M` was
fixed at 32. With `BLOCK_M=32` at `D=1536, num_warps=4` (128 threads):

```
   5 tiles × BLOCK_M=32 × D=1536  ≈  245K fp32 / program
   245K / 128 threads             ≈  1900 fp32 regs/thread
```

**1900 vs 255 cap → catastrophic spill.** So yes, BLOCK_M=32 + full D
spills. But the inline path notices that BLOCK_M is *also* a free
parameter — `_pick_fused_add_norm_tile_config(M, BLOCK_D, n_live_tiles=5)` derives
`BLOCK_M ≤ 1638·nw / BLOCK_D` (= 256 reg/thread cap ÷ 5 tiles, then
distributed over nw·32 threads) and rounds to a pow-of-2. At D=1536
that drops BLOCK_M to 4 (regs ≈ 160), at D=4096 to 1 (also ≈ 160).
Comfortable, no spill.

Same key idea as the D-split fallback (cut whichever dimension keeps
the tile in registers), just applied to the M axis where each row is
independent and there's no cross-row reduction to worry about. The
fallback only fires when shrinking BLOCK_M to 1 *still* isn't enough —
i.e. when BLOCK_D alone (= full D) blows the budget.

### 2.3 d_summed_external is fused into the bwd kernel

The op returns two outputs (`y`, `summed`), so backward receives two
gradients (`dy`, `d_summed_external`). The first is the norm's output
gradient; the second is from the caller using `summed` directly
downstream.

The naïve setup would compute `d_summed_from_norm` in the bwd kernel,
then do a Python `d_summed_total = d_summed_from_norm + d_summed_external`.
That's an extra torch elementwise op — its own kernel launch (~10-30 μs)
plus a 4·M·D HBM round-trip (~10-15 μs for d24 shape).

Instead, the bwd kernel takes `d_summed_external` as an extra input
and folds the add into the same tile store:

```python
d_summed_tile = rms_inv * (g_eff - y_norm * inner) + d_ext   # ← `+ d_ext` is the fuse
tl.store(d_summed_ptr + offs, d_summed_tile.to(...), mask=mask)
```

Pure register-level add, no extra HBM traffic, no extra kernel launch.

### 2.4 Sizing — applying Chapter 1's budgets

All tile-sized kernels here (fwd, bwd inline, fallback inner) share
one helper for sizing decisions:

```python
def _pick_fused_add_norm_tile_config(M, BLOCK_D, n_live_tiles) -> TileConfig:
    # Register-budget model (Ampere 255 fp32 reg/thread spill cap):
    #     regs/thread ≈ n_live_tiles × (BLOCK_M × BLOCK_D) / (nw × 32)
    # At the 256-reg target: tile ≤ (8192 / n_live_tiles) × nw
    tile_per_nw = 8192 // n_live_tiles                    # 4096 @ n=2, 1638 @ n=5
    base_nw = 4                                           # initial guess
    BLOCK_M = max(1, min(
        triton.next_power_of_2(max(1, M // 64)),          # M-saturation: grid ≳ 64
        triton.next_power_of_2(max(1, tile_per_nw * base_nw // BLOCK_D)),
    ))
    tile = BLOCK_M * BLOCK_D
    num_warps = max(4, min(16, triton.next_power_of_2(max(1, tile // tile_per_nw))))
    est_regs = n_live_tiles * tile // (num_warps * 32)
    return TileConfig(BLOCK_M, num_warps, est_regs)
```

`n_live_tiles` is the per-kernel knob — peak number of fp32 tiles
alive simultaneously in the hot path:

| Kernel | n_live_tiles | Why |
|---|---|---|
| fwd | 2 | auto-promoted summed-as-fp32 + `y_f32` |
| inner pre-compute (fallback) | 2 | `y_norm` + `g_eff` briefly |
| bwd inline (HAS_NW=True) | 5 | `y_norm`, `g_eff`, `dy_t`, `d_ext`, `d_summed` |
| bwd inline (HAS_NW=False) | 4 | as above but `y_norm`/`g_eff` alias `src`/`dy_t` |

The `TileConfig.fits_reg_budget` property (`est_regs ≤ 256`) is what
`_fused_add_norm_bwd_impl` queries to choose inline vs 2-kernel fallback.

Worked examples at d24 shape (M=2048, D=1536, BLOCK_D=2048):

**fwd (n=2)**: `tile_per_nw = 4096`. BLOCK_M = min(next_pow2(32),
next_pow2(4096·4/2048)) = min(32, 8) = 8. tile = 16K. nw = next_pow2(16K/4096) = 4.
→ `BLOCK_M=8, nw=4`, ~256 reg/thread. Grid = cdiv(2048, 8) = 256
programs.

**bwd inline (n=5)**: `tile_per_nw = 1638`. BLOCK_M = min(32,
next_pow2(1638·4/2048)) = min(32, 4) = 4. tile = 8K. nw = next_pow2(8K/1638) = 8.
→ `BLOCK_M=4, nw=8`, ~160 reg/thread. Grid = cdiv(2048, 4) = 512.

**2-kernel bwd fallback**: uses **fixed config** `BLOCK_M=32, BLOCK_D=64,
num_warps=4` instead of `_pick_fused_add_norm_tile_config`. Reason isn't perf
(autotune's picks were similar); it's that Triton's autotune dispatch
path retains some operations that don't survive CUDA Graph stream
capture. Hard-coding makes the fallback graph-friendly. Only the
inner pre-compute kernel uses `_pick_fused_add_norm_tile_config` (with n_live=2).

### 2.5 Numerical precision

This kernel is **close to but not bit-tight** with `F.rms_norm` on
bf16 inputs. PyTorch's `F.rms_norm` internally promotes bf16 → fp32
**before** squaring; we save one register tile by leaving the square
in bf16 and only promoting the accumulator:

```python
# F.rms_norm  : summed.to(fp32) → (fp32 * fp32)² → fp32 sum
# our fwd     : (bf16 * bf16)² → fp32 sum  (via `tl.sum(..., dtype=fp32)`)
```

The bf16-truncated square plus a long-D accumulation drifts about
~1.5e-1 max forward diff on adversarial seeds (well below the
gradient-noise floor on typical inputs). The `test_triton_norm_mlp.py`
bf16 atol is set to 1.5e-1 to allow this.

| Operation | Where it lives | Why |
|---|---|---|
| HBM load of `x`, `r`, `nw` | bf16 (caller's dtype) | Cheaper bandwidth |
| `summed = x + r`, residual-stream store | bf16 in registers | Caller expects bf16 |
| `summed * summed` square | **bf16 in registers** (lossy) | Saves one fp32 register tile |
| `tl.sum(..., dtype=tl.float32)` | **fp32 accumulator** | Prevents long-D mantissa overflow |
| `summed * rms_inv` / `* nw` | **auto-promoted to fp32** | rms_inv is fp32, Triton lifts on multiply |
| Final `y` store | cast back to bf16 | Caller expects bf16 |

**Net register pressure is roughly the same as a fully-fp32-internal
implementation** (fwd kernel: n_regs≈255 at tile=16384, nw=4) — the
auto-promote in `summed * rms_inv` materializes an fp32 tile anyway;
we save the SSA name of the explicit `summed_f32` intermediate but not
the underlying register. The bf16 path is about HBM dtype compatibility,
not register savings.

> Want strict bit-parity with `F.rms_norm` instead? Add `summed.to(tl.float32)`
> before the square (costing one register tile, matching the fully-fp32
> implementation). The current code chose the precision-vs-register
> tradeoff that's worth more in practice.

### 2.6 Expected savings — HBM and launch ledger

Before measuring, work out what fusion should save on paper. The
short version: **forward saves a real HBM round-trip; backward
primary (inline) path also saves bytes; backward fallback (2-kernel)
saves launches and intermediate buffers but not HBM bytes**.

#### Forward

Naive native (two ops: `summed = x + r`, then `y = F.rms_norm(summed)`):
```
torch.add (x + r → summed):
  read x        M·D
  read r        M·D
  write summed  M·D
F.rms_norm (summed → y):
  read summed   M·D                     ← this is what fusion eliminates
  write y       M·D
─────────────────────────────────────────
Total:          5·M·D
```

Fused (one kernel):
```
read x          M·D
read r          M·D
write summed    M·D    (caller residual stream still needs it)
write y         M·D
─────────────────────────────────────────
Total:          4·M·D
```

Net forward savings:
- **HBM: 1·M·D bytes** — the `summed` value never leaves registers
  between `x + r` and the norm reduction.
- **Kernel launches: 1** (two ops collapse to one Triton kernel).
- **Intermediate buffer: 1** (no separate `summed` allocation for
  the torch.add output).

For d24 (M=2048, D=1536, bf16): 1·M·D = 6.3 MB / 936 GB/s ≈ 6.7 μs
of HBM time, plus ~10-30 μs of avoided launch overhead.

#### Backward

Backward has two kernel paths (see §2.2 for the dispatch). Common
case is the **inline single-pass kernel**; the **2-kernel D-split
fallback** only engages when the inline tile would exceed the 255
reg/thread cap (HAS_NW=True at D > 16K; HAS_NW=False at D > 32K
which essentially never triggers). Ledger them both.

Both compare against PyTorch-optimized native. The math itself needs
two reductions over D (`inner[m] = mean_d(dy · y_norm)` and then
`dx = rms_inv · (dy − y_norm · inner)`), so "1 read" below assumes
the **1-pass shared-memory pattern** PyTorch ships: one CUDA block
per row, load `summed` and `dy` into shared mem once, reduce there,
then revisit shared mem for the per-element `dx`. D=1536 fp32 +
dy = 12 KB / row, easily fits in 3090's 100 KB / SM shared-mem
budget. A literally-naive 2-pass implementation (write `inner` to
HBM, re-load `summed`/`dy` for the second pass) would double the
read bytes — call it out if you ever benchmark against one.

All ledgers below assume `HAS_NW=False` (no learnable affine weight,
which matches nanchat's setup — see `nanchat/gpt.py:9`). HAS_NW=True
adds a `dnw_partials` write (~M·D fp32, larger in inline since
BLOCK_M is smaller there) + a `.sum(dim=0)` reduction (~M·D read,
~D write) on both fused paths, and a separate per-channel dW kernel
(~2·M·D read, ~D write) on native — roughly cancels out in
comparisons, so the relative wins stay the same.

Native (1-pass shared-mem + Python accumulation of external grad):
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
Total:                     6·M·D
```

##### Path A — Inline (primary): bytes + launches + buffer win

The inline kernel fits the full row in one tile (`BLOCK_D = next_pow_of_2(D)`)
and computes `inner` in registers, no precompute kernel, no inner
HBM buffer. `d_summed_external` is folded into the kernel's d_summed
store via a register-level add — same trick as the 2-kernel path:

```
_fused_add_norm_inline_bwd_kernel:
  read summed/y            M·D
  read dy                  M·D
  read d_summed_external   M·D
  write d_summed           M·D
─────────────────────────────────────────
Total:                     4·M·D
```

Net inline savings vs native:
- **HBM: 2·M·D bytes** — no `d_summed_from_norm` intermediate
  flowing through HBM (the kernel computes the norm gradient and
  folds in `d_ext` inside the same row's registers).
- **Kernel launches: 1** (one Triton kernel vs PyTorch's
  `rms_norm.backward` + Python add).
- **Intermediate buffer: 1** (no `d_summed_from_norm` allocation).

For d24 (M=2048, D=1536, bf16): 2·M·D = 12.6 MB / 936 GB/s ≈ 13 μs
of HBM time saved, plus ~10-30 μs of avoided launch overhead. This
matches the symmetry with forward — both fwd and bwd primary paths
get the clean three-way win.

##### Path B — 2-kernel fallback: bytes tie, launches still saved

When inline would spill (HAS_NW=True at D > 16K, see §2.2), the
dispatch falls back to the 2-kernel pair. The split-D structure
**can't** use the 1-pass shared-mem trick — no single program owns
a full row — so we read `summed`/`dy` twice (once in inner, once
in bwd):

```
_fused_add_norm_inner_bwd_kernel:
  read summed/y            M·D
  read dy                  M·D
  write inner_buf          ~0  (M floats)                   ─┐ subtotal 2·M·D
                                                             │
_fused_add_norm_split_bwd_kernel:                                  │
  read summed/y            M·D   ← repeat read (L2 likely hits)
  read dy                  M·D   ← repeat read (L2 likely hits)
  read d_summed_external   M·D
  read inner_buf           ~0
  write d_summed           M·D                              ─┘ subtotal 4·M·D
─────────────────────────────────────────
Total:                     6·M·D    (= native)
```

**Bytes tie native** — fused reads `summed`/`dy` twice (2·M·D each)
where native reads them once, but native's separate `+` kernel adds
back a 3·M·D round-trip on `d_summed_from_norm`. Net: even. Wins
shrink to:
- **1 intermediate buffer**: no `d_summed_from_norm` (same as inline).
- **1 kernel launch**: trade native's Python add for the
  inner-precompute kernel — net launch count ties native, but the
  fused launches are closer together (better L2 reuse on the repeat
  reads in the bwd kernel).

#### Takeaway

| Path | bytes | launches | buffer |
|---|---|---|---|
| Forward fused | ✓ save 1·M·D | ✓ save 1 | ✓ save 1 |
| Backward inline (primary) | ✓ save 2·M·D | ✓ save 1 | ✓ save 1 |
| Backward 2-kernel (fallback) | — tie | ~ tie (trade) | ✓ save 1 |

The lesson: **whenever you can re-use values from a previous step
in registers instead of round-tripping through HBM, that's a real
bandwidth saving**. The inline path does exactly that for both fwd
and bwd. The 2-kernel fallback is forced to re-read `summed`/`dy`
because the per-program tile can't hold a full row — but that
fallback only fires when the alternative (catastrophic register
spill) would be much worse.

### 2.7 Performance reality

Measured at d24 fwd-only (M=2048, D=1536, bf16, HAS_NW=False) on
RTX 3090:

| Mode | fused | native | ratio |
|---|---|---|---|
| Kernel-only timing (direct kernel launch, no autograd) | **~72 μs** | ~88 μs | **fused 1.22× faster** |
| Plain eager `fused_add_norm(...)` call (fwd-only) | ~163 μs | ~88 μs | fused 1.85× *slower* |
| Plain eager `fused_add_norm(...) + backward` | ~1075 μs | ~618 μs | fused 1.74× *slower* |

The eager-mode slowdown is **not** the kernel itself — it's the
`autograd.Function` + Triton dispatch overhead per call (~90 μs of
fixed cost per fwd: tensor allocs, `save_for_backward`, ctx setup,
kernel launch arg packing). This overhead vanishes inside CUDA Graph
capture or a larger `torch.compile`-wrapped pipeline.

Bigger shapes flip the verdict: at `M=2048, D=4096` (also fwd+bwd,
HAS_NW=False) fused wins **1.21×** end-to-end. The crossover is where
the kernel's actual GPU work exceeds the ~90 μs framework overhead.

The fundamental lesson: **for kernels this short, Python framework
overhead can easily exceed the kernel's actual GPU work**. A
single-op `torch.compile(fused_add_norm)` makes things *worse*
because the compile dispatcher adds its own overhead. The fusion
only pays off when:
1. The op is invoked inside a larger model wrapped in
   `torch.compile(model)` so the dispatcher overhead is amortized, OR
2. The whole training step is `torch.cuda.CUDAGraph`-captured so the
   per-call Python work happens only at capture time.

This is the same reason production transformer kernels are
*bigger* (Flash Attention covers the whole `Q@K^T → softmax → @V`
chain, not individual ops): kernel launches and Python dispatchers
are constant per-op overhead, so longer kernels win on amortization
even if their per-element throughput isn't faster than a chain of
small kernels.

That's also why nanchat's production path mostly skips this standalone
op-boundary kernel: `fused_mlp` folds norm directly into its
matmul path, and `norm_qkv_projection` reuses only the RMSNorm
materialization sub-kernel before its Q/K/V projection. The bigger
production kernels are where these patterns pay off.

---

## Chapter 3 — `fused_mlp`: production-level, fwd+bwd all Triton

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
~1300 μs. fp32 IEEE path per-stage shared mem is `(256·32 + 64·32)·4
= 40 KB`, ×2 stages = 80 KB, within the 100 KB SM budget on 3090, so
the same config is safe for parity tests too.

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

**D's config bifurcation + shared-mem budget**: the d24 bf16 sweep
winner is `(BLOCK_K=64, BLOCK_N=64, nw=4, st=2)`. The fp32 IEEE path
(which parity tests take) uses a more conservative staging choice, so
the caller bifurcates on `ieee`:

```python
if ieee:
    BLOCK_K_D, BLOCK_N_D, NW_D, ST_D = 64, 64, 8, 3   # fp32-safe (80 KB / 2 stages)
else:
    BLOCK_K_D, BLOCK_N_D, NW_D, ST_D = 64, 64, 4, 2   # bf16 winner
```

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
   prerequisite (no autotune dispatch), but the wrapper still has
   `if has_nw` / `if ieee` Python branches that need specialization
   before capture. Another +1-2% possible.

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

7. **Compute the shared-mem budget along the dtype path** — a sweep
   winner for the bf16 path can exceed 100 KB on the fp32 IEEE path,
   and Triton **silently miscompiles** (doesn't fail the launch,
   just produces wrong answers). Step D had to bifurcate configs in
   the caller on `ieee`. autotune dispatch can't avoid this trap (it
   runs every candidate and picks the "fastest", including wrong
   ones), so manual sweeps + caller-locked configs are the only safe
   path.

8. **atomic_add is free on small target buffers** — L2-friendly,
   hardware atomic units handle contention. Simpler than
   scratchpad+reduce and not slower.
