# nanochat-3090: nanochat on RTX 3090 — a teaching fork

> 中文版：[README_zh.md](README_zh.md)

**Why RTX 3090 (and not a rented H100).** As a beginner you'll spend
the vast majority of your time **debugging and learning**, not actually
training: reading the source, stepping through a backward in a
debugger, swapping one implementation detail for another, comparing
loss curves against the PyTorch reference, re-running a 20-iter probe,
profiling memory. Renting an H100 ($2-4/hr spot) for those phases is
~10-20× more expensive per useful learning hour — you're paying for
flops you mostly aren't using.

The cheapest setups for each learning goal:
  - **Just ops + training internals** → **single RTX 3090**
    (~$0.18/hr spot, or own one outright once → $0/hr ongoing).
    Trains everything in this repo up through d24 with the offload
    stack; no distributed surprises to debug.
  - **Adding NCCL / DDP / collective comm to the learning goals**
    → **dual RTX 3090** (~$0.36/hr spot). The smallest setup where
    real `dist.all_reduce` / `dist.reduce_scatter` actually run
    cross-device (a 1-GPU torchrun box gives you env vars but no
    actual cross-rank network), so you can profile NCCL bottlenecks,
    test ZeRO sharding strategies, and reproduce DDP-specific bugs.

H100s become rational only when wall-time per training run dominates
your debug iteration time — typically once you've built confidence in
the code and just want throughput.

---

This fork extends [karpathy/nanochat](https://github.com/karpathy/nanochat)
with two intertwined goals:

1. **`nanoops/` — fill in the part of nanchat's teaching story that
   stops at PyTorch ops.** nanchat is excellent at showing the whole
   LLM training pipeline (tokenizer → training loop → eval → chat UI)
   end-to-end, but uses PyTorch's built-in ops as black boxes —
   `F.linear`, `F.scaled_dot_product_attention`, `F.cross_entropy`, etc.
   nanoops opens those boxes: every PyTorch op nanchat uses (`Mm`,
   `Linear`, `RMSNorm`, `Softmax`, `CrossEntropy`,
   `ScaledDotProductAttention`, `ApplyRotaryEmb`, sliding-window
   attention, ...) is rewritten as a custom `torch.autograd.Function`
   with explicit forward + backward, an in-place / memory-aware
   implementation, and a math derivation in
   [`nanoops/README.md`](nanoops/README.md)'s appendix
   (also [bilingual 中文版](nanoops/README_zh.md)). Read the source to
   see how `softmax_backward` via `addcmul_` fusion, ctx trade-offs,
   GQA `repeat_interleave + unflatten/sum`, online softmax / chunked
   LSE, and segmented-sum embedding backward actually look in code —
   not just on a whiteboard.

2. **Optimize nanchat training on a 24 GiB GPU along two axes: speed
   AND model size.** nanoops now has two execution layers. The teaching
   layer is the Python `autograd.Function` rewrite above. The training
   layer is a Triton full-fuse path for the hot transformer block pieces:
   `fused_mlp`, `fused_attn_qkv_projection`, and
   `fused_attn_spda_and_output`. The old Triton fused CE tail was removed;
   lm-head / softcap / CE stay on the standard nanoops functional path.
   Together with optimizer-state CPU offload and the expandable-segments
   allocator, the fused path lets consumer 24 GiB GPUs run d24 while also
   improving the compiled training step.

### What this means in practice

**`--depth=24` is nanchat's reference model size — and consumer-grade
GPUs (RTX 3090 / 4090 / etc., 24 GiB class) cannot normally run it.**
With nanchat's stock code, attempting to train d24 on a 24 GiB card
OOMs at every batch size: 1.5 B parameters auto-widened to
`n_embd=1536` × 24 layers + AdamW state + bf16 gradients + the full
`(L, L)` attention probability matrix per sliding layer simply doesn't
fit. The reference hardware is an 8× H100 node — well out of reach for
anyone learning at home or on a small budget.

This fork's memory stack — banded/sliding attention, optional activation
checkpointing, optimizer-state CPU offload, and the
`expandable_segments` allocator — frees enough peak GPU memory and
tames allocator fragmentation enough that d24 actually fits at
`--device-batch-size=1` on a 24 GiB consumer card, whether you have
**one** or **two** of them. The full-fuse Triton path then attacks the
training step itself and defaults to `--device-batch-size=2` on dual
3090s: fused MLP, fused QKV/rotary/QK-norm/VE, and fused
SDPA/output-projection handoff. The 2-GPU run still finishes in about
half the wall time of the 1-GPU run (DDP data-parallel, so it's just
more tokens/sec at the same per-iter peak memory). **The point is to
put nanchat's default training within reach of a beginner's hardware
budget.**

| Config            | nanchat default | nanoops, 1× 24 GiB GPU | nanoops, 2× 24 GiB GPUs |
| ----------------- | --------------- | ---------------------- | ----------------------- |
| `--depth=20`, B=4 | OOM (no FA3)    | (recipe applies)       | **~30.5k tok/s**, ~31 h |
| `--depth=24`, B=1/2 | OOM at all B | **~8k tok/s**, ~200 h  | **~16k tok/s** checkpoint-heavy B=1; **~19.5k tok/s** full-fuse B=2 |

**Concretely:** at typical spot-rental rates of ~$0.18/GPU/hr for an
RTX 3090, a single 3090 costs ~$0.18/hr (~$30/week) and a 2× 3090 rig
~$0.36/hr (~$60/week). A full `--depth=24` pretraining run costs
roughly **$33-35 over ~3.8-4.0 days on dual-GPU** with the current
full-fuse B=2 path. The pure training-step time is ~3.5 days at the
measured ~54.2 s/step; the extra wall time is the default validation,
checkpoint, CORE, and sampling cadence. The single-GPU memory-first
path is still roughly **~$36 over ~8.3 days**. `--depth=20` on dual-GPU
finishes in ~31 h for under **$12**. The same training is otherwise
targeted at 8× H100 nodes; this fork makes it feasible on a desktop
with one or two consumer GPUs.

**Good fit for learners.** Even at the heavier d24 budget there's still
~$25 / ~3 days of GPU time left in a week to break the code on —
read an op in `nanoops/functional.py`, swap an in-place trick out, add
a print to a `.backward()`, kick off a 20-iter run, and watch the loss
curve / MFU drift. The whole stack is small enough to step through in
a debugger, and the bundled tests (`tests/test_nanoops_e2e.py`,
`tests/test_sdpa_parity.py`, ...) cross-check every op against
PyTorch's reference — so you always have ground truth to compare against.

### Current d24 performance (2× RTX 3090)

| Config | dt/step | tok/sec | MFU | Wall time / cost |
| ------ | ------- | ------- | --- | ---------------- |
| Checkpoint-heavy memory-first, B=1 | ~66.5 s | ~15.8k | ~53% | ~61 h counted training ETA; lower memory headroom pressure |
| **Full-fuse, B=2 (default)** | **~54.2 s** | **~19.3k** | **~65%** | **~3.5 days pure step time; ~3.8-4.0 days / $33-35 with default eval+checkpoint cadence** |

The full-fuse row is from the current d24 run after compile/warmup, with
`NANOOPS_FUSED=1`, fused MLP, fused QKV, fused SDPA/output, no activation
checkpointing, optimizer-state CPU offload, and `device-batch-size=2`.
Use `NANOOPS_FUSED=0` when memory headroom matters more than step time.

### Measured speedup journey (d20 base_train on a 3090, dual-GPU numbers)

| Config                                | tok/sec    | MFU       | Peak GPU mem | vs baseline |
| ------------------------------------- | ---------- | --------- | ------------ | ----------- |
| PyTorch SDPA, B=2 (baseline)          | 22,725     | 46.2%     | 16.5 GiB     | —           |
| nanoops Lookup default, B=2           | 28,800     | 58.5%     | 19.7 GiB     | +27%        |
| + SlidingWindowSDPA, B=2              | 30,594     | 62.2%     | 17.6 GiB     | +35%        |
| + B=4 + expandable_segments           | 32,678     | 66.4%     | 22.7 GiB     | +44%        |
| **+ MLP_CHECKPOINT (current default)**| **30,500** | **62.0%** | **19.0 GiB** | **+34%, B=4 holds w/ headroom for d24** |

Loss curves match across all rows to within bf16 rounding noise.
Full A/B autopsy lives in the
[`SlidingWindowSDPA` docstring](nanoops/functional.py).

### What `NANOOPS_FUSED=1` covers

The default fused path is not just a faster `F.linear` swap. It replaces the
hot transformer-block chains with compile-visible Triton custom ops:

- `fused_mlp`: pre-RMSNorm, `c_fc`, ReLU², `c_proj`, residual add, and the
  matching backward path.
- `fused_attn_qkv_projection`: residual/x0 mixing, RMSNorm, separate Q/K/V
  projections, rotary, Q/K RMSNorm+scale, optional value embedding gate/lookup,
  and the matching backward path.
- `fused_attn_spda_and_output`: GQA/sliding-window SDPA plus the attention
  output projection/residual tail, including the backward delta/output-proj
  work that used to be separate.
- `fused_add_norm`: standalone add+RMSNorm Triton op, also reused by the block
  fused kernels.

The lm-head / softcap / cross-entropy tail intentionally stays on the standard
nanoops functional path; the experimental Triton CE tail was removed because it
saved memory but lost too much throughput versus the native GEMM-heavy path.

### Try it

```bash
# Drop-in replacement for speedrun.sh's base_train step. Defaults to
# --depth=24. The default path is full-fuse with device-batch-size=2:
# fused MLP + fused QKV + fused SDPA/output, no activation checkpoints,
# optimizer state offloaded to CPU, and expandable_segments set before CUDA init.
bash nanoops/train.sh                       # full-fuse, uses both visible GPUs
NPROC=1 bash nanoops/train.sh               # single GPU — same recipe still fits
NANOOPS_FUSED=0 bash nanoops/train.sh       # memory-first checkpoint-heavy path

# Or override defaults — e.g. fastest throughput on dual 3090:
bash nanoops/train.sh --depth=20 --device-batch-size=4

# Effective defaults / env vars:
#   NANOOPS=1                                       activates the integration
#   PYTORCH_ALLOC_CONF=expandable_segments:True     recovers fragmentation
#   NANOOPS_OFFLOAD_OPTIM=1                         Muon+AdamW state on CPU;
#                                                    needed to fit d24+B=1
#   NANOOPS_FUSED_MLP=1                             fused Triton MLP block
#   NANOOPS_FUSED_ATTN=1                            fused QKV + SDPA/output
#   NANOOPS_DEVICE_BATCH_SIZE=2                     full-fuse default micro-batch
#   NANOOPS_SAVE_EVERY=200                          checkpoint cadence
#
# Mode knobs:
#   NANOOPS_FUSED=1                                  default: full-fuse speed path
#   NANOOPS_FUSED=0                                  memory-first checkpoint path
#   NANOOPS_MLP_CHECKPOINT=1                         memory-first MLP ckpt
#   NANOOPS_L_ATTN_CHECKPOINT=1                      memory-first L-attn ckpt
#   NANOOPS_LOOKUP_SORTED=1                          segmented embedding bwd
```

See [`nanoops/README.md`](nanoops/README.md) for the op-by-op TODO list +
math derivations, and [`nanoops/integration.py`](nanoops/integration.py)
for how the swap-in monkey-patches plug into nanchat without touching
the upstream model code.

The rest of this README is the unchanged upstream nanchat documentation.

---

# nanochat (upstream)

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat is the simplest experimental harness for training LLMs. It is designed to run on a single GPU node, the code is minimal/hackable, and it covers all major LLM stages including tokenization, pretraining, finetuning, evaluation, inference, and a chat UI. For example, you can train your own GPT-2 capability LLM (which cost ~$43,000 to train in 2019) for only $48 (~2 hours of 8XH100 GPU node) and then talk to it in a familiar ChatGPT-like web UI. On a spot instance, the total cost can be closer to ~$15. More generally, nanochat is configured out of the box to train an entire miniseries of compute-optimal models by setting one single complexity dial: `--depth`, the number of layers in the GPT transformer model (GPT-2 capability happens to be approximately depth 26). All other hyperparameters (the width of the transformer, number of heads, learning rate adjustments, training horizons, weight decays, ...) are calculated automatically in an optimal way.

For questions about the repo, I recommend either using [DeepWiki](https://deepwiki.com/karpathy/nanochat) from Devin/Cognition to ask questions about the repo, or use the [Discussions tab](https://github.com/karpathy/nanochat/discussions), or come by the [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) channel on Discord.

## Time-to-GPT-2 Leaderboard

Presently, the main focus of development is on tuning the pretraining stage, which takes the most amount of compute. Inspired by the modded-nanogpt repo and to incentivise progress and community collaboration, nanochat maintains a leaderboard for a "GPT-2 speedrun", which is the wall-clock time required to train a nanochat model to GPT-2 grade capability, as measured by the DCLM CORE score. The [runs/speedrun.sh](runs/speedrun.sh) script always reflects the reference way to train GPT-2 grade model and talk to it. The current leaderboard looks as follows:

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | change dataset to NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

The primary metric we care about is "time to GPT-2" - the wall clock time needed to outperform the GPT-2 (1.6B) CORE metric on an 8XH100 GPU node. The GPT-2 CORE score is 0.256525. In 2019, the training of GPT-2 cost approximately $43,000 so it is incredible that due to many advances over 7 years across the stack, we can now do so much faster and for well below $100 (e.g. at the current ~$3/GPU/hr, an 8XH100 node is ~$24/hr, so 2 hours is ~$48).

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for more docs on how to interpret and contribute to the leaderboard.

## Getting started

### Setup

nanochat uses [uv](https://docs.astral.sh/uv/) for dependency management. To install:

```bash
uv sync --extra gpu    # Use for CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) Use for CPU-only / MPS
source .venv/bin/activate
```

For development (adds pytest, matplotlib, ipykernel, transformers, etc.):

```bash
uv sync --extra gpu --group dev
```

### Reproduce and talk to GPT-2

The most fun you can have is to train your own GPT-2 and talk to it. The entire pipeline to do so is contained in the single file [runs/speedrun.sh](runs/speedrun.sh), which is designed to be run on an 8XH100 GPU node. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash runs/speedrun.sh
```

You may wish to do so in a screen session as this will take ~3 hours to run. Once it's done, you can talk to it via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

A few more notes:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device-batch-size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't personally exercised all of these code paths so there might be sharp edges.

## Research

If you are a researcher and wish to help improve nanochat, two scripts of interest are [runs/scaling_laws.sh](runs/scaling_laws.sh) and [runs/miniseries.sh](runs/miniseries.sh). See [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) for related documentation. For quick experimentation (~5 min pretraining runs) my favorite scale is to train a 12-layer model (GPT-1 sized), e.g. like this:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

This uses wandb (run name "d12"), only runs the CORE metric on last step, and it doesn't sample and save intermediate checkpoints. I like to change something in the code, re-run a d12 (or a d16 etc) and see if it helped, in an iteration loop. To see if a run helps, I like to monitor the wandb plots for:

1. `val_bpb` (validation loss in vocab-size-invariant units of bits per byte) as a function of `step`, `total_training_time` and `total_training_flops`.
2. `core_metric` (the DCLM CORE score)
3. VRAM utilization, `train/mfu` (Model FLOPS utilization), `train/tok_per_sec` (training throughput)

See an example [here](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044).

The important thing to note is that nanochat is written and configured around one single dial of complexity - the depth of the transformer. This single integer automatically determines all other hyperparameters (the width of the transformer, number of heads, learning rate adjustments, training horizons, weight decays, ...) so that the trained model comes out compute optimal. The idea is that the user doesn't have to think about or set any of this, they are simply asking for a smaller or bigger model using `--depth`, and everything "just works". By sweeping out the depth, you achieve the nanochat miniseries of compute optimal models at various sizes. GPT-2 capability model (which is of most interest at the moment) happens to be somewhere around d24-d26 range with the current code. But any candidate changes to the repo have to be principled enough that they work for all settings of depth.

## Running on CPU / MPS

The script [runs/runcpu.sh](runs/runcpu.sh) shows a very simple example of running on CPU or Apple Silicon. It dramatically shrinks the LLM that is being trained to make things fit into a reasonable time interval of a few ten minutes of training. You will not get strong results in this way.

## Precision / dtype

nanochat does not use `torch.amp.autocast`. Instead, precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/common.py`). By default this is auto-detected based on your hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | No reduced-precision tensor cores |

You can override the default with the `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # force bf16
```

How it works: model weights are stored in fp32 (for optimizer precision), but our custom `Linear` layer casts them to `COMPUTE_DTYPE` during the forward pass. Embeddings are stored directly in `COMPUTE_DTYPE` to save memory. This gives us the same mixed-precision benefit as autocast but with full explicit control over what runs in which precision.

Note: `float16` training automatically enables a `GradScaler` in `base_train.py` to prevent gradient underflow. SFT supports this too but RL currently does not. Inference in fp16 works fine everywhere.

## Guides

I've published a number of guides that might contain helpful information, most recent to least recent:

- [Feb 1 2026: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) documents the first nanochat miniseries of models.
- To add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).
- To customize your nanochat, see [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139) in Discussions, which describes how you can tune your nanochat's personality through synthetic data generation and mixing that data into the SFT stage.
- [Oct 13 2025: original nanochat post](https://github.com/karpathy/nanochat/discussions/1) introducing nanochat, though now it contains some deprecated information and the model is a lot older (with worse results) than current master.

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # Pretraining data shard generation
├── nanochat
│   ├── __init__.py                 # empty
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── optim.py                    # AdamW + Muon optimizer, 1GPU and distributed
│   ├── report.py                   # Utilities for writing the nanochat Report
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for nanochat frontend
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries training script
│   ├── runcpu.sh                   # Small example of how to run on CPU/MPS
│   ├── scaling_laws.sh             # Scaling laws experiments
│   └── speedrun.sh                 # Train the ~$100 nanochat d20
├── scripts
│   ├── base_eval.py                # Base model: CORE score, bits per byte, samples
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model: talk to over CLI
│   ├── chat_eval.py                # Chat model: eval tasks
│   ├── chat_rl.py                  # Chat model: reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model: talk to over WebUI
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_engine.py
└── uv.lock
```

## Contributing

The goal of nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there are no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a ChatGPT model you can talk to. Currently, the most interesting part personally is speeding up the latency to GPT-2 (i.e. getting a CORE score above 0.256525). Currently this takes ~3 hours, but by improving the pretraining stage we can improve this further.

Current AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
