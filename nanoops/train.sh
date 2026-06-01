#!/bin/bash
# Launch base_train with nanoops swapped into nanchat's F namespace.
#
# What this does: sets NANOOPS=1 (the env var that scripts/base_train.py
# reads to call nanoops.integration.patch_nanchat() at startup), then
# launches a d24 base-training run sized for 2× RTX 3090.
#
# Usage:
#   bash nanoops/train.sh                       # defaults below
#   NANOOPS_FUSED=1 bash nanoops/train.sh       # full fuse, no activation checkpoints
#   bash nanoops/train.sh --num-iterations=10   # pass extra args through
#   NPROC=4 bash nanoops/train.sh               # override GPU count
#   WANDB_RUN=myrun bash nanoops/train.sh       # enable wandb logging
#
# Compared to runs/speedrun.sh this is just the base_train step in isolation
# (no tokenizer / dataset / SFT / eval) — meant for iterating on nanoops
# itself, not for end-to-end training.

set -e
source .venv/bin/activate

export NANOOPS=1
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
# Note: PYTORCH_ALLOC_CONF=expandable_segments:True is set by
# scripts/base_train.py (line 15) before any CUDA call, so we don't
# re-export it here.

NANOOPS_FUSED=${NANOOPS_FUSED:-}

# Optimizer CPU offload ON by default — moves DistMuonAdamW's per-rank
# Muon + AdamW state (~2.5 GB + ~300 MB on d24 under ZeRO-1) to CPU
# pinned memory; H2D/D2H per optimizer step adds ~0.4% wall time but
# frees the GPU headroom that finally cleared d24+B=1's iter-3 OOM
# cliff (every Python-level checkpoint/LSE trick we tried first failed
# to escape that exact 21.58 GiB allocated + 1.35 GiB fragmentation
# pattern). Opt out with empty value.
export NANOOPS_OFFLOAD_OPTIM="${NANOOPS_OFFLOAD_OPTIM:-1}"

if [ -n "$NANOOPS_FUSED" ]; then
    # Full-fuse d24 path for performance runs: use fused MLP + fused QKV
    # + fused SDPA/output tail, but do not activation-checkpoint MLP or
    # full-attention layers. Fused CE is intentionally not part of this
    # default path: its streaming vocab tiles save logits memory but lose the
    # large lm-head GEMM efficiency and are too slow for the training path.
    #
    # Current d24 + B=1 + 2x RTX 3090 run, after compile/eval warmup:
    #   dt        ~55.8 s/step
    #   tok/sec   ~18,800
    #   MFU       ~63%
    # First step is slower (~125 s) because it includes validation and compile.
    export NANOOPS_MLP_CHECKPOINT=
    export NANOOPS_L_ATTN_CHECKPOINT=
    export NANOOPS_FUSED_MLP=1
    export NANOOPS_FUSED_ATTN=1
    export NANOOPS_FUSED_CROSS_ENTROPY="${NANOOPS_FUSED_CROSS_ENTROPY:-}"
else
    # Checkpoint-heavy d24 path. This keeps the historical defaults:
    # fused kernels are enabled, and activation checkpointing is on to
    # maximize memory headroom on 24 GiB cards.
    export NANOOPS_MLP_CHECKPOINT="${NANOOPS_MLP_CHECKPOINT:-1}"
    export NANOOPS_L_ATTN_CHECKPOINT="${NANOOPS_L_ATTN_CHECKPOINT:-1}"
    export NANOOPS_FUSED_MLP="${NANOOPS_FUSED_MLP:-1}"
    export NANOOPS_FUSED_ATTN="${NANOOPS_FUSED_ATTN-1}"
fi
SAVE_EVERY=${NANOOPS_SAVE_EVERY:-200}

NPROC=${NPROC:-2}
WANDB_RUN=${WANDB_RUN:-dummy}

BASE_TRAIN_ARGS=(
    --depth=24
    --target-param-data-ratio=8
    --device-batch-size=1
    --val-device-batch-size=16
    --save-every="$SAVE_EVERY"
    --save-keep-last=3
    --resume-from-step=-2
    --run="$WANDB_RUN"
)

echo "nanoops fused: ${NANOOPS_FUSED:-0} (NPROC=$NPROC, save_every=$SAVE_EVERY, run=$WANDB_RUN)"

# NPROC=1: launch via plain python (NOT torchrun). torchrun unconditionally
# sets RANK / LOCAL_RANK / WORLD_SIZE in env, which makes nanchat's
# is_ddp_requested() return True and Factory pick DistMuonAdamW. We want
# single-GPU to use the simpler MuonAdamW class — both because it's the
# right tool for the job and because nanoops.cpu_offload's
# patched_step_adamw / patched_step_muon (which handle the GPU-transient
# release dance that single-GPU memory pressure needs) live on MuonAdamW.
# Going via DistMuonAdamW for single-GPU wires through patched_compute_*
# which is missing those single-GPU mitigations.
#
# `python -u` forces unbuffered stdout — torchrun's launcher does this
# automatically, but plain python block-buffers stdout when redirected
# to a file, so step lines / patch list would never appear in the log
# until the buffer fills or the process exits.
if [ "$NPROC" = "1" ]; then
    python -u -m scripts.base_train "${BASE_TRAIN_ARGS[@]}" "$@"
else
    torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
        "${BASE_TRAIN_ARGS[@]}" "$@"
fi
# --depth=24 / device-batch-size=1 on 2× RTX 3090 (24 GiB each):
# d24 auto-widens to D=1536, n_layer=24, ~1.5B params, ~1.67× heavier than
# d20. Even with all three optimizations active (sliding window +
# expandable_segments + MLP_CHECKPOINT) only B=1 fits — B=2 OOMs by ~20 MiB,
# B=4 OOMs by ~1 GiB.
#
# Historical checkpoint-heavy numbers at d24 + B=1:
#   tok/sec  ~15,800
#   MFU       ~53% (B=1 micro-batches don't saturate GEMMs)
#   dt        ~66.5 s/iter (256 grad-accum steps × 270 ms)
#   peak mem  ~22.3 GiB (1.7 GiB headroom)
#   ETA       ~61 h for full 3320-iter run
#
# Drop to --depth=20 --device-batch-size=4 for the throughput sweet spot
# (~30.5k tok/s, ~31h ETA, much better MFU).
