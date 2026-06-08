#!/bin/bash
# Launch chat_sft with nanoops swapped into nanchat's F namespace.
#
# What this does: sets NANOOPS=1 (the env var that scripts/chat_sft.py
# reads to call nanoops.integration.maybe_patch_nanchat() at startup), then
# launches SFT from the d24 base checkpoint produced by nanoops/train.sh.
#
# Usage:
#   bash nanoops/sft.sh
#   NANOOPS_FUSED=0 bash nanoops/sft.sh
#   NANOOPS_MODEL_TAG=d20 bash nanoops/sft.sh
#   NANOOPS_MODEL_STEP=5568 bash nanoops/sft.sh
#   NANOOPS_DEVICE_BATCH_SIZE=1 bash nanoops/sft.sh
#   NPROC=1 bash nanoops/sft.sh
#   WANDB_RUN=myrun bash nanoops/sft.sh
#   bash nanoops/sft.sh --chatcore-every=-1  # pass extra args through

set -e
source .venv/bin/activate

export NANOOPS=1
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

# chat_sft needs this small identity dataset in the SFT mixture. Keep this
# local to the SFT launcher so nanoops/train.sh remains base-train only.
IDENTITY_DATA="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ ! -f "$IDENTITY_DATA" ]; then
    mkdir -p "$NANOCHAT_BASE_DIR"
    curl -L -o "$IDENTITY_DATA" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

# Full-fuse ON by default. Set NANOOPS_FUSED=0 (or empty) to use the
# checkpoint-heavy memory-first path.
export NANOOPS_FUSED="${NANOOPS_FUSED-1}"

# Optimizer CPU offload ON by default, same default as nanoops/train.sh.
export NANOOPS_OFFLOAD_OPTIM="${NANOOPS_OFFLOAD_OPTIM:-1}"

if [ -n "$NANOOPS_FUSED" ] && [ "$NANOOPS_FUSED" != "0" ]; then
    export NANOOPS_MLP_CHECKPOINT=
    export NANOOPS_L_ATTN_CHECKPOINT=
    export NANOOPS_FUSED_MLP=1
    export NANOOPS_FUSED_ATTN=1
    DEVICE_BATCH_SIZE="${NANOOPS_DEVICE_BATCH_SIZE:-2}"
else
    export NANOOPS_MLP_CHECKPOINT="${NANOOPS_MLP_CHECKPOINT:-1}"
    export NANOOPS_L_ATTN_CHECKPOINT="${NANOOPS_L_ATTN_CHECKPOINT:-1}"
    export NANOOPS_FUSED_MLP="${NANOOPS_FUSED_MLP:-1}"
    export NANOOPS_FUSED_ATTN="${NANOOPS_FUSED_ATTN-1}"
    DEVICE_BATCH_SIZE="${NANOOPS_DEVICE_BATCH_SIZE:-1}"
fi

NPROC=${NPROC:-2}
WANDB_RUN=${WANDB_RUN:-dummy}
MODEL_TAG=${NANOOPS_MODEL_TAG:-d24}
MODEL_STEP=${NANOOPS_MODEL_STEP:-}

SFT_ARGS=(
    --model-tag="$MODEL_TAG"
    --device-batch-size="$DEVICE_BATCH_SIZE"
    --run="$WANDB_RUN"
)

if [ -n "$MODEL_STEP" ]; then
    SFT_ARGS+=(--model-step="$MODEL_STEP")
fi

echo "nanoops sft fused: ${NANOOPS_FUSED:-0} (NPROC=$NPROC, device_batch_size=$DEVICE_BATCH_SIZE, model_tag=$MODEL_TAG, model_step=${MODEL_STEP:-latest}, run=$WANDB_RUN)"

# NPROC=1: launch via plain python (NOT torchrun), same reason as
# nanoops/train.sh: single-GPU should use MuonAdamW instead of forcing the
# distributed optimizer through torchrun's RANK / WORLD_SIZE env.
if [ "$NPROC" = "1" ]; then
    python -u -m scripts.chat_sft "${SFT_ARGS[@]}" "$@"
else
    torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- \
        "${SFT_ARGS[@]}" "$@"
fi
