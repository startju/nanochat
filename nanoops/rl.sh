#!/bin/bash
# Launch chat_rl with nanoops swapped into nanchat's F namespace.
#
# Usage:
#   bash nanoops/rl.sh
#   NANOOPS_MODEL_STEP=486 bash nanoops/rl.sh
#   NANOOPS_FUSED=0 bash nanoops/rl.sh
#   NANOOPS_DEVICE_BATCH_SIZE=1 bash nanoops/rl.sh
#   NPROC=1 bash nanoops/rl.sh
#   WANDB_RUN=myrun bash nanoops/rl.sh
#   bash nanoops/rl.sh --num-epochs=1  # pass extra args through

set -e
source .venv/bin/activate

export NANOOPS=1
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

# Full-fuse ON by default. Set NANOOPS_FUSED=0 (or empty) to use the
# checkpoint-heavy memory-first path.
export NANOOPS_FUSED="${NANOOPS_FUSED-1}"

# Optimizer CPU offload ON by default, same default as nanoops/train.sh and
# nanoops/sft.sh.
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

RL_ARGS=(
    --model-tag="$MODEL_TAG"
    --device-batch-size="$DEVICE_BATCH_SIZE"
    --run="$WANDB_RUN"
)

if [ -n "$MODEL_STEP" ]; then
    RL_ARGS+=(--model-step="$MODEL_STEP")
fi

echo "nanoops rl fused: ${NANOOPS_FUSED:-0} (NPROC=$NPROC, device_batch_size=$DEVICE_BATCH_SIZE, model_tag=$MODEL_TAG, model_step=${MODEL_STEP:-latest}, run=$WANDB_RUN)"

# NPROC=1: launch via plain python, matching nanoops/train.sh and
# nanoops/sft.sh so single-GPU uses MuonAdamW instead of DistMuonAdamW.
if [ "$NPROC" = "1" ]; then
    python -u -m scripts.chat_rl "${RL_ARGS[@]}" "$@"
else
    torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_rl -- \
        "${RL_ARGS[@]}" "$@"
fi
