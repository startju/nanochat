#!/bin/bash
# Serve the latest RL checkpoint with the nanochat web UI.
#
# Usage:
#   bash nanoops/web.sh
#   NANOOPS_MODEL_TAG=d20 bash nanoops/web.sh
#   NANOOPS_MODEL_STEP=466 bash nanoops/web.sh
#   NANOOPS_NUM_GPUS=2 bash nanoops/web.sh
#   NANOOPS_PORT=8001 bash nanoops/web.sh
#   bash nanoops/web.sh --temperature=0.7  # pass extra args through

set -e
source .venv/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

MODEL_TAG=${NANOOPS_MODEL_TAG:-d24}
MODEL_STEP=${NANOOPS_MODEL_STEP:-}
NUM_GPUS=${NANOOPS_NUM_GPUS:-1}
HOST=${NANOOPS_HOST:-0.0.0.0}
PORT=${NANOOPS_PORT:-8000}
TEMPERATURE=${NANOOPS_TEMPERATURE:-0.8}
TOP_K=${NANOOPS_TOP_K:-50}
MAX_TOKENS=${NANOOPS_MAX_TOKENS:-512}

WEB_ARGS=(
    --source=rl
    --model-tag="$MODEL_TAG"
    --num-gpus="$NUM_GPUS"
    --host="$HOST"
    --port="$PORT"
    --temperature="$TEMPERATURE"
    --top-k="$TOP_K"
    --max-tokens="$MAX_TOKENS"
)

if [ -n "$MODEL_STEP" ]; then
    WEB_ARGS+=(--step="$MODEL_STEP")
fi

echo "nanoops web rl (num_gpus=$NUM_GPUS, model_tag=$MODEL_TAG, step=${MODEL_STEP:-latest}, host=$HOST, port=$PORT)"
python -u -m scripts.chat_web "${WEB_ARGS[@]}" "$@"
