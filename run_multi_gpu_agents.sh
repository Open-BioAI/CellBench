#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main"
cd "$PROJECT_DIR"

export PYTHONPATH=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src:$PYTHONPATH
export HYDRA_FULL_ERROR=1

SWEEP_ID="xinjiemao60-westlake-university/perturbench/o8c3jzqe"
PYTHON_BIN="/fs-computility-new/upzd_share/maoxinjie/miniconda3/envs/qianhong_env/bin/python"

# 建议先从 1 个开始；确认没问题再改成 0..7
GPU_LIST=(0 1 2 3 4 5 6 7)

mkdir -p "$PROJECT_DIR/wandb_logs" "$PROJECT_DIR/wandb_cache"

for GPU_ID in "${GPU_LIST[@]}"; do
  echo "Starting agent on GPU $GPU_ID ..."

  # 每个 agent 独立目录，避免锁冲突
  export WANDB_DIR="$PROJECT_DIR/wandb_logs/gpu${GPU_ID}"
  export WANDB_CACHE_DIR="$PROJECT_DIR/wandb_cache/gpu${GPU_ID}"
  mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

  CUDA_VISIBLE_DEVICES=$GPU_ID \
  "$PYTHON_BIN" -m wandb agent "$SWEEP_ID" \
    > "$PROJECT_DIR/wandb_agent_gpu${GPU_ID}.log" 2>&1 &
done

echo "Launched ${#GPU_LIST[@]} wandb agents."
