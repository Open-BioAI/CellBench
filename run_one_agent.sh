#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# SWEEP_ID="xinjiemao60-westlake-university/perturbench/9z168o0z"
SWEEP_ID="${SWEEP_ID:?SWEEP_ID env is required}"

PYTHON_BIN="/fs-computility-new/upzd_share/maoxinjie/miniconda3/envs/qianhong_env/bin/python"

# 用平台注入的 role index（如果没有，就用随机数）区分本地缓存目录
JOB_ID="${MLP_ROLE_INDEX:-${RANDOM}}"

export WANDB_DIR="$PROJECT_DIR/wandb_logs/job_${JOB_ID}"
export WANDB_CACHE_DIR="$PROJECT_DIR/wandb_cache/job_${JOB_ID}"
export NUMBA_CACHE_DIR="$PROJECT_DIR/numba_cache/job_${JOB_ID}"
export TORCH_EXTENSIONS_DIR="$PROJECT_DIR/torch_extensions/job_${JOB_ID}"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache/job_${JOB_ID}"

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$NUMBA_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"

# 设置 TMPDIR 到项目目录，避免跨设备移动 checkpoint 时出错
export TMPDIR="$PROJECT_DIR/.tmp/job_${JOB_ID}"
mkdir -p "$TMPDIR"

echo "HOSTNAME=$(hostname)"
echo "JOB_ID=$JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "WANDB_DIR=$WANDB_DIR"
echo "SWEEP_ID=$SWEEP_ID"

# 让一个任务持续吃队列：一次跑一个 run，跑完继续下一个
while true; do
  "$PYTHON_BIN" -m wandb agent "$SWEEP_ID" --count 1 || true
  sleep 2
done
