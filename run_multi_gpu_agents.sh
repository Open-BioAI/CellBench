#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY="c24d277403208674c2360ed46c8a8812a74911b6"
export WANDB_MODE=online   # 可选，显式指定

PROJECT_DIR="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main"
cd "$PROJECT_DIR"

export PYTHONPATH=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src:$PYTHONPATH
export HYDRA_FULL_ERROR=1

SWEEP_ID="xinjiemao60-westlake-university/perturbench/la51jylw"
PYTHON_BIN="/fs-computility-new/upzd_share/maoxinjie/miniconda3/envs/qianhong_env/bin/python"

# 建议先从 1 个开始；确认没问题再改成 0..7
GPU_LIST=(0 1 2 3)

mkdir -p "$PROJECT_DIR/wandb_logs" "$PROJECT_DIR/wandb_cache"

# 设置 TMPDIR 到项目目录，避免跨设备移动 checkpoint 时出错
export TMPDIR="$PROJECT_DIR/.tmp"
mkdir -p "$TMPDIR"

launch_agent() {
  local gpu_id="$1"
  local agent_log="$PROJECT_DIR/wandb_agent_gpu${gpu_id}.log"
  
  echo "[$(date)] Starting agent on GPU $gpu_id ..." | tee -a "$agent_log"
  
  # 每个 agent 独立目录，避免锁冲突
  export WANDB_DIR="$PROJECT_DIR/wandb_logs/gpu${gpu_id}"
  export WANDB_CACHE_DIR="$PROJECT_DIR/wandb_cache/gpu${gpu_id}"
  mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

  # 使用循环确保 agent 在失败后自动重启（容错机制）
  # 不加 --count 参数：agent 会持续运行，自动从 sweep 队列中获取 run，直到 sweep 完成
  # 如果 agent 因为异常崩溃，循环会自动重启它
  while true; do
    echo "[$(date)] Starting wandb agent on GPU $gpu_id..." | tee -a "$agent_log"
    CUDA_VISIBLE_DEVICES=$gpu_id \
    "$PYTHON_BIN" -m wandb agent "$SWEEP_ID" >> "$agent_log" 2>&1 || {
      exit_code=$?
      echo "[$(date)] Wandb agent exited with code $exit_code on GPU $gpu_id" | tee -a "$agent_log"
      # 如果退出码是 0，说明 sweep 完成了（所有 run 都执行完毕）
      if [ $exit_code -eq 0 ]; then
        echo "[$(date)] Sweep completed. Exiting agent on GPU $gpu_id." | tee -a "$agent_log"
        break
      fi
      # 否则说明 agent 异常退出，等待后自动重启
      echo "[$(date)] Agent crashed. Waiting 5 seconds before restart on GPU $gpu_id..." | tee -a "$agent_log"
      sleep 5
    }
  done
}

for GPU_ID in "${GPU_LIST[@]}"; do
  launch_agent "$GPU_ID" &
done

echo "Launched ${#GPU_LIST[@]} wandb agents in background."
echo "To check status: tail -f $PROJECT_DIR/wandb_agent_gpu*.log"
echo "To stop all agents: pkill -f 'wandb agent'"
