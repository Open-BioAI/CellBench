#!/usr/bin/env bash
set -euo pipefail

SWEEP_ID="${1:-xinjiemao60-westlake-university/perturbench/yxdqwdad}"

CONDA_BASE="/fs-computility-new/upzd_share/maoxinjie/miniconda3"
ENV_NAME="qianhong_env"
PROJECT_DIR="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main"

GPU_LIST=(0 1 2 3)

# ====== 新增：自动登录 wandb ======
# 强烈建议你把 key 存在一个文件里，而不是写死在脚本。
# 例如 ~/.wandb_api_key
WANDB_KEY_FILE="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/.wandb_api_key"

if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY="$(cat $WANDB_KEY_FILE)"
else
    echo "[ERROR] NO WANDB API KEY FOUND. Please create: .wandb_api_key"
    exit 1
fi

export WANDB_AGENT_PROCESS_ON_STDOUT=true
export WANDB_SILENT=false

# ====== 激活环境 ======
source "$CONDA_BASE/bin/activate" "$ENV_NAME"
cd "$PROJECT_DIR"
export TMPDIR=/tmp  # 避免 AF_UNIX path too long

echo "[INFO] Sweep ID: $SWEEP_ID"
echo "[INFO] GPUs: ${GPU_LIST[*]}"
echo "[INFO] Using WANDB key from $WANDB_KEY_FILE"

pids=()

for gpu in "${GPU_LIST[@]}"; do
  log_file="$PROJECT_DIR/wandb_agent_gpu${gpu}.log"
  echo "[INFO] Launching wandb agent on GPU $gpu, log: $log_file"

  CUDA_VISIBLE_DEVICES="$gpu" \
    python -m wandb agent "$SWEEP_ID" \
    > "$log_file" 2>&1 &

  pids+=($!)
done

echo "[INFO] Started ${#GPU_LIST[@]} agents. Waiting for them to finish..."
wait "${pids[@]}"
echo "[INFO] All agents finished."
