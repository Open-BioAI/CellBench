#!/usr/bin/env bash
set -euo pipefail

# =========================
# W&B auth: 强烈建议从控制面板/Secret 注入，而不是写在脚本里
# export WANDB_API_KEY="YOUR_KEY"
export WANDB_MODE="${WANDB_MODE:-online}"

# =========================
PROJECT_DIR="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

SWEEP_ID="${SWEEP_ID:-xinjiemao60-westlake-university/perturbench/ejfagq7v}"
PYTHON_BIN="${PYTHON_BIN:-/fs-computility-new/upzd_share/maoxinjie/miniconda3/envs/qianhong_env/bin/python}"

# 你要的“提高并行度”开关：每张 GPU 启动几个 agent/训练进程
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"          # 建议先 2，再视显存/吞吐调整
GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})

# 可选：限制 CPU 线程，避免多进程把 CPU 打爆反而拖慢
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# 全局 wandb 目录设置（确保所有 wandb 文件都写到挂载盘，不写 /root/.local/share/wandb）
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR/wandb_logs}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PROJECT_DIR/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PROJECT_DIR/wandb_config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$PROJECT_DIR/wandb_data}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"

echo "=========================================="
echo "Launching W&B sweep agents with higher parallelism"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "SWEEP_ID=$SWEEP_ID"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "GPU_LIST=${GPU_LIST[*]}"
echo "PROCS_PER_GPU=$PROCS_PER_GPU"
echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_CACHE_DIR=$WANDB_CACHE_DIR"
echo "=========================================="

launch_one_agent () {
  local gpu_id="$1"
  local slot_id="$2"
  local job_tag="gpu${gpu_id}_p${slot_id}"

  # 每个进程独立目录，避免锁冲突（覆盖全局设置）
  export WANDB_DIR="$PROJECT_DIR/wandb_logs/${job_tag}"
  export WANDB_CACHE_DIR="$PROJECT_DIR/wandb_cache/${job_tag}"
  export WANDB_CONFIG_DIR="$PROJECT_DIR/wandb_config/${job_tag}"
  export WANDB_DATA_DIR="$PROJECT_DIR/wandb_data/${job_tag}"
  export XDG_CACHE_HOME="$PROJECT_DIR/.cache/${job_tag}"
  export NUMBA_CACHE_DIR="$PROJECT_DIR/numba_cache/${job_tag}"
  export TORCH_EXTENSIONS_DIR="$PROJECT_DIR/torch_extensions/${job_tag}"
  mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" \
           "$XDG_CACHE_HOME" "$NUMBA_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

  # 设置 TMPDIR 到项目目录，避免跨设备移动 checkpoint 时出错
  export TMPDIR="$PROJECT_DIR/.tmp/${job_tag}"
  mkdir -p "$TMPDIR"

  # 关键：将多个进程绑定到同一张 GPU（并行"叠加"）
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  "$PYTHON_BIN" -m wandb agent "$SWEEP_ID" \
    > "$PROJECT_DIR/wandb_agent_${job_tag}.log" 2>&1 &
}

# 主循环：每张 GPU 启动 PROCS_PER_GPU 个进程
for GPU_ID in "${GPU_LIST[@]}"; do
  for SLOT in $(seq 0 $((PROCS_PER_GPU - 1))); do
    echo "Starting agent: GPU=$GPU_ID SLOT=$SLOT ..."
    launch_one_agent "$GPU_ID" "$SLOT"
    sleep 0.2
  done
done

echo "Launched $((${#GPU_LIST[@]} * PROCS_PER_GPU)) agents total."
echo "Logs: $PROJECT_DIR/wandb_agent_gpu*_p*.log"

# 让脚本前台阻塞，避免控制面板认为任务结束
wait
