#!/usr/bin/env bash
# 这是给 MLP 平台"入口命令"字段使用的单行或多行命令
# 可以直接复制到平台的"入口命令"配置中

cd /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main && \
export PYTHONPATH=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src:$PYTHONPATH && \
export HYDRA_FULL_ERROR=1 && \
export WANDB_DIR=$PWD/wandb_logs/job_${MLP_ROLE_INDEX:-0} && \
export WANDB_CACHE_DIR=$PWD/wandb_cache/job_${MLP_ROLE_INDEX:-0} && \
export NUMBA_CACHE_DIR=$PWD/numba_cache/job_${MLP_ROLE_INDEX:-0} && \
export TORCH_EXTENSIONS_DIR=$PWD/torch_extensions/job_${MLP_ROLE_INDEX:-0} && \
export XDG_CACHE_HOME=$PWD/.cache/job_${MLP_ROLE_INDEX:-0} && \
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$NUMBA_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" && \
echo "HOSTNAME=$(hostname)" && \
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" && \
echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-unset}" && \
nvidia-smi || true && \
bash /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/run_mlp_agent.sh

