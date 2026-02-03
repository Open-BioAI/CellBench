#!/bin/bash
# 快速运行所有模型的脚本
# 使用方法: bash run_all_models.sh <model_name> [gpu_id]
# 例如: bash run_all_models.sh latent_additive 0

# 设置环境变量
export PYTHONPATH=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src:${PYTHONPATH}
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=c24d277403208674c2360ed46c8a8812a74911b6
export WANDB_DIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs
export WANDB_CACHE_DIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs/.cache
export TMPDIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/tmp

# 设置工作目录
cd /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main

# Python可执行文件路径
PYTHON=/fs-computility-new/upzd_share/maoxinjie/miniconda3/envs/qianhong_env/bin/python
SCRIPT=src/perturbench/modelcore/train.py

# 默认GPU ID
GPU_ID=${2:-0}
DATA_PATH="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/all_cell_line_filterdrug.h5ad"

# 根据模型名称运行对应的命令
MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "使用方法: bash run_all_models.sh <model_name> [gpu_id]"
    echo "可用模型: latent_additive, biolord, cpa, gears, genepert, prnet, sams_vae, decoder_only, linear_additive, sclambda, squidiff, state"
    exit 1
fi

case $MODEL_NAME in
    latent_additive)
        echo "运行 Latent Additive 模型..."
        $PYTHON $SCRIPT \
            trainer.devices=[$GPU_ID] \
            data=mix_pert \
            data.task=unseen_cell \
            data.data_path="$DATA_PATH" \
            model=latent_additive \
            logger=wandb \
            logger.wandb.project="perturbench" \
            logger.wandb.name="latent_additive" \
            trainer.log_every_n_steps=5 \
            trainer.max_epochs=20 \
            trainer.min_epochs=1
        ;;
    
    biolord)
        echo "运行 Biolord 模型..."
        $PYTHON $SCRIPT \
            data=mix_pert \
            data.task=srivatsantrapnell2020_sciplex3 \
            data.data_path="$DATA_PATH" \
            model=biolord \
            logger=wandb \
            logger.wandb.project=perturbench \
            logger.wandb.name=biolord_srivatsantrapnell2020_sciplex3 \
            trainer.log_every_n_steps=10 \
            trainer.max_epochs=2 \
            trainer.min_epochs=0 \
            trainer.devices=[$GPU_ID] \
            model.use_mask=true \
            train=true \
            test=true \
            test_ckpt_type=pcc \
            data.transform.use_covs=true
        ;;
    
    cpa)
        echo "运行 CPA 模型..."
        $PYTHON $SCRIPT \
            data=mix_pert \
            data.task=srivatsantrapnell2020_sciplex3 \
            data.data_path="$DATA_PATH" \
            model=cpa \
            logger=wandb \
            logger.wandb.project=perturbench \
            logger.wandb.name=cpa_srivatsantrapnell2020_sciplex3 \
            trainer.log_every_n_steps=10 \
            trainer.max_epochs=1 \
            trainer.min_epochs=0 \
            trainer.devices=[$GPU_ID] \
            model.use_mask=true \
            train=true \
            test=true \
            test_ckpt_type=pcc
        ;;
    
    gears)
        echo "运行 GEARS 模型..."
        $PYTHON $SCRIPT \
            data=gears \
            model=gears \
            trainer=gears \
            trainer.devices=[$GPU_ID]
        ;;
    
    genepert)
        echo "运行 GenePert 模型..."
        $PYTHON $SCRIPT \
            trainer.devices=[$GPU_ID] \
            data=mix_pert \
            model=genepert \
            data.data_path="$DATA_PATH"
        ;;
    
    prnet)
        echo "运行 PRNet 模型..."
        $PYTHON $SCRIPT \
            trainer.devices=[$GPU_ID] \
            data=mix_pert \
            logger=wandb \
            model=prnet \
            data.data_path="$DATA_PATH" \
            trainer.min_epochs=50 \
            trainer.max_epochs=500 \
            logger.wandb.project="perturbench" \
            logger.wandb.name="prnet" \
            trainer.log_every_n_steps=5
        ;;
    
    sams_vae)
        echo "运行 SAMS VAE 模型..."
        $PYTHON $SCRIPT \
            trainer.devices=[$GPU_ID] \
            data=mix_pert \
            model=sams_vae \
            data.data_path="$DATA_PATH"
        ;;
    
    decoder_only)
        echo "运行 Decoder Only 模型..."
        $PYTHON $SCRIPT \
            trainer.devices=[$GPU_ID] \
            data=mix_pert \
            model=decoder_only \
            data.data_path="$DATA_PATH"
        ;;
    
    linear_additive)
        echo "运行 Linear Additive 模型..."
        $PYTHON $SCRIPT \
            trainer.devices=[$GPU_ID] \
            data=mix_pert \
            model=linear_additive \
            data.data_path="$DATA_PATH"
        ;;
    
    sclambda)
        echo "运行 scLAMBDA 模型..."
        $PYTHON $SCRIPT \
            trainer.devices=[$GPU_ID] \
            data=mix_pert \
            model=sclambda \
            data.data_path="$DATA_PATH"
        ;;
    
    squidiff)
        echo "运行 Squidiff 模型..."
        $PYTHON $SCRIPT \
            trainer.devices=[$GPU_ID] \
            data=mix_pert \
            model=squidiff \
            data.data_path="$DATA_PATH"
        ;;
    
    state)
        echo "运行 State 模型..."
        $PYTHON $SCRIPT \
            trainer.devices=[$GPU_ID] \
            trainer.strategy='ddp' \
            data=mix_pert \
            model=state_sm \
            data.data_path="$DATA_PATH"
        ;;
    
    *)
        echo "未知模型: $MODEL_NAME"
        echo "可用模型: latent_additive, biolord, cpa, gears, genepert, prnet, sams_vae, decoder_only, linear_additive, sclambda, squidiff, state"
        exit 1
        ;;
esac
