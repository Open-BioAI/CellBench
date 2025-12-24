# export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
# HYDRA_FULL_ERROR=1 train trainer.devices=[0] data=mix_pert model=biolord data.data_path="/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess/total/all_cell_line_filterdrug_subsampled.h5ad"#!/bin/bash

# Set environment variables
export PYTHONPATH=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src:${PYTHONPATH}
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=c24d277403208674c2360ed46c8a8812a74911b6
export WANDB_DIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs
export WANDB_CACHE_DIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs/.cache
export TMPDIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/tmp

# Set working directory
cd /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main

# Python executable
PYTHON=/fs-computility-new/upzd_share/maoxinjie/miniconda3/envs/qianhong_env/bin/python

# Training script
SCRIPT=src/perturbench/modelcore/train.py

# Run training with Hydra arguments
$PYTHON $SCRIPT \
    data=mix_pert \
    data.task=srivatsantrapnell2020_sciplex3 \
    data.data_path=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/all_cell_line_filterdrug.h5ad \
    model=biolord \
    # Removed callbacks=lr_monitor to use default callbacks (includes checkpoint callbacks)
    logger=wandb \
    logger.wandb.project=perturbench \
    logger.wandb.name=biolord_srivatsantrapnell2020_sciplex3_updatePCC_useCovs \
    trainer.log_every_n_steps=10 \
    trainer.max_epochs=2 \
    trainer.min_epochs=0 \
    trainer.devices=[1] \
    model.use_mask=true \
    train=true \
    test=true \
    test_ckpt_type=pcc \
    data.transform.use_covs=true
