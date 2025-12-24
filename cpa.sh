export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export WANDB_API_KEY=c24d277403208674c2360ed46c8a8812a74911b6
export WANDB_DIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs
export WANDB_CACHE_DIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs/.cache
export TMPDIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/tmp
# train trainer.devices=[0] data=mix_pert data.task=unseen_cell data.data_path="./data/all_cell_line_filterdrug.h5ad" model=cpa logger=wandb logger.wandb.project="perturbench" logger.wandb.name="cpa_unseen_cell" trainer.log_every_n_steps=1 trainer.max_epochs=20 trainer.min_epochs=1


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
    model=cpa \
    # Removed callbacks=lr_monitor to use default callbacks (includes checkpoint callbacks)
    logger=wandb \
    logger.wandb.project=perturbench \
    logger.wandb.name=cpa_srivatsantrapnell2020_sciplex3 \
    trainer.log_every_n_steps=10 \
    trainer.max_epochs=1 \
    trainer.min_epochs=0 \
    trainer.devices=[0] \
    model.use_mask=true \
    train=true \
    test=true \
    test_ckpt_type=pcc