export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export WANDB_API_KEY=c24d277403208674c2360ed46c8a8812a74911b6

train \
trainer.devices=[0] \
data=mix_pert \
logger=wandb \
model=prnet \
data.data_path='/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess/total/all_cell_line_filterdrug_subsampled.h5ad' \
trainer.min_epochs=50 \
trainer.max_epochs=500 \
logger.wandb.project="perturbench" \
logger.wandb.name="prnet" \
trainer.log_every_n_steps=5
# callbacks.early_stopping.patience=5