export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export TMPDIR=/tmp  # 避免 AF_UNIX path too long
HYDRA_FULL_ERROR=1 train trainer.devices=[0] \
 data=mix_pert \
 model=decoder_only \
 data.data_path='./data/norman_hvg_emb.h5ad' \
 logger=wandb \
 trainer.log_every_n_steps=5 \
 trainer.max_epochs=20 \
 trainer.min_epochs=1 \