export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export TMPDIR=/tmp  # 避免 AF_UNIX path too long
HYDRA_FULL_ERROR=1 train trainer.devices=[0] \
trainer.min_epochs=0 \
trainer.max_epochs=1 \
data=mix_pert \
data.embedding_key=null \
data.cov_keys=[split_category] \
data.result_avg_keys=[split_category] \
data.train_batch_size=300 \
data.sample_mode='cell' \
data.transform.gene_map_path='./ESM2_pert_features.pt' \
model=cpa \
model.use_cell_emb=false \
model.use_mask=false \
logger=wandb \
data.data_path='./tasks/unseen_perts/norman2019_comb.h5ad' 