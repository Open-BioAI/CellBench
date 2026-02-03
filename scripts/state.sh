export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export TMPDIR=/tmp  # 避免 AF_UNIX path too long
HYDRA_FULL_ERROR=1 train trainer.devices=[0] \
trainer.min_epochs=0 \
trainer.max_epochs=1 \
data=mix_pert \
data.embedding_key='scgpt_embeddings' \
data.train_batch_size=8 \
data.val_batch_size=8 \
data.test_batch_size=8 \
data.sample_mode='set' \
data.cell_set_len=128 \
model=state_sm \
model.use_cell_emb=false \
model.use_mask=false \
logger=wandb \
data.data_path='./data/norman_hvg_emb.h5ad' 