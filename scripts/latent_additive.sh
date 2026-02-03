export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export TMPDIR=/tmp  # 避免 AF_UNIX path too long
HYDRA_FULL_ERROR=1 train trainer.devices=[0] \
 data=mix_pert \
 data.embedding_key='scgpt_embeddings' \
 data.data_path='./data/norman_hvg_emb.h5ad' \
 model=latent_additive \
 model.use_cell_emb=true \
 model.use_mask=false \
 logger=wandb \
 trainer.log_every_n_steps=5 \
 trainer.max_epochs=20 \
 trainer.min_epochs=1 \
