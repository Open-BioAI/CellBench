export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export TMPDIR=/tmp  # 避免 AF_UNIX path too long
HYDRA_FULL_ERROR=1 train trainer.devices=[0] \
 data=mix_pert \
 model=sclambda \
 data.embedding_key='scgpt_embeddings' \
 model.use_cell_emb=true \
 model.use_mask=false \
 logger=wandb \
 data.data_path='./data/norman_hvg_emb.h5ad'

