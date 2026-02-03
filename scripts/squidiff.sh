export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export TMPDIR=/tmp  # 避免 AF_UNIX path too long
HYDRA_FULL_ERROR=1 train trainer.devices=[0] \
  data=mix_pert \
  model=squidiff \
  data.data_path='./data/norman_hvg_emb.h5ad' \
  data.embedding_key='scgpt_embeddings' \
  model.use_cell_emb=False \
  model.use_mask=false \
  model.diffusion_steps=2 \
  model.n_selected_genes=10 \
  trainer.min_epochs=0 \
  trainer.max_epochs=1 \
  logger=wandb 