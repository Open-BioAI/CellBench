export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export WANDB_API_KEY=c24d277403208674c2360ed46c8a8812a74911b6


train trainer.devices=[0] data=mix_pert data.task=unseen_cell data.data_path="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/all_cell_line_filterdrug.h5ad" model=latent_additive logger=wandb logger.wandb.project="perturbench" logger.wandb.name="latent_additive_newloss" trainer.log_every_n_steps=5 trainer.max_epochs=20 trainer.min_epochs=1
