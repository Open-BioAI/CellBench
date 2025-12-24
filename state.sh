export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
HYDRA_FULL_ERROR=1 train trainer.devices=[0,1,2,3] \
      trainer.strategy='ddp' \
      data=mix_pert \
      model=state_sm \
      data.data_path='/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess/total/all_cell_line_filterdrug_subsampled.h5ad'