cd /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main
export PYTHONPATH="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export TMPDIR="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/tmp"
export WANDB_DIR="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs"
export WANDB_CACHE_DIR="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs/.cache"


# wandb sweep --entity xinjiemao60-westlake-university --project perturbench sweep.yaml
# wandb agent <新的SweepID>
