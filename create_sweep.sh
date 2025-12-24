#!/bin/bash
# Script to create Wandb Sweep with correct entity
# Based on launch.json configuration

# Set environment variables
export PYTHONPATH="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c24d277403208674c2360ed46c8a8812a74911b6"
export TMPDIR="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/tmp"
export WANDB_DIR="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs"
export WANDB_CACHE_DIR="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs/.cache"

# Change to project directory
cd /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main

# Entity and project from your configuration
ENTITY="xinjiemao60-westlake-university"
PROJECT="perturbench"
SWEEP_CONFIG="sweep.yaml"

# Create sweep with specified entity
echo "Creating Wandb Sweep..."
echo "  Entity: $ENTITY"
echo "  Project: $PROJECT"
echo "  Config: $SWEEP_CONFIG"
echo ""

wandb sweep --entity $ENTITY --project $PROJECT $SWEEP_CONFIG

echo ""
echo "After creation, you'll get a sweep ID. Then run:"
echo "  wandb agent --entity $ENTITY $ENTITY/$PROJECT/<sweep_id>"

