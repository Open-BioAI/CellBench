#!/bin/bash
# Script to update existing Wandb Sweep with new configuration
# Usage: ./update_sweep.sh <sweep_id>

# Set environment variables
export WANDB_API_KEY="c24d277403208674c2360ed46c8a8812a74911b6"

# Change to project directory
cd /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main

# Check if sweep ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <sweep_id>"
    echo ""
    echo "Example:"
    echo "  $0 qwen2832346109-none/perturbench/bc0atqwr"
    exit 1
fi

SWEEP_ID=$1

echo "Updating Wandb Sweep..."
echo "  Sweep ID: $SWEEP_ID"
echo "  Config: sweep.yaml"
echo ""

wandb sweep --update $SWEEP_ID sweep.yaml

echo ""
echo "Sweep updated! Now you can run the agent:"
echo "  wandb agent --entity <entity> $SWEEP_ID"


