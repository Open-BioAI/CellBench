#!/usr/bin/env bash
# Script to stop all wandb agents gracefully

PROJECT_DIR="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main"
STOP_FLAG="$PROJECT_DIR/.stop_agents"

echo "Stopping all wandb agents..."

# Create stop flag file
touch "$STOP_FLAG"
echo "✅ Stop flag created: $STOP_FLAG"

# Wait a moment for agents to detect the flag
sleep 2

# Force kill any remaining agents
pkill -f "wandb agent" && echo "✅ Force killed remaining wandb agent processes" || echo "No wandb agent processes found"

# Kill the launch script
pkill -f "run_multi_gpu_agents.sh" && echo "✅ Stopped run_multi_gpu_agents.sh script" || echo "No launch script found"

# Remove stop flag after a moment
sleep 1
rm -f "$STOP_FLAG"

echo "✅ All agents stopped. You can restart with: bash run_multi_gpu_agents.sh"

