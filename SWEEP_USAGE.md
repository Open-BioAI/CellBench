# Wandb Sweep 使用说明

## 当前已创建的 Sweep

根据之前的输出，你已经创建了以下 sweep：
- **Sweep ID**: `qwen2832346109-none/perturbench/bc0atqwr`
- **Entity**: `qwen2832346109-none` (这是创建时使用的默认 entity)
- **Project**: `perturbench`

## 使用已创建的 Sweep

### 方法 1: 使用脚本（推荐）

```bash
cd /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main

# 使用完整的 sweep ID
./run_sweep_agent.sh qwen2832346109-none/perturbench/bc0atqwr

# 或者只使用 sweep ID（脚本会自动使用创建时的 entity）
./run_sweep_agent.sh bc0atqwr
```

### 方法 2: 手动运行

```bash
cd /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main

# 设置环境变量
export WANDB_API_KEY="c24d277403208674c2360ed46c8a8812a74911b6"
export PYTHONPATH="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src:${PYTHONPATH}"

# 运行 agent（使用创建时的 entity）
wandb agent --entity qwen2832346109-none qwen2832346109-none/perturbench/bc0atqwr
```

## 创建新的 Sweep（使用正确的 Entity）

如果你想使用 `xinjiemao60-westlake-university` 作为 entity，需要重新创建 sweep：

```bash
cd /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main

# 使用脚本创建
./create_sweep.sh

# 或手动创建
wandb sweep --entity xinjiemao60-westlake-university --project perturbench sweep.yaml
```

## 注意事项

1. **Entity 不能更改**: Sweep 一旦创建，entity 就固定了，无法更改
2. **运行 Agent 时**: 必须使用创建 sweep 时使用的 entity
3. **多个 Agent**: 可以在不同终端或 GPU 上同时运行多个 agent 来加速搜索

## 查看 Sweep 状态

在 Wandb 网页界面查看：
- 旧 entity: https://wandb.ai/qwen2832346109-none/perturbench/sweeps/bc0atqwr
- 新 entity (如果重新创建): https://wandb.ai/xinjiemao60-westlake-university/perturbench/sweeps/xxx


