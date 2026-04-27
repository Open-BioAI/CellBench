# VCBench

VCBench 是一个用于单细胞扰动效应预测与公平评测的统一框架，支持多数据集、多模型训练，以及基于 Hydra + Weights & Biases（W&B）的可复现实验流程。

Paper Demo 网站：<https://maoxinjie.github.io/VCBench-demo/>

---

## 1. 环境安装

在项目根目录执行：

```bash
conda env create -f ./vcbench.yml
conda activate vcbench
pip install -e .
```

配置 W&B：

```bash
wandb login
```

---

## 2. 项目结构（核心）

```text
VCBench/
├── src/perturbench/
│   ├── configs/                 # Hydra 配置
│   │   └── model/               # 各模型配置文件
│   ├── modelcore/
│   │   ├── models/              # 模型实现与注册
│   │   └── train.py             # 训练入口（支持 Hydra + CLI override）
│   └── data/                    # 数据与数据集构建逻辑
├── sweep/                       # W&B Sweep 配置
├── NEW_MODEL_INTEGRATION.md     # 新模型接入详细教程
└── RUN_TUTORIAL.md              # 复现实验运行教程
```

---

## 3. 数据准备与路径配置

### 3.1 下载数据

从 Google Drive 下载并放到项目根目录：

- `unseen_perts`
- `model_related`

下载地址：<https://drive.google.com/drive/folders/1GrPW9x5_npnT7ILwDVsFWvfDIcqaSjdk?usp=sharing>

### 3.2 设置数据文件路径（sweep 配置）

- `./sweep/norman/*.yaml` 中 `parameters.data.data_path` 设置为：

```yaml
./unseen_perts/norman2019_comb_stack.h5ad
```

- `./sweep/replogle/*.yaml` 中 `parameters.data.data_path` 设置为：

```yaml
./unseen_perts/ReplogleWeissman2022_K562_stack_hvg_split.h5ad
```

- `./sweep/sciplex/*.yaml` 中 `parameters.data.data_path` 设置为：

```yaml
./unseen_perts/SrivatsanTrapnell2020_sciplex3_stack_hvg_split.h5ad
```

### 3.3 设置特征映射路径

- Norman / Replogle：`parameters.data.transform.gene_map_path`

```yaml
./model_related/ESM2_pert_features.pt
```

- Sciplex：`parameters.data.transform.drug_map_path`

```yaml
./model_related/SMILES_pert_features.pt
```

### 3.4 GEARS 额外配置

- `./sweep/norman/no-stack/gears.yaml` 中 `parameters.model.data_path`：

```yaml
./gears_norman
```

- `./sweep/replogle/no-stack/gears.yaml` 中 `parameters.model.data_path`：

```yaml
./gears_replogle
```

- `./src/perturbench/configs/model/gears.yaml` 中：

```yaml
gene2go_path: ./model_related/gene2go.pkl
gene_set_path: ./model_related/essential_all_data_pert_genes.pkl
```

---

## 4. 运行实验

### 4.1 使用 train.py（Hydra）

```bash
python src/perturbench/modelcore/train.py model=latent_additive train=true test=true
```

快速检查配置（不训练）：

```bash
python src/perturbench/modelcore/train.py model=latent_additive train=false test=false
```

### 4.2 使用 W&B Sweep

```bash
wandb sweep [yaml_path]
wandb agent [entity/project/sweep_id]
```

可先做单次试跑：

```bash
wandb agent --count 1 [entity/project/sweep_id]
```

---

## 5. 新模型接入（最小可用流程）

目标：让模型可被 `train.py` 和 `wandb sweep` 正常调用，并贯通训练/验证/测试。

### 步骤 1：新增模型代码

在 `src/perturbench/modelcore/models/<your_model>.py` 新建模型，推荐继承 `PerturbationModel` 并实现：

- `forward(...)`
- `training_step(...)`
- `validation_step(...)`
- `predict(...)`

建议复用基类能力：

- `auto_mse(...)`（损失）
- `_get_mask(batch)`（mask 逻辑）

参考实现：

- `src/perturbench/modelcore/models/linear_additive.py`
- `src/perturbench/modelcore/models/latent_additive.py`

### 步骤 2：新增模型配置

在 `src/perturbench/configs/model/<your_model>.yaml` 新建配置，示例：

```yaml
_target_: perturbench.modelcore.models.YourModel

use_cell_emb: false
use_mask: true
lr: 1e-4
wd: 1e-6
lr_scheduler_mode: onecycle
```

注意：

- `_target_` 必须指向正确的 Python 类
- YAML 参数名需与 `__init__` 对齐

### 步骤 3：注册模型

在 `src/perturbench/modelcore/models/__init__.py` 添加：

```python
from .your_model import YourModel
```

### 步骤 4：如需 sweep 注入新参数，更新 train.py 参数映射

当前 `train.py` 使用“argparse + Hydra override”混合机制。若新增例如 `model.foo`：

1. 在 `_build_arg_parser()` 增加：

```text
--model.foo
```

2. 在 `_apply_cli_overrides()` 的 `mapping` 增加：

```text
"model_foo": "model.foo"
```

否则 sweep 参数可能无法生效。

### 步骤 5：最小验证顺序

```bash
# 1) 仅检查配置
python src/perturbench/modelcore/train.py model=<your_model> train=false test=false

# 2) 1 个 epoch
python src/perturbench/modelcore/train.py model=<your_model> trainer.max_epochs=1 train=true test=false

# 3) 训练 + 测试
python src/perturbench/modelcore/train.py model=<your_model> train=true test=true
```

---

## 6. 常见问题排查

- `_target_` 找不到类：检查 YAML 路径、类名、`models/__init__.py` 注册
- 训练时报 batch 字段缺失：对齐现有 batch 协议（如 `pert_cell_counts`、`control_cell_counts`）
- sweep 参数不生效：确认 `train.py` 的 parser 和 mapping 已覆盖新参数
- 协变量维度不匹配：检查 `use_covs` 与 transform 设置是否一致

---

## 7. 推荐实践

- 新模型先做“最小可跑版本”，再逐步叠加复杂模块
- 尽量复用基类提供的 mask、scheduler、logging，降低重复代码
- 每新增一个 sweep 参数，同步更新 `train.py` 参数映射，避免线上空跑

---

## 8. 相关文档

- `NEW_MODEL_INTEGRATION.md`：新模型接入完整说明
- `RUN_TUTORIAL.md`：数据准备与实验运行说明