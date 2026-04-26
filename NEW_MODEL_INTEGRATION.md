# 新模型集成教程（PerturBench）

本文面向当前仓库，给出一套可直接落地的新模型集成流程。你可以把它当成 checklist：按顺序做完，模型通常就能被 `train.py` 和 `wandb sweep` 正常调用。

---

## 1. 集成目标

完成后应满足：

- 能通过 Hydra 选择模型并启动训练：
  - `python src/perturbench/modelcore/train.py model=<your_model> ...`
- 能在 sweep 中注入超参数并执行：
  - `wandb sweep <yaml>`
  - `wandb agent <entity/project/sweep_id>`
- 训练/验证/测试评估链路不报接口错误。

---

## 2. 需要改哪些位置

最少涉及 3 类文件：

1. **模型代码文件**  
   `src/perturbench/modelcore/models/<your_model>.py`

2. **模型配置文件**  
   `src/perturbench/configs/model/<your_model>.yaml`

3. **模型注册入口**  
   `src/perturbench/modelcore/models/__init__.py`

> 如果要跑 `wandb sweep`，还要检查 `src/perturbench/modelcore/train.py` 的 CLI 参数解析白名单（见第 6 节）。

---

## 3. 模型代码规范（建议模板）

仓库中大多数模型继承 `PerturbationModel`，并沿用统一训练接口。推荐结构：

1. 继承：
   - `class YourModel(PerturbationModel):`
2. `__init__`：
   - 接收 `datamodule`、`lr`、`wd`、`use_mask`、scheduler 参数等
   - 调用 `super(..., datamodule=datamodule, ...)`
3. 实现核心方法：
   - `forward(...)`
   - `training_step(...)`
   - `validation_step(...)`
   - `predict(...)`
4. loss：
   - 优先复用基类的 `auto_mse(...)`
   - mask 逻辑优先复用 `_get_mask(batch)`

建议优先参考：

- `src/perturbench/modelcore/models/linear_additive.py`
- `src/perturbench/modelcore/models/latent_additive.py`

---

## 4. 配置文件写法（Hydra）

在 `src/perturbench/configs/model/` 下新建 `<your_model>.yaml`，核心是 `_target_`：

```yaml
_target_: perturbench.modelcore.models.YourModel

use_cell_emb: false
use_mask: true

lr: 1e-4
wd: 1e-6

lr_scheduler_mode: onecycle
```

关键点：

- `_target_` 必须能定位到你的 Python 类。
- 参数名必须和模型 `__init__` 对齐。
- 与数据侧联动参数（如 `use_covs`）建议保持与现有模型一致语义。

---

## 5. 模型注册

在 `src/perturbench/modelcore/models/__init__.py` 增加导入：

```python
from .your_model import YourModel
```

不注册通常会导致 `_target_` 可读性和统一入口变差，也容易在后续维护中遗漏。

---

## 6. Sweep 参数注入注意事项（重要）

当前仓库的 `train.py` 使用了“argparse + Hydra override”混合模式。

这意味着：

- sweep 传入的参数如果不在 `train.py` 的 parser 或映射中，可能无法按预期落到最终配置。

如果你给模型新增 sweep 超参（例如 `model.foo`），建议在：

- `_build_arg_parser()` 增加 `parser.add_argument("--model.foo", ...)`
- `_apply_cli_overrides()` 的 `mapping` 增加：
  - `"model_foo": "model.foo"`

这样最稳，尤其在你们当前 `wandb` 通过 `${args}` 注入参数时。

---

## 7. 最小验证流程

按下面顺序检查，定位问题最快：

1. **仅构建配置（不训练）**
   - `python src/perturbench/modelcore/train.py model=<your_model> train=false test=false`
2. **1 epoch 训练**
   - `python src/perturbench/modelcore/train.py model=<your_model> trainer.max_epochs=1 train=true test=false`
3. **训练+测试**
   - `python src/perturbench/modelcore/train.py model=<your_model> train=true test=true`
4. **sweep 单次 trial**
   - `wandb sweep <your_sweep.yaml>`
   - `wandb agent --count 1 <entity/project/sweep_id>`

---

## 8. 以 `latent_additive` 集成为例

这里用仓库现有的 `latent_additive` 演示“一个模型是如何被接入完整链路”的。

### 8.1 代码入口

- 模型实现：`src/perturbench/modelcore/models/latent_additive.py`
- 模型配置：`src/perturbench/configs/model/latent_additive.yaml`
- 注册入口：`src/perturbench/modelcore/models/__init__.py`

### 8.2 配置如何指向代码

`latent_additive.yaml` 通过：

```yaml
_target_: perturbench.modelcore.models.LatentAdditive
```

把 Hydra 选择器 `model=latent_additive` 绑定到 Python 类 `LatentAdditive`。

同文件中参数（如 `n_layers`、`encoder_width`、`latent_dim`、`dropout`、`softplus_output`）会直接传入类构造函数。

### 8.3 模型类做了什么

`LatentAdditive` 的典型路径：

1. 调用 `PerturbationModel` 父类初始化，接入统一优化器/调度器/mask逻辑；
2. 从 `datamodule` 读取维度信息（基因维度、扰动编码维度、协变量维度）；
3. 使用 `MixedPerturbationEncoder` 编码 perturbation；
4. 将 control 输入编码到 latent 空间，与 perturbation latent 做加和；
5. 解码到基因表达空间，输出预测；
6. 训练/验证时用 `auto_mse` 计算损失并打日志。

### 8.4 与数据配置的联动

该模型支持 `use_covs` 自动联动：

- 若 `datamodule.train_dataset.transform.use_covs=True`，则模型侧自动启用协变量拼接（即使配置里没显式开）。

这就是为什么在 sweep 中，`data.transform.use_covs` 与模型表现会直接相关。

### 8.5 在 train.py 中如何被调用

训练入口 `src/perturbench/modelcore/train.py` 会：

1. 解析 CLI / sweep 注入参数；
2. 通过 Hydra 实例化 `cfg.data` 与 `cfg.model`；
3. 执行 `trainer.fit(...)` 与可选 `trainer.test(...)`。

对 `latent_additive` 来说，你只要保证：

- `model=latent_additive`
- 相关超参与 `__init__` 对齐

就可以直接进入训练流程。

### 8.6 复制该模式接入新模型

如果你要新加 `my_model`，最省事的方式是：

1. 复制 `latent_additive.py` 为模板改网络结构；
2. 新建 `my_model.yaml` 并替换 `_target_`；
3. 在 `models/__init__.py` 注册；
4. 若 sweep 传新参数，补 `train.py` 参数白名单；
5. 按第 7 节做最小验证。

---

## 9. 常见问题速查

- **`_target_` 找不到类**
  - 检查 yaml 路径和类名是否一致
  - 检查 `models/__init__.py` 是否注册

- **训练时 batch 字段不存在**
  - 对齐现有 batch 协议（`pert_cell_counts`、`control_cell_counts`、covariate keys）
  - 优先参考 `linear_additive.py` 的字段访问方式

- **sweep 参数不生效**
  - 检查 `train.py` 的 `argparse` + `mapping` 是否包含新增参数

- **covariates 维度不匹配**
  - 检查 `use_covs` 是否与 transform 保持一致
  - 打印拼接前后 tensor shape

---

## 10. 推荐实践

- 新模型先做“最小可跑版本”，保证接口稳定后再加复杂机制。
- 尽量复用基类能力（mask、scheduler、logging），减少重复代码。
- 每加一个 sweep 参数，就同步补 `train.py` 映射，避免线上 trial 空跑。

