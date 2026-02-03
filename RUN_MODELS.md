# 模型运行指令整理

本文档整理了项目中所有现有模型的运行shell指令。

## 环境设置

所有模型运行前需要设置以下环境变量：

```bash
# 设置Python路径
export PYTHONPATH=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src:${PYTHONPATH}

# 设置Wandb相关环境变量（可选，如果使用wandb）
export WANDB_API_KEY=c24d277403208674c2360ed46c8a8812a74911b6
export WANDB_DIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs
export WANDB_CACHE_DIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs/.cache

# 设置临时目录。若报错 OSError: AF_UNIX path too long，请改用短路径：export TMPDIR=/tmp
export TMPDIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/tmp

# 设置工作目录
cd /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main
```

## Python可执行文件路径

```bash
PYTHON=/fs-computility-new/upzd_share/maoxinjie/miniconda3/envs/qianhong_env/bin/python
SCRIPT=src/perturbench/modelcore/train.py
```

## 模型运行指令

### 1. Latent Additive

**方式一：使用shell脚本**
```bash
bash scripts/latent_additive.sh
```

**方式二：直接运行命令**
```bash
export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export WANDB_API_KEY=c24d277403208674c2360ed46c8a8812a74911b6

train trainer.devices=[0] \
    data=mix_pert \
    data.task=unseen_cell \
    data.data_path="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/all_cell_line_filterdrug.h5ad" \
    model=latent_additive \
    logger=wandb \
    logger.wandb.project="perturbench" \
    logger.wandb.name="latent_additive_newloss" \
    trainer.log_every_n_steps=5 \
    trainer.max_epochs=20 \
    trainer.min_epochs=1
```

**方式三：使用Python直接调用**
```bash
$PYTHON $SCRIPT \
    trainer.devices=[0] \
    data=mix_pert \
    data.task=unseen_cell \
    data.data_path="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/all_cell_line_filterdrug.h5ad" \
    model=latent_additive \
    logger=wandb \
    logger.wandb.project="perturbench" \
    logger.wandb.name="latent_additive_newloss" \
    trainer.log_every_n_steps=5 \
    trainer.max_epochs=20 \
    trainer.min_epochs=1
```

---

### 2. Biolord

**方式一：使用shell脚本**
```bash
bash scripts/biolord.sh
```

**方式二：直接运行命令**
```bash
export PYTHONPATH=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src:${PYTHONPATH}
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=c24d277403208674c2360ed46c8a8812a74911b6
export WANDB_DIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs
export WANDB_CACHE_DIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs/.cache
export TMPDIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/tmp

cd /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main
PYTHON=/fs-computility-new/upzd_share/maoxinjie/miniconda3/envs/qianhong_env/bin/python
SCRIPT=src/perturbench/modelcore/train.py

$PYTHON $SCRIPT \
    data=mix_pert \
    data.task=srivatsantrapnell2020_sciplex3 \
    data.data_path=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/all_cell_line_filterdrug.h5ad \
    model=biolord \
    logger=wandb \
    logger.wandb.project=perturbench \
    logger.wandb.name=biolord_srivatsantrapnell2020_sciplex3_updatePCC_useCovs \
    trainer.log_every_n_steps=10 \
    trainer.max_epochs=2 \
    trainer.min_epochs=0 \
    trainer.devices=[1] \
    model.use_mask=true \
    train=true \
    test=true \
    test_ckpt_type=pcc \
    data.transform.use_covs=true
```

---

### 3. CPA

**方式一：使用shell脚本**
```bash
bash scripts/cpa.sh
```

**方式二：直接运行命令**
```bash
export PYTHONPATH=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src:${PYTHONPATH}
export WANDB_API_KEY=c24d277403208674c2360ed46c8a8812a74911b6
export WANDB_DIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs
export WANDB_CACHE_DIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/wandb_logs/.cache
export TMPDIR=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/tmp

cd /fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main
PYTHON=/fs-computility-new/upzd_share/maoxinjie/miniconda3/envs/qianhong_env/bin/python
SCRIPT=src/perturbench/modelcore/train.py

$PYTHON $SCRIPT \
    data=mix_pert \
    data.task=srivatsantrapnell2020_sciplex3 \
    data.data_path=/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/all_cell_line_filterdrug.h5ad \
    model=cpa \
    logger=wandb \
    logger.wandb.project=perturbench \
    logger.wandb.name=cpa_srivatsantrapnell2020_sciplex3 \
    trainer.log_every_n_steps=10 \
    trainer.max_epochs=1 \
    trainer.min_epochs=0 \
    trainer.devices=[0] \
    model.use_mask=true \
    train=true \
    test=true \
    test_ckpt_type=pcc
```

---

### 4. GEARS

**方式一：使用shell脚本**
```bash
bash scripts/gears.sh
```

**方式二：直接运行命令**
```bash
train \
    data=gears \
    model=gears \
    trainer=gears \
    trainer.devices=[0,1,2,3]
```

---

### 5. GenePert

**方式一：使用shell脚本**
```bash
bash scripts/genepert.sh
```

**方式二：直接运行命令**
```bash
export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export HYDRA_FULL_ERROR=1

train trainer.devices=[0] \
    data=mix_pert \
    model=genepert \
    data.data_path="/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess/total/all_cell_line_filterdrug_subsampled.h5ad"
```

---

### 6. PRNet

**方式一：使用shell脚本**
```bash
bash scripts/prnet.sh
```

**方式二：直接运行命令**
```bash
export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export WANDB_API_KEY=c24d277403208674c2360ed46c8a8812a74911b6

train \
    trainer.devices=[0] \
    data=mix_pert \
    logger=wandb \
    model=prnet \
    data.data_path='/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess/total/all_cell_line_filterdrug_subsampled.h5ad' \
    trainer.min_epochs=50 \
    trainer.max_epochs=500 \
    logger.wandb.project="perturbench" \
    logger.wandb.name="prnet" \
    trainer.log_every_n_steps=5
```

---

### 7. SAMS VAE

**方式一：使用shell脚本**
```bash
bash scripts/sams_vae.sh
```

**方式二：直接运行命令**
```bash
export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export HYDRA_FULL_ERROR=1

train trainer.devices=[0] \
    data=mix_pert \
    model=sams_vae \
    data.data_path="/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess/total/all_cell_line_filterdrug_subsampled.h5ad"
```

---

### 8. Decoder Only

**方式一：使用shell脚本**
```bash
bash scripts/decoder_only.sh
```

**方式二：直接运行命令**
```bash
export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export HYDRA_FULL_ERROR=1

train trainer.devices=[0] \
    data=mix_pert \
    model=decoder_only \
    data.data_path="/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess/total/all_cell_line_filterdrug_subsampled.h5ad"
```

---

### 9. Linear Additive

**方式一：使用shell脚本**
```bash
bash scripts/linear_additive.sh
```

**方式二：直接运行命令**
```bash
export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export HYDRA_FULL_ERROR=1

train trainer.devices=[0] \
    data=mix_pert \
    model=linear_additive \
    data.data_path="/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess/total/all_cell_line_filterdrug_subsampled.h5ad"
```

---

### 10. scLAMBDA

**方式一：使用shell脚本**
```bash
bash scripts/sclambda.sh
```

**方式二：直接运行命令**
```bash
export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export HYDRA_FULL_ERROR=1

train trainer.devices=[0] \
    data=mix_pert \
    model=sclambda \
    data.data_path="/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess/total/all_cell_line_filterdrug_subsampled.h5ad"
```

---

### 11. Squidiff

**方式一：使用shell脚本**
```bash
bash scripts/squidiff.sh
```

**方式二：直接运行命令**
```bash
export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export HYDRA_FULL_ERROR=1

train trainer.devices=[0] \
    data=mix_pert \
    model=squidiff \
    data.data_path="/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess/total/all_cell_line_filterdrug_subsampled.h5ad"
```

---

### 12. State

**方式一：使用shell脚本**
```bash
bash scripts/state.sh
```

**方式二：直接运行命令**
```bash
export PYTHONPATH=$PYTHONPATH:/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src
export HYDRA_FULL_ERROR=1

train trainer.devices=[0,1,2,3] \
    trainer.strategy='ddp' \
    data=mix_pert \
    model=state_sm \
    data.data_path='/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess/total/all_cell_line_filterdrug_subsampled.h5ad'
```

---

## 通用运行模板

如果`train`命令已配置为别名或通过pip安装，可以使用以下通用模板：

```bash
# 基础模板
train \
    trainer.devices=[0] \
    data=mix_pert \
    data.task=<task_name> \
    data.data_path="<data_path>" \
    model=<model_name> \
    logger=wandb \
    logger.wandb.project="perturbench" \
    logger.wandb.name="<run_name>" \
    trainer.log_every_n_steps=10 \
    trainer.max_epochs=<max_epochs> \
    trainer.min_epochs=<min_epochs> \
    train=true \
    test=true \
    test_ckpt_type=pcc
```

如果`train`命令不可用，使用Python直接调用：

```bash
# Python直接调用模板
PYTHON=/fs-computility-new/upzd_share/maoxinjie/miniconda3/envs/qianhong_env/bin/python
SCRIPT=src/perturbench/modelcore/train.py

$PYTHON $SCRIPT \
    trainer.devices=[0] \
    data=mix_pert \
    data.task=<task_name> \
    data.data_path="<data_path>" \
    model=<model_name> \
    logger=wandb \
    logger.wandb.project="perturbench" \
    logger.wandb.name="<run_name>" \
    trainer.log_every_n_steps=10 \
    trainer.max_epochs=<max_epochs> \
    trainer.min_epochs=<min_epochs> \
    train=true \
    test=true \
    test_ckpt_type=pcc
```

## 常用参数说明

- `trainer.devices=[0]`: 使用的GPU设备编号，可以是单个`[0]`或多个`[0,1,2,3]`
- `data=mix_pert`: 数据集配置组
- `data.task`: 具体任务名称，如`unseen_cell`、`srivatsantrapnell2020_sciplex3`等
- `data.data_path`: 数据文件路径
- `model`: 模型名称，如`latent_additive`、`biolord`、`cpa`等
- `logger`: 日志记录器，常用`wandb`
- `trainer.max_epochs`: 最大训练轮数
- `trainer.min_epochs`: 最小训练轮数
- `trainer.log_every_n_steps`: 每N步记录一次日志
- `train=true`: 是否训练
- `test=true`: 是否测试
- `test_ckpt_type`: 测试时使用的checkpoint类型，`pcc`或`loss`
- `model.use_mask`: 是否使用mask（某些模型支持）
- `data.transform.use_covs`: 是否使用协变量（某些模型支持）

## 注意事项

1. 确保conda环境已激活：`conda activate qianhong_env`
2. 确保数据文件路径正确
3. 根据可用GPU数量调整`trainer.devices`参数
4. 多GPU训练时，某些模型需要使用`trainer.strategy='ddp'`
5. 如果遇到路径问题，确保`PYTHONPATH`已正确设置
6. 若报错 `OSError: AF_UNIX path too long`：运行前执行 `export TMPDIR=/tmp` 使用短临时目录即可
