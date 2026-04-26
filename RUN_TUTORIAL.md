# 运行代码教程

## 1. 数据准备

我们在 Google Drive 提供了一部分复现实验所需要的数据：
https://drive.google.com/drive/folders/1GrPW9x5_npnT7ILwDVsFWvfDIcqaSjdk?usp=sharing

我们在 Google Drive 的目录下提供了该论文用到的数据。

## 2. 实验教程

我们将给出跑通 unseen perturbation 实验作为教程。

## 3. 下载数据

分别将 Google Drive 的 `unseen_perts` 目录和 `model_related` 目录下载到项目的根目录下。

## 4. 配置数据路径

### Norman 数据集

将 `./sweep/norman` 下的所有 `.yaml` 文件中 `parameters.data.data_path` 全都设置为：
```yaml
./unseen_perts/norman2019_comb_stack.h5ad
```

### Replogle 数据集

将 `./sweep/replogle` 下的所有 `.yaml` 文件中 `parameters.data.data_path` 全都设置为：
```yaml
./unseen_perts/ReplogleWeissman2022_K562_stack_hvg_split.h5ad
```

### Sciplex 数据集

将 `./sweep/sciplex` 下的所有 `.yaml` 文件中 `parameters.data.data_path` 全都设置为：
```yaml
./unseen_perts/SrivatsanTrapnell2020_sciplex3_stack_hvg_split.h5ad
```

## 5. 配置特征映射路径

### Norman 和 Replogle 数据集

将 `./sweep/norman` 和 `./sweep/replogle` 下的所有 `.yaml` 文件中 `parameters.data.transform.gene_map_path` 全都设置为：
```yaml
./model_related/ESM2_pert_features.pt
```

### Sciplex 数据集

将 `./sweep/sciplex` 下的所有 `.yaml` 文件中 `parameters.data.transform.drug_map_path` 全都设置为：
```yaml
./model_related/SMILES_pert_features.pt
```

## 6. GEARS 模型配置

对于 GEARS，需要如下设置：

### 设置数据路径

- 将 `./sweep/norman/no-stack/gears.yaml` 的 `parameters.model.data_path` 全都设置为：
  ```yaml
  ./gears_norman
  ```

- 将 `./sweep/replogle/no-stack/gears.yaml` 的 `parameters.model.data_path` 全都设置为：
  ```yaml
  ./gears_replogle
  ```

### 设置基因映射路径

将 `./src/perturbench/configs/model/gears.yaml` 中的以下参数设置：

- `gene2go_path` 设置为：
  ```yaml
  ./model_related/gene2go.pkl
  ```

- `gene_set_path` 设置为：
  ```yaml
  ./model_related/essential_all_data_pert_genes.pkl
  ```

## 7. 环境配置

### 安装并激活环境

首先在终端切换到该项目根目录下：

```bash
conda env create -f ./vcbench.yml
conda activate vcbench
```

### 配置 WandB

```bash
wandb login
```

输入你的 API KEY。

## 8. 运行实验

对 `./sweep` 下所有 `.yaml` 文件都能这样运行：

```bash
wandb sweep [yaml_path]
```

终端会返回：Run Sweep Agent With: xxxxx

将 `xxxxx` 复制粘贴在终端再按回车就能启动进程。

## 9. 查看进度

在浏览器登录 WandB 查看进程。