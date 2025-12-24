#!/bin/bash
# ==========================================
# 一键安装 Miniconda + perturbench 环境 + Jupyter kernel
# ==========================================

# 设置安装路径和环境路径
CONDA_DIR="/fs-computility/prime/maoxinjie/miniconda3"
ENV_NAME="perturbench"
ENV_PATH="$CONDA_DIR/envs/$ENV_NAME"
REQ_FILE="/fs-computility/requirements_clean.txt"  # 修改为你requirements路径

# 下载 Miniconda（如果没下载的话）
MINICONDA_SH="Miniconda3-latest-Linux-x86_64.sh"
if [ ! -f "$MINICONDA_SH" ]; then
    wget https://repo.anaconda.com/miniconda/$MINICONDA_SH
fi

# 安装 Miniconda
bash $MINICONDA_SH -b -p $CONDA_DIR

# 配置 PATH
export PATH="$CONDA_DIR/bin:$PATH"

# 更新 conda
conda update -n base -c defaults conda -y

# 创建 perturbench 环境
conda create -n $ENV_NAME python=3.11 -y

# 激活环境
source $CONDA_DIR/bin/activate $ENV_NAME

# 安装 pip 依赖
if [ -f "$REQ_FILE" ]; then
    pip install --upgrade pip
    pip install -r $REQ_FILE
else
    echo "Warning: requirements file not found: $REQ_FILE"
fi

# 安装 Jupyter kernel
pip install ipykernel
python -m ipykernel install --name $ENV_NAME --display-name "$ENV_NAME" --sys-prefix

echo "========================================="
echo "Installation complete!"
echo "Activate your environment with:"
echo "    conda activate $ENV_NAME"
echo "Use Jupyter Notebook/Lab and select kernel: $ENV_NAME"
echo "========================================="
