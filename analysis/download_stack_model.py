#!/usr/bin/env python3
"""
下载 Stack-Large 模型从 HuggingFace。

这个脚本会从 HuggingFace Hub 下载 Stack-Large 模型到本地目录。
如果目标目录已存在且不为空，脚本会报错退出。
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download


def download_stack_model(
    repo_id: str = "arcinstitute/Stack-Large",
    target_dir: str = "/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/stack",
    revision: str = "main",
    local_dir_use_symlinks: bool = False,
    resume_download: bool = True,
) -> None:
    """
    下载 Stack 模型从 HuggingFace Hub。

    Parameters
    ----------
    repo_id : str
        HuggingFace 仓库 ID，默认为 "arcinstitute/Stack-Large"
    target_dir : str
        本地保存目录路径
    revision : str
        模型版本/分支，默认为 "main"
    local_dir_use_symlinks : bool
        是否使用符号链接，默认为 False（使用硬拷贝）
    resume_download : bool
        是否支持断点续传，默认为 True
    """
    target_dir = Path(target_dir)
    
    # 检查目标目录是否已存在且不为空
    if target_dir.exists() and any(target_dir.iterdir()):
        raise SystemExit(
            f"Error: '{target_dir}' already exists and is not empty. "
            f"Please remove it first or choose a different target directory."
        )
    
    # 创建目标目录（如果不存在）
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Stack model from HuggingFace...")
    print(f"  Repository: {repo_id}")
    print(f"  Revision: {revision}")
    print(f"  Target directory: {target_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=str(target_dir),
            local_dir_use_symlinks=local_dir_use_symlinks,
            resume_download=resume_download,
        )
        print(f"\n✓ Successfully downloaded Stack model to: {target_dir}")
    except Exception as e:
        print(f"\n✗ Failed to download Stack model: {e}")
        raise


def main():
    """主函数：下载 Stack-Large 模型"""
    download_stack_model()


if __name__ == "__main__":
    main()

