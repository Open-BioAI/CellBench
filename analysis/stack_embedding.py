#!/usr/bin/env python3
"""
使用 Stack 模型生成 cell embeddings。

在运行 stack-embedding 之前，会自动为 adata 添加 'organism' 列（值为 'Homo sapiens'），
以满足 Stack 模型的要求。

输出格式：
- 默认输出 npy 文件（包含 embeddings）
- 可选：使用 --save-h5ad 将 embeddings 加载到 adata.obsm 并保存为 h5ad

示例：
    # 只生成 npy 文件
    python stack_embedding.py \\
        --checkpoint data/stack/bc_large.ckpt \\
        --adata input.h5ad \\
        --genelist data/stack/basecount_1000per_15000max.pkl \\
        --output embeddings.npy \\
        --gene-name-col feature_name

    # 生成 npy 并保存到 h5ad
    python stack_embedding.py \\
        --checkpoint data/stack/bc_large.ckpt \\
        --adata input.h5ad \\
        --genelist data/stack/basecount_1000per_15000max.pkl \\
        --output embeddings.npy \\
        --save-h5ad output_with_embeddings.h5ad
"""

import argparse
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil

import numpy as np
import scanpy as sc
import anndata as ad
import h5py


def add_organism_column(adata: ad.AnnData, organism: str = "Homo sapiens") -> ad.AnnData:
    """
    为 adata 添加 'organism' 列。

    Parameters
    ----------
    adata : ad.AnnData
        输入的 AnnData 对象
    organism : str
        生物体名称，默认为 "Homo sapiens"

    Returns
    -------
    ad.AnnData
        添加了 'organism' 列的 AnnData 对象（可能是原对象的副本或视图）
    """
    # 如果已经存在 organism 列，先检查是否都是 Homo sapiens
    if "organism" in adata.obs.columns:
        existing_org = adata.obs["organism"].unique()
        if len(existing_org) == 1 and existing_org[0] == organism:
            print(f"✓ 'organism' column already exists with value '{organism}'")
            return adata
        else:
            print(f"⚠ Warning: 'organism' column exists with values: {existing_org}")
            print(f"  Overwriting with '{organism}'")
    
    # 添加或覆盖 organism 列
    adata.obs["organism"] = organism
    print(f"✓ Added 'organism' column with value '{organism}'")
    return adata


def run_stack_embedding(
    checkpoint: str,
    adata_path: str,
    genelist: str,
    output_path: str,
    batch_size: int = 32,
    gene_name_col: str | None = None,
    use_temp_file: bool = True,
    save_h5ad: str | None = None,
    embed_key: str = "stack-embed",
) -> None:
    """
    运行 stack-embedding 命令生成 embeddings。

    Parameters
    ----------
    checkpoint : str
        Stack 模型 checkpoint 路径
    adata_path : str
        输入 h5ad 文件路径
    genelist : str
        基因列表文件路径
    output_path : str
        输出文件路径（npy 格式）
    batch_size : int
        批处理大小，默认为 32
    gene_name_col : str | None
        基因名称列名（如果 adata.var 中有该列），默认为 None
    use_temp_file : bool
        是否使用临时文件（先添加 organism 列后保存），默认为 True
    save_h5ad : str | None
        如果提供，会将 embeddings 加载到 adata 并保存为 h5ad，默认为 None
    embed_key : str
        保存到 adata.obsm 的键名，默认为 "stack-embed"
    """
    checkpoint = Path(checkpoint)
    adata_path = Path(adata_path)
    genelist = Path(genelist)
    output_path = Path(output_path)

    # 检查输入文件是否存在
    for path, name in [
        (checkpoint, "Checkpoint"),
        (adata_path, "Input h5ad"),
        (genelist, "Genelist"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    # 读取 adata 并添加 organism 列
    print(f"Reading AnnData from: {adata_path}")
    try:
        adata = sc.read_h5ad(adata_path)
    except Exception as e:
        # 如果遇到 IORegistryError（null encoding），尝试清理 uns 中的问题键
        if "IORegistryError" in str(type(e).__name__) or "null" in str(e).lower():
            print(f"⚠ Warning: Encountered encoding error: {e}")
            print("  Attempting to fix by cleaning uns metadata...")
            import h5py
            # 读取并清理有问题的 uns 键
            with h5py.File(adata_path, 'r+') as f:
                if 'uns/log1p' in f:
                    del f['uns/log1p']
                    print("  Removed problematic 'uns/log1p' key")
            # 重新读取
            adata = sc.read_h5ad(adata_path)
        else:
            raise
    print(f"  Shape: {adata.shape}")

    adata = add_organism_column(adata, organism="Homo sapiens")

    # 确定使用的 adata 路径
    temp_adata_path = None
    if use_temp_file:
        # 创建临时文件
        temp_dir = Path(tempfile.gettempdir())
        temp_adata_path = temp_dir / f"stack_embedding_temp_{adata_path.stem}.h5ad"
        print(f"\nWriting temporary AnnData with 'organism' column to: {temp_adata_path}")
        adata.write(temp_adata_path)
        final_adata_path = temp_adata_path
    else:
        # 直接覆盖原文件（不推荐，但可以节省空间）
        print(f"\n⚠ Warning: Overwriting input file with 'organism' column")
        adata.write(adata_path)
        final_adata_path = adata_path

    # 构建 stack-embedding 命令
    cmd = [
        "stack-embedding",
        "--checkpoint", str(checkpoint),
        "--adata", str(final_adata_path),
        "--genelist", str(genelist),
        "--output", str(output_path),
        "--batch-size", str(batch_size),
    ]
    
    # 添加 gene-name-col 参数（如果提供）
    if gene_name_col is not None:
        cmd.extend(["--gene-name-col", gene_name_col])

    print(f"\nRunning stack-embedding command:")
    print(f"  {' '.join(cmd)}")
    print()

    # 运行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Successfully generated embeddings!")
        print(f"  Output saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ stack-embedding failed with exit code {e.returncode}")
        raise
    finally:
        # 清理临时文件
        if use_temp_file and temp_adata_path.exists():
            print(f"\nCleaning up temporary file: {temp_adata_path}")
            temp_adata_path.unlink()
    
    # 如果指定了 save_h5ad，加载 npy 并保存到 h5ad
    if save_h5ad is not None:
        save_h5ad = Path(save_h5ad)
        print(f"\nLoading embeddings from: {output_path}")
        embeddings = np.load(output_path)
        print(f"  Embeddings shape: {embeddings.shape}")
        
        # 验证形状匹配
        if embeddings.shape[0] != adata.n_obs:
            raise ValueError(
                f"Embeddings shape {embeddings.shape[0]} doesn't match adata.n_obs {adata.n_obs}"
            )
        
        # 加载到 adata.obsm
        adata.obsm[embed_key] = embeddings
        print(f"✓ Added embeddings to adata.obsm['{embed_key}']")
        
        # 保存 h5ad
        print(f"\nSaving AnnData with embeddings to: {save_h5ad}")
        save_h5ad.parent.mkdir(parents=True, exist_ok=True)
        adata.write(save_h5ad)
        print(f"✓ Saved AnnData with embeddings to: {save_h5ad}")


def main():
    """主函数：解析命令行参数并运行 stack embedding"""
    parser = argparse.ArgumentParser(
        description="Generate cell embeddings using Stack model. "
                    "Automatically adds 'organism' column to adata before processing."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Stack model checkpoint file (e.g., bc_large.ckpt)",
    )
    parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="Path to input h5ad file",
    )
    parser.add_argument(
        "--genelist",
        type=str,
        required=True,
        help="Path to genelist file (e.g., basecount_1000per_15000max.pkl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output npy file with embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )
    parser.add_argument(
        "--gene-name-col",
        type=str,
        default=None,
        help="Gene name column in adata.var (e.g., 'feature_name')",
    )
    parser.add_argument(
        "--save-h5ad",
        type=str,
        default=None,
        help="Optional: Save adata with embeddings loaded to adata.obsm['stack-embed'] as h5ad file",
    )
    parser.add_argument(
        "--embed-key",
        type=str,
        default="stack-embed",
        help="Key name for embeddings in adata.obsm (default: 'stack-embed')",
    )
    parser.add_argument(
        "--no-temp-file",
        action="store_true",
        help="Don't use temporary file (will overwrite input file with organism column)",
    )

    args = parser.parse_args()

    run_stack_embedding(
        checkpoint=args.checkpoint,
        adata_path=args.adata,
        genelist=args.genelist,
        output_path=args.output,
        batch_size=args.batch_size,
        gene_name_col=args.gene_name_col,
        use_temp_file=not args.no_temp_file,
        save_h5ad=args.save_h5ad,
        embed_key=args.embed_key,
    )


if __name__ == "__main__":
    main()

