#!/usr/bin/env python3
"""
预处理数据脚本：将 split CSV 信息匹配到 h5ad 数据文件中

这个脚本按照 modules.py 中的逻辑，提前处理数据，生成已经包含 split 信息的数据文件。
这样在训练时就不需要每次都进行 CSV 匹配了。
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import issparse

def _choose_csv(path: Path) -> Path:
    """选择 CSV 文件，逻辑与 modules.py 一致"""
    if path.is_file():
        return path
    # If multiple files, pick the first sorted for determinism
    csvs = sorted([p for p in path.glob("*.csv") if p.is_file()])
    if not csvs:
        raise FileNotFoundError(f"No CSV found under {path}")
    return csvs[0]


def preprocess_data_with_split(
    data_path: str,
    task: str,
    split_dir: str | None = None,
    output_path: str | None = None,
):
    """
    预处理数据，将 split CSV 信息匹配到 h5ad 数据文件中
    
    Args:
        data_path: 输入的 h5ad 文件路径
        task: task 名称（如 'srivatsantrapnell2020_sciplex3'）
        split_dir: split 目录路径，如果为 None 则使用默认路径
        output_path: 输出文件路径，如果为 None 则自动生成
    """
    print(f"Loading data from: {data_path}")
    adata = sc.read_h5ad(data_path)
    print(f"  Original adata shape: {adata.shape}")
    
    # 确定 split 目录
    if split_dir is None:
        # 默认 split 目录在项目根目录下的 split 文件夹
        current_file = Path(__file__)
        project_root = current_file.parent
        split_dir = project_root / "split"
        if not split_dir.exists():
            split_dir = Path("./split")
    else:
        split_dir = Path(split_dir)
    
    # 根据 task 确定 CSV 文件路径
    task_lower = str(task).lower()
    split_subdir = split_dir / task_lower
    
    # 选择 CSV 文件（所有 task 都使用相同的逻辑）
    csv_path = _choose_csv(split_subdir)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV file not found: {csv_path}")
    
    print(f"Loading split CSV from: {csv_path}")
    # 读取 CSV，确保 split 列是字符串类型，不是 Categorical
    split_df = pd.read_csv(csv_path, index_col=0)
    # 去掉列名两端可能的空格或不可见字符
    split_df.columns = split_df.columns.astype(str).str.strip()
    
    if 'split' not in split_df.columns:
        raise KeyError(
            f"'split' column not found in CSV {csv_path}. "
            f"Available columns: {list(split_df.columns)}"
        )
    
    # 将 split 列转换为字符串类型（处理可能的 NaN 或空字符串）
    split_df['split'] = split_df['split'].astype(str)
    # 将空字符串和 'nan' 字符串替换为空字符串，然后过滤掉
    split_df['split'] = split_df['split'].replace(['nan', 'NaN', ''], '')
    
    # 确保 CSV 的索引（cell ID）与 adata 的索引匹配
    valid_split_mask = split_df['split'].isin(['train', 'val', 'test'])
    valid_cells = split_df[valid_split_mask].index
    
    # 检查哪些细胞在 adata 中存在
    cells_in_adata = adata.obs.index.isin(valid_cells)
    n_matched = cells_in_adata.sum()
    n_total_valid = len(valid_cells)
    
    print(f"  Found {n_matched:,} cells in adata matching CSV (out of {n_total_valid:,} valid cells in CSV)")
    print(f"  Original adata shape: {adata.shape}")
    
    # 只保留在 CSV 中有有效 split 值的细胞
    adata = adata[cells_in_adata].copy()
    
    # 如果原来的 split 列存在，先删除以避免类型冲突
    if 'split' in adata.obs.columns:
        del adata.obs['split']
    
    split_series = split_df.loc[adata.obs.index, 'split']
    split_values = split_series.values.astype(str)
    split_series_new = pd.Series(split_values, index=adata.obs.index, dtype='object')
    adata._obs.loc[:, 'split'] = split_series_new
    adata._obs['split'] = adata._obs['split'].astype('object')
    
    assert not pd.api.types.is_categorical_dtype(adata.obs['split']), \
        "Failed to set split column as non-Categorical type"
    
    # If CSV contains sample_count, duplicate cells accordingly (with replacement)
    if 'sample_count' in split_df.columns:
        print("  Applying sample_count from CSV...")
        counts = pd.to_numeric(
            split_df.loc[adata.obs.index, 'sample_count'],
            errors='coerce'
        ).fillna(1).astype(int)
        counts[counts < 0] = 0  # negative counts are treated as zero
        repeat_idx = np.repeat(np.arange(adata.n_obs), counts.values)
        if len(repeat_idx) == 0:
            raise ValueError("After applying sample_count, no cells remain to subset.")
        adata = adata[repeat_idx].copy()
        # Ensure unique obs names after duplication
        adata.obs_names_make_unique()
        print(f"  After sample_count duplication: {adata.shape}")
    
    # Drop cells whose expression is all zeros (sparse/dense safe)
    X = adata.X
    if issparse(X):
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        col_sums = np.asarray(X.sum(axis=0)).ravel()
    else:
        row_sums = np.sum(X, axis=1)
        col_sums = np.sum(X, axis=0)
    
    # Drop cells with all-zero expression
    nonzero_mask = row_sums != 0
    n_drop = int((~nonzero_mask).sum())
    if n_drop > 0:
        print(f"  Dropping {n_drop:,} all-zero expression cells")
        adata = adata[nonzero_mask].copy()
        # recompute column sums after cell filtering
        X = adata.X
        if issparse(X):
            col_sums = np.asarray(X.sum(axis=0)).ravel()
        else:
            col_sums = np.sum(X, axis=0)
    
    # Drop genes (columns) that are all zero after subsetting
    gene_nonzero = col_sums != 0
    n_genes_drop = int((~gene_nonzero).sum())
    if n_genes_drop > 0:
        print("  Dropping {n_genes_drop:,} all-zero genes")
        adata = adata[:, gene_nonzero].copy()
    
    print("  Final adata shape: {adata.shape}")
    
    # 验证 split 列
    split_counts = adata.obs['split'].value_counts()
    print("  Split distribution:")
    for split_name, count in split_counts.items():
        print(f"    {split_name}: {count:,} cells")
    
    # 确定输出路径
    if output_path is None:
        data_path_obj = Path(data_path)
        output_path = data_path_obj.parent / f"{data_path_obj.stem}_with_split.h5ad"
    
    print(f"\nSaving processed data to: {output_path}")
    adata.write_h5ad(output_path, compression='gzip')
    print("✓ Successfully saved processed data!")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess h5ad data by matching split CSV information"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/all_cell_line_filterdrug.h5ad",
        help="Path to input h5ad file"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="srivatsantrapnell2020_sciplex3",
        help="Task name (e.g., 'srivatsantrapnell2020_sciplex3')"
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Directory containing split CSV files (default: project_root/split)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output file path (default: {data_path}_with_split.h5ad)"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = preprocess_data_with_split(
            data_path=args.data_path,
            task=args.task,
            split_dir=args.split_dir,
            output_path=args.output_path,
        )
        print("\n✓ Processing completed successfully!")
        print(f"  Output file: {output_path}")
    except Exception as e:
        print(f"\n✗ Error during processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

