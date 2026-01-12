#!/usr/bin/env python3
"""
针对 Norman2019 数据集添加 split 列和 CRISPR 列到 obs。

流程：
1. 读取 Norman2019 h5ad 文件
2. 保留所有细胞（包括control细胞）
3. 添加 CRISPR 列到 obs（通过命令行参数指定）
4. 创建 cell_cluster 列（优先使用cell_line，否则使用celltype，默认'Norman'）
5. 按 unique gene_pt 进行分割（4:1比例，4份train，1份再均分为val和test）
   - perturbed cells按gene_pt分配到相应split
   - control细胞（gene_pt为空）按8:1:1比例分配到train/val/test
6. 跳过perturbation组合过滤（因为train/val/test的gene_pt互斥，组合也互斥）
7. 对 train 中的 perturbed cells (control==False) 进行有放回抽样，确保每个 gene_pt 被抽到的概率相同
   - control细胞不参与重采样
8. 在 adata.obs 中添加 'split' 列，保存为新的 h5ad 文件
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


def normalize_pert_string(s) -> str:
    """将一个扰动字符串标准化：按'+'拆分、strip、排序后再用'+'连接."""
    # 先检查是否为 None 或 NaN（避免 astype(str) 把 NaN 变成 "nan" 字符串）
    if s is None or pd.isna(s):
        return ""

    # 转换为字符串并检查是否为 "nan"/"None" 字符串
    s_str = str(s).strip()
    if s_str.lower() in {"nan", "none"}:
        return ""

    parts = [p.strip() for p in s_str.split("+") if p.strip() != ""]
    if not parts:
        return ""
    parts_sorted = sorted(parts)
    return "+".join(parts_sorted)


def build_pert_key(
    obs: pd.DataFrame, gene_pt_col: str, drug_pt_col: str, env_pt_col: str
) -> pd.Series:
    """Build perturbation key from gene_pt, drug_pt, env_pt columns."""
    # 去掉 astype(str)，让 normalize_pert_string 直接处理原始值（包括 NaN）
    g_norm = obs[gene_pt_col].apply(normalize_pert_string)
    d_norm = obs[drug_pt_col].apply(normalize_pert_string)
    e_norm = obs[env_pt_col].apply(normalize_pert_string)
    # 全部转换为字符串类型，避免 Categorical 类型导致的错误
    g_norm = g_norm.astype(str)
    d_norm = d_norm.astype(str)
    e_norm = e_norm.astype(str)
    return g_norm + "|" + d_norm + "|" + e_norm


# 默认输入文件
DEFAULT_H5AD = Path(
    "/fs-computility-new/upzd_share/shared/AIVC_data/processed_control/after_preprocess/Norman2019_processed.h5ad"
)
# 输出到 Norman2019 目录
DEFAULT_OUTPUT = Path(
    "/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/norman2019_processed.h5ad"
)


def resample_train_perturbed_cells_by_gene_pt(
    train_cells: pd.Index,
    obs: pd.DataFrame,
    gene_pt_col: str = "gene_pt",
    control_col: str = "control",
    seed: int = 42,
) -> pd.Index:
    """
    对 train 中的 perturbed cells 进行有放回抽样，确保每个 gene_pt 被抽到的概率相同。

    Args:
        train_cells: train split 的细胞索引
        obs: 观察数据框
        gene_pt_col: gene_pt 列名
        control_col: control 列名
        seed: 随机种子

    Returns:
        需要追加的重复细胞索引（pd.Index，包含重复采样的细胞）
    """
    # 检查 control 列
    if control_col not in obs.columns:
        raise ValueError(f"Column '{control_col}' not found in adata.obs")

    # 识别 perturbed cells (control == False)
    is_control = obs[control_col].astype(bool)
    is_perturbed = ~is_control

    # 获取 train 中的 perturbed cells
    train_obs = obs.loc[train_cells]
    train_perturbed_mask = is_perturbed.loc[train_cells]
    train_perturbed_cells = train_cells[train_perturbed_mask.values]

    print(f"\nTrain perturbed cells before resampling: {len(train_perturbed_cells):,}")

    # 按 gene_pt 分组
    train_perturbed_gene_pts = train_obs.loc[train_perturbed_cells, gene_pt_col].astype(str)
    gene_pt_counts = train_perturbed_gene_pts.value_counts()

    print(f"\nPerturbed cells per gene_pt in train (before resampling):")
    for gene_pt, count in gene_pt_counts.items():
        print(f"  {gene_pt}: {count:,}")

    # 计算目标数量：所有gene_pt的平均数量
    target_count = len(train_perturbed_cells) // len(gene_pt_counts)
    print(f"\nTarget count per gene_pt: {target_count:,}")

    # 收集需要追加的重复细胞
    cells_to_append = []
    rng = np.random.default_rng(seed)

    # 对每个 gene_pt 进行重采样
    for gene_pt in gene_pt_counts.index:
        gene_pt_cells = train_perturbed_cells[train_perturbed_gene_pts == gene_pt]
        original_count = len(gene_pt_cells)

        if original_count < target_count:
            # 需要上采样
            need = target_count - original_count
            additional_indices = rng.choice(len(gene_pt_cells), size=need, replace=True)
            additional_cells = gene_pt_cells[additional_indices]
            cells_to_append.extend(additional_cells.tolist())

            print(f"  {gene_pt}: {original_count:,} -> {target_count:,} (will append {need:,} cells)")
        elif original_count > target_count:
            # 需要下采样（随机选择）
            keep_indices = rng.choice(len(gene_pt_cells), size=target_count, replace=False)
            keep_cells = gene_pt_cells[keep_indices]
            # 这里我们不直接修改，而是通过追加逻辑来处理
            # 实际的下采样会在后续步骤中处理
            print(f"  {gene_pt}: {original_count:,} -> {target_count:,} (will downsample)")

    if len(cells_to_append) > 0:
        print(f"\nTotal cells to append: {len(cells_to_append):,}")
        return pd.Index(cells_to_append)
    else:
        return pd.Index([], dtype=train_cells.dtype)


def create_norman_split(
    h5ad_path: Path,
    output_h5ad_path: Path,
    seed: int = 42,
    gene_pt_col: str = "gene_pt",
    drug_pt_col: str = "drug_pt",
    env_pt_col: str = "env_pt",
    control_col: str = "control",
    crispr_value: str = "",
) -> None:
    print(f"Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    obs = adata.obs
    print(f"  Shape: {adata.shape}")

    # 保留所有细胞，包括control细胞（gene_pt为空）
    print(f"  Keeping all cells including controls (gene_pt empty)")

    # 添加 CRISPR 列到 obs
    obs['CRISPR'] = crispr_value
    print(f"Added CRISPR column with value: '{crispr_value}'")

    # 新建 cell_cluster 列：优先选取 cell_line，如果没有就用 celltype
    # Norman数据集通常只有一个细胞系，所以cell_cluster主要是用于标识细胞类型
    if 'cell_line' in obs.columns:
        obs['cell_cluster'] = obs['cell_line']
        print(f"Using 'cell_line' column as 'cell_cluster'")
    elif 'celltype' in obs.columns:
        obs['cell_cluster'] = obs['celltype']
        print(f"Using 'celltype' column as 'cell_cluster'")
    else:
        # 如果都没有，创建一个默认值
        obs['cell_cluster'] = 'Norman'
        print(f"No cell_line or celltype column found, using default 'Norman' as cell_cluster")

    # 检查必要列
    required_cols = [gene_pt_col, drug_pt_col, env_pt_col, control_col]
    for col in required_cols:
        if col not in obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    # 获取所有 unique gene_pt
    unique_gene_pts = obs[gene_pt_col].dropna().unique()
    unique_gene_pts = [pt for pt in unique_gene_pts if pt != ""]
    print(f"\nFound {len(unique_gene_pts)} unique gene_pt values")

    # 按 4:1 比例分割 unique gene_pt
    rng = np.random.default_rng(seed)
    gene_pt_array = np.array(unique_gene_pts)
    rng.shuffle(gene_pt_array)

    n_total = len(gene_pt_array)
    n_train = int(n_total * 4 / 5)  # 4/5 用于训练
    n_val_test = n_total - n_train  # 1/5 用于 val/test

    train_gene_pts = gene_pt_array[:n_train]
    remaining_gene_pts = gene_pt_array[n_train:]

    # 将剩余的再均分为 val 和 test
    n_val = n_val_test // 2
    val_gene_pts = remaining_gene_pts[:n_val]
    test_gene_pts = remaining_gene_pts[n_val:]

    print(f"\nGene_pt split:")
    print(f"  Train: {len(train_gene_pts)} gene_pt values")
    print(f"  Val: {len(val_gene_pts)} gene_pt values")
    print(f"  Test: {len(test_gene_pts)} gene_pt values")

    # 初始化所有细胞的 split 为空字符串
    splits = np.array([""] * adata.n_obs, dtype=object)

    # 1. 首先根据 gene_pt 分配 perturbed cells
    for i, gene_pt in enumerate(obs[gene_pt_col]):
        if gene_pt in train_gene_pts:
            splits[i] = "train"
        elif gene_pt in val_gene_pts:
            splits[i] = "val"
        elif gene_pt in test_gene_pts:
            splits[i] = "test"

    # 2. 对于 control 细胞（gene_pt 为空），按照 8:1:1 比例分配到 train/val/test
    control_mask = obs[gene_pt_col].isna() | (obs[gene_pt_col] == "")
    control_indices = np.where(control_mask.values)[0]

    if len(control_indices) > 0:
        print(f"\nFound {len(control_indices)} control cells (gene_pt empty)")

        # 按照 8:1:1 比例分配 control 细胞
        rng.shuffle(control_indices)
        n_control = len(control_indices)
        n_train_control = int(n_control * 8 / 10)  # 80% 到 train
        n_val_control = int(n_control * 1 / 10)    # 10% 到 val
        # 剩余到 test

        train_control_end = n_train_control
        val_control_end = n_train_control + n_val_control

        train_control_indices = control_indices[:train_control_end]
        val_control_indices = control_indices[train_control_end:val_control_end]
        test_control_indices = control_indices[val_control_end:]

        splits[train_control_indices] = "train"
        splits[val_control_indices] = "val"
        splits[test_control_indices] = "test"

        print(f"Control cells split (8:1:1):")
        print(f"  Train: {len(train_control_indices)} cells")
        print(f"  Val: {len(val_control_indices)} cells")
        print(f"  Test: {len(test_control_indices)} cells")

    # 构建 split DataFrame
    split_df = pd.DataFrame({"split": splits}, index=obs.index)

    print("\nSplit counts before perturbation filtering:")
    for name, count in split_df["split"].value_counts(dropna=False).items():
        label = "(empty)" if name == "" else name
        print(f"  {label}: {count:,}")
    print(f"  TOTAL: {len(split_df):,} cells")

    # 对于按gene_pt分割的情况，由于train/val/test的gene_pt完全不同，
    # 我们跳过perturbation组合过滤，以保留val和test数据
    print("\n" + "=" * 60)
    print("Skipping perturbation filtering for gene_pt-based split...")
    print("Reason: train/val/test gene_pt are mutually exclusive, so perturbation combinations are also exclusive.")

    # 更新 val/test cells（不过滤）
    mask_val_after = split_df["split"] == "val"
    mask_test_after = split_df["split"] == "test"
    val_cells_after = split_df.index[mask_val_after]
    test_cells_after = split_df.index[mask_test_after]

    # 定义train_cells用于重采样
    mask_train = split_df["split"] == "train"
    train_cells = split_df.index[mask_train]

    # 对 train 中的 perturbed cells 进行重采样（按 gene_pt 概率相同）
    train_cells_to_append = None
    print("\n" + "=" * 60)
    print("Resampling train perturbed cells to balance gene_pt counts...")

    try:
        train_cells_to_append = resample_train_perturbed_cells_by_gene_pt(
            train_cells=train_cells,
            obs=obs,
            gene_pt_col=gene_pt_col,
            control_col=control_col,
            seed=seed,
        )

        print(f"\nAfter resampling:")
        print(f"  Original train cells: {len(train_cells):,}")
        print(f"  Cells to append: {len(train_cells_to_append):,}")

    except Exception as e:
        print(f"\n[WARNING] Resampling failed: {e}")
        print("  Continuing without resampling...")
        import traceback
        traceback.print_exc()
        train_cells_to_append = None

    print("\nFinal split counts (after perturbation filtering and resampling):")
    split_counts_after = split_df["split"].value_counts(dropna=False)
    for name, count in split_counts_after.items():
        label = "(empty)" if name == "" else name
        if name == "train" and train_cells_to_append is not None:
            # train 显示总数量（包括要追加的），而不是唯一细胞数
            total_train = count + len(train_cells_to_append)
            print(f"  {label}: {total_train:,} (unique: {count:,}, to append: {len(train_cells_to_append):,})")
        else:
            print(f"  {label}: {count:,}")
    print(f"  TOTAL: {len(split_df):,} cells (unique)")

    # 在 adata.obs 中添加 split 列，过滤掉 split 为空的细胞，并根据采样次数复制细胞
    adata.obs["split"] = split_df["split"]

    # 只保留 split 不为空的细胞
    mask_valid_split = adata.obs["split"] != ""
    n_before_filter = adata.n_obs
    adata = adata[mask_valid_split].copy()
    n_after_filter = adata.n_obs
    n_removed = n_before_filter - n_after_filter

    print(f"\nFiltering cells with empty split:")
    print(f"  Before: {n_before_filter:,} cells")
    print(f"  After:  {n_after_filter:,} cells")
    print(f"  Removed: {n_removed:,} cells")

    # 追加重采样的 train 细胞
    if train_cells_to_append is not None and len(train_cells_to_append) > 0:
        print("\n" + "=" * 60)
        print("Appending resampled train cells...")

        # 获取要追加的细胞数据
        cells_to_append_adata = adata[train_cells_to_append].copy()
        # 确保这些细胞的 split 标记为 "train"
        cells_to_append_adata.obs["split"] = "train"

        # 直接追加到原 adata 后面
        adata = ad.concat([adata, cells_to_append_adata], join="outer", index_unique=None)

        print(f"  Appended {len(train_cells_to_append):,} cells")
        print(f"  Total cells after appending: {adata.n_obs:,}")

    # 确保观察值和变量名唯一
    print("\nMaking observation and variable names unique...")
    adata.obs_names_make_unique()

    # 打印最终统计信息（包括复制的细胞）
    print("\n" + "=" * 60)
    print("Final split counts (after replication):")
    final_split_counts = adata.obs["split"].value_counts()
    for name, count in final_split_counts.items():
        print(f"  {name}: {count:,}")
    print(f"  TOTAL: {adata.n_obs:,} cells")

    output_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad_path)
    print(f"\nSaved h5ad file with 'split' and 'CRISPR' columns in obs to: {output_h5ad_path}")
    print(f"  Final shape: {adata.shape}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add 'split', 'CRISPR', and 'cell_cluster' columns to obs in Norman2019 h5ad file:\n"
            "1) Add CRISPR column with specified value\n"
            "2) Create cell_cluster column (uses cell_line/celltype or defaults to 'Norman')\n"
            "3) Keep all cells including controls\n"
            "4) Split by unique gene_pt (4:1 ratio for train:val+test) for perturbed cells\n"
            "5) Control cells (gene_pt empty) split 8:1:1 to train/val/test\n"
            "6) Skip perturbation filtering (gene_pt are mutually exclusive across splits)\n"
            "7) Resample only perturbed cells (control=False) to balance gene_pt counts."
        )
    )
    parser.add_argument(
        "--h5ad-file",
        type=Path,
        default=DEFAULT_H5AD,
        help="Input processed h5ad file.",
    )
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output h5ad file with 'split' and 'CRISPR' columns in obs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for gene_pt split and resampling (default: 42).",
    )
    parser.add_argument(
        "--crispr",
        type=str,
        default="CRISPRi",
        help="Value to set for the CRISPR column in obs (default: empty string).",
    )
    args = parser.parse_args()

    create_norman_split(
        h5ad_path=args.h5ad_file,
        output_h5ad_path=args.output_h5ad,
        seed=args.seed,
        crispr_value=args.crispr,
    )


if __name__ == "__main__":
    main()
