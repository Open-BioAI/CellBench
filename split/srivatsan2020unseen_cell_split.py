#!/usr/bin/env python3
"""
从 global all_cell_line_filterdrug.h5ad 中为 SrivatsanTrapnell2020_sciplex3 生成 unseen_cell split CSV。

流程：
1. 读取 all_cell_line_filterdrug.h5ad
2. 用 obs.dataset 过滤出 dataset 含有 'SrivatsanTrapnell2020_sciplex3' 的细胞
3. 在该子集上：
   - 所有 cell_cluster == "MCF7" 或 "A549" 的细胞设为 train
   - 所有 cell_cluster == "K562" 的细胞，在内部随机 1:1 划分为 val / test
4. 按 (gene_pt, drug_pt, env_pt) 构建 perturbation 组合，只保留在 train 中出现过的组合在 val/test 中的细胞
5. 导出 CSV，列为 ['cell', 'split']，index 不保存
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def normalize_pert_string(s: str) -> str:
    """将一个扰动字符串标准化：按'+'拆分、strip、排序后再用'+'连接."""
    if s is None or pd.isna(s):
        return ""
    parts = [p.strip() for p in str(s).split("+") if p.strip() != ""]
    if not parts:
        return ""
    parts_sorted = sorted(parts)
    return "+".join(parts_sorted)


def build_pert_key(
    obs: pd.DataFrame, gene_pt_col: str, drug_pt_col: str, env_pt_col: str
) -> pd.Series:
    """Build perturbation key from gene_pt, drug_pt, env_pt columns."""
    g_norm = obs[gene_pt_col].astype(str).apply(normalize_pert_string)
    d_norm = obs[drug_pt_col].astype(str).apply(normalize_pert_string)
    e_norm = obs[env_pt_col].astype(str).apply(normalize_pert_string)
    return g_norm + "|" + d_norm + "|" + e_norm


# 全局 all_cell_line_filterdrug.h5ad
DEFAULT_H5AD = Path(
    "/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/all_cell_line_filterdrug.h5ad"
)
# 输出到 SrivatsanTrapnell2020_sciplex3 目录
DEFAULT_OUTPUT_CSV = Path(
    "/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/split/srivatsantrapnell2020_sciplex3/unseen_cell.csv"
)
CELL_CLUSTER_COL = "cell_cluster"


def create_unseen_cell_split(
    h5ad_path: Path,
    output_csv_path: Path,
    seed: int = 42,
    dataset_filter: str = "SrivatsanTrapnell2020_sciplex3",
    cell_cluster_col: str = CELL_CLUSTER_COL,
    gene_pt_col: str = "gene_pt",
    drug_pt_col: str = "drug_pt",
    env_pt_col: str = "env_pt",
) -> None:
    print(f"Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    obs = adata.obs
    print(f"  Shape: {adata.shape}")

    # 1) 按 dataset 定义 SrivatsanTrapnell2020_sciplex3 子集，但不丢弃其他细胞
    if "dataset" not in obs.columns:
        raise ValueError("obs does not contain 'dataset' column; cannot filter by dataset.")

    ds = obs["dataset"].astype(str)
    mask_ds = ds.str.contains(dataset_filter, case=False, na=False)
    n_before = adata.n_obs
    n_after = int(mask_ds.sum())
    print(
        f"\nFiltering by dataset containing '{dataset_filter}' (for assigning splits only): "
        f"{n_before:,} -> {n_after:,} cells in target dataset"
    )
    if n_after == 0:
        raise ValueError(
            f"No cells found with dataset containing '{dataset_filter}'. "
            f"Please check obs['dataset'] values."
        )

    # 2) 检查必要列（在全体 obs 上）
    required_cols = [cell_cluster_col, gene_pt_col, drug_pt_col, env_pt_col]
    for col in required_cols:
        if col not in obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    clusters = obs[cell_cluster_col].astype(str)
    print("\ncell_cluster value counts (all cells):")
    print(clusters.value_counts())

    # 初始化所有细胞的 split 为空字符串（与 single_dataset_test 格式一致）
    splits = np.array([""] * adata.n_obs, dtype=object)

    # 仅在目标 dataset 子集内赋值 split
    _ds_idx = np.where(mask_ds.values)[0]

    # 3) MCF7 & A549 -> train（仅限目标 dataset）
    mask_mcf7 = mask_ds & (clusters == "MCF7")
    mask_a549 = mask_ds & (clusters == "A549")
    idx_mcf7 = np.where(mask_mcf7.values)[0]
    idx_a549 = np.where(mask_a549.values)[0]
    splits[idx_mcf7] = "train"
    splits[idx_a549] = "train"
    print(f"\nMCF7 cells: {len(idx_mcf7):,} -> train")
    print(f"A549 cells: {len(idx_a549):,} -> train")

    # 4) K562 -> 在 K562 内部随机 1:1 分成 val/test（仅限目标 dataset）
    mask_k562 = mask_ds & (clusters == "K562")
    idx_k562 = np.where(mask_k562.values)[0]
    n_k562 = len(idx_k562)
    print(f"\nK562 cells: {n_k562:,}")

    if n_k562 > 0:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx_k562)
        n_val = n_k562 // 2
        val_idx = idx_k562[:n_val]
        test_idx = idx_k562[n_val:]
        splits[val_idx] = "val"
        splits[test_idx] = "test"
        print(f"K562 split -> val: {len(val_idx):,}, test: {len(test_idx):,}")
    else:
        print("[WARNING] No K562 cells found; val/test will be empty.")

    # 构建 split DataFrame
    split_df = pd.DataFrame({"split": splits}, index=obs.index)

    print("\nSplit counts before perturbation filtering:")
    for name, count in split_df["split"].value_counts(dropna=False).items():
        label = "(empty)" if name == "" else name
        print(f"  {label}: {count:,}")
    print(f"  TOTAL: {len(split_df):,} cells")

    # 5) perturbation 过滤：只保留 train 中出现过的 (gene_pt, drug_pt, env_pt)
    print("\n" + "=" * 60)
    print("Filtering val and test cells by perturbation combinations...")

    pert_keys = build_pert_key(obs, gene_pt_col, drug_pt_col, env_pt_col)

    # train perturbations
    mask_train = split_df["split"] == "train"
    train_cells = split_df.index[mask_train]
    train_pert_keys = pert_keys.loc[train_cells]
    train_perts = set(train_pert_keys.unique())

    print(f"Unique (gene_pt, drug_pt, env_pt) combinations in train: {len(train_perts)}")

    # 过滤 val/test
    mask_val = split_df["split"] == "val"
    mask_test = split_df["split"] == "test"

    val_cells = split_df.index[mask_val]
    test_cells = split_df.index[mask_test]

    val_pert_keys = pert_keys.loc[val_cells]
    test_pert_keys = pert_keys.loc[test_cells]

    val_keep_mask = val_pert_keys.isin(train_perts)
    test_keep_mask = test_pert_keys.isin(train_perts)

    n_val_before = int(mask_val.sum())
    n_test_before = int(mask_test.sum())

    val_cells_to_remove = val_keep_mask[~val_keep_mask].index
    test_cells_to_remove = test_keep_mask[~test_keep_mask].index

    split_df.loc[val_cells_to_remove, "split"] = ""
    split_df.loc[test_cells_to_remove, "split"] = ""

    n_val_removed = len(val_cells_to_remove)
    n_test_removed = len(test_cells_to_remove)
    n_val_after = n_val_before - n_val_removed
    n_test_after = n_test_before - n_test_removed

    print("\nFiltering results:")
    print(f"  val:  {n_val_before:,} -> {n_val_after:,} (removed {n_val_removed:,} cells)")
    print(f"  test: {n_test_before:,} -> {n_test_after:,} (removed {n_test_removed:,} cells)")
    print(f"  Total removed: {n_val_removed + n_test_removed:,} cells (split set to empty)")

    print("\nFinal split counts (after perturbation filtering):")
    split_counts_after = split_df["split"].value_counts(dropna=False)
    for name, count in split_counts_after.items():
        label = "(empty)" if name == "" else name
        print(f"  {label}: {count:,}")
    print(f"  TOTAL: {len(split_df):,} cells")

    # 6) 保存 CSV，第一列列名为 'cell'
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = split_df["split"].rename("split").reset_index()
    out_df.columns = ["cell", "split"]
    out_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved split CSV to: {output_csv_path} (columns: cell, split)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create unseen_cell.csv for SrivatsanTrapnell2020_sciplex3 from all_cell_line_filterdrug.h5ad:\n"
            "1) Filter obs.dataset by substring (default: SrivatsanTrapnell2020_sciplex3)\n"
            "2) MCF7/A549 -> train, K562 -> val/test 1:1 within K562\n"
            "3) Remove val/test cells whose (gene_pt, drug_pt, env_pt) combination is unseen in train."
        )
    )
    parser.add_argument(
        "--h5ad-file",
        type=Path,
        default=DEFAULT_H5AD,
        help="Input all_cell_line_filterdrug.h5ad file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV file (default: SrivatsanTrapnell2020_sciplex3/unseen_cell.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for K562 val/test split (default: 42).",
    )
    parser.add_argument(
        "--dataset-filter",
        type=str,
        default="SrivatsanTrapnell2020_sciplex3",
        help="Substring in obs.dataset to select cells (default: SrivatsanTrapnell2020_sciplex3).",
    )
    args = parser.parse_args()

    create_unseen_cell_split(
        h5ad_path=args.h5ad_file,
        output_csv_path=args.output_csv,
        seed=args.seed,
        dataset_filter=args.dataset_filter,
    )


if __name__ == "__main__":
    main()


