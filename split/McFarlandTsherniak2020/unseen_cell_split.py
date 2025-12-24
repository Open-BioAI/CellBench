#!/usr/bin/env python3
"""
Create train/val/test split for McFarlandTsherniak2020:

- Input: a h5ad file with obs containing:
    - 'cell_cluster'  : cell line / cluster name，用它来做 8:1:1 的划分单元
    - 'gene_pt','drug_pt','env_pt' : 组合定义一个 perturbation
- Split:
    1. 按 cell_cluster 随机打乱，按 8:1:1 比例把 cluster 分成 train/val/test
    2. 生成每个 cell 的 split（train/val/test）
    3. 删除 val/test 中那些 (gene_pt, drug_pt, env_pt) 组合从未在 train 出现过的细胞
- Output:
    - CSV: unseen_cell.csv, 两列：cell, split（index 为 cell id）
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def normalize_pert_string(s: str) -> str:
    """将一个扰动字符串标准化：按'+'拆分、strip、排序后再用'+'连接.
    例如 'geneA+geneB' 和 'geneB + geneA' 都会变成 'geneA+geneB'。
    空字符串或全是空白时返回空字符串。
    """
    if s is None or pd.isna(s):
        return ""
    parts = [p.strip() for p in str(s).split("+") if p.strip() != ""]
    if not parts:
        return ""
    parts_sorted = sorted(parts)
    return "+".join(parts_sorted)


def build_pert_key(obs: pd.DataFrame, gene_pt_col: str, drug_pt_col: str, env_pt_col: str) -> pd.Series:
    """Build perturbation key from gene_pt, drug_pt, env_pt columns."""
    # 先标准化每一列内部的多重扰动（顺序无关）
    # 注意原列可能是 categorical，所以先转成字符串再处理，避免 Series + str 报错
    g_norm = obs[gene_pt_col].astype(str).apply(normalize_pert_string)
    d_norm = obs[drug_pt_col].astype(str).apply(normalize_pert_string)
    e_norm = obs[env_pt_col].astype(str).apply(normalize_pert_string)
    # 再把三列组合成一个 key（都已是普通字符串 Series）
    return g_norm + "|" + d_norm + "|" + e_norm


DEFAULT_H5AD = Path(
    "/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess/total/all_cell_line_filterdrug.h5ad"
)
DEFAULT_OUTPUT_CSV = Path("unseen_cell.csv")
DEFAULT_CLUSTER_COL = "cell_cluster"


def random_split_by_cell_cluster(
    h5ad_path: Path,
    output_csv_path: Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    cell_cluster_col: str = DEFAULT_CLUSTER_COL,
    gene_pt_col: str = "gene_pt",
    drug_pt_col: str = "drug_pt",
    env_pt_col: str = "env_pt",
) -> None:
    print(f"Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    obs = adata.obs
    print(f"  Shape: {adata.shape}")

    # 检查必要列
    required_cols = [cell_cluster_col, gene_pt_col, drug_pt_col, env_pt_col]
    for col in required_cols:
        if col not in obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    clusters = obs[cell_cluster_col].astype(str).values
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    print(f"\nFound {n_clusters} unique '{cell_cluster_col}' values:")
    print(f"  {unique_clusters}")

    # 检查比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, "
            f"test={test_ratio}, sum={total_ratio}"
        )

    # 计算每个 split 的 cluster 数
    n_train = int(round(n_clusters * train_ratio))
    n_val = int(round(n_clusters * val_ratio))
    n_test = n_clusters - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes with n_clusters={n_clusters}: "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

    print(
        f"\nCluster split counts (total {n_clusters} clusters):\n"
        f"  train: {n_train}\n  val: {n_val}\n  test: {n_test}"
    )

    # 随机打乱 cluster，划分 8:1:1
    rng = np.random.default_rng(seed)
    shuffled_clusters = unique_clusters.copy()
    rng.shuffle(shuffled_clusters)

    train_clusters = shuffled_clusters[:n_train]
    val_clusters = shuffled_clusters[n_train : n_train + n_val]
    test_clusters = shuffled_clusters[n_train + n_val :]

    print("\nRandom cluster assignment (by cell_cluster, seed={}):".format(seed))
    print(f"  train clusters: {sorted(train_clusters)}")
    print(f"  val clusters:   {sorted(val_clusters)}")
    print(f"  test clusters:  {sorted(test_clusters)}")

    # 为每个 cell 赋 split
    cluster_to_split: dict[str, str] = {}
    for c in train_clusters:
        cluster_to_split[str(c)] = "train"
    for c in val_clusters:
        cluster_to_split[str(c)] = "val"
    for c in test_clusters:
        cluster_to_split[str(c)] = "test"

    splits: list[str] = []
    unknown_clusters: set[str] = set()
    for cl in clusters:
        cl_str = str(cl)
        split = cluster_to_split.get(cl_str, "unknown")
        if split == "unknown":
            unknown_clusters.add(cl_str)
        splits.append(split)

    # 构建 split DataFrame
    split_df = pd.DataFrame({"split": splits}, index=obs.index)

    print("\nCell counts per split (before perturbation filtering):")
    split_counts = split_df["split"].value_counts(dropna=False)
    for name, count in split_counts.items():
        label = "(unknown)" if name == "unknown" else name
        print(f"  {label}: {count:,}")
    print(f"  TOTAL: {len(split_df):,} cells")

    if unknown_clusters:
        print("\n[WARNING] Some cell_cluster values were not assigned to any split:")
        print(f"  {sorted(unknown_clusters)}")

    # 2) 按 perturbation 过滤：只保留在 train 中出现过的 (gene_pt, drug_pt, env_pt)
    print("\n" + "=" * 60)
    print("Filtering val and test cells by perturbation combinations...")

    pert_keys = build_pert_key(obs, gene_pt_col, drug_pt_col, env_pt_col)

    # train 中的所有 perturbation 组合
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

    # 保存 CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    split_df["split"].to_csv(output_csv_path, header=True)
    print(f"\nSaved split CSV to: {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test split for McFarlandTsherniak2020 using cell_cluster (8:1:1) and filter unseen perturbations."
    )
    parser.add_argument(
        "--h5ad-file",
        type=Path,
        default=DEFAULT_H5AD,
        help="Input h5ad file with cell_cluster, gene_pt, drug_pt, env_pt.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV file (default: unseen_cell.csv in current directory).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio over cell_cluster (default: 0.8).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Val ratio over cell_cluster (default: 0.1).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test ratio over cell_cluster (default: 0.1).",
    )
    args = parser.parse_args()

    random_split_by_cell_cluster(
        h5ad_path=args.h5ad_file,
        output_csv_path=args.output_csv,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()


