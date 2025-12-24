#!/usr/bin/env python3
"""
从 global all_cell_line_filterdrug.h5ad 中为 TianKampamann2019_iPSC 生成 unseen_cell split CSV。

规则：
1. 读取 all_cell_line_filterdrug.h5ad
2. 使用 obs.dataset 中包含 "TianKampamann2019_iPSC" 的细胞作为目标子集
3. 在该子集上，按 (gene_pt, drug_pt, env_pt) 构建唯一 perturbation 组合
4. 随机打乱所有 unique perturbation（seed=42），
   - 约 80% 的 perturbation 标记为 train
   - 剩余约 20% 按 1:1 划分为 val / test
5. 所有属于某个 perturbation 的细胞，split 一致（train/val/test）
6. 其他不在目标 dataset 内的细胞，split 为空字符串（兼容现有代码）
7. 导出 CSV，列为 ['cell', 'split']，保存到
   split/TianKampamann2019_iPSC/unseen_cell.csv
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


DEFAULT_H5AD = Path(
    "/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/all_cell_line_filterdrug.h5ad"
)

DEFAULT_OUTPUT_CSV = Path(
    "/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/split/tiankampamann2019_ipsc/unseen_cell.csv"
)


def create_tian_kampmann_unseen_cell_split(
    h5ad_path: Path,
    output_csv_path: Path,
    seed: int = 42,
    dataset_filter: str = "TianKampmann2019_iPSC",
    gene_pt_col: str = "gene_pt",
    drug_pt_col: str = "drug_pt",
    env_pt_col: str = "env_pt",
) -> None:
    print(f"Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    obs = adata.obs
    print(f"  Shape: {adata.shape}")

    # 1) 按 dataset 过滤目标子集
    if "dataset" not in obs.columns:
        raise ValueError("obs does not contain 'dataset' column; cannot filter by dataset.")

    ds = obs["dataset"].astype(str)
    mask_ds = ds.str.contains(dataset_filter, case=False, na=False)
    n_before = adata.n_obs
    n_after = int(mask_ds.sum())
    print(
        f"\nFiltering by dataset containing '{dataset_filter}': "
        f"{n_before:,} -> {n_after:,} cells in target dataset"
    )
    if n_after == 0:
        raise ValueError(
            f"No cells found with dataset containing '{dataset_filter}'. "
            f"Please check obs['dataset'] values."
        )

    # 2) 构建 perturbation key（在全体 obs 上，便于保持与其他脚本一致）
    required_cols = [gene_pt_col, drug_pt_col, env_pt_col]
    for col in required_cols:
        if col not in obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    pert_keys = build_pert_key(obs, gene_pt_col, drug_pt_col, env_pt_col)

    # 目标子集中的 cell index
    target_idx = np.where(mask_ds.values)[0]
    target_cells = obs.index[target_idx]
    target_perts = pert_keys.loc[target_cells]

    # 3) 在目标 dataset 内的 unique perturbation 上做 80/10/10 划分
    unique_perts = np.array(sorted(target_perts.unique()))
    n_perts = len(unique_perts)
    print(f"\nFound {n_perts:,} unique perturbations in dataset '{dataset_filter}'.")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_perts)

    n_train = int(round(0.8 * n_perts))
    n_remaining = n_perts - n_train
    n_val = n_remaining // 2
    n_test = n_remaining - n_val  # 保证总数一致

    train_perts = set(unique_perts[:n_train])
    val_perts = set(unique_perts[n_train : n_train + n_val])
    test_perts = set(unique_perts[n_train + n_val :])

    print(f"Perturbation split (by unique (gene_pt, drug_pt, env_pt)):")
    print(f"  train perts: {len(train_perts):,}")
    print(f"  val perts:   {len(val_perts):,}")
    print(f"  test perts:  {len(test_perts):,}")

    # 4) 初始化所有细胞 split 为空字符串
    splits = np.array([""] * adata.n_obs, dtype=object)

    # 只对目标 dataset 内的细胞赋值 split；其他 dataset 留空
    target_perts_array = target_perts.values
    for i, cell in enumerate(target_cells):
        pk = target_perts_array[i]
        if pk in train_perts:
            splits[adata.obs.index.get_loc(cell)] = "train"
        elif pk in val_perts:
            splits[adata.obs.index.get_loc(cell)] = "val"
        elif pk in test_perts:
            splits[adata.obs.index.get_loc(cell)] = "test"
        # 否则保持为空（理论上不会发生）

    split_df = pd.DataFrame({"split": splits}, index=obs.index)

    print("\nFinal split counts (all cells, including empty):")
    for name, count in split_df["split"].value_counts(dropna=False).items():
        label = "(empty)" if name == "" else name
        print(f"  {label}: {count:,}")
    print(f"  TOTAL: {len(split_df):,} cells")

    # 5) 保存 CSV，列名 ['cell', 'split']
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = split_df["split"].rename("split").reset_index()
    out_df.columns = ["cell", "split"]
    out_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved split CSV to: {output_csv_path} (columns: cell, split)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create unseen_cell.csv for TianKampamann2019_iPSC from all_cell_line_filterdrug.h5ad:\n"
            "1) Filter obs.dataset by substring (default: TianKampamann2019_iPSC)\n"
            "2) Build unique perturbations by (gene_pt, drug_pt, env_pt)\n"
            "3) Randomly assign ~80% perts to train, remaining ~20% split 1:1 to val/test (seed=42)."
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
        help="Output CSV file (default: TianKampamann2019_iPSC/unseen_cell.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for perturbation-level split (default: 42).",
    )
    parser.add_argument(
        "--dataset-filter",
        type=str,
        default="TianKampamann2019_iPSC",
        help="Substring in obs.dataset to select cells (default: TianKampamann2019_iPSC).",
    )
    args = parser.parse_args()

    create_tian_kampmann_unseen_cell_split(
        h5ad_path=args.h5ad_file,
        output_csv_path=args.output_csv,
        seed=args.seed,
        dataset_filter=args.dataset_filter,
    )


if __name__ == "__main__":
    main()


