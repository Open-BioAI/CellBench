#!/usr/bin/env python3
"""
Split cells from a specific dataset (Replogle+K562) into train/val/test sets.

- Read all_cell_line_filterdrug.h5ad
- Filter cells where obs.dataset contains both "Replogle" and "K562"
- Randomly split these cells into train/val/test with ratio 8:1:1
- Generate a CSV file with ALL cells, but only filtered cells have split values
- Other cells have empty split values
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


DEFAULT_H5AD = "./data/all_cell_line_filterdrug.h5ad"
DEFAULT_OUTPUT_DIR = "./split/single_dataset_test"
DEFAULT_OUTPUT_FILE = "split_single_dataset.csv"


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


def split_single_dataset(
    h5ad_path: Path,
    output_csv_path: Path | None = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    dataset_col: str = "dataset",
    filter_keywords: list[str] | None = None,
):
    """
    Split cells from a specific dataset into train/val/test sets and save to CSV.
    
    Args:
        h5ad_path: Path to input h5ad file
        output_csv_path: Path to output CSV file
        seed: Random seed for shuffling cells
        train_ratio: Ratio of cells for training (default: 0.8)
        val_ratio: Ratio of cells for validation (default: 0.1)
        test_ratio: Ratio of cells for testing (default: 0.1)
        dataset_col: Name of dataset column in obs
        filter_keywords: List of keywords that must all be present in dataset column
    """
    if filter_keywords is None:
        filter_keywords = ["Replogle", "K562"]
    
    print(f"Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    if dataset_col not in adata.obs.columns:
        raise ValueError(f"Column '{dataset_col}' not found in adata.obs")
    
    obs = adata.obs
    dataset_values = obs[dataset_col].astype(str)
    
    # Filter cells where dataset contains all keywords
    print(f"\nFiltering cells where '{dataset_col}' contains all of: {filter_keywords}")
    mask_filtered = pd.Series(True, index=obs.index)
    for keyword in filter_keywords:
        mask_filtered = mask_filtered & dataset_values.str.contains(keyword, case=False, na=False)
    
    filtered_indices = obs.index[mask_filtered]
    n_filtered = len(filtered_indices)
    n_total = len(obs)
    
    print(f"  Total cells: {n_total:,}")
    print(f"  Filtered cells (containing all keywords): {n_filtered:,}")
    print(f"  Other cells: {n_total - n_filtered:,}")
    
    if n_filtered == 0:
        raise ValueError(f"No cells found matching all keywords: {filter_keywords}")
    
    # Verify ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, "
            f"test={test_ratio}, sum={total_ratio}"
        )
    
    # Randomly shuffle filtered cell indices with fixed seed
    rng = np.random.default_rng(seed)
    shuffled_indices = np.array(filtered_indices)
    rng.shuffle(shuffled_indices)
    
    # Compute number of cells for each split
    n_train = int(round(n_filtered * train_ratio))
    n_val = int(round(n_filtered * val_ratio))
    n_test = n_filtered - n_train - n_val  # Remaining cells go to test
    
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes with n_filtered={n_filtered}: "
            f"train={n_train}, val={n_val}, test={n_test}"
        )
    
    print(
        f"\nCell split counts (from {n_filtered:,} filtered cells):\n"
        f"  train: {n_train}\n  val: {n_val}\n  test: {n_test}"
    )
    
    # Assign splits
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:n_train + n_val]
    test_indices = shuffled_indices[n_train + n_val:]
    
    print(f"\nRandom cell assignment (seed={seed}):")
    print(f"  train: {len(train_indices):,} cells")
    print(f"  val: {len(val_indices):,} cells")
    print(f"  test: {len(test_indices):,} cells")
    
    # Create DataFrame with ALL cells
    # Initialize all splits as empty string
    cell_ids = obs.index.values
    splits = [""] * len(cell_ids)
    
    # Set split values for filtered cells
    split_df = pd.DataFrame({"cell": cell_ids, "split": splits})
    split_df.set_index("cell", inplace=True)
    
    # Assign splits to filtered cells
    split_df.loc[train_indices, "split"] = "train"
    split_df.loc[val_indices, "split"] = "val"
    split_df.loc[test_indices, "split"] = "test"
    
    # Summary of cell counts per split (before perturbation filtering)
    print("\nCell counts per split (before perturbation filtering):")
    split_counts = split_df["split"].value_counts(dropna=False)
    for split_name, count in split_counts.items():
        if split_name == "":
            print(f"  (empty): {count:,} cells")
        else:
            print(f"  {split_name}: {count:,} cells")
    print(f"  TOTAL: {len(split_df):,} cells")
    
    # Step 2: Filter val and test cells by perturbation combinations
    # Remove cells in val/test whose perturbation combination is not in train
    gene_pt_col = "gene_pt"
    drug_pt_col = "drug_pt"
    env_pt_col = "env_pt"
    
    # Check if perturbation columns exist
    pert_cols_exist = all(col in obs.columns for col in [gene_pt_col, drug_pt_col, env_pt_col])
    
    if pert_cols_exist:
        print("\n" + "="*60)
        print("Filtering val and test cells by perturbation combinations...")
        
        pert_keys = build_pert_key(obs, gene_pt_col, drug_pt_col, env_pt_col)
        
        # Get perturbation combinations in train
        mask_train = split_df["split"] == "train"
        train_cells = split_df[mask_train].index
        train_pert_keys = pert_keys.loc[train_cells]
        train_perts = set(train_pert_keys.unique())
        
        print(f"Unique (gene_pt, drug_pt, env_pt) combinations in train: {len(train_perts)}")
        
        # Filter val and test cells
        mask_val = split_df["split"] == "val"
        mask_test = split_df["split"] == "test"
        
        val_cells = split_df[mask_val].index
        test_cells = split_df[mask_test].index
        
        # Check which val/test cells have perturbation combinations not in train
        val_pert_keys = pert_keys.loc[val_cells]
        test_pert_keys = pert_keys.loc[test_cells]
        
        val_keep_mask = val_pert_keys.isin(train_perts)
        test_keep_mask = test_pert_keys.isin(train_perts)
        
        # Set split to empty for cells that should be removed
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
    else:
        print("\n[WARNING] Perturbation columns (gene_pt, drug_pt, env_pt) not found. Skipping perturbation filtering.")
    
    # Summary of cell counts per split (after perturbation filtering)
    print("\nCell counts per split (after perturbation filtering):")
    split_counts_after = split_df["split"].value_counts(dropna=False)
    for split_name, count in split_counts_after.items():
        if split_name == "":
            print(f"  (empty): {count:,} cells")
        else:
            print(f"  {split_name}: {count:,} cells")
    print(f"  TOTAL: {len(split_df):,} cells")
    
    # Save CSV file
    if output_csv_path is None:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        output_csv_path = output_dir / DEFAULT_OUTPUT_FILE
    else:
        output_csv_path = Path(output_csv_path)
    
    print(f"\nSaving split CSV to: {output_csv_path}")
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv_path)
    print("✓ Saved successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Split cells from a specific dataset (Replogle+K562) into train/val/test (8:1:1) and save to CSV"
    )
    parser.add_argument(
        "--h5ad-file",
        type=Path,
        default=Path(DEFAULT_H5AD),
        help="Input h5ad file",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=f"Output CSV file (default: {DEFAULT_OUTPUT_DIR}/{DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling cells (default: 42)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of cells for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of cells for validation (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Ratio of cells for testing (default: 0.1)",
    )
    parser.add_argument(
        "--dataset-col",
        type=str,
        default="dataset",
        help="obs column name for dataset (default: dataset)",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        default=["Replogle", "K562"],
        help="Keywords that must all be present in dataset column (default: Replogle K562)",
    )
    args = parser.parse_args()

    split_single_dataset(
        h5ad_path=args.h5ad_file,
        output_csv_path=args.output_csv,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        dataset_col=args.dataset_col,
        filter_keywords=args.keywords,
    )


if __name__ == "__main__":
    main()

