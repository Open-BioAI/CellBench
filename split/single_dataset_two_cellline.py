import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


DEFAULT_H5AD = "./data/all_cell_line_filterdrug.h5ad"
DEFAULT_OUTPUT_DIR = "./split/single_dataset_two_cellline"
DEFAULT_OUTPUT_FILE = "split_single_dataset_two_cellline.csv"
DEFAULT_CELLLINE_COL = "cell_cluster"


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


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test split for single_dataset_two_cellline (K562/RPE1 in Replogle)."
    )
    parser.add_argument(
        "--h5ad-file",
        type=Path,
        default=Path(DEFAULT_H5AD),
        help="Input all_cell_line_filterdrug.h5ad file",
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
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.5,
        help="Val ratio within K562 (default: 0.5, test gets the rest)",
    )
    args = parser.parse_args()

    print(f"Loading h5ad: {args.h5ad_file}")
    adata = sc.read_h5ad(args.h5ad_file)
    obs = adata.obs

    required_cols = ["dataset", "cell_cluster"]
    for col in required_cols:
        if col not in obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    dataset_mask = obs["dataset"].astype(str).str.contains("Replogle", case=False, na=False)
    print(f"Cells with dataset containing 'Replogle': {int(dataset_mask.sum()):,} / {adata.n_obs:,}")

    splits = np.array([""] * adata.n_obs, dtype=object)

    # RPE1 -> train (全部)
    mask_rpe1 = dataset_mask & (obs["cell_cluster"].astype(str) == "RPE1")
    idx_rpe1 = np.where(mask_rpe1.values)[0]
    n_rpe1 = len(idx_rpe1)
    print(f"RPE1 cells: {n_rpe1:,} -> all assigned to train")
    splits[idx_rpe1] = "train"

    # K562 -> val/test (1:1)
    mask_k562 = dataset_mask & (obs["cell_cluster"].astype(str) == "K562")
    idx_k562 = np.where(mask_k562.values)[0]
    n_k562 = len(idx_k562)
    print(f"K562 cells: {n_k562:,}")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx_k562)
    n_val = int(round(n_k562 * args.val_ratio))
    n_val = min(n_val, n_k562)
    n_test = n_k562 - n_val

    val_idx = idx_k562[:n_val]
    test_idx = idx_k562[n_val:]
    splits[val_idx] = "val"
    splits[test_idx] = "test"
    
    print(f"  K562 split: val={n_val:,}, test={n_test:,}")

    # Create split DataFrame
    split_df = pd.DataFrame({"split": splits}, index=obs.index)

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

    # Summary
    split_series = split_df["split"]
    print("\nSplit counts (after perturbation filtering):")
    for name, count in split_series.value_counts(dropna=False).items():
        label = "(empty)" if name == "" else name
        print(f"  {label}: {count:,}")

    # Save
    if args.output_csv is None:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        output_csv = output_dir / DEFAULT_OUTPUT_FILE
    else:
        output_csv = Path(args.output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_series.to_csv(output_csv, header=True)
    print(f"\nSaved split CSV to: {output_csv}")


if __name__ == "__main__":
    main()

