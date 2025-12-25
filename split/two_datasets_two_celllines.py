import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from single_dataset_two_cellline import build_pert_key


DEFAULT_H5AD = "./data/all_cell_line_filterdrug.h5ad"
DEFAULT_OUTPUT_DIR = "./split/two_datasets_two_celllines"
DEFAULT_OUTPUT_FILE = "split_two_datasets_two_celllines.csv"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create train/val/test split for two_datasets_two_celllines:\n"
            "  - train:   Replogle RPE1\n"
            "  - val/test: XuCao HEK293 (1:1 split)\n"
            "Val/test are further filtered so that only perturbations present in train remain."
        )
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
        help="Val ratio within XuCao HEK293 (default: 0.5, test gets the rest)",
    )
    args = parser.parse_args()

    print(f"Loading h5ad: {args.h5ad_file}")
    adata = sc.read_h5ad(args.h5ad_file)
    obs = adata.obs

    required_cols = ["dataset", "cell_cluster"]
    for col in required_cols:
        if col not in obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    # -------------------------
    # Build initial split: train / val / test
    # -------------------------
    splits = np.array([""] * adata.n_obs, dtype=object)

    # Train = Replogle RPE1
    mask_train = (
        obs["dataset"].astype(str).str.contains("Replogle", case=False, na=False)
        & (obs["cell_cluster"].astype(str) == "RPE1")
    )
    idx_train = np.where(mask_train.values)[0]
    n_train = len(idx_train)
    print(f"Train (Replogle RPE1) cells: {n_train:,}")
    splits[idx_train] = "train"

    # Val/Test = XuCao HEK293 (1:1)
    mask_xucao_hek293 = (
        obs["dataset"].astype(str).str.contains("XuCao", case=False, na=False)
        & (obs["cell_cluster"].astype(str) == "HEK293")
    )
    idx_hek293 = np.where(mask_xucao_hek293.values)[0]
    n_hek293 = len(idx_hek293)
    print(f"XuCao HEK293 cells: {n_hek293:,}")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx_hek293)
    n_val = int(round(n_hek293 * args.val_ratio))
    n_val = min(n_val, n_hek293)
    n_test = n_hek293 - n_val

    val_idx = idx_hek293[:n_val]
    test_idx = idx_hek293[n_val:]
    splits[val_idx] = "val"
    splits[test_idx] = "test"

    print(f"  XuCao HEK293 split: val={n_val:,}, test={n_test:,}")

    # Create split DataFrame
    split_df = pd.DataFrame({"split": splits}, index=obs.index)

    # -------------------------
    # Filter val/test perturbations to those seen in train
    # -------------------------
    gene_pt_col = "gene_pt"
    drug_pt_col = "drug_pt"
    env_pt_col = "env_pt"

    pert_cols_exist = all(col in obs.columns for col in [gene_pt_col, drug_pt_col, env_pt_col])
    if pert_cols_exist:
        print("\n" + "=" * 60)
        print("Filtering val and test cells by perturbation combinations...")

        pert_keys = build_pert_key(obs, gene_pt_col, drug_pt_col, env_pt_col)

        # Train perturbation combinations
        mask_train_split = split_df["split"] == "train"
        train_cells = split_df[mask_train_split].index
        train_pert_keys = pert_keys.loc[train_cells]
        train_perts = set(train_pert_keys.unique())

        print(f"Unique (gene_pt, drug_pt, env_pt) combinations in train: {len(train_perts)}")

        # Val/test cells
        mask_val = split_df["split"] == "val"
        mask_test = split_df["split"] == "test"

        val_cells = split_df[mask_val].index
        test_cells = split_df[mask_test].index

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
    else:
        print(
            "\n[WARNING] Perturbation columns (gene_pt, drug_pt, env_pt) not found. "
            "Skipping perturbation filtering."
        )

    # -------------------------
    # Summary & Save
    # -------------------------
    split_series = split_df["split"]
    print("\nSplit counts (after perturbation filtering):")
    for name, count in split_series.value_counts(dropna=False).items():
        label = "(empty)" if name == "" else name
        print(f"  {label}: {count:,}")

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


