#!/usr/bin/env python3
"""
Manually split clusters in all_cell_line_filterdrug.h5ad into train/val/test sets.

- Read all_cell_line_filterdrug.h5ad
- Use obs.cluster as units for splitting
- Accept user-specified cluster lists for train/val (test gets remaining)
- Then apply the same post-processing as random_split_by_cluster:
  * normalize perturbation strings
  * filter val/test cells whose perturbation combinations are unseen in train
  * sample train/val with replacement, equalizing cell-line probability
  * keep all test cells once; write sample_count to CSV
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from random_split_by_cluster import (
    #normalize_pert_string,
    build_pert_key,
    #compute_cellline_sampling_weights,
    sample_counts_for_split,
    DEFAULT_H5AD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_CELLLINE_COL,
)


def parse_cluster_list(arg: str) -> list[int]:
    """
    Parse a comma-separated list of integers, e.g. "1,2,5".
    Empty string -> empty list.
    """
    if arg is None or arg.strip() == "":
        return []
    parts = [p.strip() for p in arg.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def manual_split_by_cluster(
    h5ad_path: Path,
    output_csv_path: Path | None = None,
    train_clusters: list[int] | None = None,
    val_clusters: list[int] | None = None,
    seed: int = 42,
    cluster_col: str = "cluster",
    cellline_col: str = DEFAULT_CELLLINE_COL,
    gene_pt_col: str = "gene_pt",
    drug_pt_col: str = "drug_pt",
    env_pt_col: str = "env_pt",
):
    """
    Manually split clusters into train/val/test sets and save to CSV.
    test = remaining clusters not in train/val.
    """
    train_clusters = train_clusters or []
    val_clusters = val_clusters or []

    print(f"Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    if cluster_col not in adata.obs.columns:
        raise ValueError(f"Column '{cluster_col}' not found in adata.obs")

    for col in (gene_pt_col, drug_pt_col, env_pt_col, cellline_col):
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    clusters = adata.obs[cluster_col].values
    unique_clusters = np.unique(clusters)
    print(f"\nUnique clusters: {unique_clusters}")

    # Validate overlaps
    overlap = set(train_clusters) & set(val_clusters)
    if overlap:
        raise ValueError(f"train and val clusters overlap: {sorted(overlap)}")

    # Compute test clusters as remaining
    assigned = set(train_clusters) | set(val_clusters)
    test_clusters = [int(c) for c in unique_clusters if int(c) not in assigned]
    print("\nManual cluster assignment:")
    print(f"  train clusters: {sorted(train_clusters)}")
    print(f"  val clusters:   {sorted(val_clusters)}")
    print(f"  test clusters:  {sorted(test_clusters)}")

    cluster_to_split: dict[int, str] = {}
    for cid in train_clusters:
        cluster_to_split[int(cid)] = "train"
    for cid in val_clusters:
        cluster_to_split[int(cid)] = "val"
    for cid in test_clusters:
        cluster_to_split[int(cid)] = "test"

    # Assign split per cell
    splits: list[str] = []
    for cid in clusters:
        if pd.isna(cid):
            splits.append("unknown")
            continue
        try:
            cid_int = int(cid)
        except Exception:
            splits.append("unknown")
            continue
        splits.append(cluster_to_split.get(cid_int, "unknown"))

    cell_ids = adata.obs.index.values
    split_df = pd.DataFrame({"cell": cell_ids, "split": splits}).set_index("cell")
    split_df["sample_count"] = 0

    print("\nCell counts per split (before perturbation filtering):")
    split_counts = split_df["split"].value_counts(dropna=False)
    for name, count in split_counts.items():
        print(f"  {name}: {count:,}")

    # Filter val/test by perturbation combos in train
    print("\n" + "=" * 60)
    print("Filtering val and test cells by perturbation combinations...")
    obs = adata.obs
    pert_keys = build_pert_key(obs, gene_pt_col, drug_pt_col, env_pt_col)

    mask_train = split_df["split"] == "train"
    train_cells = split_df[mask_train].index
    train_perts = set(pert_keys.loc[train_cells].unique())
    print(f"Unique (gene_pt, drug_pt, env_pt) combinations in train: {len(train_perts)}")

    mask_val = split_df["split"] == "val"
    mask_test = split_df["split"] == "test"
    val_cells = split_df[mask_val].index
    test_cells = split_df[mask_test].index

    val_keep = pert_keys.loc[val_cells].isin(train_perts)
    test_keep = pert_keys.loc[test_cells].isin(train_perts)

    val_remove = val_keep[~val_keep].index
    test_remove = test_keep[~test_keep].index
    split_df.loc[val_remove, "split"] = ""
    split_df.loc[test_remove, "split"] = ""

    print("\nFiltering results:")
    print(f"  val removed:  {len(val_remove):,}")
    print(f"  test removed: {len(test_remove):,}")

    print("\nCell counts per split (after perturbation filtering):")
    split_counts_after = split_df["split"].value_counts(dropna=False)
    for name, count in split_counts_after.items():
        label = "(empty)" if name == "" else name
        print(f"  {label}: {count:,}")

    # Sampling: keep all test once; sample train/val with replacement
    n_test_cells = int((split_df["split"] == "test").sum())
    if n_test_cells == 0:
        raise ValueError("No test cells found after filtering; cannot define sampling targets.")

    target_train = 8 * n_test_cells
    target_val = 1 * n_test_cells

    print("\n" + "=" * 60)
    print("Sampling train/val with equal cell-line probability (with replacement)...")
    print(f"  Test cells kept: {n_test_cells}")
    print(f"  Train target:    {target_train} (8x test)")
    print(f"  Val target:      {target_val} (1x test)")

    rng = np.random.default_rng(seed)
    train_counts = sample_counts_for_split(
        adata=adata,
        split_df=split_df,
        split_name="train",
        target_n=target_train,
        cellline_col=cellline_col,
        rng=rng,
    )
    val_counts = sample_counts_for_split(
        adata=adata,
        split_df=split_df,
        split_name="val",
        target_n=target_val,
        cellline_col=cellline_col,
        rng=rng,
    )

    if not train_counts.empty:
        split_df.loc[train_counts.index, "sample_count"] += train_counts.astype(int)
    if not val_counts.empty:
        split_df.loc[val_counts.index, "sample_count"] += val_counts.astype(int)

    test_cells_kept = split_df.index[split_df["split"] == "test"]
    split_df.loc[test_cells_kept, "sample_count"] = 1

    sampled_train_total = int(train_counts.sum()) if not train_counts.empty else 0
    sampled_val_total = int(val_counts.sum()) if not val_counts.empty else 0
    print("\nSampling summary (counts include duplicates):")
    print(f"  Train sampled: {sampled_train_total:,} (target {target_train:,})")
    print(f"  Val sampled:   {sampled_val_total:,} (target {target_val:,})")
    print(f"  Test kept:     {n_test_cells:,}")

    # Save CSV
    if output_csv_path is None:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        output_csv_path = output_dir / DEFAULT_OUTPUT_FILE
    else:
        output_csv_path = Path(output_csv_path)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv_path)
    print(f"\nSaved split CSV to: {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Manual cluster split into train/val/test with fixed cluster lists."
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
        "--train-clusters",
        type=str,
        default="",
        help='Comma-separated cluster IDs for train, e.g. "1,2,3"',
    )
    parser.add_argument(
        "--val-clusters",
        type=str,
        default="",
        help='Comma-separated cluster IDs for val, e.g. "4,5"',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--cellline-col",
        type=str,
        default=DEFAULT_CELLLINE_COL,
        help=f"obs column name for cell line identifier (default: {DEFAULT_CELLLINE_COL})",
    )
    parser.add_argument(
        "--gene-pt-col",
        type=str,
        default="gene_pt",
        help="obs column name for gene perturbation (default: gene_pt)",
    )
    parser.add_argument(
        "--drug-pt-col",
        type=str,
        default="drug_pt",
        help="obs column name for drug perturbation (default: drug_pt)",
    )
    parser.add_argument(
        "--env-pt-col",
        type=str,
        default="env_pt",
        help="obs column name for env perturbation (default: env_pt)",
    )
    args = parser.parse_args()

    manual_split_by_cluster(
        h5ad_path=args.h5ad_file,
        output_csv_path=args.output_csv,
        train_clusters=parse_cluster_list(args.train_clusters),
        val_clusters=parse_cluster_list(args.val_clusters),
        seed=args.seed,
        cellline_col=args.cellline_col,
        gene_pt_col=args.gene_pt_col,
        drug_pt_col=args.drug_pt_col,
        env_pt_col=args.env_pt_col,
    )


if __name__ == "__main__":
    main()

