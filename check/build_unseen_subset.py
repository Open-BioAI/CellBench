#!/usr/bin/env python3
"""
Build an h5ad that matches split/unseen_cell/split_unseen_cell.csv with sampling.

- Read the base h5ad (all_cell_line_filterdrug.h5ad by default)
- Read split_unseen_cell.csv (must contain a 'split' column; optional 'sample_count')
- Duplicate rows according to sample_count (default 1), keep expression/metadata the same
- Preserve obs.split from CSV
- Make obs names unique at the end
- Write an output h5ad
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


DEFAULT_H5AD = "./data/all_cell_line_filterdrug.h5ad"
DEFAULT_SPLIT_CSV = "./split/unseen_cell/split_unseen_cell.csv"
DEFAULT_OUTPUT = "./check/all_cell_line_filterdrug_unseen_subset.h5ad"


def main():
    parser = argparse.ArgumentParser(
        description="Build h5ad for unseen_cell split with sampling according to sample_count."
    )
    parser.add_argument(
        "--h5ad-file",
        type=Path,
        default=Path(DEFAULT_H5AD),
        help="Input h5ad (all cells)",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path(DEFAULT_SPLIT_CSV),
        help="Split CSV with columns 'split' and optional 'sample_count'",
    )
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Output h5ad with duplicated rows per sample_count",
    )
    args = parser.parse_args()

    print(f"Loading h5ad: {args.h5ad_file}")
    adata = sc.read_h5ad(args.h5ad_file)
    print(f"  Shape: {adata.shape}")

    print(f"Loading split CSV: {args.split_csv}")
    split_df = pd.read_csv(args.split_csv, index_col=0)
    if "split" not in split_df.columns:
        raise ValueError("split CSV must contain a 'split' column.")

    if "sample_count" in split_df.columns:
        counts = pd.to_numeric(split_df["sample_count"], errors="coerce").fillna(1).astype(int)
    else:
        counts = pd.Series(1, index=split_df.index, dtype=int)

    counts[counts < 0] = 0
    split_df["sample_count"] = counts

    # Align to adata
    in_adata = adata.obs.index.isin(split_df.index)
    if not in_adata.any():
        raise ValueError("No cells in split CSV match adata.obs.index")

    adata = adata[in_adata].copy()
    split_df = split_df.loc[adata.obs.index]

    repeat_idx = np.repeat(np.arange(adata.n_obs), split_df["sample_count"].values)
    if len(repeat_idx) == 0:
        raise ValueError("No rows after applying sample_count.")

    # Handle duplicates robustly (works with sparse)
    unique_indices, inverse = np.unique(repeat_idx, return_inverse=True)
    adata_unique = adata[unique_indices].copy()
    mapped_indices = inverse.astype(np.int32)
    adata_sub = adata_unique[mapped_indices].copy()

    # Attach split column to obs (aligned with new rows)
    # Build repeated split values to match repeat_idx order
    split_repeated = np.repeat(split_df["split"].values, split_df["sample_count"].values)
    adata_sub.obs["split"] = pd.Categorical(split_repeated, categories=["train", "val", "test", ""], ordered=False)

    # Make obs names unique
    adata_sub.obs_names_make_unique()

    print(f"Final shape: {adata_sub.shape}")
    print("Split counts in output:")
    for name, count in adata_sub.obs["split"].value_counts(dropna=False).items():
        print(f"  {name}: {count:,}")

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {args.output_h5ad}")
    adata_sub.write_h5ad(args.output_h5ad)
    print("Done.")


if __name__ == "__main__":
    main()

