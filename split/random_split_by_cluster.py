#!/usr/bin/env python3
"""
Randomly split clusters in all_cell_line_filterdrug.h5ad into train/val/test sets.

- Read all_cell_line_filterdrug.h5ad
- Use obs.cluster (integer cluster IDs) as units for splitting
- With seed=42, randomly assign clusters to train/val/test with ratio 8:1:1
- Generate a CSV file with cell IDs and their split assignments
- Do NOT modify the original h5ad file
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


DEFAULT_H5AD = "./data/all_cell_line_filterdrug.h5ad"
DEFAULT_OUTPUT_DIR = "./split/unseen_cell"
DEFAULT_OUTPUT_FILE = "split_unseen_cell.csv"
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


def compute_cellline_sampling_weights(obs_split: pd.DataFrame, cellline_col: str) -> np.ndarray:
    """Compute sampling weights so each cell line has equal total probability."""
    celllines = obs_split[cellline_col].values
    # Count cells per cell line (ignore NaN)
    valid_mask = ~pd.isna(celllines)
    cellline_counts = pd.Series(celllines[valid_mask]).value_counts().to_dict()

    n_celllines = len(cellline_counts)
    if n_celllines == 0:
        raise ValueError("No valid cell lines found when computing sampling weights.")

    weights = np.zeros(len(obs_split), dtype=float)
    for i, cl in enumerate(celllines):
        if pd.isna(cl):
            continue
        n_cells_in_line = cellline_counts.get(cl, 0)
        if n_cells_in_line <= 0:
            continue
        weights[i] = 1.0 / (n_celllines * n_cells_in_line)

    total_w = weights.sum()
    if total_w <= 0:
        raise ValueError("All sampling weights are zero; check cell line column.")
    weights /= total_w
    return weights


def sample_counts_for_split(
    adata,
    split_df: pd.DataFrame,
    split_name: str,
    target_n: int,
    cellline_col: str,
    rng: np.random.Generator,
) -> pd.Series:
    """Sample cells WITH replacement for a split; return value_counts Series."""
    mask_split = split_df["split"] == split_name
    cell_ids = split_df[mask_split].index.values
    if len(cell_ids) == 0 or target_n <= 0:
        print(f"[Split={split_name}] No cells to sample (target {target_n}), skipping.")
        return pd.Series(dtype=int)

    obs_split = adata.obs.loc[cell_ids]
    if cellline_col not in obs_split.columns:
        raise ValueError(f"Column '{cellline_col}' not found in adata.obs; cannot sample by cell line.")

    weights = compute_cellline_sampling_weights(obs_split, cellline_col=cellline_col)
    chosen_local = rng.choice(len(cell_ids), size=target_n, replace=True, p=weights)
    chosen_cells = cell_ids[chosen_local]
    counts = pd.Series(chosen_cells).value_counts()

    n_unique = len(counts)
    n_duplicates = int(target_n - n_unique)
    print(
        f"[Split={split_name}] Target {target_n}, sampled {target_n} (with replacement); "
        f"{n_unique} unique cells, {n_duplicates} duplicates"
    )
    return counts


def random_split_by_cluster(
    h5ad_path: Path,
    output_csv_path: Path | None = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    cluster_col: str = "cluster",
    cellline_col: str = DEFAULT_CELLLINE_COL,
    gene_pt_col: str = "gene_pt",
    drug_pt_col: str = "drug_pt",
    env_pt_col: str = "env_pt",
):
    """
    Split clusters into train/val/test sets and save to CSV.
    
    Args:
        h5ad_path: Path to input h5ad file
        output_csv_path: Path to output CSV file (default: split_unseen_cell.csv in split directory)
        seed: Random seed for shuffling clusters
        train_ratio: Ratio of clusters for training (default: 0.8)
        val_ratio: Ratio of clusters for validation (default: 0.1)
        test_ratio: Ratio of clusters for testing (default: 0.1)
        cluster_col: Name of cluster column in obs
        cellline_col: Name of cell line column used for balanced sampling
        gene_pt_col: Name of gene perturbation column in obs
        drug_pt_col: Name of drug perturbation column in obs
        env_pt_col: Name of env perturbation column in obs
    """
    print(f"Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    if cluster_col not in adata.obs.columns:
        raise ValueError(f"Column '{cluster_col}' not found in adata.obs")
    
    # Check perturbation columns
    for col in (gene_pt_col, drug_pt_col, env_pt_col):
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs; cannot filter by perturbation.")
    if cellline_col not in adata.obs.columns:
        raise ValueError(f"Column '{cellline_col}' not found in adata.obs; cannot sample by cell line.")

    clusters = adata.obs[cluster_col].values
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    print(f"\nFound {n_clusters} unique clusters in obs['{cluster_col}']:")
    print(f"  {unique_clusters}")

    # Verify ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, "
            f"test={test_ratio}, sum={total_ratio}"
        )

    # Compute number of clusters for each split
    n_train = int(round(n_clusters * train_ratio))
    n_val = int(round(n_clusters * val_ratio))
    n_test = n_clusters - n_train - n_val  # Remaining clusters go to test

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes with n_clusters={n_clusters}: "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

    print(
        f"\nCluster split counts (total {n_clusters} clusters):\n"
        f"  train: {n_train}\n  val: {n_val}\n  test: {n_test}"
    )

    # Randomly shuffle clusters with fixed seed
    rng = np.random.default_rng(seed)
    shuffled_clusters = unique_clusters.copy()
    rng.shuffle(shuffled_clusters)

    train_clusters = shuffled_clusters[:n_train]
    val_clusters = shuffled_clusters[n_train:n_train + n_val]
    test_clusters = shuffled_clusters[n_train + n_val:]

    print("\nRandom cluster assignment (seed={}):".format(seed))
    print(f"  train clusters: {sorted(train_clusters)}")
    print(f"  val clusters: {sorted(val_clusters)}")
    print(f"  test clusters: {sorted(test_clusters)}")

    # Build mapping from cluster -> split
    cluster_to_split: dict[int, str] = {}
    for cid in train_clusters:
        cluster_to_split[int(cid)] = "train"
    for cid in val_clusters:
        cluster_to_split[int(cid)] = "val"
    for cid in test_clusters:
        cluster_to_split[int(cid)] = "test"

    # Assign split to each cell
    splits: list[str] = []
    unknown_clusters: set[int] = set()

    for cid in clusters:
        # cid could be np.int64 / float; convert to int safely if possible
        if pd.isna(cid):
            splits.append("unknown")
            continue
        try:
            cid_int = int(cid)
        except Exception:
            splits.append("unknown")
            continue

        split = cluster_to_split.get(cid_int, "unknown")
        if split == "unknown":
            unknown_clusters.add(cid_int)
        splits.append(split)

    # Create DataFrame with cell IDs and splits
    # Use index as cell identifier
    cell_ids = adata.obs.index.values
    
    split_df = pd.DataFrame({
        "cell": cell_ids,
        "split": splits
    })
    split_df.set_index("cell", inplace=True)
    split_df["sample_count"] = 0  # will be filled after sampling

    # Summary of cell counts per split (before perturbation filtering)
    print("\nCell counts per split (before perturbation filtering):")
    split_counts = pd.Series(splits).value_counts(dropna=False)
    for split_name, count in split_counts.items():
        print(f"  {split_name}: {count:,} cells")
    print(f"  TOTAL: {len(splits):,} cells")

    if unknown_clusters:
        print("\n[WARNING] Some clusters were not assigned to any split (marked as 'unknown'):")
        print(f"  {sorted(unknown_clusters)}")

    # Step 2: Filter val and test cells by perturbation combinations
    # Remove cells in val/test whose perturbation combination is not in train
    print("\n" + "="*60)
    print("Filtering val and test cells by perturbation combinations...")
    
    obs = adata.obs
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
    # pert_keys index should match adata.obs.index, which matches split_df.index
    val_pert_keys = pert_keys.loc[val_cells]
    test_pert_keys = pert_keys.loc[test_cells]
    
    val_keep_mask = val_pert_keys.isin(train_perts)
    test_keep_mask = test_pert_keys.isin(train_perts)
    
    # Set split to empty for cells that should be removed
    n_val_before = int(mask_val.sum())
    n_test_before = int(mask_test.sum())
    
    # val_keep_mask and test_keep_mask are Series with index matching val_cells/test_cells
    # Use boolean indexing: cells to remove are those where keep_mask is False
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
    
    # Summary of cell counts per split (after perturbation filtering)
    print("\nCell counts per split (after perturbation filtering):")
    split_counts_after = split_df["split"].value_counts(dropna=False)
    for split_name, count in split_counts_after.items():
        if split_name == "":
            print(f"  (empty): {count:,} cells")
        else:
            print(f"  {split_name}: {count:,} cells")
    print(f"  TOTAL: {len(split_df):,} cells")

    # Cell_cluster counts in train/val before sampling (for sanity check)
    print("\nCell_cluster counts per split (before sampling):")
    for split_name in ("train", "val"):
        mask = split_df["split"] == split_name
        if not mask.any():
            print(f"  {split_name}: (no cells)")
            continue
        counts = (
            adata.obs.loc[mask.index[mask], cellline_col]
            .astype(object)
            .fillna("")
            .astype(str)
            .value_counts()
            .sort_index()
        )
        print(f"  {split_name}:")
        for cl, c in counts.items():
            print(f"    {cl}: {c:,}")
        print(f"    Total {split_name}: {int(counts.sum()):,}")

    # Sampling step: keep all test cells once; sample train/val with replacement
    n_test_cells = int((split_df["split"] == "test").sum())
    if n_test_cells == 0:
        raise ValueError("No test cells found after filtering; cannot define sampling targets.")

    target_train = 8 * n_test_cells
    target_val = 1 * n_test_cells

    print("\n" + "="*60)
    print("Sampling train/val with equal cell-line probability (with replacement)...")
    print(
        f"  Test cells kept: {n_test_cells}\n"
        f"  Train target:    {target_train} (8x test)\n"
        f"  Val target:      {target_val} (1x test)"
    )

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

    # Apply counts to split_df
    if not train_counts.empty:
        split_df.loc[train_counts.index, "sample_count"] += train_counts.astype(int)
    if not val_counts.empty:
        split_df.loc[val_counts.index, "sample_count"] += val_counts.astype(int)

    # All test cells kept exactly once
    test_cells = split_df.index[split_df["split"] == "test"]
    split_df.loc[test_cells, "sample_count"] = 1

    sampled_train_total = int(train_counts.sum()) if not train_counts.empty else 0
    sampled_val_total = int(val_counts.sum()) if not val_counts.empty else 0

    print("\nSampling summary (counts include duplicates):")
    print(f"  Train sampled: {sampled_train_total:,} cells (target {target_train:,})")
    print(f"  Val sampled:   {sampled_val_total:,} cells (target {target_val:,})")
    print(f"  Test kept:     {n_test_cells:,} cells")
    if n_test_cells > 0:
        ratio_train_test = sampled_train_total / n_test_cells
        ratio_val_test = sampled_val_total / n_test_cells
        print(
            f"  Ratios (sampled/test): train={ratio_train_test:.2f}, "
            f"val={ratio_val_test:.2f}, test=1.00"
        )

    # Summaries: sampled counts per cell line (including duplicates)
    # Cast to plain object, fill NaN with "" then stringify (no special token)
    cellline_series = (
        adata.obs[cellline_col]
        .astype(object)
        .reindex(split_df.index)
        .fillna("")
        .astype(str)
    )
    for split_name, counts in (("train", train_counts), ("val", val_counts)):
        print(f"\n{split_name} sampled cell_line counts (including duplicates):")
        if counts.empty:
            print("  (none sampled)")
            continue
        # Map cell IDs to cell line, then aggregate sampled counts
        split_mask = split_df["split"] == split_name
        sampled_counts = (
            split_df.loc[split_mask, "sample_count"]
            .groupby(cellline_series.loc[split_mask])
            .sum()
            .sort_values(ascending=False)
        )
        for cl, c in sampled_counts.items():
            print(f"  {cl}: {int(c):,}")
        print(f"  Total sampled {split_name}: {int(sampled_counts.sum()):,}")

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
        description="Randomly split clusters into train/val/test (8:1:1) and save to CSV"
    )
    parser.add_argument(
        "--h5ad-file",
        type=Path,
        default=Path(DEFAULT_H5AD),
        help="Input all_cell_line.h5ad file",
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
        help="Random seed for shuffling clusters (default: 42)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of clusters for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of clusters for validation (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Ratio of clusters for testing (default: 0.1)",
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

    random_split_by_cluster(
        h5ad_path=args.h5ad_file,
        output_csv_path=args.output_csv,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        cellline_col=args.cellline_col,
        gene_pt_col=args.gene_pt_col,
        drug_pt_col=args.drug_pt_col,
        env_pt_col=args.env_pt_col,
    )


if __name__ == "__main__":
    main()
