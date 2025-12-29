#!/usr/bin/env python3
"""
Run preprocess() on a configurable list of datasets.
Modified version: Only selects Highly Variable Genes (HVG), not DEGs.
"""

from pathlib import Path
import argparse
import os
import re
import sys
import time
from typing import Iterable, Optional
from multiprocessing import Pool, cpu_count

import scanpy as sc
from scipy import sparse

# Set PYTHONPATH for multiprocessing subprocesses
perturbench_src = "/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/src"
sys.path.insert(0, perturbench_src)
# Also set environment variable for subprocesses
if 'PYTHONPATH' in os.environ:
    os.environ['PYTHONPATH'] = f"{perturbench_src}:{os.environ['PYTHONPATH']}"
else:
    os.environ['PYTHONPATH'] = perturbench_src

# Import the new HVG-only preprocess function from the same directory
from perturbench.analysis.preprocess_hvg_only import preprocess_hvg_only
# Default directories (can be overridden via command line)
DEFAULT_CELL_LINE_DIR = Path("/fs-computility-new/upzd_share/maoxinjie/AIVC/data/cell_line")
DEFAULT_NORMALIZED_DIR = DEFAULT_CELL_LINE_DIR / "normalized"

# Default preprocess parameters can be overridden per cell line via CUSTOM_PARAMS
DEFAULT_CFG = {
    "perturbation_key": "perturbation",
    "control_value": "control",
    # "covariate_keys": ["dataset"],
    "covariate_keys": ["dataset"],
    "combination_delimiter": "+",
    "highly_variable": 2000,
    "degs": 0,  # Set to 0 to skip DEG calculation (we only want HVG)
    "min_cells_per_group": 2,
}

# Example:
# CUSTOM_PARAMS = {
#     "Jurkat": {"degs": 100, "covariate_keys": ["batch"]},
# }
CUSTOM_PARAMS = {}


def build_datasets(cell_line_dir: Path, normalized_dir: Path) -> list[dict]:
    """Build list of dataset configurations from input directory."""
    datasets = []
    for h5ad_path in sorted(cell_line_dir.glob("*.h5ad")):
        name = h5ad_path.stem
        cfg = DEFAULT_CFG.copy()
        cfg.update(CUSTOM_PARAMS.get(name, {}))
        cfg.update(
            {
                "name": name,
                "input_path": str(h5ad_path),
                "output_path": str(normalized_dir / f"{name}_processed.h5ad"),
            }
        )
        datasets.append(cfg)
    return datasets


def merge_duplicate_var_names(adata, dataset_name=None):
    """
    Merge gene columns that differ only by a numeric suffix introduced during
    scanpy.var_names_make_unique(), e.g., TP53 and TP53-0.
    
    Only merges suffixes that are consecutive starting from 0 or 1 (as produced by
    var_names_make_unique), to avoid merging original gene names like "NKX1-2"
    that were already in the data.
    
    Args:
        adata: AnnData object
        dataset_name: Optional dataset name for logging
    """
    pattern = re.compile(r"^(?P<base>.+)-(?P<suffix>\d+)$")
    bases: dict[str, dict[str, object]] = {}
    for idx, gene in enumerate(adata.var_names):
        match = pattern.match(gene)
        if not match:
            continue
        base = match.group("base")
        suffix = int(match.group("suffix"))
        if base not in adata.var_names:
            continue
        base_idx = adata.var_names.get_loc(base)
        if not isinstance(base_idx, int):
            continue
        record = bases.setdefault(base, {"base_idx": base_idx, "dup_indices": [], "suffixes": []})
        record["dup_indices"].append(idx)
        record["suffixes"].append(suffix)

    if not bases:
        return

    target_matrix = adata.X.tolil() if sparse.issparse(adata.X) else adata.X.copy()
    columns_to_drop: set[str] = set()
    merge_details = []  # Store merge information for printing

    for base, info in bases.items():
        # Sort by suffix to check for consecutive numbering
        sorted_pairs = sorted(zip(info["suffixes"], info["dup_indices"]))
        suffixes_sorted = [s for s, _ in sorted_pairs]
        dup_indices_sorted = [i for _, i in sorted_pairs]
        
        # Only merge if suffixes are consecutive starting from 0 or 1
        # This ensures we only merge var_names_make_unique() generated duplicates,
        # not original gene names like "NKX1-2" that were already in the data
        if not suffixes_sorted:
            continue
        
        # Check if suffixes are consecutive from 0 or 1
        min_suffix = min(suffixes_sorted)
        if min_suffix > 1:
            # If the smallest suffix is > 1, this might be an original gene name
            # (e.g., "NKX1-2" without "NKX1-0" or "NKX1-1")
            continue
        
        # Check if suffixes are consecutive
        expected_suffixes = list(range(min_suffix, min_suffix + len(suffixes_sorted)))
        if suffixes_sorted != expected_suffixes:
            # Suffixes are not consecutive, skip to avoid merging original gene names
            continue
        
        # All checks passed: these are var_names_make_unique() generated duplicates
        base_gene = adata.var_names[info["base_idx"]]
        dup_genes = [adata.var_names[idx] for idx in dup_indices_sorted]
        merge_details.append((base_gene, dup_genes))
        
        indices = [info["base_idx"], *dup_indices_sorted]
        subset = adata.X[:, indices]
        if sparse.issparse(subset):
            merged = subset.mean(axis=1).A1
        else:
            merged = subset.mean(axis=1)
        if sparse.issparse(target_matrix):
            target_matrix[:, info["base_idx"]] = merged.reshape(-1, 1)
        else:
            target_matrix[:, info["base_idx"]] = merged
        columns_to_drop.update(adata.var_names[dup_indices_sorted])

    adata.X = target_matrix.tocsr() if sparse.issparse(target_matrix) else target_matrix

    if columns_to_drop:
        mask = ~adata.var_names.isin(list(columns_to_drop))
        adata._inplace_subset_var(mask)
        prefix = f"[{dataset_name}] " if dataset_name else ""
        print(
            f"{prefix}Merged {len(columns_to_drop)} duplicate gene columns into "
            f"{len(bases)} base names by averaging expression."
        )
        # print("\nMerge details:")
        # for base_gene, dup_genes in merge_details:
        #     print(f"  {base_gene} <- merged with: ', '.join(dup_genes)}")


def is_control_perturbation(pert_str: str, delimiter: str = "+") -> bool:
    """Check if a perturbation is control (all parts are 'control')."""
    if not pert_str or not isinstance(pert_str, str):
        return False
    pert_str = pert_str.strip()
    if pert_str == "":
        return False
    parts = [p.strip().lower() for p in pert_str.split(delimiter)]
    return all(part == "control" for part in parts if part)


def infer_control_value(candidates: Iterable[str], delimiter: str = "+") -> Optional[str]:
    """
    Try to infer a control label from the available perturbation names.
    We require that a rule produces a single match to avoid ambiguity.
    
    Priority:
    1. Exact match: "control" or "control+control" (all parts are control)
    2. Contains control keywords (but only if single match)
    """
    candidates_list = list(candidates)
    normalized = [(val, str(val).lower()) for val in candidates_list]
    
    # Priority 1: Check for pure control (all parts are "control")
    pure_controls = [
        orig for orig, lower in normalized 
        if is_control_perturbation(orig, delimiter)
    ]
    if len(pure_controls) == 1:
        return pure_controls[0]
    elif len(pure_controls) > 1:
        # Multiple pure controls, prefer shortest (e.g., "control" over "control+control")
        return min(pure_controls, key=len)
    
    # Priority 2: Check for exact matches with common control keywords
    rules = [
        lambda v: v[1] == "control",
        lambda v: v[1] == "ctrl",
        lambda v: v[1] == "ntc",
    ]
    for rule in rules:
        matches = [orig for orig, lower in normalized if rule((orig, lower))]
        if len(matches) == 1:
            return matches[0]
    
    # Priority 3: Check for patterns (but only if single match)
    pattern_rules = [
        lambda v: v[1].endswith("_ctrl"),
        lambda v: "ntc" in v[1] and len(v[1]) < 10,  # Short NTC variants
        lambda v: "mock" in v[1] and len(v[1]) < 10,
        lambda v: "vehicle" in v[1] and len(v[1]) < 15,
        lambda v: "dms0" in v[1],
        lambda v: "lacz" in v[1] and len(v[1]) < 10,
    ]
    for rule in pattern_rules:
        matches = [orig for orig, lower in normalized if rule((orig, lower))]
        if len(matches) == 1:
            return matches[0]
    
    return None

def run_preprocess(cfg):
    start_time = time.time()
    dataset_name = cfg['name']
    print(f"\n=== Processing {dataset_name} ===")
    output_path = Path(cfg["output_path"])
    if output_path.exists():
        print(f"[{dataset_name}] Output already exists at {output_path}, skipping.")
        return

    print(f"[{dataset_name}] Loading data from: {cfg['input_path']}")
    load_start = time.time()
    adata = sc.read_h5ad(cfg["input_path"])
    load_time = time.time() - load_start
    print(f"[{dataset_name}] Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes (took {load_time:.1f}s)")
    
    # Add dataset column (filename without .h5ad extension)
    if "dataset" not in adata.obs.columns:
        adata.obs["dataset"] = dataset_name
        print(f"[{dataset_name}] Added obs['dataset'] column with value: {dataset_name}")
    else:
        print(f"[{dataset_name}] obs['dataset'] column already exists, keeping existing values")

    # Ensure count matrix is stored in sparse CSR format
    if adata.X is not None and not sparse.issparse(adata.X):
        print(f"[{dataset_name}] Converting dense expression matrix to sparse CSR format...")
        adata.X = sparse.csr_matrix(adata.X)
    
    # Skip small datasets (too few cells for reliable statistics)
    if adata.n_obs < 50:
        print(
            f"[{dataset_name}] Dataset has only {adata.n_obs} cells; "
            f"skipping preprocess (too few cells for reliable stats)."
        )
        return
    
    # Check gene count
    if adata.n_vars == 0:
        print(
            f"[{dataset_name}] Dataset has 0 genes; "
            f"skipping preprocess (no features available)."
        )
        return
    
    # Check if data matrix is valid
    if adata.X is None:
        print(
            f"[{dataset_name}] Dataset has no expression matrix (X is None); "
            f"skipping preprocess."
        )
        return
    
    # Check if matrix has any data
    try:
        if hasattr(adata.X, 'nnz'):  # Sparse matrix
            if adata.X.nnz == 0:
                print(
                    f"[{dataset_name}] Dataset expression matrix is empty (0 non-zero values); "
                    f"skipping preprocess."
                )
                return
        else:  # Dense matrix
            if adata.X.size == 0 or (adata.X == 0).all():
                print(
                    f"[{dataset_name}] Dataset expression matrix is empty or all zeros; "
                    f"skipping preprocess."
                )
                return
    except Exception as e:
        print(
            f"[{dataset_name}] Error checking expression matrix: {e}; "
            f"skipping preprocess."
        )
        return
    
    perturbation_key = cfg["perturbation_key"]
    if perturbation_key not in adata.obs.columns:
        print(
            f"[{dataset_name}] Perturbation key '{perturbation_key}' not found. "
            f"Available columns: {list(adata.obs.columns)}. Skipping."
        )
        return
    
    available_controls = adata.obs[perturbation_key].unique().tolist()
    control_value = cfg["control_value"]
    combination_delimiter = cfg.get("combination_delimiter", "+")
    
    if control_value not in available_controls:
        inferred = infer_control_value(available_controls, delimiter=combination_delimiter)
        if inferred:
            print(
                f"[{dataset_name}] Control '{control_value}' not found; "
                f"using inferred '{inferred}'."
            )
            cfg["control_value"] = inferred
        else:
            print(
                f"[{dataset_name}] Could not find or infer control. "
                f"Available values: {available_controls}. Skipping dataset."
            )
            return
    
    # Filter out perturbations with fewer than 10 cells
    print(f"[{dataset_name}] Filtering perturbations with < 10 cells...")
    pert_counts = adata.obs[perturbation_key].value_counts()
    valid_perturbations = pert_counts[pert_counts >= 10].index
    removed_perturbations = pert_counts[pert_counts < 10]

    if len(removed_perturbations) > 0:
        print(f"[{dataset_name}] Removing {len(removed_perturbations)} perturbation(s) with < 10 cells:")
        # Only print first 10 removed perturbations to avoid too much output
        for i, (pert, count) in enumerate(removed_perturbations.items()):
            if i >= 10:
                print(f"  ... and {len(removed_perturbations) - 10} more")
                break
            print(f"  - {pert}: {count} cells")

        # Keep only cells with valid perturbations
        mask = adata.obs[perturbation_key].isin(valid_perturbations)
        adata = adata[mask].copy()
        print(f"[{dataset_name}] After filtering: {adata.n_obs} cells remaining")
    else:
        print(f"[{dataset_name}] All perturbations have >= 10 cells")

    # Select top 500 perturbations by cell count
    print(f"[{dataset_name}] Selecting top 500 perturbations by cell count...")
    pert_counts = adata.obs[perturbation_key].value_counts()

    if len(pert_counts) > 500:
        # Get top 500 perturbations
        top_500_perts = pert_counts.nlargest(500).index
        print(f"[{dataset_name}] Keeping top 500 perturbations (out of {len(pert_counts)} total)")
        print(f"[{dataset_name}] Top perturbation: {top_500_perts[0]} ({pert_counts[top_500_perts[0]]} cells)")
        print(f"[{dataset_name}] 500th perturbation: {top_500_perts[-1]} ({pert_counts[top_500_perts[-1]]} cells)")

        # Filter to keep only top 500 perturbations
        mask = adata.obs[perturbation_key].isin(top_500_perts)
        original_cells = adata.n_obs
        adata = adata[mask].copy()
        filtered_cells = adata.n_obs

        print(f"[{dataset_name}] Filtered cells: {original_cells:,} → {filtered_cells:,} ({filtered_cells/original_cells*100:.1f}%)")
    else:
        print(f"[{dataset_name}] Only {len(pert_counts)} perturbations available, keeping all")
    
    # Skip merging duplicate gene names - this step is removed
    # print(f"[{dataset_name}] Merging duplicate gene names...")
    # merge_start = time.time()
    # merge_duplicate_var_names(adata, dataset_name=dataset_name)
    # merge_time = time.time() - merge_start
    # print(f"[{dataset_name}] Merge completed (took {merge_time:.1f}s)")
    
    print(f"[{dataset_name}] Running preprocess_hvg_only (this may take a while for large datasets)...")
    print(f"[{dataset_name}] Data shape before preprocess: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    preprocess_start = time.time()
    try:
        # Use the new HVG-only preprocess function
        processed = preprocess_hvg_only(
            adata=adata,
            perturbation_key=cfg["perturbation_key"],
            covariate_keys=cfg.get("covariate_keys", []),
            control_value=cfg.get("control_value", "control"),
            combination_delimiter=cfg.get("combination_delimiter", "+"),
            highly_variable=cfg.get("highly_variable", 2000),
            min_cells_per_gene=cfg.get("min_cells_per_group", 10),
        )
        
        # Ensure processed count matrix is sparse CSR before saving
        if processed.X is not None and not sparse.issparse(processed.X):
            print(f"[{dataset_name}] Converting processed dense matrix to sparse CSR format before saving...")
            processed.X = sparse.csr_matrix(processed.X)
        preprocess_time = time.time() - preprocess_start
        print(f"[{dataset_name}] Preprocess completed (took {preprocess_time:.1f}s)")
    except IndexError as e:
        error_msg = str(e)
        if "Positions outside range of features" in error_msg:
            print(
                f"[{dataset_name}] ✗ ERROR: IndexError during QC metrics calculation. "
                f"This usually means the dataset has invalid dimensions or empty features. "
                f"Current shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes. "
                f"Skipping this dataset."
            )
            print(f"[{dataset_name}] Error details: {error_msg}")
            return
        else:
            raise
    except Exception as e:
        print(f"[{dataset_name}] ✗ ERROR during preprocess: {e}")
        print(f"[{dataset_name}] Dataset shape: {adata.shape}")
        raise
    
    print(f"[{dataset_name}] Saving to: {output_path}")
    save_start = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.write_h5ad(output_path)
    save_time = time.time() - save_start
    total_time = time.time() - start_time
    print(f"[{dataset_name}] Saved (took {save_time:.1f}s)")
    print(f"[{dataset_name}] ✓ Completed in {total_time:.1f}s total")

def main():
    parser = argparse.ArgumentParser(
        description="Run preprocess() on h5ad files in a directory (HVG only version)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_CELL_LINE_DIR,
        help=f"Input directory containing h5ad files (default: {DEFAULT_CELL_LINE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory for processed files (default: input_dir/normalized)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=f"Number of parallel jobs (default: 1, use -1 for all available CPUs)",
    )
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        output_dir = args.input_dir / "normalized"
    else:
        output_dir = args.output_dir
    
    # Build datasets list
    datasets = build_datasets(args.input_dir, output_dir)
    
    if not datasets:
        print(f"No h5ad files found in {args.input_dir}")
        return
    
    # Determine number of jobs
    if args.n_jobs == -1:
        n_jobs = cpu_count()
    else:
        n_jobs = max(1, min(args.n_jobs, cpu_count(), len(datasets)))
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(datasets)} dataset(s) to process")
    print(f"Using {n_jobs} parallel job(s)")
    print(f"NOTE: This script only selects Highly Variable Genes (HVG), not DEGs.\n")
    
    # Process datasets
    if n_jobs == 1:
        # Serial processing
        for cfg in datasets:
            run_preprocess(cfg)
    else:
        # Parallel processing
        total_start = time.time()
        with Pool(processes=n_jobs) as pool:
            results = pool.map(run_preprocess, datasets)
        total_time = time.time() - total_start
        print(f"\n{'='*70}")
        print(f"All datasets processed in {total_time:.1f}s using {n_jobs} workers")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()

