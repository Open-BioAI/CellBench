"""
Simplified preprocess function that only selects Highly Variable Genes (HVG).
No DEG calculation or perturbation gene filtering.

This function follows the same preprocessing steps as the original preprocess(),
but only selects HVG genes (no DEGs or perturbation genes).
"""

import sys
from pathlib import Path
import numpy as np
from scipy import sparse as sp
import pandas as pd

# Add perturbench src to path if not already there
_this_file = Path(__file__)
_perturbench_src = _this_file.parent.parent / "src"
if str(_perturbench_src) not in sys.path:
    sys.path.insert(0, str(_perturbench_src))

import scanpy as sc
import anndata as ad
import warnings


def preprocess_hvg_only(
    adata: ad.AnnData,
    perturbation_key: str,
    covariate_keys: list[str],
    control_value: str = "control",
    combination_delimiter: str = "+",
    highly_variable: int = 2000,
    min_cells_per_gene: int = 10,
):
    """
    Simplified preprocess function that only selects Highly Variable Genes (HVG).
    
    This function:
    1. Filters genes by minimum cells
    2. Calculates QC metrics
    3. Normalizes and log-transforms
    4. Computes HVG
    5. Subsets to HVG only
    
    Does NOT:
    - Filter by GO terms
    - Calculate DEGs
    - Include perturbation genes
    - Filter perturbations
    
    Args:
        adata: AnnData object
        perturbation_key: Key in adata.obs containing perturbations (used for merging covariates)
        covariate_keys: List of keys in adata.obs for covariates
        control_value: Control perturbation value
        combination_delimiter: Delimiter for combination perturbations
        highly_variable: Number of highly variable genes to select
        min_cells_per_gene: Minimum number of cells expressing a gene to keep it
    
    Returns:
        Processed AnnData object with only HVG genes
    """
    # Step 1: Make names unique (same as original)
    adata.raw = None
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    
    # Step 2: Remove cells with all-zero expression (same as original)
    if sp.issparse(adata.X):
        nonzero_mask = adata.X.getnnz(axis=1) > 0
    else:
        nonzero_mask = np.count_nonzero(adata.X, axis=1) > 0
    n_removed = adata.n_obs - int(nonzero_mask.sum())
    if n_removed > 0:
        print(f"[INFO] Removed {n_removed} cells with all-zero expression.")
        adata = adata[nonzero_mask].copy()
    
    # Step 3: Drop rows with missing perturbation labels (same as original)
    missing_mask = adata.obs[perturbation_key].isna()
    if missing_mask.any():
        print(
            f"Warning: Dropping {missing_mask.sum()} cells with NaN perturbation labels."
        )
        adata = adata[~missing_mask].copy()
    
    # Step 4: Remove perturbation categories that have zero cells (same as original)
    vc = adata.obs[perturbation_key].value_counts()
    valid_perts = vc[vc > 0].index
    if len(valid_perts) < len(vc):
        removed = len(vc) - len(valid_perts)
        print(
            f"[INFO] Dropping {removed} perturbation categories with zero cells "
            "before covariate merge."
        )
    adata = adata[adata.obs[perturbation_key].isin(valid_perts)].copy()
    if pd.api.types.is_categorical_dtype(adata.obs[perturbation_key]):
        adata.obs[perturbation_key] = (
            adata.obs[perturbation_key].cat.remove_unused_categories()
        )
    else:
        adata.obs[perturbation_key] = adata.obs[perturbation_key].astype("category")
    
    # Step 5: Merge covariate columns (same as original)
    from perturbench.analysis.utils import merge_cols
    adata.obs["cov_merged"] = merge_cols(adata.obs, covariate_keys)
    batch_key = "cov_merged" if len(covariate_keys) > 0 else None
    
    # Step 6: Preprocess (same as original, but note: original doesn't call calculate_qc_metrics)
    print("Preprocessing ...")
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    # NOTE: Original preprocess() does NOT call calculate_qc_metrics (line 277 is commented)
    # sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    # Step 7: Normalize (same as original)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    # Step 8: Calculate HVG with fallback mechanism (same as original, but only HVG)
    if highly_variable > 0:
        print(f"Calculating {highly_variable} highly variable genes ...")
        # Use same fallback mechanism as original
        try:
            sc.pp.highly_variable_genes(
                adata,
                batch_key=batch_key,
                flavor="seurat_v3",
                layer="counts",
                n_top_genes=int(highly_variable),
                subset=False,
            )
        except Exception as e:
            print(f"[WARN] HVG (seurat_v3 with batch_key) failed: {e}")
            try:
                # Fallback: try without batch_key
                sc.pp.highly_variable_genes(
                    adata,
                    flavor="seurat_v3",
                    layer="counts",
                    n_top_genes=int(highly_variable),
                    subset=False,
                )
                print("[INFO] Success using flavor='seurat_v3' without batch_key.")
            except Exception as e2:
                print(f"[WARN] HVG (seurat_v3 without batch_key) also failed: {e2}")
                print("[INFO] Falling back to flavor='cell_ranger' ...")
                sc.pp.highly_variable_genes(
                    adata,
                    flavor="cell_ranger",
                    layer="counts",
                    n_top_genes=int(highly_variable),
                    subset=False,
                )
                print("[INFO] HVG using flavor='cell_ranger'.")
        
        # Get HVG genes
        hvg_mask = adata.var["highly_variable"]
        n_hvg = hvg_mask.sum()
        n_total = len(adata.var_names)
        
        print(f"Selected {n_hvg:,} HVG genes out of {n_total:,} total genes ({n_hvg/n_total*100:.1f}%)")
        
        # Subset to HVG only (DIFFERENT: original includes DEGs and perturbation genes)
        adata = adata[:, hvg_mask].copy()
        
        # Store HVG list in uns
        adata.uns['hvg_genes'] = list(adata.var_names)
    else:
        print("Warning: highly_variable=0, keeping all genes")
        adata.uns['hvg_genes'] = list(adata.var_names)
    
    # Step 9: Clean up intermediate columns (same as original)
    # Remove cov_merged from obs
    if "cov_merged" in adata.obs.columns:
        adata.obs.drop(columns=["cov_merged"], inplace=True)
    
    # Remove HVG-related columns from var
    hvg_related_cols = []
    for col in adata.var.columns:
        if col.startswith("highly_variable") or col in {
            "means",
            "dispersions",
            "dispersions_norm",
            "variances",
            "variances_norm",
        }:
            hvg_related_cols.append(col)
    if hvg_related_cols:
        adata.var.drop(columns=hvg_related_cols, inplace=True)
    
    # Remove intermediate uns keys
    for key in ["hvg", "log1p", "variances", "variances_norm"]:
        if key in adata.uns:
            adata.uns.pop(key, None)
    
    print("Processed dataset summary:")
    print(adata)
    
    return adata

