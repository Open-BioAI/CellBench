#!/usr/bin/env python3
"""
Process perturbation annotations

For each cell:
1. Split perturbation by "+" and get unique values
2. Classify perturbations into:
   - gene_pt: perturbations that are in intersect_genes.txt
   - env_pt: perturbations that are in unique_cytokine_perturbations.txt
   - drug_pt: remaining perturbations (not in gene or env lists)
3. Create obs.control: True if unique perturbations only contain "control", False otherwise
4. Verify that control=True cells have empty gene_pt, drug_pt, env_pt
5. Add CRISPR and cell_cluster columns if they don't exist
"""

import argparse
import sys
from pathlib import Path
from typing import Set

import scanpy as sc
import pandas as pd
from tqdm import tqdm


def load_gene_set(gene_file: Path) -> Set[str]:
    """Load gene names from a text file (one per line)."""
    genes = set()
    with open(gene_file, 'r') as f:
        for line in f:
            gene = line.strip()
            if gene:  # Skip empty lines
                genes.add(gene)
    return genes


def load_cytokine_set(cytokine_file: Path) -> Set[str]:
    """Load cytokine names from a text file (one per line)."""
    cytokines = set()
    with open(cytokine_file, 'r') as f:
        for line in f:
            cytokine = line.strip()
            if cytokine:  # Skip empty lines
                cytokines.add(cytokine)
    return cytokines


def normalize_pert_string(s) -> str:
    """将一个扰动字符串标准化：按'+'拆分、strip、排序后再用'+'连接."""
    # 先检查是否为 None 或 NaN（避免 astype(str) 把 NaN 变成 "nan" 字符串）
    if s is None or pd.isna(s):
        return ""

    # 转换为字符串并检查是否为 "nan"/"None" 字符串
    s_str = str(s).strip()
    if s_str.lower() in {"nan", "none"}:
        return ""

    parts = [p.strip() for p in s_str.split("+") if p.strip() != ""]
    if not parts:
        return ""
    parts_sorted = sorted(parts)
    return "+".join(parts_sorted)


def build_pert_key(
    obs: pd.DataFrame, gene_pt_col: str, drug_pt_col: str, env_pt_col: str
) -> pd.Series:
    """Build perturbation key from gene_pt, drug_pt, env_pt columns."""
    # 去掉 astype(str)，让 normalize_pert_string 直接处理原始值（包括 NaN）
    g_norm = obs[gene_pt_col].apply(normalize_pert_string)
    d_norm = obs[drug_pt_col].apply(normalize_pert_string)
    e_norm = obs[env_pt_col].apply(normalize_pert_string)
    # 全部转换为字符串类型，避免 Categorical 类型导致的错误
    g_norm = g_norm.astype(str)
    d_norm = d_norm.astype(str)
    e_norm = e_norm.astype(str)
    return g_norm + "|" + d_norm + "|" + e_norm


def add_crispr_column(obs: pd.DataFrame, crispr_value: str = "") -> None:
    """Add CRISPR column to obs DataFrame."""
    obs['CRISPR'] = crispr_value
    print(f"Added CRISPR column with value: '{crispr_value}'")


def add_cell_cluster_column(obs: pd.DataFrame) -> None:
    """Add cell_cluster column to obs DataFrame.

    Priority: cell_line > celltype
    """
    if 'cell_line' in obs.columns:
        obs['cell_cluster'] = obs['cell_line']
        print("Using 'cell_line' column as 'cell_cluster'")
    elif 'celltype' in obs.columns:
        obs['cell_cluster'] = obs['celltype']
        print("Using 'celltype' column as 'cell_cluster'")
    else:
        raise ValueError("Neither 'cell_line' nor 'celltype' column found in adata.obs")


def normalize_drug_name(drug: str) -> str:
    """
    Normalize drug name: convert to lowercase and remove leading "-" if present.

    Examples:
        "-JQ1" -> "jq1"
        "IFNB" -> "ifnb"
        "control" -> "control"
    """
    if not drug or pd.isna(drug):
        return ""
    drug = str(drug).strip()
    # Convert to lowercase
    drug = drug.lower()
    # Remove leading "-" if present
    if drug.startswith("-"):
        drug = drug[1:]
    return drug


def process_perturbation(
    perturbation_str: str,
    gene_set: Set[str],
    cytokine_set: Set[str],
) -> tuple[str, str, str, bool]:
    """
    Process a single perturbation string.
    
    Returns:
        (gene_pt, drug_pt, env_pt, is_control)
    """
    if pd.isna(perturbation_str) or perturbation_str == "":
        return "", "", "", False
    
    # Split by "+" and get unique values
    perts = [p.strip() for p in str(perturbation_str).split("+")]
    unique_perts = list(set([p for p in perts if p]))  # Remove empty strings
    
    # Check if only "control"
    is_control = len(unique_perts) == 1 and unique_perts[0].lower() == "control"
    
    # Classify perturbations
    gene_perts = []
    env_perts = []
    drug_perts = []
    
    for pert in unique_perts:
        pert_lower = pert.lower()
        if pert_lower == "control":
            # Skip control in classification
            continue
        elif pert in cytokine_set:
            env_perts.append(pert)
        elif pert in gene_set:
            gene_perts.append(pert)
        else:
            # Normalize drug name: lowercase and remove leading "-"
            normalized_drug = normalize_drug_name(pert)
            drug_perts.append(normalized_drug)
    
    # Join with "+"
    gene_pt = "+".join(sorted(gene_perts)) if gene_perts else ""
    env_pt = "+".join(sorted(env_perts)) if env_perts else ""
    drug_pt = "+".join(sorted(drug_perts)) if drug_perts else ""
    
    return gene_pt, drug_pt, env_pt, is_control


def add_crispr_and_cell_cluster_columns(
    adata,
    crispr_value: str = "",
) -> None:
    """Add CRISPR and cell_cluster columns to adata.obs."""
    obs = adata.obs

    # Add CRISPR column
    add_crispr_column(obs, crispr_value)

    # Add cell_cluster column
    add_cell_cluster_column(obs)


def process_single_file(
    input_file: Path,
    output_file: Path,
    gene_set: Set[str],
    cytokine_set: Set[str],
    perturbation_key: str = "perturbation",
    crispr_value: str = "",
) -> bool:
    """Process a single h5ad file."""
    try:
        print(f"\nProcessing: {input_file.name}")
        
        # Load data
        adata = sc.read_h5ad(input_file)
        print(f"  Shape: {adata.shape}")
        
        # Check if perturbation column exists
        if perturbation_key not in adata.obs.columns:
            # Try common alternative names
            alternatives = ['perturbation', 'pert', 'perturbation_key']
            found_key = None
            for alt_key in alternatives:
                if alt_key in adata.obs.columns:
                    found_key = alt_key
                    print(f"  [INFO] Using '{alt_key}' as perturbation column")
                    break
            
            if found_key is None:
                print(f"  [WARNING] No '{perturbation_key}' column found. Available columns: {list(adata.obs.columns)[:10]}...")
                return False
            perturbation_key = found_key
        
        # Remove cells with NaN perturbation values
        initial_n_obs = adata.n_obs
        nan_mask = pd.isna(adata.obs[perturbation_key]) | (adata.obs[perturbation_key] == "")| (adata.obs[perturbation_key] == "nan")
        n_nan = nan_mask.sum()
        if n_nan > 0:
            adata = adata[~nan_mask].copy()
            print(f"  Removed {n_nan:,} cells with NaN/empty perturbation (remaining: {adata.n_obs:,} cells)")
        else:
            print(f"  No cells with NaN/empty perturbation found")
        
        # Process each cell
        print(f"  Processing {adata.n_obs:,} cells...")
        results = []
        for pert_str in tqdm(adata.obs[perturbation_key], desc="    Processing", leave=False):
            gene_pt, drug_pt, env_pt, is_control = process_perturbation(
                pert_str, gene_set, cytokine_set
            )
            results.append({
                'gene_pt': gene_pt,
                'drug_pt': drug_pt,
                'env_pt': env_pt,
                'control': is_control,
            })
        
        # Add new columns (overwrite if they already exist)
        results_df = pd.DataFrame(results, index=adata.obs.index)
        adata.obs['gene_pt'] = results_df['gene_pt']
        adata.obs['drug_pt'] = results_df['drug_pt']
        adata.obs['env_pt'] = results_df['env_pt']
        adata.obs['control'] = results_df['control']

        # Add CRISPR and cell_cluster columns if they don't exist
        if 'CRISPR' not in adata.obs.columns:
            # 如果有 gene_pt 值，则设置为 crispr_value，否则为空字符串
            adata.obs['CRISPR'] = adata.obs['gene_pt'].apply(lambda x: crispr_value if x != "" else "")
            print(f"Auto-set CRISPR column based on gene_pt values (value: '{crispr_value}' for cells with gene_pt)")
        if 'cell_cluster' not in adata.obs.columns:
            add_cell_cluster_column(adata.obs)
        
        # Verify control cells
        control_mask = adata.obs['control'] == True
        n_control = control_mask.sum()
        if n_control > 0:
            control_gene_pt = adata.obs.loc[control_mask, 'gene_pt']
            control_drug_pt = adata.obs.loc[control_mask, 'drug_pt']
            control_env_pt = adata.obs.loc[control_mask, 'env_pt']
            
            non_empty_gene = (control_gene_pt != "").sum()
            non_empty_drug = (control_drug_pt != "").sum()
            non_empty_env = (control_env_pt != "").sum()
            
            if non_empty_gene > 0 or non_empty_drug > 0 or non_empty_env > 0:
                print(f"  [WARNING] Found {n_control} control cells, but:")
                if non_empty_gene > 0:
                    print(f"    - {non_empty_gene} control cells have non-empty gene_pt")
                if non_empty_drug > 0:
                    print(f"    - {non_empty_drug} control cells have non-empty drug_pt")
                if non_empty_env > 0:
                    print(f"    - {non_empty_env} control cells have non-empty env_pt")
            else:
                print(f"  ✓ Verified: {n_control} control cells all have empty gene_pt, drug_pt, env_pt")
        
        # Print statistics
        n_with_gene = (adata.obs['gene_pt'] != "").sum()
        n_with_drug = (adata.obs['drug_pt'] != "").sum()
        n_with_env = (adata.obs['env_pt'] != "").sum()
        print(f"  Statistics:")
        print(f"    - Cells with gene_pt: {n_with_gene:,} ({n_with_gene/adata.n_obs*100:.1f}%)")
        print(f"    - Cells with drug_pt: {n_with_drug:,} ({n_with_drug/adata.n_obs*100:.1f}%)")
        print(f"    - Cells with env_pt: {n_with_env:,} ({n_with_env/adata.n_obs*100:.1f}%)")
        print(f"    - Control cells: {n_control:,} ({n_control/adata.n_obs*100:.1f}%)")
        
        # Save
        output_file.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_file)
        print(f"  ✓ Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to process {input_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Process perturbation annotations in scGPT embedding files"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Input h5ad file to process",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output h5ad file path",
    )
    parser.add_argument(
        "--gene-list",
        type=Path,
        default=Path("/fs-computility-new/upzd_share/maoxinjie/AIVC/data/perturbation/unique_gene_perturbations.txt"),
        help="Path to gene list file",
    )
    parser.add_argument(
        "--cytokine-list",
        type=Path,
        default=Path("/fs-computility-new/upzd_share/maoxinjie/AIVC/data/perturbation/unique_cytokine_perturbations.txt"),
        help="Path to cytokine list file",
    )
    parser.add_argument(
        "--crispr-value",
        type=str,
        default="",
        help="Value to set for CRISPR column (default: empty string)",
    )
    parser.add_argument(
        "--perturbation-key",
        type=str,
        default="perturbation",
        help="Name of the perturbation column in obs (default: 'perturbation')",
    )
    args = parser.parse_args()

    # Load gene and cytokine sets
    print("Loading gene set...")
    gene_set = load_gene_set(args.gene_list)
    print(f"  Loaded {len(gene_set):,} genes")
    
    print("Loading cytokine set...")
    cytokine_set = load_cytokine_set(args.cytokine_list)
    print(f"  Loaded {len(cytokine_set):,} cytokines")
    
    # Check input file exists
    if not args.input_file.exists():
        print(f"Error: Input file does not exist: {args.input_file}")
        sys.exit(1)
    
    # Process single file
    print(f"\nProcessing single file: {args.input_file.name}")
    success = process_single_file(
        args.input_file,
        args.output_file,
        gene_set,
        cytokine_set,
        args.perturbation_key,
        args.crispr_value
    )
    
    print(f"\n{'='*80}")
    if success:
        print("Processing complete!")
        print(f"  Successfully processed: {args.input_file}")
        print(f"  Output saved to: {args.output_file}")
    else:
        print("Processing failed!")
        sys.exit(1)
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

