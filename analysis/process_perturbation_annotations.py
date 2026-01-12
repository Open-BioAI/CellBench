#!/usr/bin/env python3
"""
Process perturbation annotations in scGPT embedding files.

For each cell:
1. Split perturbation by "+" and get unique values
2. Classify perturbations into:
   - gene_pt: perturbations that are in intersect_genes.txt
   - env_pt: perturbations that are in unique_cytokine_perturbations.txt
   - drug_pt: remaining perturbations (not in gene or env lists)
3. Create obs.control: True if unique perturbations only contain "control", False otherwise
4. Verify that control=True cells have empty gene_pt, drug_pt, env_pt
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


def process_single_file(
    input_file: Path,
    output_file: Path,
    gene_set: Set[str],
    cytokine_set: Set[str],
    perturbation_key: str = "perturbation",
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
        "--input-dir",
        type=Path,
        default=Path("/fs-computility-new/upzd_share/maoxinjie/AIVC/data/cell_line/all/scgpt"),
        help="Input directory containing h5ad files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess"),
        help="Output directory for processed files",
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
        "--perturbation-key",
        type=str,
        default="perturbation",
        help="Name of the perturbation column in obs (default: 'perturbation')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing gene_pt, drug_pt, env_pt, control columns if they exist",
    )
    args = parser.parse_args()
    
    # Load gene and cytokine sets
    print("Loading gene set...")
    gene_set = load_gene_set(args.gene_list)
    print(f"  Loaded {len(gene_set):,} genes")
    
    print("Loading cytokine set...")
    cytokine_set = load_cytokine_set(args.cytokine_list)
    print(f"  Loaded {len(cytokine_set):,} cytokines")
    
    # Find all h5ad files
    h5ad_files = sorted([f for f in args.input_dir.glob("*.h5ad") if not f.name.startswith(".")])
    print(f"\nFound {len(h5ad_files)} h5ad files to process")
    
    if len(h5ad_files) == 0:
        print("No h5ad files found!")
        sys.exit(1)
    
    # Process each file
    success_count = 0
    for h5ad_file in tqdm(h5ad_files, desc="Processing files"):
        output_file = args.output_dir / h5ad_file.name
        
        # Skip if output exists and not overwriting
        if output_file.exists() and not args.overwrite:
            print(f"\nSkipping {h5ad_file.name} (output already exists, use --overwrite to force)")
            continue
        
        if process_single_file(h5ad_file, output_file, gene_set, cytokine_set, args.perturbation_key):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"  Successfully processed: {success_count}/{len(h5ad_files)} files")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

