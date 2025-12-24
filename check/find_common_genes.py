#!/usr/bin/env python3
"""
Script to find genes that are expressed (>0) in ALL cell lines within each dataset,
then create a final gene set and rank genes by expression frequency across all cells.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.sparse import issparse

def find_common_expressed_genes_and_rank(h5ad_path):
    """
    Load H5AD file and find genes expressed in all cell lines for each dataset,
    then create a final gene set ranked by expression frequency across all cells.

    Args:
        h5ad_path (str): Path to the H5AD file

    Returns:
        dict: Results containing per-dataset genes and final ranked gene set
    """
    print(f"Loading data from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded dataset with {adata.shape[0]} cells and {adata.shape[1]} genes")
    print(f"Available obs columns: {list(adata.obs.columns)}")

    # Group by dataset and cell_cluster
    dataset_cell_lines = defaultdict(list)

    for dataset in adata.obs['dataset'].unique():
        mask = adata.obs['dataset'] == dataset
        cell_lines = adata.obs.loc[mask, 'cell_cluster'].unique()
        dataset_cell_lines[dataset] = list(cell_lines)
        print(f"Dataset '{dataset}': {len(cell_lines)} cell lines")

    # For each dataset, find genes expressed in ALL cell lines
    common_expressed_genes_per_dataset = {}

    for dataset, cell_lines in dataset_cell_lines.items():
        print(f"\nProcessing dataset '{dataset}' with {len(cell_lines)} cell lines: {cell_lines}")

        # Collect sets of genes that are expressed (>0) in each cell line
        expressed_gene_sets = []
        for cell_line in cell_lines:
            mask = (adata.obs['dataset'] == dataset) & (adata.obs['cell_cluster'] == cell_line)
            cell_data = adata[mask]

            if cell_data.shape[0] > 0:  # Only if there are cells for this cell line
                # Find genes expressed in at least one cell of this cell line
                X = cell_data.X
                if issparse(X):
                    # (X > 0) 对稀疏矩阵仍然是稀疏的，然后 sum(axis=0) 得列和
                    expressed_mask = (X > 0).sum(axis=0).A1 > 0  # .A1 把 (1, n_genes) 变成 1D array
                else:
                    expressed_mask = np.any(X > 0, axis=0)
                expressed_genes = set(cell_data.var_names[expressed_mask])

                expressed_gene_sets.append(expressed_genes)
                print(f"  Cell line '{cell_line}': {cell_data.shape[0]} cells, {len(expressed_genes)} expressed genes")
            else:
                print(f"  Cell line '{cell_line}': No cells found")

        # Find intersection of expressed genes across all cell lines
        if expressed_gene_sets:
            common_expressed_genes = set.intersection(*expressed_gene_sets)
            common_expressed_genes_per_dataset[dataset] = sorted(list(common_expressed_genes))
            print(f"  Genes expressed in ALL cell lines: {len(common_expressed_genes)}")
        else:
            common_expressed_genes_per_dataset[dataset] = []
            print("  No gene sets found")

    # Create final gene set (intersection of all per-dataset common genes)
    # This gives genes that are expressed in ALL cell lines across ALL datasets
    final_gene_set = None
    for genes in common_expressed_genes_per_dataset.values():
        genes_set = set(genes)
        if final_gene_set is None:
            final_gene_set = genes_set
        else:
            final_gene_set &= genes_set  # intersection

    if final_gene_set is None:
        final_gene_set = set()

    print(f"\nFinal gene set contains {len(final_gene_set)} unique genes")

    # For each gene in final set, count how many cells express it (>0)
    print("Calculating expression frequency for each gene...")
    gene_expression_counts = {}

    for gene in final_gene_set:
        if gene in adata.var_names:
            gene_idx = adata.var_names.get_loc(gene)
            X = adata.X
            col = X[:, gene_idx] > 0
            if issparse(col):
                expressed_count = int(col.sum())  # or int(col.sum().A1[0])
            else:
                expressed_count = int(np.sum(col))
            gene_expression_counts[gene] = expressed_count
        else:
            gene_expression_counts[gene] = 0

    # Sort genes by expression count (descending)
    sorted_genes = sorted(gene_expression_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        'per_dataset': common_expressed_genes_per_dataset,
        'final_gene_set': list(final_gene_set),
        'ranked_by_expression': sorted_genes,
        'total_cells': adata.shape[0]
    }

def main():
    h5ad_path = '/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/all_cell_line_filterdrug.h5ad'

    try:
        results = find_common_expressed_genes_and_rank(h5ad_path)

        print("\n" + "="*80)
        print("RESULTS - Genes Expressed in ALL Cell Lines per Dataset")
        print("="*80)

        # Show per-dataset results
        print("\nPer-dataset results:")
        for dataset, genes in results['per_dataset'].items():
            print(f"Dataset '{dataset}': {len(genes)} genes expressed in all cell lines")

        # Show final gene set summary
        print(f"\nFinal gene set: {len(results['final_gene_set'])} unique genes")
        print(f"Total cells analyzed: {results['total_cells']}")

        # Show top 20 genes by expression frequency
        print("\nTop 20 genes by expression frequency across all cells:")
        print("Rank | Gene            | Cells Expressed | Expression Ratio")
        print("-" * 60)
        for i, (gene, count) in enumerate(results['ranked_by_expression'][:20], 1):
            ratio = count / results['total_cells']
            print(f"{i:4d} | {gene:15s} | {count:14d} | {ratio:.4f}")

        # Save comprehensive results
        # 1. Per-dataset results
        dataset_output = []
        for dataset, genes in results['per_dataset'].items():
            dataset_output.append({
                'dataset': dataset,
                'num_common_expressed_genes': len(genes),
                'common_expressed_genes': ';'.join(genes)
            })

        df_dataset = pd.DataFrame(dataset_output)
        dataset_output_path = '/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/check/common_expressed_genes_per_dataset.csv'
        df_dataset.to_csv(dataset_output_path, index=False)
        print(f"\nPer-dataset results saved to: {dataset_output_path}")

        # 2. Final ranked gene list
        ranked_output = []
        for gene, count in results['ranked_by_expression']:
            ratio = count / results['total_cells']
            ranked_output.append({
                'gene': gene,
                'cells_expressed': count,
                'expression_ratio': ratio
            })

        df_ranked = pd.DataFrame(ranked_output)
        ranked_output_path = '/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/check/final_gene_set_ranked_by_expression.csv'
        df_ranked.to_csv(ranked_output_path, index=False)
        print(f"Ranked gene list saved to: {ranked_output_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
