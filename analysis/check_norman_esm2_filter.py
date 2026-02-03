#!/usr/bin/env python3
"""
查看 Norman 数据集所有扰动，若某细胞的任意扰动基因不在 ESM2_pert_features.pt 中则过滤掉。
统计过滤比例后，将过滤后的数据写回原路径（覆盖）。
对照（空扰动或单独 "control"）不参与 ESM2 过滤，一律保留。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

try:
    import anndata as ad
except ImportError:
    import scanpy as sc
    ad = sc


def _parse_pert_genes(pert_str, delim: str = "+"):
    """从扰动字符串解析出基因列表。"""
    if pert_str is None or (isinstance(pert_str, float) and np.isnan(pert_str)):
        return []
    s = str(pert_str).strip()
    if s.lower() in ("nan", "none", ""):
        return []
    return [p.strip() for p in s.split(delim) if p.strip()]


def load_esm2_pert_keys(esm2_pt_path: Path):
    """加载 ESM2_pert_features.pt，返回其中所有 perturbation/gene 名称的集合。"""
    if torch is None:
        raise RuntimeError("需要 PyTorch 以加载 .pt 文件")
    data = torch.load(esm2_pt_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        return set(data.keys())
    if hasattr(data, "keys"):
        return set(data.keys())
    raise TypeError(f"ESM2 .pt 期望 dict 或带 .keys() 的对象，得到 {type(data)}")


def main():
    parser = argparse.ArgumentParser(description="过滤 Norman 中扰动不在 ESM2 的细胞，统计后覆盖原 h5ad")
    parser.add_argument(
        "--norman-h5ad",
        type=Path,
        default=Path("/fs-computility-new/upzd_share/shared/AIVC_data/processed_control/processed/NormanWeissman2019_filtered_processed.h5ad"),
        help="Norman 数据集 h5ad 路径",
    )
    parser.add_argument(
        "--esm2-pt",
        type=Path,
        default=Path("ESM2_pert_features.pt"),
        help="ESM2 扰动特征 .pt 路径（加载后 .keys() 为扰动/基因名）",
    )
    parser.add_argument("--pert-col", type=str, default=None, help="扰动列名（默认 perturbation 或 gene_pt）")
    args = parser.parse_args()

    norman_path = args.norman_h5ad
    esm2_path = args.esm2_pt

    if not norman_path.exists():
        print(f"Norman 文件不存在: {norman_path}")
        return
    if not esm2_path.exists():
        print(f"ESM2 .pt 文件不存在: {esm2_path}")
        return

    print(f"加载 Norman: {norman_path}")
    adata = ad.read_h5ad(norman_path)
    n_total = adata.n_obs

    pert_col = args.pert_col
    if pert_col is None:
        pert_col = "perturbation" if "perturbation" in adata.obs.columns else "gene_pt"
    if pert_col not in adata.obs.columns:
        print(f"扰动列不存在: {pert_col}，可选: {list(adata.obs.columns)}")
        return

    print(f"加载 ESM2 扰动集合: {esm2_path}")
    esm2_keys = load_esm2_pert_keys(esm2_path)
    print(f"  ESM2 中扰动/基因数: {len(esm2_keys)}")

    pert_series = adata.obs[pert_col]
    keep_mask = np.ones(adata.n_obs, dtype=bool)
    n_control = 0
    missing_genes = set()

    for i, v in enumerate(pert_series):
        genes = _parse_pert_genes(v)
        if len(genes) == 0:
            n_control += 1
            continue
        # 单独一个 "control" 视为对照，不参与 ESM2 过滤
        if len(genes) == 1 and str(genes[0]).strip().lower() == "control":
            n_control += 1
            continue
        if any(g not in esm2_keys for g in genes):
            keep_mask[i] = False
            for g in genes:
                if g not in esm2_keys and str(g).strip().lower() != "control":
                    missing_genes.add(g)

    n_filtered = (~keep_mask).sum()
    n_pert_cells = n_total - n_control
    n_kept_pert = n_pert_cells - n_filtered
    n_kept_total = keep_mask.sum()

    print()
    print("=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"  总细胞数:           {n_total}")
    print(f"  对照细胞数:         {n_control}")
    print(f"  扰动细胞数:         {n_pert_cells}")
    print(f"  被过滤细胞数:       {n_filtered}（任意扰动基因不在 ESM2 中）")
    print(f"  保留扰动细胞数:     {n_kept_pert}")
    print(f"  保留总细胞数:       {n_kept_total}")
    if n_total > 0:
        pct_filtered_all = 100.0 * n_filtered / n_total
        print(f"  过滤比例(占全部):   {pct_filtered_all:.2f}%")
    if n_pert_cells > 0:
        pct_filtered_pert = 100.0 * n_filtered / n_pert_cells
        print(f"  过滤比例(占扰动):   {pct_filtered_pert:.2f}%")
    if missing_genes:
        print(f"  缺失于 ESM2 的基因数: {len(missing_genes)}")
        print(f"  缺失基因示例: {sorted(missing_genes)[:20]}{'...' if len(missing_genes) > 20 else ''}")

    adata_filtered = adata[keep_mask].copy()
    adata_filtered.write_h5ad(norman_path)
    print()
    print(f"已覆盖写入: {norman_path}")
    print(f"  过滤后 Shape: {adata_filtered.shape[0]} cells × {adata_filtered.shape[1]} genes")


if __name__ == "__main__":
    main()
