#!/usr/bin/env python
"""
Script to check the contents of h5ad files and save results to log files
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from glob import glob

# 数据目录
DATA_DIR = "/fs-computility-new/upzd_share/shared/AIVC_data/processed_control/processed"

def check_adata(file_path):
    """检查单个 h5ad 文件并返回信息字典"""
    info = {"file": os.path.basename(file_path)}
    
    try:
        adata = sc.read_h5ad(file_path)
        
        # 基本信息
        info["n_obs"] = adata.n_obs
        info["n_vars"] = adata.n_vars
        info["X_shape"] = str(adata.X.shape)
        info["X_dtype"] = str(adata.X.dtype)
        
        # obs 列
        info["obs_columns"] = list(adata.obs.columns)
        
        # var 列
        info["var_columns"] = list(adata.var.columns)
        
        # 检查常见的 pert 相关列
        pert_cols = [col for col in adata.obs.columns if 'pert' in col.lower() or 'condition' in col.lower()]
        info["pert_columns"] = pert_cols
        
        # 检查 perturbation 的唯一值数量
        for col in pert_cols[:3]:  # 只检查前3个
            info[f"{col}_nunique"] = adata.obs[col].nunique()
        
        # 检查是否有 control
        for col in pert_cols:
            if adata.obs[col].dtype == 'object' or adata.obs[col].dtype.name == 'category':
                ctrl_vals = [v for v in adata.obs[col].unique() if 'control' in str(v).lower() or 'ctrl' in str(v).lower() or v == 'non-targeting']
                if ctrl_vals:
                    info[f"{col}_control_vals"] = ctrl_vals[:3]
        
        # obsm 和 uns
        info["obsm_keys"] = list(adata.obsm.keys()) if adata.obsm else []
        info["uns_keys"] = list(adata.uns.keys())[:10] if adata.uns else []
        
        # layers
        info["layers"] = list(adata.layers.keys()) if adata.layers else []
        
        info["status"] = "OK"
        
        del adata  # 释放内存
        
    except Exception as e:
        info["status"] = f"ERROR: {str(e)}"
    
    return info


def print_adata_info(info):
    """打印单个 adata 的信息"""
    print("=" * 80)
    print(f"File: {info['file']}")
    print("-" * 80)
    
    if info["status"] != "OK":
        print(f"  Status: {info['status']}")
        return
    
    print(f"  Shape: {info['n_obs']} cells x {info['n_vars']} genes")
    print(f"  X dtype: {info['X_dtype']}")
    print(f"  obs columns ({len(info['obs_columns'])}): {info['obs_columns'][:10]}{'...' if len(info['obs_columns']) > 10 else ''}")
    print(f"  var columns ({len(info['var_columns'])}): {info['var_columns'][:10]}{'...' if len(info['var_columns']) > 10 else ''}")
    print(f"  Pert-related columns: {info['pert_columns']}")
    
    # 打印 pert 列的唯一值数量
    for key, val in info.items():
        if key.endswith('_nunique'):
            print(f"    {key}: {val}")
        if key.endswith('_control_vals'):
            print(f"    {key}: {val}")
    
    if info['obsm_keys']:
        print(f"  obsm keys: {info['obsm_keys']}")
    if info['layers']:
        print(f"  layers: {info['layers']}")
    if info['uns_keys']:
        print(f"  uns keys: {info['uns_keys']}")


def main():
    # 获取所有 h5ad 文件
    h5ad_files = sorted(glob(os.path.join(DATA_DIR, "*.h5ad")))
    
    print(f"Found {len(h5ad_files)} h5ad files in {DATA_DIR}")
    print("=" * 80)
    
    all_info = []
    
    for i, file_path in enumerate(h5ad_files):
        print(f"\n[{i+1}/{len(h5ad_files)}] Processing: {os.path.basename(file_path)}")
        info = check_adata(file_path)
        all_info.append(info)
        print_adata_info(info)
    
    # 汇总表格
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    summary_data = []
    for info in all_info:
        if info["status"] == "OK":
            summary_data.append({
                "file": info["file"],
                "n_obs": info["n_obs"],
                "n_vars": info["n_vars"],
                "pert_cols": ", ".join(info["pert_columns"][:2]),
                "status": info["status"]
            })
        else:
            summary_data.append({
                "file": info["file"],
                "n_obs": "N/A",
                "n_vars": "N/A",
                "pert_cols": "N/A",
                "status": info["status"]
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # 保存汇总到 CSV
    output_csv = "adata_check_summary.csv"
    summary_df.to_csv(output_csv, index=False)
    print(f"\nSummary saved to: {output_csv}")


if __name__ == "__main__":
    main()