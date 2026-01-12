#!/usr/bin/env python3
"""
针对 SrivatsanTrapnell2020_sciplex3_processed.h5ad 添加 split 列和 CRISPR 列到 obs。

流程：
1. 读取 SrivatsanTrapnell2020_sciplex3_processed.h5ad
2. 添加 CRISPR 列到 obs（通过命令行参数指定）
3. 在该数据集上：
   - 所有 cell_cluster == "MCF7" 或 "A549" 的细胞设为 train
   - 所有 cell_cluster == "K562" 的细胞，在内部随机 1:1 划分为 val / test
4. 按 (gene_pt, drug_pt, env_pt) 构建 perturbation 组合，只保留在 train 中出现过的组合在 val/test 中的细胞
5. 对 train 中的 perturbed cells (control==False) 进行上采样，少数类上采样到多数类数量（replace=True），多数类不变
6. 在 adata.obs 中添加 'split' 列，保存为新的 h5ad 文件
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


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


# 默认输入文件
DEFAULT_H5AD = Path(
    "/fs-computility-new/upzd_share/shared/AIVC_data/processed_control/srivatsan/after_processed/sciplex3_highdose_processed.h5ad"
)
# 输出到 SrivatsanTrapnell2020_sciplex3 目录
DEFAULT_OUTPUT = Path(
    "/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/sciplex3_highdose.h5ad"
)
CELL_CLUSTER_COL = "cell_cluster"


def resample_train_perturbed_cells(
    train_cells: pd.Index,
    obs: pd.DataFrame,
    cell_cluster_col: str,
    control_col: str = "control",
    seed: int = 42,
) -> pd.Index:
    """
    对 train 中的 perturbed cells 进行有放回抽样，使得不同 cell_cluster 的 perturbed cells 数量大致 1:1。
    
    Args:
        train_cells: train split 的细胞索引
        obs: 观察数据框
        cell_cluster_col: cell_cluster 列名
        control_col: control 列名
        seed: 随机种子
    
    Returns:
        需要追加的重复细胞索引（pd.Index，包含重复采样的细胞）
    """
    # 检查 control 列
    if control_col not in obs.columns:
        raise ValueError(f"Column '{control_col}' not found in adata.obs")
    
    # 识别 perturbed cells (control == False)
    is_control = obs[control_col].astype(bool)
    is_perturbed = ~is_control
    
    # 获取 train 中的 perturbed cells
    train_obs = obs.loc[train_cells]
    train_perturbed_mask = is_perturbed.loc[train_cells]
    train_perturbed_cells = train_cells[train_perturbed_mask.values]
    
    print(f"\nTrain perturbed cells before resampling: {len(train_perturbed_cells):,}")
    
    # 按 cell_cluster 分组
    train_perturbed_clusters = train_obs.loc[train_perturbed_cells, cell_cluster_col].astype(str)
    cluster_counts = train_perturbed_clusters.value_counts()
    
    print(f"\nPerturbed cells per cell_cluster in train (before resampling):")
    for cluster, count in cluster_counts.items():
        print(f"  {cluster}: {count:,}")
    
    # 只处理 MCF7 和 A549
    target_clusters = ["MCF7", "A549"]
    cluster_counts_filtered = {k: v for k, v in cluster_counts.items() if k in target_clusters}
    
    if len(cluster_counts_filtered) < 2:
        print(f"\n[WARNING] Only found {len(cluster_counts_filtered)} target cluster(s) in train perturbed cells. Skipping resampling.")
        # 返回空的 Index，不需要追加任何细胞
        return pd.Index([], dtype=train_cells.dtype)
    
    # 找到数量最多的 cluster 作为目标数量（上采样少数类到多数类）
    max_count = max(cluster_counts_filtered.values())
    print(f"\nTarget count (maximum): {max_count:,}")
    
    # 收集需要追加的重复细胞
    cells_to_append = []
    rng = np.random.default_rng(seed)
    
    # 对每个目标 cluster 进行重采样
    for cluster in target_clusters:
        if cluster not in cluster_counts_filtered:
            continue
        
        cluster_cells = train_perturbed_cells[train_perturbed_clusters == cluster]
        original_count = len(cluster_cells)
        
        if original_count < max_count:
            # 少数类：只追加差值部分（max_count - original_count）
            need = max_count - original_count
            additional_indices = rng.choice(len(cluster_cells), size=need, replace=True)
            additional_cells = cluster_cells[additional_indices]
            cells_to_append.extend(additional_cells.tolist())
            
            print(f"  {cluster}: {original_count:,} -> {max_count:,} (will append {need:,} cells)")
        else:
            # 多数类：不变，不需要追加
            print(f"  {cluster}: {original_count:,} (no resampling needed, majority class)")
    
    if len(cells_to_append) > 0:
        print(f"\nTotal cells to append: {len(cells_to_append):,}")
        return pd.Index(cells_to_append)
    else:
        return pd.Index([], dtype=train_cells.dtype)


def create_processed_split(
    h5ad_path: Path,
    output_h5ad_path: Path,
    seed: int = 42,
    cell_cluster_col: str = CELL_CLUSTER_COL,
    gene_pt_col: str = "gene_pt",
    drug_pt_col: str = "drug_pt",
    env_pt_col: str = "env_pt",
    control_col: str = "control",
    enable_resampling: bool = True,
    crispr_value: str = "",
) -> None:
    print(f"Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    obs = adata.obs
    print(f"  Shape: {adata.shape}")

    # 添加 CRISPR 列到 obs
    obs['CRISPR'] = crispr_value
    print(f"Added CRISPR column with value: '{crispr_value}'")

    # 新建 cell_cluster 列：优先选取 cell_line，如果没有就用 celltype
    if 'cell_line' in obs.columns:
        obs['cell_cluster'] = obs['cell_line']
        print(f"Using 'cell_line' column as 'cell_cluster'")
    elif 'celltype' in obs.columns:
        obs['cell_cluster'] = obs['celltype']
        print(f"Using 'celltype' column as 'cell_cluster'")
    else:
        raise ValueError("Neither 'cell_line' nor 'celltype' column found in adata.obs")

    # 检查必要列
    required_cols = [cell_cluster_col, gene_pt_col, drug_pt_col, env_pt_col]
    for col in required_cols:
        if col not in obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    clusters = obs[cell_cluster_col].astype(str)
    print("\ncell_cluster value counts:")
    print(clusters.value_counts())

    # 初始化所有细胞的 split 为空字符串
    splits = np.array([""] * adata.n_obs, dtype=object)

    # 1) MCF7 & A549 -> train
    mask_mcf7 = clusters == "MCF7"
    mask_a549 = clusters == "A549"
    idx_mcf7 = np.where(mask_mcf7.values)[0]
    idx_a549 = np.where(mask_a549.values)[0]
    splits[idx_mcf7] = "train"
    splits[idx_a549] = "train"
    print(f"\nMCF7 cells: {len(idx_mcf7):,} -> train")
    print(f"A549 cells: {len(idx_a549):,} -> train")

    # 2) K562 -> 在 K562 内部随机 1:1 分成 val/test
    mask_k562 = clusters == "K562"
    idx_k562 = np.where(mask_k562.values)[0]
    n_k562 = len(idx_k562)
    print(f"\nK562 cells: {n_k562:,}")

    if n_k562 > 0:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx_k562)
        n_val = n_k562 // 2
        val_idx = idx_k562[:n_val]
        test_idx = idx_k562[n_val:]
        splits[val_idx] = "val"
        splits[test_idx] = "test"
        print(f"K562 split -> val: {len(val_idx):,}, test: {len(test_idx):,}")
    else:
        print("[WARNING] No K562 cells found; val/test will be empty.")

    # 构建 split DataFrame
    split_df = pd.DataFrame({"split": splits}, index=obs.index)

    print("\nSplit counts before perturbation filtering:")
    for name, count in split_df["split"].value_counts(dropna=False).items():
        label = "(empty)" if name == "" else name
        print(f"  {label}: {count:,}")
    print(f"  TOTAL: {len(split_df):,} cells")

    # 3) perturbation 过滤：只保留 train 中出现过的 (gene_pt, drug_pt, env_pt)
    print("\n" + "=" * 60)
    print("Filtering val and test cells by perturbation combinations...")

    pert_keys = build_pert_key(obs, gene_pt_col, drug_pt_col, env_pt_col)

    # train perturbations
    mask_train = split_df["split"] == "train"
    train_cells = split_df.index[mask_train]
    train_pert_keys = pert_keys.loc[train_cells]
    train_perts = set(train_pert_keys.unique())

    print(f"Unique (gene_pt, drug_pt, env_pt) combinations in train: {len(train_perts)}")

    # 过滤 val/test
    mask_val = split_df["split"] == "val"
    mask_test = split_df["split"] == "test"

    val_cells = split_df.index[mask_val]
    test_cells = split_df.index[mask_test]

    val_pert_keys = pert_keys.loc[val_cells]
    test_pert_keys = pert_keys.loc[test_cells]

    val_keep_mask = val_pert_keys.isin(train_perts)
    test_keep_mask = test_pert_keys.isin(train_perts)

    n_val_before = int(mask_val.sum())
    n_test_before = int(mask_test.sum())

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

    # 更新 val/test cells（过滤后）
    mask_val_after = split_df["split"] == "val"
    mask_test_after = split_df["split"] == "test"
    val_cells_after = split_df.index[mask_val_after]
    test_cells_after = split_df.index[mask_test_after]

    # 4) 对 train 中的 perturbed cells 进行重采样
    train_cells_to_append = None
    if enable_resampling:
        print("\n" + "=" * 60)
        print("Resampling train perturbed cells to balance cell_cluster counts...")
        
        try:
            train_cells_to_append = resample_train_perturbed_cells(
                train_cells=train_cells,
                obs=obs,
                cell_cluster_col=cell_cluster_col,
                control_col=control_col,
                seed=seed,
            )
            
            print(f"\nAfter resampling:")
            print(f"  Original train cells: {len(train_cells):,}")
            print(f"  Cells to append: {len(train_cells_to_append):,}")
            
        except Exception as e:
            print(f"\n[WARNING] Resampling failed: {e}")
            print("  Continuing without resampling...")
            import traceback
            traceback.print_exc()
            train_cells_to_append = None

    print("\nFinal split counts (after perturbation filtering and resampling):")
    split_counts_after = split_df["split"].value_counts(dropna=False)
    for name, count in split_counts_after.items():
        label = "(empty)" if name == "" else name
        if name == "train" and train_cells_to_append is not None:
            # train 显示总数量（包括要追加的），而不是唯一细胞数
            total_train = count + len(train_cells_to_append)
            print(f"  {label}: {total_train:,} (unique: {count:,}, to append: {len(train_cells_to_append):,})")
        else:
            print(f"  {label}: {count:,}")
    print(f"  TOTAL: {len(split_df):,} cells (unique)")

    # 5) 在 adata.obs 中添加 split 列，过滤掉 split 为空的细胞，并根据采样次数复制细胞
    adata.obs["split"] = split_df["split"]
    
    # 只保留 split 不为空的细胞
    mask_valid_split = adata.obs["split"] != ""
    n_before_filter = adata.n_obs
    adata = adata[mask_valid_split].copy()
    n_after_filter = adata.n_obs
    n_removed = n_before_filter - n_after_filter
    
    print(f"\nFiltering cells with empty split:")
    print(f"  Before: {n_before_filter:,} cells")
    print(f"  After:  {n_after_filter:,} cells")
    print(f"  Removed: {n_removed:,} cells")
    
    # 6) 追加重采样的 train 细胞
    if train_cells_to_append is not None and len(train_cells_to_append) > 0:
        print("\n" + "=" * 60)
        print("Appending resampled train cells...")
        
        # 获取要追加的细胞数据
        cells_to_append_adata = adata[train_cells_to_append].copy()
        # 确保这些细胞的 split 标记为 "train"
        cells_to_append_adata.obs["split"] = "train"
        
        # 直接追加到原 adata 后面
        adata = ad.concat([adata, cells_to_append_adata], join="outer", index_unique=None)
        
        print(f"  Appended {len(train_cells_to_append):,} cells")
        print(f"  Total cells after appending: {adata.n_obs:,}")
    
    # 7) 确保观察值和变量名唯一
    print("\nMaking observation and variable names unique...")
    adata.obs_names_make_unique()
    
    # 8) 打印最终统计信息（包括复制的细胞）
    print("\n" + "=" * 60)
    print("Final split counts (after replication):")
    final_split_counts = adata.obs["split"].value_counts()
    for name, count in final_split_counts.items():
        print(f"  {name}: {count:,}")
    print(f"  TOTAL: {adata.n_obs:,} cells")
    
    output_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad_path)
    print(f"\nSaved h5ad file with 'split' column in obs to: {output_h5ad_path}")
    print(f"  Final shape: {adata.shape}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add 'split' and 'CRISPR' columns to obs in SrivatsanTrapnell2020_sciplex3_processed.h5ad:\n"
            "1) Add CRISPR column with specified value\n"
            "2) MCF7/A549 -> train, K562 -> val/test 1:1 within K562\n"
            "3) Remove val/test cells whose (gene_pt, drug_pt, env_pt) combination is unseen in train.\n"
            "4) Upsample train perturbed cells (minority class to majority class count)."
        )
    )
    parser.add_argument(
        "--h5ad-file",
        type=Path,
        default=DEFAULT_H5AD,
        help="Input processed h5ad file.",
    )
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output h5ad file with 'split' column in obs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for K562 val/test split and resampling (default: 42).",
    )
    parser.add_argument(
        "--disable-resampling",
        action="store_true",
        help="Disable resampling of train perturbed cells.",
    )
    parser.add_argument(
        "--crispr",
        type=str,
        default="",
        help="Value to set for the CRISPR column in obs (default: empty string).",
    )
    args = parser.parse_args()

    create_processed_split(
        h5ad_path=args.h5ad_file,
        output_h5ad_path=args.output_h5ad,
        seed=args.seed,
        enable_resampling=not args.disable_resampling,
        crispr_value=args.crispr,
    )


if __name__ == "__main__":
    main()

