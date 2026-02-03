#!/usr/bin/env python3
"""
根据 control 细胞的表达，对 cell_cluster 做无监督聚类，然后据此划分 train/val/test。

思路：
1. 读取 h5ad（需要包含：
   - obs.control: bool，标记 control 细胞
   - obs.cell_cluster: cell line / cluster 名称
2. 只用 control 细胞，在每个 cell_cluster 内对表达取平均，得到一个「cell_cluster × genes」的表达矩阵。
3. 对上述矩阵做 PCA（降低维度）。
4. 在 PCA 空间上对 cell_cluster 做 K-means 聚类，得到若干个「meta-cluster」。
5. 随机选择其中一部分 meta-cluster 作为 train，其余作为 holdout 组。
6. 对 holdout 组中的所有细胞（不区分 control / 非 control）在细胞级别做 1:1 切分，前一半标记为 val，后一半标记为 test。
   这样：
   - train: 来自若干 meta-cluster
   - val/test: 来自同一批 holdout meta-cluster，但具体细胞不重叠
7. 将分组结果写入 adata.obs['split']，并保存新的 h5ad 文件（或覆盖原文件）。

示例默认输入：
/fs-computility-new/upzd_share/shared/AIVC_data/processed_control/processed/McFarlandTsherniak2020_filtered_processed.h5ad
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


DEFAULT_H5AD = Path(
    "/fs-computility-new/upzd_share/shared/AIVC_data/processed_control/processed/McFarlandTsherniak2020_filtered_processed.h5ad"
)


def compute_cellcluster_means(
    adata,
    control_col: str = "control",
    cell_cluster_col: str = "cell_cluster",
) -> Tuple[np.ndarray, List[str]]:
    """
    只使用 control 细胞，按 cell_cluster 求表达均值。

    Returns
    -------
    means: np.ndarray, shape (n_clusters, n_genes)
    cluster_names: List[str], 对应的 cell_cluster 名称
    """
    obs = adata.obs

    if control_col not in obs.columns:
        raise ValueError(f"obs 中未找到列 '{control_col}'")
    if cell_cluster_col not in obs.columns:
        raise ValueError(f"obs 中未找到列 '{cell_cluster_col}'")

    control_series = obs[control_col]
    # 兼容 bool / 字符串
    if control_series.dtype == bool:
        control_mask = control_series.values
    else:
        control_mask = control_series.astype(str).str.lower().isin(["true", "1", "yes", "y"])

    n_control = int(control_mask.sum())
    if n_control == 0:
        raise ValueError("没有找到任何 control==True 的细胞，无法进行聚类。")

    print(f"Found {n_control:,} control cells.")

    cell_cluster_control = obs.loc[control_mask, cell_cluster_col].astype(str)
    unique_clusters = sorted(cell_cluster_control.unique())
    n_clusters = len(unique_clusters)

    print(f"Found {n_clusters} unique '{cell_cluster_col}' among control cells:")
    print(f"  {unique_clusters}")

    X = adata.X
    if issparse(X):
        X = X.tocsr()

    X_control = X[control_mask]

    means = []
    cluster_names: List[str] = []

    for cl in unique_clusters:
        mask_cl = (cell_cluster_control == cl).values
        if mask_cl.sum() == 0:
            continue
        if issparse(X_control):
            mean_vec = np.asarray(X_control[mask_cl].mean(axis=0)).ravel()
        else:
            mean_vec = X_control[mask_cl].mean(axis=0)
        means.append(mean_vec)
        cluster_names.append(cl)

    means = np.vstack(means)
    print(f"Shape of cluster means: {means.shape} (n_clusters, n_genes)")
    return means, cluster_names


def cluster_cellclusters(
    means: np.ndarray,
    n_meta_clusters: int = 10,
    n_pcs: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对 cell_cluster 的均值表达做 PCA + KMeans，返回每个 cell_cluster 的 meta-cluster 标号。
    """
    n_clusters, n_genes = means.shape
    n_pcs_eff = min(n_pcs, n_clusters - 1, n_genes)
    if n_pcs_eff <= 0:
        raise ValueError(
            f"有效 PCA 维度为 0（n_clusters={n_clusters}, n_genes={n_genes}, n_pcs={n_pcs}），无法进行 PCA。"
        )

    print(f"Running PCA with n_components={n_pcs_eff} ...")
    pca = PCA(n_components=n_pcs_eff, random_state=seed)
    Z = pca.fit_transform(means)
    print(f"  PCA output shape: {Z.shape}")

    if n_meta_clusters > n_clusters:
        raise ValueError(
            f"n_meta_clusters={n_meta_clusters} 大于 cell_cluster 数量 n_clusters={n_clusters}，请减小 n_meta_clusters。"
        )

    print(f"Running KMeans with n_clusters={n_meta_clusters}, random_state={seed} ...")
    km = KMeans(n_clusters=n_meta_clusters, random_state=seed, n_init="auto")
    meta_labels = km.fit_predict(Z)
    print("Meta-cluster counts:")
    _, counts = np.unique(meta_labels, return_counts=True)
    for k, c in enumerate(counts):
        print(f"  meta_cluster {k}: {c} cell_clusters")

    # 返回 KMeans 标签和 PCA 坐标（用于可视化）
    return meta_labels, Z


def plot_meta_clusters(
    Z: np.ndarray,
    meta_labels: np.ndarray,
    cluster_names: List[str],
    out_path: Path,
) -> None:
    """
    在 PCA 空间中可视化每个 cell_cluster 的 meta-cluster 结果（前两主成分）。
    """
    if Z.shape[1] < 2:
        print("PCA 维度小于 2，无法绘制二维散点图，跳过可视化。")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=meta_labels,
        cmap="tab10",
        s=60,
        edgecolors="k",
        alpha=0.8,
    )

    # 标注每个点的 cell_cluster 名称
    for (x, y, name) in zip(Z[:, 0], Z[:, 1], cluster_names):
        plt.text(x, y, str(name), fontsize=8, ha="center", va="center")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("KMeans over control cell_cluster means (PCA space)")
    plt.colorbar(scatter, label="meta-cluster id")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved KMeans meta-cluster visualization to: {out_path}")


def build_split_from_meta_clusters(
    adata,
    cluster_names: List[str],
    meta_labels: np.ndarray,
    train_cluster_ratio: float = 0.8,
    seed: int = 42,
    cell_cluster_col: str = "cell_cluster",
) -> pd.Series:
    """
    根据 meta-cluster 划分 train / holdout，再对 holdout 细胞做 1:1 划分为 val / test。
    """
    rng = np.random.default_rng(seed)

    n_meta = int(meta_labels.max()) + 1
    all_meta_ids = np.arange(n_meta)

    n_train_meta = int(round(n_meta * train_cluster_ratio))
    n_train_meta = max(1, min(n_meta - 1, n_train_meta))  # 至少 1 个 train，至少 1 个 holdout

    shuffled_meta = all_meta_ids.copy()
    rng.shuffle(shuffled_meta)

    train_meta = shuffled_meta[:n_train_meta]
    holdout_meta = shuffled_meta[n_train_meta:]

    print(f"\nMeta-cluster split (by cluster-level groups, seed={seed}):")
    print(f"  train_meta:  {sorted(train_meta.tolist())}")
    print(f"  holdout_meta: {sorted(holdout_meta.tolist())}")

    # 建立 cell_cluster -> meta_cluster 的映射
    if len(cluster_names) != len(meta_labels):
        raise ValueError("cluster_names 与 meta_labels 长度不一致。")
    cluster_to_meta: Dict[str, int] = {cl: int(m) for cl, m in zip(cluster_names, meta_labels)}

    obs = adata.obs
    if cell_cluster_col not in obs.columns:
        raise ValueError(f"obs 中未找到列 '{cell_cluster_col}'")

    cell_clusters_all = obs[cell_cluster_col].astype(str)

    # 先标记 train / holdout 组
    group = []
    unknown_clusters = set()
    for cl in cell_clusters_all:
        m = cluster_to_meta.get(cl, None)
        if m is None:
            unknown_clusters.add(cl)
            group.append("unknown")
        elif m in train_meta:
            group.append("train")
        else:
            group.append("holdout")

    if unknown_clusters:
        print("\n[WARNING] 以下 cell_cluster 未被分配 meta-cluster，将被标记为 'unknown'，不参与训练/验证/测试：")
        print(f"  {sorted(unknown_clusters)}")

    group = pd.Series(group, index=obs.index, name="group")

    # 在细胞级别上，把 holdout 组拆成 val/test（各占一半）
    split = pd.Series("", index=obs.index, name="split", dtype="object")

    # train 组：全部标记为 train
    mask_train = group == "train"
    split.loc[mask_train] = "train"

    # holdout 组：随机 1:1 划分为 val/test
    mask_holdout = group == "holdout"
    holdout_indices = split.index[mask_holdout].to_numpy()
    n_holdout = len(holdout_indices)
    print(f"\nTotal holdout cells (to be split into val/test): {n_holdout:,}")

    if n_holdout > 0:
        shuffled_idx = holdout_indices.copy()
        rng.shuffle(shuffled_idx)
        mid = n_holdout // 2
        val_cells = shuffled_idx[:mid]
        test_cells = shuffled_idx[mid:]
        split.loc[val_cells] = "val"
        split.loc[test_cells] = "test"
        print(f"  val cells:  {len(val_cells):,}")
        print(f"  test cells: {len(test_cells):,}")

    print("\nFinal split counts:")
    counts = split.value_counts(dropna=False)
    for k, v in counts.items():
        label = "(empty)" if k == "" else k
        print(f"  {label}: {v:,}")
    print(f"  TOTAL: {len(split):,} cells")

    return split


def run(
    h5ad_path: Path,
    output_h5ad: Path | None = None,
    plot_path: Path | None = None,
    n_meta_clusters: int = 10,
    n_pcs: int = 10,
    train_cluster_ratio: float = 0.8,
    seed: int = 42,
    control_col: str = "control",
    cell_cluster_col: str = "cell_cluster",
) -> None:
    """
    在 h5ad 中根据 control 细胞的 cell_cluster 聚类结果生成 split，
    并将结果写入 adata.obs['split']，最后保存到 output_h5ad（若为 None 则覆盖原文件）。
    """
    print(f"Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    means, cluster_names = compute_cellcluster_means(
        adata, control_col=control_col, cell_cluster_col=cell_cluster_col
    )
    meta_labels, Z = cluster_cellclusters(
        means,
        n_meta_clusters=n_meta_clusters,
        n_pcs=n_pcs,
        seed=seed,
    )

    # 为每个 cell_cluster 以及每个 cell 生成 meta_cluster 标注，方便下游脚本（例如 split/control_meta_cluster_split.py）使用
    cluster_to_meta: Dict[str, int] = {
        str(cl): int(m) for cl, m in zip(cluster_names, meta_labels)
    }
    cell_cluster_str = adata.obs[cell_cluster_col].astype(str)
    adata.obs["meta_cluster"] = cell_cluster_str.map(cluster_to_meta)
    print("Added obs['meta_cluster'] based on KMeans meta-clusters over control cell_cluster means.")
    split = build_split_from_meta_clusters(
        adata,
        cluster_names=cluster_names,
        meta_labels=meta_labels,
        train_cluster_ratio=train_cluster_ratio,
        seed=seed,
        cell_cluster_col=cell_cluster_col,
    )

    # 可选：可视化 meta-cluster 结果
    if plot_path is not None:
        plot_meta_clusters(Z, meta_labels, cluster_names, out_path=plot_path)

    # 写入 obs.split
    adata.obs["split"] = split

    # 确定输出路径：未指定则覆盖原文件
    if output_h5ad is None:
        output_h5ad = h5ad_path

    output_h5ad = Path(output_h5ad)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad)
    print(f"\nSaved h5ad with obs['split'] to: {output_h5ad}")


def parse_ratio(value: str) -> float:
    """
    解析比例值，支持以下格式：
    - 浮点数：'0.75' -> 0.75
    - 分数：'6/8' -> 0.75
    - 比例：'6:2' -> 0.75 (6/(6+2))
    """
    value = value.strip()
    
    # 尝试直接解析为浮点数
    try:
        return float(value)
    except ValueError:
        pass
    
    # 尝试解析分数形式 "a/b"
    if '/' in value:
        parts = value.split('/')
        if len(parts) == 2:
            try:
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator == 0:
                    raise ValueError(f"分母不能为 0: {value}")
                return numerator / denominator
            except ValueError as e:
                raise argparse.ArgumentTypeError(f"无效的分数格式 '{value}': {e}")
    
    # 尝试解析比例形式 "a:b"
    if ':' in value:
        parts = value.split(':')
        if len(parts) == 2:
            try:
                train_part = float(parts[0].strip())
                holdout_part = float(parts[1].strip())
                total = train_part + holdout_part
                if total == 0:
                    raise ValueError(f"比例总和不能为 0: {value}")
                return train_part / total
            except ValueError as e:
                raise argparse.ArgumentTypeError(f"无效的比例格式 '{value}': {e}")
    
    raise argparse.ArgumentTypeError(f"无法解析比例值 '{value}'，请使用浮点数（如 0.75）、分数（如 6/8）或比例（如 6:2）")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use control cells to cluster cell_cluster in an unsupervised way, "
            "then create train/val/test split based on these clusters."
        )
    )
    parser.add_argument(
        "--h5ad-file",
        type=Path,
        default=DEFAULT_H5AD,
        help="Input h5ad file path.",
    )
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=None,
        help="Output h5ad path with obs['split']. If not set, overwrite input file.",
    )
    parser.add_argument(
        "--n-meta-clusters",
        type=int,
        default=10,
        help="Number of meta-clusters for KMeans over cell_cluster means (default: 10, 即 8:2 集群比例).",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=10,
        help="Number of principal components for PCA (default: 10).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional path to save a PNG visualization of meta-clusters in PCA space.",
    )
    parser.add_argument(
        "--train-cluster-ratio",
        type=parse_ratio,
        default=0.8,
        help="Ratio of meta-clusters assigned to train. Supports: float (0.75), fraction (6/8), or ratio (6:2). Default: 0.8 (即 8:2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PCA/KMeans and splitting (default: 42).",
    )
    parser.add_argument(
        "--control-col",
        type=str,
        default="control",
        help="Column name in obs for control indicator (default: 'control').",
    )
    parser.add_argument(
        "--cell-cluster-col",
        type=str,
        default="cell_cluster",
        help="Column name in obs for cell line / cell cluster (default: 'cell_cluster').",
    )

    args = parser.parse_args()

    run(
        h5ad_path=args.h5ad_file,
        output_h5ad=args.output_h5ad,
        plot_path=args.plot_path,
        n_meta_clusters=args.n_meta_clusters,
        n_pcs=args.n_pcs,
        train_cluster_ratio=args.train_cluster_ratio,
        seed=args.seed,
        control_col=args.control_col,
        cell_cluster_col=args.cell_cluster_col,
    )


if __name__ == "__main__":
    main()


