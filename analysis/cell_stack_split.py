#!/usr/bin/env python3
"""
根据 control 细胞的 Stack embedding，对 cell_cluster 做无监督聚类，然后据此划分 train/val/test。

思路：
1. 读取 h5ad（需要包含：
   - obs.control: bool，标记 control 细胞
   - obs.cell_cluster: cell line / cluster 名称
2. 只用 control 细胞，使用 Stack 模型生成 cell embeddings。
3. 在每个 cell_cluster 内对 control 细胞的 embedding 取平均，得到一个「cell_cluster × embedding_dim」的矩阵。
4. 在 embedding 空间上对 cell_cluster 做 K-means 聚类，得到若干个「meta-cluster」。
5. 随机选择其中一部分 meta-cluster 作为 train，其余作为 holdout 组。
6. 对 holdout 组中的所有细胞在细胞级别做 1:1 切分，前一半标记为 val，后一半标记为 test。
   这样：
   - train: 来自若干 meta-cluster
   - val/test: 来自同一批 holdout meta-cluster，但具体细胞不重叠
7. 将分组结果写入 adata.obs['split']，并保存新的 h5ad 文件（或覆盖原文件）。

示例：
    python cell_stack_split.py \\
        --h5ad-file tasks/unseen_cells/zeroshot/McFarlandTsherniak2020_stack.h5ad \\
        --checkpoint data/stack/bc_large.ckpt \\
        --genelist data/stack/basecount_1000per_15000max.pkl \\
        --output-h5ad tasks/unseen_cells/zeroshot/McFarlandTsherniak2020_stack_split.h5ad \\
        --n-meta-clusters 10 \\
        --train-cluster-ratio 0.8
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

def add_organism_column(adata, organism: str = "Homo sapiens"):
    """为 adata 添加 'organism' 列（Stack 模型需要）。"""
    if "organism" in adata.obs.columns:
        existing_org = adata.obs["organism"].unique()
        if len(existing_org) == 1 and existing_org[0] == organism:
            return adata
        else:
            print(f"⚠ Warning: 'organism' column exists with values: {existing_org}, overwriting with '{organism}'")
    
    adata.obs["organism"] = organism
    return adata


def run_stack_embedding_command(
    checkpoint: str,
    adata_path: str,
    genelist: str,
    output_path: str,
    batch_size: int = 32,
    gene_name_col: str | None = None,
) -> None:
    """运行 stack-embedding 命令生成 embeddings。"""
    cmd = [
        "stack-embedding",
        "--checkpoint", str(checkpoint),
        "--adata", str(adata_path),
        "--genelist", str(genelist),
        "--output", str(output_path),
        "--batch-size", str(batch_size),
    ]
    
    if gene_name_col is not None:
        cmd.extend(["--gene-name-col", gene_name_col])
    
    print(f"Running stack-embedding command:")
    print(f"  {' '.join(cmd)}")
    
    result = subprocess.run(cmd, check=True, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"stack-embedding failed with exit code {result.returncode}")


def compute_cellcluster_embeddings(
    adata,
    checkpoint: str,
    genelist: str,
    control_col: str = "control",
    cell_cluster_col: str = "cell_cluster",
    batch_size: int = 32,
    gene_name_col: str | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    只使用 control 细胞，使用 Stack 模型生成 embeddings，然后按 cell_cluster 求平均。

    Returns
    -------
    means: np.ndarray, shape (n_clusters, embedding_dim)
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

    # 提取 control 细胞
    adata_control = adata[control_mask].copy()

    # 确保有 organism 列（Stack 模型需要）
    adata_control = add_organism_column(adata_control, organism="Homo sapiens")

    # 使用临时文件保存 control 细胞的 adata
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
        temp_adata_path = tmp_file.name
        adata_control.write(temp_adata_path)

    # 使用临时文件保存 embeddings
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_emb:
        temp_emb_path = tmp_emb.name

    try:
        # 生成 Stack embeddings
        print(f"\nGenerating Stack embeddings for control cells...")
        print(f"  Checkpoint: {checkpoint}")
        print(f"  Genelist: {genelist}")
        print(f"  Batch size: {batch_size}")

        run_stack_embedding_command(
            checkpoint=checkpoint,
            adata_path=temp_adata_path,
            genelist=genelist,
            output_path=temp_emb_path,
            batch_size=batch_size,
            gene_name_col=gene_name_col,
        )

        # 加载 embeddings
        embeddings = np.load(temp_emb_path)
        print(f"  Embeddings shape: {embeddings.shape}")

        # 验证形状匹配
        if embeddings.shape[0] != adata_control.n_obs:
            raise ValueError(
                f"Embeddings shape {embeddings.shape[0]} doesn't match control cells {adata_control.n_obs}"
            )

        # 按 cell_cluster 求平均 embedding
        means = []
        cluster_names: List[str] = []

        for cl in unique_clusters:
            mask_cl = (cell_cluster_control == cl).values
            if mask_cl.sum() == 0:
                continue
            mean_emb = embeddings[mask_cl].mean(axis=0)
            means.append(mean_emb)
            cluster_names.append(cl)

        means = np.vstack(means)
        print(f"Shape of cluster mean embeddings: {means.shape} (n_clusters, embedding_dim)")

    finally:
        # 清理临时文件
        import os
        if os.path.exists(temp_adata_path):
            os.unlink(temp_adata_path)
        if os.path.exists(temp_emb_path):
            os.unlink(temp_emb_path)

    return means, cluster_names


def cluster_cellclusters(
    embeddings: np.ndarray,
    n_meta_clusters: int = 10,
    seed: int = 42,
    method: str = "average",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 cell_cluster 的平均 embeddings 做层次聚类，返回每个 cell_cluster 的 meta-cluster 标号。
    
    使用余弦距离（1 - cosine similarity）进行聚类。
    
    Parameters
    ----------
    embeddings : np.ndarray
        每个 cell_cluster 的平均 embedding，shape (n_clusters, embedding_dim)
    n_meta_clusters : int
        目标 meta-cluster 数量
    seed : int
        随机种子（用于一致性，但层次聚类本身是确定性的）
    method : str
        层次聚类方法：'ward', 'complete', 'average', 'single' 等
    
    Returns
    -------
    meta_labels : np.ndarray
        每个 cell_cluster 的 meta-cluster 标签
    linkage_matrix : np.ndarray
        层次聚类的 linkage 矩阵，用于绘制 dendrogram
    """
    n_clusters, embedding_dim = embeddings.shape

    if n_meta_clusters > n_clusters:
        raise ValueError(
            f"n_meta_clusters={n_meta_clusters} 大于 cell_cluster 数量 n_clusters={n_clusters}，请减小 n_meta_clusters。"
        )

    print(f"Running hierarchical clustering with cosine distance...")
    print(f"  Input embedding shape: {embeddings.shape}")
    print(f"  Target n_meta_clusters: {n_meta_clusters}")
    print(f"  Linkage method: {method}")
    
    # 进行层次聚类（直接使用余弦距离）
    # linkage 函数会自动计算余弦距离（1 - cosine similarity）
    linkage_matrix = linkage(embeddings, method=method, metric='cosine')
    
    # 根据目标聚类数切割树
    meta_labels = fcluster(linkage_matrix, n_meta_clusters, criterion='maxclust')
    # fcluster 返回的标签从1开始，转换为从0开始
    meta_labels = meta_labels - 1
    
    print("Meta-cluster counts:")
    _, counts = np.unique(meta_labels, return_counts=True)
    for k, c in enumerate(counts):
        print(f"  meta_cluster {k}: {c} cell_clusters")

    return meta_labels, linkage_matrix


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    cluster_names: List[str],
    out_path: Path,
    n_meta_clusters: int = 10,
) -> None:
    """
    绘制层次聚类的 dendrogram（树状图）。
    
    Parameters
    ----------
    linkage_matrix : np.ndarray
        层次聚类的 linkage 矩阵
    cluster_names : List[str]
        每个 cell_cluster 的名称
    out_path : Path
        输出图片路径
    n_meta_clusters : int
        meta-cluster 数量，用于在图上标记切割点
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # 绘制 dendrogram
    dendrogram(
        linkage_matrix,
        labels=cluster_names,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=None,
    )
    
    # 计算切割阈值（使得有 n_meta_clusters 个簇）
    # 找到第 (n_clusters - n_meta_clusters) 个合并的距离
    n_clusters = len(cluster_names)
    if n_meta_clusters < n_clusters:
        # 获取所有合并距离
        distances = linkage_matrix[:, 2]
        # 排序后取第 (n_clusters - n_meta_clusters) 个距离作为阈值
        sorted_distances = np.sort(distances)
        threshold_idx = n_clusters - n_meta_clusters
        if threshold_idx >= 0 and threshold_idx < len(sorted_distances):
            threshold = sorted_distances[threshold_idx]
            plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                       label=f'Cut at {n_meta_clusters} clusters (threshold={threshold:.4f})')
    
    plt.xlabel('Cell Cluster', fontsize=12)
    plt.ylabel('Cosine Distance', fontsize=12)
    plt.title(f'Hierarchical Clustering Dendrogram (Cosine Distance)\nTarget: {n_meta_clusters} meta-clusters', 
              fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved dendrogram to: {out_path}")


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


def run(
    h5ad_path: Path,
    checkpoint: str,
    genelist: str,
    output_h5ad: Path | None = None,
    n_meta_clusters: int = 10,
    train_cluster_ratio: float = 0.8,
    seed: int = 42,
    control_col: str = "control",
    cell_cluster_col: str = "cell_cluster",
    batch_size: int = 32,
    gene_name_col: str | None = None,
) -> None:
    """
    在 h5ad 中根据 control 细胞的 Stack embedding 聚类结果生成 split，
    并将结果写入 adata.obs['split']，最后保存到 output_h5ad（若为 None 则覆盖原文件）。
    """
    print(f"Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    # 生成 Stack embeddings 并按 cell_cluster 求平均
    means, cluster_names = compute_cellcluster_embeddings(
        adata,
        checkpoint=checkpoint,
        genelist=genelist,
        control_col=control_col,
        cell_cluster_col=cell_cluster_col,
        batch_size=batch_size,
        gene_name_col=gene_name_col,
    )

    # 对平均 embeddings 做层次聚类（使用余弦距离）
    meta_labels, linkage_matrix = cluster_cellclusters(
        means,
        n_meta_clusters=n_meta_clusters,
        seed=seed,
    )

    # 为每个 cell_cluster 以及每个 cell 生成 meta_cluster 标注
    cluster_to_meta: Dict[str, int] = {
        str(cl): int(m) for cl, m in zip(cluster_names, meta_labels)
    }
    cell_cluster_str = adata.obs[cell_cluster_col].astype(str)
    adata.obs["meta_cluster"] = cell_cluster_str.map(cluster_to_meta)
    print("Added obs['meta_cluster'] based on hierarchical clustering (cosine distance) over control cell_cluster mean embeddings.")

    # 绘制 dendrogram
    dendrogram_path = output_h5ad.parent / f"{output_h5ad.stem}_dendrogram.png" if output_h5ad else h5ad_path.parent / f"{h5ad_path.stem}_dendrogram.png"
    plot_dendrogram(linkage_matrix, cluster_names, dendrogram_path, n_meta_clusters=n_meta_clusters)

    # 根据 meta-cluster 划分 train/val/test
    split = build_split_from_meta_clusters(
        adata,
        cluster_names=cluster_names,
        meta_labels=meta_labels,
        train_cluster_ratio=train_cluster_ratio,
        seed=seed,
        cell_cluster_col=cell_cluster_col,
    )

    # 写入 obs.split
    adata.obs["split"] = split

    # 确定输出路径：未指定则覆盖原文件
    if output_h5ad is None:
        output_h5ad = h5ad_path

    output_h5ad = Path(output_h5ad)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad)
    print(f"\nSaved h5ad with obs['split'] to: {output_h5ad}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use Stack embeddings of control cells to cluster cell_cluster in an unsupervised way, "
            "then create train/val/test split based on these clusters."
        )
    )
    parser.add_argument(
        "--h5ad-file",
        type=Path,
        required=True,
        help="Input h5ad file path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Stack model checkpoint file (e.g., bc_large.ckpt).",
    )
    parser.add_argument(
        "--genelist",
        type=str,
        required=True,
        help="Path to genelist file (e.g., basecount_1000per_15000max.pkl).",
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
        help="Number of meta-clusters for hierarchical clustering over cell_cluster mean embeddings (default: 10).",
    )
    parser.add_argument(
        "--train-cluster-ratio",
        type=parse_ratio,
        default=0.8,
        help="Ratio of meta-clusters assigned to train. Supports: float (0.75), fraction (6/8), or ratio (6:2). Default: 0.8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42). Note: hierarchical clustering is deterministic.",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for Stack embedding generation (default: 32).",
    )
    parser.add_argument(
        "--gene-name-col",
        type=str,
        default=None,
        help="Gene name column in adata.var (e.g., 'feature_name').",
    )

    args = parser.parse_args()

    run(
        h5ad_path=args.h5ad_file,
        checkpoint=args.checkpoint,
        genelist=args.genelist,
        output_h5ad=args.output_h5ad,
        n_meta_clusters=args.n_meta_clusters,
        train_cluster_ratio=args.train_cluster_ratio,
        seed=args.seed,
        control_col=args.control_col,
        cell_cluster_col=args.cell_cluster_col,
        batch_size=args.batch_size,
        gene_name_col=args.gene_name_col,
    )


if __name__ == "__main__":
    main()

