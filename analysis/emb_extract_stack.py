"""
使用 Stack 模型抽取 cell embeddings 并做 UMAP 可视化。
"""

import os
from pathlib import Path

import numpy as np
import scanpy as sc
import anndata as ad


def extract_stack_embeddings(
    adata: ad.AnnData,
    model_dir: str,
    checkpoint_name: str = "bc_large.ckpt",
    genelist_name: str = "basecount_1000per_15000max.pkl",
    gene_name_col: str | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> np.ndarray:
    from stack.model_loading import load_model_from_checkpoint

    model_dir = Path(model_dir)
    checkpoint_path = model_dir / checkpoint_name
    genelist_path = model_dir / genelist_name

    for path, msg in [
        (model_dir, "Model directory not found"),
        (checkpoint_path, "Checkpoint file not found"),
        (genelist_path, "Genelist file not found"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{msg}: {path}")

    print(f"Using AnnData for embedding (in-memory, will write temp file)...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Genelist:   {genelist_path}")
    print(f"Gene name column: {gene_name_col}")
    print(f"Batch size: {batch_size}, num_workers: {num_workers}")

    print(f"\nLoading Stack model from: {checkpoint_path}")
    model = load_model_from_checkpoint(str(checkpoint_path))

    # 写临时 h5ad，再按照 notebook 的方式通过路径调用
    tmp_path = Path("atlas_merged_immune_ordered_tmp.h5ad")
    print(f"Writing temporary AnnData to: {tmp_path}")
    adata.write(tmp_path)

    print("\nExtracting embeddings with Stack (notebook-style, using adata_path)...")
    embeddings, _ = model.get_latent_representation(
        adata_path=str(tmp_path),
        genelist_path=str(genelist_path),
        gene_name_col=gene_name_col,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # 清理临时文件
    if tmp_path.exists():
        print(f"Removing temporary file: {tmp_path}")
        tmp_path.unlink()

    return embeddings


def visualize_embeddings_with_umap(
    adata,
    embeddings: np.ndarray,
    embed_key: str = "stack-embed",
    color_keys: list[str] | None = None,
    save_dir: str | Path | None = None,
) -> None:

    if embeddings.shape[0] != adata.n_obs:
        raise ValueError(f"Embeddings has {embeddings.shape[0]} cells, but adata has {adata.n_obs}.")

    adata.obsm[embed_key] = embeddings

    print(f"Embeddings shape: {embeddings.shape}")
    # adata = sc.pp.subsample(adata, fraction=0.05, copy=True)

    print("Computing neighbors & UMAP...")
    sc.pp.neighbors(adata, use_rep=embed_key)
    sc.tl.umap(adata)

    if save_dir is None:
        save_dir = Path(".") / "stack_embed_figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for key in color_keys:
        if key not in adata.obs.columns:
            print(f"Skip color '{key}' (not in adata.obs).")
            continue

        adata.obs[key] = adata.obs[key].astype("category")
        fig = sc.pl.umap(
            adata,
            color=[key],
            frameon=False,
            wspace=0.4,
            title=f"Stack_{key}",
            return_fig=True,
        )
        out_path = save_dir / f"immune_stack_{key}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"Saved UMAP colored by '{key}' to: {out_path}")


def main():
    """
    示例入口：根据你自己的路径修改 base_path / adata / model_dir / obs 列名等。
    """
    base_path = "/mnt/shared-storage-gpfs2/beam-gpfs02/zhangsongming"

    # === 路径配置（按需要修改）===
    # 原始（未排序）的 AnnData
    adata_path = f"{base_path}/AIVC-TME-immune/atlas_merged_immune.h5ad"
    model_dir = f"{base_path}/weights/Stack-Large"
    checkpoint_name = "bc_large.ckpt"
    genelist_name = "basecount_1000per_15000max.pkl"

    # 1) 按 notebook 的方式，先读入并按 donor / Sample 排序后写出 *_ordered.h5ad
    donor_col = "Sample"  # 对应 tutorial 里的 'donor_id'，根据自己数据调整

    print(f"Reading raw AnnData from: {adata_path}")
    adata_raw = sc.read_h5ad(adata_path)
    if donor_col not in adata_raw.obs.columns:
        raise KeyError(f"'{donor_col}' not found in adata.obs; available columns: {list(adata_raw.obs.columns)}")
    order = np.argsort(adata_raw.obs[donor_col].values, kind="stable")
    adata_ordered = adata_raw[order, :].copy()

    # ordered_adata_path = Path(adata_path).with_name(f"{Path(adata_path).stem}_ordered.h5ad")
    # print(f"Writing ordered AnnData to: {ordered_adata_path}")
    # adata_ordered.write(ordered_adata_path)

    # 2) 抽取 embeddings（与 notebook 中的 get_latent_representation 调用一致）
    embeddings = extract_stack_embeddings(
        adata=adata_ordered,
        model_dir=model_dir,
        checkpoint_name=checkpoint_name,
        genelist_name=genelist_name,
        # gene_name_col="feature_name",  # 与 tutorial-embed.ipynb 中保持一致
        batch_size=64,
        num_workers=4,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    adata_ordered.obsm["stack_embed"] = embeddings
    adata_ordered.write(f"{base_path}/AIVC-TME-immune/immune_ordered_stack_embed.h5ad")
    print(f"Saved ordered AnnData with embeddings!")

    # 3) UMAP + 可视化（这里默认画几个常用列，你可以按需要改）
    # print(f"Visualizing embeddings with UMAP...")
    # color_keys = ["Celltype", "Tissue", "Cancer type", "cnv_status", "Organ_origin"]
    # save_dir = f"{base_path}/AIVC-TME-immune/stack_embed/figures"
    # visualize_embeddings_with_umap(
    #     adata=adata_ordered,
    #     embeddings=embeddings,
    #     embed_key="stack-embed",
    #     color_keys=color_keys,
    #     save_dir=save_dir,
    # )


if __name__ == "__main__":
    main()

