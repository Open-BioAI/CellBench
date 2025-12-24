import scanpy as sc
import numpy as np

path = '/fs-computility-new/upzd_share/shared/AIVC_data/perturbench-main/check/all_cell_line_filterdrug_unseen_subset.h5ad'
adata = sc.read_h5ad(path)

# 检查 adata.X 是稀疏矩阵还是稠密矩阵
if hasattr(adata.X, "toarray"):  # 稀疏矩阵
    X = adata.X.toarray()
else:
    X = adata.X

# 计算每个细胞的非零基因数
nonzero_per_cell = (X != 0).sum(axis=1)

# 找出全为零的细胞
zero_cells = np.where(nonzero_per_cell == 0)[0]
print(f"🔎 全零细胞数: {len(zero_cells)} / {adata.n_obs}")

if len(zero_cells) > 0:
    print("这些细胞的索引示例：", zero_cells[:10])
    print("对应的 obs 行：")
    print(adata.obs.iloc[zero_cells[:5]])
else:
    print("✅ 没有全零细胞，一切正常。")
