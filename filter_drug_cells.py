#!/usr/bin/env python3
"""
筛选所有带有 drug_pt 的细胞，保存到 onlydrug.h5ad
"""
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

# 输入和输出路径
input_path = Path("/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/all_cell_line_filterdrug.h5ad")
output_path = Path("/fs-computility-new/upzd_share/maoxinjie/AIVC/mxj/perturbench-main/data/onlydrug.h5ad")

print(f"正在读取 h5ad 文件: {input_path}")
adata = sc.read_h5ad(input_path)
print(f"  原始数据形状: {adata.shape} (cells × genes)")
print(f"  obs 列数: {len(adata.obs.columns)}")

# 检查 drug_pt 列是否存在
if 'drug_pt' not in adata.obs.columns:
    raise ValueError("'drug_pt' 列不存在于 adata.obs 中")

# 获取 drug_pt 列
drug_pt = adata.obs['drug_pt']

# 检查 drug_pt 的数据类型和值
print(f"\n'drug_pt' 列信息:")
print(f"  数据类型: {drug_pt.dtype}")
print(f"  唯一值数量: {drug_pt.nunique()}")
print(f"  前10个唯一值: {drug_pt.unique()[:10]}")

# 筛选条件：drug_pt 不为空且不是 control
# 处理可能的 Categorical 类型
if pd.api.types.is_categorical_dtype(drug_pt):
    drug_pt_str = drug_pt.astype(str)
else:
    drug_pt_str = drug_pt.astype(str)

# 筛选：drug_pt 不为空字符串、不为 'nan'、不为 'control'、不为空值
mask = (
    (drug_pt_str != '') & 
    (drug_pt_str != 'nan') & 
    (drug_pt_str != 'NaN') & 
    (drug_pt_str != 'control') &
    (drug_pt_str.notna())
)

# 统计
n_total = len(adata)
n_filtered = mask.sum()
n_dropped = n_total - n_filtered

print(f"\n筛选结果:")
print(f"  总细胞数: {n_total:,}")
print(f"  带有 drug_pt 的细胞数: {n_filtered:,}")
print(f"  被过滤掉的细胞数: {n_dropped:,}")
print(f"  保留比例: {n_filtered/n_total*100:.2f}%")

if n_filtered == 0:
    raise ValueError("没有找到任何带有 drug_pt 的细胞！请检查数据。")

# 应用筛选
adata_filtered = adata[mask].copy()

print(f"\n筛选后的数据形状: {adata_filtered.shape} (cells × genes)")

# 显示 drug_pt 的分布
print(f"\n筛选后 'drug_pt' 的值分布（前20个）:")
print(adata_filtered.obs['drug_pt'].value_counts().head(20))

# 保存到新文件
print(f"\n正在保存到: {output_path}")
adata_filtered.write_h5ad(output_path, compression='gzip')
print(f"✅ 保存完成！")

# 验证保存的文件
print(f"\n验证保存的文件...")
adata_verify = sc.read_h5ad(output_path)
print(f"  验证通过: {adata_verify.shape} == {adata_filtered.shape}")

