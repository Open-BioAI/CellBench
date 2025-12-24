data_path='/fs-computility-new/upzd_share/maoxinjie/AIVC/data/after_preprocess/total/all_cell_line_filterdrug_subsampled.h5ad'
import scanpy as sc
adata=sc.read(data_path)

print('control' in adata.obs['env_pt'].unique())
print('control' in adata.obs['gene_pt'].unique())
print('control' in adata.obs['drug_pt'].unique())


'''
adata=adata[adata.obs['split']=='test']
print(adata)
print(adata.obs['CRISPR'].value_counts())
print(adata.obs['split'].value_counts())
print(adata.obs['cell_cluster'].value_counts())
print(adata.obs['time'].value_counts())
print(adata.obs['dataset'].value_counts())
print(adata.obs['cluster'].value_counts())

print('\n\n')
for cluster in adata.obs['cluster'].unique():
    print(type(cluster))
    print('cluster:', cluster)
    mask=adata.obs['cluster']==cluster
    _adata=adata[mask]
    print(_adata.obs['cell_cluster'].value_counts())
    print(_adata.obs['dataset'].value_counts())
    print(_adata.obs['split'].value_counts())
    print(_adata.obs['CRISPR'].value_counts())
    print('\n\n')


# 找到 cluster 在 [3, 4, 19] 的细胞
mask = adata.obs['cluster'].isin([3, 4, 19])

# 1. 把 'val' 加入 split 的类别
adata.obs['split'] = adata.obs['split'].astype('category')
adata.obs['split'] = adata.obs['split'].cat.add_categories(['val'])

# 2. 修改值
adata.obs.loc[mask, 'split'] = 'val'


print(adata.obs['cluster'].value_counts())


print(adata.obs['split'].value_counts())
'''