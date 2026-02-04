from .base import TransformBase
import torch
import json
import numpy as np
import os


class MixPertTransform(TransformBase):
    def __init__(self, obs_df,
                 gene_key,
                 drug_key,
                 env_key,
                 crispr_key,
                 cov_keys,
                 crispr_map_path:str|None=None,
                 gene_map_path:str|None=None,
                 drug_map_path:str|None=None,
                 env_map_path:str|None=None,
                 regenerate_gene_map=False,
                 regenerate_drug_map=False,
                 regenerate_env_map=False,
                 regenerate_crispr_map=False,
                 regenerate_cov_maps=False,
                 keep_pert_names=True,
                 null_token='',
                 cov_maps_path:str|None=None,
                 use_covs=False,
                 use_cell_emb=False,
                 comb_delim='+',
                 max_gene_perts=3,  # 最大基因扰动数
                 max_drug_perts=3,  # 最大药物扰动数
                 ):
        super().__init__(obs_df)
        self.gene_key = gene_key
        self.drug_key = drug_key
        self.env_key = env_key
        self.cov_keys = cov_keys
        self.crispr_key= crispr_key
        self.use_cell_emb = use_cell_emb
        self.use_covs = use_covs
        self.comb_delim = comb_delim
        self.null_token = null_token
        self.keep_pert_names = keep_pert_names
        self.max_gene_perts = max_gene_perts
        self.max_drug_perts = max_drug_perts

        self.regenerate_gene_map = regenerate_gene_map
        self.regenerate_drug_map = regenerate_drug_map
        self.regenerate_env_map = regenerate_env_map
        self.regenerate_crispr_map = regenerate_crispr_map
        self.regenerate_cov_maps = regenerate_cov_maps

        self.gene_map_path = gene_map_path
        self.crispr_map_path = crispr_map_path
        self.drug_map_path = drug_map_path
        self.env_map_path = env_map_path
        self.cov_maps_path = cov_maps_path

        self.get_gene_map()
        self.get_crispr_map()
        self.get_drug_map()
        self.get_env_map()
        self.get_perts_dim()
        self.add_null_tokens()

        # 计算总扰动数量
        self.n_perts = self.calculate_total_perturbations()

        if use_covs:
            self.get_cov_maps()
            self.get_cov_dims()
            # 预缓存 cov null embeddings
            self._cov_null_embs = {
                cov_key: self.cov_maps[cov_key][self.null_token]
                for cov_key in self.cov_keys
            }
        else:
            # 即使不使用协变量，也需要初始化 cov_dims 属性
            self.cov_dims = {}
            self.n_total_covs = 0
            self._cov_null_embs = {}

        # 确保 n_total_covs 总是被设置（即使 use_covs=False，我们也需要为模型提供这个值）
        if not hasattr(self, 'n_total_covs'):
            if use_covs and hasattr(self, 'cov_maps') and self.cov_maps is not None:
                n_total_covs = 0
                for cov_map in self.cov_maps.values():
                    n_total_covs += len(list(cov_map.keys()))
                self.n_total_covs = n_total_covs
            else:
                self.n_total_covs = 0

    def add_null_tokens(self):
        if self.null_token not in self.gene_map.keys():
            self.gene_map[self.null_token] = torch.zeros(self.gene_pert_dim, dtype=torch.float32)
        if self.null_token not in self.drug_map.keys():
            self.drug_map[self.null_token] = torch.zeros(self.drug_pert_dim, dtype=torch.float32)
        if self.null_token not in self.env_map.keys():
            self.env_map[self.null_token] = torch.zeros(self.env_pert_dim, dtype=torch.float32)
        if self.null_token not in self.crispr_map.keys():
            self.crispr_map[self.null_token] = torch.zeros(self.crispr_dim, dtype=torch.float32)
        
        # 预缓存 null embedding（加速 __call__ 中的查找）
        self._gene_null_emb = self.gene_map[self.null_token]
        self._drug_null_emb = self.drug_map[self.null_token]
        self._env_null_emb = self.env_map[self.null_token]
        self._crispr_null_emb = self.crispr_map[self.null_token]

    def get_gene_map(self):
        if self.gene_map_path is not None and os.path.exists(self.gene_map_path) and not self.regenerate_gene_map:
            self.gene_map = torch.load(self.gene_map_path, weights_only=False)
        else:
            self.gene_map = self.get_onehot_dict(self.gene_key)

    def get_crispr_map(self):
        if self.crispr_map_path is not None and os.path.exists(self.crispr_map_path) and not self.regenerate_crispr_map:
            self.crispr_map = torch.load(self.crispr_map_path, weights_only=False)
        else:
            self.crispr_map = self.get_onehot_dict(self.crispr_key)

    def get_cov_maps(self):
        if self.cov_maps_path is not None and os.path.exists(self.cov_maps_path) and not self.regenerate_cov_maps:
            self.cov_maps = torch.load(self.cov_maps_path, weights_only=False)
            # 确保每个 cov_map 都包含 null_token
            for cov_key, cov_map in self.cov_maps.items():
                if self.null_token not in cov_map:
                    if len(cov_map) > 0:
                        first_val = next(iter(cov_map.values()))
                        onehot_dim = len(first_val)
                        cov_map[self.null_token] = torch.zeros(onehot_dim, dtype=torch.float32)
                    else:
                        cov_map[self.null_token] = torch.zeros(1, dtype=torch.float32)
            # Calculate n_total_covs for existing cov_maps
            n_total_covs = 0
            for cov_map in self.cov_maps.values():
                n_total_covs += len(list(cov_map.keys()))
            self.n_total_covs = n_total_covs
        else:
            cov_maps = {}
            for cov_key in self.cov_keys:
                cov_map = self.get_onehot_dict(cov_key)
                # 确保 null_token 存在于 cov_map 中
                if self.null_token not in cov_map:
                    # 获取 one-hot 向量的长度（从第一个值获取）
                    if len(cov_map) > 0:
                        first_val = next(iter(cov_map.values()))
                        onehot_dim = len(first_val)
                        cov_map[self.null_token] = torch.zeros(onehot_dim, dtype=torch.float32)
                    else:
                        # 如果 cov_map 为空，创建一个包含 null_token 的映射
                        cov_map[self.null_token] = torch.zeros(1, dtype=torch.float32)
                cov_maps[cov_key] = cov_map
            self.cov_maps = cov_maps  # 先赋值给 self.cov_maps
            # Calculate n_total_covs for new cov_maps
            n_total_covs = 0
            for cov_map in self.cov_maps.values():
                n_total_covs += len(list(cov_map.keys()))
            self.n_total_covs = n_total_covs

    def get_cov_dims(self):
        cov_dims = {}
        for cov_key in self.cov_keys:
            cov_map = self.cov_maps[cov_key]
            first_val = next(iter(cov_map.values()))
            cov_dims[cov_key] = len(first_val)  # 存维度值而非 tensor
        self.cov_dims = cov_dims

    def get_drug_map(self):
        if self.drug_map_path is not None and os.path.exists(self.drug_map_path) and not self.regenerate_drug_map:
            self.drug_map = torch.load(self.drug_map_path, weights_only=False)
        else:
            self.drug_map = self.get_onehot_dict(self.drug_key)

    def get_env_map(self):
        if self.env_map_path is not None and os.path.exists(self.env_map_path) and not self.regenerate_env_map:
            self.env_map = torch.load(self.env_map_path, weights_only=False)
        else:
            self.env_map = self.get_onehot_dict(self.env_key)

    def get_unique_vals(self, key):
        unique_obs_col = self.obs_df[key].unique()
        unique_vals = [self.null_token]
        for comb_val in unique_obs_col:
            # Skip None/NaN
            if comb_val is None:
                continue
            if isinstance(comb_val, float) and np.isnan(comb_val):
                continue
            comb_str = str(comb_val)
            if comb_str.lower() == "nan":
                continue
            unique_vals.extend(comb_str.split(self.comb_delim))
        unique_vals = np.array(list(set(unique_vals)))
        return unique_vals

    def get_onehot_dict(self, key):
        onehot_dict = {}
        unique_vals = self.get_unique_vals(key)
        for unique_val in unique_vals:
            onehot_dict[unique_val] = \
                torch.tensor(unique_vals == unique_val, dtype=torch.float32)
        return onehot_dict

    def get_perts_dim(self):

        key = list(self.gene_map.keys())[0]
        self.gene_pert_dim = len(self.gene_map[key])

        key = list(self.drug_map.keys())[0]
        self.drug_pert_dim = len(self.drug_map[key])

        key = list(self.env_map.keys())[0]
        self.env_pert_dim = len(self.env_map[key])

        key = list(self.crispr_map.keys())[0]
        self.crispr_pert_dim = len(self.crispr_map[key])

    def _safe_lookup(self, mapping, key, null_emb):
        """快速查找，使用预缓存的 null embedding"""
        return mapping.get(key, null_emb)

    def __call__(self, example):
        # 使用局部变量加速属性访问
        gene_key = self.gene_key
        drug_key = self.drug_key
        env_key = self.env_key
        crispr_key = self.crispr_key
        comb_delim = self.comb_delim
        null_token = self.null_token
        
        # 预取 null embeddings（避免重复查找）
        gene_null = self._gene_null_emb
        drug_null = self._drug_null_emb
        env_null = self._env_null_emb
        crispr_null = self._crispr_null_emb
        
        out = {}

        # 协变量处理
        if self.use_covs:
            cov_maps = self.cov_maps
            cov_null_embs = self._cov_null_embs
            for cov_key in self.cov_keys:
                out[cov_key] = cov_maps[cov_key].get(example[cov_key], cov_null_embs[cov_key])

        # 字符串分割（无法避免，但用局部变量加速）
        gene_perts = example[gene_key].split(comb_delim)
        drug_perts = example[drug_key].split(comb_delim)
        env_perts = example[env_key].split(comb_delim)

        # CRISPR 处理（简化逻辑）
        raw_crispr = example.get(crispr_key)
        if raw_crispr is None or raw_crispr == '' or (isinstance(raw_crispr, float) and raw_crispr != raw_crispr):
            crispr_type = null_token
        else:
            crispr_type = str(raw_crispr) if str(raw_crispr).lower() != "nan" else null_token

        # pert_names（直接赋值，避免字典合并）
        if self.keep_pert_names:
            out['gene_names'] = gene_perts
            out['drug_names'] = drug_perts
            out['env_names'] = env_perts

        # gene embedding（padded tensor 格式，适合多进程）
        if self.gene_pert_dim > 1:
            gene_map = self.gene_map
            max_gene = self.max_gene_perts
            n_gene = min(len(gene_perts), max_gene)
            gene_tensor = torch.zeros(max_gene, self.gene_pert_dim, dtype=torch.float32)
            for i in range(n_gene):
                gene_tensor[i] = gene_map.get(gene_perts[i], gene_null)
            out['gene_pert'] = gene_tensor
            out['gene_pert_len'] = n_gene
        
        # drug embedding（padded tensor 格式）
        if self.drug_pert_dim > 1:
            drug_map = self.drug_map
            max_drug = self.max_drug_perts
            n_drug = min(len(drug_perts), max_drug)
            drug_tensor = torch.zeros(max_drug, self.drug_pert_dim, dtype=torch.float32)
            for i in range(n_drug):
                drug_tensor[i] = drug_map.get(drug_perts[i], drug_null)
            out['drug_pert'] = drug_tensor
            out['drug_pert_len'] = n_drug
        
        # env embedding（优化：预分配结果）
        if self.env_pert_dim > 1:
            env_map = self.env_map
            if len(env_perts) == 1:
                out['env_pert'] = env_map.get(env_perts[0], env_null)
            else:
                # 手动求和，避免 torch.stack 的开销
                result = env_map.get(env_perts[0], env_null).clone()
                for ep in env_perts[1:]:
                    result += env_map.get(ep, env_null)
                out['env_pert'] = result
        
        # crispr embedding
        if self.crispr_pert_dim > 1:
            out['crispr_pert'] = self.crispr_map.get(crispr_type, crispr_null)

        # counts 处理（使用 torch.as_tensor 避免不必要的拷贝）
        pert_counts = example['pert_cell_counts']
        control_counts = example['control_cell_counts']
        
        out['pert_cell_counts'] = pert_counts.float() if isinstance(pert_counts, torch.Tensor) else torch.as_tensor(pert_counts, dtype=torch.float32)
        out['control_cell_counts'] = control_counts.float() if isinstance(control_counts, torch.Tensor) else torch.as_tensor(control_counts, dtype=torch.float32)

        # cell embedding
        pert_cell_emb = example.get('pert_cell_emb', None)
        if self.use_cell_emb and pert_cell_emb is not None:
            pert_emb = example['pert_cell_emb']
            control_emb = example['control_cell_emb']
            out['pert_cell_emb'] = pert_emb.float() if isinstance(pert_emb, torch.Tensor) else torch.as_tensor(pert_emb, dtype=torch.float32)
            out['control_cell_emb'] = control_emb.float() if isinstance(control_emb, torch.Tensor) else torch.as_tensor(control_emb, dtype=torch.float32)

        # mask
        if 'mask' in example:
            mask = example['mask']
            out['mask'] = mask.float() if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, dtype=torch.float32)

        return out
    
    def calculate_total_perturbations(self):
        total_perts = 0
        
        gene_perts = set()
        for comb_val in self.obs_df[self.gene_key].unique():
            gene_perts.update(comb_val.split(self.comb_delim))
        total_perts += len(gene_perts)
        
        drug_perts = set()
        for comb_val in self.obs_df[self.drug_key].unique():
            drug_perts.update(comb_val.split(self.comb_delim))
        total_perts += len(drug_perts)
        
        env_perts = set()
        for comb_val in self.obs_df[self.env_key].unique():
            env_perts.update(comb_val.split(self.comb_delim))
        total_perts += len(env_perts)

        crispr_perts = set()
        for comb_val in self.obs_df[self.crispr_key].unique():
            crispr_perts.update(comb_val.split(self.comb_delim))
        total_perts += len(crispr_perts)
        
        return total_perts
