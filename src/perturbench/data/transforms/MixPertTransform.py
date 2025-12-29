from .base import TransformBase
import torch
import json
import numpy as np
import os


class MixPertTransform(TransformBase):
    def __init__(self, obs_df, mode,
                 gene_key,
                 drug_key,
                 env_key,
                 crispr_key,
                 cov_keys,
                 crispr_map_path,
                 gene_map_path,
                 drug_map_path,
                 env_map_path,
                 keep_pert_names=True,
                 null_token='',
                 cov_maps_path='./',
                 use_covs=False,
                 use_cell_emb=False,
                 comb_delim='+',
                 ):
        super().__init__(obs_df, mode)
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

        self.gene_pert_dim+=self.crispr_dim

        if use_covs:
            self.get_cov_maps()
            self.get_cov_dims()
        else:
            # 即使不使用协变量，也需要初始化 cov_dims 属性
            self.cov_dims = {}
            self.n_total_covs = 0

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

    def get_gene_map(self):
        if os.path.exists(self.gene_map_path):
            self.gene_map = torch.load(self.gene_map_path, weights_only=False)
        else:
            gene_map_dir = os.path.dirname(self.gene_map_path)
            if not os.path.exists(gene_map_dir):
                os.makedirs(gene_map_dir)
            self.gene_map = self.get_onehot_dict(self.gene_key)
            torch.save(self.gene_map, self.gene_map_path)

    def get_crispr_map(self):
        if os.path.exists(self.crispr_map_path):
            self.crispr_map = torch.load(self.crispr_map_path, weights_only=False)
        else:
            crispr_map_dir = os.path.dirname(self.crispr_map_path)
            if not os.path.exists(crispr_map_dir):
                os.makedirs(crispr_map_dir)
            self.crispr_map = self.get_onehot_dict(self.crispr_key)
            torch.save(self.crispr_map, self.crispr_map_path)

    def get_cov_maps(self):
        if os.path.exists(self.cov_maps_path):
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
            cov_maps_dir = os.path.dirname(self.cov_maps_path)
            if not os.path.exists(cov_maps_dir):
                os.makedirs(cov_maps_dir)
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
            torch.save(self.cov_maps, self.cov_maps_path)  # 然后保存
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
        if os.path.exists(self.drug_map_path):
            self.drug_map = torch.load(self.drug_map_path, weights_only=False)
        else:
            drug_map_dir = os.path.dirname(self.drug_map_path)
            if not os.path.exists(drug_map_dir):
                os.makedirs(drug_map_dir)
            self.drug_map = self.get_onehot_dict(self.drug_key)
            torch.save(self.drug_map, self.drug_map_path)

    def get_env_map(self):
        if os.path.exists(self.env_map_path):
            self.env_map = torch.load(self.env_map_path, weights_only=False)
        else:
            env_map_dir = os.path.dirname(self.env_map_path)
            if not os.path.exists(env_map_dir):
                os.makedirs(env_map_dir)
            self.env_map = self.get_onehot_dict(self.env_key)
            torch.save(self.env_map, self.env_map_path)

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
        self.crispr_dim = len(self.crispr_map[key])


    def __call__(self, example):
        out = {}

        def _safe_lookup(mapping, key):
            return mapping[key] if key in mapping else mapping[self.null_token]

        if self.use_covs:
            for cov_key in self.cov_keys:
                cov_map = self.cov_maps[cov_key]
                out[cov_key] = _safe_lookup(cov_map, example[cov_key])

        gene_perts = example[self.gene_key].split(self.comb_delim)
        drug_perts = example[self.drug_key].split(self.comb_delim)
        env_perts = example[self.env_key].split(self.comb_delim)

        # CRISPR 可能为 None / NaN / 空
        if self.crispr_key and self.crispr_key in example:
            raw_crispr = example[self.crispr_key]
            if raw_crispr is None or (isinstance(raw_crispr, float) and np.isnan(raw_crispr)):
                crispr_type = self.null_token
            else:
                raw_str = str(raw_crispr)
                crispr_type = self.null_token if raw_str.lower() == "nan" else raw_str
        else:
            crispr_type = self.null_token

        if self.keep_pert_names:
            pert_names={
                'gene_names':gene_perts,
                'drug_names':drug_perts,
                'env_names':env_perts,
            }
            out={**out, **pert_names}
        
        gene_emb = torch.stack([_safe_lookup(self.gene_map, gene_pert) for gene_pert in gene_perts]).sum(dim=0)
        drug_emb_list = [_safe_lookup(self.drug_map, drug_pert) for drug_pert in drug_perts]
        env_emb = torch.stack([_safe_lookup(self.env_map, env_pert) for env_pert in env_perts]).sum(dim=0)
        crispr_emb = _safe_lookup(self.crispr_map, crispr_type)
        gene_emb=torch.concat([gene_emb, crispr_emb], dim=0)

        out['gene_pert'] = gene_emb
        out['drug_pert'] = drug_emb_list
        out['env_pert'] = env_emb

        # 确保所有输出都是float32类型
        out['gene_pert'] = out['gene_pert'].float()
        out['drug_pert'] = [pert.float() for pert in out['drug_pert']]
        out['env_pert'] = out['env_pert'].float()

        # 正确处理tensor转换，避免警告
        pert_counts = example['pert_cell_counts']
        control_counts = example['control_cell_counts']

        if isinstance(pert_counts, torch.Tensor):
            out['pert_cell_counts'] = pert_counts.detach().clone().float()
        else:
            out['pert_cell_counts'] = torch.tensor(pert_counts, dtype=torch.float32)

        if isinstance(control_counts, torch.Tensor):
            out['control_cell_counts'] = control_counts.detach().clone().float()
        else:
            out['control_cell_counts'] = torch.tensor(control_counts, dtype=torch.float32)

        if self.use_cell_emb:
            pert_emb = example['pert_cell_emb']
            control_emb = example['control_cell_emb']

            if isinstance(pert_emb, torch.Tensor):
                out['pert_cell_emb'] = pert_emb.detach().clone().float()
            else:
                out['pert_cell_emb'] = torch.tensor(pert_emb, dtype=torch.float32)

            if isinstance(control_emb, torch.Tensor):
                out['control_cell_emb'] = control_emb.detach().clone().float()
            else:
                out['control_cell_emb'] = torch.tensor(control_emb, dtype=torch.float32)

        # 保留额外的keys，比如expression masks
        extra_keys = ['pert_expression_mask', 'control_expression_mask']
        for key in extra_keys:
            if key in example:
                value = example[key]
                if isinstance(value, torch.Tensor):
                    out[key] = value.detach().clone().float()
                else:
                    out[key] = torch.tensor(value, dtype=torch.float32)

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
        
        return total_perts
