import warnings
from pathlib import Path
from collections.abc import Callable, Mapping
import gc
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from omegaconf import DictConfig
from torch.utils.data import Dataset
import anndata as ad
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset


class scTrainDataset(Dataset):
    """
    高性能版本：
    - 不使用 AnnData 切片
    - 所有信息提前提取到 numpy 数组
    - 不使用 Pandas 过滤
    """

    def __init__(
            self,
            pert_adata: ad.AnnData,
            control_adata: ad.AnnData,
            cov_keys: List[str],
            transform: DictConfig,
            pert_key: str|None=None,
            gene_key: str|None=None,
            drug_key:str|None=None,
            crispr_key: str|None=None,
            use_mix_pert: bool=False,
            env_key: str|None=None,
            embedding_key: str | None = None,
            raw_counts_key: str | None = None,
            cov_avg_sampling: bool = False,
            cell_set_len: int | None = None,
            add_keys: List[str] | None = None,
            predict_controls: bool = False,
            expression_mask: np.ndarray | None = None,
            **kwargs
    ):
        super().__init__()

        self.use_mix_pert=use_mix_pert

        if predict_controls:
            pert_adata=ad.concat([pert_adata,control_adata])

        self.pert_adata=pert_adata
        self.control_adata=control_adata

        self.pert_obs=pert_adata.obs
        self.control_obs=control_adata.obs

        # Generate expression masks for perturbation and control data
        self.pert_expression_mask = (pert_adata.X > 0).astype(np.float32)#没啥用，删了吧，还不如一句mask=batch.pert_cell_counts!=0
        self.control_expression_mask = (control_adata.X > 0).astype(np.float32)

        # -----------------------------------
        # 1. 提前提取所有数据（极大提速）
        # -----------------------------------
        self.X_pert = pert_adata.X
        self.X_ctrl = control_adata.X

        self.embedding_key = embedding_key
        if embedding_key:
            self.emb_pert = pert_adata.obsm[embedding_key]
            self.emb_ctrl = control_adata.obsm[embedding_key]
        else:
            self.emb_pert = None
            self.emb_ctrl = None

        self.raw_counts_key = raw_counts_key
        if raw_counts_key:
            self.raw_pert = pert_adata.obs[raw_counts_key].to_numpy()
            self.raw_ctrl = control_adata.obs[raw_counts_key].to_numpy()
        else:
            self.raw_pert = None
            self.raw_ctrl = None

        # -----------------------------------
        # 2. 提前构建 obs 信息
        # -----------------------------------
        self.pert_key = pert_key
        self.gene_key = gene_key
        self.drug_key=drug_key
        self.env_key = env_key
        self.crispr_key = crispr_key
        self.cov_keys = cov_keys
        self.cov_avg_sampling = cov_avg_sampling
        self.cell_set_len = cell_set_len
        self.add_keys = add_keys

        pert_obs = pert_adata.obs
        ctrl_obs = control_adata.obs

        # 组合 cov 信息
        if cov_keys:
            pert_cov = self.merge_cols(pert_obs, cov_keys)
            ctrl_cov = self.merge_cols(ctrl_obs, cov_keys)
        else:
            pert_cov = np.array(["_"] * len(pert_obs))
            ctrl_cov = np.array(["_"] * len(ctrl_obs))

        self.pert_cov = pert_cov
        self.ctrl_cov = ctrl_cov

        # pert only
        if not self.use_mix_pert:
            self.pert_labels=pert_obs[pert_key].astype(str).to_numpy()
        else:
            # 处理 gene_key，先转换为字符串类型避免 Categorical 问题
            gene_col = pert_obs[gene_key]
            if pd.api.types.is_categorical_dtype(gene_col):
                gene_col = gene_col.astype(str)
            self.gene_pert_labels = gene_col.fillna('').astype(str).to_numpy()

            # 处理 drug_key
            drug_col = pert_obs[drug_key]
            if pd.api.types.is_categorical_dtype(drug_col):
                drug_col = drug_col.astype(str)
            self.drug_pert_labels = drug_col.fillna('').astype(str).to_numpy()

            # 处理 env_key
            env_col = pert_obs[env_key]
            if pd.api.types.is_categorical_dtype(env_col):
                env_col = env_col.astype(str)
            self.env_pert_labels = env_col.fillna('').astype(str).to_numpy()

            # 处理 crispr_key
            crispr_col = pert_obs[crispr_key]
            if pd.api.types.is_categorical_dtype(crispr_col):
                crispr_col = crispr_col.astype(str)
            # CRISPR 列可能有很多空值（只有有 gene_pt 的列才有 CRISPR），将 NaN 转换为空字符串
            self.crispr_labels = crispr_col.fillna('').astype(str).to_numpy()

        # -----------------------------------
        # 3. 建立索引（代替 Pandas 过滤）
        # -----------------------------------
        #（重要）为每个 cov / (cov,pert) 分组提前建立索引列表
        self.index_by_cov_pert = {}
        self.index_by_cov_ctrl = {}

        for i, c in enumerate(pert_cov):
            if self.use_mix_pert:
                key=(c,self.gene_pert_labels[i],self.drug_pert_labels[i],self.env_pert_labels[i],self.crispr_labels[i])
            else:
                key = (c, self.pert_labels[i])
            self.index_by_cov_pert.setdefault(key, []).append(i)

        for i, c in enumerate(ctrl_cov):
            self.index_by_cov_ctrl.setdefault(c, []).append(i)

        # weak check
        if any(len(v) == 0 for v in self.index_by_cov_ctrl.values()):
            print("⚠ warning: 某些 control cov 为空")

        # cov unique
        self.unique_covs = np.unique(pert_cov)
        self.unique_keys = list(self.index_by_cov_pert.keys())

        # -----------------------------------
        # 4. transform 初始化
        # -----------------------------------
        self.transform = transform(
            obs_df=pert_obs.copy(),
            mode="train"
        )

        # 记录总长度
        self.length = len(pert_obs) if predict_controls \
            else len(pert_obs)+ len(ctrl_obs)

    def merge_cols(self, obs_df, cols):
        merged = obs_df[cols[0]].astype(str).to_numpy()
        for c in cols[1:]:
            merged = merged + "<>" + obs_df[c].astype(str).to_numpy()
        return merged

    def __len__(self):
        if self.cell_set_len:
            return self.length // self.cell_set_len
        else:
            return self.length

    def __getitem__(self, idx):
        if self.cell_set_len:
            return self.get_set_sample()
        else:
            return self.get_single_sample()

    # ---------------------------------------------------------
    # 采样一个单细胞
    # ---------------------------------------------------------
    def get_single_sample(self):
        # 1. 随机选一个 (cov, pert) 或者 (cov,gene_pert,drug_pert,env_pert)
        key = self.unique_keys[
            np.random.randint(len(self.unique_keys))
        ]

        # 2. 从对应 bucket 取 idx
        pert_idx = np.random.choice(self.index_by_cov_pert[key])
        ctrl_idx = np.random.choice(self.index_by_cov_ctrl[key[0]])

        if self.use_mix_pert:
            return self.build_output(pert_idx, ctrl_idx, key[0], (key[1],key[2],key[3],key[4]))
        else:
            return self.build_output(pert_idx, ctrl_idx, key[0], key[1])

    # ---------------------------------------------------------
    # 采样一个 set（多个细胞）
    # ---------------------------------------------------------
    def get_set_sample(self):
        key = self.unique_keys[
            np.random.randint(len(self.unique_keys))
        ]

        pert_list = self.index_by_cov_pert[key]
        ctrl_list = self.index_by_cov_ctrl[key[0]]

        replace_p = self.cell_set_len > len(pert_list)
        replace_c = self.cell_set_len > len(ctrl_list)

        pert_idxs = np.random.choice(pert_list, size=self.cell_set_len, replace=replace_p)
        ctrl_idxs = np.random.choice(ctrl_list, size=self.cell_set_len, replace=replace_c)

        if self.use_mix_pert:
            return self.build_output(pert_idxs, ctrl_idxs,key[0], (key[1],key[2],key[3],key[4]))
        else:
            return self.build_output(pert_idxs, ctrl_idxs, key[0], key[1])

    # ---------------------------------------------------------
    # 构建 transform 输入字典（纯 numpy → torch）
    # ---------------------------------------------------------
    def build_output(self, pert_idx, ctrl_idx, cov, pert):
        out = {}

        covs=cov.split('<>')
        for idx,cov_key in enumerate(self.cov_keys):
            out[cov_key]=covs[idx]

        if self.use_mix_pert:
            gene_pert,drug_pert,env_pert,crispr_type=pert
            out[self.gene_key]=gene_pert
            out[self.drug_key]=drug_pert
            out[self.env_key]=env_pert
            out[self.crispr_key]=crispr_type
        else:
            out[self.pert_key] = pert

        # counts
        out["pert_cell_counts"] = torch.tensor(self.X_pert[pert_idx])
        out["control_cell_counts"] = torch.tensor(self.X_ctrl[ctrl_idx])

        # expression mask for loss calculation
        out["pert_expression_mask"] = torch.tensor(self.pert_expression_mask[pert_idx])
        out["control_expression_mask"] = torch.tensor(self.control_expression_mask[ctrl_idx])

        # embedding
        if self.emb_pert is not None:
            out["pert_cell_emb"] = torch.tensor(self.emb_pert[pert_idx])
            out["control_cell_emb"] = torch.tensor(self.emb_ctrl[ctrl_idx])

        # raw
        if self.raw_pert is not None:
            out["pert_raw_counts"] = torch.tensor(self.raw_pert[pert_idx])
            out["control_raw_counts"] = torch.tensor(self.raw_ctrl[ctrl_idx])

        # transform（保持兼容）
        return self.transform(out)

    def get_gene_names(self):
        return self.pert_adata.var_names

    def get_embedding_width(self):
        if self.embedding_key:
            return self.pert_adata.obsm[self.embedding_key].shape[1]
        else:
            return None
