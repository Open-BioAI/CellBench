import warnings
from pathlib import Path
from collections.abc import Callable, Mapping
import gc
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from omegaconf import DictConfig
import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import perturbench.data.datasplitter as datasplitter
import anndata as ad
import numpy as np
import hydra
import scanpy as sc
import pandas as pd
import torch

class scInferenceDataset(Dataset):

    def __init__(
        self,
        pert_adata: ad.AnnData,
        control_adata: ad.AnnData,
        transform: DictConfig,
        pert_key: str | None = None,
        gene_key: str | None = None,
        drug_key: str | None = None,
        env_key: str | None = None,
        crispr_key: str | None = None,
        use_mix_pert: bool = False,
        cov_keys: List[str] | None = None,
        embedding_key: str | None = None,
        raw_counts_key: str | None = None,
        cell_set_len: int | None = None,
        add_keys: List[str] | None = None,
        expression_mask: np.ndarray | None = None,
        **kwargs
    ):
        super().__init__()

        self.merge_delim = '<>'
        self.pert_key = pert_key
        self.gene_key=gene_key
        self.drug_key=drug_key
        self.env_key=env_key
        self.crispr_key=crispr_key
        self.cov_keys = cov_keys
        self.add_keys = add_keys
        self.embedding_key = embedding_key
        self.raw_counts_key = raw_counts_key
        self.cell_set_len = cell_set_len

        self.use_mix_pert = use_mix_pert

        # ====== 1) 预取 obs-level 信息（避免重复 pandas 操作） ======
        pert_obs = pert_adata.obs.copy()
        ctrl_obs = control_adata.obs.copy()

        pert_obs["cov_merged"] = self.merge_cols(pert_obs, cov_keys, self.merge_delim)
        ctrl_obs["cov_merged"] = self.merge_cols(ctrl_obs, cov_keys, self.merge_delim)

        self.pert_adata=pert_adata
        self.control_adata=control_adata

        self.pert_obs = pert_obs
        self.control_obs = ctrl_obs

        # ====== 2) 预构建控制组索引映射 ======
        self.ctrl_group_map = {
            cov: ctrl_obs.index[ctrl_obs["cov_merged"] == cov].to_numpy()
            for cov in ctrl_obs["cov_merged"].unique()
        }

        # ====== 3) 把矩阵一次性取出为 numpy，避免 AnnData 切片 ======
        self.pert_X = np.asarray(pert_adata.X)
        self.ctrl_X = np.asarray(control_adata.X)

        # Generate expression masks for perturbation and control data
        self.pert_expression_mask = (self.pert_X > 0).astype(np.float32)
        self.ctrl_expression_mask = (self.ctrl_X > 0).astype(np.float32)

        if embedding_key:
            self.pert_emb = pert_adata.obsm[embedding_key]
            self.ctrl_emb = control_adata.obsm[embedding_key]
        else:
            self.pert_emb = self.ctrl_emb = None

        if raw_counts_key:
            self.pert_raw = pert_obs[raw_counts_key].to_numpy()
            self.ctrl_raw = ctrl_obs[raw_counts_key].to_numpy()
        else:
            self.pert_raw = self.ctrl_raw = None

        # 预记录 index mapping，避免反复查名字
        self.pert_index_map = {name: i for i, name in enumerate(pert_obs.index)}
        self.ctrl_index_map = {name: i for i, name in enumerate(ctrl_obs.index)}

        # ====== 初始化 transform ======
        self.transform = transform(obs_df=pert_obs.copy(), mode="eval")

        # ====== 构建 chunks ======
        if cell_set_len:
            self.chunks = self.build_sets()
        else:
            self.chunks = self.build_singles()

    # ----------------------------------------------------------------------
    # 快速打包表达信息
    # ----------------------------------------------------------------------
    def pack_expr(self, pert_i, ctrl_i):
        out = {
            "pert_cell_counts": torch.from_numpy(self.pert_X[pert_i]),
            "control_cell_counts": torch.from_numpy(self.ctrl_X[ctrl_i]),
        }

        # expression mask for loss calculation
        out["pert_expression_mask"] = torch.from_numpy(self.pert_expression_mask[pert_i])
        out["control_expression_mask"] = torch.from_numpy(self.ctrl_expression_mask[ctrl_i])

        if self.embedding_key:
            out["pert_cell_emb"] = torch.from_numpy(self.pert_emb[pert_i])
            out["control_cell_emb"] = torch.from_numpy(self.ctrl_emb[ctrl_i])

        if self.raw_counts_key:
            out["pert_raw_counts"] = torch.from_numpy(self.pert_raw[pert_i])
            out["control_raw_counts"] = torch.from_numpy(self.ctrl_raw[ctrl_i])

        return out

    # ----------------------------------------------------------------------
    def build_singles(self):
        chunks = []
        n = len(self.pert_obs)

        # 一次性拿 meta 信息
        pert_cov = self.pert_obs["cov_merged"].to_numpy()
        if self.use_mix_pert:
            gene_pert=self.pert_obs[self.gene_key].to_numpy()
            drug_pert=self.pert_obs[self.drug_key].to_numpy()
            env_pert=self.pert_obs[self.env_key].to_numpy()
            crispr_type=self.pert_obs[self.crispr_key].to_numpy()
        else:
            pert_pert = self.pert_obs[self.pert_key].to_numpy()

        add_keys_arrays = {
            k: self.pert_obs[k].to_numpy() for k in (self.add_keys or [])
        }

        for i in range(n):
            cov = pert_cov[i]
            ctrl_idxs = self.ctrl_group_map[cov]
            ctrl_name = np.random.choice(ctrl_idxs)
            ctrl_i = self.ctrl_index_map[ctrl_name]

            meta = {c: self.pert_obs[c].iloc[i] for c in self.cov_keys}
            if self.use_mix_pert:
                meta[self.gene_key]=gene_pert[i]
                meta[self.drug_key]=drug_pert[i]
                meta[self.env_key]=env_pert[i]
                meta[self.crispr_key]=crispr_type[i]
            else:
                meta[self.pert_key] = pert_pert[i]
            for k in add_keys_arrays:
                meta[k] = add_keys_arrays[k][i]

            expr = self.pack_expr(i, ctrl_i)
            chunks.append((self.transform({**meta, **expr}), self.pert_obs.iloc[[i]]))

        return chunks

    # ----------------------------------------------------------------------
    def build_sets(self):
        chunks = []
        if self.use_mix_pert:
            obs_merged = self.merge_cols(
                self.pert_obs, self.cov_keys + [self.gene_key,self.drug_key,self.env_key,self.crispr_key], self.merge_delim
            ).to_numpy()

        else:
            obs_merged = self.merge_cols(
                self.pert_obs, self.cov_keys + [self.pert_key], self.merge_delim
            ).to_numpy()

        unique_groups = np.unique(obs_merged)

        for g in unique_groups:
            idxs = np.where(obs_merged == g)[0]
            cov_vals = g.split(self.merge_delim)
            if self.use_mix_pert:
                pert_val=cov_vals[len(self.cov_keys):]
                cov_vals = cov_vals[:len(self.cov_keys)]
                gene_pert,drug_pert,env_pert,crispr_type=pert_val
                sample_meta={
                    self.gene_key:gene_pert,
                    self.drug_key:drug_pert,
                    self.env_key:env_pert,
                    self.crispr_key:crispr_type,
                }
            else:
                pert_val = cov_vals[-1]
                cov_vals = cov_vals[:-1]
                sample_meta = {self.pert_key: pert_val}

            for ci, c in enumerate(self.cov_keys):
                sample_meta[c] = cov_vals[ci]

            ctrl_cov = self.merge_delim.join(cov_vals)
            # Fallback: 如果找不到匹配的 control 组，使用所有 control 细胞
            if ctrl_cov in self.ctrl_group_map:
                ctrl_idxs = self.ctrl_group_map[ctrl_cov]
            else:
                # 使用所有 control 细胞作为 fallback
                ctrl_idxs = self.all_ctrl_idxs

            for start in range(0, len(idxs), self.cell_set_len):
                sub = idxs[start:start+self.cell_set_len]
                sub_n = len(sub)

                ctrl_sample = np.random.choice(
                    ctrl_idxs, size=sub_n, replace=(len(ctrl_idxs)<sub_n)
                )

                # pack batch
                out=self.pack_expr(sub, [self.ctrl_index_map[c] for c in ctrl_sample])

                # add_keys
                if self.add_keys:
                    for k in self.add_keys:
                        out[k] = self.pert_obs[k].iloc[sub].to_numpy()

                chunks.append((self.transform({**sample_meta, **out}),
                               self.pert_obs.iloc[sub]))

        return chunks

    # ----------------------------------------------------------------------

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

    @staticmethod
    def merge_cols(df, cols, delim):
        x = df[cols[0]].astype(str)
        for c in cols[1:]:
            x = x + delim + df[c].astype(str)
        return x.astype("category")
