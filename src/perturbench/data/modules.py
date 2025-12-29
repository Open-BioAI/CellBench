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
from .datasets import  scTrainDataset,scInferenceDataset
from scipy.sparse import issparse,csr_matrix
from .collate import inference_collate,train_collate
from .utils import build_co_expression_graph

# 全局缓存：用于缓存已加载的 adata 对象，避免重复读取
_adata_cache: Dict[str, ad.AnnData] = {}

class PertDataModule(L.LightningDataModule):
    def __init__(self,
                 data_path,
                 train_batch_size:int,
                 val_batch_size:int,
                 test_batch_size:int,
                 control_val: str,
                 cov_keys: List[str],
                 transform: DictConfig,
                 splitter: DictConfig|None = None,
                 pert_key: str|None=None,
                 gene_key: str | None=None,
                 drug_key:str|None=None,
                 env_key:str|None=None,
                 crispr_key:str|None=None,#CRISPRi,CRISPRa,CRISPRko,null_token
                 result_avg_keys: List[str] | None = None,  # 比如说这个如果是[celltype]那么最终结果聚合时会在celltype上求平均
                 evaluation: DictConfig | None =None,
                 perturbation_combination_delimiter: str | None=None,
                 embedding_key: str | None = None,
                 raw_counts_key: str | None = None,
                 cov_avg_sampling: bool = False,
                 sample_mode: str = "cell",  # "cell" | "set" - determines data packaging mode
                 cell_set_len: int | None = None,  # Only used when sample_mode="set"
                 add_keys: List[str] | None = None,
                 train_num_workers: int | None = 12,
                 val_num_workers: int | None = 12,
                 test_num_workers: int | None = 12,
                 co_expression_graph_config: DictConfig|None=None,
                 inference_top_hvg: int|None=None,
                 sharing_controls: bool=False,
                 predict_controls: bool=False,
                 task: str | None = None,  # Task name: unseen_cell, single_dataset_test, single_dataset_two_cellline, two_datasets_two_celllines
                 split_dir: str | None = None,  # Base directory for split CSV files
                 cache_data: bool = True,  # 是否启用数据缓存（pin 到内存）
                 **kwargs
                 ):

        super().__init__()
        self.use_mix_pert = pert_key is None and (gene_key is not None and drug_key is not None and env_key is not None)
        assert self.use_mix_pert or \
               (pert_key is not None and (gene_key is None and drug_key is None and env_key is None))

        self.result_avg_keys = result_avg_keys
        if self.result_avg_keys is None:
            self.result_avg_keys =cov_keys

        # 使用缓存机制：如果数据已经在内存中，直接使用；否则加载并缓存
        # 注意：缓存的是原始数据，每个实例会基于缓存数据创建副本并应用自己的 task/split 处理
        data_path_str = str(Path(data_path).resolve())
        if cache_data and data_path_str in _adata_cache:
            print(f"Using cached adata for {data_path_str} (shape: {_adata_cache[data_path_str].shape})")
            # 使用深拷贝，避免多个实例之间相互影响
            self.adata = _adata_cache[data_path_str].copy()
        else:
            print(f"Loading adata from {data_path_str}...")
            self.adata = sc.read_h5ad(data_path)
            # 缓存原始数据（在应用 task/split 处理之前）
            if cache_data:
                _adata_cache[data_path_str] = self.adata.copy()
                print(f"Cached adata for {data_path_str} (shape: {self.adata.shape})")
        
        # 读取数据切分表
        # 如果指定了 task，从外部 CSV 文件读取 split 信息并筛选细胞
        # 如果没有指定 task，跳过 CSV 匹配，直接使用输入的 data_path 数据
        if task is not None:
            if split_dir is None:
                # 默认 split 目录在项目根目录下的 split 文件夹
                # 尝试从当前文件位置推断项目根目录
                current_file = Path(__file__)
                # modules.py 在 src/perturbench/data/ 下，向上4级到项目根目录
                project_root = current_file.parent.parent.parent.parent
                split_dir = project_root / "split"
                # 如果推断的路径不存在，使用绝对路径
                if not split_dir.exists():
                    split_dir = Path("./split")
            else:
                split_dir = Path(split_dir)
            
            # 根据 task 确定 CSV 文件路径
            # task 对应 split 下的子目录名
            task_lower = str(task).lower()
            split_subdir = split_dir / task_lower

            def _choose_csv(path: Path) -> Path:
                if path.is_file():
                    return path
                # If multiple files, pick the first sorted for determinism
                csvs = sorted([p for p in path.glob("*.csv") if p.is_file()])
                if not csvs:
                    raise FileNotFoundError(f"No CSV found under {path}")
                return csvs[0]

            # 选择 CSV 文件（所有 task 都使用相同的逻辑）
            csv_path = _choose_csv(split_subdir)
            
            if not csv_path.exists():
                raise FileNotFoundError(f"Split CSV file not found: {csv_path}")
            
            print(f"Loading split CSV from: {csv_path}")
            # 读取 CSV，确保 split 列是字符串类型，不是 Categorical
            split_df = pd.read_csv(csv_path, index_col=0)
            # 去掉列名两端可能的空格或不可见字符（例如 ' split', 'split '）
            split_df.columns = split_df.columns.astype(str).str.strip()

            if 'split' not in split_df.columns:
                raise KeyError(
                    f"'split' column not found in CSV {csv_path}. "
                    f"Available columns: {list(split_df.columns)}"
                )
            
            # 将 split 列转换为字符串类型（处理可能的 NaN 或空字符串）
            split_df['split'] = split_df['split'].astype(str)
            # 将空字符串和 'nan' 字符串替换为空字符串，然后过滤掉
            split_df['split'] = split_df['split'].replace(['nan', 'NaN', ''], '')
            
            # 确保 CSV 的索引（cell ID）与 adata 的索引匹配
            valid_split_mask = split_df['split'].isin(['train', 'val', 'test'])
            valid_cells = split_df[valid_split_mask].index
            
            # 检查哪些细胞在 adata 中存在
            cells_in_adata = self.adata.obs.index.isin(valid_cells)
            n_matched = cells_in_adata.sum()
            n_total_valid = len(valid_cells)
            
            print(f"  Found {n_matched:,} cells in adata matching CSV (out of {n_total_valid:,} valid cells in CSV)")
            print(f"  Original adata shape: {self.adata.shape}")
            
            # 只保留在 CSV 中有有效 split 值的细胞
            self.adata = self.adata[cells_in_adata].copy()
            
            # 如果原来的 split 列存在，先删除以避免类型冲突
            if 'split' in self.adata.obs.columns:
                del self.adata.obs['split']
            
            split_series = split_df.loc[self.adata.obs.index, 'split']
            split_values = split_series.values.astype(str)
            split_series_new = pd.Series(split_values, index=self.adata.obs.index, dtype='object')
            self.adata._obs.loc[:, 'split'] = split_series_new
            self.adata._obs['split'] = self.adata._obs['split'].astype('object')

            assert not pd.api.types.is_categorical_dtype(self.adata.obs['split']), \
                "Failed to set split column as non-Categorical type"
            # If CSV contains sample_count, duplicate cells accordingly (with replacement)
            if 'sample_count' in split_df.columns:
                counts = pd.to_numeric(
                    split_df.loc[self.adata.obs.index, 'sample_count'],
                    errors='coerce'
                ).fillna(1).astype(int)
                counts[counts < 0] = 0  # negative counts are treated as zero
                repeat_idx = np.repeat(np.arange(self.adata.n_obs), counts.values)
                if len(repeat_idx) == 0:
                    raise ValueError("After applying sample_count, no cells remain to subset.")
                self.adata = self.adata[repeat_idx].copy()
                # Ensure unique obs names after duplication
                self.adata.obs_names_make_unique()

            # Drop cells whose expression is all zeros (sparse/dense safe)
            X = self.adata.X
            if issparse(X):
                row_sums = np.asarray(X.sum(axis=1)).ravel()
                col_sums = np.asarray(X.sum(axis=0)).ravel()
            else:
                row_sums = np.sum(X, axis=1)
                col_sums = np.sum(X, axis=0)

            # Drop cells with all-zero expression
            nonzero_mask = row_sums != 0
            n_drop = int((~nonzero_mask).sum())
            if n_drop > 0:
                print(f"  Dropping {n_drop:,} all-zero expression cells")
                self.adata = self.adata[nonzero_mask].copy()
                # recompute column sums after cell filtering
                X = self.adata.X
                if issparse(X):
                    col_sums = np.asarray(X.sum(axis=0)).ravel()
                else:
                    col_sums = np.sum(X, axis=0)

            # Drop genes (columns) that are all zero after subsetting
            gene_nonzero = col_sums != 0
            n_genes_drop = int((~gene_nonzero).sum())
            if n_genes_drop > 0:
                print(f"  Dropping {n_genes_drop:,} all-zero genes")
                self.adata = self.adata[:, gene_nonzero].copy()
        else:
            # 如果没有提供 task，跳过 CSV 匹配，直接使用输入的 data_path 数据
            print("No task specified. Skipping split CSV matching. Using data directly from data_path.")
            # 如果数据中已经有 split 列，保留它；如果没有，后续会通过 splitter 生成或报错


        # Store common attributes early for downstream use
        self.splitter = splitter
        self.pert_key = pert_key
        self.gene_key = gene_key
        self.drug_key=drug_key
        self.env_key=env_key
        self.control_val = control_val
        self.cov_keys = cov_keys
        self.crispr_key=crispr_key
        self.perturbation_combination_delimiter = perturbation_combination_delimiter
        self.embedding_key = embedding_key
        self.raw_counts_key = raw_counts_key
        self.cov_avg_sampling = cov_avg_sampling
        self.sample_mode = sample_mode
        # Only use cell_set_len when sample_mode is "set"
        if sample_mode == "set":
            self.cell_set_len = cell_set_len if cell_set_len is not None else 128  # Default to 128 for set mode
        else:
            self.cell_set_len = None  # Force None for cell mode
        self.add_keys = add_keys
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.evaluation = evaluation
        self.predict_controls = predict_controls
        if self.use_mix_pert:
            # Build merged perturbation column and robust control mask
            merged_pert_col = self.merge_cols(
                self.adata.obs,
                cols=[gene_key, drug_key, env_key, crispr_key],
            )

            def _col_to_str(col_name):
                if col_name is None or col_name not in self.adata.obs.columns:
                    return pd.Series([""] * self.adata.n_obs, index=self.adata.obs.index)
                col = self.adata.obs[col_name]
                if pd.api.types.is_categorical_dtype(col):
                    col = col.astype(str)
                return col.fillna("").astype(str)

            # Prefer an existing boolean control column if present; otherwise derive
            if control_val in self.adata.obs.columns and pd.api.types.is_bool_dtype(
                self.adata.obs[control_val]
            ):
                is_control = self.adata.obs[control_val].astype(bool).copy()
            else:
                g_col = _col_to_str(gene_key)
                d_col = _col_to_str(drug_key)
                e_col = _col_to_str(env_key)
                c_col = _col_to_str(crispr_key)

                def _is_control_col(series: pd.Series) -> pd.Series:
                    return (series == "") | (series == str(self.control_val))

                is_control = (
                    _is_control_col(g_col)
                    & _is_control_col(d_col)
                    & _is_control_col(e_col)
                    & _is_control_col(c_col)
                )

            merged_pert_col = pd.Series(merged_pert_col, index=self.adata.obs.index).astype(str)
            merged_pert_col.loc[is_control] = self.control_val
            self.adata.obs['_merged_pert_col_'] = merged_pert_col
            # Ensure a clean boolean control column for downstream splits
            self.adata.obs[self.control_val] = is_control.astype(bool)

        # Only enforce batch_size=1 for set mode (when cell_set_len is used)
        if self.sample_mode == "set" and self.cell_set_len:
            self.val_batch_size=1
            self.test_batch_size=1

        self.transform = transform

        if splitter:
            split_dict = datasplitter.PerturbationDataSplitter.split_dataset(
                splitter_config=splitter,
                obs_dataframe=self.adata.obs,
                perturbation_key=pert_key,
                perturbation_combination_delimiter=perturbation_combination_delimiter,
                perturbation_control_value=control_val,
            )
        else:
            # 如果没有提供 splitter，尝试从 obs 中读取 split 列
            # 这可能来自：
            # 1. 通过 task 加载的外部 CSV（已经添加到 obs 中）
            # 2. 数据文件中已有的 split 列
            if 'split' not in self.adata.obs.columns:
                raise ValueError(
                    "'split' column not found in adata.obs. "
                    "Please provide one of the following:\n"
                    "  1. task parameter (to load split from CSV file)\n"
                    "  2. splitter config (to generate split automatically)\n"
                    "  3. split column in the input data file"
                )
            split = self.adata.obs['split']
            split_dict = {}
            split_dict['train'] = split == 'train'
            split_dict['val'] = split == 'val'
            split_dict['test'] = split == 'test'



        '''全转化为dense格式并确保float32类型'''
        self.adata.X = self.to_dense(self.adata.X).astype(np.float32)
        if self.embedding_key:
            self.adata.obsm[self.embedding_key] = self.to_dense(self.adata.obsm[self.embedding_key]).astype(np.float32)
        if self.raw_counts_key:
            self.adata.layers[self.raw_counts_key] = self.to_dense(self.adata.layers[self.raw_counts_key]).astype(np.float32)


        if inference_top_hvg and 'highly_variable_rank' in self.adata.var:
            self.inference_top_hvg=self.adata.var['highly_variable_rank'].\
                                       fillna(np.inf).values.argsort()[:inference_top_hvg]

        # 在 mix_pert 模式下，使用 _merged_pert_col_ 而不是 pert_key
        if self.use_mix_pert:
            sharing_control_adata=self.adata[self.adata.obs['_merged_pert_col_']==self.control_val]
        else:
            sharing_control_adata=self.adata[self.adata.obs[self.pert_key]==self.control_val]

        train_adata = self.adata[split_dict["train"]]
        train_pert_adata=train_adata[~train_adata.obs[control_val]]
        train_control_adata = train_adata[train_adata.obs[control_val]]
        if sharing_controls:
            train_control_adata=sharing_control_adata

        val_adata = self.adata[split_dict["val"]]
        val_pert_adata = val_adata[~val_adata.obs[control_val]]
        val_control_adata = val_adata[val_adata.obs[control_val]]
        if sharing_controls:
            val_control_adata=sharing_control_adata

        test_adata = self.adata[split_dict["test"]]
        test_pert_adata = test_adata[~test_adata.obs[control_val]]
        test_control_adata = test_adata[test_adata.obs[control_val] ]
        if sharing_controls:
            test_control_adata=sharing_control_adata

        if co_expression_graph_config:
            build_co_expression_graph(**co_expression_graph_config,control_adata=sharing_control_adata)


        '''获取dataset'''
        self.train_dataset =scTrainDataset(
            pert_adata=train_pert_adata,
            control_adata=train_control_adata,
            pert_key=self.pert_key,
            gene_key=self.gene_key,
            crispr_key=self.crispr_key,
            drug_key=self.drug_key,
            env_key=self.env_key,
            use_mix_pert=self.use_mix_pert,
            cov_keys=self.cov_keys,
            transform=self.transform,
            embedding_key=self.embedding_key,
            raw_counts_key=self.raw_counts_key,
            cov_avg_sampling=self.cov_avg_sampling,
            cell_set_len=self.cell_set_len,
            add_keys=self.add_keys,
            predict_controls=self.predict_controls,
            **kwargs
        )

        self.val_dataset=scInferenceDataset(
            pert_adata=val_pert_adata,
            control_adata=val_control_adata,
            pert_key=self.pert_key,
            gene_key=self.gene_key,
            crispr_key=self.crispr_key,
            drug_key=self.drug_key,
            env_key=self.env_key,
            use_mix_pert=self.use_mix_pert,
            cov_keys=self.cov_keys,
            transform=self.transform,
            embedding_key=self.embedding_key,
            raw_counts_key=self.raw_counts_key,
            cell_set_len=self.cell_set_len,
            add_keys=self.add_keys,
            **kwargs
        )

        self.test_dataset = scInferenceDataset(
            pert_adata=test_pert_adata,
            control_adata=test_control_adata,
            pert_key=self.pert_key,
            gene_key=self.gene_key,
            crispr_key=self.crispr_key,
            drug_key=self.drug_key,
            env_key=self.env_key,
            use_mix_pert=self.use_mix_pert,
            cov_keys=self.cov_keys,
            transform=self.transform,
            embedding_key=self.embedding_key,
            raw_counts_key=self.raw_counts_key,
            cell_set_len=self.cell_set_len,
            add_keys=self.add_keys,
            **kwargs
        )

        self.infer_collect_fn=inference_collate()
        self.train_collect_fn=train_collate()
        
    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            collate_fn=self.train_collect_fn,
            drop_last=True,
            shuffle=True,
        )
    def val_dataloader(self) -> DataLoader | None:

        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.val_num_workers,
            collate_fn=self.infer_collect_fn,
            shuffle=False,
        )
    def test_dataloader(self) -> DataLoader | None:

        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.test_num_workers,
            shuffle=False,
            collate_fn=self.infer_collect_fn,
        )
    def to_dense(self,X):
        if isinstance(X, csr_matrix):
            return X.toarray()
        elif issparse(X):
            # 是其他类型稀疏矩阵 (例如 csc_matrix)，可以选择是否转换
            return X.toarray()
        else:
            return X

    def merge_cols(self, obs_df, cols):
        # 处理第一列，先转换为字符串类型避免 Categorical 问题
        first_col = obs_df[cols[0]]
        # 如果是 Categorical，先转换为字符串
        if pd.api.types.is_categorical_dtype(first_col):
            first_col = first_col.astype(str)
        merged = first_col.fillna('').astype(str).to_numpy()

        for c in cols[1:]:
            # 处理后续列，先转换为字符串类型避免 Categorical 问题
            col_data = obs_df[c]
            # 如果是 Categorical，先转换为字符串
            if pd.api.types.is_categorical_dtype(col_data):
                col_data = col_data.astype(str)
            col_values = col_data.fillna('').astype(str).to_numpy()
            merged = merged + "<>" + col_values
        return merged


