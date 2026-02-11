import lightning as L
import torch
from abc import ABC
import pandas as pd
import numpy as np
import anndata as ad
import os
import gc
from hydra.core.hydra_config import HydraConfig
from ...analysis.benchmarks.evaluation import Evaluation
from lightning_utilities.core.apply_func import apply_to_collection

class Batch:
    def __init__(self,batch_dict):
        self.batch_dict = batch_dict
        for k in batch_dict:
            setattr(self, k, batch_dict[k])
    def __getitem__(self,key):
        return self.batch_dict[key]
    def __len__(self):
        return len(list(self.batch_dict.values())[0])
    def __iter__(self):
        for key in self.batch_dict:
            yield key
    def get(self,key,default=None):
        return self.batch_dict.get(key,default)
    def keys(self):
        return self.batch_dict.keys()
    def items(self):
        return self.batch_dict.items()
    def values(self):
        return self.batch_dict.values()


class PerturbationModel(L.LightningModule, ABC):

    def __init__(
        self,
        datamodule: L.LightningDataModule | None = None,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: float | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: float | None = None,
        lr_scheduler_factor: float | None = None,
        lr_scheduler_mode: str | None = None,  # "plateau", "onecycle", "step"
        lr_scheduler_max_lr: float | None = None,  # For OneCycleLR
        lr_scheduler_total_steps: int | None = None,  # For OneCycleLR
        lr_monitor_key: str | None = None,
        use_infer_top_hvgs: bool=False,
        use_mask: bool = False,  # Unified mask switch for both training and evaluation
        **kwargs,
    ):
        super(PerturbationModel, self).__init__()

        self.lr = 1e-3 if lr is None else lr
        self.wd = 1e-5 if wd is None else wd
        self.lr_scheduler_freq = 1 if lr_scheduler_freq is None else lr_scheduler_freq
        self.lr_scheduler_interval = (
            "epoch" if lr_scheduler_interval is None else lr_scheduler_interval
        )
        self.lr_scheduler_patience = (
            5 if lr_scheduler_patience is None else lr_scheduler_patience
        )
        self.lr_scheduler_factor = (
            0.2 if lr_scheduler_factor is None else lr_scheduler_factor
        )
        self.lr_scheduler_mode = lr_scheduler_mode or "plateau"
        self.lr_scheduler_max_lr = lr_scheduler_max_lr or (self.lr * 10)  # Default 10x current lr
        self.lr_scheduler_total_steps = lr_scheduler_total_steps
        self.lr_monitor_key = "val_loss" if lr_monitor_key is None else lr_monitor_key

        self.use_infer_top_hvgs=use_infer_top_hvgs
        self.use_mask = use_mask  # Unified mask switch for training loss and evaluation

        if datamodule is not None:

            self.datamodule = datamodule

            self.use_mix_pert=datamodule.use_mix_pert

            if self.use_mix_pert:
                self.gene_key=datamodule.gene_key
                self.drug_key=datamodule.drug_key
                self.env_key=datamodule.env_key
                self.gene_pert_dim=datamodule.train_dataset.transform.gene_pert_dim
                self.drug_pert_dim=datamodule.train_dataset.transform.drug_pert_dim
                self.env_pert_dim=datamodule.train_dataset.transform.env_pert_dim
                self.crispr_pert_dim=datamodule.train_dataset.transform.crispr_pert_dim
                
                if datamodule.train_dataset.transform.use_covs:
                    self.cov_keys=datamodule.train_dataset.transform.cov_keys
                    self.cov_dims=datamodule.train_dataset.transform.cov_dims
                else:
                    self.cov_keys=[]
                    self.cov_dims={}
            else:
                self.pert_key = datamodule.pert_key
                self.cov_keys = datamodule.cov_keys
                self.cov_dims = {}

            self.result_avg_keys=datamodule.result_avg_keys
            self.control_val = datamodule.control_val

            self.gene_names=datamodule.train_dataset.get_gene_names()
            self.n_genes=len(self.gene_names)
            self.embedding_dim=datamodule.train_dataset.get_embedding_width()

            self.evaluation_config = datamodule.evaluation

            if self.use_infer_top_hvgs and hasattr(datamodule, "inference_top_hvg"):
                self.infer_gene_ids=datamodule.inference_top_hvg
            
            self.mask_type=self.datamodule.mask_type
            self.cellclass_mask_dict=self.datamodule.cellclass_mask_dict

    def _ensure_2d(self, t: torch.Tensor | None) -> torch.Tensor | None:
        """Convert [B,S,G] -> [B*S,G], keep [N,G] as-is."""
        if t is None:
            return None
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        if t.dim() == 2:
            return t
        if t.dim() == 3:
            return t.reshape(-1, t.size(-1))
        raise ValueError(f"Expected 2D or 3D tensor, got dim={t.dim()}, shape={tuple(t.shape)}")

    def _get_mask(self, batch) -> torch.Tensor:
        if not self.use_mask:
            return None
        return batch.mask
    
    def  auto_mse(self, pred, target,mask=None):
        import torch.nn.functional as F
        if mask is not None:
            masked_loss = F.mse_loss(pred*mask, target*mask, reduction="none")
            valid = mask.sum(dim=1)
            loss_per_batch = (masked_loss * mask).sum(dim=1)
            loss = (loss_per_batch / valid).nanmean()
        else:
            loss = F.mse_loss(pred, target, reduction="mean")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )

        if self.lr_scheduler_mode == "onecycle":
            # OneCycleLR: 需要 max_lr 和总步数
            max_lr = self.lr_scheduler_max_lr
            if max_lr is None:
                max_lr = self.lr 

            total_steps = self.lr_scheduler_total_steps
            if total_steps is None and hasattr(self, 'trainer') and self.trainer is not None:
                # 动态计算总步数：steps_per_epoch * max_epochs
                try:
                    if hasattr(self.trainer, 'max_epochs') and hasattr(self.trainer.datamodule, 'train_dataloader'):
                        # 获取训练 dataloader 的长度
                        train_dl = self.trainer.datamodule.train_dataloader()
                        steps_per_epoch = len(train_dl)
                        total_steps = steps_per_epoch * self.trainer.max_epochs
                        print(f"OneCycleLR: dynamically calculated total_steps = {steps_per_epoch} * {self.trainer.max_epochs} = {total_steps}")
                except Exception as e:
                    print(f"Could not calculate total_steps dynamically: {e}")
                    total_steps = 100 * 100  # fallback
            elif total_steps is None:
                # 默认假设 100 个 epoch，每个 epoch 有 100 步
                total_steps = 100 * 100
                print(f"OneCycleLR: using default total_steps = {total_steps}")

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                anneal_strategy='cos',
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",  # OneCycleLR 基于 step
            }
        elif self.lr_scheduler_mode == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=getattr(self, 'lr_scheduler_step_size', None) or 10,  # 每 N 个 epoch 降低一次
                gamma=getattr(self, 'lr_scheduler_gamma', None) or 0.1,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        else:  # Default to ReduceLROnPlateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": self.lr_monitor_key,
                "frequency": self.lr_scheduler_freq,
                "interval": self.lr_scheduler_interval,
            }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_test_start(self) -> None:
        super().on_test_start()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.preds_list=[]
        self.unique_aggregations=set()
        for eval_dict in self.evaluation_config.evaluation_pipelines:
            self.unique_aggregations.add(eval_dict["aggregation"])
        self.summary_metrics=None

    def test_step(
        self,
        data_tuple:tuple[any,pd.DataFrame],
        batch_idx: int,
    ):

        batch,obs_df=data_tuple
        predicted_expression = self.predict(batch)
        # Only convert to numpy for storage at the end (move to CPU only when needed)
        # This minimizes CPU-GPU transfers
        if isinstance(predicted_expression, torch.Tensor):
            # Detach to avoid gradient computation, move to CPU only for storage
            pred_np = predicted_expression.detach().cpu().numpy()
        else:
            pred_np = np.asarray(predicted_expression)
        self.preds_list.append((pred_np, obs_df))

    def predict(self, batch):
        pass

    def _gather_predictions(self):
        """Gather predictions from all distributed ranks."""
        import torch.distributed as dist

        local_expr = np.concatenate([expr for expr, _ in self.preds_list])
        local_obs = pd.concat([obs for _, obs in self.preds_list])

        is_distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_distributed else 1
        rank = dist.get_rank() if is_distributed else 0

        gathered_data = [None for _ in range(world_size)]

        if is_distributed:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            dist.all_gather_object(gathered_data, (local_expr, local_obs))
        else:
            gathered_data[0] = (local_expr, local_obs)

        return gathered_data, is_distributed, rank

    def _build_anndata(self, gathered_data):
        """Build predicted and reference AnnData objects from gathered data."""
        gathered_expr = np.concatenate([expr for expr, _ in gathered_data])
        gathered_obs = pd.concat([obs for _, obs in gathered_data], ignore_index=True)

        gene_names = self.gene_names
        if hasattr(self, 'infer_gene_ids'):
            gene_names = gene_names[self.infer_gene_ids]

        control_adata = self.datamodule.test_dataset.control_adata[:, gene_names]
        pert_adata = self.datamodule.test_dataset.pert_adata[:, gene_names]

        predicted_adata = ad.AnnData(
            X=gathered_expr,
            obs=gathered_obs,
            var=pd.DataFrame(index=gene_names),
        )
        predicted_adata = ad.concat([predicted_adata, control_adata])
        predicted_adata.obs_names_make_unique()
        reference_adata = ad.concat([pert_adata, control_adata])

        return predicted_adata, reference_adata, gene_names

    def _compute_sample_level_pcc(self, predicted_adata, reference_adata, eval_features):
        """
        计算样本级别的 Pearson 相关系数 (PCC)，不进行聚合。
        对每个样本计算预测值和真实值之间的 PCC，然后返回平均值。
        
        Args:
            predicted_adata: 预测的 AnnData 对象
            reference_adata: 参考的 AnnData 对象
            eval_features: 用于评估的基因子集
            
        Returns:
            float: 所有样本 PCC 的平均值
        """
        from scipy.stats import pearsonr
        
        # 获取 eval_features 对应的基因索引
        gene_mask = np.isin(predicted_adata.var_names, eval_features)
        
        # 提取预测和参考的表达矩阵
        pred_X = np.asarray(predicted_adata.X[:, gene_mask])
        ref_X = np.asarray(reference_adata.X[:, gene_mask])
        
        # 确保样本数相同
        n_samples = min(pred_X.shape[0], ref_X.shape[0])
        
        # 计算每个样本的 PCC
        pcc_values = []
        for i in range(n_samples):
            pred_row = pred_X[i].flatten()
            ref_row = ref_X[i].flatten()
            
            # 跳过全零或常数行
            if np.std(pred_row) > 0 and np.std(ref_row) > 0:
                pcc, _ = pearsonr(pred_row, ref_row)
                if not np.isnan(pcc):
                    pcc_values.append(pcc)
        
        # 返回平均 PCC
        if len(pcc_values) > 0:
            return float(np.mean(pcc_values))
        else:
            return 0.0

    def _compute_ot_distances(self, predicted_adata, reference_adata, eval_features):
        """
        计算分布距离指标：
        1) Energy Distance（无超参统计距离）
        2) Sinkhorn Divergence（GeomLoss, Wasserstein-2）

        按 cov_pert 分组后求平均
        """
        import numpy as np
        import torch
        from scipy.spatial.distance import cdist
        from geomloss import SamplesLoss

        # ================= Energy Distance =================
        def energy_distance(X, Y):
            d_xy = cdist(X, Y, metric='euclidean')
            d_xx = cdist(X, X, metric='euclidean')
            d_yy = cdist(Y, Y, metric='euclidean')

            n = len(X)
            m = len(Y)

            term_xy = 2.0 * d_xy.mean()
            term_xx = d_xx.sum() / (n * (n - 1))
            term_yy = d_yy.sum() / (m * (m - 1))

            return term_xy - term_xx - term_yy

        # ================= Sinkhorn (GeomLoss) =================
        device = "cuda" if torch.cuda.is_available() else "cpu"

        sinkhorn_loss = SamplesLoss(
            loss="sinkhorn",   # Sinkhorn divergence（已 debiased）
            p=2,               # squared Euclidean cost → Wasserstein-2
            blur=0.05,         # 正则强度（默认稳健）
            scaling=0.9,
            backend="tensorized"
        )

        def sinkhorn_distance_geomloss(X, Y):
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
            return sinkhorn_loss(X_t, Y_t).item()

        # ================= 数据准备 =================
        gene_mask = np.isin(predicted_adata.var_names, eval_features)

        pred_X = np.asarray(predicted_adata.X[:, gene_mask])
        ref_X = np.asarray(reference_adata.X[:, gene_mask])

        pert_col = '_merged_pert_col_' if self.use_mix_pert else self.pert_key
        cov_cols = [k for k in self.cov_keys if k not in self.result_avg_keys]

        def get_group_key(obs, idx):
            parts = [str(obs[pert_col].iloc[idx])]
            for c in cov_cols:
                parts.append(str(obs[c].iloc[idx]))
            return "_".join(parts)

        # 分组
        pred_groups = {}
        for i in range(len(predicted_adata)):
            key = get_group_key(predicted_adata.obs, i)
            pred_groups.setdefault(key, []).append(i)

        ref_groups = {}
        for i in range(len(reference_adata)):
            key = get_group_key(reference_adata.obs, i)
            ref_groups.setdefault(key, []).append(i)

        # ================= 逐组计算 =================
        energy_values = []
        sinkhorn_values = []

        common_groups = set(pred_groups.keys()) & set(ref_groups.keys())

        for group_key in common_groups:
            pred_idx = pred_groups[group_key]
            ref_idx = ref_groups[group_key]

            pred_samples = pred_X[pred_idx]
            ref_samples = ref_X[ref_idx]

            if len(pred_samples) < 2 or len(ref_samples) < 2:
                continue

            try:
                e_dist = energy_distance(pred_samples, ref_samples)
                s_dist = sinkhorn_distance_geomloss(pred_samples, ref_samples)

                energy_values.append(float(e_dist))
                sinkhorn_values.append(float(s_dist))

            except Exception:
                continue

        mean_energy = float(np.mean(energy_values)) if energy_values else 0.0
        mean_sinkhorn = float(np.mean(sinkhorn_values)) if sinkhorn_values else 0.0

        return mean_energy, mean_sinkhorn

    def _compute_pca_metrics(self, predicted_adata, reference_adata, eval_features, model_name, n_components=50):
        """
        在 PCA 降维空间下计算指标：Evaluation 聚合指标、样本级别 PCC、Energy Distance、Sinkhorn Divergence。
        
        使用参考数据拟合 PCA，然后将预测数据和参考数据都投影到该 PCA 空间中进行评估。
        复用 Evaluation、_compute_sample_level_pcc 和 _compute_ot_distances 函数进行计算。
        所有指标名称都带有 "pca_" 前缀。
        
        Args:
            predicted_adata: 预测的 AnnData 对象
            reference_adata: 参考的 AnnData 对象
            eval_features: 用于评估的基因子集
            model_name: 模型名称，用于 Evaluation
            n_components: PCA 降维的维度数（默认 50）
            
        Returns:
            dict: 包含以下指标的字典（所有指标名都带 "pca_" 前缀）
                - pca_{metric}_{aggr}: Evaluation 计算的聚合指标
                - pca_{metric}_rank_{aggr}: Evaluation 计算的排名指标（如配置了 rank）
                - pca_pcc_no_aggr: PCA 空间下的样本级别 PCC
                - pca_energy_distance: PCA 空间下的 Energy Distance
                - pca_sinkhorn_divergency: PCA 空间下的 Sinkhorn Divergence
        """
        from sklearn.decomposition import PCA
        
        # ================= 数据准备 =================
        gene_mask = np.isin(predicted_adata.var_names, eval_features)
        pred_X = np.asarray(predicted_adata.X[:, gene_mask])
        ref_X = np.asarray(reference_adata.X[:, gene_mask])
        
        # 确定实际使用的 PCA 维度（不超过特征数和样本数）
        actual_n_components = min(n_components, pred_X.shape[1], ref_X.shape[0] - 1)
        
        # ================= PCA 降维 =================
        # 使用参考数据拟合 PCA
        pca = PCA(n_components=actual_n_components)
        ref_pca = pca.fit_transform(ref_X)
        pred_pca = pca.transform(pred_X)
        
        # ================= 构建 PCA 空间的临时 AnnData =================
        # 创建 PCA 维度的特征名（作为 eval_features 传递给复用函数）
        pca_feature_names = [f"PC{i+1}" for i in range(actual_n_components)]
        
        # 构建包含 PCA 数据的临时 AnnData，保留原始 obs 信息
        pred_pca_adata = ad.AnnData(
            X=pred_pca,
            obs=predicted_adata.obs.copy(),
            var=pd.DataFrame(index=pca_feature_names),
        )
        ref_pca_adata = ad.AnnData(
            X=ref_pca,
            obs=reference_adata.obs.copy(),
            var=pd.DataFrame(index=pca_feature_names),
        )
        
        pca_metrics_dict = {}
        
        # ================= 1. 使用 Evaluation 计算聚合指标（PCA 空间）=================
        cov_cols = [k for k in self.cov_keys if k not in self.result_avg_keys]
        
        ev = Evaluation(
            model_adatas=[pred_pca_adata],
            model_names=[model_name],
            ref_adata=ref_pca_adata,
            pert_col='_merged_pert_col_' if self.use_mix_pert else self.pert_key,
            cov_cols=cov_cols,
            ctrl=self.control_val,
            features=pca_feature_names,
        )
        
        for aggr in self.unique_aggregations:
            ev.aggregate(aggr_method=aggr)
        
        for eval_dict in self.evaluation_config.evaluation_pipelines:
            aggr = eval_dict["aggregation"]
            metric = eval_dict["metric"]
            ev.evaluate(aggr_method=aggr, metric=metric)
            
            df = ev.evals[aggr][metric].copy()
            avg = df.groupby("model").mean("metric")
            pca_metrics_dict[f"pca_{metric}_{aggr}"] = avg["metric"].iloc[0]
            
            if eval_dict.get("rank"):
                ev.evaluate_pairwise(aggr_method=aggr, metric=metric)
                ev.evaluate_rank(aggr_method=aggr, metric=metric)
                rank_df = ev.rank_evals[aggr][metric].copy()
                avg_rank = rank_df.groupby("model").mean("rank")
                pca_metrics_dict[f"pca_{metric}_rank_{aggr}"] = avg_rank["rank"].iloc[0]
        
        # ================= 2. 样本级别 PCC（PCA 空间）=================
        pca_pcc = self._compute_sample_level_pcc(pred_pca_adata, ref_pca_adata, pca_feature_names)
        pca_metrics_dict["pca_pcc_no_aggr"] = pca_pcc
        
        # ================= 3. OT 距离（PCA 空间）=================
        pca_energy, pca_sinkhorn = self._compute_ot_distances(pred_pca_adata, ref_pca_adata, pca_feature_names)
        pca_metrics_dict["pca_energy_distance"] = pca_energy
        pca_metrics_dict["pca_sinkhorn_divergency"] = pca_sinkhorn
        
        return pca_metrics_dict

    def _compute_deg_metrics(self, predicted_adata, reference_adata, eval_features, top_n_deg=50):
        """
        计算 DEG（差异表达基因）相关指标：IoU, Precision, Recall。
        
        对于每个 perturbation：
        1. 在 predicted_adata 中计算该 pert 相对于 control 的 top DEG（按表达差异绝对值排序）
        2. 在 reference_adata 中计算该 pert 相对于 control 的 top DEG
        3. 计算两个 DEG 集合的 IoU, Precision, Recall
        4. 在所有 pert 上求平均
        
        Args:
            predicted_adata: 预测的 AnnData 对象
            reference_adata: 参考的 AnnData 对象
            eval_features: 用于评估的基因子集
            top_n_deg: 每个 pert 选取的 top DEG 数量（默认 50）
            
        Returns:
            tuple: (mean_iou, mean_precision, mean_recall)
                - IoU = |pred_DEG ∩ ref_DEG| / |pred_DEG ∪ ref_DEG|
                - Precision = |pred_DEG ∩ ref_DEG| / |pred_DEG|  (预测的 DEG 中有多少在真实 DEG 中)
                - Recall = |pred_DEG ∩ ref_DEG| / |ref_DEG|  (真实的 DEG 中有多少被预测到)
        """
        # 获取 pert 列名
        pert_col = '_merged_pert_col_' if self.use_mix_pert else self.pert_key
        
        # 获取 eval_features 对应的基因索引
        gene_mask = np.isin(predicted_adata.var_names, eval_features)
        
        # 提取表达矩阵
        pred_X = np.asarray(predicted_adata.X[:, gene_mask])
        ref_X = np.asarray(reference_adata.X[:, gene_mask])
        
        # 获取基因名
        genes = np.array(predicted_adata.var_names)[gene_mask]
        
        # 分离 control 和 perturbation 样本
        pred_ctrl_mask = (predicted_adata.obs[pert_col] == self.control_val).values
        ref_ctrl_mask = (reference_adata.obs[pert_col] == self.control_val).values
        
        # 计算 control 的平均表达
        pred_ctrl_mean = pred_X[pred_ctrl_mask].mean(axis=0) if pred_ctrl_mask.sum() > 0 else np.zeros(pred_X.shape[1])
        ref_ctrl_mean = ref_X[ref_ctrl_mask].mean(axis=0) if ref_ctrl_mask.sum() > 0 else np.zeros(ref_X.shape[1])
        
        # 获取所有 perturbation（排除 control）
        all_perts = set(predicted_adata.obs[pert_col].unique()) | set(reference_adata.obs[pert_col].unique())
        perts = [p for p in all_perts if p != self.control_val]
        
        iou_values = []
        precision_values = []
        recall_values = []
        
        for pert in perts:
            # 获取该 pert 的样本 mask
            pred_pert_mask = (predicted_adata.obs[pert_col] == pert).values
            ref_pert_mask = (reference_adata.obs[pert_col] == pert).values
            
            # 跳过不存在的 pert
            if pred_pert_mask.sum() == 0 or ref_pert_mask.sum() == 0:
                continue
            
            # 计算该 pert 的平均表达
            pred_pert_mean = pred_X[pred_pert_mask].mean(axis=0)
            ref_pert_mean = ref_X[ref_pert_mask].mean(axis=0)
            
            # 计算差异（相对于 control 的绝对差值）
            pred_diff = np.abs(pred_pert_mean - pred_ctrl_mean)
            ref_diff = np.abs(ref_pert_mean - ref_ctrl_mean)
            
            # 确定实际使用的 top_n（不超过基因总数）
            actual_top_n = min(top_n_deg, len(genes))
            
            # 选取 top N DEG（按差异绝对值排序，取最大的 N 个）
            pred_top_indices = np.argsort(pred_diff)[-actual_top_n:]
            ref_top_indices = np.argsort(ref_diff)[-actual_top_n:]
            
            pred_deg_set = set(genes[pred_top_indices])
            ref_deg_set = set(genes[ref_top_indices])
            
            # 计算 IoU, Precision, Recall
            intersection = len(pred_deg_set & ref_deg_set)
            union = len(pred_deg_set | ref_deg_set)
            
            iou = intersection / union if union > 0 else 0.0
            # Precision: 预测的 DEG 中有多少在真实 DEG 中
            precision = intersection / len(pred_deg_set) if len(pred_deg_set) > 0 else 0.0
            # Recall: 真实的 DEG 中有多少被预测到
            recall = intersection / len(ref_deg_set) if len(ref_deg_set) > 0 else 0.0
            
            iou_values.append(iou)
            precision_values.append(precision)
            recall_values.append(recall)
        
        mean_iou = float(np.mean(iou_values)) if iou_values else 0.0
        mean_precision = float(np.mean(precision_values)) if precision_values else 0.0
        mean_recall = float(np.mean(recall_values)) if recall_values else 0.0
        
        return mean_iou, mean_precision, mean_recall

    def _run_evaluation(self, predicted_adata, reference_adata, eval_features, model_name):
        """Run evaluation and return summary metrics."""
        if eval_features is None:
            eval_features = reference_adata.var_names
            
        cov_cols = [k for k in self.cov_keys if k not in self.result_avg_keys]
        
        ev = Evaluation(
            model_adatas=[predicted_adata],
            model_names=[model_name],
            ref_adata=reference_adata,
            pert_col='_merged_pert_col_' if self.use_mix_pert else self.pert_key,
            cov_cols=cov_cols,
            ctrl=self.control_val,
            features=eval_features,
        )

        for aggr in self.unique_aggregations:
            ev.aggregate(aggr_method=aggr)

        summary_metrics_dict = {}
        for eval_dict in self.evaluation_config.evaluation_pipelines:
            aggr = eval_dict["aggregation"]
            metric = eval_dict["metric"]
            ev.evaluate(aggr_method=aggr, metric=metric)

            df = ev.evals[aggr][metric].copy()
            avg = df.groupby("model").mean("metric")
            summary_metrics_dict[f"{metric}_{aggr}"] = avg["metric"]

            if eval_dict.get("rank"):
                ev.evaluate_pairwise(aggr_method=aggr, metric=metric)
                ev.evaluate_rank(aggr_method=aggr, metric=metric)
                rank_df = ev.rank_evals[aggr][metric].copy()
                avg_rank = rank_df.groupby("model").mean("rank")
                summary_metrics_dict[f"{metric}_rank_{aggr}"] = avg_rank["rank"]

        # ====== 样本级别 PCC (no-aggr) 计算，不使用 evaluation 包 ======
        sample_pcc = self._compute_sample_level_pcc(predicted_adata, reference_adata, eval_features)
        summary_metrics_dict["pcc_no_aggr"] = pd.Series({model_name: sample_pcc})

        # ====== OT 距离计算 (MMD, Sinkhorn)，按 cov_pert 分组后求平均 ======
        try:
            mean_mmd, mean_sinkhorn = self._compute_ot_distances(predicted_adata, reference_adata, eval_features)
            summary_metrics_dict["energy_distance"] = pd.Series({model_name: mean_mmd})
            summary_metrics_dict["sinkhorn_divergency"] = pd.Series({model_name: mean_sinkhorn})
        except Exception as e:
            print(f"Warning: OT distance computation failed: {e}")
            summary_metrics_dict["energy_distance"] = pd.Series({model_name: 0.0})
            summary_metrics_dict["sinkhorn_divergency"] = pd.Series({model_name: 0.0})

        # ====== PCA 空间下的指标计算 (Evaluation聚合指标, PCC, Energy Distance, Sinkhorn) ======
        try:
            pca_metrics = self._compute_pca_metrics(
                predicted_adata, reference_adata, eval_features, model_name, n_components=50
            )
            # 将所有 PCA 指标添加到 summary_metrics_dict（已带 "pca_" 前缀）
            for metric_name, metric_value in pca_metrics.items():
                summary_metrics_dict[metric_name] = pd.Series({model_name: metric_value})
        except Exception as e:
            print(f"Warning: PCA metrics computation failed: {e}")
            # 设置默认的 PCA 指标为 0.0
            summary_metrics_dict["pca_pcc_no_aggr"] = pd.Series({model_name: 0.0})
            summary_metrics_dict["pca_energy_distance"] = pd.Series({model_name: 0.0})
            summary_metrics_dict["pca_sinkhorn_divergency"] = pd.Series({model_name: 0.0})

        # ====== DEG 指标计算 (IoU, Precision, Recall)，按 pert 分组后求平均 ======
        try:
            deg_iou, deg_precision, deg_recall = self._compute_deg_metrics(
                predicted_adata, reference_adata, eval_features, top_n_deg=50
            )
            summary_metrics_dict["deg_iou"] = pd.Series({model_name: deg_iou})
            summary_metrics_dict["deg_precision"] = pd.Series({model_name: deg_precision})
            summary_metrics_dict["deg_recall"] = pd.Series({model_name: deg_recall})
        except Exception as e:
            print(f"Warning: DEG metrics computation failed: {e}")
            summary_metrics_dict["deg_iou"] = pd.Series({model_name: 0.0})
            summary_metrics_dict["deg_precision"] = pd.Series({model_name: 0.0})
            summary_metrics_dict["deg_recall"] = pd.Series({model_name: 0.0})

        summary_metrics = pd.DataFrame(summary_metrics_dict).T.applymap(
            lambda x: float(np.format_float_positional(
                x, precision=4, unique=False, fractional=False, trim="k"
            ))
        )

        return ev, summary_metrics, summary_metrics_dict

    def _get_output_dir(self):
        """Get output directory from hydra or logger."""
        try:
            return HydraConfig.get().runtime.output_dir
        except Exception:
            if self.logger is not None:
                logger_obj = self.logger[0] if isinstance(self.logger, (list, tuple)) and len(self.logger) > 0 else self.logger
                return getattr(logger_obj, "save_dir", None) or self.evaluation_config.save_dir
            return self.evaluation_config.save_dir

    def _save_results(self, ev, summary_metrics, predicted_adata, output_dir):
        """Save evaluation results, summary metrics, and predictions."""
        summary_dir = os.path.join(output_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(self.evaluation_config.save_dir, exist_ok=True)

        ev.save(self.evaluation_config.save_dir)

        # Save summary CSV
        csv_path = os.path.join(self.evaluation_config.save_dir, "summary.csv")
        summary_metrics.to_csv(csv_path, index_label="metric")

        ckpt_type = getattr(self, "current_test_ckpt_type", None)
        suffix = f"_{ckpt_type}" if ckpt_type and ckpt_type != "unknown" else ""
        
        summary_csv_path = os.path.join(summary_dir, f"summary_metrics{suffix}.csv")
        summary_metrics.to_csv(summary_csv_path, index_label="metric")

        # Save predictions
        pred_h5ad_path = os.path.join(summary_dir, f"predictions{suffix}.h5ad")
        try:
            predicted_adata.write(pred_h5ad_path)
        except OSError as e:
            if e.errno == 122:
                print(f"WARNING: Disk quota exceeded, skipping prediction save.")
            else:
                print(f"WARNING: Failed to save predictions: {e}")
        except Exception as e:
            print(f"WARNING: Failed to save predictions: {e}")

        return csv_path, summary_dir

    def _log_to_wandb(self, summary_metrics_dict):
        """Log metrics to wandb if enabled."""
        save_preds_to_wandb = self.evaluation_config.get("save_predictions_to_wandb", False)
        if not save_preds_to_wandb or self.logger is None:
            return

        try:
            loggers = self.logger if isinstance(self.logger, (list, tuple)) else [self.logger]
            for logger in loggers:
                if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                    test_metrics_dict = {f"test_{k}": float(v) for k, v in summary_metrics_dict.items()}
                    logger.experiment.log(test_metrics_dict)
        except Exception:
            pass  # Silently skip wandb logging errors

    def _run_evaluation_per_cellclass(self, predicted_adata, reference_adata, model_name, output_dir):
        """
        针对每个 cellclass 分别运行评估（使用 obs['cellclass'] 和 cellclass_mask_dict）。
        
        此函数根据数据中是否存在 cellclass 分组信息，决定采用不同的评估策略：
        - 若无 cellclass 信息：对全量数据进行统一评估
        - 若有 cellclass 信息：按每个 cellclass 分组评估，并汇总结果
        
        Args:
            predicted_adata (AnnData): 模型预测的表达矩阵，obs 中应包含 'cellclass' 列（如适用）
            reference_adata (AnnData): 参考（真实）表达矩阵，obs 中应包含 'cellclass' 列（如适用）
            model_name (str): 模型名称，用于评估报告中标识模型
            output_dir (str): 输出目录的根路径，评估结果将保存于此
        
        Returns:
            tuple: (summary_metrics, summary_metrics_dict, csv_path)
                - summary_metrics (pd.DataFrame | None): 所有 cellclass 的平均评估指标 DataFrame
                - summary_metrics_dict (dict): 按 cellclass 分组的指标字典 {cellclass: {metric: value}}
                - csv_path (str | None): 汇总 CSV 文件的保存路径
        
        输出目录结构（当有 cellclass 分组时）:
        -----------------------------------------------
        {output_dir}/
        ├── cellclass_evaluation/           # 按 cellclass 分组的评估结果
        │   ├── {cellclass_1}/              # 第一个 cellclass 的结果目录
        │   │   ├── summary.csv             # 该 cellclass 的评估指标摘要
        │   │   ├── predictions.h5ad        # 该 cellclass 的预测结果 AnnData
        │   │   └── ... (Evaluation.save() 生成的其他文件)
        │   ├── {cellclass_2}/              # 第二个 cellclass 的结果目录
        │   │   └── ...
        │   └── ...
        └── summary/                        # 汇总目录
            ├── summary_by_cellclass.csv    # 所有 cellclass 的详细指标（含 cellclass 列）
            └── summary_avg.csv             # 所有 cellclass 的平均指标
        
        输出目录结构（当无 cellclass 分组时）:
        -----------------------------------------------
        {output_dir}/
        └── summary/
            ├── summary_metrics_{ckpt_type}.csv  # 评估指标摘要
            └── predictions_{ckpt_type}.h5ad     # 预测结果 AnnData
        {evaluation_config.save_dir}/
            ├── summary.csv                      # 评估指标摘要（复制）
            └── ... (Evaluation.save() 生成的其他文件)
        """
        # ========== 步骤1：检查是否需要按 cellclass 分组评估 ==========
        # 如果 reference_adata.obs 中没有 'cellclass' 列，或者没有配置 cellclass_mask_dict，
        # 则采用全量评估模式，直接对所有数据进行统一评估
        if 'cellclass' not in reference_adata.obs or not getattr(self, 'cellclass_mask_dict', None):
            ev, metrics, mdict = self._run_evaluation(
                predicted_adata, reference_adata, None, model_name
            )
            return metrics, mdict, self._save_results(ev, metrics, predicted_adata, output_dir)[0]
        
        # ========== 步骤2：获取预测和参考数据中共有的 cellclass 列表 ==========
        # 取交集确保只评估两者都存在的 cellclass，并按字母顺序排序
        groups = sorted(set(predicted_adata.obs['cellclass'].unique()) & set(reference_adata.obs['cellclass'].unique()))
        all_metrics, all_mdict = {}, {}  # 存储每个 cellclass 的评估结果
        cc_dir = os.path.join(output_dir, "cellclass_evaluation")  # cellclass 评估结果的根目录
        
        # ========== 步骤3：遍历每个 cellclass 进行独立评估 ==========
        for cc in groups:
            # 3.1 按 cellclass 筛选子集
            _cellclass_mask=self.cellclass_mask_dict.get(cc,None)
            
            pred_sub = predicted_adata[predicted_adata.obs['cellclass'] == cc]
            ref_sub = reference_adata[reference_adata.obs['cellclass'] == cc]
            
            # 3.2 跳过空数据集（无预测或无参考样本）
            if pred_sub.n_obs == 0 or ref_sub.n_obs == 0:
                continue
            
            try:
                # 3.4 对该 cellclass 子集运行评估
                # 确保 _cellclass_mask 是一维 numpy 数组，用于正确索引 gene_names
                if _cellclass_mask is not None:
                    _cellclass_mask = np.asarray(_cellclass_mask).flatten()
                gene_names_arr = np.array(self.gene_names)
                eval_gene_names = gene_names_arr[_cellclass_mask] if _cellclass_mask is not None else gene_names_arr
                ev, metrics, mdict = self._run_evaluation(pred_sub, ref_sub,
                                                          eval_gene_names, model_name)
                # 3.5 创建该 cellclass 的保存目录（将 '/' 替换为 '_' 避免路径问题）
                save_dir = os.path.join(cc_dir, str(cc).replace('/', '_'))
                os.makedirs(save_dir, exist_ok=True)
                
                # 3.6 保存评估结果
                ev.save(save_dir)  # 保存 Evaluation 对象的完整结果
                metrics.to_csv(os.path.join(save_dir, "summary.csv"), index_label="metric")  # 保存指标摘要
                
                # 3.7 尝试保存预测的 AnnData（可能因磁盘空间等问题失败，静默处理）
                try:
                    pred_sub.write(os.path.join(save_dir, "predictions.h5ad"))
                except Exception:
                    pass
                
                # 3.8 记录该 cellclass 的评估结果
                all_metrics[cc], all_mdict[cc] = metrics, mdict
                print(f"[{cc}] genes={pred_sub.n_vars}, saved to {save_dir}")
                
            except Exception as e:
                # 记录失败的 cellclass（不中断整体流程）
                print(f"[{cc}] failed: {e}")
        
        # ========== 步骤4：检查是否有成功的评估结果 ==========
        if not all_metrics:
            return None, {}, None
        
        # ========== 步骤5：汇总所有 cellclass 的评估结果 ==========
        summary_dir = os.path.join(output_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # 5.1 保存按 cellclass 分组的详细指标表（每行带有 cellclass 标签）
        pd.concat([df.assign(cellclass=cc) for cc, df in all_metrics.items()]).to_csv(
            os.path.join(summary_dir, "summary_by_cellclass.csv"), index_label="metric"
        )
        
        # 5.2 计算并保存所有 cellclass 的平均指标
        avg = pd.concat(all_metrics.values()).groupby(level=0).mean()
        avg.to_csv(os.path.join(summary_dir, "summary_avg.csv"), index_label="metric")
        
        return avg, all_mdict, os.path.join(summary_dir, "summary_by_cellclass.csv")

    def on_test_end(self) -> None:
        """
        测试阶段结束后的回调函数，负责汇总预测结果并进行评估。
        
        此函数是 PyTorch Lightning 的生命周期钩子，在所有测试批次处理完成后自动调用。
        主要完成以下工作：
        1. 从所有分布式进程收集预测结果
        2. 构建 AnnData 对象并运行评估
        3. 保存评估结果和预测数据
        4. 同步评估指标到所有进程
        5. 清理内存资源
        
        分布式训练说明:
        ----------------
        - 在多 GPU/多节点训练时，每个进程只持有部分预测结果
        - 本函数通过 all_gather_object 收集所有进程的预测结果
        - 评估仅在 rank 0 进程执行，然后广播结果给其他进程
        - 使用 barrier 同步确保所有进程在继续前达到一致状态
        
        完整输出目录结构:
        ----------------
        {output_dir}/                           # 由 Hydra 或 Logger 决定的根目录
        ├── cellclass_evaluation/               # 按 cellclass 分组的评估结果（如适用）
        │   ├── {cellclass_1}/
        │   │   ├── summary.csv                 # 该 cellclass 的评估指标
        │   │   ├── predictions.h5ad            # 该 cellclass 的预测 AnnData
        │   │   ├── aggregations/               # Evaluation.save() 生成的聚合结果
        │   │   │   ├── {aggr_method_1}.h5ad
        │   │   │   └── ...
        │   │   └── evaluations/                # Evaluation.save() 生成的评估详情
        │   │       ├── {aggr_method}_{metric}.csv
        │   │       └── ...
        │   ├── {cellclass_2}/
        │   │   └── ...
        │   └── ...
        └── summary/                            # 汇总目录
            ├── summary_by_cellclass.csv        # 所有 cellclass 的详细指标（有 cellclass 分组时）
            ├── summary_avg.csv                 # 所有 cellclass 的平均指标（有 cellclass 分组时）
            ├── summary_metrics_{ckpt_type}.csv # 评估指标摘要（无 cellclass 分组时）
            └── predictions_{ckpt_type}.h5ad    # 完整预测结果（无 cellclass 分组时）
        
        {evaluation_config.save_dir}/           # 配置指定的评估保存目录
        ├── summary.csv                         # 评估指标摘要副本
        ├── aggregations/                       # 聚合后的表达数据
        │   └── ...
        └── evaluations/                        # 评估详细结果
            └── ...
        
        Attributes Modified:
            self.summary_metrics: 更新为最终评估指标 DataFrame
            self.preds_list: 清空以释放内存
        
        Note:
            - 此函数依赖 on_test_start() 初始化的 self.preds_list
            - 依赖 test_step() 在每个批次后填充的预测结果
        """
        import torch.distributed as dist
        super().on_test_end()
        
        # ========== 步骤1：准备工作 ==========
        # 从类名提取模型名称（用于评估报告标识）
        # 例如：<class 'perturbench.models.MyModel'> -> "MyModel"
        model_name = str(self.__class__).split(".")[-1].replace("'>", "")
        
        # 收集所有分布式进程的预测结果
        # gathered_data: List[(np.ndarray, pd.DataFrame)]，每个元素是一个进程的 (表达矩阵, obs DataFrame)
        # is_distributed: bool，是否处于分布式训练模式
        # rank: int，当前进程的 rank（0 为主进程）
        gathered_data, is_distributed, rank = self._gather_predictions()
        summary_metrics, summary_metrics_dict = None, {}

        # ========== 步骤2：在主进程 (rank 0) 执行评估 ==========
        # 仅在 rank 0 执行评估，避免重复计算和 I/O 冲突
        if rank == 0:
            # 2.1 构建 AnnData 对象
            # predicted_adata: 模型预测的表达矩阵 + 对照组数据
            # reference_adata: 真实扰动数据 + 对照组数据
            predicted_adata, reference_adata, _ = self._build_anndata(gathered_data)
            
            # 2.2 获取输出目录（优先从 Hydra 配置获取，否则从 Logger 或配置文件获取）
            output_dir = self._get_output_dir()

            # 2.3 按 cellclass 分组运行评估（或全量评估，取决于数据配置）
            summary_metrics, summary_metrics_dict, csv_path = self._run_evaluation_per_cellclass(
                predicted_adata, reference_adata, model_name, output_dir
            )

            # 2.4 打印评估摘要（如果配置允许）
            if self.evaluation_config.print_summary and summary_metrics is not None:
                print(f"\n===== Average Summary Metrics =====\n{summary_metrics}\n")
            if csv_path:
                print(f"Evaluation finished. Results saved to {csv_path}")

            # 2.5 记录到 Weights & Biases（如果启用）
            # 先记录每个 cellclass 的指标，再记录所有 cellclass 的平均指标
            if summary_metrics_dict:
                # 2.5.1 记录每个 cellclass 的独立指标（带 cellclass 前缀）
                for cc, cc_metrics in summary_metrics_dict.items():
                    cc_prefix = str(cc).replace('/', '_')
                    self._log_to_wandb({f"{cc_prefix}/{k}": v for k, v in cc_metrics.items()})

                # 2.5.2 计算并记录所有 cellclass 的平均指标
                avg_dict = {}
                for cc_metrics in summary_metrics_dict.values():
                    for k, v in cc_metrics.items():
                        avg_dict.setdefault(k, []).append(float(v) if hasattr(v, '__float__') else v)
                self._log_to_wandb({f"mean/{k}": np.mean(v) for k, v in avg_dict.items()})

        # ========== 步骤3：同步评估结果到所有进程 ==========
        # 在分布式场景下，确保所有进程都能访问 summary_metrics
        # 这对于后续可能依赖评估结果的逻辑很重要（如模型选择、早停等）
        if is_distributed:
            # 使用 broadcast_object_list 从 rank 0 广播 summary_metrics 到所有进程
            obj_list = [summary_metrics]
            dist.broadcast_object_list(obj_list, src=0)
            self.summary_metrics = obj_list[0]
            
            # 同步屏障：确保所有进程都收到广播后再继续
            # 防止快的进程过早进入下一阶段（如开始新的训练轮次）
            dist.barrier()
        else:
            # 非分布式模式，直接赋值即可
            self.summary_metrics = summary_metrics

        # ========== 步骤4：清理资源 ==========
        # 释放预测列表占用的内存（可能很大，特别是测试集较大时）
        self.preds_list = []
        
        # 手动触发垃圾回收，确保内存及时释放
        # 在 GPU 训练场景下尤为重要，避免后续操作因内存不足而失败
        gc.collect()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):

        # Case 1: dict batch
        if isinstance(batch, dict):
            batch_dict=apply_to_collection(
                batch,
                torch.Tensor,
                lambda x: x.to(device)
            )
            return Batch(batch_dict)

        # Case 2: (dict, pandas_df)
        if isinstance(batch, tuple):
            batch_dict, obs_df = batch

            batch_dict = apply_to_collection(
                batch_dict,
                torch.Tensor,
                lambda x: x.to(device)
            )

            # 注意：obs_df 不要递归 to(device)
            return Batch(batch_dict), obs_df

        return batch
