"""
BSD 3-Clause License

Copyright (c) 2024, <anonymized authors of NeurIPS submission #1306>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

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

    def _compute_masked_pcc(
        self,
        predictions: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute Pearson Correlation Coefficient, optionally with mask.

        Args:
            predictions: Predicted expression values [batch_size, n_genes]
            observed: Observed expression values [batch_size, n_genes]
            mask: Optional expression mask [batch_size, n_genes], 1 for valid genes

        Returns:
            Mean PCC across batch
        """
        x = predictions
        y = observed

        if mask is not None:
            # Ensure mask is on the same device as predictions
            mask = mask.to(x.device)
            # Masked PCC: only compute on expressed genes
            valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # Avoid division by zero
            x_mean = (x * mask).sum(dim=1, keepdim=True) / valid_counts
            y_mean = (y * mask).sum(dim=1, keepdim=True) / valid_counts
            x_centered = (x - x_mean) * mask
            y_centered = (y - y_mean) * mask
        else:
            # Full gene PCC
            x_mean = x.mean(dim=1, keepdim=True)
            y_mean = y.mean(dim=1, keepdim=True)
            x_centered = x - x_mean
            y_centered = y - y_mean

        num = (x_centered * y_centered).sum(dim=1)
        den = (
            x_centered.pow(2).sum(dim=1).sqrt()
            * y_centered.pow(2).sum(dim=1).sqrt()
            + 1e-8
        )
        pcc_per_cell = num / den
        return pcc_per_cell.mean()

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

        # Get pert_cell_counts from batch (handle both dict and Batch object)
        if isinstance(batch, dict):
            pert_cell_counts = batch.get("pert_cell_counts", None)
        else:
            pert_cell_counts = getattr(batch, "pert_cell_counts", None)

        if pert_cell_counts is None:
            return None

        # Handle sparse tensors - convert to dense for element-wise operations
        if pert_cell_counts.is_sparse:
            pert_cell_counts = pert_cell_counts.to_dense()

        # Simple and effective: mask = (pert_cell_counts != 0)
        mask = (pert_cell_counts != 0).float()

        return mask

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
        
        # Compute test loss with mask (similar to validation_step)
        # Get ground truth labels
        if isinstance(batch, dict):
            y = batch.get('pert_cell_counts', None)
        else:
            y = getattr(batch, 'pert_cell_counts', None)
        
        if y is not None and predicted_expression is not None:
            # Get device from y (usually GPU)
            device = y.device
            
            # Ensure predictions are torch.Tensor and on same device as y (GPU)
            # All computations will be done on GPU for efficiency
            if not isinstance(predicted_expression, torch.Tensor):
                predicted_expression = torch.as_tensor(predicted_expression, device=device, dtype=y.dtype)
            else:
                predicted_expression = predicted_expression.to(device, non_blocking=True)
            
            # Get expression mask using unified method from base class
            mask = self._get_mask(batch)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            
            # ---- normalize shapes to [N, G] ----
            pred_2d = self._ensure_2d(predicted_expression)
            y_2d = self._ensure_2d(y)
            mask_2d = self._ensure_2d(mask) if mask is not None else None

            # Safety check: must match exactly on gene dim
            if pred_2d.shape != y_2d.shape:
                raise ValueError(f"Shape mismatch in test_step: pred={pred_2d.shape}, y={y_2d.shape}")

            # ---- masked MSE on [N,G] ----
            mse = (pred_2d - y_2d).pow(2)
            if mask_2d is not None:
                valid = mask_2d.sum(dim=1).clamp_min(1.0)
                test_loss_per_row = (mse * mask_2d).sum(dim=1)
                test_loss = (test_loss_per_row / valid).nanmean()
            else:
                test_loss = mse.mean()
            
            self.log("test_loss", test_loss, prog_bar=True, logger=True,
                     batch_size=y_2d.shape[0], on_step=False, on_epoch=True)
            
            # ---- PCC on [N,G] ----
            test_pcc = self._compute_masked_pcc(pred_2d, y_2d, mask_2d)
            self.log("test_PCC", test_pcc, prog_bar=True, logger=True,
                     batch_size=y_2d.shape[0], on_step=False, on_epoch=True)
        
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

    def on_test_end(self) -> None:
        import torch.distributed as dist
        import sys

        super().on_test_end()
        model_name = str(self.__class__).split(".")[-1].replace("'>", "")

        # --- Gather predictions from all ranks ---
        local_preds_expr = []
        local_preds_obs = []
        for expr, obs in self.preds_list:
            local_preds_expr.append(expr)
            local_preds_obs.append(obs)

        # Concatenate locally first
        local_expr = np.concatenate(local_preds_expr)
        local_obs = pd.concat(local_preds_obs)

        # Determine distributed environment
        is_distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_distributed else 1
        rank = dist.get_rank() if is_distributed else 0

        # Each rank prepares its own data to gather
        gathered_data = [None for _ in range(world_size)]

        if is_distributed:
            # Move data to CPU before gathering to avoid GPU memory allocation
            # This prevents CUDA out of memory during all_gather_object
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear any cached GPU memory

            # All ranks must call this - data will be transferred to CPU internally by all_gather_object
            dist.all_gather_object(gathered_data, (local_expr, local_obs))
        else:
            gathered_data[0] = (local_expr, local_obs)

        summary_metrics=None
        # ---- Only rank 0 performs aggregation ----
        if rank == 0:
            print(f"[Rank 0] Gathering data from {world_size} ranks ...")
            sys.stdout.flush()

            gathered_expr, gathered_obs = [], []
            for expr, obs in gathered_data:
                gathered_expr.append(expr)
                gathered_obs.append(obs)

            gathered_expr = np.concatenate(gathered_expr)
            gathered_obs = pd.concat(gathered_obs, ignore_index=True)

            # # Ensure that X (gathered_expr) and obs (gathered_obs) have matching row counts
            # # for AnnData construction. For some models (e.g., set-based models) we may
            # # produce multiple predictions per observation; in that case, aggregate them.
            # n_x, n_obs = gathered_expr.shape[0], len(gathered_obs)
            # if n_x != n_obs:
            #     if n_x % n_obs != 0:
            #         raise ValueError(
            #             f"Mismatch between prediction rows (X={n_x}) and obs rows ({n_obs}), "
            #             "and cannot infer an integer aggregation factor."
            #         )
            #     factor = n_x // n_obs
            #     gathered_expr = gathered_expr.reshape(n_obs, factor, -1).mean(axis=1)

            print(f"[Rank 0] Total gathered samples: {len(gathered_obs)}")
            sys.stdout.flush()

            # ---- Build AnnData objects ----
            gene_names=self.gene_names
            if hasattr(self,'infer_gene_ids'):
                print("[Rank 0] Use infer gene ids")
                sys.stdout.flush()
                gene_names=gene_names[self.infer_gene_ids]

            print(len(gene_names))
            sys.stdout.flush()

            control_adata = self.datamodule.test_dataset.control_adata[:,gene_names]
            pert_adata = self.datamodule.test_dataset.pert_adata[:,gene_names]

            predicted_adata = ad.AnnData(
                X=gathered_expr,
                obs=gathered_obs,
                var=pd.DataFrame(index=gene_names),
            )

            predicted_adata = ad.concat([predicted_adata, control_adata])
            predicted_adata.obs_names_make_unique()
            reference_adata = ad.concat([pert_adata, control_adata])

            print(f"[Rank 0] Built AnnData: predicted={predicted_adata.shape}, reference={reference_adata.shape}")
            sys.stdout.flush()

            # ---- Determine evaluation features (gene subset) ----
            eval_features = None
            single_celline_mask = reference_adata.X.sum(axis=0) != 0
            # Convert boolean mask to numpy array if needed
            if hasattr(single_celline_mask, 'values'):
                single_celline_mask = single_celline_mask.values
            elif hasattr(single_celline_mask, 'A1'):  # sparse matrix
                single_celline_mask = single_celline_mask.A1
            elif hasattr(single_celline_mask, 'toarray'):  # sparse matrix
                single_celline_mask = single_celline_mask.toarray().flatten()
            elif hasattr(single_celline_mask, 'A'):  # numpy matrix
                single_celline_mask = single_celline_mask.A.flatten()
            # Ensure it's a numpy array
            single_celline_mask = np.asarray(single_celline_mask, dtype=bool)

            # Convert gene_names to numpy array for boolean indexing
            gene_names_array = np.array(self.gene_names)
            eval_features = gene_names_array[single_celline_mask]

            # ---- Perform evaluation ----
            ev = Evaluation(
                model_adatas=[predicted_adata],
                model_names=[model_name],
                ref_adata=reference_adata,
                pert_col='_merged_pert_col_' if self.use_mix_pert else self.pert_key ,
                cov_cols=self.result_avg_keys,
                ctrl=self.control_val,
                features=eval_features,  # Pass gene subset if mask is enabled
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

            summary_metrics = pd.DataFrame(summary_metrics_dict).T.applymap(
                lambda x: float(np.format_float_positional(
                    x, precision=4, unique=False, fractional=False, trim="k"
                ))
            )

            if self.evaluation_config.print_summary:
                print(f"\n===== Summary Metrics =====\n{summary_metrics}\n")
                sys.stdout.flush()

            # Get output directory (contains timestamp from hydra)
            # Try to get from hydra runtime, fallback to logger save_dir or evaluation_config.save_dir
            try:
                output_dir = HydraConfig.get().runtime.output_dir
            except Exception:
                # Fallback: try to get from logger
                if self.logger is not None:
                    # Handle both single logger and list of loggers
                    logger_obj = self.logger[0] if isinstance(self.logger, (list, tuple)) and len(self.logger) > 0 else self.logger
                    output_dir = getattr(logger_obj, "save_dir", None) or self.evaluation_config.save_dir
                else:
                    output_dir = self.evaluation_config.save_dir
            
            # Create summary directory in output_dir (contains timestamp)
            summary_dir = os.path.join(output_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            
            # Also keep the original evaluation save_dir for backward compatibility
            os.makedirs(self.evaluation_config.save_dir, exist_ok=True)
            ev.save(self.evaluation_config.save_dir)

            # Save summary files to both locations
            csv_path = os.path.join(self.evaluation_config.save_dir, "summary.csv")
            summary_metrics.to_csv(csv_path, index_label="metric")
            
            # Also save to summary directory
            # Add checkpoint type suffix if available
            ckpt_type = getattr(self, "current_test_ckpt_type", None)
            if ckpt_type and ckpt_type != "unknown":
                summary_csv_path = os.path.join(summary_dir, f"summary_metrics_{ckpt_type}.csv")
            else:
                summary_csv_path = os.path.join(summary_dir, "summary_metrics.csv")
            summary_metrics.to_csv(summary_csv_path, index_label="metric")

            # Always save predictions to summary directory (regardless of wandb)
            # Add checkpoint type suffix to distinguish predictions from different checkpoints
            if ckpt_type and ckpt_type != "unknown":
                pred_h5ad_path = os.path.join(summary_dir, f"predictions_{ckpt_type}.h5ad")
                # ref_h5ad_path = os.path.join(summary_dir, f"reference_{ckpt_type}.h5ad")  # Disabled: no longer saving reference files
            else:
                pred_h5ad_path = os.path.join(summary_dir, "predictions.h5ad")
                # ref_h5ad_path = os.path.join(summary_dir, "reference.h5ad")  # Disabled: no longer saving reference files
            # Track whether files were saved successfully
            files_saved = False
            try:
                predicted_adata.write(pred_h5ad_path)
                # reference_adata.write(ref_h5ad_path)  # Disabled: no longer saving reference files to save disk space
                files_saved = True
                print(f"[Rank 0] Prediction files saved to: {summary_dir}")
                print(f"[Rank 0]   - predictions.h5ad: {pred_h5ad_path}")
                # print(f"[Rank 0]   - reference.h5ad: {ref_h5ad_path}")  # Disabled: no longer saving reference files
                sys.stdout.flush()
            except OSError as e:
                # Handle disk quota exceeded and other disk-related errors gracefully
                if e.errno == 122:  # Disk quota exceeded
                    print("[Rank 0] WARNING: Disk quota exceeded. Skipping prediction file save.")
                    print("[Rank 0]   Metrics are still saved to CSV and logged to wandb.")
                    print("[Rank 0]   File size would be ~{predicted_adata.X.nbytes / 1024**3:.2f} GB")
                    print("[Rank 0]   To save predictions, free up disk space or use a different output directory.")
                else:
                    print("[Rank 0] WARNING: Failed to save prediction files (OSError {e.errno}): {e}")
                    print("[Rank 0]   Metrics are still saved to CSV and logged to wandb.")
                sys.stdout.flush()
            except Exception as e:
                print(f"[Rank 0] WARNING: Failed to save prediction files: {e}")
                print("[Rank 0]   Metrics are still saved to CSV and logged to wandb.")
                sys.stdout.flush()

            print(f"[Rank 0] Evaluation finished. Results saved to {csv_path}")
            if files_saved:
                print(f"[Rank 0] Summary files also saved to {summary_dir}")
                print("[Rank 0] Test predictions saved to:")
                print(f"[Rank 0]   - Predictions: {pred_h5ad_path}")
                # print(f"[Rank 0]   - Reference: {ref_h5ad_path}")  # Disabled: no longer saving reference files
            else:
                print("[Rank 0] WARNING: Prediction files were not saved due to disk space issues.")
            print(f"[Rank 0]   - Summary metrics: {summary_csv_path}")
            sys.stdout.flush()

            # ---- Optional: Log test metrics and predictions to wandb (if enabled and available) ----
            save_preds_to_wandb = self.evaluation_config.get("save_predictions_to_wandb", False)  # Default to False
            if save_preds_to_wandb and self.logger is not None:
                try:
                    # Handle both single logger and list of loggers
                    loggers = self.logger if isinstance(self.logger, (list, tuple)) else [self.logger]
                    for logger in loggers:
                        # Check if it's WandbLogger
                        if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                            try:
                                # Convert summary_metrics_dict to dict format for wandb
                                test_metrics_dict = {}
                                for metric_name, value in summary_metrics_dict.items():
                                    test_metrics_dict[f"test_{metric_name}"] = float(value)
                                
                                # Log metrics to wandb (only metrics, no artifacts to avoid upload delays)
                                logger.experiment.log(test_metrics_dict)
                                print(f"[Rank 0] Test metrics logged to wandb: {list(test_metrics_dict.keys())}")
                                # Note: Artifact upload removed to prevent hanging. Files are still saved locally.
                                sys.stdout.flush()
                            except Exception as e:
                                print(f"[Rank 0] Warning: Failed to log to wandb (files are still saved locally): {e}")
                                sys.stdout.flush()
                except Exception as e:
                    print(f"[Rank 0] Warning: Wandb logging skipped (files are still saved locally): {e}")
            sys.stdout.flush()

            # Save result for external access
        #broadcast summary_metrics to all processes
        if is_distributed:
            obj_list=[summary_metrics]
            dist.broadcast_object_list(obj_list,src=0)
            self.summary_metrics=obj_list[0]
        else:
            self.summary_metrics = summary_metrics

        # ---- Synchronize and cleanup ----
        if is_distributed:
            dist.barrier()

        self.preds_list = []
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
