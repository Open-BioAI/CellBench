from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import lightning as L

from .base import PerturbationModel
from perturbench.data.types import Batch
from ..nn import MixedPerturbationEncoder

from perturbench.modelcore.nn.squidiff.script_util import (
    create_model_and_diffusion,
)

from perturbench.modelcore.nn.squidiff.resample import create_named_schedule_sampler

class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=2048, n_layers=2, dropout=0.1):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ConditionEncoder(nn.Module):
    def __init__(self, base_width: int, n_perts: int, covar_uniques: dict[str, list] | None,
                 latent_dim: int = 60, hidden: int = 2048, dropout: float = 0.1,
                 use_mix_pert: bool = True, mix_pert_encoder: nn.Module | None = None):
        super().__init__()
        self.enc_basal = _MLP(base_width, latent_dim, hidden, 2, dropout)
        self.use_mix_pert = use_mix_pert
        self.mix_pert_encoder = mix_pert_encoder if use_mix_pert else None
        if not self.use_mix_pert:
            self.enc_pert  = _MLP(n_perts,   latent_dim, hidden, 1, dropout)
        elif self.mix_pert_encoder is None:
            raise ValueError("mix_pert_encoder must be provided when use_mix_pert is True")
        self.enc_cov   = nn.ModuleDict()
        if covar_uniques is not None:
            for k, vals in covar_uniques.items():
                if len(vals) > 1:
                    self.enc_cov[k] = _MLP(len(vals), latent_dim, hidden, 1, dropout)

    def forward(self, base: torch.Tensor, pert: Optional[torch.Tensor],
                covs: Optional[dict[str, torch.Tensor]], batch=None) -> torch.Tensor:
        z = self.enc_basal(base)
        if self.mix_pert_encoder is not None:
            z = z + self.mix_pert_encoder(batch)
        else:
            z = z + self.enc_pert(pert)
        if covs is not None:
            for k, enc in self.enc_cov.items():
                z = z + enc(covs[k])
        return z

############################################
#输入batch:
#gene_expression
#pert_cell_counts
#pert_cell_counts
#b

class Squidiff(PerturbationModel):
    def __init__(
        self,
        z_latent_dim: int = 60,
        cond_hidden: int = 2048,
        cond_dropout: float = 0.1,
        use_covariates: bool = True,
        lr: float = 1e-4,
        wd: float = 1e-8,
        lr_scheduler_factor= 0.5,  # LR decay factor
        lr_scheduler_patience= 10 , # Wait N epochs before reducing LR
        lr_scheduler_freq=1,  # Check the monitor every N epochs
        lr_scheduler_interval= "epoch",  # Run scheduler once per epoch
        lr_scheduler_mode: str | None = None,
        lr_scheduler_max_lr: float | None = None,
        lr_scheduler_total_steps: int | None = None,
        cov_keys: list[str] = None,
        pert_key: str=None,
        diffusion_steps: int = 1000,
        noise_schedule: str = "linear",
        timestep_respacing: str = "",
        learn_sigma: bool = False,
        predict_xstart: bool = False,
        rescale_timesteps: bool = False,
        rescale_learned_sigmas: bool = False,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = True,
        use_fp16: bool = False,
        use_encoder: bool = True,
        use_drug_structure: bool = False,
        drug_dimension: int = 1024,
        comb_num: int = 1,

        schedule_sampler: str = "uniform",
        use_pretrained_cell_emb:bool= False,
        use_cell_emb: bool | None = None,
        use_mask: bool = False,  # 控制是否使用mask计算loss，默认不启用
        datamodule: Optional[L.LightningDataModule] = None,
    ):
        super().__init__(datamodule=datamodule, lr=lr, wd=wd,
                         lr_scheduler_freq=lr_scheduler_freq,
                         lr_scheduler_interval=lr_scheduler_interval,
                         lr_scheduler_patience=lr_scheduler_patience,
                         lr_scheduler_factor=lr_scheduler_factor,
                         lr_scheduler_mode=lr_scheduler_mode,
                         lr_scheduler_max_lr=lr_scheduler_max_lr,
                         lr_scheduler_total_steps=lr_scheduler_total_steps,
                         use_mask=use_mask)

        self.cov_keys = cov_keys
        self.pert_key=pert_key
        self.use_covariates = use_covariates
        self.n_perts=datamodule.train_dataset.transform.n_perts
        self.use_pretrained_cell_emb = use_pretrained_cell_emb if use_cell_emb is None else use_cell_emb
        self.use_cell_emb = self.use_pretrained_cell_emb

        if use_covariates:
            train_pert_adata=self.datamodule.train_dataset.pert_adata
            covar_uniques = {cov_key:train_pert_adata.obs[cov_key].unique() for cov_key in cov_keys}
        else:
            covar_uniques = None

        assert (self.use_pretrained_cell_emb and self.embedding_dim) or self.use_pretrained_cell_emb is False
        if self.use_pretrained_cell_emb:
            base_width=self.embedding_dim
        else:
            base_width=self.n_genes

        self.mix_pert_encoder = None
        if getattr(self, "use_mix_pert", False):
            self.mix_pert_encoder = MixedPerturbationEncoder(
                gene_pert_dim=self.gene_pert_dim,
                drug_pert_dim=self.drug_pert_dim,
                env_pert_dim=self.env_pert_dim,
                final_embed_dim=z_latent_dim,
                dropout=cond_dropout,
            )

        self.cond_encoder = ConditionEncoder(
            base_width=base_width,
            n_perts=self.n_perts,
            covar_uniques=covar_uniques,
            latent_dim=z_latent_dim,
            hidden=cond_hidden,
            dropout=cond_dropout,
            use_mix_pert=self.use_mix_pert,
            mix_pert_encoder=self.mix_pert_encoder,
        )

        self.model, self.diffusion = create_model_and_diffusion(
            gene_size=self.n_genes,
            num_layers=3,
            z_latent_dim=z_latent_dim,
            output_dim=self.n_genes,
            class_cond=False,
            learn_sigma=learn_sigma,
            num_channels=128,
            dropout=dropout,
            diffusion_steps=diffusion_steps,
            noise_schedule=noise_schedule,
            timestep_respacing=timestep_respacing,
            use_kl=False,
            predict_xstart=predict_xstart,
            rescale_timesteps=rescale_timesteps,
            rescale_learned_sigmas=rescale_learned_sigmas,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
            use_fp16=use_fp16,
            use_encoder=use_encoder,
            use_drug_structure=use_drug_structure,
            drug_dimension=drug_dimension,
            comb_num=comb_num,
        )
        self.schedule_sampler = create_named_schedule_sampler(schedule_sampler, self.diffusion)
        self.save_hyperparameters(ignore=["datamodule"])

    def _get_control_expression(self, batch: Batch) -> torch.Tensor:
        if hasattr(batch, "controls"):
            return batch.controls
        if hasattr(batch, "control_cell_counts"):
            return batch.control_cell_counts
        return batch.control_cell_emb

    def _build_conditions(self, batch) -> Dict[str, torch.Tensor]:
        base = self._get_control_expression(batch)
        pert = None if self.use_mix_pert else batch[self.pert_key].squeeze()

        covs = None
        if self.use_covariates and self.cov_keys:
            covs = {cov_key: batch[cov_key] for cov_key in self.cov_keys}

        z_sem = self.cond_encoder(
            base,
            pert,
            covs,
            batch=batch if self.use_mix_pert else None,
        )
        return {"z_mod": z_sem}

    def _loss_on_batch(self, batch) -> torch.Tensor:
        x0 = batch.pert_cell_counts.squeeze()
        B = x0.shape[0]
        t, weights = self.schedule_sampler.sample(B, x0.device)
        cond = self._build_conditions(batch)

        # Get mask using unified method from base class
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(x0.device)

        losses = self.diffusion.training_losses(self.model, x0, t, model_kwargs=cond, mask=mask)
        return (losses["loss"] * weights).mean()

    def training_step(self, batch, batch_idx: int):
        loss = self._loss_on_batch(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch))

        # Compute training PCC (use mask if enabled)
        # Note: For diffusion models, we compute PCC on actual sample vs observed during training
        # This is expensive, so we only do it occasionally
        if batch_idx % 100 == 0:  # Log PCC every 100 batches to reduce overhead
            with torch.no_grad():
                predictions = self.predict(batch)
                observed = batch.pert_cell_counts.squeeze()
                mask = self._get_mask(batch)
                if mask is not None:
                    mask = mask.to(predictions.device)
                train_pcc = self._compute_masked_pcc(predictions, observed, mask)
                self.log("train_PCC", train_pcc, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, data_tuple, batch_idx: int):
        batch,_=data_tuple
        loss = self._loss_on_batch(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        # Compute validation PCC (use mask if enabled)
        with torch.no_grad():
            predictions = self.predict(batch)
            observed = batch.pert_cell_counts.squeeze()
            mask = self._get_mask(batch)
            if mask is not None:
                mask = mask.to(predictions.device)
            val_pcc = self._compute_masked_pcc(predictions, observed, mask)
            self.log("val_PCC", val_pcc, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        return loss

    @torch.no_grad()
    def predict(self, batch) -> torch.Tensor:
        self.model.eval()
        B = len(batch)
        cond = self._build_conditions(batch)

        sample_fn = (
            self.diffusion.ddim_sample_loop
            if isinstance(self.hparams.timestep_respacing, str) and "ddim" in self.hparams.timestep_respacing.lower()
            else self.diffusion.p_sample_loop
        )
        out = sample_fn(self.model, shape=(B, self.n_genes), model_kwargs=cond, noise=None)
        return out


    @torch.no_grad()
    def sample_with_z(self, z_sem: torch.Tensor, n: int | None = None) -> torch.Tensor:
        n = n or z_sem.shape[0]
        return self.diffusion.ddim_sample_loop(self.model, shape=(n, self.n_genes), model_kwargs={"z_mod": z_sem})

    def configure_optimizers(self):
        """
        Optimizer + scheduler settings for Squidiff.
        Supports multiple scheduler modes: onecycle, step, or plateau (default).
        """

        # ---- Optimizer ----
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
        )

        # ---- LR Scheduler ----
        if self.lr_scheduler_mode == "onecycle":
            # OneCycleLR scheduler
            total_steps = self.lr_scheduler_total_steps
            if total_steps is None:
                try:
                    steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
                    total_steps = steps_per_epoch * self.trainer.max_epochs
                except Exception:
                    total_steps = 100 * 100  # fallback
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr_scheduler_max_lr or self.lr,
                total_steps=total_steps,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
            }
        elif self.lr_scheduler_mode == "step":
            # StepLR scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=getattr(self, 'lr_scheduler_step_size', None) or 10,
                gamma=getattr(self, 'lr_scheduler_gamma', None) or 0.1,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        else:
            # Default: ReduceLROnPlateau with Squidiff-specific tuned parameters
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_scheduler_factor,  # e.g. 0.5
                patience=self.lr_scheduler_patience,  # e.g. 15
                threshold=1e-3,
                threshold_mode="rel",
                cooldown=20,
                min_lr=1e-6,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": self.lr_monitor_key,  # usually "val_loss"
                "interval": "epoch",
                "frequency": 1,
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
