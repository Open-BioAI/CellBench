import torch
import torch.nn.functional as F
import lightning as L

from ..nn.mlp import MLP
from .base import PerturbationModel
from ..nn import MixedPerturbationEncoder


class BiolordStar(PerturbationModel):
    """
    A version of Biolord
    """

    def __init__(
        self,
        n_layers: int = 2,
        encoder_width: int = 128,
        latent_dim: int = 32,
        penalty_weight: float = 10000.0,
        noise: float = 0.1,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
        lr_scheduler_mode: str | None = None,
        lr_scheduler_max_lr: float | None = None,
        lr_scheduler_total_steps: int | None = None,
        dropout: float | None = None,
        softplus_output: bool = True,
        n_total_covariates: int | None = None,
        use_mask: bool = False,  # Unified mask switch for training loss and evaluation
        datamodule: L.LightningDataModule | None = None,
            **kwargs,
    ):
        """
        The constructor for the BiolordStar class.

        Args:
            n_genes: Number of genes to use for prediction
            n_perts: Number of perturbations in the dataset
                (not including controls)
            n_layers: Number of layers in the encoder/decoder
            encoder_width: Width of the hidden layers in the encoder/decoder
            latent_dim: Dimension of the latent space
            lr: Learning rate
            wd: Weight decay
            lr_scheduler_freq: How often the learning rate scheduler checks
                val_loss
            lr_scheduler_interval: Whether the learning rate scheduler checks
                every epoch or step
            lr_scheduler_patience: Learning rate scheduler patience
            lr_scheduler_factor: Factor by which to reduce learning rate when
                learning rate scheduler triggers
            lr_scheduler_mode: Learning rate scheduler mode ("plateau", "onecycle", "step")
            lr_scheduler_max_lr: Maximum learning rate for OneCycleLR
            lr_scheduler_total_steps: Total training steps for OneCycleLR
            dropout: Dropout rate or None for no dropout.
            softplus_output: Whether to apply a softplus activation to the
                output of the decoder to enforce non-negativity
            datamodule: The datamodule used to train the model
        """
        super(BiolordStar, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_mode=lr_scheduler_mode,
            lr_scheduler_max_lr=lr_scheduler_max_lr,
            lr_scheduler_total_steps=lr_scheduler_total_steps,
            use_mask=use_mask,  # Pass use_mask to base class
        )
        self.save_hyperparameters(ignore=["datamodule"])

        n_total_covariates = datamodule.train_dataset.transform.n_total_covs
        self.n_perts = datamodule.train_dataset.transform.n_perts

        decoder_input_dim = 3 * latent_dim
        self.lord_embedding = torch.nn.Parameter(
            torch.randn(latent_dim, n_total_covariates)
        )
        self.gene_encoder = MLP(
            self.n_genes, encoder_width, latent_dim, n_layers, dropout
        )
        self.decoder = MLP(
            decoder_input_dim, encoder_width, self.n_genes, n_layers, dropout
        )
        self.pert_encoder = MixedPerturbationEncoder(gene_pert_dim=self.gene_pert_dim,
                                                     drug_pert_dim=self.drug_pert_dim,
                                                     env_pert_dim=self.env_pert_dim,
                                                     hidden_dims=[latent_dim]*(n_layers-1) if n_layers>1 else [],
                                                     final_embed_dim=latent_dim)

        self.penalty_weight = penalty_weight
        self.noise = noise
        self.dropout = dropout
        self.softplus_output = softplus_output

    def forward(
        self,
        input_expression: torch.Tensor,
        batch,
        add_noise: bool = True
    ):
        """
        Forward pass: predict perturbed expression from input expression + perturbation.
        
        Args:
            input_expression: Input expression (typically control expression during training/prediction,
                             or perturbed expression as fallback)
            batch: Batch containing perturbation and covariate information
            add_noise: Whether to add noise to latent representation (only for training, not validation/test)
        """
        covariates = {cov_key: batch[cov_key] for cov_key in self.cov_keys}
        latent_input_expression = self.gene_encoder(
            input_expression
        )
        # Only add noise during training, not during validation/test
        if add_noise:
            latent_input_expression += self.noise * torch.randn_like(
                latent_input_expression
            )
        latent_perturbation = self.pert_encoder(batch)
        # # 修复：检查协变量是否存在且有效
        # if "cell_cluster" in covariates and len(covariates["cell_cluster"]) > 0:
        #     latent_covariates = torch.vstack(
        #         [self.lord_embedding[:, cov.bool()].T for cov in covariates["cell_cluster"]]
        #     )
        # 处理协变量：将所有 covariate 的 one-hot 向量拼接，然后通过 lord_embedding 转换为嵌入
        if covariates and any(cov is not None and len(cov) > 0 for cov in covariates.values()):
            # 拼接所有 covariate 的 one-hot 向量
            # covariates 是一个字典，每个值是一个 tensor，形状为 [batch_size, onehot_dim]
            cov_tensors = [cov for cov in covariates.values() if cov is not None and len(cov) > 0]
            if cov_tensors:
                batch_size = cov_tensors[0].shape[0]
                # 按顺序拼接所有 covariate 的 one-hot 向量，形状变为 [batch_size, total_onehot_dim]
                merged_cov = torch.cat(cov_tensors, dim=1)  # 在特征维度上拼接
                # 通过 lord_embedding 转换为嵌入向量
                # merged_cov 是 one-hot 向量，需要找到每个样本中为 1 的位置
                # 使用矩阵乘法：lord_embedding @ merged_cov.T 然后转置
                latent_covariates = (self.lord_embedding @ merged_cov.T).T  # [batch_size, latent_dim]
            else:
                batch_size = latent_input_expression.shape[0]
                latent_covariates = torch.zeros(
                    batch_size, 
                    self.lord_embedding.shape[0],
                    device=input_expression.device
                )
        else:
            # 当没有协变量时，使用全零向量，并确保在正确的设备上
            batch_size = latent_input_expression.shape[0]
            latent_covariates = torch.zeros(
                batch_size, 
                self.lord_embedding.shape[0],
                device=input_expression.device  # 确保在相同设备上
            )
    
        latent_perturbed_expression = torch.cat(
            [
                latent_input_expression,
                latent_perturbation,
                latent_covariates,
            ],
            dim=-1,
        )

        predicted_perturbed_expression = self.decoder(latent_perturbed_expression)

        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression, (latent_covariates**2).sum()

    def training_step(self, batch, batch_idx: int):
        observed_perturbed_expression = batch.pert_cell_counts
        
        # Get control expression for training (consistent with prediction)
        # Must have control expression to avoid training contamination
        if hasattr(batch, "control_cell_counts"):
            control_expression = batch.control_cell_counts
        elif hasattr(batch, "control_cell_emb"):
            control_expression = batch.control_cell_emb
        elif hasattr(batch, "controls"):
            control_expression = batch.controls
        else:
            raise AttributeError(
                f"Batch does not have control expression for training. "
                f"Biolord requires 'control_cell_counts', 'control_cell_emb', or 'controls' attribute. "
                f"Available attributes: {list(batch.keys()) if hasattr(batch, 'keys') else dir(batch)}"
            )

        # Training: add noise for regularization
        predicted_perturbed_expression, penalty = self.forward(
            control_expression, batch, add_noise=True
        )
        # Use expression mask for loss calculation - only compute loss on expressed genes
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(predicted_perturbed_expression.device)
            masked_loss = F.mse_loss(
                predicted_perturbed_expression,
                observed_perturbed_expression,
                reduction="none",
            )
            valid = mask.sum()
            if valid > 0:
                recon_loss = (masked_loss * mask).sum() / valid
            else:
                recon_loss = masked_loss.mean()
        else:
            # Fallback to standard MSE
            recon_loss = F.mse_loss(
                predicted_perturbed_expression,
                observed_perturbed_expression,
                reduction="mean",
            )

        # Total loss includes penalty (this is what we optimize)
        penalty_term = self.penalty_weight * penalty
        total_loss = recon_loss + penalty_term

        # Compute training PCC (use mask if enabled)
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(predicted_perturbed_expression.device)
        train_pcc = self._compute_masked_pcc(predicted_perturbed_expression, observed_perturbed_expression, mask)

        # Log both reconstruction loss and total loss (with penalty)
        self.log("train_recon_loss", recon_loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)
        self.log("train_penalty", penalty, prog_bar=False, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)
        self.log("train_penalty_term", penalty_term, prog_bar=False, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)
        self.log("train_loss", total_loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)
        self.log("train_PCC", train_pcc, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, data_tuple, batch_idx: int):
        batch,_=data_tuple
        observed_perturbed_expression = batch.pert_cell_counts
        
        # Get control expression for validation (consistent with prediction)
        # Must have control expression to avoid validation contamination
        if hasattr(batch, "control_cell_counts"):
            control_expression = batch.control_cell_counts
        elif hasattr(batch, "control_cell_emb"):
            control_expression = batch.control_cell_emb
        elif hasattr(batch, "controls"):
            control_expression = batch.controls
        else:
            raise AttributeError(
                f"Batch does not have control expression for validation. "
                f"Biolord requires 'control_cell_counts', 'control_cell_emb', or 'controls' attribute. "
                f"Available attributes: {list(batch.keys()) if hasattr(batch, 'keys') else dir(batch)}"
            )

        # Validation: no noise (deterministic evaluation)
        predicted_perturbed_expression, penalty = self.forward(
            control_expression, batch, add_noise=False
        )
        # Use expression mask for loss calculation - only compute loss on expressed genes
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(predicted_perturbed_expression.device)
            masked_loss = F.mse_loss(
                predicted_perturbed_expression,
                observed_perturbed_expression,
                reduction="none",
            )
            valid = mask.sum()
            if valid > 0:
                val_recon_loss = (masked_loss * mask).sum() / valid
            else:
                val_recon_loss = masked_loss.mean()
        else:
            # Fallback to standard MSE
            val_recon_loss = F.mse_loss(
                predicted_perturbed_expression,
                observed_perturbed_expression,
                reduction="mean",
            )
        
        # Total loss includes penalty
        penalty_term = self.penalty_weight * penalty
        val_loss = val_recon_loss + penalty_term

        # Compute validation PCC (use mask if enabled)
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(predicted_perturbed_expression.device)
        val_pcc = self._compute_masked_pcc(predicted_perturbed_expression, observed_perturbed_expression, mask)

        # Log both reconstruction loss and total loss (with penalty)
        self.log("val_recon_loss", val_recon_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_penalty", penalty, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_penalty_term", penalty_term, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch))
        self.log("val_PCC", val_pcc, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch))
        return val_loss

    def predict(self, batch):
        # Get control expression - handle different batch formats
        if hasattr(batch, "controls"):
            control_expression = batch.controls
        elif hasattr(batch, "control_cell_counts"):
            control_expression = batch.control_cell_counts
        elif hasattr(batch, "control_cell_emb"):
            control_expression = batch.control_cell_emb
        else:
            raise AttributeError(
                f"Batch does not have 'controls', 'control_cell_counts', or 'control_cell_emb' attribute. "
                f"Available attributes: {dir(batch)}"
            )
        
        control_expression = control_expression.to(self.device)

        # Prediction: no noise (deterministic)
        predicted_perturbed_expression, _ = self.forward(
            control_expression,
            batch,
            add_noise=False
        )
        return predicted_perturbed_expression


