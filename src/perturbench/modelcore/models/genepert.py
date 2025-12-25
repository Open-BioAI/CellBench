import torch
import torch.nn.functional as F
import lightning as L
import logging
from .base import PerturbationModel
from ..nn import MixedPerturbationEncoder
from ..nn.genepert_networks import GenePertMLP

log = logging.getLogger(__name__)


class GenePert(PerturbationModel):

    def __init__(
            self,
            hidden_size: int = 128,
            use_cell_emb: bool = False,
            use_mask: bool = False,  # Unified mask switch for training loss and evaluation
            lr: float = 1e-3,
            wd: float = 1e-5,
            lr_scheduler_freq: int | None = None,
            lr_scheduler_interval: str | None = None,
            lr_scheduler_patience: int | None = None,
            lr_scheduler_factor: float | None = None,
            lr_scheduler_mode: str | None = None,
            lr_scheduler_max_lr: float | None = None,
            lr_scheduler_total_steps: int | None = None,
            datamodule: L.LightningDataModule | None = None,
            **kwargs
    ):
        super(GenePert, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_mode=lr_scheduler_mode,
            lr_scheduler_max_lr=lr_scheduler_max_lr,
            lr_scheduler_total_steps=lr_scheduler_total_steps,
            use_mask=use_mask,  # Pass use_mask to base class
        )


        self.hidden_size = hidden_size
        self.use_cell_emb = use_cell_emb

        # Perturbation encoder / embedding dimension setup
        self.pert_encoder = None
        if getattr(self, "use_mix_pert", False):
            self.pert_encoder = MixedPerturbationEncoder(
                gene_pert_dim=self.gene_pert_dim,
                drug_pert_dim=self.drug_pert_dim,
                env_pert_dim=self.env_pert_dim,
                final_embed_dim=self.hidden_size,
            )
            self.embedding_dim = self.hidden_size
        else:
            self.embedding_dim = self.datamodule.train_dataset.transform.embedding_dim

        # Initialize MLP model
        self.mlp = GenePertMLP(
            input_dim=self.embedding_dim,
            output_dim=self.n_genes,
            hidden_size=self.hidden_size
        )

        self._compute_ctrl_mean()

    def _compute_ctrl_mean(self):
        """Compute mean control expression from training data."""
        train_ctrl_mean=torch.tensor(self.datamodule.train_dataset.control_adata.X.mean(axis=0))
        val_ctrl_mean = torch.tensor(self.datamodule.val_dataset.control_adata.X.mean(axis=0))
        test_ctrl_mean = torch.tensor(self.datamodule.test_dataset.control_adata.X.mean(axis=0))

        self.register_buffer("train_ctrl_mean", train_ctrl_mean)
        self.register_buffer("val_ctrl_mean", val_ctrl_mean)
        self.register_buffer("test_ctrl_mean", test_ctrl_mean)

    def _encode_perturbation(self, batch) -> torch.Tensor:
        """Encode perturbation signals from the batch."""
        if self.pert_encoder is not None:
            return self.pert_encoder(batch)
        return batch["pert_emb"]

    def forward(self, perturbation_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            perturbation_embeddings: Tensor of shape [batch_size, embedding_dim]

        Returns:
            Predicted gene expression delta [batch_size, n_genes]
        """
        # MLP predicts the delta (change from control)
        delta = self.mlp(perturbation_embeddings)
        return delta

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Training batch
            batch_idx: Batch index

        Returns:
            Training loss
        """

        # Get observed expression and perturbation names
        observed_expression = batch['pert_cell_counts']

        # Get perturbation embeddings
        pert_embeddings = self._encode_perturbation(batch)

        # Forward pass: predict delta from control
        predicted_delta = self.forward(pert_embeddings)

        # Compute target delta (observed - control mean)
        target_delta = observed_expression - self.train_ctrl_mean.unsqueeze(0)

        # Use expression mask for loss calculation - only compute loss on expressed genes
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(predicted_delta.device)
            masked_loss = F.mse_loss(
                predicted_delta,
                target_delta,
                reduction="none",
            )
            # 这样才算给每个batch上有效gene算好mse_loss以后在batch上求平均
            valid = mask.sum(dim=1)  # 指定维度[batch]
            loss_per_batch = (masked_loss * mask).sum(dim=1)  # [batch]
            loss = (loss_per_batch / valid).nanmean()
        else:
            # Fallback to standard MSE
            loss = F.mse_loss(predicted_delta, target_delta)

        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        # Compute training PCC (use mask if enabled)
        # predictions = delta + ctrl_mean, observed = observed_expression
        predictions = predicted_delta + self.train_ctrl_mean.unsqueeze(0)
        train_pcc = self._compute_masked_pcc(predictions, observed_expression, mask)
        self.log("train_PCC", train_pcc, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, data_tuple, batch_idx):
        """
        Validation step.

        Args:
            batch: Validation batch
            batch_idx: Batch index

        Returns:
            Validation loss
        """
        batch,_=data_tuple
        # Get observed expression and perturbation names
        observed_expression = batch['pert_cell_counts']

        # Get perturbation embeddings
        pert_embeddings = self._encode_perturbation(batch)

        # Forward pass: predict delta from control
        predicted_delta = self.forward(pert_embeddings)

        # Compute target delta
        target_delta = observed_expression - self.val_ctrl_mean.unsqueeze(0)

        # Use expression mask for loss calculation - only compute loss on expressed genes
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(predicted_delta.device)
            masked_loss = F.mse_loss(
                predicted_delta,
                target_delta,
                reduction="none",
            )
            valid = mask.sum(dim=1)
            loss_per_batch = (masked_loss * mask).sum(dim=1)
            loss = (loss_per_batch / valid).nanmean()
        else:
            # Fallback to standard MSE
            loss = F.mse_loss(predicted_delta, target_delta)

        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        # Compute validation PCC (use mask if enabled)
        predictions = predicted_delta + self.val_ctrl_mean.unsqueeze(0)
        val_pcc = self._compute_masked_pcc(predictions, observed_expression, mask)
        self.log("val_PCC", val_pcc, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        return loss

    def predict(self,batch):
        # Get perturbation embeddings
        pert_embeddings = self._encode_perturbation(batch)

        # Forward pass: predict delta from control
        predicted_delta = self.forward(pert_embeddings)

        preds = predicted_delta + batch['control_cell_counts']
        return preds
