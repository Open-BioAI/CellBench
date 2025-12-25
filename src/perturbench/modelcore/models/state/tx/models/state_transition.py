import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from geomloss import SamplesLoss
from typing import Tuple

from ....base import PerturbationModel
from .....nn import MixedPerturbationEncoder
from .decoders_nb import NBDecoder, nb_nll
from .utils import build_mlp, get_activation_class, get_transformer_backbone, apply_lora
from .base import LatentToGeneDecoder


logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """
    Combined Sinkhorn + Energy loss
    """

    def __init__(self, sinkhorn_weight=0.001, energy_weight=1.0, blur=0.05):
        super().__init__()
        self.sinkhorn_weight = sinkhorn_weight
        self.energy_weight = energy_weight
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)
        self.energy_loss = SamplesLoss(loss="energy", blur=blur)

    def forward(self, pred, target):
        sinkhorn_val = self.sinkhorn_loss(pred, target)
        energy_val = self.energy_loss(pred, target)
        return self.sinkhorn_weight * sinkhorn_val + self.energy_weight * energy_val


class ConfidenceToken(nn.Module):
    """
    Learnable confidence token that gets appended to the input sequence
    and learns to predict the expected loss value.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Learnable confidence token embedding
        self.confidence_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Projection head to map confidence token output to scalar loss prediction
        self.confidence_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU(),  # Ensure positive loss prediction
        )

    def append_confidence_token(self, seq_input: torch.Tensor) -> torch.Tensor:
        """
        Append confidence token to the sequence input.

        Args:
            seq_input: Input tensor of shape [B, S, E]

        Returns:
            Extended tensor of shape [B, S+1, E]
        """
        batch_size = seq_input.size(0)
        # Expand confidence token to batch size
        confidence_tokens = self.confidence_token.expand(batch_size, -1, -1)
        # Concatenate along sequence dimension
        return torch.cat([seq_input, confidence_tokens], dim=1)

    def extract_confidence_prediction(self, transformer_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract main output and confidence prediction from transformer output.

        Args:
            transformer_output: Output tensor of shape [B, S+1, E]

        Returns:
            main_output: Tensor of shape [B, S, E]
            confidence_pred: Tensor of shape [B, 1]
        """
        # Split the output
        main_output = transformer_output[:, :-1, :]  # [B, S, E]
        confidence_output = transformer_output[:, -1:, :]  # [B, 1, E]

        # Project confidence token output to scalar
        confidence_pred = self.confidence_projection(confidence_output).squeeze(-1)  # [B, 1]

        return main_output, confidence_pred



class StateTransitionPerturbationModel(PerturbationModel):
    """
    This model:
      1) Projects basal expression and perturbation encodings into a shared latent space.
      2) Uses an OT-based distributional loss (energy, sinkhorn, etc.) from geomloss.
      3) Enables cells to attend to one another, learning a set-to-set function rather than
      a sample-to-sample single-cell map.
    """

    def __init__(
        self,
        datamodule,
        hidden_dim: int,
        batch_dim: int = None,
        basal_mapping_strategy: str = "random",
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "GPT2",
        transformer_backbone_kwargs: dict = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            input_dim: dimension of the input expression (e.g. number of genes or embedding dimension).
            hidden_dim: not necessarily used, but required by PerturbationModel signature.
            output_dim: dimension of the output space (genes or latent).
            pert_dim: dimension of perturbation embedding.
            gpt: e.g. "TranslationTransformerSamplesModel".
            model_kwargs: dictionary passed to that model's constructor.
            loss: choice of distributional metric ("sinkhorn", "energy", etc.).
            **kwargs: anything else to pass up to PerturbationModel or not used.
        """
        # Call the parent PerturbationModel constructor
        super().__init__(datamodule,lr =kwargs.get('lr', 1e-4))

        # Save or store relevant hyperparams
        self.input_dim = self.embedding_dim if self.embedding_dim else self.n_genes
        self.output_dim = self.embedding_dim if self.embedding_dim else self.n_genes
        self.use_cell_emb = kwargs.get("use_cell_emb", bool(self.embedding_dim))
        if getattr(self, "use_mix_pert", False):
            # MixPertTransform 没有单一的 pert_dim，后续直接用 gene/drug/env encoder
            self.pert_dim = self.gene_pert_dim + self.drug_pert_dim + self.env_pert_dim
        else:
            self.pert_dim = datamodule.train_dataset.transform.pert_dim
        self.gene_dim = self.n_genes

        decoder_dropout=kwargs.get("decoder_dropout", 0.1)
        residual_decoder=kwargs.get("residual_decoder", False)
        decoder_hidden_dims = kwargs.get("decoder_hidden_dims", [1024, 1024, 512])

        decoder_config=dict(
            latent_dim=self.output_dim,
            gene_dim=self.gene_dim,
            decoder_dropout=decoder_dropout,
            residual_decoder=residual_decoder,
            hidden_dims=decoder_hidden_dims,
        )

        self.hidden_dim=hidden_dim
        self.predict_residual = predict_residual
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.cell_sentence_len = kwargs.get("cell_set_len", 1)
        self.decoder_loss_weight = kwargs.get("decoder_weight", 1.0)
        self.regularization = kwargs.get("regularization", 0.0)
        self.detach_decoder = kwargs.get("detach_decoder", False)
        self.dropout=dropout

        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.transformer_backbone_kwargs["n_positions"] = self.cell_sentence_len + kwargs.get("extra_tokens", 0)

        self.distributional_loss = distributional_loss

        self.gene_decoder_bool = kwargs.get("gene_decoder_bool", True)
        if self.gene_decoder_bool:
            self.gene_decoder=LatentToGeneDecoder(**decoder_config)
        else:
            self.gene_decoder=None

        # Build the distributional loss from geomloss
        blur = kwargs.get("blur", 0.05)
        loss_name = kwargs.get("loss", "energy")
        if loss_name == "energy":
            self.loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
        elif loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "se":
            sinkhorn_weight = kwargs.get("sinkhorn_weight", 0.01)  # 1/100 = 0.01
            energy_weight = kwargs.get("energy_weight", 1.0)
            self.loss_fn = CombinedLoss(sinkhorn_weight=sinkhorn_weight, energy_weight=energy_weight, blur=blur)
        elif loss_name == "sinkhorn":
            self.loss_fn = SamplesLoss(loss="sinkhorn", blur=blur)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        self.use_basal_projection = kwargs.get("use_basal_projection", True)

        # Build the underlying neural OT network
        self._build_networks(lora_cfg=kwargs.get("lora", None))

        # Add an optional encoder that introduces a batch variable
        self.batch_encoder = None
        self.batch_dim = None
        self.predict_mean = kwargs.get("predict_mean", False)
        if kwargs.get("batch_encoder", False) and batch_dim is not None:
            self.batch_encoder = nn.Embedding(
                num_embeddings=batch_dim,
                embedding_dim=hidden_dim,
            )
            self.batch_dim = batch_dim

        # if the model is outputting to counts space, apply relu
        # otherwise its in embedding space and we don't want to
        if self.gene_decoder is None:
            self.relu = torch.nn.ReLU()

        self.use_batch_token = kwargs.get("use_batch_token", False)
        self.basal_mapping_strategy = basal_mapping_strategy
        # Disable batch token only for truly incompatible cases
        disable_reasons = []
        if self.batch_encoder and self.use_batch_token:
            disable_reasons.append("batch encoder is used")
        if basal_mapping_strategy == "random" and self.use_batch_token:
            disable_reasons.append("basal mapping strategy is random")

        if disable_reasons:
            self.use_batch_token = False
            logger.warning(
                f"Batch token is not supported when {' or '.join(disable_reasons)}, setting use_batch_token to False"
            )
            try:
                self.hparams["use_batch_token"] = False
            except Exception:
                pass

        self.batch_token_weight = kwargs.get("batch_token_weight", 0.1)
        self.batch_token_num_classes: Optional[int] = batch_dim if self.use_batch_token else None

        if self.use_batch_token:
            if self.batch_token_num_classes is None:
                raise ValueError("batch_token_num_classes must be set when use_batch_token is True")
            self.batch_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
            self.batch_classifier = build_mlp(
                in_dim=self.hidden_dim,
                out_dim=self.batch_token_num_classes,
                hidden_dim=self.hidden_dim,
                n_layers=1,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.batch_token = None
            self.batch_classifier = None

        # Internal cache for last token features (B, S, H) from transformer for aux loss
        self._batch_token_cache: Optional[torch.Tensor] = None

        # initialize a confidence token
        self.confidence_token = None
        self.confidence_loss_fn = None
        if kwargs.get("confidence_token", False):
            self.confidence_token = ConfidenceToken(hidden_dim=self.hidden_dim, dropout=self.dropout)
            self.confidence_loss_fn = nn.MSELoss()

        # Backward-compat: accept legacy key `freeze_pert`
        self.freeze_pert_backbone = kwargs.get("freeze_pert_backbone", kwargs.get("freeze_pert", False))
        if self.freeze_pert_backbone:
            # Freeze backbone base weights but keep LoRA adapter weights (if present) trainable
            for name, param in self.transformer_backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # Freeze projection head as before
            for param in self.project_out.parameters():
                param.requires_grad = False

        if kwargs.get("nb_decoder", False):
            self.gene_decoder = NBDecoder(
                latent_dim=self.output_dim + (self.batch_dim or 0),
                gene_dim=self.gene_dim,
                hidden_dims=[512, 512, 512],
                dropout=self.dropout,
            )

        print(self)

    def _build_networks(self, lora_cfg=None):
        """
        Here we instantiate the actual GPT2-based model.
        """
        if getattr(self, "use_mix_pert", False):
            self.pert_encoder = MixedPerturbationEncoder(
                gene_pert_dim=self.gene_pert_dim,
                drug_pert_dim=self.drug_pert_dim,
                env_pert_dim=self.env_pert_dim,
                hidden_dims=[self.hidden_dim] * (self.n_encoder_layers - 1) if self.n_encoder_layers > 1 else [],
                per_modality_embed_dim=self.hidden_dim,
                final_embed_dim=self.hidden_dim,
                dropout=self.dropout,
            )
        else:
            self.pert_encoder = build_mlp(
                in_dim=self.pert_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )

        # Simple linear layer that maintains the input dimension
        if self.use_basal_projection:
            self.basal_encoder = build_mlp(
                in_dim=self.input_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.basal_encoder = nn.Linear(self.input_dim, self.hidden_dim)

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
        )

        # Optionally wrap backbone with LoRA adapters
        if lora_cfg and lora_cfg.get("enable", False):
            self.transformer_backbone = apply_lora(
                self.transformer_backbone,
                self.transformer_backbone_key,
                lora_cfg,
            )

        # Project from input_dim to hidden_dim for transformer input
        # self.project_to_hidden = nn.Linear(self.input_dim, self.hidden_dim)

        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        if self.gene_decoder is None:
            self.final_down_then_up = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 8),
                nn.GELU(),
                nn.Linear(self.output_dim // 8, self.output_dim),
            )

    def _wrap_mix_batch(self, batch: dict | SimpleNamespace) -> SimpleNamespace:
        """
        Ensure MixedPerturbationEncoder receives attribute-style access.
        """
        if hasattr(batch, "gene_pert"):
            return batch  # already attribute-accessible
        return SimpleNamespace(
            gene_pert=batch["gene_pert"],
            drug_pert=batch["drug_pert"],
            env_pert=batch["env_pert"],
        )

    def encode_perturbation(self, pert: torch.Tensor, batch: dict | None = None) -> torch.Tensor:
        """If needed, define how we embed the raw perturbation input."""
        if getattr(self, "use_mix_pert", False):
            if batch is None:
                raise ValueError("batch is required for mixed perturbation encoding")
            mix_batch = self._wrap_mix_batch(batch)
            return self.pert_encoder(mix_batch)
        return self.pert_encoder(pert)

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we embed basal state input, if needed."""
        return self.basal_encoder(expr)

    def _get_control_tensor(self, batch: dict, padded: bool) -> torch.Tensor:
        """
        Fetch control input (embedding or counts) with backward-compatible key handling.
        """
        if "ctrl_cell_emb" in batch:
            ctrl = batch["ctrl_cell_emb"]
        elif "control_cell_emb" in batch:
            ctrl = batch["control_cell_emb"]
        elif "ctrl_cell_counts" in batch:
            ctrl = batch["ctrl_cell_counts"]
        elif "control_cell_counts" in batch:
            ctrl = batch["control_cell_counts"]
        else:
            raise KeyError("No control cell tensor found (ctrl_cell_emb/control_cell_emb/ctrl_cell_counts/control_cell_counts)")

        if padded:
            return ctrl.reshape(-1, self.cell_sentence_len, self.input_dim)
        return ctrl.reshape(1, -1, self.input_dim)

    def _get_target_tensor(self, batch: dict, padded: bool) -> torch.Tensor:
        """
        Fetch target tensor in embedding or gene space.
        """
        if "pert_cell_emb" in batch:
            tgt = batch["pert_cell_emb"]
        elif "pert_cell_counts" in batch:
            tgt = batch["pert_cell_counts"]
        else:
            raise KeyError("No perturbation target tensor found (pert_cell_emb/pert_cell_counts)")

        if padded:
            return tgt.reshape(-1, self.cell_sentence_len, self.output_dim)
        return tgt.reshape(1, -1, self.output_dim)

    def forward(self, batch: dict, padded=True) -> torch.Tensor:
        """
        The main forward call. Batch is a flattened sequence of cell sentences,
        which we reshape into sequences of length cell_sentence_len.

        Expects input tensors of shape (B, S, N) where:
        B = batch size
        S = sequence length (cell_sentence_len)
        N = feature dimension

        The `padded` argument here is set to True if the batch is padded. Otherwise, we
        expect a single batch, so that sentences can vary in length across batches.
        """
        basal = self._get_control_tensor(batch, padded)
        if getattr(self, "use_mix_pert", False):
            pert = None
        else:
            if padded:
                pert = batch["pert_emb"].unsqueeze(1).repeat(1, self.cell_sentence_len, 1)
            else:
                # we are inferencing on a single batch, so accept variable length sentences
                pert = batch["pert_emb"].unsqueeze(1).repeat(1, basal.shape[1], 1)

        # Shape: [B, S, input_dim]
        if getattr(self, "use_mix_pert", False):
            mix_pert = self.encode_perturbation(None, batch=batch)  # (B, H)
            # Expand across sequence length to match basal sentence
            seq_len = self.cell_sentence_len if padded else basal.shape[1]
            pert_embedding = mix_pert.unsqueeze(1).repeat(1, seq_len, 1)
        else:
            pert_embedding = self.encode_perturbation(pert)
        control_cells = self.encode_basal_expression(basal)

        # Add encodings in input_dim space, then project to hidden_dim
        combined_input = pert_embedding + control_cells  # Shape: [B, S, hidden_dim]
        seq_input = combined_input  # Shape: [B, S, hidden_dim]

        if self.batch_encoder is not None:
            # Extract batch indices (assume they are integers or convert from one-hot)
            batch_indices = batch["batch"]

            # Handle one-hot encoded batch indices
            if batch_indices.dim() > 1 and batch_indices.size(-1) == self.batch_dim:
                batch_indices = batch_indices.argmax(-1)

            # Reshape batch indices to match sequence structure
            if padded:
                batch_indices = batch_indices.unsqueeze(1).repeat(1, self.cell_sentence_len)
            else:
                batch_indices = batch_indices.unsqueeze(1).repeat(1, basal.shape[1])

            # Get batch embeddings and add to sequence input
            batch_embeddings = self.batch_encoder(batch_indices.long())  # Shape: [B, S, hidden_dim]
            seq_input = seq_input + batch_embeddings

        if self.use_batch_token and self.batch_token is not None:
            batch_size, _, _ = seq_input.shape
            # Prepend the batch token to the sequence along the sequence dimension
            # [B, S, H] -> [B, S+1, H], batch token at position 0
            seq_input = torch.cat([self.batch_token.expand(batch_size, -1, -1), seq_input], dim=1)

        confidence_pred = None
        if self.confidence_token is not None:
            # Append confidence token: [B, S, E] -> [B, S+1, E] (might be one more if we have the batch token)
            seq_input = self.confidence_token.append_confidence_token(seq_input)

        # forward pass + extract CLS last hidden state
        if self.hparams.get("mask_attn", False):
            batch_size, seq_length, _ = seq_input.shape
            device = seq_input.device
            self.transformer_backbone._attn_implementation = "eager"   # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

            # create a [1,1,S,S] mask (now S+1 if confidence token is used)
            base = torch.eye(seq_length, device=device, dtype=torch.bool).view(1, 1, seq_length, seq_length)
            
            # Get number of attention heads from model config
            num_heads = self.transformer_backbone.config.num_attention_heads

            # repeat out to [B,H,S,S]
            attn_mask = base.repeat(batch_size, num_heads, 1, 1)

            outputs = self.transformer_backbone(inputs_embeds=seq_input, attention_mask=attn_mask)
            transformer_output = outputs.last_hidden_state
        else:
            outputs = self.transformer_backbone(inputs_embeds=seq_input)
            transformer_output = outputs.last_hidden_state

        # Extract outputs accounting for optional prepended batch token and optional confidence token at the end
        if self.confidence_token is not None and self.use_batch_token and self.batch_token is not None:
            # transformer_output: [B, 1 + S + 1, H] -> batch token at 0, cells 1..S, confidence at -1
            batch_token_pred = transformer_output[:, :1, :]  # [B, 1, H]
            res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(
                transformer_output[:, 1:, :]
            )
            # res_pred currently excludes the confidence token and starts from former index 1
            self._batch_token_cache = batch_token_pred
        elif self.confidence_token is not None:
            # Only confidence token appended at the end
            res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(transformer_output)
            self._batch_token_cache = None
        elif self.use_batch_token and self.batch_token is not None:
            # Only batch token prepended at the beginning
            batch_token_pred = transformer_output[:, :1, :]  # [B, 1, H]
            res_pred = transformer_output[:, 1:, :]  # [B, S, H]
            self._batch_token_cache = batch_token_pred
        else:
            # Neither special token used
            res_pred = transformer_output
            self._batch_token_cache = None

        # add to basal if predicting residual
        if self.predict_residual and self.gene_decoder is None:
            # Project control_cells to hidden_dim space to match res_pred
            # control_cells_hidden = self.project_to_hidden(control_cells)
            # treat the actual prediction as a residual sum to basal
            out_pred = self.project_out(res_pred) + basal
            out_pred = self.final_down_then_up(out_pred)
        elif self.predict_residual:
            out_pred = self.project_out(res_pred + control_cells)
        else:
            out_pred = self.project_out(res_pred)


        if self.gene_decoder is None:
            out_pred=self.relu(out_pred)

        output = out_pred.reshape(-1, self.output_dim)

        if confidence_pred is not None:
            return output, confidence_pred
        else:
            return output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # Get model predictions (in latent space)
        confidence_pred = None
        if self.confidence_token is not None:
            pred, confidence_pred = self.forward(batch, padded=padded)
        else:
            pred = self.forward(batch, padded=padded)

        target = self._get_target_tensor(batch, padded)

        # GeomLoss expects pred/target to have matching dimensions (e.g. [B, S, D]).
        # Forward() returns a flattened [B * S, D] tensor, so we reshape it back here
        # to be consistent with `_get_target_tensor` when `padded=True`.
        if padded:
            pred_for_loss = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            B = pred_for_loss.shape[0]
        else:
            pred_for_loss = pred.reshape(1, -1, self.output_dim)
            B = 1

        main_loss = self.loss_fn(pred_for_loss, target).nanmean()
        self.log("train_loss", main_loss, on_step=True, on_epoch=True, batch_size=B)

        # Log individual loss components if using combined loss
        if hasattr(self.loss_fn, "sinkhorn_loss") and hasattr(self.loss_fn, "energy_loss"):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred_for_loss, target).nanmean()
            energy_component = self.loss_fn.energy_loss(pred_for_loss, target).nanmean()
            self.log("train/sinkhorn_loss", sinkhorn_component)
            self.log("train/energy_loss", energy_component)

        # Process decoder if available
        decoder_loss = None
        total_loss = main_loss

        if self.use_batch_token and self.batch_classifier is not None and self._batch_token_cache is not None:
            logits = self.batch_classifier(self._batch_token_cache)  # [B, 1, C]
            batch_token_targets = batch["batch"]
            # B = logits.shape[0]
            C = logits.size(-1)

            # Prepare one label per sequence (all S cells share the same batch)
            if batch_token_targets.dim() > 1 and batch_token_targets.size(-1) == C:
                # One-hot labels; reshape to [B, S, C]
                if padded:
                    target_oh = batch_token_targets.reshape(-1, self.cell_sentence_len, C)
                else:
                    target_oh = batch_token_targets.reshape(1, -1, C)
                sentence_batch_labels = target_oh.argmax(-1)
            else:
                # Integer labels; reshape to [B, S]
                if padded:
                    sentence_batch_labels = batch_token_targets.reshape(-1, self.cell_sentence_len)
                else:
                    sentence_batch_labels = batch_token_targets.reshape(1, -1)

            if sentence_batch_labels.shape[0] != B:
                sentence_batch_labels = sentence_batch_labels.reshape(B, -1)

            if self.basal_mapping_strategy == "batch":
                uniform_mask = sentence_batch_labels.eq(sentence_batch_labels[:, :1]).all(dim=1)
                if not torch.all(uniform_mask):
                    bad_indices = torch.where(~uniform_mask)[0]
                    label_strings = []
                    for idx in bad_indices:
                        labels = sentence_batch_labels[idx].detach().cpu().tolist()
                        logger.error("Batch labels for sentence %d: %s", idx.item(), labels)
                        label_strings.append(f"sentence {idx.item()}: {labels}")
                    raise ValueError(
                        "Expected all cells in a sentence to share the same batch when "
                        "basal_mapping_strategy is 'batch'. "
                        f"Found mixed batch labels: {', '.join(label_strings)}"
                    )

            target_idx = sentence_batch_labels[:, 0]

            # Safety: ensure exactly one target per sequence
            if target_idx.numel() != B:
                target_idx = target_idx.reshape(-1)[:B]

            ce_loss = F.cross_entropy(logits.reshape(B, -1, C).squeeze(1), target_idx.long())
            self.log("train/batch_token_loss", ce_loss)
            total_loss = total_loss + self.batch_token_weight * ce_loss

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]
            # Train decoder to map latent predictions to gene space

            if self.detach_decoder:
                # with some random change, use the true targets
                if np.random.rand() < 0.1:
                    latent_preds = target.reshape_as(pred).detach()
                else:
                    latent_preds = pred.detach()
            else:
                latent_preds = pred

            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_preds)
                if padded:
                    gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                else:
                    gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = total_loss.detach().clone().unsqueeze(0) * 10

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("train/confidence_loss", confidence_loss)
            self.log("train/actual_loss", loss_target.mean())

            # Add to total loss with weighting
            confidence_weight = 0.1  # You can make this configurable
            total_loss = total_loss + confidence_weight * confidence_loss

            # Add to total loss
            total_loss = total_loss + confidence_loss

        if self.regularization > 0.0:
            ctrl_cell_emb = batch["ctrl_cell_emb"].reshape_as(pred)
            delta = pred - ctrl_cell_emb

            # compute l1 loss
            l1_loss = torch.abs(delta).mean()

            # Log the regularization loss
            self.log("train/l1_regularization", l1_loss)

            # Add regularization to total loss
            total_loss = total_loss + self.regularization * l1_loss

        self.log('_train_loss',
                 total_loss,
                 on_step=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=B)

        # Compute and log training PCC (use mask if enabled)
        # Reshape pred and target for PCC computation
        pred_2d = pred.reshape(-1, self.output_dim)
        target_2d = target.reshape(-1, self.output_dim)
        
        # Get mask if use_mask is enabled
        mask = self._get_mask(batch)
        if mask is not None:
            # Reshape mask to match pred_2d shape if needed
            if mask.dim() == 2:
                mask_2d = mask
            else:
                mask_2d = mask.reshape(-1, self.output_dim) if mask.numel() == pred_2d.numel() else None
            # Ensure mask is on the same device as predictions
            if mask_2d is not None:
                mask_2d = mask_2d.to(pred_2d.device)
        else:
            mask_2d = None
        
        # Compute PCC (use mask if enabled)
        train_pcc = self._compute_masked_pcc(pred_2d, target_2d, mask_2d)
        self.log("train_PCC", train_pcc, prog_bar=True, logger=True, batch_size=B, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, data_tuple:tuple[any,pd.DataFrame], batch_idx: int) -> None:
        """Validation step logic."""
        batch,_=data_tuple
        if self.confidence_token is None:
            pred, confidence_pred = self.forward(batch,padded=False), None
        else:
            pred, confidence_pred = self.forward(batch,padded=False)

        pred = pred.reshape(1, -1, self.output_dim)
        target = self._get_target_tensor(batch, padded=False)

        loss = self.loss_fn(pred, target).mean()
        self.log("val_loss", loss, batch_size=1, on_step=True, on_epoch=True)

        # Log individual loss components if using combined loss
        if hasattr(self.loss_fn, "sinkhorn_loss") and hasattr(self.loss_fn, "energy_loss"):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).mean()
            energy_component = self.loss_fn.energy_loss(pred, target).mean()
            self.log("val/sinkhorn_loss", sinkhorn_component, batch_size=1)
            self.log("val/energy_loss", energy_component, batch_size=1)

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]

            # Get model predictions from validation step
            latent_preds = pred

            # Train decoder to map latent predictions to gene space
            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                # Get decoder predictions
                pert_cell_counts_preds = self.gene_decoder(latent_preds).reshape(
                    1, -1, self.gene_decoder.gene_dim()
                )
                gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())
                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log the validation metric
            self.log("val/decoder_loss", decoder_loss, batch_size=1)
            loss = loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone() * 10

            # Ensure proper shapes for confidence loss computation
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:  # confidence_pred is [B,]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("val/confidence_loss", confidence_loss, batch_size=1)
            self.log("val/actual_loss", loss_target.mean(), batch_size=1)

        self.log('_val_loss',loss,
                 on_step=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=1)

        # Compute and log validation PCC (use mask if enabled)
        # Reshape pred and target from [1, -1, output_dim] to [-1, output_dim] for PCC computation
        pred_2d = pred.reshape(-1, self.output_dim)
        target_2d = target.reshape(-1, self.output_dim)
        
        # Get mask if use_mask is enabled
        mask = self._get_mask(batch)
        if mask is not None:
            # Reshape mask to match pred_2d shape if needed
            if mask.dim() == 2:
                mask_2d = mask
            else:
                mask_2d = mask.reshape(-1, self.output_dim) if mask.numel() == pred_2d.numel() else None
            # Ensure mask is on the same device as predictions
            if mask_2d is not None:
                mask_2d = mask_2d.to(pred_2d.device)
        else:
            mask_2d = None
        
        # Compute PCC (use mask if enabled)
        val_pcc = self._compute_masked_pcc(pred_2d, target_2d, mask_2d)
        self.log("val_PCC", val_pcc, prog_bar=True, logger=True, batch_size=1, on_step=True, on_epoch=True)

        return {"loss": loss, "predictions": pred}

    def test_step(self, data_tuple:tuple[any,pd.DataFrame], batch_idx: int) -> None:
        """Test step: 将 per-cell 预测在 cell 维度上聚合为 per-observation 预测。"""
        batch, obs_df = data_tuple

        # 使用关键字参数避免与 Lightning 的 hook 签名冲突
        out_dict = self.predict_step(batch, padded=False, batch_idx=batch_idx)

        pred_expr = out_dict.get("pert_cell_counts_preds", None)
        if pred_expr is None:
            pred_expr = out_dict.get("preds", None)

        # 转成 numpy
        pred_expr_np = pred_expr.detach().cpu().numpy()
        n_pred, n_gene = pred_expr_np.shape
        n_obs = len(obs_df)

        # 当前 state 配置下：一个 obs（条件）对应若干个 cell 的预测（例如 32 个），
        # 因此这里将 [n_pred, G] 重塑并在 cell 维上做平均，得到 [n_obs, G]，
        # 与其它模型的“每个 obs 一行预测”语义对齐。
        if n_obs == 0:
            raise ValueError("obs_df has zero rows in test_step.")
        if n_pred % n_obs != 0:
            raise ValueError(
                f"Prediction rows ({n_pred}) not an integer multiple of obs rows ({n_obs}); "
                "cannot aggregate per-observation predictions."
            )
        factor = n_pred // n_obs  # 每个 obs 对应的 cell 数
        pred_expr_np = pred_expr_np.reshape(n_obs, factor, n_gene).mean(axis=1)

        self.preds_list.append((pred_expr_np, obs_df))

    def predict(self, batch):

        out_dict = self.predict_step(batch, padded=False)
        pred_expr = out_dict.get('pert_cell_counts_preds', None)
        if pred_expr is None:
            pred_expr = out_dict.get('preds', None)
        return pred_expr


    def predict_step(self, batch, padded=False, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:s
         returning 'preds', 'X', 'pert_name', etc.
        """
        if self.confidence_token is None:
            latent_output = self.forward(batch, padded=padded)  # shape [B, ...]
            confidence_pred = None
        else:
            latent_output, confidence_pred = self.forward(batch, padded=padded)

        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
            "pert_cell_barcode": batch.get("pert_cell_barcode", None),
            "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
        }

        # Add confidence prediction to output if available
        if confidence_pred is not None:
            output_dict["confidence_pred"] = confidence_pred

        if self.gene_decoder is not None:
            if isinstance(self.gene_decoder, NBDecoder):
                mu, _ = self.gene_decoder(latent_output)
                pert_cell_counts_preds = mu
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_output)

            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict

    def configure_optimizers(self):
        """
        Configure a single optimizer for both the main model and the gene decoder.
        """
        # Use a single optimizer for all parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
