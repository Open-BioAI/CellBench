import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as kl
from torchmetrics.functional import accuracy
from typing import Optional, Literal
import logging
import lightning as L

from ..nn.vae import VariationalEncoder
from ..nn.decoders import (
    DeepIsotropicGaussian,
)
from ..nn.mlp import MLP
from ..nn import MixedPerturbationEncoder

from .base import PerturbationModel

log = logging.getLogger(__name__)


class CPA(PerturbationModel):
    """
    CPA module using Gaussian/NegativeBinomial/Zero-InflatedNegativeBinomial Likelihood
    """

    def __init__(
            self,
            use_cell_emb: bool =False,
            use_mask: bool = False,  # Unified mask switch for training loss and evaluation
            context: dict | None = None,
            n_latent: int = 128,
            library_size: Literal["learned", "observed"] | None = None,
            hidden_dim: int = 256,
            n_layers_encoder: int = 3,
            n_layers_pert_emb: int = 2,
            n_layers_covar_emb: int = 1,
            adv_classifier_hidden_dim: int = 128,
            adv_classifier_n_layers: int = 2,
            variational: bool = True,
            lr: float = 1e-3,
            wd: float = 1e-8,
            lr_scheduler_freq: int | None = None,
            lr_scheduler_interval: str | None = None,
            lr_scheduler_patience: int | None = None,
            lr_scheduler_factor: float | None = None,
            lr_scheduler_mode: str | None = None,
            lr_scheduler_max_lr: float | None = None,
            lr_scheduler_total_steps: int | None = None,
            lr_scheduler_step_size: int | None = None,
            lr_scheduler_gamma: float | None = None,
            kl_weight: float = 1.0,
            adv_weight: float = 1.0,
            dropout: float = 0.1,
            penalty_weight: float = 10.0,
            adv_steps: int = 7,
            n_warmup_epochs: int = 5,
            use_adversary: bool = True,
            use_covariates: bool = True,
            softplus_output: bool = False,
            elementwise_affine: bool = False,
            datamodule: L.LightningDataModule | None = None,
            # add decoder_distribution
            decoder_distribution: str = "IsotropicGaussian",  # 👈 加上这一行
            **kwargs
    ):
        """The constructor for the CPA module class.
        Args:
            n_genes: Number of genes.
            n_perts: Number of perturbations.
            drug_embeddings: Drug embeddings.
            n_latent: Number of latent variables.
            decoder_distribution: Distribution of the decoder.
            hidden_dim: Hidden dimension.
            n_layers_encoder: Number of encoder layers.
            n_layers_decoder: Number of decoder layers.
            n_layers_pert_emb: Number of perturbation embedding layers.
            n_layers_covar_emb: Number of covariate embedding layers.
            adv_classifier_hidden_dim: Adversarial classifier hidden dimension.
            adv_classifier_n_layers: Number of adversarial classifier layers.
            variational: Whether to use variational autoencoder.
            seed: Random seed.
            lr: Learning rate.
            wd: Weight decay.
            kl_weight: KL divergence weight.
            adv_weight: Adversarial weight.
            dropout: Dropout rate.
            penalty_weight: Penalty weight.
            datamodule: Data module.
            adv_steps: Number of adversarial steps.
            n_warmup_epochs: Number of warmup epochs for the autoencoder
            use_adversary: Whether to use the adversarial component.
            use_covariates: Whether to use additive covariate conditioning.
            softplus_output: Whether to apply a softplus activation to the output.
            elementwise_affine: Whether to use elementwise affine in the layer norms
        """

        super(CPA, self).__init__(
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
            lr_scheduler_step_size=lr_scheduler_step_size,
            lr_scheduler_gamma=lr_scheduler_gamma,
            use_mask=use_mask,  # Pass use_mask to base class
        )
        self.n_input_features=self.embedding_dim if use_cell_emb else self.n_genes
        self.use_cell_emb = use_cell_emb
        self.save_hyperparameters(ignore=["datamodule"])
        self.automatic_optimization = False
        base_n_perts = datamodule.train_dataset.transform.n_perts
        self.n_perts = (
            self.gene_pert_dim + self.drug_pert_dim + self.env_pert_dim
            if getattr(self, "use_mix_pert", False)
            else base_n_perts
        )
        self.n_latent = n_latent
        self.variational = variational
        self.hidden_dim = hidden_dim
        self.n_layers_pert_emb = n_layers_pert_emb
        self.n_layers_covar_emb = n_layers_covar_emb
        self.n_layers_encoder = n_layers_encoder
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight
        self.penalty_weight = penalty_weight
        self.adv_classifier_hidden_dim = adv_classifier_hidden_dim
        self.adv_classifier_n_layers = adv_classifier_n_layers
        self.dropout = dropout
        self.adv_steps = adv_steps
        self.n_warmup_epochs = n_warmup_epochs
        self.softplus_output = softplus_output
        self.multi_label_perts = bool(getattr(self, "use_mix_pert", False))
        self.adv_loss_drugs = nn.BCEWithLogitsLoss() if self.multi_label_perts else nn.CrossEntropyLoss()
        self.adv_loss_fn = nn.CrossEntropyLoss()
        self.use_adversary = use_adversary
        # Only enable covariates if the transform actually provides maps
        transform_has_covs = hasattr(datamodule.train_dataset.transform, "cov_maps")
        self.use_covariates = use_covariates and transform_has_covs
        # add decoder_distribution
        self.decoder_distribution = decoder_distribution

        self.encoder = VariationalEncoder(
            input_dim=self.n_input_features,
            hidden_dim=self.hidden_dim,
            latent_dim=self.n_latent,
            n_layers=self.n_layers_encoder,
            dropout=self.dropout,
        )

        if self.use_covariates:
            self.covars_encoder = getattr(datamodule.train_dataset.transform, "cov_maps", {})

        else:
            self.covars_encoder = {}

        self.decoder = DeepIsotropicGaussian(
            self.n_latent,
            self.hidden_dim,
            self.n_genes,
            self.n_layers_encoder,
            self.dropout,
            self.softplus_output,
        )

        self.pert_network = None
        self.pert_encoder = None
        if getattr(self, "use_mix_pert", False):
            hidden_dims = [self.hidden_dim] * (self.n_layers_pert_emb - 1) if self.n_layers_pert_emb > 1 else []
            self.pert_encoder = MixedPerturbationEncoder(
                gene_pert_dim=self.gene_pert_dim,
                drug_pert_dim=self.drug_pert_dim,
                env_pert_dim=self.env_pert_dim,
                hidden_dims=hidden_dims,
                final_embed_dim=self.n_latent,
                dropout=self.dropout,
            )
        else:
            self.pert_network = MLP(
                input_dim=self.n_perts,
                hidden_dim=self.hidden_dim,
                output_dim=self.n_latent,
                n_layers=self.n_layers_pert_emb,
                dropout=self.dropout,
                elementwise_affine=elementwise_affine,
            )

        self.perturbation_adversary_classifier = MLP(
            self.n_latent,
            self.adv_classifier_hidden_dim,
            self.n_perts,
            self.adv_classifier_n_layers,
            self.dropout,
            elementwise_affine=elementwise_affine,
        )

        if self.use_covariates:
            self.covars_embeddings = nn.ModuleDict(
                {
                    key: MLP(
                        input_dim=len(list(map.keys())),
                        output_dim=n_latent,
                        hidden_dim=hidden_dim,
                        n_layers=self.n_layers_covar_emb,
                        dropout=dropout,
                        elementwise_affine=elementwise_affine,
                    )
                    for key, map in self.covars_encoder.items()
                    if len(list(map.keys())) > 1
                }
            )

            self.covars_adversary_classifiers = dict()
            # Iterate over covars_encoder (dict) to get the number of classes for each covariate
            # Only create classifiers for covariates that have embeddings
            for covar, cov_map in self.covars_encoder.items():
                if covar in self.covars_embeddings:
                    self.covars_adversary_classifiers[covar] = MLP(
                        self.n_latent,
                        self.adv_classifier_hidden_dim,
                        len(list(cov_map.keys())),  # Use cov_map (dict) instead of MLP
                        self.adv_classifier_n_layers,
                        self.dropout,
                        elementwise_affine=elementwise_affine,
                    )
            self.covars_adversary_classifiers = nn.ModuleDict(
                self.covars_adversary_classifiers
            )

        else:
            self.covars_embeddings = None
            self.covars_adversary_classifiers = None

        gen_modules = [self.encoder, self.decoder]
        gen_modules.append(self.pert_encoder if self.pert_encoder is not None else self.pert_network)
        if self.covars_embeddings is not None:
            gen_modules.append(self.covars_embeddings)
        self.generative_modules = nn.ModuleList([m for m in gen_modules if m is not None])

        adv_modules = [self.perturbation_adversary_classifier]
        if self.covars_adversary_classifiers is not None:
            adv_modules.append(self.covars_adversary_classifiers)
        self.adversary_modules = nn.ModuleList([m for m in adv_modules if m is not None])

    @property
    def start_adv_training(self):
        if self.n_warmup_epochs:
            return self.current_epoch > self.n_warmup_epochs
        else:
            return True

    def unpack_batch(self, batch):

        if hasattr(batch, "controls"):
            x = batch.controls  # batch_size, n_genes
        else:
            x = batch.control_cell_emb if self.use_cell_emb else batch.control_cell_counts

        perts = batch if getattr(self, "use_mix_pert", False) else batch[self.pert_key]
        covars_dict = {cov_key:batch[cov_key] for cov_key in self.cov_keys}

        return dict(
            x=x,
            perts=perts,
            covars_dict=covars_dict,
        )

    def inference(
            self,
            x: torch.Tensor,
            perts: torch.Tensor,
            covars_dict: dict[str, torch.Tensor],
            n_samples: int = 1,
            covars_to_add: Optional[list] = None,
    ):

        x_ = x
        library = None

        if self.variational:
            qz, z_basal = self.encoder(x_)
        else:
            qz, z_basal = None, self.encoder(x_)

        if self.variational and n_samples > 1:
            sampled_z = qz.sample((n_samples,))
            z_basal = self.encoder.z_transformation(sampled_z)

        if getattr(self, "use_mix_pert", False):
            z_pert_true = self.pert_encoder(perts)  # perturbation encoder for mixed perturbations
        else:
            z_pert_true = self.pert_network(perts)  # perturbation encoder
        z_pert = z_pert_true
        z_covs_dict = torch.zeros_like(z_basal)  # ([n_samples,] batch_size, n_latent)

        if covars_to_add is None:
            covars_to_add = list(self.covars_encoder.keys())

        z_covs_dict = {}
        if self.covars_embeddings is not None:
            for covar in self.covars_embeddings:
                if covar in covars_to_add:
                    covars_input = covars_dict[covar]
                    z_cov = self.covars_embeddings[covar](covars_input)
                    z_covs_dict[covar] = z_cov

        if len(z_covs_dict) > 0:
            z_covs = torch.stack(list(z_covs_dict.values()), dim=0).sum(dim=0)
        else:
            z_covs = torch.zeros_like(z_basal)

        z = z_basal + z_pert + z_covs
        z_no_pert = z_basal + z_covs

        return dict(
            z=z,
            z_no_pert=z_no_pert,
            z_basal=z_basal,
            z_covs=z_covs,
            z_pert=z_pert.sum(dim=1),
            library=library,
            qz=qz,
        )

    def generative(
            self,
            z: torch.Tensor,
            library: torch.Tensor | None = None,
    ):
        predictions = self.decoder(z)
        return {
            "predictions": predictions,
            "pz": Normal(torch.zeros_like(z), torch.ones_like(z)),
        }

    def _aggregate_drug_pert(self, drug_pert_batch, device, dtype):
        if isinstance(drug_pert_batch, torch.Tensor):
            return drug_pert_batch.to(device=device, dtype=dtype)

        aggregated = []
        for perts in drug_pert_batch:
            if isinstance(perts, torch.Tensor):
                if perts.dim() > 1:
                    aggregated.append(perts.to(device=device, dtype=dtype).sum(dim=0))
                else:
                    aggregated.append(perts.to(device=device, dtype=dtype))
            elif isinstance(perts, (list, tuple)) and len(perts) > 0:
                stacked = torch.stack(perts).to(device=device, dtype=dtype)
                aggregated.append(stacked.sum(dim=0))
            else:
                aggregated.append(torch.zeros(self.drug_pert_dim, device=device, dtype=dtype))

        return torch.stack(aggregated, dim=0)

    def _get_perturbations(self, batch):
        if getattr(self, "use_mix_pert", False):
            gene = batch.gene_pert
            env = batch.env_pert
            drug = self._aggregate_drug_pert(batch.drug_pert, gene.device, gene.dtype)
            targets = torch.cat([gene, drug, env], dim=1)
            return (targets > 0).float()

        return batch[self.pert_key]

    def loss(
            self,
            x: torch.Tensor,
            perturbations: torch.Tensor,
            covariates: dict[str, torch.Tensor],
            inference_outputs: dict[str, torch.Tensor],
            generative_outputs: dict[str, torch.Tensor],
            batch_idx: int,
            batch: Optional[dict] = None,
    ):
        """Computes the reconstruction loss (AE) or the ELBO (VAE)"""
        # Use expression mask for loss calculation - only compute loss on expressed genes
        mask = self._get_mask(batch)
        if mask is not None:
            # Get per-gene loss (shape: [batch_size, n_genes])
            predictions = generative_outputs["predictions"]
            masked_loss = F.mse_loss(
                predictions,
                x,
                reduction='none'
            )  # shape: [batch_size, n_genes]

            mask = mask.to(masked_loss.device)
            valid = mask.sum()

            if valid > 0:
                recon_loss = (masked_loss * mask).sum() / valid
            else:
                # Fallback to mean if mask is all zeros
                recon_loss = masked_loss.mean()
        else:
            # Fallback to standard reconstruction loss when use_mask=False or no mask available
            recon_loss = self.decoder.reconstruction_loss(
                generative_outputs["predictions"], x
            )

        if self.variational:
            qz = inference_outputs["qz"]
            pz = generative_outputs["pz"]  # just a standard gaussian

            kl_divergence_z = kl(qz, pz).sum(dim=1)
            kl_loss = kl_divergence_z.mean()
        else:
            kl_loss = torch.zeros_like(recon_loss)

        if self.use_adversary:
            adv_loss = self.adversarial_loss(
                perturbations, covariates, inference_outputs["z_basal"], self.training
            )
        else:
            adv_loss = {
                "adv_loss": torch.zeros_like(recon_loss),
                "penalty_adv": torch.zeros_like(recon_loss),
                "penalty_covars": torch.zeros_like(recon_loss),
                "penalty_perts": torch.zeros_like(recon_loss),
                "acc_perts": torch.zeros_like(recon_loss),
                "covariate_classfier_loss": torch.zeros_like(recon_loss),
                "perturbation_classifier_loss": torch.zeros_like(recon_loss),
            }

        total_loss = (
                recon_loss
                + self.kl_weight * kl_loss
                - self.adv_weight * adv_loss["adv_loss"]
        )

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "adv_loss": adv_loss["adv_loss"],
            "covariate_classfier_loss": adv_loss["covariate_classfier_loss"],
            "perturbation_classifier_loss": adv_loss["perturbation_classifier_loss"],
            "penalty_adv": adv_loss["penalty_adv"],
            "penalty_covars": adv_loss["penalty_covars"],
            "penalty_perts": adv_loss["penalty_perts"],
            "acc_perts": adv_loss["acc_perts"],
        }

    def adversarial_loss(
            self,
            perturbations: torch.Tensor,
            covariates: dict[str, torch.Tensor],
            z_basal: torch.Tensor,
            compute_penalty: bool = True,
    ):
        """Computes adversarial classification losses and regularizations"""
        if compute_penalty:
            z_basal = z_basal.requires_grad_(True)

        covars_pred_logits = {}
        if self.covars_embeddings is not None:
            for covar in self.covars_embeddings.keys():
                if self.covars_adversary_classifiers[covar] is not None:
                    covars_pred_logits[covar] = self.covars_adversary_classifiers[covar](
                        z_basal
                    )
                else:
                    covars_pred_logits[covar] = None

        adv_results = {}

        # Classification losses for different covariates
        if self.covars_embeddings is not None:
            for covar in self.covars_embeddings.keys():
                adv_results[f"adv_{covar}"] = (
                    self.adv_loss_fn(  # we've removed mixup for now
                        covars_pred_logits[covar],
                        covariates[covar],
                    )
                    if covars_pred_logits[covar] is not None
                    else torch.as_tensor(0.0).to(self.device)
                )

                adv_results[f"acc_{covar}"] = (
                    accuracy(
                        covars_pred_logits[covar].argmax(1),
                        covariates[covar].argmax(1),
                        task="multiclass",
                        num_classes=len(self.covars_encoder[covar]),  # Use covars_encoder dict
                    )
                    if covars_pred_logits[covar] is not None
                    else torch.as_tensor(0.0).to(self.device)
                )

        if self.covars_embeddings is not None and len(self.covars_embeddings) > 0:
            adv_results["covariate_classfier_loss"] = sum(
                [adv_results[f"adv_{key}"] for key in self.covars_embeddings.keys()]
            )
        else:
            adv_results["covariate_classfier_loss"] = torch.as_tensor(0.0).to(
                self.device
            )

        # TODO Not using mixups for now.
        perturbations_pred_logits = self.perturbation_adversary_classifier(z_basal)

        if self.multi_label_perts:
            target_multi = (perturbations > 0).float()
            adv_results["perturbation_classifier_loss"] = self.adv_loss_drugs(
                perturbations_pred_logits, target_multi
            )
            adv_results["acc_perts"] = accuracy(
                torch.sigmoid(perturbations_pred_logits),
                target_multi,
                task="multilabel",
                num_labels=self.n_perts,
            )
        else:
            target_perts = (
                perturbations.argmax(1)
                if perturbations.dim() > 1
                else perturbations.long().view(-1)
            )
            adv_results["perturbation_classifier_loss"] = self.adv_loss_drugs(
                perturbations_pred_logits, target_perts
            )

            adv_results["acc_perts"] = accuracy(
                perturbations_pred_logits.argmax(1),
                target_perts,
                average="macro",
                num_classes=self.n_perts,
                task="multiclass",
            )

        adv_results["adv_loss"] = (
                adv_results["covariate_classfier_loss"]
                + adv_results["perturbation_classifier_loss"]
        )

        if compute_penalty:
            # Penalty losses
            if self.covars_embeddings is not None:
                for covar in self.covars_embeddings.keys():
                    adv_results[f"penalty_{covar}"] = (
                        (
                            torch.autograd.grad(
                                covars_pred_logits[covar].sum(),
                                z_basal,
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True,
                            )[0]
                            .pow(2)
                            .mean()
                        )
                        if covars_pred_logits[covar] is not None
                        else torch.as_tensor(0.0).to(self.device)
                    )

                if len(self.covars_embeddings) > 0:
                    adv_results["penalty_covars"] = sum(
                        [
                            adv_results[f"penalty_{covar}"]
                            for covar in self.covars_embeddings.keys()
                        ]
                    )
                else:
                    adv_results["penalty_covars"] = torch.as_tensor(0.0).to(self.device)
            else:
                adv_results["penalty_covars"] = torch.as_tensor(0.0).to(self.device)

            adv_results["penalty_perts"] = (
                torch.autograd.grad(
                    perturbations_pred_logits.sum(),
                    z_basal,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                .pow(2)
                .mean()
            )

            adv_results["penalty_adv"] = (
                    adv_results["penalty_perts"] + adv_results["penalty_covars"]
            )
        else:
            if self.covars_embeddings is not None:
                for covar in self.covars_embeddings.keys():
                    adv_results[f"penalty_{covar}"] = torch.as_tensor(0.0).to(self.device)
            else:
                adv_results["penalty_covars"] = torch.as_tensor(0.0).to(self.device)
                adv_results["penalty_perts"] = torch.as_tensor(0.0).to(self.device)
                adv_results["penalty_adv"] = torch.as_tensor(0.0).to(self.device)
                return adv_results

            adv_results["penalty_covars"] = torch.as_tensor(0.0).to(self.device)
            adv_results["penalty_perts"] = torch.as_tensor(0.0).to(self.device)
            adv_results["penalty_adv"] = torch.as_tensor(0.0).to(self.device)

        return adv_results

    def _get_dict_if_none(param):
        param = {} if not isinstance(param, dict) else param

        return param

    def forward(
            self,
            batch,
    ):
        x, perts, covars_dict = self.unpack_batch(batch).values()
        inference_outputs = self.inference(x, perts, covars_dict)

        generative_outputs = self.generative(
            inference_outputs["z"], inference_outputs["library"]
        )
        return inference_outputs, generative_outputs

    def training_step(self, batch, batch_idx: int):
        optimizer_generative, optimizer_adversary = self.optimizers()

        perturbations = self._get_perturbations(batch)
        inference_outputs, generative_outputs = self.forward(batch)
        losses = self.loss(
            batch.pert_cell_counts,
            perturbations,
            {cov_key:batch[cov_key] for cov_key in self.cov_keys},
            inference_outputs,
            generative_outputs,
            batch_idx,
            batch=batch,
        )

        total_loss = losses["total_loss"]
        adv_loss = (
                self.adv_weight * losses["adv_loss"]
                + self.penalty_weight * losses["penalty_adv"]
        )

        if self.start_adv_training:
            if batch_idx % self.adv_steps == 0:
                self.toggle_optimizer(optimizer_generative)
                optimizer_generative.zero_grad()
                self.manual_backward(total_loss)
                optimizer_generative.step()
                self.untoggle_optimizer(optimizer_generative)

            else:
                self.toggle_optimizer(optimizer_adversary)
                optimizer_adversary.zero_grad()
                self.manual_backward(adv_loss)
                optimizer_adversary.step()
                self.untoggle_optimizer(optimizer_adversary)

        else:
            gen_loss = losses["recon_loss"] + self.kl_weight * losses["kl_loss"]
            self.toggle_optimizer(optimizer_generative)
            optimizer_generative.zero_grad()
            self.manual_backward(gen_loss)
            optimizer_generative.step()
            self.untoggle_optimizer(optimizer_generative)

        if self.training:
            for key, value in losses.items():
                self.log(
                    "train_" + key,
                    value,
                    prog_bar=True,
                    logger=True,
                    batch_size=len(batch),
                    on_step=True,
                    on_epoch=True,
                )
            
            # Compute training PCC (use mask if enabled)
            predictions = generative_outputs["predictions"]
            observed = batch.pert_cell_counts
            mask = self._get_mask(batch)
            if mask is not None:
                mask = mask.to(predictions.device)
            train_pcc = self._compute_masked_pcc(predictions, observed, mask)
            self.log("train_PCC", train_pcc, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        # =================== [FIX] Manual scheduler step for manual_optimization mode
        # Since automatic_optimization=False, Lightning won't call scheduler.step() automatically
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            # schedulers can be a single scheduler or a list
            if isinstance(schedulers, (list, tuple)):
                for sch in schedulers:
                    if sch is not None:
                        sch.step()
            else:
                schedulers.step()
        # =================== [FIX] End

    def validation_step(self, data_tuple, batch_idx: int):
        batch,_=data_tuple
        perturbations = self._get_perturbations(batch)
        inference_outputs, generative_outputs = self.forward(batch)
        losses = self.loss(
            batch.pert_cell_counts,
            perturbations,
            {cov_key:batch[cov_key] for cov_key in self.cov_keys},
            inference_outputs,
            generative_outputs,
            batch_idx,
            batch=batch,
        )
        total_loss = losses["total_loss"]
        # Log both total_loss and recon_loss for better monitoring
        # Note: total_loss can be negative due to adversarial training design
        self.log("val_loss", total_loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)
        self.log("val_recon_loss", losses["recon_loss"], prog_bar=False, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)
        
        # Compute validation PCC (use mask if enabled)
        predictions = generative_outputs["predictions"]
        observed = batch.pert_cell_counts
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(predictions.device)
        val_pcc = self._compute_masked_pcc(predictions, observed, mask)
        self.log("val_PCC", val_pcc, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)
        return total_loss

    def predict(self, batch):
        _, generative_outputs = self.forward(batch)
        return generative_outputs["predictions"]

    def configure_optimizers(self):
        """
        Configure optimizers and schedulers for CPA.
        Supports multiple scheduler modes: onecycle, step, or plateau (default).
        Both generative and adversary optimizers use the same scheduler mode.
        """
        optimizer_generative = torch.optim.Adam(
            self.generative_modules.parameters(), lr=self.lr, weight_decay=self.wd
        )
        optimizer_adversary = torch.optim.Adam(
            self.adversary_modules.parameters(), lr=self.lr, weight_decay=self.wd
        )

        # ============================== [NEW] Unified scheduler selection - supports onecycle/step/plateau modes
        # Select scheduler based on lr_scheduler_mode
        if self.lr_scheduler_mode == "onecycle":
            # OneCycleLR scheduler
            total_steps = self.lr_scheduler_total_steps
            if total_steps is None:
                try:
                    steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
                    total_steps = steps_per_epoch * self.trainer.max_epochs
                except Exception:
                    total_steps = 100 * 100  # fallback

            scheduler_generative = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_generative,
                max_lr=self.lr_scheduler_max_lr or self.lr,
                total_steps=total_steps,
            )
            scheduler_adversary = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_adversary,
                max_lr=self.lr_scheduler_max_lr or self.lr,
                total_steps=total_steps,
            )
            lr_scheduler_generative = {"scheduler": scheduler_generative, "interval": "step"}
            lr_scheduler_adversary = {"scheduler": scheduler_adversary, "interval": "step"}

        elif self.lr_scheduler_mode == "step":
            # StepLR scheduler
            scheduler_generative = torch.optim.lr_scheduler.StepLR(
                optimizer_generative,
                step_size=getattr(self, 'lr_scheduler_step_size', None) or 10,
                gamma=getattr(self, 'lr_scheduler_gamma', None) or 0.1,
            )
            scheduler_adversary = torch.optim.lr_scheduler.StepLR(
                optimizer_adversary,
                step_size=getattr(self, 'lr_scheduler_step_size', None) or 10,
                gamma=getattr(self, 'lr_scheduler_gamma', None) or 0.1,
            )
            lr_scheduler_generative = {"scheduler": scheduler_generative, "interval": "epoch"}
            lr_scheduler_adversary = {"scheduler": scheduler_adversary, "interval": "epoch"}

        else:
            # Default: ReduceLROnPlateau
            scheduler_generative = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_generative,
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
            )
            scheduler_adversary = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_adversary,
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
            )
            lr_scheduler_generative = {
                "scheduler": scheduler_generative,
                "monitor": self.lr_monitor_key,
                "frequency": self.lr_scheduler_freq,
                "interval": self.lr_scheduler_interval,
            }
            lr_scheduler_adversary = {
                "scheduler": scheduler_adversary,
                "monitor": self.lr_monitor_key,
                "frequency": self.lr_scheduler_freq,
                "interval": self.lr_scheduler_interval,
            }
        # ======================================= [NEW] End unified scheduler selection

        return [
            {
                "optimizer": optimizer_generative,
                "lr_scheduler": lr_scheduler_generative,
            },
            {"optimizer": optimizer_adversary, "lr_scheduler": lr_scheduler_adversary},
        ]
