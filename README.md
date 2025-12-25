# CellBench

We present a comprehensive framework, **CellBench**, for predicting the effects of perturbations in single cells. Designed to standardize benchmarking in this rapidly evolving field, it includes a user-friendly platform, diverse datasets, metrics for fair model comparison, and detailed performance analysis.

## Installation

Bash

```
conda create -n cellbench python=3.11
conda activate cellbench
git clone https://github.com/Open-BioAI/CellBench.git
cd CellBench
pip install -e .
```

------

## Project Structure

Plaintext

```
perturbench/
├── src/perturbench/
│   ├── configs/          # Hydra configuration files
│   │   ├── model/        # Model configurations (*.yaml)
│   │   ├── data/         # Dataset configurations
│   │   └── experiment/   # Experiment configurations
│   ├── modelcore/
│   │   ├── models/       # Model implementations
│   │   │   ├── base.py   # PerturbationModel base class
│   │   │   ├── latent_additive.py
│   │   │   ├── cpa.py
│   │   │   └── ...
│   │   └── train.py      # Training entry point
│   └── data/             # Data loading modules
└── data/                 # Directory for storing datasets
```

------

## Quick Start: Running Training

Bash

```
# Run using a predefined experiment configuration
train experiment=neurips2024/norman19/latent_best_params_norman19

# Override parameters via command line
train model=latent_additive data=srivatsan2020 trainer.max_epochs=50
```

------

## How to Add a New Model

### Workflow Flowchart

Plaintext

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Create Model Class                                     │
│  src/perturbench/modelcore/models/my_model.py                   │
│  Inherit PerturbationModel, implement forward() & training_step()│
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Register Model                                         │
│  src/perturbench/modelcore/models/__init__.py                   │
│  Add: from .my_model import MyModel                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Create Configuration File                              │
│  src/perturbench/configs/model/my_model.yaml                    │
│  Define _target_ and hyperparameters                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Run Experiment                                         │
│  train model=my_model data=srivatsan2020                        │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: Create the Model Class

Create `my_model.py` in `src/perturbench/modelcore/models/`:

Python

```
import torch
import torch.nn.functional as F
import lightning as L
from .base import PerturbationModel
from ..nn import MixedPerturbationEncoder

class MyModel(PerturbationModel):
    """Your model description here"""

    def __init__(
        self,
        datamodule: L.LightningDataModule | None = None,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        **kwargs
    ):
        # 1. Initialize parent class (datamodule is required)
        super().__init__(datamodule, lr=lr, **kwargs)

        # 2. Save hyperparameters (ignore datamodule object)
        self.save_hyperparameters(ignore=['datamodule'])

        # 3. Build network layers
        self.encoder = torch.nn.Linear(self.n_genes, hidden_dim)
        self.decoder = torch.nn.Linear(hidden_dim, self.n_genes)

        # 4. Use MixedPerturbationEncoder to handle perturbation embeddings
        self.pert_encoder = MixedPerturbationEncoder(
            gene_pert_dim=self.gene_pert_dim,
            drug_pert_dim=self.drug_pert_dim,
            env_pert_dim=self.env_pert_dim,
            final_embed_dim=hidden_dim
        )

    def forward(self, batch) -> torch.Tensor:
        """
        Forward Propagation

        Args:
            batch: A Batch object containing:
                - control_cell_counts: Control group expression [B, n_genes]
                - gene_pert / drug_pert / env_pert: Perturbation encodings
                - pert_cell_counts: Post-perturbation expression (for training)

        Returns:
            predicted_expression: Predicted post-perturbation expression [B, n_genes]
        """
        # Encode control cell state
        control_expr = batch.control_cell_counts
        latent = self.encoder(control_expr)

        # Encode perturbation effects
        pert_effect = self.pert_encoder(batch)

        # Additive model: latent state + perturbation effect
        perturbed_latent = latent + pert_effect

        # Decode to predict expression
        return self.decoder(perturbed_latent)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Training Step - Loss Calculation"""
        pred = self.forward(batch)
        target = batch["pert_cell_counts"]

        # Optional: Use a mask to calculate loss only on expressed genes
        mask = self._get_mask(batch)
        if mask is not None:
            loss = (F.mse_loss(pred, target, reduction='none') * mask).sum() / mask.sum()
        else:
            loss = F.mse_loss(pred, target)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        """Validation Step"""
        pred = self.forward(batch)
        target = batch["pert_cell_counts"]
        loss = F.mse_loss(pred, target)
        self.log("val_loss", loss, prog_bar=True)
```

### Step 2: Register the Model

Edit `src/perturbench/modelcore/models/__init__.py`:

Python

```
from .base import PerturbationModel
from .my_model import MyModel  # Add this line
# ... other models
```

### Step 3: Create the Configuration File

Create `src/perturbench/configs/model/my_model.yaml`:

YAML

```
_target_: perturbench.modelcore.models.MyModel
hidden_dim: 128
lr: 1e-3
wd: 1e-5
use_mask: true

# LR Scheduler configuration
lr_scheduler_mode: plateau
lr_scheduler_patience: 5
lr_scheduler_factor: 0.5
```

### Step 4: Run Training

Bash

```
train model=my_model data=srivatsan2020 trainer.max_epochs=100
```

------

## Key API Reference

### Core Attributes of `PerturbationModel` Base Class

| **Attribute**        | **Type** | **Description**                                |
| -------------------- | -------- | ---------------------------------------------- |
| `self.n_genes`       | int      | Number of genes                                |
| `self.gene_pert_dim` | int      | Dimension of gene perturbation encoding        |
| `self.drug_pert_dim` | int      | Dimension of drug perturbation encoding        |
| `self.env_pert_dim`  | int      | Dimension of environment perturbation encoding |
| `self.cov_keys`      | list     | List of covariate keys                         |
| `self.control_val`   | str      | Identifier value for the control group         |

### Batch Data Structure

Python

```
batch.control_cell_counts   # [B, n_genes] Control group expression
batch.pert_cell_counts      # [B, n_genes] Post-perturbation ground truth
batch.gene_pert             # [B, gene_pert_dim] Gene perturbation encoding
batch.drug_pert             # [B, drug_pert_dim] Drug perturbation encoding
batch.env_pert              # [B, env_pert_dim] Environment perturbation encoding
batch.pert_expression_mask  # [B, n_genes] Expressed gene mask (optional)
batch[cov_key]              # Covariates (e.g., cell_type)
```

------

## Configuration System (Hydra)

### Command Line Overrides

Bash

```
train model=my_model trainer.max_epochs=50 model.hidden_dim=256
```

### Creating Experiment Configurations

Create `src/perturbench/configs/experiment/my_exp.yaml`:

YAML

```
# @package _global_
defaults:
  - override /model: my_model
  - override /data: srivatsan2020

trainer:
  max_epochs: 100

model:
  hidden_dim: 256
  lr: 5e-4
```

Run: `train experiment=my_exp`

------

## Evaluation Metrics

The built-in evaluation pipelines run automatically. Configuration is located at `src/perturbench/configs/data/evaluation/`:

YAML

```
evaluation_pipeline:
  - aggregation: average   # average, logfc, logp, var
    metric: rmse           # rmse, mse, mae, cosine, pearson, r2_score
    rank: True
  - aggregation: logfc
    metric: cosine
```

------

## Existing Models

| **Model**                          | **Config File**        | **Description**                                                      |
| ---------------------------------- | ---------------------- | -------------------------------------------------------------------- |
| LatentAdditive                     | `latent_additive.yaml` | Additive model in latent space                                       |
| LinearAdditive                     | `linear_additive.yaml` | Additive model in linear (gene expression) space                     |
| DecoderOnly                        | `decoder_only.yaml`    | Decoder-only model without control cell encoding                     |
| CPA                                | `cpa.yaml`             | Compositional Perturbation Autoencoder with VAE                      |
| GEARS                              | `gears.yaml`           | Graph-Enhanced Gene Activation Regularized System with GO annotation |
| GenePert                           | `genepert.yaml`        | Perturbation model using GenePT gene embeddings                      |
| scLAMBDA                           | `sclambda.yaml`        | VAE with Mutual Information minimization and adversarial training    |
| Squidiff                           | `squidiff.yaml`        | Diffusion-based perturbation prediction model                        |
| SparseAdditiveVAE                  | `sams_vae.yaml`        | Sparse Additive Mechanism Shift VAE (NeurIPS 2024)                   |
| Biolord                            | `biolord.yaml`         | Biolord disentangled representation learning                         |
| PRNet                              | `prnet.yaml`           | Perturbation Response Network with uncertainty estimation            |
| StateTransitionPerturbationModel   | `state_sm.yaml`        | Transformer-based state transition model with LoRA support           |
| scDFM                              | `scDFM.yaml`           | Distributional Flow Matching for single-cell perturbation            |

