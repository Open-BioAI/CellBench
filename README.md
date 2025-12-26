# CellBench

A comprehensive framework for predicting perturbation effects in single cells, featuring standardized benchmarking, diverse datasets, and fair model comparison.

## Installation

```bash
conda create -n cellbench python=3.11
conda activate cellbench
git clone https://github.com/Open-BioAI/CellBench.git
cd CellBench
pip install -e .
```

## Project Structure

```
CellBench/
├── src/perturbench/
│   ├── configs/          # Hydra configuration files
│   ├── modelcore/
│   │   ├── models/       # Model implementations
│   │   └── train.py      # Training entry point
│   └── data/             # Data loading modules
├── scripts/              # Training shell scripts
└── data/                 # Datasets
```

## Quick Start

```bash
# Using shell scripts
bash scripts/biolord.sh
bash scripts/cpa.sh

# Using train command
train model=latent_additive data=srivatsan2020 trainer.max_epochs=50
```

## Adding a New Model

1. **Create Model Class** in `src/perturbench/modelcore/models/my_model.py`:

```python
from .base import PerturbationModel

class MyModel(PerturbationModel):
    def __init__(self, datamodule=None, hidden_dim=128, lr=1e-3, **kwargs):
        super().__init__(datamodule, lr=lr, **kwargs)
        self.save_hyperparameters(ignore=['datamodule'])
        # Build your network layers here

    def forward(self, batch):
        # Return predicted expression
        pass

    def training_step(self, batch, batch_idx):
        # Compute and return loss
        pass
```

2. **Register Model** in `src/perturbench/modelcore/models/__init__.py`:

```python
from .my_model import MyModel
```

3. **Create Config** at `src/perturbench/configs/model/my_model.yaml`:

```yaml
_target_: perturbench.modelcore.models.MyModel
hidden_dim: 128
lr: 1e-3
```

4. **Run Training**: `train model=my_model data=srivatsan2020`

## API Reference

### PerturbationModel Attributes

| Attribute | Description |
|-----------|-------------|
| `self.n_genes` | Number of genes |
| `self.gene_pert_dim` | Gene perturbation encoding dimension |
| `self.drug_pert_dim` | Drug perturbation encoding dimension |
| `self.env_pert_dim` | Environment perturbation encoding dimension |

### Batch Structure

```python
batch.control_cell_counts   # [B, n_genes] Control expression
batch.pert_cell_counts      # [B, n_genes] Perturbed expression
batch.gene_pert             # [B, gene_pert_dim] Gene perturbation
batch.drug_pert             # [B, drug_pert_dim] Drug perturbation
batch.env_pert              # [B, env_pert_dim] Environment perturbation
```

## Supported Models

| Model | Config | Description |
|-------|--------|-------------|
| LatentAdditive | `latent_additive.yaml` | Additive model in latent space |
| LinearAdditive | `linear_additive.yaml` | Additive model in expression space |
| CPA | `cpa.yaml` | Compositional Perturbation Autoencoder |
| GEARS | `gears.yaml` | Graph-Enhanced with GO annotation |
| GenePert | `genepert.yaml` | Using GenePT embeddings |
| scLAMBDA | `sclambda.yaml` | VAE with MI minimization |
| Squidiff | `squidiff.yaml` | Diffusion-based model |
| SparseAdditiveVAE | `sams_vae.yaml` | Sparse Additive Mechanism Shift VAE |
| Biolord | `biolord.yaml` | Disentangled representation learning |
| PRNet | `prnet.yaml` | With uncertainty estimation |
| scDFM | `scDFM.yaml` | Distributional Flow Matching |
