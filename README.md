# PerturBench
We present a comprehensive framework, PerturBench for predicting the effects of perturbations in single cells, designed to standardize benchmarking in this rapidly evolving field. We include a user-friendly platform, diverse datasets, metrics for fair model comparison, and detailed performance analysis.

If you use PerturBench in your work, please consider citing [Wu, Wershof, Shmon, Nassar, Osinski, and Eksi et al, 2024](https://arxiv.org/abs/2408.10609):
```
@misc{wu2024perturbenchbenchmarkingmachinelearning,
      title={PerturBench: Benchmarking Machine Learning Models for Cellular Perturbation Analysis}, 
      author={Yan Wu and Esther Wershof and Sebastian M Schmon and Marcel Nassar and Błażej Osiński and Ridvan Eksi and Kun Zhang and Thore Graepel},
      year={2024},
      eprint={2408.10609},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.10609}, 
}
```

## Install PerturBench
```
conda create -n [env-name] python=3.11
conda activate [env-name]
cd [/path/to/PerturBench/]
pip3 install -e .
# or
pip3 install -e .[cli]
```
for command line extras such as the `rich` package, which gives you neater progress bars.

## Downloading and Preparing Datasets

### Dataset Access
For convenience, we've uploaded all processed datasets as gzipped h5ad files to Hugging Face at: https://huggingface.co/datasets/altoslabs/perturbench/tree/main.

We also provide accessor functions to automatically download and cache the Srivatsan20, Norman19, and Frangieh21 datasets as either [AnnData objects](https://anndata.readthedocs.io/en/latest/) or a [PyTorch Datasets](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
```
from perturbench.data.accessors.srivatsan20 import Sciplex3

srivatsan20_accessor = Sciplex3()
adata = srivatsan20_accessor.get_anndata() ## Get the preprocessed anndata object
torch_dataset = srivatsan20_accessor.get_dataset() ## Get a PyTorch Dataset
```

To reproduce the curation and preprocessing steps used to generate these datasets, first create a local cache directory (i.e. `~/perturbench_data`) and set the `data_cache_dir` variable in the curation notebooks in `notebooks/neurips2024/data_curation` to the cache you created. Please also set the `data_dir` variable in `src/configs/paths/default.yaml` to the correct data cache path as well.

Once you've set the correct local cache paths, please run the curation notebooks and scripts which will download the datasets, curate the metadata, and run standard scRNA-seq preprocessing with scanpy. Note that the McFalineFigueroa23 and Jiang24 data curation requires two steps as the downloaded files are Seurat objects and need to be converted to anndata h5ad files.

### Data Splitting
The Frangieh21 and Jiang24 datasets require manual splits that can be generated using the `notebooks/build_jiang24_frangieh21_splits.ipynb` notebook. For the McFalineFigueroa23 data scaling experiments and the Srivatsan20 imbalance experiments, you can generate the custom splits using the `notebooks/build_data_scaling_splits.ipynb` and `notebooks/build_imbalance_splits.ipynb` notebooks respectively.

## Usage

### Evaluator Class
If you just want to use our suite of metrics on your own custom model or generated predictions, we provide an `Evaluator` class. This class requires predicted scRNA-seq responses to perturbations as anndata objects, and will return a dataframe of average metrics for each model. More in-depth examples of using the `Evaluator` class can be found in the `notebooks/demos/evaluator_demo.ipynb` notebook.

### Hydra Training Script
To run end-to-end model training, inference, and evaluation, we provide a training script that is integrated with the Hydra configuration system. This script can be found under `src/perturbench/modelcore/train.py` and can be executed as follows:

```python
python <path-to-repo-folder>/src/perturbench/modelcore/train.py <config-options>
```
The configuration options are discussed in the Configuration System section. If the repo is installed using pip (setuptools) via `pip install`, the `train.py`` is added to the environment as an executable script. As a result the above command can be shortened and called anywhere as follows:
```python
train <config-options>
```

### Automated Evaluation
Model evaluation is built into the `src/perturbench/modelcore/train.py` script and by default will run automatically. Evaluation parameters are specified in the `src/perturbench/configs/data/evaluation/default.yaml` and the specific set of metrics used is controlled by the `evaluation_pipelines` parameter. 

To specify an evaluation pipeline, the user first needs to specify an aggregation (`aggregation`) method (`average`, `logfc`, `logp`, `var`) which generates an aggregate measure of expression/change in expression due to perturbation. The user also needs to specify an evaluation metric (`metric`) that compares observed vs predicted changes (`cosine`, `pearson`, `rmse`, `mse`, `mae`, `r2_score`). Finally, if the user wants this pipeline to also generate rank metrics, they need to set `rank: True` in the pipeline.

The default pipeline is:
```
evaluation_pipeline:
  - aggregation: average
    metric: rmse
    rank: True
  - aggregation: logfc
    metric: cosine
```

To add another pipeline, simply add another list element. For example to add `logp` aggregation which uses log-pvalues (similar to the NeurIPS competition):
```
evaluation_pipeline:
  - aggregation: logp
    metric: cosine
```

To run evaluation on a pre-trained model, the user can simply set `train: False` in the main `train.yaml` config and specify a path to the model checkpoint to use in the `ckpt_path` parameter. An example experiment config that runs evaluation only is at `src/perturbench/configs/experiment/evaluation_only_example.yaml`.

### Prediction
To generate predictions using a pre-trained model, we'll need:
- A trained model checkpoint
- A path to a dataset to use for inference (only control cells will be used)
- A csv file containing the desired counterfactual perturbations to predict and relevant covariates. An example of how to generate this csv file is at `notebooks/demos/generate_prediction_dataframe.ipynb`

To generate predictions:

```python
predict <config-options>
```

The configuration options are controlled by the default `src/perturbench/configs/predict.yaml` config file, and further discussed in the Configuration System section.

### Configuration System

This repo uses Hydra for configuration system management. A configuration setup and default settings are stored in `src/perturbench/configs`. This folders and the contained files should not be modified unless there are updates to the model codebase (such as adding new models, datasets, loggers, ...).

Below we describe potential workflows to use the configuration system to scale your experimentation:

1) Override any configuration parameter from the commandline
```python
train trainer.max_epoch=100 model=gene_sampling model.ngenes=10000 model.nsamples=5
```
This command overrides the `cfg.trainer.max_epoch` value and sets it to `100`. After that, it overrides the `cfg.model` to be the `gene_sampling` model and sets its parameters `ngenes` and `nsamples` to `10000` and `5`, respectively. 

2) Add any additional parameters that were not defined in the configuration system
```python
python train.py +trainer.gradient_clip_val=0.5
``` 
This will add an attribute field `gradient_clip_val` to the trainer.

3) A differentially written experiment configuration that overrides/augments the default configuration, for example,
```python
train experiment=example
```
where `experiment/example.yaml` contains the following: 
```yaml
# @package _global_

defaults:
  - override /data: mnist
  - override /model: mnist
  - override /callbacks: default
  - override /trainer: default

seed: 12345

trainer:
  max_epochs: 10
  gradient_clip_val: 0.5

data:
  batch_size: 64

logger:
  aim:
    experiment: "mnist"

```
This files uses the global configuration setup but overrides the `data`, `model`, `callbacks`, and `trainer`. After that it further sets the values of `seed`, `trainer.max_epochs`, `trainer.gradient_clip_val`,` data.batch_size`, and `logger.aim.experiment`.

 
4) Define your local experimental configuration

In many cases, it might not be desirable to work on the configuration that is part of the library. As a result, it would be desirable to have a local configuration that the user can modify as they debug/develop their model. 

Assume the user would like to use a local configuration directory `<local-path>/my_configs` to set his configurations. One way is to setup an experiment configuration based on the `experiment` configuration schema described in the previous point. Thus, the used will create the following files `<local-path>/my_configs/experiment/my_experiment.yaml`. Then, the user can execute his experiment as follows:
```python
train -cd <local-path>/my_configs experiment=my_experiment
```

_**Note**_: Any part of the configuration can be replicated in the local directory and would be augment to the library configuration.


### Hyperparameter Optimization with Ray

You can run HPO trials in parallel on an instance with multiple GPUs. This is enabled by Hydra's Optuna sweeper plugin and Ray launcher plugin. 

An example of additional config for the plugins can be found at `src/perturbench/configs/hpo/local.yaml`

Example command to run HPO on a single instance with multiple GPUs:

```
CUDA_VISIBLE_DEVICES=0,1 train hpo=latent_additive_hpo experiment=neurips2024/norman19/latent_best_params_norman19
```

### Reproducing arXiv results
To reproduce the results from our ArXiv preprint, we provide best-params configs for each model and dataset. For example, to reproduce the linear model results for the Norman19 dataset, you can run:
```
train experiment=neurips2024/norman19/linear_best_params_norman19
```

## Model development requirements

When creating a new model, you'll need to:
1. Subclass the base `PerturbationModel` class which implements inference methods 
```
from .base import PerturbationModel

class MyModel(PerturbationModel):
```

2. Pass the datamodule when initializing the superclass which enables the transforms and other key training information to be saved with the model checkpoint. You also need to save hyperparameters used to initialize the model (excluding the datamodule) so that the model can be easily instantiated for inference.
```
def __init__(
  ...,
  datamodule: L.LightningDataModule | None=None,
):
    super(LatentAdditive, self).__init__(datamodule)
    self.save_hyperparameters(ignore=['datamodule'])
```

3. Define a `predict` method that takes in a batch of data and outputs a counterfactual prediction
```
def predict(self, batch):
    control_expression = batch.gene_expression.squeeze()
    perturbation = batch.perturbations.squeeze()
    covariates = {
        k:v.to(self.device) for k,v in batch.covariates.items()
    }
    
    predicted_perturbed_expression = self.forward(
        control_expression, 
        perturbation,
        covariates,
    )
    return predicted_perturbed_expression
```

## Adding a new dataset
This section describes how to add a new dataset to benchmark against.

### Data curation
First download the dataset from GEO, figshare, or the desired database. [scPerturb](http://projects.sanderlab.org/scperturb/) and [pertpy](https://pertpy.readthedocs.io/en/latest/usage/usage.html#datasets) provide indexes of perturbational datasets with single cell readouts that might be of interest.

If the dataset is not stored as an [anndata](https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html) file, you will need to convert it to an anndata file. It also may help to clean up some of the metadata columns. Example scripts of converting Seurat objects to anndata and metadata curation notebooks can be found at `notebooks/curation`.

### Data preprocessing
Most downloaded datasets will contain raw counts, which will need to be processed before model training. We provide a default preprocessing pipeline that applies standard log-normalization and filters for highly variable or differentially expressed genes. Specifically the counts for each cell are divided by the total counts for that cell, multiplied by a scaling factor (`1e4`), and then log-transformed. The dataset is then subset to the top 4000 highly variable genes and top 50 differentially expressed genes per perturbation (computed on a per covariate basis). If the perturbations are genetic, those genes are also included in the expression matrix by default. Datasets ending in `_preprocessed.h5ad` have been preprocessed.

The unnormalized raw counts can be accessed in the `adata.layers['counts']` slot. To use raw counts instead of log-normalized expression add
```
data:
  use_counts: True
```
to your experiment config.

To preprocess a new dataset, use the `preprocess` function in `src/analysis/preprocess.py`.

### Data config
Once the dataset is preprocessed, you will need to create a dataset config file where you will specify which metadata columns contain the perturbations and covariates, as well as dataloader parameters. Example configs can be found at `src/configs/data`. You will also specifically need to specify how you want to split the data. You can select from a predefined split in the `src/configs/data/splitter` directory such as cross cell type or combination prediction splits. You can also specify a custom split saved as a `csv`. The data config is also where you specify the evaluation parameters, such as which metrics you want to evaluate. Configs that specify those parameters can be found in `src/configs/data/evaluation`.
