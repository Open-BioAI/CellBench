## Evaluation wrapper class
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import os

from hydra import initialize_config_module, compose
from hydra.core.hydra_config import HydraConfig

from perturbench.data.accessors.srivatsan20 import Sciplex3
from perturbench.data.accessors.norman19 import Norman19
from perturbench.data.accessors.frangieh21 import Frangieh21
from .evaluation import Evaluation
import perturbench.data.datasplitter as datasplitter


class Evaluator:
    """A class for benchmarking model predictions on a specific task."""

    @staticmethod
    def list_tasks():
        """List the tasks that the Evaluator class can evaluate models on."""
        return [
            "srivatsan20-transfer",
            "mcfaline23-transfer",
            "norman19-combo",
            "frangieh21-transfer",
            "jiang24-transfer",
        ]

    @staticmethod
    def get_task_data(
        task: str,
        local_data_cache: str = "perturbench_data",
    ):
        """Pulls the anndata object for a specific task into the local cache."""
        if task not in Evaluator.list_tasks():
            raise ValueError(f"Task {task} is not supported.")

        if not os.path.exists(local_data_cache):
            os.makedirs(local_data_cache)

        if task == "srivatsan20-transfer":
            adata = Sciplex3(data_cache_dir=local_data_cache).get_anndata()

        elif task == "norman19-combo":
            adata = Norman19(data_cache_dir=local_data_cache).get_anndata()

        elif task == "frangieh21-transfer":
            adata = Frangieh21(data_cache_dir=local_data_cache).get_anndata()

        elif task == "mcfaline23-transfer":
            local_data_path = f"{local_data_cache}/mcfaline23_gxe_processed.h5ad"
            try:
                adata = sc.read_h5ad(local_data_path, backed="r")
            except FileNotFoundError:
                raise NotImplementedError(
                    "Automatic McFaline23 dataset access not yet supported. Please run the \
                    notebooks in the notebooks/neurips2024/data_curation/ \
                        directory first to preprocess the data."
                )

        elif task == "jiang24-transfer":
            local_data_path = f"{local_data_cache}/jiang24_processed.h5ad"
            try:
                adata = sc.read_h5ad(local_data_path, backed="r")
            except FileNotFoundError:
                raise NotImplementedError(
                    "Automatic Jiang24 dataset access not yet supported. Please run the notebooks in \
                    the notebooks/neurips2024/data_curation/ directory first to \
                        preprocess the data."
                )

        return adata

    @staticmethod
    def get_task_config(task: str):
        """Returns the metadata columns for a specific task."""

        if task == "srivatsan20-transfer":
            data_override = ["data=sciplex3"]
        elif task == "norman19-combo":
            data_override = ["data=norman19"]
        elif task == "frangieh21-transfer":
            data_override = ["data=frangieh21"]
        elif task == "mcfaline23-transfer":
            data_override = ["data=mcfaline23"]
        elif task == "jiang24-transfer":
            data_override = ["data=jiang24"]
        else:
            raise ValueError(f"Task {task} is not supported.")

        with initialize_config_module(
            version_base="1.3", config_module="perturbench.configs"
        ):
            cfg = compose(
                config_name="train",
                overrides=data_override + ["data.splitter.save=False"],
                return_hydra_config=True,
            )
            HydraConfig.instance().set_config(cfg)

        return cfg.data

    def __init__(
        self,
        task: str,
        split_value_to_evaluate: str | None = "val",
        local_data_cache: str = "perturbench_data",
    ):
        """The constructor for the Evaluation class.

        Args:
            task: The task that the model is being evaluated on. Must be one of
              "srivatsan20-transfer", "norman19-combo", "mcfaline23-transfer".
            local_data_cache: The local directory where the task data is stored.
        """
        if task not in Evaluator.list_tasks():
            raise ValueError(f"Task {task} is not supported.")

        # Load observed anndata object
        ref_adata = Evaluator.get_task_data(task, local_data_cache)
        task_config = Evaluator.get_task_config(task)

        if split_value_to_evaluate is not None:
            split_dict = datasplitter.PerturbationDataSplitter.split_dataset(
                splitter_config=task_config.splitter,
                obs_dataframe=ref_adata.obs,
                perturbation_key=task_config.perturbation_key,
                perturbation_combination_delimiter=task_config.perturbation_combination_delimiter,
                perturbation_control_value=task_config.perturbation_control_value,
            )
            self.split_dict = split_dict
            ref_adata = ref_adata[split_dict[split_value_to_evaluate]].to_memory()

        else:
            ref_adata = ref_adata.to_memory()

        self.ref_adata = ref_adata
        self.task_config = task_config

    def get_split(self):
        return self.split_dict

    def evaluate(
        self,
        model_predictions: dict[str, ad.AnnData],
        return_metrics_dataframe: bool = False,
        print_metrics: bool = False,
    ):
        """Evaluates the model predictions on the task.

        Args:
            model_predictions: A dictionary mapping model names to their predictions
              as anndata objects.
            return_metrics_dataframe: Whether to return the summarized metrics
              as a pandas dataframe.
            print_metrics: Whether to print the summarized metrics to the console.
        """

        ev = Evaluation(
            model_adatas=model_predictions,
            ref_adata=self.ref_adata,
            pert_col=self.task_config["perturbation_key"],
            cov_cols=self.task_config["covariate_keys"],
            ctrl=self.task_config["perturbation_control_value"],
        )

        summary_metrics_dict = {}
        for aggr in ["average", "logfc"]:
            ev.aggregate(aggr_method=aggr)

            if aggr == "average":
                metric = "rmse"
            elif aggr == "logfc":
                metric = "cosine"

            ev.evaluate(aggr_method=aggr, metric=metric)
            ev.evaluate_pairwise(aggr_method=aggr, metric=metric)
            ev.evaluate_rank(aggr_method=aggr, metric=metric)

            df = ev.evals[aggr][metric].copy()
            avg = df.groupby("model").mean("metric")
            summary_metrics_dict[metric + "_" + aggr] = avg["metric"]

            rank_df = ev.rank_evals[aggr][metric].copy()
            avg_rank = rank_df.groupby("model").mean("rank")
            summary_metrics_dict[metric + "_rank_" + aggr] = avg_rank["rank"]

        summary_metrics = pd.DataFrame(summary_metrics_dict).T.applymap(
            lambda x: float(
                np.format_float_positional(
                    x, precision=4, unique=False, fractional=False, trim="k"
                )
            ),
        )
        self.summary_metrics = summary_metrics
        self.ev = ev

        if print_metrics:
            print(summary_metrics)

        if return_metrics_dataframe:
            return summary_metrics
