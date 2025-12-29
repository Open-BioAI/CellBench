import scanpy as sc
import os
from scipy.sparse import csr_matrix

from perturbench.analysis.preprocess import preprocess
from perturbench.data.accessors.base import (
    download_scperturb_adata,
    download_file,
    Accessor,
)


class Norman19(Accessor):
    def __init__(self, data_cache_dir="../perturbench_data"):
        super().__init__(
            data_cache_dir=data_cache_dir,
            dataset_hf_url="https://huggingface.co/datasets/altoslabs/perturbench/resolve/main/norman19_preprocessed.h5ad.gz",
            dataset_orig_url="https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad?download=1",
            dataset_name="norman19",
        )

    def get_anndata(self):
        """
        Downloads, curates, and preprocesses the norman19 dataset from either
        Hugging Face or the scPerturb database. Saves the preprocessed data to
        disk and returns it in-memory.

        Returns:
            adata (anndata.AnnData): Anndata object containing the processed data.

        """
        self.processed_data_path = (
            f"{self.data_cache_dir}/{self.dataset_name}_processed.h5ad"
        )
        if os.path.exists(self.processed_data_path):
            print("Loading processed data from:", self.processed_data_path)
            adata = sc.read_h5ad(self.processed_data_path)

        else:
            try:
                hf_filename = f"{self.dataset_name}_processed.h5ad.gz"
                download_file(self.dataset_hf_url, self.data_cache_dir, hf_filename)
                adata = sc.read_h5ad(self.processed_data_path)
                
            except Exception as e:
                print(f"Error downloading file from {self.dataset_hf_url}: {e}")
                print(f"Downloading file from {self.dataset_orig_url}")

                adata = download_scperturb_adata(
                    self.dataset_orig_url,
                    self.data_cache_dir,
                    filename=f"{self.dataset_name}_downloaded.h5ad",
                )

                adata.obs.rename(
                    columns={
                        "nCount_RNA": "ncounts",
                        "nFeature_RNA": "ngenes",
                        "percent.mt": "percent_mito",
                        "cell_line": "cell_type",
                    },
                    inplace=True,
                )

                adata.obs["perturbation"] = adata.obs["perturbation"].str.replace("_", "+")
                adata.obs["perturbation"] = adata.obs["perturbation"].astype("category")
                adata.obs["condition"] = adata.obs.perturbation.copy()

                adata.X = csr_matrix(adata.X)

                adata = preprocess(
                    adata,
                    perturbation_key="condition",
                    covariate_keys=["cell_type"],
                )

                adata = adata.copy()
                adata.write_h5ad(self.processed_data_path)
            
            print("Saved processed data to:", self.processed_data_path)

        return adata
