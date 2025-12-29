import scanpy as sc
import os

from perturbench.data.accessors.base import (
    download_file,
    Accessor,
)


class McFaline23(Accessor):
    def __init__(self, data_cache_dir="../perturbench_data"):
        super().__init__(
            data_cache_dir=data_cache_dir,
            dataset_hf_url="https://huggingface.co/datasets/altoslabs/perturbench/blob/main/mcfaline23_gxe_preprocessed.h5ad.gz",
            dataset_orig_url=None,
            dataset_name="mcfaline23",
        )

    def get_anndata(self):
        """
        Downloads, curates, and preprocesses the McFalineFigueroa23 dataset from
        Hugging Face. Saves the preprocessed data to disk and returns it in-memory.

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
                raise ValueError(
                    "Automatic data curation not available for this dataset. \
                     Use the notebooks in notebooks/neurips2025/data_curation \
                     to download and preprocess the data."
                )
            
            print("Saved processed data to:", self.processed_data_path)

        return adata
