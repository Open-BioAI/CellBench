import pickle
import torch
from .base import TransformBase
import os

class GenePertTransform(TransformBase):
    def __init__(self,
                 obs_df,
                 mode,
                 pert_key,
                 pert_emb_path,
                 delimiter
                 ):
        super().__init__(obs_df,mode)
        self.obs_df=obs_df
        self.mode=mode
        self.pert_key=pert_key
        self.pert_emb_path=pert_emb_path
        self.delimiter=delimiter

        assert os.path.exists(self.pert_emb_path)
        with open(pert_emb_path, "rb") as f:
            self.pert_emb_dict = pickle.load(f)

        self.embedding_dim = len(next(iter(self.pert_emb_dict.values())))
        self.pert_encoder=GeneEmbeddingProcessor(embeddings=self.pert_emb_dict,
                                                 embedding_dim=self.embedding_dim)

    def __call__(self,example_data):
        control_cell_counts=example_data['control_cell_counts']
        pert_cell_counts=example_data["pert_cell_counts"]
        pert=example_data[self.pert_key]
        pert_emb=self.pert_encoder.process_perturbation(pert,self.delimiter,normalize=True)

        out = {
            "control_cell_counts":control_cell_counts,
            "pert_cell_counts":pert_cell_counts,
            "pert_emb":pert_emb,
        }

        # Pass through expression masks for masked loss calculation
        if 'pert_expression_mask' in example_data:
            out['pert_expression_mask'] = example_data['pert_expression_mask']
        if 'control_expression_mask' in example_data:
            out['control_expression_mask'] = example_data['control_expression_mask']

        return out







class GeneEmbeddingProcessor:
    """
    Processor for gene embeddings in GenePert.

    This class handles the loading and processing of gene embeddings,
    including normalization and combination for multi-gene perturbations.

    Args:
        embeddings: Dictionary mapping gene names to embedding vectors
        embedding_dim: Dimension of the embeddings
    """

    def __init__(self, embeddings: dict, embedding_dim: int):
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim

    def get_embedding(self, gene_name: str, normalize: bool = True) -> torch.Tensor:
        """
        Get embedding for a single gene.

        Args:
            gene_name: Name of the gene
            normalize: Whether to L2-normalize the embedding

        Returns:
            Gene embedding tensor
        """
        if gene_name in self.embeddings:
            emb = torch.tensor(self.embeddings[gene_name], dtype=torch.float32)
            if normalize:
                emb = emb / torch.norm(emb)
            return emb
        else:
            # Return random normalized vector for unknown genes
            emb = torch.randn(self.embedding_dim)
            if normalize:
                emb = emb / torch.norm(emb)
            return emb

    def combine_embeddings(
            self,
            gene_names: list[str],
            method: str = 'sum',
            normalize: bool = True
    ) -> torch.Tensor:
        """
        Combine embeddings for multiple genes (for combination perturbations).

        Args:
            gene_names: List of gene names to combine
            method: Combination method ('sum' or 'mean')
            normalize: Whether to normalize the combined embedding

        Returns:
            Combined embedding tensor
        """
        embeddings = [self.get_embedding(gene, normalize=False) for gene in gene_names]

        if method == 'sum':
            combined = torch.sum(torch.stack(embeddings), dim=0)
        elif method == 'mean':
            combined = torch.mean(torch.stack(embeddings), dim=0)
        else:
            raise ValueError(f"Unknown combination method: {method}")

        if normalize:
            combined = combined / torch.norm(combined)

        return combined

    def process_perturbation(
            self,
            pert,
            delimiter: str = '+',
            normalize: bool = True
    ) -> torch.Tensor:
        """
        Process a batch of perturbations into embeddings.

        Args:
            perturbation_list: List of perturbations, each as a list of gene names
            delimiter: Delimiter for multi-gene perturbations (not used, kept for compatibility)
            normalize: Whether to normalize embeddings

        Returns:
            Batch of perturbation embeddings of shape (batch_size, embedding_dim)
        """
        pert=pert.split(delimiter)

        if len(pert) == 0:  # Control condition
            # Return zero embedding for control
            emb = torch.zeros(self.embedding_dim)
        elif len(pert) == 1:  # Single gene perturbation
            emb = self.get_embedding(pert[0], normalize=normalize)
        else:  # Multi-gene perturbation
            emb = self.combine_embeddings(pert, method='sum', normalize=normalize)

        return emb

