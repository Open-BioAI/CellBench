import torch
import torch.nn as nn
from typing import Iterable, Optional, Sequence

a=torch.tensor([[1],[3],[2],[1]])
print(a[torch.tensor([0,1,0,0],dtype=torch.bool)])

class MLP(nn.Module):
    """Simple MLP block reused by modality encoders and the final fusion layer."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: Optional[Iterable[int]] = None,
            dropout: float = 0.0,
            activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims or [])
        dims = [input_dim] + hidden_dims + [output_dim]

        layers: Sequence[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PertAggregator(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            output_dim: int,
            hidden_dims: Optional[Iterable[int]] = None,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp = MLP(emb_dim, output_dim, hidden_dims, dropout)

    def forward(self,pert_batch) -> torch.Tensor:
        batch_size = len(pert_batch)
        pos_in_batch=[]
        stack_pert_emb=[]

        for idx,pert_emb_list in enumerate(pert_batch):
            for pert_emb in pert_emb_list:
                pos_in_batch.append(idx)
                stack_pert_emb.append(pert_emb)

        if len(stack_pert_emb) == 0:
            # 如果没有perturbation，返回零向量
            return torch.zeros(batch_size, self.mlp.net[-1].out_features, device=next(self.mlp.parameters()).device)

        stack_pert_emb=torch.stack(stack_pert_emb)
        pos_in_batch=torch.tensor(pos_in_batch,device=stack_pert_emb.device)
        stack_pert_emb = self.mlp(stack_pert_emb)

        agged_pert_emb=[]
        for idx in range(batch_size):  # 遍历batch size，而不是perturbation总数，以免出现如果有空字符串，多个“+”导致的错误
            agged_pert_emb.append(stack_pert_emb[pos_in_batch==idx].sum(dim=0))
        agged_pert_emb=torch.stack(agged_pert_emb)

        return agged_pert_emb

class MixedPerturbationEncoder(nn.Module):

    def __init__(
            self,
            gene_pert_dim: int,
            drug_pert_dim: int,
            env_pert_dim:int ,
            hidden_dims: Optional[Iterable[int]] = None,
            per_modality_embed_dim: int = 128,
            final_embed_dim: int = 128,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gene_encoder=MLP(gene_pert_dim,per_modality_embed_dim,hidden_dims,dropout=dropout)
        self.drug_encoder = PertAggregator(drug_pert_dim, per_modality_embed_dim, hidden_dims, dropout=dropout)
        self.env_encoder = MLP(env_pert_dim, per_modality_embed_dim, hidden_dims, dropout=dropout)
        self.per_modality_embed_dim = per_modality_embed_dim
        self.fusion_mlp = MLP(per_modality_embed_dim, final_embed_dim)

    def forward(self,batch: torch.Tensor,) -> torch.Tensor:
        gene_pert_emb = self.gene_encoder(batch.gene_pert)
        drug_pert_emb=self.drug_encoder(batch.drug_pert)
        env_pert_emb=self.env_encoder(batch.env_pert)

        final_pert_emb=gene_pert_emb+drug_pert_emb+env_pert_emb

        return self.fusion_mlp(final_pert_emb)  # (B, final_embed_dim)