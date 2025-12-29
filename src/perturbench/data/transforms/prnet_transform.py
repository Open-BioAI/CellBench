import torch
from .base import TransformBase
from .onehot_transform import OneHotTransform

class PRNetTransform(OneHotTransform):
    def __init__(
            self,
            obs_df,
            mode,
            pert_key,
            cov_keys,
            pert_map_path,
            cov_maps_path,
            use_embedding_key=False,
            pert_comb_delim='+'
    ):
        super().__init__(obs_df,
                         mode,
                         pert_key,
                         cov_keys,
                         pert_map_path,
                         cov_maps_path,
                         use_embedding_key,
                         pert_comb_delim)

    def __call__(self,example):
        comb_pert = example[self.pert_key]
        pert_idx_list = []
        for pert in comb_pert.split(self.pert_comb_delim):
            pert_idx_list+= [self.pert_map[pert].argmax()]

        if self.use_embedding_key:
            controls = example['control_cell_emb']
        else:
            controls = example['control_cell_counts']

        pert_cell_counts = example['pert_cell_counts']

        cov_embs = {}
        for cov_key in self.cov_keys:
            cov_embs[cov_key] = self.cov_maps[cov_key][example[cov_key]]

        out = {
            'controls': controls,
            'control_cell_counts': example['control_cell_counts'],  # Keep original field name for model access
            'pert_cell_counts': pert_cell_counts,  # Fixed: use consistent field name
            self.pert_key: pert_idx_list,
            **cov_embs
        }

        # Pass through expression masks for masked loss calculation
        if 'pert_expression_mask' in example:
            out['pert_expression_mask'] = example['pert_expression_mask']
        if 'control_expression_mask' in example:
            out['control_expression_mask'] = example['control_expression_mask']

        return out
