from .base import TransformBase
import torch
import numpy as np
class SCLambdaTransform(TransformBase):
    def __init__(self,obs_df,mode,
                 pert_key):
        super().__init__(obs_df,mode)
        self.pert_key = pert_key

    def __call__(self, example):
        gene_expression = example['pert_cell_counts']
        perturbation=example[self.pert_key]
        controls=example['control_cell_counts']

        out = {
            'gene_expression': gene_expression,
            'pert_cell_counts': gene_expression,  # Keep consistent field name for masked loss
            'perturbation': perturbation,
            'controls': controls,
            'control_cell_counts': controls,  # Keep consistent field name for masked loss
        }

        # Pass through expression masks for masked loss calculation
        if 'pert_expression_mask' in example:
            out['pert_expression_mask'] = example['pert_expression_mask']
        if 'control_expression_mask' in example:
            out['control_expression_mask'] = example['control_expression_mask']

        return out