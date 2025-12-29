import torch
from torch import nn
from .base import TransformBase
import os
class scDFMTransform(TransformBase):
    def __init__(self,
                 obs_df,
                 mode,
                 pert_key,
                 perturbation_combination_delimiter,
                 pert_map_path):
        super().__init__(obs_df, mode)

        self.pert_map_path = pert_map_path
        self.pert_key = pert_key
        self.perturbation_combination_delimiter = perturbation_combination_delimiter
        self.pert_map_path=pert_map_path

        if os.path.exists(self.pert_map_path):
            self.pert_map=torch.load(self.pert_map_path,weights_only=False)
        else:
            self.generate_multihot_map()

        self.check()


    def check(self):
        perts = []
        for pert in self.obs_df[self.pert_key].unique():
            perts.extend(pert.split(self.perturbation_combination_delimiter))
        perts = set(perts)

        assert len(perts-set(self.pert_map.keys()))==0,\
            f"These perts for model are not all in perts map:\n{' '.join(perts - set(self.pert_map.keys()))}"


    def generate_multihot_map(self):
        self.pert_map={}
        perts=[]
        for pert in self.obs_df[self.pert_key].unique():
            perts.extend(pert.split(self.perturbation_combination_delimiter))
        perts=list(set(perts))

        for idx,pert in enumerate(perts):
            self.pert_map[pert]=idx

    def __call__(self, input_dict):
        pert=input_dict[self.pert_key]
        pert_emb=[]

        for p in pert.split(self.perturbation_combination_delimiter):
            pert_emb.append(self.pert_map[p])

        pert_emb=torch.tensor(pert_emb)

        pert_cell_counts=input_dict['pert_cell_counts']
        control_cell_counts=input_dict['control_cell_counts']

        return {
            'pert_cell_counts':pert_cell_counts,
            'control_cell_counts':control_cell_counts,
            self.pert_key:pert_emb
        }

