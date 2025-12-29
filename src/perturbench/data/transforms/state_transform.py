from .base import TransformBase
import torch
import os


class StateTransform(TransformBase):
    def __init__(self, obs_df=None,mode=None,**kwargs):
        super().__init__(obs_df=obs_df,mode=mode)
        self.pert_map_path=kwargs.get('pert_map_path',None)
        self.pert_key=kwargs.get('pert_key')
        self.embedding_key=kwargs.get('embedding_key',None)
        self.perturbation_combination_delimiter=kwargs.get('perturbation_combination_delimiter','+')

        if self.pert_map_path:
            if os.path.exists(self.pert_map_path):
                self.pert_map=torch.load(self.pert_map_path,weights_only=False)
            elif self.mode=='train':
                self.generate_pert_map()
                torch.save(self.pert_map, self.pert_map_path)
        elif self.mode=='train':
            self.generate_pert_map()

        assert hasattr(self,'pert_map'),'pert_map_path does not exist'

        if self.mode=='eval':
            self.check_pert_map()


        self.pert_dim=len(self.pert_map.keys())

        self.obs_df=obs_df

    def __call__(self, input_dict):
        #可根据self.mode的情况分类进行

        pert_cell_counts=input_dict['pert_cell_counts']
        control_cell_counts=input_dict['control_cell_counts']

        if self.embedding_key:
            ctrl_cell_emb=input_dict['control_cell_emb']
            pert_cell_emb=input_dict['pert_cell_emb']
            out_dict={
                'ctrl_cell_emb':ctrl_cell_emb,
                'pert_cell_emb':pert_cell_emb,
                'pert_cell_counts':pert_cell_counts,
            }
        else:
            out_dict={
                'ctrl_cell_emb':control_cell_counts,
                'pert_cell_emb':pert_cell_counts,
            }


        pert_emb=self.pert_trans(input_dict[self.pert_key])
        out_dict['pert_emb']=pert_emb
        return out_dict

    def generate_pert_map(self):
        perts =[]
        for pert in self.obs_df[self.pert_key].unique():
            perts.extend(pert.split(self.perturbation_combination_delimiter))
        perts=list(set(perts))
        n_perts = len(perts)
        pert_to_idx={pert:idx for idx,pert in enumerate(perts)}
        self.pert_map= {}
        for pert in perts:
            emb=torch.zeros(n_perts,dtype=torch.float32)
            idx=pert_to_idx[pert]
            emb[idx]=1
            self.pert_map[pert]=emb

    def check_pert_map(self):
        perts=[]
        for pert in self.obs_df[self.pert_key].unique():
            perts.extend(pert.split(self.perturbation_combination_delimiter))
        perts = set(perts)
        assert len(set(perts)-set(self.pert_map.keys()))==0,'Some perts are not in pert_map.keys()'


    def pert_trans(self,pert):
        out=0
        for split_pert in pert.split(self.perturbation_combination_delimiter):
            out+=self.pert_map[split_pert]
        return out

