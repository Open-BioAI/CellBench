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

        # Add covariate support
        self.use_covs = kwargs.get('use_covs', False)
        self.cov_keys = kwargs.get('cov_keys', [])
        self.cov_maps = {}
        self.cov_dims = {}

        if self.use_covs and self.cov_keys and self.obs_df is not None:
            for cov_key in self.cov_keys:
                if cov_key not in self.obs_df.columns:
                    continue
                vals = self.obs_df[cov_key].dropna().unique().tolist()
                vals = [str(v) for v in vals]              # 防止 mixed type / numpy type
                vals = sorted(set(vals))
                dim = len(vals)
                self.cov_dims[cov_key] = dim

                cov_map = {}
                eye = torch.eye(dim, dtype=torch.float32)
                for i, v in enumerate(vals):
                    cov_map[v] = eye[i]
                self.cov_maps[cov_key] = cov_map

        if self.pert_map_path:
            if os.path.exists(self.pert_map_path):
                self.pert_map=torch.load(self.pert_map_path, map_location="cpu")
            elif self.mode=='train':
                self.generate_pert_map()
                torch.save(self.pert_map, self.pert_map_path)
        elif self.mode=='train':
            self.generate_pert_map()

        assert hasattr(self,'pert_map'),'pert_map_path does not exist'

        if self.mode=='eval':
            self.check_pert_map()

        self.pert_dim=len(self.pert_map.keys())

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

        # Add covariates if enabled
        if self.use_covs and self.cov_keys:
            cov_tensors = []
            for cov_key in self.cov_keys:
                dim = self.cov_dims.get(cov_key, 0)
                if dim == 0:
                    continue

                cov_val = input_dict.get(cov_key, None)
                cov_val = None if cov_val is None else str(cov_val)  # 统一成 str key

                cov_map = self.cov_maps.get(cov_key, {})
                if cov_val is not None and cov_val in cov_map:
                    cov_tensors.append(cov_map[cov_val])
                else:
                    cov_tensors.append(torch.zeros(dim, dtype=torch.float32))

            if cov_tensors:
                out_dict["covariates"] = torch.cat(cov_tensors, dim=-1)  # [cov_dim]

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

