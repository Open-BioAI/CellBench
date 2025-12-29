import torch
class TransformBase:
    '''
    perturbench.data.transforms.Base
    '''
    def __init__(self,obs_df,mode):
        #assert obs_df
        #assert mode
        self.obs_df = obs_df
        self.mode=mode#train/eval

    def __call__(self, example):
        pass



