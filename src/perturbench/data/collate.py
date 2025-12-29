import collections
import pandas as pd
import torch
import numpy as np


class noop_collate:
    """No operation collate function. Returns the batch as is."""

    def __call__(self, batch: list):
        if len(batch) == 1:
            return batch[0]
        else:
            return batch

class inference_collate:

    def __call__(self, data_list: list):
        #[(example_dict,obs),(example_dict,obs),.............]
        keys=data_list[0][0].keys()
        batch_dict=collections.defaultdict(list)
        batch_obs=[]

        for example_dict,obs in data_list:
            for key in keys:
                val=example_dict[key]
                if isinstance(val, np.ndarray):
                    batch_dict[key].append(val)
                elif isinstance(val, torch.Tensor):
                    batch_dict[key].append(val)
                elif isinstance(val, int):
                    batch_dict[key].append(torch.tensor(val))
                elif isinstance(val, float):
                    batch_dict[key].append(torch.tensor(val))
                elif isinstance(val, str):
                    batch_dict[key].append(np.array(val))
                else:
                    batch_dict[key].append(val)
            batch_obs.append(obs)

        batch_obs=pd.concat(batch_obs)
        for key in keys:
            _type=type(batch_dict[key][0])
            if _type is torch.Tensor:
                batch_dict[key]=torch.stack(batch_dict[key])
            elif _type is np.ndarray:
                batch_dict[key]=np.stack(batch_dict[key])

        return batch_dict,batch_obs


class train_collate:
    def __call__(self, examples: list):
        #[example_dict1,example_dict2,example_dict3.........]
        keys=examples[0].keys()
        batch_dict=collections.defaultdict(list)

        for example_dict in examples:
            for key in keys:
                val = example_dict[key]
                if isinstance(val, np.ndarray):
                    batch_dict[key].append(val)
                elif isinstance(val, torch.Tensor):
                    batch_dict[key].append(val)
                elif isinstance(val, int):
                    batch_dict[key].append(torch.tensor(val))
                elif isinstance(val, float):
                    batch_dict[key].append(torch.tensor(val))
                elif isinstance(val, str):
                    batch_dict[key].append(np.array(val))
                else:
                    batch_dict[key].append(val)

        for key in keys:
            _type = type(batch_dict[key][0])
            if _type is torch.Tensor:
                batch_dict[key] = torch.stack(batch_dict[key])
            elif _type is np.ndarray:
                batch_dict[key] = np.stack(batch_dict[key])
            elif _type is list:
                # 对于列表类型（如drug_pert），保持为列表
                batch_dict[key] = batch_dict[key]

        return batch_dict

