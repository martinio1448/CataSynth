import torch

class ExpandToRGB:
    def __init__(self):
        a = 1
        # print(self.displacement.shape)

    def __call__(self, sample: torch.tensor):
        if(len(sample.shape)==3):
            ret = sample.repeat_interleave(3, dim=0)
            return ret
        
        else:
            return sample.reeat_interleave(3, dim=1)