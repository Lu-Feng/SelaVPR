import torch
import torch.nn.functional as F
from local_matching import local_sim

class LocalFeatureLoss(torch.nn.Module):
    def __init__(self):
        super(LocalFeatureLoss,self).__init__()
        return
    def forward(self, feature_data):
        anchor, positive, negative = feature_data[0], feature_data[1], feature_data[2]
        simP = local_sim(anchor,positive,trainflag=True)
        simN = local_sim(anchor,negative,trainflag=True)
        loss = torch.sum(torch.clamp(-simP+simN+0., min=0.))
        return loss