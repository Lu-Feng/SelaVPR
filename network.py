import torch
import logging
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Flatten(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
    
class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.aggregation = nn.Sequential(L2Norm(), GeM(work_with_tokens=None), Flatten())
        self.upconv = torch.nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.upconv2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.backbone(x)
        patch_feature = x["x_norm_patchtokens"].view(-1,16,16,1024)

        x1 = patch_feature.permute(0, 3, 1, 2)
        x1 = self.aggregation(x1) 
        global_feature = torch.nn.functional.normalize(x1, p=2, dim=-1)

        x0 = patch_feature.permute(0, 3, 1, 2)
        x0 = self.upconv(x0)
        x0 = self.relu(x0)
        x0 = self.upconv2(x0)
        x0 = x0.permute(0, 2, 3, 1)
        local_feature = torch.nn.functional.normalize(x0, p=2, dim=-1)
        return local_feature, global_feature


def get_backbone(args):
    backbone = vit_large(patch_size=14,img_size=518,init_values=1,block_chunks=0) 
    assert not (args.foundation_model_path is None and args.resume is None), "Please specify foundation model path."
    if args.foundation_model_path:
        model_dict = backbone.state_dict()
        state_dict = torch.load(args.foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    args.features_dim = 1024
    return backbone

