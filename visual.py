import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

import imp
import os
import sys
import torch
from torchvision import transforms
import parser
import commons
from network import GeoLocalizationNet
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

######################################### SETUP #########################################
args = parser.parse_arguments()
commons.make_deterministic(args.seed)


t = transforms.Compose([transforms.Resize((224, 224)), #128, 128
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def show_feature_map(imgpath, conv_features):
    img = Image.open(imgpath).convert('RGB')
    heat = conv_features.squeeze(0)
    heat_mean = torch.mean(heat,dim=0)
    heatmap = heat_mean.detach().cpu().numpy()
    print(heatmap)
    heatmap = -heatmap  # Take the opposite number if necessary
    heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap)) 
    heatmap = cv2.resize(heatmap,(img.size[0],img.size[1]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    cv2.imwrite('heatmap_original.jpg',heatmap)

    plt.imshow(heatmap)
    plt.show()

    superimg = heatmap*0.6+np.array(img)[:,:,::-1] # Image overlay. Notice flipping the channel, opencv uses BGR
    cv2.imwrite('heatmap_result.jpg',superimg)

imgpath = "./image/img_pair/img0.jpg" # Path to the image you want to visualize
img = Image.open(imgpath)
img = t(img).unsqueeze(0).to(args.device)

##### Load trained model and extract feature map
model = GeoLocalizationNet(args)
model = model.to(args.device)
model = torch.nn.DataParallel(model)
if args.resume != None:
    state_dict = torch.load(args.resume)["model_state_dict"]
    model.load_state_dict(state_dict)
feature = model.module.backbone(img)

##### Load pre-trained DINOv2 and extract feature map
# from model.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
# model = vit_large(patch_size=14,img_size=518,init_values=1,block_chunks=0)
# state_dict = torch.load("/home/lufeng/data/VPR/dinov2_vitl14_pretrain.pth") # Path to pre-trained DINOv2
# model.load_state_dict(state_dict)
# model = model.to(args.device)
# model = torch.nn.DataParallel(model)
# feature = model(img)

#print(feature["x_norm_patchtokens"].shape)
feature = feature["x_norm_patchtokens"].view(-1,16,16,1024).permute(0, 3, 1, 2)
#print(feature.shape)

show_feature_map(imgpath,feature)