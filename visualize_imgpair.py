import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

import parser
import os
import network
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

imgpath0 = "./image/img_pair/img0.jpg"
imgpath1 = "./image/img_pair/img1.jpg"

args = parser.parse_arguments()
t = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_patchfeature(model,imgpath):
    img = Image.open(imgpath)
    img = t(img).unsqueeze(0).to(args.device)
    feature = model.module.backbone(img)
    feature = feature["x_norm_patchtokens"]
    feature = torch.nn.functional.normalize(feature, p=2, dim=-1)
    return feature

def get_keypoints(img_size): 
    H,W = img_size
    patch_size = 14 #224/16
    N_h = H//patch_size
    N_w = W//patch_size
    keypoints = np.zeros((2, N_h*N_w), dtype=int) #(x,y)
    keypoints[0] = np.tile(np.linspace(patch_size//2, W-patch_size//2, N_w, 
                                       dtype=int), N_h)
    keypoints[1] = np.repeat(np.linspace(patch_size//2, H-patch_size//2, N_h,
                                         dtype=int), N_w)
    return np.transpose(keypoints)

def match_batch_tensor(fm1, fm2, img_size):
    '''
    fm1: (l,D)
    fm2: (N,l,D)
    mask1: (l)
    mask2: (N,l)
    '''
    M = torch.matmul(fm2, fm1.T) # (N,l,l)
    
    max1 = torch.argmax(M, dim=1) #(N,l)
    max2 = torch.argmax(M, dim=2) #(N,l)
    m = max2[torch.arange(M.shape[0]).reshape((-1,1)), max1] #(N, l)
    valid = torch.arange(M.shape[-1]).repeat((M.shape[0],1)).cuda() == m #(N, l) bool

    kps = get_keypoints(img_size)
    for i in range(fm2.shape[0]):
        idx1 = torch.nonzero(valid[i,:]).squeeze()
        idx2 = max1[i,:][idx1]
        assert idx1.shape==idx2.shape

        cv_im_one = cv2.resize(cv2.imread(imgpath0),(224,224))
        cv_im_two = cv2.resize(cv2.imread(imgpath1),(224,224))

        kps = get_keypoints(img_size)
        inlier_keypoints_one = kps[idx1.cpu().numpy()]
        inlier_keypoints_two = kps[idx2.cpu().numpy()]
        kp_all1 = []
        kp_all2 = []
        matches_all = []
        print("Number of matched point pairs:", len(inlier_keypoints_one))
        #for this_inlier_keypoints_one, this_inlier_keypoints_two in zip(inlier_keypoints_one, inlier_keypoints_two):
        for k in range(inlier_keypoints_one.shape[0]):
            kp_all1.append(cv2.KeyPoint(inlier_keypoints_one[k, 0].astype(float), inlier_keypoints_one[k, 1].astype(float), 1, -1, 0, 0, -1))
            kp_all2.append(cv2.KeyPoint(inlier_keypoints_two[k, 0].astype(float), inlier_keypoints_two[k, 1].astype(float), 1, -1, 0, 0, -1))
            matches_all.append(cv2.DMatch(k, k, 0))

        im_allpatch_matches = cv2.drawMatches(cv_im_one, kp_all1, cv_im_two, kp_all2,
                                            matches_all, None, matchColor=(0, 255, 0), flags=2)
        cv2.imwrite("patch_matches.jpg",im_allpatch_matches)

model = network.GeoLocalizationNet(args)
model = model.to(args.device)
model = torch.nn.DataParallel(model)
state_dict = torch.load(args.resume)["model_state_dict"]
model.load_state_dict(state_dict)

patch_feature0 = get_patchfeature(model,imgpath0)
patch_feature1 = get_patchfeature(model,imgpath1)

print("Size of patch tokens:",patch_feature1.shape[1:])
match_batch_tensor(patch_feature0[0], patch_feature1, img_size=(224,224))