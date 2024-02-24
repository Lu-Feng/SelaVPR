
import torch
import random
from glob import glob

import torch.nn.functional as F
import datetime
import numpy as np
import cv2
import math

def get_keypoints(img_size):
    # flaten by x 
    H,W = img_size
    patch_size = 1#14
    N_h = H//patch_size
    N_w = W//patch_size
    keypoints = np.zeros((2, N_h*N_w), dtype=int)
    keypoints[0] = np.tile(np.linspace(patch_size//2, W-patch_size//2, N_w, 
                                       dtype=int), N_h)
    keypoints[1] = np.repeat(np.linspace(patch_size//2, H-patch_size//2, N_h,
                                         dtype=int), N_w)
    return np.transpose(keypoints)

def match_batch_tensor(fm1, fm2, trainflag, grid_size):
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
    
    scores = torch.zeros(fm2.shape[0]).cuda()

    kps = get_keypoints(grid_size)
    for i in range(fm2.shape[0]):
        idx1 = torch.nonzero(valid[i,:]).squeeze()
        idx2 = max1[i,:][idx1]
        assert idx1.shape==idx2.shape

        if trainflag:
            if len(idx1.shape)>0:      
                similarity = torch.mean(torch.sum(fm1[idx1] * fm2[i][idx2],dim=1),dim=0)
            else:
                print("No mutual nearest neighbors!")
                similarity = torch.mean(torch.sum(fm1 * fm2[i],dim=1),dim=0)
            return similarity
        
        else:
            if len(idx1.shape)<1:
                scores[i] = 0
            else:
                scores[i] = len(idx1)
    return scores

def local_sim(features_1, features_2, trainflag=False):
    B, H, W, C = features_2.shape
    if trainflag:
        queries = features_1
        preds = features_2
        queries,preds = queries.view(B, H*W, C),preds.view(B, H*W, C)
        similarity = torch.zeros(B).cuda()
        for i in range(B):
            query,pred = queries[i],preds[i].unsqueeze(0)
            similarity[i] = match_batch_tensor(query, pred, trainflag, grid_size=(61,61))
        return similarity
    else:
        query = features_1
        preds = features_2
        query,preds = query.view(H*W, C),preds.view(B, H*W, C)
        scores = match_batch_tensor(query, preds,trainflag, grid_size=(61,61))
        return scores