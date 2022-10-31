import json
import math
from random import randrange
from turtle import forward
import numpy as np
from numpy.core.numeric import outer
import pandas as pd
import torch
import os
import sys
import shutil
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
from network.vit_16_base_feat_middle_gal_v2 import vit_base_patch16_224
import pdb


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

def calculate_threshold(probs, labels, threshold):
    TN, FN, FP, TP = eval_state(probs, labels, threshold)
    ACC = (TP + TN) / labels.shape[0]
    return ACC
		
def fit_inv_covariance(samples):
    return torch.Tensor(LedoitWolf().fit(samples.cpu()).precision_).to(
        samples.device
    )

def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched mahalanobis distance.

        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert inv_covariance.dim() == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
        return dist.sqrt()


import pdb

class Attention_Correlation_weight_reshape_loss(nn.Module):
    def __init__(self, c_out, c_in, c_cross) -> None:
        super(Attention_Correlation_weight_reshape_loss, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.c_cross = c_cross
        self.device = c_out.device
    def forward(self, correlation_map_real, correlation_map_fake, fake_weight):
        if correlation_map_real.shape[0] == 0:
            loss_real = 0
        else:
            B, PP, PP = correlation_map_real.shape
            C_matrix = torch.eye(PP).to(self.device) * (1-self.c_out) + self.c_out*torch.ones(PP).to(self.device)
            C_matrix = C_matrix.expand(B, -1, -1)
            loss_real = torch.sum(torch.abs(correlation_map_real - C_matrix))/(B*(PP*PP-PP))

        if correlation_map_fake.shape[0] == 0:
            loss_fake = 0
        else:
            B, PP, PP = correlation_map_fake.shape
            C_matrix_fake_all = []
            for b in range(B):
                fake_index = torch.where(fake_weight[b].reshape(-1)>0)[0]
                real_index = torch.where(fake_weight[b].reshape(-1)<=0)[0]
                C_matrix_fake = torch.zeros((PP, PP)).to(self.device) + self.c_cross
                for i in fake_index:
                    C_matrix_fake[i, fake_index] = self.c_in
                for j in real_index:
                    C_matrix_fake[j, real_index] = self.c_out
                C_matrix_fake_all.append(C_matrix_fake)
            C_matrix_fake_all = torch.stack(C_matrix_fake_all).to(self.device)
            # C_matrix_fake = C_matrix_fake.expand(B, -1, -1).cuda()
            loss_fake = torch.sum(torch.abs(correlation_map_fake - C_matrix_fake_all))/(B*(PP*PP-PP))

        loss = loss_fake + loss_real
        return loss


def estimate_MVG(feat_tensorlist_real, feat_tensorlist_fake):
    ## for compute MVG parameter
    feat_tensorlist = torch.cat(feat_tensorlist_real, dim=0).cuda()
    inv_covariance_real = fit_inv_covariance(feat_tensorlist_real).cpu()
    mean_real = feat_tensorlist.mean(dim=0).cpu()  # mean features.

    feat_tensorlist_fake = torch.cat(feat_tensorlist_fake, dim=0).cuda()
    inv_covariance_fake = fit_inv_covariance(feat_tensorlist_fake).cpu()
    mean_fake = feat_tensorlist_fake.mean(dim=0).cpu()  # mean features.

    return mean_real, mean_fake, inv_covariance_real, inv_covariance_fake


if __name__ == "__main__":
    ## 
    model = vit_base_patch16_224(pretrained=True, num_classes=2)
    model.train()
    ## create your own dataloader
    train_iter = iter(dataloader_train)
    img_data, img_label = train_iter.next()
    img_label = img_label.numpy().astype(np.float)

    [c_cross, c_in, c_out] = [nn.Parameter(torch.tensor(float(i)).cuda()) for i in opt.c_initilize.split()] 
    attention_loss = Attention_Correlation_weight_reshape_loss(c_out=c_out, c_in=c_in, c_cross=c_cross)
    classes, feat_patch, attn_map = model(img_data)
    B, H, W, C = feat_patch.size()

    ## acquire the mean attention map from multiple layers of the network
    realindex = np.where(img_label==0.0)[0]
    attn_map_real = torch.sigmoid(torch.mean(attn_map[realindex,:, 1:, 1:], dim=1))
    fakeindex = np.where(img_label==1.0)[0]
    attn_map_fake = torch.sigmoid(torch.mean(attn_map[fakeindex,:, 1:, 1:], dim=1))

    ## forgery localization 
    ## should accumulate the real/fake samples every iteration to acquire feat_tensorlist_real, feat_tensorlist_fake, then update MVG
    feat_tensorlist_real.append(feat_patch[realindex[:4]].reshape(B*H*W, C).cpu().detach())
    feat_tensorlist_fake.append(feat_patch[realindex[:4]].reshape(B*H*W, C).cpu().detach())

    if True: ## after training for a fix number of epoch
        mean_real, mean_fake, inv_covariance_real, inv_covariance_fake = estimate_MVG(feat_tensorlist_real, feat_tensorlist_fake)
    
    maha_patch_1 = mahalanobis_distance(feat_patch.reshape((B*H*W, C)), mean_real.cuda(), inv_covariance_real.cuda())
    maha_patch_2 = mahalanobis_distance(feat_patch.reshape((B*H*W, C)), mean_fake.cuda(), inv_covariance_fake.cuda())
    ## index_map is the pseudo localization map
    index_map = torch.relu(maha_patch_1 - maha_patch_2).reshape((B, H, W))

    ## calculate the consistency loss
    loss_inter_frame = attention_loss(attn_map_real, attn_map_fake, index_map[fakeindex,:])
