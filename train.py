import sys
import os
import random
import torch
from torch import nn, relu, threshold
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.utils.data
from dataset.faceforensics_dataset_stand import faceforensicsDataset_all
import torchvision.transforms as transforms
import time
from timeit import default_timer as timer
from tqdm import tqdm
import argparse
from sklearn import metrics
from network.vit_16_base_feat_middle_gal_v2 import vit_base_patch16_224
from utils.utils import time_to_str, get_EER_states, calculate_threshold
from utils.utils import Attention_Correlation_weight_reshape_loss, fit_inv_covariance, mahalanobis_distance
import json


def random_seed(seed):
    ## fix the random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    return

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        pass

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vit')
parser.add_argument('--dataset', default ="/data/FaceForensics++/manipulated_sequences/src/c23/faces/", help='path to root dataset')
parser.add_argument('--originroot', default ="/data/FaceForensics++/original_sequences/youtube/c23/faces/", help='path to root original')
parser.add_argument('--train_set', default ="./dataset/train_stand.json', help='train set')
parser.add_argument('--val_set', default ='./dataset/valid_stand.json', help='validation set')
parser.add_argument('--test_set', default ='./dataset/test_stand.json', help='test set')
parser.add_argument('--src', default="Deepfakes Face2Face FaceSwap NeuralTextures")
parser.add_argument('--test_src', default='Deepfakes Face2Face FaceSwap NeuralTextures')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=96, help='input batch size')
parser.add_argument('--batchSize_val', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--framenum', default=20)
parser.add_argument('--framenum_val', default=10)
parser.add_argument('--niter', type=int, default=8, help='number of epochs to train for first stage, get inital MVG param')
parser.add_argument('--update_epoch', type=float, default=0.5)
parser.add_argument('--optim', default='Adam')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.7)
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--eps', type=float, default=1e-08, help='epsilon')
parser.add_argument('--outf', default='./ckpt/', help='folder to output images and model checkpoints')
parser.add_argument('--log_root', default='./logs/', help='folder to record logs')
parser.add_argument('--manualSeed', type=int, help='manual seed', default=666)
parser.add_argument('--lambda_corr', type=float, default=0.06)
parser.add_argument('--lambda_c_param', default='0.5 0.05 0.05', help='lambda for c_cross, c_in, c_out')
parser.add_argument('--c_initilize', default='0.2 0.6 0.6')
parser.add_argument('--min_threshold_H', default='3 11')
parser.add_argument('--min_threshold', default='3 11', help="min_threshold_W")
parser.add_argument('--patch', type=int, default=16)
parser.add_argument('--attn_blk', type=str, default='8 9 10 11 12')
parser.add_argument('--feat_blk', type=int, default=6)
parser.add_argument('--k_weight', type=float, default=12.0)
parser.add_argument('--k_thr', type=float, default=0.7)
opt = parser.parse_args()

if opt.feat_blk :
    opt.feat_blk = int(opt.feat_blk)
opt.attn_blk = [int(i) for i in opt.attn_blk.split()]
if len(opt.attn_blk) == 1:
    opt.attn_blk = opt.attn_blk[0]
patch_num = int(opt.imageSize / opt.patch)

if __name__ == "__main__":
	os.makedirs(opt.outf, exist_ok=True)
    log = Logger(filename=os.path.join(opt.outf, 'train.log'))
    log.write(str(opt)+'\n')
    
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random_seed(opt.manualSeed)
    os.makedirs(opt.outf, exist_ok=True)
					
	model = vit_base_patch16_224(pretrained=True, num_classes=2)  
    network_loss = nn.CrossEntropyLoss()
    [c_cross, c_in, c_out] = [nn.Parameter(torch.tensor(float(i)).cuda()) for i in opt.c_initilize.split()]     
    min_threshold_H = [int(i) for i in opt.min_threshold_H.split()]
    min_threshold = [int(i) for i in opt.min_threshold.split()]
    attention_loss = Attention_Correlation_weight_reshape_loss(c_out=c_out, c_in=c_in, c_cross=c_cross)
    ###
    best_valacc = 0.0
    best_valauc = 0.0
    loss_train = 0
    loss_val = 0
    start = timer()
    epoch = 0
    start_iter = 0
    count = 0
    ###

    model = nn.DataParallel(model) 
    model.cuda()
    network_loss.cuda()
    if opt.lambda_c_param == '0 0 0':
        optimizer_dict = [{"params": model.parameters(), 'lr': opt.lr}]
    else:
        optimizer_dict = [{"params": model.parameters(), 'lr': opt.lr},
                        {"params": [c_in, c_out, c_cross], 'lr': opt.lr},]

    if opt.optim == 'Adam':
        optimizer = Adam(optimizer_dict, lr=opt.lr, betas=(opt.beta1, 0.999), eps=opt.eps)
    elif opt.optim == 'SGD':
        optimizer = SGD(optimizer_dict, lr=opt.lr, momentum=0.9, weight_decay=5e-4)
   
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    model.train(mode=True) 

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])


    dataset_train = faceforensicsDataset_all(rootpath=opt.dataset, origin=opt.originroot, datapath=opt.train_set, src=opt.src, framenum=opt.framenum, transform=transform_fwd)
    assert dataset_train
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

    dataset_val = faceforensicsDataset_all(rootpath=opt.dataset, origin=opt.originroot, datapath=opt.val_set, src=opt.test_src, framenum=opt.framenum_val, transform=transform_fwd)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize_val, shuffle=False, num_workers=int(opt.workers))
    
    
    tol_label = np.array([], dtype=np.float) 
    tol_pred = np.array([], dtype=np.float)
    feat_tensorlist = []
    feat_tensorlist_fake = []
    is_best_list = []
    train_iter = iter(dataloader_train)
    iter_per_epoch = len(train_iter)
    iter_per_valid = int(iter_per_epoch / 10.0)
    start_iter = epoch * iter_per_epoch + 1
    max_iter = opt.niter * iter_per_epoch
    for iter_num in range(start_iter, max_iter+1):
        if (iter_num != 0 and iter_num % int(opt.update_epoch * iter_per_epoch) == 0):
            ## for compute MVG parameter
            feat_tensorlist = torch.cat(feat_tensorlist, dim=0).cuda(2)
            inv_covariance_real = fit_inv_covariance(feat_tensorlist).cpu()
            mean_real = feat_tensorlist.mean(dim=0).cpu()  # mean features.
            gauss_param_real = {'mean': mean_real.tolist(), 'covariance': inv_covariance_real.tolist()}
            with open(os.path.join(opt.outf, 'gauss_param_real.json'),'w') as f:
                json.dump(gauss_param_real, f)
            feat_tensorlist = []

            feat_tensorlist_fake = torch.cat(feat_tensorlist_fake, dim=0).cuda(1)
            inv_covariance_fake = fit_inv_covariance(feat_tensorlist_fake).cpu()
            mean_fake = feat_tensorlist_fake.mean(dim=0).cpu()  # mean features.
            gauss_param_fake = {'mean': mean_fake.tolist(), 'covariance': inv_covariance_fake.tolist()}
            with open(os.path.join(opt.outf, 'gauss_param_fake.json'),'w') as f:
                json.dump(gauss_param_fake, f)
            feat_tensorlist_fake = []
            

        if (iter_num !=0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
            save_name = 'checkpoint.pth.tar'
            train_iter = iter(dataloader_train)
            torch.save({
                'epoch': epoch,
                'best_validacc': best_valacc,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'gauss_param_real': gauss_param_real,
                'gauss_param_fake': gauss_param_fake,
                }, os.path.join(opt.outf, save_name))
        
        
        model.train(True)
        ### load data
        img_data, labels_data = train_iter.next()
        img_label = labels_data.numpy().astype(np.float)
        img_data = img_data.cuda()
        labels_data = labels_data.cuda()

        step = iter_num / max_iter
        classes, feat_patch, attn_map = model(img_data, step=step, attn_blk=opt.attn_blk, feat_blk=opt.feat_blk, k=opt.k_weight, thr=opt.k_thr) 

        ### learn MVG parameters
        realindex = np.where(img_label==0.0)[0]
        attn_map_real = torch.sigmoid(torch.mean(attn_map[realindex,:, 1:, 1:], dim=1))
        feat_patch_real = feat_patch[realindex[:4]]
        B, H, W, C = feat_patch_real.size()
        feat_tensorlist.append(feat_patch_real.reshape(B*H*W, C).cpu().detach())

        fakeindex = np.where(img_label==1.0)[0]
        feat_patch_fake = feat_patch[fakeindex[:4]]
        attn_map_fake = torch.sigmoid(torch.mean(attn_map[fakeindex,:, 1:, 1:], dim=1))
        # attn_map_fake = torch.mean(attn_map[fakeindex,:, 1:, 1:], dim=1)

        del attn_map

        feat_patch_fake_inner = feat_patch_fake[:, min_threshold_H[0]:min_threshold_H[1], min_threshold[0]:min_threshold[1], :]
        B, H, W, C = feat_patch_fake_inner.size()
        feat_tensorlist_fake.append(feat_patch_fake_inner.reshape(B*H*W, C).cpu().detach())

        if epoch > 0 and iter_num >= int(opt.update_epoch * iter_per_epoch) and opt.lambda_corr > 0:
            fakeindex = np.where(img_label==1.0)[0]
            B, H, W, C = feat_patch.size()
            maha_patch_1 = mahalanobis_distance(feat_patch.reshape((B*H*W, C)), mean_real.cuda(), inv_covariance_real.cuda())
            maha_patch_2 = mahalanobis_distance(feat_patch.reshape((B*H*W, C)), mean_fake.cuda(), inv_covariance_fake.cuda())
            del feat_patch
            index_map = torch.relu(maha_patch_1 - maha_patch_2).reshape((B, H, W))
            
            loss_dis = network_loss(classes, labels_data.data)

            ### for attetion correlation loss
            loss_inter_frame = attention_loss(attn_map_real, attn_map_fake, index_map[fakeindex,:])
            lambda_c = [float(p) for p in opt.lambda_c_param.split()]
            loss_dis_tatol = loss_dis + opt.lambda_corr * loss_inter_frame + lambda_c[0]*torch.abs(c_cross) + lambda_c[1]/torch.abs(c_in) + lambda_c[2]/torch.abs(c_out)
        else:
            loss_dis = network_loss(classes, labels_data.data) 
            loss_dis_tatol = loss_dis
        

        loss_dis_tatol.backward()
        optimizer.step()
        optimizer.zero_grad()
        output_pred = F.softmax(classes, dim=1).cpu().data.numpy()[:, 1]
        tol_label = np.concatenate((tol_label, img_label)) 
        tol_pred = np.concatenate((tol_pred, output_pred))
        count += 1
        loss_train += loss_dis_tatol.item()
        
        if (iter_num != 0 and (iter_num+1) % iter_per_valid ==0):
            acc_train = metrics.accuracy_score(tol_label, np.where(tol_pred<0.5, 0, 1)) 
            loss_train /= count
            tol_label = np.array([], dtype=np.float) 
            tol_pred = np.array([], dtype=np.float)
            tol_pred_maha = np.array([],dtype=np.float)
            count = 0
            ########################################################################
            # do checkpointing & validation 
            model.eval() 
            tol_label_val = np.array([], dtype=np.float)
            tol_pred_val = np.array([], dtype=np.float)
            feat_patch = []
            count_val = 0
            with torch.no_grad():
                for img_data, labels_data in dataloader_val:
                    img_label = labels_data.numpy().astype(np.float)
                    img_data = img_data.cuda()
                    labels_data = labels_data.cuda()

                    classes, feat, _ = model(img_data, step=step, attn_blk=opt.attn_blk, feat_blk=opt.feat_blk, k=opt.k_weight, thr=opt.k_thr)
                   
                    if epoch > 0 and iter_num >= int(opt.update_epoch * iter_per_epoch) and opt.lambda_corr > 0:
                        ## for MVG predict acc and auc
                        B, H, W, C = feat.size()
                        feat = feat.reshape((B*H*W, C)).cpu().detach()
                        maha_real = mahalanobis_distance(feat, mean_real, inv_covariance_real).reshape((B, H*W)).cpu().data/torch.tensor(C**(0.5))
                        maha_fake = mahalanobis_distance(feat, mean_fake, inv_covariance_fake).reshape((B, H*W)).cpu().data/torch.tensor(C**(0.5))
                        maha = maha_real - maha_fake
                        maha = maha.numpy()
                        tol_pred_maha = np.concatenate((tol_pred_maha, np.array(np.mean(maha, axis=1), dtype=float)))

                    loss_dis = network_loss(classes, labels_data.data)
                    loss_dis_data = loss_dis.item()
                    output_pred = F.softmax(classes, dim=1).cpu().data.numpy()[:, 1]
                   
                    tol_label_val = np.concatenate((tol_label_val, img_label))
                    tol_pred_val = np.concatenate((tol_pred_val, output_pred))

                    loss_val += loss_dis_data
                    count_val += 1
            ####
            if epoch > 0 and opt.lambda_corr > 0:
                auc_MVG = metrics.roc_auc_score(tol_label_val, tol_pred_maha)
                # _, MVG_threshold, _, _ = get_EER_states(tol_pred_maha, tol_label_val)
                acc_MVG = calculate_threshold(tol_pred_maha, tol_label_val, 0)
            else:
                acc_MVG=0
                auc_MVG=0
            
            acc_val = metrics.accuracy_score(tol_label_val, np.where(tol_pred_val<0.5, 0, 1))
            auc_val = metrics.roc_auc_score(tol_label_val, tol_pred_val)
            loss_val /= count_val
            scheduler.step(acc_val)
            is_best = False
            if acc_val > best_valacc:
                is_best = True
                is_best_list = []
                best_valacc = acc_val
                best_valauc = auc_val
                torch.save(model.state_dict(), os.path.join(opt.outf, 'model_bestacc.pt'))
                torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optim_bestacc.pt'))
                if epoch > 0 and iter_num >= int(opt.update_epoch * iter_per_epoch) and opt.lambda_corr > 0:
                    with open(os.path.join(opt.outf, 'gauss_param_real_bestacc.json'),'w') as f:
                        json.dump(gauss_param_real, f)
                    with open(os.path.join(opt.outf, 'gauss_param_fake_bestacc.json'),'w') as f:
                        json.dump(gauss_param_fake, f)
            is_best_list.append(is_best)
            
            log.write('[Epoch%4.1f] Train loss:%5.3f  acc:%6.2f | Val loss:%5.3f acc:%6.2f auc:%6.2f acc_MVG:%6.2f auc_MVG:%6.2f| Best model acc:%6.2f  auc:%6.2f | Time:%s | c_in:%.3f  c_out:%.3f  c_cross:%.3f\n'  
            % ((iter_num+1) / iter_per_epoch, loss_train, acc_train*100, loss_val, acc_val*100, auc_val*100, acc_MVG*100, auc_MVG*100, best_valacc*100, best_valauc*100,
              time_to_str(timer() - start, 'min'), c_in.data, c_out.data, c_cross.data))
 

