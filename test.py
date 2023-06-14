import sys

sys.setrecursionlimit(15000)
import os
import random
from PIL import Image
import torch
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm
import glob
import argparse
from network.vit_16_base_feat_middle_gal_v2 import vit_base_patch16_224
from sklearn.metrics import roc_auc_score
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vit')
parser.add_argument('--dataset', default ="", help='path to dataset')
parser.add_argument('--originroot', default ="", help='path to root original')
parser.add_argument('--test_set', default ='./dataset/test.json', help='test set')
parser.add_argument('--src', default="Deepfakes Face2Face FaceSwap NeuralTextures")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchsize', type=int, default=128, help='input batch size')
parser.add_argument('--framenum', type=int, default=110)
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--patch', type=int, default=16)
parser.add_argument('--outf', default='', help='folder to output log and load model checkpoints')
parser.add_argument('--testmodel', default="model_bestacc.pt")
parser.add_argument('--record_error', default=True)
parser.add_argument('--attn_blk', type=str, default='8 9 10 11 12')
parser.add_argument('--feat_blk', type=int, default=6)
parser.add_argument('--k_weight', type=float, default=12.0)
parser.add_argument('--k_thr', type=float, default=0.7)
parser.add_argument('--step', type=float, default=1.0)
parser.add_argument('--is_progressive', type=int, default=1)
opt = parser.parse_args()
print(opt)

opt.attn_blk = [int(i) for i in opt.attn_blk.split()]

class TestSet(Dataset):
    def __init__(self, test_frames, face_path):
        random.seed(8664)
        self.facepath = face_path
        self.frames = test_frames
        self.transform = transforms.Compose([
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
        self.imgs = self._get_img_list()
        

    def _get_img_list(self):
        imgs = []
        filelist = glob.glob(os.path.join(self.facepath, '*_0.png'))
        try:
            for path in filelist[:self.frames]:
                imgs.append(path)
        except(IndexError):
            pass       
    
        return imgs

    def __len__(self):
        return(len(self.imgs))

    def _load_image(self, path):
        image = Image.open(path).convert('RGB')
        return image
    
    def __getitem__(self,idx):
        img_name = self.imgs[idx]
        image = self._load_image(img_name)
        image = self.transform(image)
        return image

def get_datalist_stand(datapath):
    with open(datapath, 'r') as f:
        data_dict = json.load(f)
    output = []
    for pair in data_dict:
        name1, name2 = pair[0], pair[1]
        output.append([name1, '0'])
        output.append([name2, '0'])
        output.append([name1 + '_' + name2, '1'])
        output.append([name2 + '_' + name1, '1'])
    return output


def test(model, testroot, origin, testlist, test_frame, batchsize, record_error):
    video_list = get_datalist_stand(testlist)
    tol_label = np.array([])
    tol_label_frame = np.array([])
    tol_pred = np.array([])
    tol_pred_frame = np.array([])
   
    if record_error:
        error_vid_file = open(os.path.join(opt.outf, 'test_error_message.txt'), 'w')
    for item in tqdm(video_list):
        vid_name, label = item
        if label == "0":
            face_path = os.path.join(origin, vid_name)
            testdataset = TestSet(test_frames=test_frame, face_path=face_path)
            testloder = DataLoader(
                testdataset,
                batch_size=test_frame,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            pred = test_video(testloder, model)
            ## calculate accuracy in frame level
            target = [float(label)] * len(pred)
            tol_pred_frame = np.concatenate((tol_pred_frame, pred))
            tol_label_frame = np.concatenate((tol_label_frame, target))
            ## calculate accuracy in video level
            preds = np.array([np.mean(pred)], dtype=float)
            label = np.array([float(label)], dtype=float)
            tol_label = np.concatenate((tol_label, label))
            tol_pred = np.concatenate((tol_pred, preds))
            if record_error:
                pred_label = 1.0 if preds[0] >= 0.5 else 0.0
                if pred_label != label[0]:
                    txt = vid_name.split('/')[-1] + ' ' + str(preds[0]) + '\n'
                    error_vid_file.write(txt) 
        else:
            # forgery = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
            forgery = [s for s in opt.src.split()]
            for f in forgery:
                face_path = os.path.join(testroot.replace('src', f), vid_name)
                testdataset = TestSet(test_frames=test_frame, face_path=face_path)
                testloder = DataLoader(
                    testdataset,
                    batch_size=test_frame,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )
                
                pred = test_video(testloder, model)
                ## calculate accuracy in frame level
                target = [float(label)] * len(pred)

                tol_pred_frame = np.concatenate((tol_pred_frame, pred))
                tol_label_frame = np.concatenate((tol_label_frame, target))
                ## calculate accuracy in video level
                preds = np.array([np.mean(pred)], dtype=float)
                label = np.array([float(label)], dtype=float)
                tol_label = np.concatenate((tol_label, label))
                tol_pred = np.concatenate((tol_pred, preds))
                if record_error:
                    pred_label = 1.0 if preds[0] >= 0.5 else 0.0
                    if pred_label != label[0]:
                        txt = f + '/'+ vid_name + ' ' + str(preds[0]) + '\n'
                        error_vid_file.write(txt) 
    
    auc = roc_auc_score(tol_label, tol_pred)
    auc_frame  =roc_auc_score(tol_label_frame, tol_pred_frame)

    top1_acc = compute_accuracy(tol_label, tol_pred)
    top1_acc_frame = compute_accuracy(tol_label_frame, tol_pred_frame)
    

    if record_error:
        error_vid_file.close()
    return top1_acc_frame, auc_frame, top1_acc, auc

def compute_accuracy(target, output):
    pred = output > 0.5
    return np.sum(pred==target)/target.shape[0]

def test_video(testloder, model):
    model.eval()
    with torch.no_grad():
        preds = np.array([])
        for img in testloder:
            img = img.cuda()
            cls_out, _, _ = model(img, step=opt.step, attn_blk=opt.attn_blk, feat_blk=opt.feat_blk, k=opt.k_weight, thr=opt.k_thr, is_progressive=opt.is_progressive)
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            preds = np.concatenate((preds, prob))

    return preds 



if __name__ == '__main__':

    text_writer = open(os.path.join(opt.outf, 'test.txt'), 'a')
    net = vit_base_patch16_224(pretrained=False, num_classes=2)

    print('\n')
    print("**Testing** Get test files done!")
    print("Testroot : " + opt.dataset)
    print("Test Model: " + opt.outf+opt.testmodel)
    # load model
    net = torch.nn.DataParallel(net).cuda()
    net_ = torch.load(os.path.join(opt.outf, opt.testmodel))
    net.load_state_dict(net_)
    # test model
    test_args = test(net, opt.dataset, opt.originroot, opt.test_set, 
                opt.framenum, opt.batchsize, opt.record_error)
    print('\n===========Test Info===========\n')
    print('Test ACC_frame: %6.3f' % (test_args[0]*100))
    print('Test AUC_frame: %6.3f' % (test_args[1]*100))
    print('Test ACC_vid: %6.3f' % (test_args[2]*100))
    print('Test AUC_vid: %6.3f' % (test_args[3]*100))
    print('\n===============================\n')


        

