import os
import glob
import random
from PIL import Image
from PIL import ImageFile
from imageio import imread
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import copy
import json

def get_datalist_stand(datapath):
    with open(datapath, 'r') as f:
        data_dict = json.load(f)
    output = []
    for pair in data_dict:
        name1, name2 = pair[0], pair[1]
        output.append(name1 + ' 0')
        output.append(name2 + ' 0')
        output.append(name1 + '_' + name2 + ' 1')
        output.append(name2 + '_' + name1 + ' 1')
    return output


class faceforensicsDataset_all(Dataset):
    def __init__(self, rootpath, origin, datapath, src, framenum, transform=None):
        assert 'src' in rootpath
        imgsfolderPath = get_datalist_stand(datapath)
        imgs = []
        # manipulations = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        manipulations = [s for s in src.split()]
        framenum_list = []
        if isinstance(framenum, int):
            framenum_list = [framenum] * len(manipulations)
        elif isinstance(framenum, str):
            framenum_list = [int(u) for u in framenum.split()]
        for line in imgsfolderPath:
            words = line.split()
            if words[1] == '1':
                for i, src in enumerate(manipulations):
                    filelist = glob.glob(os.path.join(rootpath.replace('src', src), words[0], '*_0.png'))
                    if 'train' in datapath:
                        random.shuffle(filelist)
                    for frame in filelist[:framenum_list[i]]:
                        imagepath = frame
                        imgs.append((imagepath, int(words[1])))

            else:
                filelist = glob.glob(os.path.join(origin, words[0], '*_0.png'))
                if 'train' in datapath:
                    random.shuffle(filelist)
                    framenum_real = sum(framenum_list)
                else:
                    framenum_real = int(sum(framenum_list)/len(manipulations))
                for frame in filelist[:framenum_real]:
                    imagepath = frame
                    imgs.append((imagepath, int(words[1])))

        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        #print(len(self.imgs))
        return(len(self.imgs))


    def __getitem__(self,idx):
        img_name, label = self.imgs[idx]
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label

