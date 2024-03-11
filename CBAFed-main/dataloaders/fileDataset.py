import copy
import os
import shutil

import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from PIL import Image # ？？？  查看官方文档
from .augmentation.randaugment import RandAugment
class fileDataset(Dataset):
    """
    将path文件夹下的图片文件作为数据，
    图片的命名格式 ：标签_文件名，
    """

    def __init__(self,data_np,transform,data_idxs=None):
        super(fileDataset, self).__init__()
        self.images = data_np
        self.weak_trans = transform
        strong_trans = copy.deepcopy(self.weak_trans)
        strong_trans.transforms.insert(0, RandAugment(3, 5))
        self.strong_transform = strong_trans
    def __getitem__(self, index):

        image = Image.fromarray(self.images[index]).convert('RGB')
        weak_aug = self.weak_trans(image)
        aug_img = self.strong_transform(image)
        return index, [weak_aug, aug_img], torch.FloatTensor([0])

    def __len__(self):
        return len(self.images)