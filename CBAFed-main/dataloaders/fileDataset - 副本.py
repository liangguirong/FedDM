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

    def __init__(self,
                 path,is_labeled=False,
                 num_classes=None,
                 transform=None,strong_transform=None,
                 is_gray = True,
                 idxs = None,
                 path_filter=None,is_dming=False,
                 *args, **kwargs):
        super(fileDataset, self).__init__()
        # self.alg = alg
        self.path = path
        self.img_files = os.listdir(path)
        weak_trans = transform
        if idxs != None:
            # self.img_files = np.array(self.img_files)   #.astype(str)  # array可以索引列表选择， ndarray 不可以索引列表选择元素
            temp = []
            for idx in idxs:
                temp += [ self.img_files[idx] ]
            self.img_files = temp
        self.labels = [ int(file.split('_')[0]) for file in self.img_files ]
        self.is_labeled = is_labeled
        self.num_classes = num_classes
        self.transform = weak_trans
        self.is_gray = is_gray
        self.is_dming = is_dming
        if path_filter != None:
            self.path_filter = path_filter
            os.makedirs(path_filter, exist_ok= True)


    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.img_files[idx])
        img = Image.open(img_path)  # 此句有问题， 导入的是RGB图
        # if self.is_gray:
        #     img = img.convert('L')  # 转灰度图，  PIL 了解
        target = int(self.img_files[idx].split('_')[0])
        # if self.is_dming:
        #     w_img = img
        #     aug_img = img
        # else:
        w_img = self.transform(img)
        return idx, [w_img, w_img], torch.FloatTensor([target])  # ToTensor()(img) 声明类 ，然后调用

    def __len__(self):
        return len(self.img_files)

    def remove_sample(self, idxs, is_save = False):
        # print("idxs:",idxs)
        # file_list = [self.img_files_fix[n] for n in idxs]   # 获取 索引对应的 元素
        file_list = [self.img_files[n] for n in idxs]   # 不能直接用索引删除，因为删除时列表长度会改变
        # file_list = self.img_files_fix[idxs]  # 获取 索引对应的 元素
        num = len(idxs)
        # print(idxs)
        # 按照元素删除，按照索引删除会出问题，因为长度的改变
        for element in file_list:
            # if element in self.img_files:   # 为什么会出现不索引的样本，问题在第一行
            self.img_files.remove(element)
            # self.img_files = np.delete(self.img_files, np.where(self.img_files == element ))  # 删除元素
            # 保存问题图片
            if is_save:
                self.save_image(
                    os.path.join(self.path, element)
                )

        l = len(self.img_files)
        print("Now, {} samples! ,del {} !".format(l, num))

    def save_image(self,srcfile,  verbose= False):
        dstpath = self.path_filter
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        # if not os.path.exists(dstpath):
        #     os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, os.path.join( dstpath , fname ))  # 复制文件
        if verbose:
            print("copy %s -> %s" % (srcfile, dstpath + fname))

def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot