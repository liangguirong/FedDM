import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import logging
import os
from torch.utils.data import Dataset
import copy
from dataloaders import dataset
from dataloaders.augmentation.randaugment import RandAugment
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from datasets import CIFAR10_truncated, SVHN_truncated, CIFAR100_truncated,FashionMNIST_truncated
import pandas as pd
from PIL import Image
from torchvision.datasets import MNIST, EMNIST, STL10, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils



def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_fmnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = FashionMNIST(datadir, True, download=True, transform=transforms.ToTensor())
    test_data = FashionMNIST(datadir, False, download=True, transform=transforms.ToTensor())
    #train_data = FashionMNIST_truncated(datadir, True, download=True, transform=transform)
    #test_data = FashionMNIST_truncated(datadir, False, download=True, transform=transform)
    X_train, y_train = train_data.data.numpy(), train_data.targets.numpy()
    X_test, y_test = test_data.data.numpy(), test_data.targets.numpy()
    return (X_train, y_train, X_test, y_test)
def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = MNIST(datadir, True, download=True, transform=transforms.ToTensor())
    test_data = MNIST(datadir, False, download=True, transform=transforms.ToTensor())
    X_train, y_train = train_data.data.numpy(), train_data.targets.numpy()
    X_test, y_test = test_data.data.numpy(), test_data.targets.numpy()
    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_STL10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = STL10(root=datadir, split = 'train', transform=transforms.ToTensor(),download=True)
    test_data = STL10(root=datadir, split = 'test', transform=transforms.ToTensor(),download=True)
    X_train, y_train = train_data.data, train_data.labels
    X_test, y_test = test_data.data, test_data.labels

    return (X_train, y_train, X_test, y_test)
def load_SVHN_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    SVHN_train_ds = SVHN_truncated(datadir, split='train', download=True, transform=transform)
    SVHN_test_ds = SVHN_truncated(datadir, split='test', download=True, transform=transform)

    X_train, y_train = SVHN_train_ds.data, SVHN_train_ds.target
    X_test, y_test = SVHN_test_ds.data, SVHN_test_ds.target

    return (X_train, y_train, X_test, y_test)

def load_skin_data(args):  # idxs相对所有data
    file = pd.read_csv(args.csv_file_train)
    X_train = file['ImageID'].values
    y_train = file.iloc[:, 1:].values.astype(int)
    _, y_train = torch.max(torch.tensor(y_train), dim=-1)
    file = pd.read_csv(args.csv_file_test)
    X_test = file['ImageID'].values
    y_test = file.iloc[:, 1:].values.astype(int)
    _, y_test = torch.max(torch.tensor(y_test), dim=-1)
    total = 0
    images_train = []
    for path in X_train:
        img_path = os.path.join(args.root_path, path + '.jpg')
        img = Image.open(img_path)
        img = np.array(img)
        total = total + 1
        images_train.append(img)
    images_train = np.array(images_train)
    images_test = []
    total = 0
    for path in X_test:
        img_path = os.path.join(args.root_path, path + '.jpg')
        img = Image.open(img_path)
        img = np.array(img)
        total = total + 1
        images_test.append(img)
    images_test = np.array(images_test)
    return images_train, y_train, images_test, y_test

def record_net_data_stats(y_train, net_dataidx_map,n_classes):
    ## usage: ?
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = {i: 0 for i in range(n_classes)}
        for i in range(len(unq)):
            net_cls_counts[net_i][unq[i]] = unq_cnt[i]

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, labeled_num, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)

    state = np.random.get_state()
    #let X_train and y_train have the same shuffle state
    np.random.shuffle(X_train)
    # print(a)
    # result:[6 4 5 3 7 2 0 1 8 9]
    np.random.set_state(state)
    np.random.shuffle(y_train)
    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        # min_require_size = 100
        sup_size = int(len(y_train) / 10)
        N = y_train.shape[0] - sup_size
        net_dataidx_map = {}
        for sup_i in range(labeled_num):
            net_dataidx_map[sup_i] = [i for i in range(sup_i * sup_size, (sup_i + 1) * sup_size)]

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties - labeled_num)]
            for k in range(K):
                idx_k = np.where(y_train[int(labeled_num * len(y_train) / 10):] == k)[0] + sup_size
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(
                    [p * (len(idx_j) < N / (n_parties - labeled_num)) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties - labeled_num):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j + labeled_num] = idx_batch[j]
    return (X_train, y_train, X_test, y_test, net_dataidx_map)
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def partition_data_allnoniid(args,dataset, datadir, train_idxs=None, test_idxs=None, partition="noniid", n_parties=10,
                             beta=0.8,n_classes=10):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'SVHN':
        X_train, y_train, X_test, y_test = load_SVHN_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'skin':
        X_train, y_train, X_test, y_test = load_skin_data(args)
    elif dataset == 'STL10':
        X_train, y_train, X_test, y_test = load_STL10_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    n_train = y_train.shape[0]
    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, n_classes)

    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = n_classes

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
        # repeat this process unitl the min_size of one party is greater than requirement
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # whether the data number in party k is enough
                proportions = np.array(
                    [p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # normalize proportions
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
        if dataset == 'fmnist':
            idx_batch1 = net_dataidx_map[0]
            net_dataidx_map[0] = net_dataidx_map[1]
            net_dataidx_map[1] = idx_batch1
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map,n_classes)
    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


def get_dataloader(args,data_np, label_np, dataset_type, datadir, train_bs, is_labeled=None, data_idxs=None,
                   is_testing=False, pre_sz=40, input_sz=32, data_list=None):
    if dataset_type == 'SVHN':
        normalize = transforms.Normalize(mean=[0.4376821, 0.4437697, 0.47280442],
                                         std=[0.19803012, 0.20101562, 0.19703614])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'
    elif dataset_type == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'
    elif dataset_type == 'skin':
        normalize = transforms.Normalize(mean=[0.7630332, 0.5456457, 0.57004654],
                                         std=[0.14092809, 0.15261231, 0.16997086])
        pre_sz = 224
        input_sz = 224

    elif dataset_type == 'cifar10':
        normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                         std=[0.24703233, 0.24348505, 0.26158768])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'
    elif dataset_type == 'STL10':
        pre_sz = 96
        input_sz = 96
        normalize = transforms.Normalize(mean=[x / 255 for x in [112.4, 109.1, 98.6]],
                                         std=[x / 255 for x in [68.4, 66.6, 68.5]])
        assert pre_sz == 96 and input_sz == 96, 'Error: Wrong input size for 32*32 dataset'
    elif dataset_type == 'fmnist':
        pre_sz = 36
        input_sz = 28
        normalize = transforms.Normalize(mean=[0.2860402],
                                         std=[0.3530239])
        assert pre_sz == 36 and input_sz == 28, 'Error: Wrong input size for 32*32 dataset'

    if not is_testing:
        if is_labeled:
            trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.RandomCrop(input_sz, padding=4, padding_mode='reflect'),
                                       transforms.ToTensor(),
                                       normalize])
            ds = dataset.CheXpertDataset(dataset_type,args.root_path, data_np, label_np, pre_sz, pre_sz, lab_trans=trans,
                                         is_labeled=True, is_testing=False)

        else:
            weak_trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(input_sz, padding=4, padding_mode='reflect'),
                                        transforms.ToTensor(),
                                        normalize])
            strong_trans = copy.deepcopy(weak_trans)
            strong_trans.transforms.insert(0, RandAugment(3, 5))
            ds = StrongDataset(dataset_type,args.root_path, data_np, label_np, pre_sz, pre_sz,
                               un_trans_wk=weak_trans,
                               un_trans_st=strong_trans,
                               data_idxs=data_idxs,
                               is_labeled=False,
                               is_testing=False)
        if data_list == None:
            dl = data.DataLoader(dataset=ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=0)
        else:
            dl = data.DataLoader(dataset=ds + data_list, batch_size=train_bs, drop_last=True, shuffle=True,
                                 num_workers=0)
    else:
        ds = dataset.CheXpertDataset(dataset_type,args.root_path, data_np, label_np, input_sz, input_sz,
                                     lab_trans=transforms.Compose([
                                         # K.RandomCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize
                                     ]), is_labeled=True, is_testing=True)
        dl = data.DataLoader(dataset=ds, batch_size=train_bs, drop_last=False, shuffle=False, num_workers=0)
    return dl, ds, normalize
class StrongDataset(Dataset):
    def __init__(self, dataset_type,root_path, data_np, label_np, pre_w, pre_h, lab_trans=None, un_trans_wk=None, un_trans_st=None,
                 data_idxs=None,
                 is_labeled=False,
                 is_testing=False):
        """
        Args:

            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(StrongDataset, self).__init__()
        self.root_path = root_path
        self.images = data_np
        self.labels = label_np
        self.is_labeled = is_labeled
        self.dataset_type = dataset_type
        self.is_testing = is_testing
        self.resize = transforms.Compose([transforms.Resize((pre_w, pre_h))])
        if not is_testing:
            if is_labeled == True:
                self.transform = lab_trans
            else:
                self.data_idxs = data_idxs
                self.weak_trans = un_trans_wk
                self.strong_trans = un_trans_st
        else:
            self.transform = lab_trans

        print('Total # images:{}, labels:{}'.format(len(self.images), len(self.labels)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        if self.dataset_type == 'skin1':
            img_path = os.path.join(self.root_path, self.images[index]+ '.jpg')
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.fromarray(self.images[index]).convert('RGB')

        image_resized = image
        label = self.labels[index]

        if not self.is_testing:
            if self.is_labeled == True:
                if self.transform is not None:
                    image = self.transform(image_resized).squeeze()
                    # image=image[:,:224,:224]
                    return index, image, torch.FloatTensor([label])
            else:
                if self.weak_trans and self.data_idxs is not None:
                    weak_aug = self.weak_trans(image_resized)
                    strong_aug = self.weak_trans(image_resized)
                    idx_in_all = self.data_idxs[index]

                    for idx in range(len(weak_aug)):
                        weak_aug[idx] = weak_aug[idx].squeeze()
                        strong_aug[idx] = strong_aug[idx].squeeze()
                    return index, [weak_aug, strong_aug], torch.FloatTensor([label])
        else:
            image = self.transform(image_resized)
            return index, image, torch.FloatTensor([label])
            # return index, weak_aug, strong_aug, torch.FloatTensor([label])

    def __len__(self):
        return len(self.labels)

