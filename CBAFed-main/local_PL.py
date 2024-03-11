from torch.utils.data import Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
import logging
from torchvision import transforms
import torchvision.models as torch_models
import torch.nn as nn
from utils import DiffAugment,ParamDiffAug

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        items, index, weak_aug, strong_aug, label = self.dataset[self.idxs[item]]
        return items, index, weak_aug, strong_aug, label


class PLUpdate(object):
    def __init__(self, args, idxs, n_classes):
        if args.model == 'Res18':
            net = torch_models.resnet18(pretrained=args.Pretrained)
            net.fc = nn.Linear(net.fc.weight.shape[1], n_classes)
        self.model = net.cuda()
        self.data_idxs = idxs
        self.epoch = 0
        self.iter_num = 0
        self.flag = True
        self.softmax = nn.Softmax()
        self.max_grad_norm = args.max_grad_norm
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.ema = args.ema
        self.max_step = args.rounds * round(len(self.data_idxs) / args.batch_size)
        self.ulb_prob_t = torch.ones((args.n_classes)) / args.n_classes
        self.prob_max_mu_t = 1.0 / args.n_classes
        args.dsa_param = ParamDiffAug()
        self.prob_max_std_t = 0.1
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.unsup_lr,momentum=0.9, weight_decay=5e-4)

    def update_prob_t(self, ulb_probs):
        ulb_prob_t = ulb_probs.mean(0)
        self.ulb_prob_t = self.ema * self.ulb_prob_t + (1 - self.ema) * ulb_prob_t
        max_probs, max_idx = ulb_probs.max(dim=-1)
        prob_max_mu_t = torch.mean(max_probs)
        prob_max_std_t = torch.std(max_probs, unbiased=True)
        self.prob_max_mu_t = self.ema * self.prob_max_mu_t + (1 - self.ema) * prob_max_mu_t.item()
        self.prob_max_std_t = self.ema * self.prob_max_std_t + (1 - self.ema) * prob_max_std_t.item()
    def train(self, args, net_w, op_dict, train_dl_local, n_classes, class_confident=None,avg_local_label=None, include_second=False,is_train=True,):
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.model.train()
        self.model.cuda()
        self.optimizer.load_state_dict(op_dict)
        logger = logging.getLogger()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = args.unsup_lr
        loss_fn = torch.nn.CrossEntropyLoss()
        for i in range(1):
            train_data = []
            train_label = []
            num = 0
            total = 0
            for i, (_, weak_image_batch, label_batch) in enumerate(train_dl_local):
                # obtain pseudo labels
                with torch.no_grad():
                    image_batch = weak_image_batch[0]
                    total = total + len(image_batch)
                    image_batch = image_batch.cuda()
                    if args.dsa:
                        image_batch = DiffAugment(image_batch, args.dsa_strategy, param=args.dsa_param)
                    self.model.eval()
                    outputs = self.model(image_batch)
                    if len(outputs.shape) != 2:
                        outputs = outputs.unsqueeze(dim=0)
                    guessed = F.softmax(outputs, dim=1).cpu()
                    pseu = torch.argmax(guessed, dim=1).cpu()
                    confident_threshold = torch.zeros(pseu.shape)
                    self.update_prob_t(guessed)
                    local_label = (self.ulb_prob_t / sum(self.ulb_prob_t))
                    class_confident = avg_local_label+self.prob_max_mu_t-avg_local_label.std()
                    class_confident[class_confident >= 1] = 1
                    for i in range(len(pseu)):
                        confident_threshold[i] = class_confident[pseu[i]]
                        #confident_threshold[i] = 0.5
                pl = pseu[torch.max(guessed, dim=1)[0] > confident_threshold]
                num = num + len(pl)
                select_samples = image_batch[torch.max(guessed, dim=1)[0] > confident_threshold]
                train_label.append(pl)
                train_data.append(select_samples)
                self.iter_num = self.iter_num + 1
            logger.info("selected number {}".format(num))
            train_data = torch.cat(train_data, dim=0)
            train_label = torch.cat(train_label, dim=0)
            class_num = torch.zeros(n_classes)
            for i in range(n_classes):
                class_num[i] = (train_label==i).float().sum()
            if is_train:
                for j in range(1):
                    for i in range(0, len(train_data), args.batch_size):
                        self.model.train()
                        data_batch = train_data[i:min(len(train_data), i+args.batch_size)].cuda()
                        if(len(data_batch)==1):
                            continue
                        label_batch = train_label[i:min(len(train_label), i + args.batch_size)].cuda()
                        outputs = self.model(data_batch)
                        if len(label_batch.shape) == 0:
                            label_batch = label_batch.unsqueeze(dim=0)
                        if len(outputs.shape) != 2:
                            outputs = outputs.unsqueeze(dim=0)
                        loss_classification = loss_fn(outputs, label_batch)
                        loss = loss_classification
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       max_norm=self.max_grad_norm)
                        self.optimizer.step()
        logger.info("local_label:{},max_mu:{},max_std:{},class_confident:{}".format(local_label,self.prob_max_mu_t,local_label.std(),class_confident))
        self.model.cpu()
        return self.model.state_dict(), copy.deepcopy(self.optimizer.state_dict()), sum(class_num), class_num,class_confident,self.prob_max_mu_t