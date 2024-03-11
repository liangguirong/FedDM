import numpy as np
import torch
import torch.optim
import copy
from FedAvg import FedAvg, model_dist
import torch.nn.functional as F
from train_utils import consistency_loss
import logging
from train_utils import TBLog, get_optimizer, get_cosine_schedule_with_warmup
from validation import epochVal_metrics_test
from cifar_load import get_dataloader
import contextlib
import torchvision.models as torch_models
import torch.nn as nn
from utils import DiffAugment,ParamDiffAug

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets.long(), reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


def test(args,epoch, checkpoint, data_test, label_test, n_classes):
    if args.model == 'Res18':
        net = torch_models.resnet18(pretrained=args.Pretrained)
        net.fc = nn.Linear(net.fc.weight.shape[1], n_classes)
    # net = get_network(args.model, args.channel, args.n_classes, args.imsize).cuda()
    if len(args.gpu.split(',')) > 1:
        net = torch.nn.DataParallel(net, device_ids=[i for i in range(round(len(args.gpu) / 2))])
    model = net.cuda()
    model.load_state_dict(checkpoint)

    if args.dataset == 'SVHN' or args.dataset == 'cifar100' or args.dataset == 'cifar10':
        test_dl, test_ds,_ = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True)
    elif args.dataset == 'skin' or args.dataset == 'STL10' or args.dataset == 'fmnist' or args.dataset == 'mnist':
        test_dl, test_ds,_ = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True, pre_sz=args.pre_sz, input_sz=args.input_sz)

    AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, n_classes=n_classes)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()

    return AUROC_avg, Accus_avg
def setattr_cls_from_kwargs(cls, kwargs):
    # if default values are in the cls,
    # overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls, key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])
def net_builder(net_name, from_name: bool, net_conf=None, is_remix=False):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
    """
    if from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(models.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return models.__dict__[net_name]

    else:
        if net_name == 'WideResNet':
            import models.nets.wrn as net
            builder = getattr(net, 'build_WideResNet')()
        elif net_name == 'WideResNetVar':
            import models.nets.wrn_var as net
            builder = getattr(net, 'build_WideResNetVar')()
        elif net_name == 'ResNet50':
            import models.nets.resnet50 as net
            builder = getattr(net, 'build_ResNet50')(is_remix)
        else:
            assert Exception("Not Implemented Error")

        if net_name != 'ResNet50':
            setattr_cls_from_kwargs(builder, net_conf)
        return builder.build

class SupervisedLocalUpdate(object):
    def __init__(self, args, idxs, n_classes):
        self.epoch = 0
        self.iter_num = 0
        self.base_lr = args.base_lr
        self.data_idx = idxs
        self.con = 0
        self.max_grad_norm = args.max_grad_norm
        if args.model == 'Res18':
            net = torch_models.resnet18(pretrained=args.Pretrained)
            net.fc = nn.Linear(net.fc.weight.shape[1], n_classes)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net, device_ids=[i for i in range(round(len(args.gpu) / 2))])
        self.model = net.cuda()
        args.dsa_param = ParamDiffAug()
        self.ulb_prob_t = torch.ones((args.n_classes)).cuda() / args.n_classes
        self.sum_p = []
        self.prob_max_mu_t = 1.0 / args.n_classes
        self.prob_max_std_t = 0.1
        self.ema = args.ema
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                    2 ** 20,
                                                    num_warmup_steps=2 ** 20 * 0)
    def update_prob_t(self, ulb_probs):
        ulb_prob_t = ulb_probs.mean(0)
        self.ulb_prob_t = self.ema * self.ulb_prob_t + (1 - self.ema) * ulb_prob_t
        max_probs, max_idx = ulb_probs.max(dim=-1)
        prob_max_mu_t = torch.mean(max_probs)
        prob_max_std_t = torch.std(max_probs, unbiased=True)
        self.prob_max_mu_t = self.ema * self.prob_max_mu_t + (1 - self.ema) * prob_max_mu_t.item()
        self.prob_max_std_t = self.ema * self.prob_max_std_t + (1 - self.ema) * prob_max_std_t.item()

    @torch.no_grad()
    def cal_time_p_and_p_model(self, logits_x_ulb_w, time_p, p_model):
        prob_w = torch.softmax(logits_x_ulb_w, dim=1)
        max_probs, max_idx = torch.max(prob_w, dim=-1)
        if time_p is None:
            time_p = max_probs.mean()
        else:
            time_p = time_p * 0.999 + max_probs.mean() * 0.001
        if p_model is None:
            p_model = torch.mean(prob_w, dim=0)
        else:
            p_model = p_model * 0.999 + torch.mean(prob_w, dim=0) * 0.001
        return time_p, p_model
    def train(self, args, net_w, op_dict, dataloader, n_classes, is_test=False, local_w=None, X_test=None, y_test=None, res=False, stage=2,class_confident=None,avg_local_label=None,gdataloader=None):
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.model.cuda().train()
        self.optimizer.load_state_dict(op_dict)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr
        loss_fn = torch.nn.CrossEntropyLoss()
        epoch_loss = []
        test_acc_avg = []
        logger = logging.getLogger()
        if stage==1:
            if args.dataset=='cifar100':
                s_epoch = 1001
            else:
                s_epoch = 1
                #s_epoch = 501
        else:
            s_epoch = 1
        p_model = (torch.ones(args.n_classes) / args.n_classes).cuda()
        time_p = p_model.mean()
        amp_cm = contextlib.nullcontext
        threshold =None
        for epoch in range(s_epoch):
            self.model.train()
            batch_loss = []
            accuracy = []
            total = 0
            train_label = []
            class_num = torch.zeros(n_classes)
            trainloader_l_iter = enumerate(dataloader)
            if gdataloader==None:
                self.steps = len(dataloader)
                for i, (_, image_batch, label_batch) in enumerate(dataloader):

                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    label_batch = label_batch.long().squeeze()
                    inputs = image_batch
                    if args.dsa:
                        inputs = DiffAugment(inputs, args.dsa_strategy, param=args.dsa_param)
                    outputs = self.model(inputs)
                    total = total + len(image_batch)
                    train_label.append(label_batch)
                    loss_classification = loss_fn(outputs, label_batch)
                    loss = loss_classification
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   max_norm=self.max_grad_norm)
                    self.optimizer.step()

                    batch_loss.append(loss.item())
                    self.iter_num = self.iter_num + 1
                    with torch.no_grad():
                        accuracy.append((torch.argmax(outputs, dim=1) == label_batch).float().mean())
            else:
                self.steps = max(len(dataloader),len(gdataloader))
                trainloader_u_iter = enumerate(gdataloader)
                for i in range(self.steps):
                    # Check if the label loader has a batch available
                    try:
                        _, sample_batched = next(trainloader_l_iter)
                    except:
                        # Curr loader doesn't have data, then reload data
                        del trainloader_l_iter
                        trainloader_l_iter = enumerate(dataloader)
                        _, sample_batched = next(trainloader_l_iter)
                    try:
                        _, sample_batched_u = next(trainloader_u_iter)
                    except:
                        del trainloader_u_iter
                        trainloader_u_iter = enumerate(gdataloader)
                        _, sample_batched_u = next(trainloader_u_iter)

                    x_lb, x_ulb_w, x_ulb_s = sample_batched[1].cuda(), sample_batched_u[1][0].cuda(), sample_batched_u[1][1].cuda()
                    y_lb = sample_batched[2].cuda()
                    # inputs = x_lb
                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                    if args.dsa:
                        inputs = DiffAugment(inputs, args.dsa_strategy, param=args.dsa_param)
                    # inference and calculate sup/unsup losses
                    with amp_cm():
                        num_lb = x_lb.shape[0]
                        logits = self.model(inputs)
                        logits_x_lb = logits[:num_lb]
                        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                        y_lb = y_lb.squeeze()
                        sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                        # hyper-params for update
                        time_p, p_model = self.cal_time_p_and_p_model(logits_x_ulb_w, time_p, p_model)
                        unsup_loss, max_idx, threshold,_ = consistency_loss(args.dataset, logits_x_ulb_s, logits_x_ulb_w,
                                                            time_p, p_model,
                                                            'ce',class_confident,avg_local_label,use_hard_labels=True)
                        train_label.append(y_lb)
                        train_label.append(max_idx)
                        loss = sup_loss + unsup_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   max_norm=self.max_grad_norm)
                    self.optimizer.step()

                    batch_loss.append(loss.item())
                    self.iter_num = self.iter_num + 1
                    with torch.no_grad():
                        accuracy.append((torch.argmax(logits_x_lb, dim=1) == y_lb).float().mean())
            self.epoch = self.epoch + 1
            train_label = torch.cat(train_label, dim=0)
            for i in range(n_classes):
                class_num[i] = (train_label == i).float().sum()
            epoch_loss.append(np.array(batch_loss).mean())
            # weight connection with previous epoch
            res = False
            if res:
                if (epoch==0):
                    print('res weight connection')
                    record_w = (copy.deepcopy(self.model.cpu().state_dict()))
                    self.model.cuda()
                if (epoch>0 and epoch%5==0):
                    print('res weight connection')
                    w_l = [record_w, copy.deepcopy(self.model.cpu().state_dict())]
                    if args.dataset == 'skin':
                        if stage==1:
                            n_l = [4., 1.]
                        else:
                            n_l = [4., 1.]
                    else:
                        n_l = [4., 1.]
                    print(n_l)
                    w = FedAvg(record_w, w_l, n_l)
                    record_w = copy.deepcopy(w)
                    self.model.load_state_dict(w)
                    self.model.cuda()
            if epoch>0 and epoch%1==0 and type(X_test) != type(None):
                w = self.model.cpu().state_dict()
                AUROC_avg, Accus_avg = test(args,epoch, w, X_test, y_test, n_classes)
                logger.info("epoch {}, AUROC_avg {}, AUROC_avg {}".format(epoch, AUROC_avg,Accus_avg))
                test_acc_avg.append(Accus_avg)
                logger.info("max test_acc_avg {}, test_acc_avg {}".format(max(test_acc_avg), test_acc_avg))
                logger.info("class_num {}, threshold {}".format(class_num, threshold))
                self.model.cuda()
            if s_epoch == epoch+1:
                logger.info("epoch {}, accuracy {}".format(epoch, sum(accuracy)/len(accuracy)))
        self.model.cpu()
        return self.model.state_dict(), copy.deepcopy(self.optimizer.state_dict()),  sum(class_num), class_num,threshold
