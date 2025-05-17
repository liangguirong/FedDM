from validation import epochVal_metrics_test
from options import args_parser
import os
import sys
import logging
import random
from FedAvg import FedAvg, model_dist
import json
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from utils import *
import copy
from PIL import Image
import torchvision.models as torch_models
from dataloaders.fileDataset import fileDataset
from local_supervised import SupervisedLocalUpdate
from local_PL import PLUpdate
from tqdm import tqdm,trange
from cifar_load import get_dataloader, partition_data, partition_data_allnoniid


def split(dataset, num_users):
    ## randomly split basicdataset into equally num_users parties
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def select_samlple(x,y,n_classes):
    class_n = torch.zeros(n_classes)
    for i in range(n_classes):
        class_n[i] = (y==i).float().sum()
    X_new = []
    Y_new = []
    select_n = torch.ones(n_classes)*min(class_n)
    select_list = random.shuffle([i for i in range(len(y))])
    print(select_list)
    #print(y.shape, select_list[i], y[select_list[i]])
    for i in range(len(x)):
        if select_n[y[select_list[i]]] > 0:
            X_new.append(x[select_list[i]])
            Y_new.append(y[select_list[i]])
            select_n[y[select_list[i]]] = select_n[y[select_list[i]]] - 1

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
        test_dl, test_ds, _ = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True)
    elif args.dataset == 'skin' or args.dataset == 'STL10' or args.dataset == 'fmnist':
        test_dl, test_ds, _ = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True, pre_sz=args.pre_sz, input_sz=args.input_sz)

    AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, n_classes=n_classes)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()

    return AUROC_avg, Accus_avg
def get_data_random_idx(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])     # set -> list
    return dict_users

def load_models( args,normalize):
    path_data_file = args.gen_path
    images_all = []
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.RandomCrop(args.imsize[0], padding=4, padding_mode='reflect'),
                                     transforms.ToTensor(),
                                     normalize])
    # 随机划分
    print("BUILDING DATASET")

    total = 0
    img_files = os.listdir(path_data_file)
    for path in img_files:
        img_path = os.path.join(path_data_file, path)
        img = Image.open(img_path)
        img = np.array(img)
        total = total + 1
        images_all.append(img)
    images_all = np.array(images_all)
    if args.dataset == 'fmnist':
        numbers = 20
    else:
        numbers = 20
    dict_users = get_data_random_idx(images_all, numbers)
    data_list = []
    for i in range(numbers):
        idxs = dict_users[i]  # 数据集中相应的数据的索引
        dataset = fileDataset(images_all[dict_users[i]],transform,dict_users[i])
        data_list.append(dataset)
    # =================== 保存数据分布 =================
    logger.info("---数据统计---")
    sum_data = 0
    for i in range(10):
        sum_data = sum_data +len(dict_users[i])
    logger.info("---sum add data: {} ----".format(sum_data))
    return data_list
if __name__ == '__main__':
    #weight_path = '/home/ubuntu/federated_semi_supervised_learning/RSCFed-main/'
    args = args_parser()
    channel_dict = {
        "cifar10": 3,
        "cifar100": 3,
        "SVHN": 3,
        "fmnist": 3,
        "STL10": 3,
        "skin": 3,
    }
    imsize_dict = {
        "cifar10": (32, 32),
        "cifar100": (32, 32),
        "SVHN": (32, 32),
        "fmnist": (28, 28),
        "STL10": (96, 96),
        "skin": (224, 224),
    }
    args.imsize = imsize_dict[args.dataset]
    args.channel = channel_dict[args.dataset]
    supervised_user_id = [0]
    unsupervised_user_id = list(range(len(supervised_user_id), args.unsup_num + len(supervised_user_id)))
    sup_num = len(supervised_user_id)
    unsup_num = len(unsupervised_user_id)
    total_num = sup_num + unsup_num

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(args.gpu)
    torch.cuda.set_device(0)
    time_current = 'attempt0'
    log_path = '.log'
    logging.basicConfig(filename=os.path.join(args.logdir, log_path), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    if args.deterministic:
        print('deterministic operation')
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    if not os.path.isdir('tensorboard'):
        os.mkdir('tensorboard')

    snapshot_path = 'model/'
    if not os.path.isdir(snapshot_path):
        os.mkdir(snapshot_path)
    if args.dataset == 'SVHN':
        snapshot_path = 'model/SVHN/'
    if args.dataset == 'cifar100':
        snapshot_path = 'model/cifar100/'
    if args.dataset == 'fmnist':
        args.dsa = False
    if args.dataset == 'skin':
        args.n_classes = 7
        snapshot_path = 'model/skin/'
    logger.info(str(args))
    print('==> Reloading data partitioning strategy..')
    n_classes = args.n_classes
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data_allnoniid(args,
        args.dataset, args.datadir, partition=args.partition, n_parties=total_num, beta=args.beta,n_classes=n_classes) #加了net_dataidx_map, traindata_cls_counts
    if args.dataset == 'SVHN' or args.dataset == 'STL10':
        X_train = X_train.transpose([0, 2, 3, 1])
        X_test = X_test.transpose([0, 2, 3, 1])

    torch.save({'X_train': X_train, 'y_train': y_train,'net_dataidx_map': net_dataidx_map}, 'noniid_10%labeled.pth')
    if args.dataset == 'SVHN' or args.dataset == 'cifar100' or args.dataset == 'cifar10':
        test_dl, test_ds, _ = get_dataloader(args, X_test, y_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True)
    elif args.dataset == 'skin' or args.dataset == 'STL10' or args.dataset == 'fmnist':
        test_dl, test_ds, _ = get_dataloader(args, X_test, y_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True, pre_sz=args.pre_sz, input_sz=args.input_sz)
    if args.model == 'Res18':
        net_glob = torch_models.resnet18(pretrained=args.Pretrained)
        net_glob.fc = nn.Linear(net_glob.fc.weight.shape[1], n_classes)
    # net_glob = get_network(args.model, args.channel, args.n_classes, args.imsize)

    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('warmup/SVHN.pth')

        net_glob.load_state_dict(checkpoint['state_dict'])
    else:
        start_epoch = 0

    if len(args.gpu.split(',')) > 1:
        net_glob = torch.nn.DataParallel(net_glob, device_ids=[i for i in range(round(len(args.gpu) / 2))])  #

    net_glob.train()
    w_glob = net_glob.state_dict()
    w_locals = []
    w_locals_trans = []
    w_ema_unsup = []
    lab_trainer_locals = []
    unlab_trainer_locals = []
    pl_trainer_locals = []
    sup_net_locals = []
    sup_net_locals_trans = []
    unsup_net_locals = []
    unsup_net_locals_trans = []
    pl_net_locals = []
    pl_net_locals_trans = []
    sup_optim_locals = []
    #sup_optim_locals_trans = []
    unsup_optim_locals = []
    pl_optim_locals = []
    pl_optim_locals_trans = []
    dist_scale_f = args.dist_scale
    total_lenth = sum([len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))])
    each_lenth = [len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))]
    client_freq = [len(net_dataidx_map[i]) / total_lenth for i in range(len(net_dataidx_map))]

    #load supervised trainer
    for i in supervised_user_id:
        lab_trainer_locals.append(SupervisedLocalUpdate(args, net_dataidx_map[i], n_classes))
        w_locals.append(copy.deepcopy(w_glob))
        w_locals_trans.append(copy.deepcopy(w_locals_trans))
        sup_net_locals.append(copy.deepcopy(net_glob))
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(sup_net_locals[i].parameters(), lr=args.base_lr,
                                         betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(sup_net_locals[i].parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
        elif args.opt == 'adamw':
            optimizer = torch.optim.AdamW(sup_net_locals[i].parameters(), lr=args.base_lr, weight_decay=0.02)
            optimizer_trans = torch.optim.AdamW(sup_net_locals_trans[i].parameters(), lr=args.base_lr, weight_decay=0.02)
        if args.resume:
            optimizer.load_state_dict(checkpoint['sup_optimizers'][i])
        sup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))
        #sup_optim_locals_trans.append(copy.deepcopy(optimizer_trans.state_dict()))

    # load pseudo labelling trainer
    for i in unsupervised_user_id:
        pl_trainer_locals.append(PLUpdate(args, net_dataidx_map[i], n_classes))
        w_locals.append(copy.deepcopy(w_glob))
        pl_net_locals.append(copy.deepcopy(net_glob))
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(pl_net_locals[i - sup_num].parameters(), lr=args.unsup_lr,
                                         betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(pl_net_locals[i - sup_num].parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
        elif args.opt == 'adamw':
            optimizer = torch.optim.AdamW(pl_net_locals[i - sup_num].parameters(), lr=args.unsup_lr,
                                          weight_decay=0.02)
            optimizer_trans = torch.optim.AdamW(pl_net_locals_trans[i - sup_num].parameters(), lr=0.03,
                                          weight_decay=0.02)
        pl_optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    sup_p = torch.zeros(n_classes)
    record_accuracy = []
    _, _, normalize = get_dataloader(args, X_train,
                                     y_train,
                                     args.dataset, args.datadir, args.batch_size,
                                     is_labeled=True,
                                     data_idxs=net_dataidx_map[0],
                                     pre_sz=args.pre_sz, input_sz=args.input_sz)
    images_train, labels_train = None, None
    #全监督
    for com_round in trange(201):
        print("************* Communication round %d begins *************" % com_round)
        w_l = []
        n_l = []
        for client_idx in supervised_user_id:
            sup_label = traindata_cls_counts[client_idx]
            loss_locals = []
            clt_this_comm_round = []
            w_per_meta = []
            local = lab_trainer_locals[client_idx]
            optimizer = sup_optim_locals[client_idx]
            train_dl_local, train_ds_local, _ = get_dataloader(args, X_train[net_dataidx_map[client_idx]],
                                                                       y_train[net_dataidx_map[client_idx]],
                                                                       args.dataset, args.datadir, args.batch_size,
                                                                       is_labeled=True,
                                                                       data_idxs=net_dataidx_map[client_idx],
                                                                       pre_sz=args.pre_sz, input_sz=args.input_sz)

            w, op, _,_,_ = local.train(args, sup_net_locals[client_idx].state_dict(),
                                               optimizer,train_dl_local, n_classes, X_test=X_test, y_test=y_test, res=True,stage=1, gdataloader=None)  # network, loss, optimizer
            w_l.append(w)
            n_l.append(len(net_dataidx_map[client_idx]))
            sup_optim_locals[client_idx] = copy.deepcopy(op)
        w = FedAvg(net_glob.state_dict(), w_l, n_l)
        net_glob.load_state_dict(w)
        if com_round%10==0:
            AUROC_avg, Accus_avg = test(args,com_round, net_glob.state_dict(), X_test, y_test, n_classes)
            print(AUROC_avg, Accus_avg)
            record_accuracy.append(Accus_avg)
            # print('adding lambda')
            print(record_accuracy)
        for i in supervised_user_id:
            sup_net_locals[i].load_state_dict(w)
    net_glob.load_state_dict(w)
    torch.save({'state_dict': net_glob.state_dict()}, 'net_glob.pth')
    # load supervised pretrained models
    state = torch.load('net_glob.pth')
    w = state['state_dict']
    record_w = copy.deepcopy(w)
    net_glob.load_state_dict(w)
    for i in supervised_user_id:
        sup_net_locals[i].load_state_dict(w)
    for i in unsupervised_user_id:
        pl_net_locals[i - sup_num].load_state_dict(w)
    record_accuracy = []
    predict_accuracy = []

    T_base = 0.84

    T_lower = 0.03

    T_higher = 0.1
    T_upper = 0.95
    all_local = []
    # load number of classes in labeled clients
    # sup_label = torch.load('partition_strategy/svhn_beta0.8_sup.pth')
    tensor_dict = {key: torch.tensor(value) for key, value in sup_label.items()}
    # 将字典的值（Tensor）转换为一个Tensor列表
    tensor_list = [tensor for tensor in tensor_dict.values()]
    tensor_list = torch.stack(tensor_list, dim=0)
    sup_label = tensor_list
    temp_sup_label = copy.deepcopy(tensor_list)
    temp_sup_label = (temp_sup_label / sum(temp_sup_label)) * (n_classes / 10)
    second_class = []
    second_h = []
    for i in range(len(temp_sup_label)):
        if temp_sup_label[i] < T_lower:
            second_class.append(i)
        if temp_sup_label[i] > T_higher:
            second_h.append(i)
    if min(temp_sup_label) < T_lower:
        include = True
    else:
        include = False
    avg_local_label = temp_sup_label
    class_confident = temp_sup_label + T_base - temp_sup_label.std()
    if args.dataset == 'skin' or args.dataset == 'SVHN':
        class_confident[class_confident >= 0.9] = 0.9
    else:
        class_confident[class_confident >= T_upper] = T_upper
    print(class_confident)
    sc = 10
    if args.dataset == 'cifar100':
        total_epoch = 1001
    else:
        total_epoch = 501
    data_list = load_models(args, normalize)
    dm_dl = data.DataLoader(dataset=data_list[total_num] + data_list[total_num + 1], batch_size=args.batch_size * 4,
                            drop_last=True, shuffle=True, num_workers=0)
    for com_round in trange(total_epoch):
        temp_p_a = []
        print("************* Communication round %d begins *************" % com_round)
        local_w = []
        local_num = []
        local_label = torch.zeros(n_classes)
        avg_max_mu_t = 0
        for client_idx in supervised_user_id:
            loss_locals = []
            clt_this_comm_round = []
            w_per_meta = []
            local = lab_trainer_locals[client_idx]
            optimizer = sup_optim_locals[client_idx]
            train_dl_local, train_ds_local, _ = get_dataloader(args, X_train[net_dataidx_map[client_idx]],
                                                            y_train[net_dataidx_map[client_idx]],
                                                            args.dataset, args.datadir, args.batch_size,
                                                            is_labeled=True,
                                                            data_idxs=net_dataidx_map[client_idx],
                                                            pre_sz=args.pre_sz, input_sz=args.input_sz)
            dl = data.DataLoader(dataset=data_list[client_idx], batch_size=args.batch_size * 4, drop_last=True,
                                 shuffle=True,num_workers=0)
            w, op, total, sup_label, confident = local.train(args, sup_net_locals[client_idx].state_dict(), optimizer,
                                      train_dl_local, n_classes, res=True,class_confident=class_confident,avg_local_label=avg_local_label, gdataloader=dl)
            local_w.append(w)
            sup_optim_locals[client_idx] = copy.deepcopy(op)
            if args.dataset == 'skin':
                local_num.append(len(net_dataidx_map[client_idx])*sc)#看不懂乘以10
            else:
                local_num.append(total)
            local_label = local_label + sup_label
        for client_idx in unsupervised_user_id:
            local = pl_trainer_locals[client_idx - sup_num]
            optimizer = pl_optim_locals[client_idx - sup_num]
            train_dl_local, train_ds_local, _ = get_dataloader(args, X_train[net_dataidx_map[client_idx]],
                                                            y_train[net_dataidx_map[client_idx]],
                                                            args.dataset, args.datadir, args.batch_size * 4,
                                                            is_labeled=False,
                                                            data_idxs=net_dataidx_map[client_idx],
                                                            pre_sz=args.pre_sz, input_sz=args.input_sz,data_list=data_list[client_idx])
            w, op, num, train_label, confident,prob_max_mu_t = local.train(args, pl_net_locals[client_idx - sup_num].state_dict(), optimizer,
                                      train_dl_local, n_classes, class_confident,avg_local_label, include, True)  # network, loss, optimizer
            avg_max_mu_t = avg_max_mu_t + prob_max_mu_t
            local_w.append(w)
            pl_optim_locals[client_idx - sup_num] = copy.deepcopy(op)
            local_num.append(num)
            local_label = local_label + train_label
        local_label = (local_label / sum(local_label)) * (n_classes / 10)
        avg_max_mu_t = avg_max_mu_t/len(unsupervised_user_id)
        second_class = []
        second_h = []
        for i in range(len(local_label)):
            if (local_label[i] < T_lower):
                second_class.append(i)
            if (local_label[i] > T_higher):
                second_h.append(i)
        print(local_label)

        if min(local_label) < T_lower:
            include = True
        else:
            include = False
        avg_local_label = local_label
        class_confident = local_label + avg_max_mu_t - local_label.std()
        class_confident[class_confident >= 1] = 1
        w = FedAvg(net_glob.state_dict(), local_w, local_num)
        if args.dataset == 'skin':
            if com_round>500 and com_round%5==0:
                print('res weight connection 5 epoch')
                w_l = [record_w, copy.deepcopy(w)]
                n_l = [1., 1.]
                print(n_l)
                w = FedAvg(record_w, w_l, n_l)
                record_w = copy.deepcopy(w)
        else:
            if com_round>500 and com_round%5==0:
                print('res weight connection 5 epoch')
                w_l = [record_w, copy.deepcopy(w)]
                n_l = [1., 1.]
                print(n_l)
                w = FedAvg(record_w, w_l, n_l)
                record_w = copy.deepcopy(w)
        net_glob.load_state_dict(w)
        if com_round % 1 == 0:
            AUROC_avg, Accus_avg = test(args, com_round, net_glob.state_dict(), X_test, y_test, n_classes)
            logger.info("AUROC_avg:{},Accus_avg:{}".format(AUROC_avg, Accus_avg))
            datadistill(args, net_glob, dm_dl,class_confident, dsa=True)
            AUROC_avg, Accus_avg = test(args, com_round, net_glob.state_dict(), X_test, y_test, n_classes)
            logger.info("AUROC_avg:{},Accus_avg:{}".format(AUROC_avg, Accus_avg))
            record_accuracy.append(Accus_avg)
            print(record_accuracy)
            print(max(record_accuracy))
        for i in supervised_user_id:
            sup_net_locals[i].load_state_dict(net_glob.state_dict())
        for i in unsupervised_user_id:
            pl_net_locals[i - sup_num].load_state_dict(net_glob.state_dict())

