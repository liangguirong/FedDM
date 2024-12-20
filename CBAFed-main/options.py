import argparse




def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'skin', 'SVHN', 'cifar100', 'fmnist', 'STL10'], default='cifar10',
                        help='dataset used for training')
    parser.add_argument('--model', type=str, default='Res18', help='neural network used in training')
    parser.add_argument('--unsup_num', type=int, default=9, help='number of unsupervised clients')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size per gpu')

    parser.add_argument('--lambda_u', type=float, default=0.02, help='start_epoch')
    parser.add_argument('--opt', type=str, default='sgd', help='sgd or adam or adamw')
    parser.add_argument("--save_dir", type=str, default="save", help="Path to save results")
    parser.add_argument('--base_lr', type=float, default=0.03,
                        help='maximum epoch number to train')  # adam:2e-4 sgd:2e-3 adamw:2e-3?
    parser.add_argument('--unsup_lr', type=float, default=0.02,
                        help='lr of unsupervised clients')
    parser.add_argument("--gen_path", type=str, default="/public/home/jd_yangfeng/lgr/FedDM/fed-ddim-main/exp/image_samples/cifar10",
                        help="Path of generated dataset")

    parser.add_argument("--max_grad_norm",
                        dest="max_grad_norm",
                        type=float,
                        default=5,
                        help="max gradient norm allowed (used for gradient clipping)")
    parser.add_argument('--rounds', type=int, default=1000, help='local epoch')
    parser.add_argument('--meta_round', type=int, default=3, help='number of sub-consensus groups')
    parser.add_argument('--w_mul_times', type=int, default=6,
                        help='times of amplifying weight of supervised model when aggregating')
    parser.add_argument('--sup_scale', type=float or int, default=100, help='scale factor for labeled clients when computing model distance')
    parser.add_argument('--dist_scale', type=float or int, default=1e4,
                        help='scale factor when computing model distance')
    parser.add_argument('--ema', type=float, default=0.999,
                        help='EMA')
    parser.add_argument('--local_pl_ep', type=int, default=10, help='local epoch')
    parser.add_argument('--n_classes', type=int, default=10, help='local epoch')
    parser.add_argument('--conf-threshold',
                        '--confidence-threshold',
                        dest="confidence_threshold",
                        default=0.9,
                        type=float,
                        help="confidence threshold for pair loss and unsupervised loss")
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    # 3407 114514
    parser.add_argument('--seed', type=int, default=114514, help='random seed')
    parser.add_argument('--Pretrained', type=bool, default=False, help='use imagenet pretrained model')

    parser.add_argument('--input_sz', type=int, default=32, help='actual input size')
    parser.add_argument('--pre_sz', type=int, default=40, help='image size for pre-processing')
    # skin 240 224

    parser.add_argument('--root_path', type=str, default='D:/Deeplearning/FedIRM-main/data/HAM10000_images_part_1/',
                        help='dataset root dir')
    parser.add_argument('--csv_file_train', type=str, default='data/skin_split/train.csv',
                        help='training set csv file')
    parser.add_argument('--csv_file_val', type=str, default='data/skin_split/validation.csv',
                        help='validation set csv file')
    parser.add_argument('--csv_file_test', type=str, default='data/skin_split/test.csv', help='testing set csv file')
    parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
    parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--local_ep', type=int, default=1, help='local epoch')
    parser.add_argument('--num_users', type=int, default=10, help='local epoch')
    parser.add_argument('--num_labeled', type=int, default=1, help='local epoch')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--logdir', type=str, default='logs/', help='The log file name')
    parser.add_argument('--beta', type=float, default=0.8,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    ### tune
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch')
    #parser.add_argument('--weight_decay', dest="weight_decay", default=0.02, type=float, help='weight decay')

    ### costs
    parser.add_argument('--label_uncertainty', type=str, default='U-Ones', help='label type')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema_decay')
    parser.add_argument('--consistency', type=float, default=1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=30, help='consistency_rampup')

    ### unlabeled client training parameters
    parser.add_argument('--num-warmup-epochs',
                        '--num-warm-up-epochs',
                        dest="num_warmup_epochs",
                        default=0,
                        type=int,
                        help='number of warm-up epochs for unsupervised loss ramp-up during training'
                             'set to 0 to disable ramp-up')

    parser.add_argument('--lr-step-size',
                        '--learning-rate-step-size',
                        dest="lr_step_size",
                        default=5,
                        type=int,
                        help='step size for step learning rate decay')

    ### unsupervised loss

    parser.add_argument('--test', action='store_true', help='resume from checkpoint')

    ### meta
    parser.add_argument('--meta_client_num', type=int, default=5, help='number of clients in each sub-consensus group')
    parser.add_argument('--from_labeled', action='store_true',
                        help='whether the warm-up checkpoint is trained only on labeled client')
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--ipc', type=int, default=20, help='image(s) per class')
    args = parser.parse_args()

    args.dsa = True
    return args
