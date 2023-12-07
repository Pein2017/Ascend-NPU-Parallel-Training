import argparse
import torchvision.models as models


def manually_get_parse() -> argparse.ArgumentParser:
    # 创建一个 ArgumentParser 实例
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR-10 Training with ResNet18')

    # 设置默认参数
    parser.set_defaults(
        data='/home/HW/Pein/cifar10_data/',  # 数据集路径
        arch='resnet34',  # 使用的网络架构
        workers=4,  # 数据加载时的工作进程数
        epochs=100,  # 训练的总轮次
        start_epoch=1,  # 起始轮次
        batch_size=512,  # 每批处理的样本数量
        lr=0.015,  # 学习率
        momentum=0.9,  # 动量
        weight_decay=1e-4,  # 权重衰减
        print_freq=50,  # 打印频率
        resume='',  # 恢复训练的模型路径
        evaluate=True,  # 是否在验证集上评估模型
        pretrained=True,  # 是否使用预训练模型
        world_size=1,  # 分布式训练的世界大小
        rank=0,  # 分布式训练的节点排名
        dist_url='tcp://192.168.10.31:23456',  # 分布式训练的URL
        dist_backend='hccl',  # 分布式训练使用的后端
        seed=17,  # 随机种子
        gpu=None,  # 使用的GPU ID
        multiprocessing_distributed=True,  # 是否使用多进程分布式训练
        dummy=False,  # 是否使用假数据进行基准测试
        device='npu',  # 使用的设备类型，如npu或gpu
        addr='192.168.10.31',  # 主节点地址
        device_list='0,1,2,3,4,5,6,7',  # 使用的设备列表
        amp=True,  # 是否使用自动混合精度
        loss_scale=1024.,  # 混合精度训练的损失缩放，
        opt_level='O2',
    )

    return parser


def get_parse_args() -> argparse.Namespace:
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch Cifar-10 Training')

    parser.add_argument('data',
                        metavar='DIR',
                        nargs='?',
                        default='imagenet',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a',
                        '--arch',
                        metavar='ARCH',
                        default='resnet18',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j',
                        '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs',
                        default=90,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument(
        '-b',
        '--batch-size',
        default=256,
        type=int,
        metavar='N',
        help='mini-batch size (default: 256), this is the total '
        'batch size of all GPUs on the current node when '
        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')
    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p',
                        '--print-freq',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e',
                        '--evaluate',
                        dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained',
                        dest='pretrained',
                        action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank',
                        default=-1,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://224.66.41.62:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend',
                        default='nccl',
                        type=str,
                        help='distributed backend')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument(
        '--multiprocessing-distributed',
        action='store_true',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs. This is the '
        'fastest way to use PyTorch for either single node or '
        'multi node data parallel training')
    parser.add_argument('--dummy',
                        action='store_true',
                        help="use fake data to benchmark")

    parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
    parser.add_argument('--addr',
                        default='127.0.0.1',
                        type=str,
                        help='master addr')
    parser.add_argument('--device_list',
                        default='0,1,2,3,4,5,6,7',
                        type=str,
                        help='device id list')
    parser.add_argument('--amp',
                        default=False,
                        action='store_true',
                        help='use amp to train the model')
    parser.add_argument(
        '--loss_scale',
        default=1024.,
        type=float,
        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--dist_backend',
                        default='hccl',
                        type=str,
                        help='distributed backend')
    parser.add_argument(
        '--opt-level',
        default='O2',
        type=str,
        help='loss scale using in amp, default -1 means dynamic')

    return parser.parse_args()
