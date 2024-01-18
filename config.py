import argparse
import torchvision.models as models

default_config = {
    'data': '/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/cifar10_data',
    'arch': 'resnet18',
    'workers': 4,
    'epochs': 20,
    'start_epoch': 1,
    'batch_size': 512,
    'lr': 0.015,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'print_freq': 5,
    'checkpoint_path':
    '/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/checkpoints',
    'resume': False,
    'evaluate': True,
    'pretrained': True,
    'world_size': 1,
    'rank': 0,
    'dist_url': 'tcp://192.168.10.31:23456',
    'dist_backend': 'hccl',
    'seed': 17,
    'gpu': None,
    'multiprocessing_distributed': True,
    'dummy': False,
    'device': 'npu',
    'addr': '192.168.10.31',
    'device_list': '0,1,2,3,4,5,6,7',
    'amp': True,
    'loss_scale': 1024.,
    'opt_level': 'O2',
}


def manually_get_parse(config: dict) -> argparse.ArgumentParser:
    """
    创建并配置命令行参数解析器。

    :param config: 包含默认配置的字典。
    :return: 配置好的解析器
    """
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')

    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(f'--{key}',
                                type=lambda x: (str(x).lower() == 'true'),
                                default=value,
                                help=f'{key} (default: {value})')
        elif value is None:
            parser.add_argument(f'--{key}',
                                default=value,
                                help=f'{key} (default: {value})')
        else:
            parser.add_argument(f'--{key}',
                                type=type(value),
                                default=value,
                                help=f'{key} (default: {value})')

    return parser


configured_parser = manually_get_parse(default_config)


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
