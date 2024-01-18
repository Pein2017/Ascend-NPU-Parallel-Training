import torch
import torch_npu  # noqa: F401
from apex import amp
import os
import random
import warnings
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
from typing import Optional, Union
from argparse import Namespace

from worker import main_worker
from config import configured_parser
from train import train, validate, run_training_loop
from data_loader import get_dataloaders
from model import CIFAR10Net, load_or_create_model
from utilis import (device_id_to_process_device_map, set_device,
                    init_distributed_training, save_checkpoint)


def setup_deterministic_mode(seed: int) -> None:
    """设置确定性训练模式"""
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    warnings.warn('Deterministic mode can slow down training.')


def setup_environment(args: Namespace) -> None:
    """配置环境变量和随机种子"""
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '17'  # 选择一个闲置端口

    if args.seed is not None:
        setup_deterministic_mode(args.seed)


def verify_and_download_dataset(args: Namespace) -> None:
    """验证数据集是否存在，若不存在则下载"""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'cifar10_data')
    if not os.path.exists(data_path) or not os.listdir(data_path):
        os.makedirs(data_path, exist_ok=True)
        _ = get_dataloaders(data_path=data_path,
                            batch_size=1,
                            distributed=False,
                            download=True)
        print(f"Data downloaded to '{data_path}'.")


def run_training(args: Namespace) -> None:
    """执行训练流程"""

    # 设置分布式URL和world_size
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # 确定是否启用分布式处理
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # 设备映射
    args.process_device_map = device_id_to_process_device_map(args.device_list)

    # 确定每个节点的设备数量
    ngpus_per_node = len(
        args.process_device_map
    ) if args.device == 'npu' else torch.cuda.device_count()

    # 处理多进程分布式训练
    if args.multiprocessing_distributed:
        args.world_size *= ngpus_per_node
        print('loading model...')
        model = load_or_create_model(args)
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        raise Exception('Not support single process training now!')
        main_worker(args.gpu, ngpus_per_node, args)


def main():
    args = configured_parser.parse_args()
    setup_environment(args)
    verify_and_download_dataset(args)
    run_training(args)


# def main():

#     best_acc1 = 0

#     # 确保数据集已经下载
#     current_script_path = os.path.abspath(__file__)
#     # 获取当前脚本所在的目录
#     current_script_dir = os.path.dirname(current_script_path)
#     # 设置数据集存储路径为该目录下的子目录 'cifar10_data'
#     data_path = os.path.join(current_script_dir, 'cifar10_data')

#     # 检查数据路径是否存在且不为空
#     if os.path.exists(data_path) and os.listdir(data_path):
#         print(
#             f"Dataset '{data_path}' already exists and is not empty. Skipping download."
#         )
#     else:
#         # 确保数据路径存在
#         os.makedirs(data_path, exist_ok=True)
#         # 执行get_dataloaders
#         _ = get_dataloaders(data_path=data_path,
#                             batch_size=1,
#                             distributed=False,
#                             download=True)
#         print(f"Data downloaded to '{data_path}'.")
#     # 解析命令行参数
#     args = configured_parser.parse_args()

#     # 设置主节点的地址和端口
#     os.environ['MASTER_ADDR'] = args.addr
#     os.environ['MASTER_PORT'] = '17'  # **为端口号，请根据实际选择一个闲置端口填写

#     # 如果提供了种子参数，设置随机种子以确保结果的可重复性
#     if args.seed is not None:
#         random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         cudnn.deterministic = True
#         cudnn.benchmark = True
#         # 发出警告，因为确定性设置可能会减慢训练速度，并且从检查点重启时可能表现出不可预测的行为
#         warnings.warn('You have chosen to seed training. '
#                       'This will turn on the CUDNN deterministic setting, '
#                       'which can slow down your training considerably! '
#                       'You may see unexpected behavior when restarting '
#                       'from checkpoints.')

#     # 如果指定了特定的GPU，发出警告，因为这会禁用数据并行处理
#     if args.gpu is not None:
#         warnings.warn('You have chosen a specific GPU. This will completely '
#                       'disable data parallelism.')

#     # 如果使用环境变量设置分布式URL，并且world_size为-1，则从环境变量中获取world_size
#     print(args)
#     if args.dist_url == "env://" and args.world_size == -1:
#         # 这里的world_size未设定
#         args.world_size = int(os.environ["WORLD_SIZE"])

#     # 设置是否使用分布式处理
#     args.distributed = args.world_size > 1 or args.multiprocessing_distributed
#     # 根据设备列表设置设备映射
#     args.process_device_map = device_id_to_process_device_map(args.device_list)

#     # 如果使用NPU，设置每个节点的NPU数量；否则，使用CUDA设备数量
#     if args.device == 'npu':
#         ngpus_per_node = len(args.process_device_map)
#     else:
#         ngpus_per_node = torch.cuda.device_count()

#     # 如果启用了多进程分布式处理
#     if args.multiprocessing_distributed:
#         # 调整总的world_size
#         args.world_size = ngpus_per_node * args.world_size
#         print('world_size:{}'.format(args.world_size))
#         # 使用torch.multiprocessing.spawn来启动分布式处理的进程
#         mp.spawn(main_worker,
#                  nprocs=ngpus_per_node,
#                  args=(ngpus_per_node, args))
#     else:
#         # 如果不是多进程分布式处理，则直接调用main_worker函数
#         main_worker(args.gpu, ngpus_per_node, args)

if __name__ == '__main__':
    import time
    # 开始计时
    start_time = time.time()
    main()
    # 结束计时
    end_time = time.time()
    print('total time:{}'.format(end_time - start_time))
