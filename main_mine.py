'''逐行测试时启用'''
# import sys

# sys.path.append('/data/Pein/Pytorch/Ascend-NPU-Parallel-Training')

import os
import random
import warnings
from argparse import Namespace

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch_npu  # noqa: F401

# warnings.filterwarnings("ignore")

from config import configured_parser
from data_loader import get_dataloaders
from model import load_or_create_model
from utilis import (device_id_to_process_device_map, init_distributed_training)
from worker import main_worker


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
    download = False
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f'{args.dataset_name}_data')
    if not os.path.exists(data_path) or not os.listdir(data_path):
        os.makedirs(data_path, exist_ok=True)
        _ = get_dataloaders(data_path=data_path,
                            dataset_name=args.dataset_name,
                            batch_size=1,
                            distributed=False,
                            download=True)
        print(f"Data downloaded to '{data_path}'.")
    else:
        print(f"Data downloaded at '{data_path}'.")


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
        print('Loading model...')
        model = load_or_create_model(args)
        print(f'Model {args.arch} loaded')
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


if __name__ == '__main__':
    import time

    # 开始计时
    start_time = time.time()
    main()
    # 结束计时
    end_time = time.time()
    print('total time:{}'.format(end_time - start_time))
