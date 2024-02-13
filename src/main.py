'''逐行测试时启用'''
# import sys

# sys.path.append('/data/Pein/Pytorch/Ascend-NPU-Parallel-Training')

import os
import random
import warnings
from typing import Dict

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch_npu  # noqa: F401

# warnings.filterwarnings("ignore")

from config import config_from_yaml as config
from data_loader import get_dataloaders
from model import load_or_create_model
from utilis import device_id_to_process_device_map
from worker import main_worker


def setup_deterministic_mode(seed: int) -> None:
    """设置确定性训练模式"""
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    warnings.warn('Deterministic mode can slow down training.')


def setup_environment(config: Dict) -> None:
    """配置环境变量和随机种子"""

    os.environ['MASTER_ADDR'] = config['distributed_training']['addr']
    os.environ['MASTER_PORT'] = str(config['distributed_training']['port'])

    seed = config['training'].get('seed', None)
    if seed is not None:
        setup_deterministic_mode(seed)


def verify_and_download_dataset(config: Dict) -> None:
    """验证数据集是否存在，若不存在则下载"""
    dataset_name = config['data']['dataset_name']
    data_path = config['data'].get('path', None)
    if data_path is None:

        # 数据下载到上一级".."目录下
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..")
        data_path = os.path.join(base_path, f'{dataset_name}-data')

        os.makedirs(data_path, exist_ok=True)
        print(f"Data is downloading to '{data_path}'.")
        _ = get_dataloaders(data_path=data_path,
                            dataset_name=dataset_name,
                            batch_size=1,
                            distributed=False,
                            download=True)
    elif data_path is not None:
        if not os.path.exists(data_path) or not os.listdir(data_path):
            raise FileNotFoundError(
                f"Data:{dataset_name} was not found at '{data_path}'.")
    else:
        print(f"Data downloaded at '{data_path}'.")


def start_worker(config: Dict) -> None:
    """执行训练流程"""

    # 设置分布式URL和world_size
    dist_url = config['distributed_training']['addr']
    world_size = config['distributed_training']['world_size']
    multiprocessing_distributed = config['distributed_training'].get(
        'multiprocessing_distributed', False)

    device = config['distributed_training']['device']
    device_list = config['distributed_training']['device_list']

    if dist_url == "env://" and world_size == -1:
        world_size = int(os.environ["WORLD_SIZE"])

    # 确定是否启用分布式处理
    distributed = world_size > 1 or multiprocessing_distributed

    # 设备映射
    process_device_map = device_id_to_process_device_map(device_list)
    # 加入到config里
    config['distributed_training']['process_device_map'] = process_device_map

    # 确定每个节点的设备数量
    ngpus_per_node = len(
        process_device_map) if device == 'npu' else torch.cuda.device_count()

    # 处理多进程分布式训练
    if multiprocessing_distributed:
        world_size *= ngpus_per_node
        config['distributed_training']['world_size'] = world_size
        print('Loading model...')
        arch = config['model']['arch']
        pretrained = config['model']['pretrained']
        dataset_name = config['data']['dataset_name']

        # NOTE:为了确保pre-trained模型参数已经被下载
        model = load_or_create_model(
            arch=arch,
            dataset_name=dataset_name,
            pretrained=pretrained,
        )
        print(f'Model {arch} loaded')

        mp.spawn(
            main_worker,
            args=(ngpus_per_node, model, config),
            nprocs=ngpus_per_node,
        )
        print('finished mp.spawn ')
    else:
        raise Exception('Not support single process training now!')


def main():

    seed = config['training'].get('seed', None)
    setup_environment(config)
    verify_and_download_dataset(config)

    start_worker(config)


if __name__ == '__main__':
    import time

    # 开始计时
    start_time = time.time()
    main()
    # 结束计时
    end_time = time.time()
    # 将时间转换为小时和分钟的形式,如，1小时，30分钟
    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    print("Total time cost：%02dh:%02dmins:%02ds" % (h, m, s))
