from argparse import Namespace
from ast import Dict
from math import dist
from typing import Optional, Tuple, Union, Dict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from apex import amp
from data_loader import get_dataloaders
from model import load_or_create_model
from optimizer import CriterionManager, OptimizerManager
from train import run_training_loop, validate
from utilis import (init_distributed_training, load_checkpoint,
                    save_checkpoint, set_device)

# warnings.filterwarnings("ignore")


def main_worker(gpu: Optional[Union[str, int]], ngpus_per_node: int,
                model: Module, config: Dict) -> Tuple[float, int]:
    """
    主工作函数，用于初始化和执行分布式训练。

    :param gpu: 用于训练的 GPU/NPU 的标识符，可以是设备编号或名称。传入的是具体的 GPU/NPU 编号。
    :param ngpus_per_node: 每个节点上的 GPU 数量。
    :param model: 要进行训练的模型，是 torch.nn.Module 的一个实例。
    :param config: 包含训练配置的字典，包括分布式训练的设置等。
    :return: 训练过程中的最佳准确率和对应的 epoch（best_acc1, best_epoch）。
    """
    # 通过process_device_map将逻辑GPU映射到物理GPU
    process_device_map = config['distributed_training']['process_device_map']
    gpu = process_device_map.get(str(gpu), gpu)  # 确保gpu是字符串类型的键

    # 从配置中提取设备类型
    device_type = config['distributed_training']['device']

    # 检查设备类型是否支持，并打印相关信息
    if device_type not in ['gpu', 'npu']:
        print(
            'Set device to CPU. Note: CPU may not be supported for distributed training.'
        )
        device_type = 'cpu'  # 为了安全起见，将设备类型设为'cpu'

    # 调用set_device函数设置设备，并获取torch.device对象
    device = set_device(device_type, gpu)

    # 创建或加载模型
    model.device = device
    model.unfreeze()
    model.to_device()

    # 调整batch_size和workers
    batch_size = config['training']['batch_size']
    workers = config['training']['workers']
    batch_size = int(batch_size / ngpus_per_node)

    workers = int((workers + ngpus_per_node - 1) / ngpus_per_node)

    distributed = config['distributed_training']['distributed']
    world_size = config['distributed_training']['world_size']

    distributed_params = {
        'distributed':
        distributed,
        'dist_url':
        config['distributed_training']['dist_url'],
        'rank':
        config['distributed_training']['rank'],
        'dist_backend':
        config['distributed_training']['dist_backend'],
        'world_size':
        world_size,
        'multiprocessing_distributed':
        config['distributed_training']['multiprocessing_distributed']
    }

    init_distributed_training(**distributed_params,
                              ngpus_per_node=ngpus_per_node,
                              gpu=gpu)

    # 数据加载代码
    if config['data']['dummy']:
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        print("=> Dummy data is used!")
        # 使用假数据来模拟CIFAR-10的数据结构和大小
        train_dataset = datasets.FakeData(1000, (3, 32, 32), 10,
                                          transforms.ToTensor())
        val_dataset = datasets.FakeData(100, (3, 32, 32), 10,
                                        transforms.ToTensor())
        test_dataset = datasets.FakeData(100, (3, 32, 32), 10,
                                         transforms.ToTensor())

        # 根据是否进行分布式训练来创建采样器
        if distributed_params['distributed']:
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset,
                                             shuffle=False,
                                             drop_last=True)
        else:
            train_sampler = None
            val_sampler = None

        # 创建 DataLoader 实例
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=workers)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=workers)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=workers)
    else:
        # CIFAR数据集的实际目录
        data_path = config['data']['path']
        dataset_name = config['data']['dataset_name']
        num_workers = config['training']['workers']
        split_ratio = config['training']['split_ratio']
        # 获取数据加载器和采样器
        train_loader, val_loader, test_loader, train_sampler, val_sampler = get_dataloaders(
            data_path=data_path,
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
            split_ratio=split_ratio,
            distributed=distributed,
        )

    # 从config中提取优化器参数
    optimizer_params = {
        'lr':
        float(config['training']['lr']),
        'momentum':
        float(config['optimizer'].get('momentum', 0.9)),
        'weight_decay':
        float(config['optimizer']['weight_decay']),
        'betas':
        tuple(float(x) for x in config['optimizer'].get('betas', [0.9, 0.95])),
    }
    # 先初始化 OptimizerManager
    optimizer_manager = OptimizerManager(
        model.parameters(),
        optimizer_type=config['optimizer']['name'],
        optimizer_params=optimizer_params,
        patience=int(config['early_stopping']['patience']),
        early_stop_delta=float(config['early_stopping']['delta']),
    )

    # 尝试从检查点加载模型和优化器
    checkpoint_path = config['logging'].get('checkpoint_path', None)
    if checkpoint_path is not None:
        model_state_dict, optimizer_state_dict, best_epoch, best_acc1 = load_checkpoint(
            checkpoint_path)
        # 更新模型和优化器的状态
        if model_state_dict:
            model.load_state_dict(model_state_dict)
        if optimizer_state_dict:
            optimizer_manager.update_optimizer_state(optimizer_state_dict,
                                                     params_to_restore=['lr'])

    if 'scheduler' in config:
        scheduler_config = config['scheduler']
        # 明确转换为期望的类型
        scheduler_type = scheduler_config['type']
        scheduler_patience = int(scheduler_config.get('patience', 10))
        mode = scheduler_config.get('mode', 'min')
        factor = float(scheduler_config.get('factor', 0.5))

        optimizer_manager.create_scheduler(
            scheduler_type=scheduler_type,
            scheduler_patience=scheduler_patience,
            mode=mode,
            factor=factor,
        )

    amp_enabled = config['amp']['enabled']
    opt_level = config['amp']['opt_level']
    loss_scale = config['amp']['loss_scale']

    if amp_enabled:
        model, optimizer_manager.optimizer = amp.initialize(
            model,
            optimizer_manager.optimizer,
            opt_level=opt_level,
            loss_scale=loss_scale)

    # 对于NPU，False
    cudnn.benchmark = False
    verbose = config['training'].get('verbose', False)
    print_freq = config['training'].get('print_freq', 100)
    start_epoch = config['training'].get('start_epoch', 0)

    criterion_type = config['optimizer']['criterion']
    criterion_manager = CriterionManager(criterion_type)
    criterion_manager.to_device(device)
    criterion = criterion_manager.criterion

    debug_mode = config['training']['debug_mode']

    best_acc1, best_epoch = run_training_loop(
        start_epoch=start_epoch,
        epochs=config['training']['epochs'],
        distributed=distributed,
        world_size=world_size,
        batch_size=batch_size,
        arch=config['model']['arch'],
        lr=config['training']['lr'],
        optimizer_name=config['optimizer']['name'],
        tb_log_path=config['logging'].get('tb_log_path', None),
        checkpoint_folder=config['logging'].get('checkpoint_folder',
                                                './checkpoints'),
        amp_enabled=config['amp'].get('enabled', True),
        verbose=verbose,
        print_freq=print_freq,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer_manager=optimizer_manager,
        device=device,
        train_sampler=train_sampler,
        ngpus_per_node=ngpus_per_node,
        amp=amp,
        debug_mode=debug_mode)

    # 如果指定了评估，则执行验证过程并返回
    if config['evaluation']['evaluate'] and gpu == 0:
        print('\n')
        print('-' * 20)
        print(f'best acc1: {best_acc1:3f} at epoch {best_epoch}')
        print('\n')
        # print('Final validating at NPU:{}'.format(gpu))
        # validate(test_loader, model, criterion, args, current_epoch=best_epoch)
