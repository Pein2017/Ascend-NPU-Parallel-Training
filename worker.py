from argparse import Namespace
from typing import Optional, Union, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from apex import amp
from data_loader import get_dataloaders
from model import load_or_create_model
from train import run_training_loop, validate
from utilis import (init_distributed_training, load_checkpoint,
                    save_checkpoint, set_device)

# warnings.filterwarnings("ignore")


def main_worker(gpu: Optional[Union[str, int]], ngpus_per_node: int,
                args: Namespace) -> Tuple[float, int]:
    """
    主工作函数，用于初始化和执行分布式训练。

    :param gpu: 用于训练的 GPU/NPU 的标识符，可以是设备编号或名称。
    :param ngpus_per_node: 每个节点上的 GPU 数量。
    :param args: 包含训练配置和参数的命名空间。
    :return: 训练过程中的最佳准确率和对应的 epoch（best_acc1, best_epoch）。
    """

    args.gpu = args.process_device_map[gpu]

    if args.gpu is not None:
        print("Use GPU/NPU: {} for training".format(args.gpu))

    # 设置设备

    device = set_device(args)
    print('after set_device')  ## TODO: delete

    # 创建或加载模型

    model = load_or_create_model(args, device)

    model.to_device()

    # 调整batch_size和workers
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    init_distributed_training(args, ngpus_per_node, gpu)

    # 数据加载代码
    if args.dummy:
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
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset,
                                             shuffle=False,
                                             drop_last=True)
        else:
            train_sampler = None
            val_sampler = None

        # 创建 DataLoader 实例
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.workers)
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.workers)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.workers)

    else:
        # CIFAR数据集的实际目录
        data_path = args.data_path
        dataset_name = args.dataset_name
        batch_size = args.batch_size
        num_workers = args.workers
        if_distributed = args.distributed
        split_ratio = args.split_ratio
        # 获取数据加载器和采样器
        train_loader, val_loader, test_loader, train_sampler, val_sampler = get_dataloaders(
            data_path=data_path,
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
            split_ratio=split_ratio,
            distributed=if_distributed,
        )

    # 定义损失函数（标准）和优化器
    criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数也移到相应设备
    # optimizer = optim.Adam(model.parameters(),
    #                        lr=args.lr,
    #                        weight_decay=args.weight_decay)

    optimizer = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optimizer = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                  mode='min',
    #                                                  factor=0.1,
    #                                                  patience=10,
    #                                                  verbose=True)

    # 如果使用自动混合精度（AMP）
    if args.amp:
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args.opt_level,
                                          loss_scale=args.loss_scale)

    # 如果使用分布式数据并行
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu] if args.device == 'gpu' else None,
            broadcast_buffers=False)
    elif args.device == 'gpu':
        print('nonono')
        model = torch.nn.DataParallel(model, device_ids=None)
    else:
        raise ValueError('Only DistributedDataParallel is supported.')

    if args.resume:
        model, optimizer, args.start_epoch, best_acc1 = load_checkpoint(
            args.resume, model, optimizer, args, device)

    # 对于NPU，False
    cudnn.benchmark = False

    # 保存检查点的函数
    checkpoint_folder = args.checkpoint_path

    # 定义checkpoint保存函数
    save_checkpoint_fn = lambda checkpoint, is_best: save_checkpoint(
        checkpoint, is_best, checkpoint_folder)

    # 运行训练循环

    best_acc1, best_epoch = run_training_loop(args,
                                              train_loader,
                                              val_loader,
                                              model,
                                              criterion,
                                              optimizer,
                                              device,
                                              save_checkpoint_fn,
                                              train_sampler,
                                              ngpus_per_node=ngpus_per_node,
                                              amp=amp)

    # 如果指定了评估，则执行验证过程并返回
    if args.evaluate and args.gpu == 0:
        print('\n')
        print('-' * 20)
        print(f'best acc1: {best_acc1:3f} at epoch {best_epoch}')
        print('\n')
        print('Final validating at NPU:{}'.format(args.gpu))
        validate(test_loader, model, criterion, args)
