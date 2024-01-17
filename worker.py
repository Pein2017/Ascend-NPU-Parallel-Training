import os
from typing import Optional, Union, Namespace

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from apex import amp

from model import load_or_create_model
from data_loader import get_dataloaders
from train import validate, run_training_loop
from utilis import init_distributed_training, set_device, save_checkpoint, load_checkpoint


def main_worker(gpu: Optional[Union[str, int]], ngpus_per_node: int,
                args: Namespace):
    """
    主工作函数，用于初始化和执行分布式训练。

    :param gpu: 用于训练的 GPU/NPU 的标识符，可以是设备编号或名称。
    :param ngpus_per_node: 每个节点上的 GPU 数量。
    :param args: 包含训练配置和参数的命名空间。

    全局变量:
    best_acc1: 用于跟踪目前为止的最佳精度。
    """

    global best_acc1  # 全局变量，用于跟踪最佳精度

    args.gpu = args.process_device_map[gpu]

    if args.gpu is not None:
        print("Use GPU/NPU: {} for training".format(args.gpu))

    # 设置设备
    device = set_device(args)

    # 创建或加载模型
    model = load_or_create_model(args)
    model.to(device)

    # 调整batch_size和workers
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    # 数据加载代码
    if args.dummy:
        print("=> Dummy data is used!")
        # 使用假数据来模拟CIFAR-10的数据结构和大小
        train_dataset = datasets.FakeData(50000, (3, 32, 32), 10,
                                          transforms.ToTensor())
        val_dataset = datasets.FakeData(10000, (3, 32, 32), 10,
                                        transforms.ToTensor())
    else:
        # CIFAR-10数据集的实际目录
        data_path = args.data  # 例如：'/home/HW/Pein/cifar10_data/cifar-10-batches-py'
        batch_size = args.batch_size

        # 获取数据加载器和采样器
        train_loader, val_loader, test_loader, train_sampler, val_sampler = get_dataloaders(
            data_path, args.batch_size, distributed=args.distributed)

    # 定义损失函数（标准）和优化器
    criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数也移到相应设备
    optimizer = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # 如果使用自动混合精度（AMP）
    if args.amp:
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args.opt_level,
                                          loss_scale=args.loss_scale)

    # 如果使用分布式数据并行
    if args.distributed:
        init_distributed_training(args, ngpus_per_node, gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu] if args.device == 'gpu' else None,
            broadcast_buffers=False)
    elif args.device == 'gpu':
        model = torch.nn.DataParallel(model, device_ids=None)
    else:
        raise ValueError('Only DistributedDataParallel is supported.')

    if args.resume:
        model, optimizer, args.start_epoch, best_acc1 = load_checkpoint(
            args.resume, model, optimizer, args, device)

    # 对于NPU，False
    cudnn.benchmark = False

    # 保存检查点的函数
    save_checkpoint_fn = lambda checkpoint, is_best: save_checkpoint(
        checkpoint, is_best)

    # 运行训练循环
    run_training_loop(args,
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
        print('Fianl validating at NPU:{}'.format(args.gpu))
        validate(val_loader, model, criterion, args)
        return
