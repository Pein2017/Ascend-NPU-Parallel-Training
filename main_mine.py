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
# 获取当前文件所在目录的绝对路径
current_root_path = '/home/HW/Pein/DistributedTrain'

# 将当前文件所在目录的绝对路径加入到环境变量中
sys.path.append(current_root_path)
# flake8: noqa: E402
from config import get_parse_args, manually_get_parse
from train import train, validate, run_training_loop
from data import get_dataloaders
from model import CIFAR10Net, load_or_create_model
from utilis import (device_id_to_process_device_map, set_device,
                    init_distributed_training, save_checkpoint)

# parser = get_parse_args()
# 测试，先用manual parse
parser = manually_get_parse()
best_acc1 = 0


def main_worker(gpu: Optional[Union[str, int]], ngpus_per_node: int,
                args: Namespace):
    global best_acc1  # 假设这是一个全局变量，用于跟踪最佳精度

    args.gpu = args.process_device_map[gpu]

    if args.gpu is not None:
        print("Use GPU/NPU: {} for training".format(args.gpu))

    # 初始化分布式训练环境
    init_distributed_training(args, ngpus_per_node, gpu)

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
    optimizer = torch.optim.SGD(model.parameters(),
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
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu] if args.device == 'gpu' else None,
            broadcast_buffers=False)
    elif args.device == 'gpu':
        model = torch.nn.DataParallel(model, device_ids=None)
    else:
        raise ValueError('Only DistributedDataParallel is supported.')

    # 可选地从检查点恢复
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # 加载检查点，确保将数据加载到正确的设备上
            checkpoint = torch.load(args.resume, map_location=device)

            # 恢复训练的起始epoch和最佳精度
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            # 加载模型和优化器的状态
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            # 如果使用AMP，加载AMP的状态
            if args.amp:
                amp.load_state_dict(checkpoint['amp'])

            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            # 如果找不到检查点文件，打印警告信息
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 为了提高效率，启用cudnn的benchmark模式
    # 这会让cudnn自动寻找最适合当前配置的算法
    # （特别是对于固定输入大小的情况）
    cudnn.benchmark = True

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

    # 修改：仅在主进程（rank 0）中执行验证
    if args.evaluate and args.gpu == 0:
        print('Fianl validating at NPU:{}'.format(args.gpu))
        validate(val_loader, model, criterion, args)
        return


def main():
    # 解析命令行参数
    args = parser.parse_args()

    # 设置主节点的地址和端口
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '17'  # **为端口号，请根据实际选择一个闲置端口填写

    # 如果提供了种子参数，设置随机种子以确保结果的可重复性
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
        # 发出警告，因为确定性设置可能会减慢训练速度，并且从检查点重启时可能表现出不可预测的行为
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # 如果指定了特定的GPU，发出警告，因为这会禁用数据并行处理
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # 如果使用环境变量设置分布式URL，并且world_size为-1，则从环境变量中获取world_size
    print(args)
    if args.dist_url == "env://" and args.world_size == -1:
        # 这里的world_size未设定
        args.world_size = int(os.environ["WORLD_SIZE"])

    # 设置是否使用分布式处理
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    # 根据设备列表设置设备映射
    args.process_device_map = device_id_to_process_device_map(args.device_list)

    # 如果使用NPU，设置每个节点的NPU数量；否则，使用CUDA设备数量
    if args.device == 'npu':
        ngpus_per_node = len(args.process_device_map)
    else:
        ngpus_per_node = torch.cuda.device_count()

    # 如果启用了多进程分布式处理
    if args.multiprocessing_distributed:
        # 调整总的world_size
        args.world_size = ngpus_per_node * args.world_size
        # 使用torch.multiprocessing.spawn来启动分布式处理的进程
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # 如果不是多进程分布式处理，则直接调用main_worker函数
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    import time
    # 开始计时
    start_time = time.time()
    main()
    # 结束计时
    end_time = time.time()
    print('total time:{}'.format(end_time - start_time))
