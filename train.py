import time
import torch
from utilis import AverageMeter, ProgressMeter, accuracy
from argparse import Namespace
from typing import Optional, Callable
from torch.utils.data import DataLoader, Sampler
from torch import nn, optim


def train(train_loader: DataLoader, model: torch.nn.Module,
          criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
          epoch: int, device: torch.device, args: Namespace) -> None:
    """
    训练模型的函数。

    参数:
    train_loader (DataLoader): 训练数据加载器。
    model (torch.nn.Module): 训练模型。
    criterion (torch.nn.Module): 损失函数。
    optimizer (torch.optim.Optimizer): 优化器。
    epoch (int): 当前训练周期。
    device (torch.device): 计算设备。
    args (argparse.Namespace): 训练过程中的参数。
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('损失', ':.4e')
    top1 = AverageMeter('准确率@1', ':6.2f')
    top5 = AverageMeter('准确率@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()  # 切换到训练模式

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader: DataLoader, model: torch.nn.Module,
             criterion: torch.nn.Module, args: Namespace) -> float:
    """
    在验证数据集上验证模型的性能。

    参数:
    val_loader (DataLoader): 验证数据加载器。
    model (torch.nn.Module): 验证模型。
    criterion (torch.nn.Module): 损失函数。
    args (argparse.Namespace): 验证过程中的参数。

    返回:
    float: 验证集上的平均准确率。
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    model.eval()  # 切换到评估模式

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)

    progress.display_summary()

    return top1.avg


def run_training_loop(args: Namespace,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      model: nn.Module,
                      criterion: nn.Module,
                      optimizer: optim.Optimizer,
                      device: torch.device,
                      save_checkpoint: Callable,
                      train_sampler: Optional[Sampler] = None,
                      ngpus_per_node: int = 1,
                      amp: Optional[object] = None) -> None:
    """
    运行训练循环。

    参数:
    args (Namespace): 命令行参数的命名空间。
    train_loader (DataLoader): 训练集的 DataLoader。
    val_loader (DataLoader): 验证集的 DataLoader。
    model (nn.Module): 要训练的模型。
    criterion (nn.Module): 损失函数。
    optimizer (optim.Optimizer): 优化器。
    device (torch.device): 训练使用的设备。
    save_checkpoint (Callable): 保存模型检查点的函数。
    train_sampler (Optional[Sampler]): 训练集的采样器，用于分布式训练。
    ngpus_per_node (int): 每个节点的 GPU 数量。
    amp (Optional[object]): 自动混合精度对象，用于混合精度训练。

    返回:
    None
    """

    best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # 训练一个epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)
        print(f'trained on epoch on npu:{args.gpu} \n')
        # 在全局rank为0的GPU上执行验证
        if args.gpu == 0:
            acc1 = validate(val_loader, model, criterion, args)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            # 在全局rank为0的GPU上保存检查点
            print(f'saving checkpoint at GPU:{args.gpu} \n ')
            checkpoint = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }
            if args.amp and amp is not None:
                checkpoint['amp'] = amp.state_dict()
            save_checkpoint(checkpoint, is_best)
