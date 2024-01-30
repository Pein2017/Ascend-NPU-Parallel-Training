import time
from argparse import Namespace
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from train_utilis import create_meters, process_batch, update_meters

from optimizer import OptimizerManager
from tb_log_visualization import export_tb_log_to_figure


def train(train_loader: DataLoader,
          model: torch.nn.Module,
          criterion: torch.nn.Module,
          optimizer_manager: OptimizerManager,
          current_epoch: int,
          device: torch.device,
          args: Namespace,
          writer: SummaryWriter = None) -> None:
    """
    训练模型的函数。

    :param train_loader: 训练数据加载器。
    :param model: 训练模型。
    :param criterion: 损失函数。
    :param optimizer: 优化器。
    :param current_epoch: 当前训练周期。
    :param device: 计算设备。
    :param args: 训练参数。
    """

    meters, progress = create_meters(len(train_loader),
                                     f"Epoch:[{current_epoch}]")
    losses_meter, top1, top5, batch_processing_time, data_loading_time = meters

    model.train()  # 切换到训练模式

    end = time.time()
    for i, batch in enumerate(train_loader):
        data_loading_time.update(time.time() - end)
        loss, acc1, acc5 = process_batch(batch,
                                         model,
                                         criterion,
                                         device,
                                         is_training=True)

        update_meters([losses_meter, top1, top5], loss, acc1, acc5,
                      batch[0].size(0))

        optimizer_manager.zero_grad()
        loss.backward()
        optimizer_manager.step()

        batch_processing_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.gpu == 0:
            progress.display(i + 1)
            print('\n')

    # 同步设备间的度量数据
    losses_meter.all_reduce()
    top1.all_reduce()
    top5.all_reduce()
    # 显示训练过程的摘要
    if args.gpu == 0:
        progress.display_summary()
        if writer:
            writer.add_scalar('Loss/train', losses_meter.avg, current_epoch)
            writer.add_scalar('Top1/train', top1.avg, current_epoch)


def validate(val_loader: DataLoader,
             model: torch.nn.Module,
             criterion: torch.nn.Module,
             args: Namespace,
             current_epoch: int,
             writer: SummaryWriter = None) -> float:
    """
    在验证数据集上验证模型的性能。

    :param val_loader: 验证数据加载器。
    :param model: 验证模型。
    :param criterion: 损失函数。
    :param args: 验证过程中的参数。

    :return: 验证集上的平均准确率。
    """
    meters, progress = create_meters(len(val_loader), "Test:")
    batch_processing_time, _, losses_meter, top1, top5 = meters

    model.eval()  # 切换到评估模式
    end = time.time()

    for i, batch in enumerate(val_loader):

        loss, acc1, acc5 = process_batch(batch,
                                         model,
                                         criterion,
                                         args.device,
                                         is_training=False)
        update_meters([losses_meter, top1, top5], loss, acc1, acc5,
                      batch[0].size(0))

        batch_processing_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    if writer:
        writer.add_scalar('Loss/val', losses_meter.avg, current_epoch)
        writer.add_scalar('Top1/val', top1.avg, current_epoch)

    progress.display_summary()

    return top1.avg


def run_training_loop(args: Namespace,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      model: nn.Module,
                      criterion: nn.Module,
                      optimizer_manager: OptimizerManager,
                      device: torch.device,
                      save_checkpoint: Callable,
                      train_sampler: Optional[Sampler] = None,
                      ngpus_per_node: int = 1,
                      amp: Optional[object] = None) -> Tuple[float, int]:
    """
    运行训练循环。

    :param args: 命令行参数的命名空间。
    :param train_loader: 训练集的 DataLoader。
    :param val_loader: 验证集的 DataLoader。
    :param model: 要训练的模型。
    :param criterion: 损失函数。
    :param optimizer_manager: 管理优化器的对象。
    :param device: 训练使用的设备。
    :param save_checkpoint: 保存模型检查点的函数。
    :param train_sampler: 训练集的采样器，用于分布式训练。
    :param ngpus_per_node: 每个节点的 GPU 数量。
    :param amp: 自动混合精度对象，用于混合精度训练。

    :return: 最佳准确率及其对应的epoch（best_acc1, best_epoch）。
    """

    writer = None
    if args.gpu == 0 and getattr(args, 'tb_log_path', None) is not None:
        # 获取 args 中的属性，构建自定义后缀
        arch = getattr(args, 'arch', 'default_arch')
        world_size = getattr(args, 'world_size', 'default_world_size')
        batch_size = world_size * getattr(args, 'batch_size',
                                          'default_batch_size')
        lr = getattr(args, 'lr', 'default_lr')
        custom_suffix = f"{arch}-batch:{batch_size}-lr:{lr}"

        writer = SummaryWriter(log_dir=args.tb_log_path,
                               filename_suffix=custom_suffix)
        print('TensorBoard enabled at', args.tb_log_path)

    best_acc1 = 0
    best_epoch = args.start_epoch

    for current_epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(current_epoch)
        # 训练一个epoch
        train(train_loader,
              model,
              criterion,
              optimizer_manager,
              current_epoch,
              device,
              args,
              writer=writer)

        # 在全局rank为0的GPU上执行验证
        if args.gpu == 0:
            acc1 = validate(val_loader,
                            model,
                            criterion,
                            args,
                            current_epoch=current_epoch,
                            writer=writer)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            best_epoch = current_epoch if is_best else best_epoch

            # 在全局rank为0的GPU上保存检查点
            checkpoint = {
                'epoch': current_epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer_manager.optimizer.state_dict(),
            }
            if args.amp and amp is not None:
                checkpoint['amp'] = amp.state_dict()
            save_checkpoint(checkpoint, is_best)

    if writer:
        writer.close()

        # 定义保存图表的文件名
        fig_name = 'train_val_metrics.png'

        # 定义 TensorBoard 日志的自定义后缀
        custom_suffix

        # 调用 export_tb_log_to_figure 函数
        export_tb_log_to_figure(custom_suffix, fig_name, args.tb_log_path)

    return (best_acc1, best_epoch)
