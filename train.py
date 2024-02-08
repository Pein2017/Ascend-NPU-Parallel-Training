from logging import raiseExceptions
import time
import os
from argparse import Namespace
from typing import Callable, Optional, Tuple

from tqdm import tqdm
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from train_utilis import create_meters, process_batch, update_meters, broadcast_early_stop

from utilis import save_checkpoint
from optimizer import OptimizerManager
from tb_log_visualization import TBLogExporter


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

        if args.verbose:
            if i % args.print_freq == 0 and args.gpu == 0:
                progress.display(i + 1)
                print('\n')

    # 同步设备间的度量数据
    losses_meter.all_reduce()
    top1.all_reduce()
    top5.all_reduce()
    # 显示训练过程的摘要
    if args.gpu == 0:
        if args.verbose:
            progress.display_summary()
        if writer:
            current_lr = optimizer_manager.optimizer.param_groups[0]['lr']
            writer.add_scalar('Loss/train', losses_meter.avg, current_epoch)
            writer.add_scalar('Top1/train', top1.avg, current_epoch)
            writer.add_scalar('Learning_Rate', current_lr, current_epoch)


def validate(val_loader: DataLoader,
             model: torch.nn.Module,
             criterion: torch.nn.Module,
             args: Namespace,
             current_epoch: int,
             writer: SummaryWriter = None) -> Tuple[float, float]:
    """
    在验证数据集上验证模型的性能。

    :param val_loader: 验证数据加载器。
    :param model: 验证模型。
    :param criterion: 损失函数。
    :param args: 验证过程中的参数。
    :param current_epoch: 当前的训练周期。
    :param writer: 用于TensorBoard日志记录的SummaryWriter实例。
    :return: 验证集上的平均准确率和平均损失值。
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

        if args.verbose and i % args.print_freq == 0:
            progress.display(i + 1)

    if writer:
        writer.add_scalar('Loss/val', losses_meter.avg, current_epoch)
        writer.add_scalar('Top1/val', top1.avg, current_epoch)

    if args.verbose:
        progress.display_summary()

    return top1.avg, losses_meter.avg


def run_training_loop(args: Namespace,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      model: nn.Module,
                      criterion: nn.Module,
                      optimizer_manager: OptimizerManager,
                      device: torch.device,
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
    :param train_sampler: 训练集的采样器，用于分布式训练。
    :param ngpus_per_node: 每个节点的 GPU 数量。
    :param amp: 自动混合精度对象，用于混合精度训练。

    :return: 最佳准确率及其对应的epoch（best_acc1, best_epoch）。
    """

    writer = None
    '''在全局rank为0的GPU上初始化SummaryWriter'''
    if args.gpu == 0:
        world_size = getattr(args, 'world_size', 'default_world_size')
        batch_size = world_size * getattr(args, 'batch_size',
                                          'default_batch_size')
        # 检查 tb_log_path 是否非空
        if getattr(args, 'tb_log_path', None) is not None:
            # 继续获取其它属性，用于构建自定义后缀
            arch = getattr(args, 'arch', 'default_arch')
            lr = getattr(args, 'lr', 'default_lr')
            optimizer_name = getattr(args, 'optimizer_name',
                                     'default_optimizer_name')
            tb_event_path = os.path.join(getattr(args, 'tb_log_path'),
                                         'events')
            # 构建自定义后缀
            # NOTE: 这里添加了'-'用于区分log_event默认的保存格式。
            custom_suffix = f"-{arch}-batch:{batch_size}-lr:{lr}-{optimizer_name}"

            # 初始化 SummaryWriter
            save_event_path = os.path.join(args.tb_log_path, 'events')
            writer = SummaryWriter(log_dir=save_event_path,
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

        early_stop_decision = False
        is_best = False
        # 在全局rank为0的GPU上执行验证和保存模型的逻辑
        if args.gpu == 0:
            acc1, val_loss = validate(val_loader,
                                      model,
                                      criterion,
                                      args,
                                      current_epoch=current_epoch,
                                      writer=writer)
            if args.scheduler:
                optimizer_manager.scheduler_step(val_loss)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1) if is_best else best_acc1
            best_epoch = current_epoch if is_best else best_epoch

            # 检查早停
            optimizer_manager.check_early_stopping(val_loss)
            if optimizer_manager.early_stop:
                print("Early stopping triggered")
                early_stop_decision = True

            # 保存检查点
            checkpoint = {
                'best_epoch': best_epoch,
                'best_acc1': best_acc1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer_manager.optimizer.state_dict(),
                'scheduler': optimizer_manager.scheduler.state_dict(),
            }

            if args.amp and amp is not None:
                checkpoint['amp'] = amp.state_dict()

            checkpoint_folder = args.checkpoint_folder
            check_point_suffix = custom_suffix[1::]

            save_checkpoint(checkpoint,
                            is_best,
                            checkpoint_folder=checkpoint_folder,
                            check_point_suffix=check_point_suffix)

        # 广播早停决策给所有进程
        if args.distributed:
            early_stop_tensor = torch.tensor([int(early_stop_decision)],
                                             dtype=torch.int,
                                             device=device)
            dist.broadcast(early_stop_tensor, src=0)
            early_stop_decision = bool(early_stop_tensor.item())

        if early_stop_decision:
            print("Early stopping triggered across all processes." if args.
                  gpu == 0 else "")
            break

    if writer:
        writer.close()

        # 定义保存图表的文件名
        # NOTE: 最终文件命名为{custom_suffix}-{fig_name}
        fig_name = 'metrics.png'

        if not custom_suffix:
            raiseExceptions(
                'should specify custom_suffix when using TensorBoard')

        # 实例化 TBLogExporter 类
        exporter = TBLogExporter(tb_log_path=args.tb_log_path,
                                 custom_suffix=custom_suffix[1:])
        # 定义想要绘制的指标
        grouped_metrics = {
            'Loss': ['Loss/train', 'Loss/val'],
            'Top1': ['Top1/train', 'Top1/val'],
            'Learning Rate': ['Learning_Rate'],
        }

        # 调用 export 方法，传入指标和图表文件名
        exporter.export(grouped_metrics=grouped_metrics, fig_name=fig_name)

    return (best_acc1, best_epoch)
