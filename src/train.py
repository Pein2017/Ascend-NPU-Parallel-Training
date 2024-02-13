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


def train(
    train_loader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer_manager: OptimizerManager,
    current_epoch: int,
    gpu: int,
    device: torch.device,
    verbose: bool = False,
    print_freq: int = 100,
    writer: SummaryWriter = None,
) -> None:
    """
    训练模型的函数。

    :param train_loader: 训练数据加载器。
    :param model: 训练模型。
    :param criterion: 损失函数。
    :param optimizer_manager: 包含优化器的管理器。
    :param current_epoch: 当前训练周期。
    :param gpu: 当前使用的GPU索引。
    :param device: 计算设备。
    :param verbose: 是否在训练过程中打印详细信息。
    :param print_freq: 打印信息的频率（以批次为单位）。
    :param writer: TensorBoard的SummaryWriter实例，用于日志记录。
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

        if verbose and i % print_freq == 0 and gpu == 0:
            progress.display(i + 1)
            print('\n')

    # 同步设备间的度量数据
    losses_meter.all_reduce()
    top1.all_reduce()
    top5.all_reduce()

    # 显示训练过程的摘要
    if gpu == 0:
        if verbose:
            progress.display_summary()
        if writer:
            current_lr = optimizer_manager.optimizer.param_groups[0]['lr']
            writer.add_scalar('Loss/train', losses_meter.avg, current_epoch)
            writer.add_scalar('Top1/train', top1.avg, current_epoch)
            writer.add_scalar('Learning_Rate', current_lr, current_epoch)


def validate(val_loader: DataLoader,
             model: torch.nn.Module,
             criterion: torch.nn.Module,
             current_epoch: int,
             gpu: int,
             device: torch.device,
             verbose: bool = False,
             print_freq: int = 100,
             writer: SummaryWriter = None) -> Tuple[float, float]:
    """
    在验证数据集上验证模型的性能。

    :param val_loader: 验证数据加载器。
    :param model: 验证模型。
    :param criterion: 损失函数。
    :param current_epoch: 当前的训练周期。
    :param gpu: 使用的GPU索引，用于确定是否在特定GPU上打印信息。
    :param device: 计算设备。
    :param verbose: 是否打印详细信息。
    :param print_freq: 打印信息的频率。
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
                                         device,
                                         is_training=False)
        update_meters([losses_meter, top1, top5], loss, acc1, acc5,
                      batch[0].size(0))

        batch_processing_time.update(time.time() - end)
        end = time.time()

        if verbose and i % print_freq == 0 and gpu == 0:
            progress.display(i + 1)

    if verbose:
        progress.display_summary()

    if writer:
        writer.add_scalar('Loss/val', losses_meter.avg, current_epoch)
        writer.add_scalar('Top1/val', top1.avg, current_epoch)

    return top1.avg, losses_meter.avg


def run_training_loop(start_epoch: int,
                      epochs: int,
                      distributed: bool,
                      world_size: int,
                      batch_size: int,
                      arch: str,
                      lr: float,
                      optimizer_name: str,
                      tb_log_path: Optional[str],
                      checkpoint_folder: str,
                      amp_enabled: bool,
                      verbose: bool,
                      print_freq: int,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      model: nn.Module,
                      criterion: nn.Module,
                      optimizer_manager: OptimizerManager,
                      device: torch.device,
                      train_sampler: Optional[Sampler] = None,
                      ngpus_per_node: int = 1,
                      amp: Optional[object] = None,
                      debug_mode: bool = False) -> Tuple[float, int]:
    """
    改造后的运行训练循环函数，接受显式参数。
    """

    gpu = device.index

    # 初始化TensorBoard
    writer = None
    if gpu == 0:
        batch_size_total = world_size * batch_size
        custom_suffix = f"-{arch}-batch:{batch_size_total}-lr:{lr}-{optimizer_name}"
        if debug_mode:
            custom_suffix = '-debug' + custom_suffix
        if tb_log_path:
            save_event_path = os.path.join(tb_log_path, 'events')
            writer = SummaryWriter(log_dir=save_event_path,
                                   filename_suffix=custom_suffix)
            print('TensorBoard enabled at', tb_log_path)

    best_acc1 = 0
    best_epoch = start_epoch

    for current_epoch in range(start_epoch, epochs):
        if distributed:
            train_sampler.set_epoch(current_epoch)

        train(
            train_loader,
            model,
            criterion,
            optimizer_manager,
            current_epoch,
            gpu,
            device,
            verbose=verbose,  # 假设这个变量已经根据config或其他方式被设置
            print_freq=print_freq,  # 同上，确保这个变量已经被正确设置
            writer=writer,
        )

        early_stop_decision = False
        is_best = False
        if gpu == 0:
            acc1, val_loss = validate(
                val_loader,
                model,
                criterion,
                current_epoch=current_epoch,
                gpu=gpu,
                device=device,
                verbose=verbose,
                print_freq=print_freq,
                writer=writer,
            )

            if optimizer_manager.scheduler:
                optimizer_manager.scheduler_step(val_loss)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1) if is_best else best_acc1
            best_epoch = current_epoch if is_best else best_epoch

            optimizer_manager.check_early_stopping(val_loss)
            if optimizer_manager.early_stop:
                print("Early stopping triggered")
                early_stop_decision = True

            checkpoint = {
                'best_epoch':
                best_epoch,
                'best_acc1':
                best_acc1,
                'arch':
                arch,
                'state_dict':
                model.state_dict(),
                'optimizer':
                optimizer_manager.optimizer.state_dict(),
                'scheduler':
                optimizer_manager.scheduler.state_dict()
                if optimizer_manager.scheduler else None,
            }

            if amp_enabled and amp:
                checkpoint['amp'] = amp.state_dict()

            save_checkpoint(checkpoint,
                            is_best,
                            checkpoint_folder=checkpoint_folder,
                            check_point_suffix=custom_suffix[1::])

        # 广播早停决策给所有进程
        if distributed:
            early_stop_tensor = torch.tensor([int(early_stop_decision)],
                                             dtype=torch.int,
                                             device=device)
            dist.broadcast(early_stop_tensor, src=0)
            early_stop_decision = bool(early_stop_tensor.item())

        if early_stop_decision:
            print("Early stopping triggered across all processes." if gpu ==
                  0 else "")
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
        exporter = TBLogExporter(tb_log_path=tb_log_path,
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
