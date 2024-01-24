import time
from argparse import Namespace
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
import torch.distributed as dist
from torch import Tensor, nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader, Sampler
from utilis import AverageMeter, ProgressMeter, accuracy


<<<<<<< HEAD
def create_meters(batch_size: int,
=======
def create_meters(batch_size: into,
>>>>>>> b8f076adc2545ae3e406a7a80b071fea9a695191
                  prefix: str) -> Tuple[List[AverageMeter], ProgressMeter]:
    """
    创建用于跟踪训练或验证过程中的各项指标的度量器。

    :param batch_size: 批次大小，用于初始化进度度量器。
    :param prefix: 显示信息前的前缀字符串。
    :return: 包含度量器列表和进度度量器的元组。
    """
<<<<<<< HEAD

=======
>>>>>>> b8f076adc2545ae3e406a7a80b071fea9a695191
    batch_processing_time = AverageMeter('批次处理时间', ':6.3f')
    data_loading_time = AverageMeter('数据加载时间', ':6.3f')
    losses_meter = AverageMeter('损失', ':.4e')
    top1 = AverageMeter('准确率@1', ':6.2f')
    top5 = AverageMeter('准确率@5', ':6.2f')

    meters = [
        batch_processing_time, data_loading_time, losses_meter, top1, top5
    ]
    progress = ProgressMeter(batch_size, meters, prefix=prefix)

    return meters, progress


def process_batch(batch: Tuple[Tensor, Tensor], model: Module,
                  criterion: Module, device: torch.device,
                  is_training: bool) -> Tuple[Tensor, Tensor, Tensor]:
    """
    处理单个批次的数据，执行模型的前向传播和损失计算。

    :param batch: 数据批次，包含特征和目标。
    :param model: 要使用的模型。
    :param criterion: 损失函数。
    :param device: 计算设备。
    :param is_training: 指示是否为训练模式。
    :return: 包含损失和准确率的元组。
    """
    images, target = batch
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    if is_training:
        output = model(images)
    else:
        with torch.no_grad():
            output = model(images)

    loss = criterion(output, target)
    acc1, acc5 = accuracy(output, target, topk=(1, 5))

    return loss, acc1, acc5


def update_meters(meters: List[AverageMeter], loss: Tensor, acc1: Tensor,
                  acc5: Tensor, batch_size: int) -> None:
    """
    更新训练或验证过程中的度量器。

    :param meters: 包含度量器的列表。
    :param loss: 当前批次的损失。
    :param acc1: 当前批次的top1准确率。
    :param acc5: 当前批次的top5准确率。
    :param batch_size: 当前批次的大小。
    """
    losses_meter, top1, top5 = meters
    losses_meter.update(loss.item(), batch_size)
    top1.update(acc1[0], batch_size)
    top5.update(acc5[0], batch_size)
