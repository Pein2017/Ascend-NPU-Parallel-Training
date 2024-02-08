import time
from typing import Callable, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist
from utilis import MetricTracker, ProgressMeter, accuracy


def create_meters(batch_size: int,
                  prefix: str) -> Tuple[List[MetricTracker], ProgressMeter]:
    """
    创建用于跟踪训练或验证过程中的各项指标的度量器。

    :param batch_size: 批次大小，用于初始化进度度量器。
    :param prefix: 显示信息前的前缀字符串。
    :return: 包含度量器列表和进度度量器的元组。
    """
    batch_processing_time = MetricTracker('批次处理时间:', ':1.2f')
    data_loading_time = MetricTracker('数据加载时间:', ':1.2f')
    losses_meter = MetricTracker('loss:', ':1.4e')
    top1 = MetricTracker('top-1:', ':1.3f')
    top5 = MetricTracker('top-5', ':1.3f')

    meters = [
        losses_meter,
        top1,
        top5,
        batch_processing_time,
        data_loading_time,
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
    if loss < 0 or acc1 > 100 or acc5 > 100:
        print(
            f"Error from process_batch, batch information: {batch[0].size(0)} , target: {target.size(0)}"
        )
        raise ValueError(
            f"During training, Loss: {loss}, Top1: {acc1}, Top5: {acc5}. "
            "Loss and accuracy should be non-negative and less than 100")

    return loss, acc1, acc5


def update_meters(meters: List[MetricTracker], loss: Tensor, acc1: Tensor,
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


def broadcast_early_stop(early_stop_decision, device):
    """
    广播早停决策给所有进程。
    
    :param early_stop_decision: 是否需要早停的决策（在全局rank为0的进程中为True或False）。
    :param device: 指定广播操作使用的设备。
    """
    if dist.is_initialized():
        # 将决策转换为一个tensor
        early_stop_tensor = torch.tensor([int(early_stop_decision)],
                                         dtype=torch.int,
                                         device=device)
        # 广播早停决策
        dist.broadcast(early_stop_tensor, src=0)
        # 返回广播后的早停决策
        return bool(early_stop_tensor.item())
    else:
        raise RuntimeError("Distribution is not initialized.")
