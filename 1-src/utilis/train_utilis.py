from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from metric_utilis import MetricTracker, MetricType, ProgressMeter, get_topk_acc
from torch import Tensor
from torch import device as TorchDevice
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def create_meters(
    batch_size: int, prefix: str, device: TorchDevice
) -> Tuple[List[MetricTracker], ProgressMeter]:
    """
    Create metric trackers for monitoring various metrics during training or validation.

    Args:
        batch_size (int): Batch size, used to initialize the progress meter.
        prefix (str): Prefix string for display before information.

    Returns:
        Tuple[List[MetricTracker], ProgressMeter]: Tuple containing a list of metric trackers and a progress meter.
    """
    # Define the metric trackers with appropriate format specifications

    losses_meter = MetricTracker(
        name="Loss",
        fmt=":1.4e",
        metric_type=MetricType.AVERAGE,
        device=device,
    )
    top1 = MetricTracker(
        name="Top-1 accuracy",
        fmt=":1.3f",
        metric_type=MetricType.AVERAGE,
        device=device,
    )
    top5 = MetricTracker(
        name="Top-5 accuracy",
        fmt=":1.3f",
        metric_type=MetricType.AVERAGE,
        device=device,
    )

    batch_processing_time = MetricTracker(
        name="Batch Training:",
        fmt=":1.2f",
        metric_type=MetricType.AVERAGE,
        device=device,
    )
    data_loading_time = MetricTracker(
        "Data Loading:",
        fmt=":1.2f",
        metric_type=MetricType.AVERAGE,
        device=device,
    )
    sync_time = MetricTracker(
        "Synchronization:",
        fmt=":1.2f",
        metric_type=MetricType.AVERAGE,
        device=device,
    )

    meters: List[MetricTracker] = [
        losses_meter,
        top1,
        top5,
        batch_processing_time,
        data_loading_time,
        sync_time,
    ]
    progress = ProgressMeter(total_batches=batch_size, trackers=meters, prefix=prefix)

    return meters, progress


def process_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    is_training: bool,
) -> Tuple[torch.Tensor, float, float]:
    """
    Process a single batch of data, perform forward propagation and loss calculation.

    Args:
        batch (Tuple[torch.Tensor, torch.Tensor]): Data batch, containing features and targets.
        model (nn.Module): The model to use.
        criterion (nn.Module): Loss function.
        device: torch.device): The computation device.
        is_training (bool): Indicates whether it is in training mode.

    Returns:
        Tuple[torch.Tensor, float, float]: A tuple containing the loss, top-1 get_topk_acc, and top-5 get_topk_acc.
    """

    images, target = batch
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # Set model to the correct mode and perform forward pass
    if is_training:
        model.train()
        output = model(images)
    else:
        model.eval()
        with torch.no_grad():
            output = model(images)

    # Compute loss and accuracy
    loss = criterion(output, target)
    acc1, acc5 = get_topk_acc(output, target, topk=(1, 5))

    return loss, acc1, acc5


def update_meters(
    meters: List[MetricTracker], loss: Tensor, acc1: float, acc5: float, batch_size: int
) -> None:
    """
    Update the metric trackers during training or validation.

    Args:
        meters (List[MetricTracker]): List containing the metric trackers.
        loss (Tensor): The current batch's loss.
        acc1 (float): The current batch's top-1 get_topk_acc.
        acc5 (float): The current batch's top-5 get_topk_acc.
        batch_size (int): The size of the current batch.
    """
    losses_meter, top1, top5 = meters
    losses_meter.update(loss.item(), batch_size)
    top1.update(acc1, batch_size)
    top5.update(acc5, batch_size)


def broadcast_early_stop(early_stop_decision, device):
    """
    Broadcast an early stop decision to all processes.

    :param early_stop_decision: The decision to early stop (True or False) determined at global rank 0.
    :param device: The device to use for the broadcast operation.

    :returns: The early stop decision after broadcasting.
    :raises RuntimeError: If the distributed environment is not properly initialized.
    """
    if dist.is_initialized():
        # Convert the decision to a tensor, ensuring it's on the correct device
        early_stop_tensor = torch.tensor(
            [early_stop_decision], dtype=torch.int, device=device
        )

        # Broadcast the early stop decision from process with rank 0
        dist.broadcast(early_stop_tensor, src=0)

        # Convert tensor back to a boolean and return
        return bool(early_stop_tensor.item())
    else:
        # Raise an error if the PyTorch distributed environment is not initialized
        raise RuntimeError(
            "Distributed environment is not initialized, cannot broadcast early stop decision."
        )


def record_metrics(
    writer: SummaryWriter, metrics_dict: Dict, global_step: int, prefix: str = ""
) -> None:
    """
    Record metrics to TensorBoard.

    Args:
        writer (SummaryWriter): The TensorBoard writer.
        metrics_dict (dict): Dictionary containing metric names and their values.
        global_step (int): The global step value to associate with the scalar.
        prefix (str): Prefix for the metric tags (helps differentiate training and validation).
    """
    for key, value in metrics_dict.items():
        tag: str = f"{prefix}{key}"
        writer.add_scalar(tag=tag, scalar_value=value, global_step=global_step)
