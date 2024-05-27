from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

# from metric_utilis import MetricTracker, MetricType, ProgressMeter, get_topk_acc
from metric_utilis import MetricProgressTracker, MetricType, get_topk_acc
from torch.utils.tensorboard import SummaryWriter


# Create metrics and progress meter
def create_meters(
    batch_size: int, total_batches: int, prefix: str, device: torch.device
) -> List[MetricProgressTracker]:
    trackers = [
        MetricProgressTracker(
            name="Loss",
            total_batches=total_batches,
            fmt=":1.4e",
            metric_type=MetricType.AVERAGE,
            device=device,
            prefix=prefix,
        ),
        MetricProgressTracker(
            name="Top-1 accuracy",
            total_batches=total_batches,
            fmt=":1.3f",
            metric_type=MetricType.AVERAGE,
            device=device,
            prefix=prefix,
        ),
        MetricProgressTracker(
            name="Top-5 accuracy",
            total_batches=total_batches,
            fmt=":1.3f",
            metric_type=MetricType.AVERAGE,
            device=device,
            prefix=prefix,
        ),
        MetricProgressTracker(
            name="Batch Training",
            total_batches=total_batches,
            fmt=":1.2f",
            metric_type=MetricType.AVERAGE,
            device=device,
            prefix=prefix,
        ),
        MetricProgressTracker(
            name="Data Loading",
            total_batches=total_batches,
            fmt=":1.2f",
            metric_type=MetricType.AVERAGE,
            device=device,
            prefix=prefix,
        ),
        MetricProgressTracker(
            name="Total Batch Time",
            total_batches=total_batches,
            fmt=":1.2f",
            metric_type=MetricType.AVERAGE,
            device=device,
            prefix=prefix,
        ),
        MetricProgressTracker(
            name="Backward Time",
            total_batches=total_batches,
            fmt=":1.2f",
            metric_type=MetricType.AVERAGE,
            device=device,
            prefix=prefix,
        ),
    ]
    return trackers


# Process batch
def process_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    is_training: bool,
) -> Tuple[torch.Tensor, float, float]:
    images, target = batch
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    if is_training:
        model.train()
        output = model(images)
    else:
        model.eval()
        with torch.no_grad():
            output = model(images)

    loss = criterion(output, target)
    acc1, acc5 = get_topk_acc(output, target, topk=(1, 5))

    return loss, acc1, acc5


"""
# Example usage
metric_values = {
    "Loss": (loss.item(), batch_size),
    "Top-1 accuracy": (acc1, batch_size),
    "Top-5 accuracy": (acc5, batch_size),
    "Batch Training": (batch_time, batch_size),
    "Data Loading": (data_time, batch_size),
    "Synchronization": (sync_time, batch_size),
}
"""


# Update meters
def update_meters(meters: List[MetricProgressTracker], metric_values: dict) -> None:
    """
    Update the metric trackers during training or validation.

    Args:
        meters (List[MetricProgressTracker]): List containing the metric trackers.
        metric_values (dict): Dictionary containing the metric names and their corresponding values.
    """
    for meter in meters:
        if meter.name in metric_values:
            value, batch_size = metric_values[meter.name]
            meter.update(value, batch_size)


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


# def create_meters(
#     batch_size: int, prefix: str, device: TorchDevice
# ) -> Tuple[List[MetricTracker], ProgressMeter]:
#     """
#     Create metric trackers for monitoring various metrics during training or validation.

#     Args:
#         batch_size (int): Batch size, used to initialize the progress meter.
#         prefix (str): Prefix string for display before information.

#     Returns:
#         Tuple[List[MetricTracker], ProgressMeter]: Tuple containing a list of metric trackers and a progress meter.
#     """
#     # Define the metric trackers with appropriate format specifications

#     losses_meter = MetricTracker(
#         name="Loss",
#         fmt=":1.4e",
#         metric_type=MetricType.AVERAGE,
#         device=device,
#     )
#     top1 = MetricTracker(
#         name="Top-1 accuracy",
#         fmt=":1.3f",
#         metric_type=MetricType.AVERAGE,
#         device=device,
#     )
#     top5 = MetricTracker(
#         name="Top-5 accuracy",
#         fmt=":1.3f",
#         metric_type=MetricType.AVERAGE,
#         device=device,
#     )

#     batch_processing_time = MetricTracker(
#         name="Batch Training:",
#         fmt=":1.2f",
#         metric_type=MetricType.AVERAGE,
#         device=device,
#     )
#     data_loading_time = MetricTracker(
#         "Data Loading:",
#         fmt=":1.2f",
#         metric_type=MetricType.AVERAGE,
#         device=device,
#     )
#     sync_time = MetricTracker(
#         "Synchronization:",
#         fmt=":1.2f",
#         metric_type=MetricType.AVERAGE,
#         device=device,
#     )

#     meters: List[MetricTracker] = [
#         losses_meter,
#         top1,
#         top5,
#         batch_processing_time,
#         data_loading_time,
#         sync_time,
#     ]
#     progress = ProgressMeter(total_batches=batch_size, trackers=meters, prefix=prefix)

#     return meters, progress


# def process_batch(
#     batch: Tuple[torch.Tensor, torch.Tensor],
#     model: nn.Module,
#     criterion: nn.Module,
#     device: torch.device,
#     is_training: bool,
# ) -> Tuple[torch.Tensor, float, float]:
#     """
#     Process a single batch of data, perform forward propagation and loss calculation.

#     Args:
#         batch (Tuple[torch.Tensor, torch.Tensor]): Data batch, containing features and targets.
#         model (nn.Module): The model to use.
#         criterion (nn.Module): Loss function.
#         device: torch.device): The computation device.
#         is_training (bool): Indicates whether it is in training mode.

#     Returns:
#         Tuple[torch.Tensor, float, float]: A tuple containing the loss, top-1 get_topk_acc, and top-5 get_topk_acc.
#     """

#     images, target = batch
#     images = images.to(device, non_blocking=True)
#     target = target.to(device, non_blocking=True)

#     # Set model to the correct mode and perform forward pass
#     if is_training:
#         model.train()
#         output = model(images)
#     else:
#         model.eval()
#         with torch.no_grad():
#             output = model(images)

#     # Compute loss and accuracy
#     loss = criterion(output, target)
#     acc1, acc5 = get_topk_acc(output, target, topk=(1, 5))

#     return loss, acc1, acc5
