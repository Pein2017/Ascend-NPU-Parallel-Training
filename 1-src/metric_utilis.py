from enum import Enum
from typing import List, Tuple

import torch
import torch.distributed as dist


class MetricType(Enum):
    NONE = 0
    AVERAGE = 1
    TOTAL_SUM = 2
    COUNT = 3


class MetricTracker:
    """
    Initialize the MetricTracker.

    Args:
        name (str): The name of the metric.
        fmt (str): Format string for outputting metric values.
        metric_type (MetricType): Determines how the metric is calculated and displayed.
    """

    def __init__(
        self,
        name: str,
        fmt: str = ":f",
        metric_type: MetricType = MetricType.AVERAGE,
        device=None,
    ):
        self.name = name
        self.fmt = fmt
        self.metric_type = metric_type
        if device is None:
            if torch.npu.is_available():
                self.device = torch.device(f"npu:{dist.get_rank()}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.total = torch.zeros(2, device=self.device)
        self.reset()

    def reset(self):
        """Reset all metrics to initial state."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.total.zero_()

    def update(self, val: float, n: int = 1):
        """
        Update the metric tracker with a new value.

        Args:
            val (float): The new value to add to the metric.
            n (int): The number of occurrences of this value (useful for batched updates).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        # Update the tensor for distributed computation
        self.total += torch.tensor([val * n, n], device=self.device)

    def all_reduce(self):
        """Synchronize metrics across all processes in distributed mode."""
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.total, op=dist.ReduceOp.SUM)
            self.sum, self.count = self.total.tolist()
            self.avg = self.sum / self.count if self.count != 0 else 0
        else:
            # Provide a detailed error message about what is likely misconfigured
            error_msg = (
                "Failed to perform all_reduce operation because the "
                "distributed environment is not properly initialized. "
                "Please ensure that torch.distributed is available and "
                "initialized correctly before calling this method."
            )
            raise RuntimeError(error_msg)

    def __str__(self) -> str:
        """Returns a formatted string representation of the current metric state."""
        if "时间" in self.name:
            return f"{self.name} {self.val:.3f}秒 (平均: {self.avg:.3f}秒)"
        else:
            # Correctly format the value and average according to the format specifier in self.fmt
            formatted_val: str = "{val:{fmt}}".format(
                val=self.val, fmt=self.fmt.strip(":")
            )
            formatted_avg = "{avg:{fmt}}".format(avg=self.avg, fmt=self.fmt.strip(":"))
            return f"{self.name} {formatted_val} (平均: {formatted_avg})"

    def summary(self) -> str:
        """Returns a summary based on the metric_type."""
        fmtstr = {
            MetricType.NONE: "",
            MetricType.AVERAGE: f"{self.name} Avg: {self.avg:.3f}",
            MetricType.TOTAL_SUM: f"{self.name} Total Sum: {self.sum:.3f}",
            MetricType.COUNT: f"{self.name} Count: {self.count}",
        }.get(self.metric_type, "")

        if fmtstr == "":
            raise ValueError(f"Invalid summary type {self.metric_type}")

        # Depending on the metric type, format the corresponding attribute
        return fmtstr.format(
            name=self.name, avg=self.avg, sum=self.sum, count=self.count
        )


class ProgressMeter:
    """
    A utility class for displaying progress and metrics during training.
    """

    def __init__(
        self, total_batches: int, trackers: List[MetricTracker], prefix: str = ""
    ):
        """
        Initialize the progress meter.

        Args:
            total_batches (int): Total number of batches in the epoch.
            trackers (List[MetricTracker]): List of MetricTracker objects to track and display.
            prefix (str): Prefix string to show before the progress information.
        """
        self.total_batches = total_batches
        self.trackers = trackers
        self.prefix = prefix
        self.batch_format_str = self._generate_batch_format_string(total_batches)

    def _generate_batch_format_string(self, total_batches: int) -> str:
        """
        Generate a format string for the batch progress based on the total number of batches.

        Args:
            total_batches (int): Total number of batches in the epoch.

        Returns:
            str: A format string for batch progress display.
        """
        total_digits = len(str(total_batches))
        return f"{{:>{total_digits}}} / {total_batches}"

    def display(self, current_batch: int) -> str:
        """
        Display the current batch's progress and metrics.

        Args:
            current_batch (int): Current batch number.
        """
        entries = [f"{self.prefix}{self.batch_format_str.format(current_batch)}"]
        entries += [str(tracker) for tracker in self.trackers]
        print("\t".join(entries))
        return "\t".join(entries)

    def display_summary(self) -> str:
        """Displays a summary of all metrics at the end of an epoch or training session."""
        entries = [f"{self.prefix} *"]
        entries += [tracker.summary() for tracker in self.trackers]
        print(" ".join(entries))
        return "\t".join(entries)

    def _generate_batch_format_string(self, total_batches: int) -> str:
        """Generates a format string for batch progress display."""
        num_digits = len(str(total_batches))
        return f"[{{:>{num_digits}}} / {total_batches}]"


def get_topk_acc(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...]
) -> List[torch.Tensor]:
    """
    Compute the top-k accuracies for the specified values of k.

    Args:
        output (torch.Tensor): The output logits from the model.
        target (torch.Tensor): The true labels for the data.
        topk (Tuple[int, ...]): Tuple of integers defining the top-k accuracies to compute.

    Returns:
        List[torch.Tensor]: A list containing the get_topk_acc percentages for each top-k specified.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get the indices of the top maxk predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # Transpose for easier indexing

        # Compare predictions with the ground truth
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # Check the correct predictions within the top-k
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            get_topk_acc_k = correct_k.mul_(
                100.0 / batch_size
            ).item()  # Convert to percentage
            res.append(get_topk_acc_k)
            # Sanity check to ensure calculated get_topk_acc is within bounds
            if get_topk_acc_k > 100:
                raise ValueError(
                    f"Calculated get_topk_acc is greater than 100%: {get_topk_acc_k}"
                )

        return res
