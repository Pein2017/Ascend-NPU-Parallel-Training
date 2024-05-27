from enum import Enum
from typing import List, Tuple

import torch
import torch.distributed as dist


# Define the MetricType Enum
class MetricType(Enum):
    NONE = 0
    AVERAGE = 1
    TOTAL_SUM = 2
    COUNT = 3


# Unified class for Metric Tracking and Progress Reporting
class MetricProgressTracker:
    def __init__(
        self,
        name: str,
        total_batches: int,
        fmt: str = ":f",
        metric_type: MetricType = MetricType.AVERAGE,
        device=None,
        prefix: str = "",
    ) -> None:
        if device is None:
            raise ValueError(
                "device must be specified in MetricTracker due to all reduce operation."
            )

        self.name = name
        self.fmt = fmt
        self.metric_type = metric_type
        self.device = device
        self.total_batches = total_batches
        self.prefix = prefix
        self.total = torch.zeros(2, device=self.device)
        self.batch_format_str = self._generate_batch_format_string(total_batches)
        self.reset()

    def _generate_batch_format_string(self, total_batches: int) -> str:
        total_digits = len(str(total_batches))
        return f"-batch:{{:>{total_digits}}}/{total_batches}"

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.total.zero_()

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        self.total += torch.tensor([val * n, n], device=self.device)

    def all_reduce(self):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.total, op=dist.ReduceOp.SUM)
            self.sum, self.count = self.total.tolist()
            self.avg = self.sum / self.count if self.count != 0 else 0
        else:
            raise RuntimeError("Distributed environment is not properly initialized.")

    def __str__(self) -> str:
        formatted_val = "{val:{fmt}}".format(val=self.val, fmt=self.fmt.strip(":"))
        formatted_avg = "{avg:{fmt}}".format(avg=self.avg, fmt=self.fmt.strip(":"))
        return f"{self.name} {formatted_val} (平均: {formatted_avg})"

    def summary(self) -> str:
        fmtstr = {
            MetricType.NONE: "",
            MetricType.AVERAGE: f"{self.name} avg: {self.avg:.3f}",
            MetricType.TOTAL_SUM: f"{self.name} total Sum: {self.sum:.3f}",
            MetricType.COUNT: f"{self.name} count: {self.count}",
        }.get(self.metric_type, "")
        if fmtstr == "":
            raise ValueError(f"Invalid summary type {self.metric_type}")
        return fmtstr.format(
            name=self.name, avg=self.avg, sum=self.sum, count=self.count
        )

    def display(self, current_batch: int) -> str:
        entries = [f"{self.prefix}{self.batch_format_str.format(current_batch)}"]
        entries.append(str(self))
        return "\n".join(entries)

    def display_summary(self) -> str:
        entries = [f"{self.prefix} Summary:"]
        entries.append(self.summary())
        return "\n".join(entries)


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
