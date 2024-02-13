import os
import shutil
from argparse import Namespace
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from uu import Error

import torch
import torch.distributed as dist
import torch_npu
from torch.nn import Module
from torch.optim import Optimizer

# warnings.filterwarnings("ignore")


def load_checkpoint(
        checkpoint_path: str) -> Tuple[dict, dict, int, Optional[float]]:
    """
    简化的加载训练检查点功能。

    :param checkpoint_path: 检查点文件的路径，类型为字符串。
    :return: 一个元组，包含模型的state_dict，优化器的状态字典，最佳训练周期（best_epoch），和最佳准确率（best_acc1）。
    """

    if checkpoint_path is not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    print(f"=> loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)

    model_state_dict = checkpoint.get('state_dict')
    optimizer_state_dict = checkpoint.get('optimizer')
    best_epoch = checkpoint.get('best_epoch', 0)
    best_acc1 = checkpoint.get('best_acc1', None)

    print(
        f"=> loaded checkpoint '{checkpoint_path}' (epoch {best_epoch}, best_acc1={best_acc1})"
    )

    return model_state_dict, optimizer_state_dict, best_epoch, best_acc1


def device_id_to_process_device_map(
        device_list: Union[str, List[Union[str, int]]]) -> Dict[int, int]:
    """
    此函数接收一个设备ID列表，然后为每个设备ID分配一个进程ID。

    :param device_list: 可以是逗号分隔的设备ID字符串或包含设备ID的列表，ID可以是字符串或整数类型。
    :return: 一个字典，其中键是进程ID（整数），值是对应的设备ID（整数）。
    """
    # 检测输入类型并转换为整数列表
    if isinstance(device_list, str):
        devices = [int(x) for x in device_list.split(",")]
    elif isinstance(device_list, list):
        devices = [int(x) for x in device_list]
    else:
        raise ValueError(
            "device_list must be either a comma-separated string or a list of integers/strings."
        )

    devices.sort()

    process_device_map = {
        process_id: device_id
        for process_id, device_id in enumerate(devices)
    }

    return process_device_map


def set_device(device: str, gpu: Optional[Union[str, int]]) -> torch.device:
    """设置训练设备为GPU或NPU"""
    if device == 'npu':
        loc = 'npu:{}'.format(gpu)
        torch_npu.npu.set_device(loc)
    elif device == 'gpu':
        loc = 'cuda:{}'.format(gpu)
        torch.cuda.set_device(loc)
    else:
        loc = 'cpu'

    print(f'Set device {loc} for training.')
    return torch.device(loc)


def init_distributed_training(distributed: bool, dist_url: str, rank: int,
                              dist_backend: str, world_size: int,
                              multiprocessing_distributed: bool,
                              ngpus_per_node: int,
                              gpu: Optional[Union[str, int]]) -> None:
    """
    初始化分布式训练环境，显式传入所需参数。

    :param distributed: 是否启用分布式训练。
    :param dist_url: 分布式训练的URL，用于进程间通信。
    :param rank: 当前进程的全局排名。
    :param dist_backend: 分布式训练使用的后端。
    :param world_size: 分布式训练中的总进程数。
    :param multiprocessing_distributed: 是否使用多进程分布式训练。
    :param ngpus_per_node: 每个节点上的GPU数量。
    :param gpu: 指定当前进程应使用的GPU索引，可以是字符串或整数。
    """
    if distributed:
        if dist_url == "env://" and rank == -1:
            rank = int(os.environ["RANK"])
        if multiprocessing_distributed:
            rank = rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=dist_backend,
            init_method=dist_url,
            world_size=world_size,
            rank=rank,
        )


def save_checkpoint(state: dict, is_best: bool, checkpoint_folder: str,
                    check_point_suffix: str) -> None:
    """
    保存训练过程中的模型检查点。
    如果当前检查点是最佳模型，检查点保存为“model_best.pth”。

    :param state: 要保存的模型状态(参数和其他信息)
    :param is_best: 布尔值，指示当前检查点是否是迄今为止最佳的模型。
    :param checkpoint_folder: 保存检查点的文件夹路径。
    :param check_point_suffix: 检查点文件的名称。
    """
    try:
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        file_path = os.path.join(checkpoint_folder,
                                 check_point_suffix + f'-checkpoint.pth')
        torch.save(state, file_path)

        if is_best:
            shutil.copyfile(
                file_path,
                os.path.join(checkpoint_folder,
                             check_point_suffix + f'-best_model.pth'))
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


class MetricType(Enum):
    NONE = 0
    AVERAGE = 1
    TOTAL_SUM = 2
    COUNT = 3


class MetricTracker:
    """
    计算并存储平均值和当前值。

    此类用于在训练过程中跟踪和计算任何数值（如损失、准确率等）的平均值、总和和计数。
    支持分布式环境下的all-reduce。

    Attributes:
        name (str): 计量器的名称。
        fmt (str): 输出格式。
        metric_type (MetricType): 决定输出计量的形式。
        val (float): 最新的数值。
        avg (float): 平均值。
        sum (float): 总和。
        count (int): 计数。
    """

    def __init__(self,
                 name: str,
                 fmt: str = ":f",
                 metric_type: MetricType = MetricType.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.metric_type = metric_type
        self.total = torch.tensor([0, 0], dtype=torch.float32)  # 初始化为零张量
        self.reset()

    def reset(self):
        """重置所有统计量为初始状态。"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        更新统计量。

        :param val: 新加入的数值。
        :param n: 新加入的数值的数量。
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def all_reduce(self):
        """当使用分布式训练时，同步不同进程间的统计数据。"""
        current_device = torch.device(f"npu:{dist.get_rank()}")
        world_size = dist.get_world_size()
        self.total = torch.tensor([self.sum, self.count],
                                  dtype=torch.float32,
                                  device=current_device)
        dist.all_reduce(self.total, dist.ReduceOp.SUM, async_op=False)
        self.sum = self.total[0].item()
        self.count = self.total[1].item()

        # 在计算平均值前确保count不为0
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self):
        """返回格式化的统计信息字符串"""
        if '时间' in self.name:
            fmtstr = "{name} {val:.3f}秒 (平均: {avg:.3f}秒)"
        else:
            fmtstr = "{name} {val" + self.fmt + "} (平均: {avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        """根据metric_type返回相应的摘要信息。"""
        fmtstr = {
            MetricType.NONE: "",
            MetricType.AVERAGE: "{name} {avg:.3f}",
            MetricType.TOTAL_SUM: "{name} {sum:.3f}",
            MetricType.COUNT: "{name} {count:.3f}"
        }.get(self.metric_type, "")

        if fmtstr == "":
            raise ValueError("Invalid summary type %r" % self.metric_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """
    进度度量器，用于显示训练过程中的进度和指标。
    """

    def __init__(self,
                 total_batches: int,
                 trackers: List[MetricTracker],
                 prefix: str = ""):
        """
        初始化进度度量器。
        :param total_batches: 批次总数。
        :param trackers: MetricTracker 对象的列表。
        :param prefix: 显示信息前的前缀字符串。
        """
        self.batch_format_str = self._generate_batch_format_string(
            total_batches)
        self.trackers = trackers
        self.prefix = prefix

    def display(self, current_batch: int):
        """
        显示当前批次的进度和度量指标。
        :param current_batch: 当前批次编号。
        """
        entries = [
            f"{self.prefix}{self.batch_format_str.format(current_batch)}"
        ]
        entries += [str(tracker) for tracker in self.trackers]
        print("\t".join(entries))

    def display_summary(self):
        """显示所有度量指标的摘要信息。"""
        entries = [f"{self.prefix} *"]
        entries += [tracker.summary() for tracker in self.trackers]
        print(" ".join(entries))

    def _generate_batch_format_string(self, total_batches: int):
        """生成批次格式化字符串。"""
        num_digits = len(str(total_batches))
        return f"[{{:>{num_digits}}} / {total_batches}]"


def accuracy(output: torch.Tensor, target: torch.Tensor,
             topk: Tuple[int, ...]) -> List[float]:
    """
    计算指定k值的top-k预测准确率。

    :param output: 模型的输出，通常是一个logits向量。
    :param target: 真实的标签。
    :param topk: 一个元组，包含要计算准确率的k值。
    :return: 一个列表，包含每个k值的准确率。
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    # 如果res中有值大于1，raise Error
    for i in res:
        if i > 100.0:
            print(f'Error from utilis.accuracy, get value of {i}')
            raise ValueError("Accuracy cannot be larger than 100")
    return res
