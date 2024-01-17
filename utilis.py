from enum import Enum
import os
import shutil
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu
from torch.nn import Module
from torch.optim import Optimizer
from argparse import Namespace
from apex import amp


def load_checkpoint(
        checkpoint_path: str, model: Module, optimizer: Optimizer,
        args: Namespace,
        device: str) -> Tuple[Module, Optimizer, int, Optional[float]]:
    """
    加载训练检查点。

    :param checkpoint_path: 检查点文件的路径，类型为字符串。
    :param model: 要加载状态的模型，类型为 torch.nn.Module。
    :param optimizer: 优化器，类型为 torch.optim.Optimizer。
    :param args: 包含训练配置和参数的命名空间，类型为 argparse.Namespace。
    :param device: 计算设备，类型为字符串。
    :return: 更新后的模型、优化器、开始epoch和最佳精度。返回类型为元组。
    """
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)

        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint.get('best_acc1', None)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        if 'amp' in checkpoint and args.amp:
            amp.load_state_dict(checkpoint['amp'])

        print("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))
        best_acc1 = None

    return model, optimizer, args.start_epoch, best_acc1


def device_id_to_process_device_map(
        device_list: List[Union[str, int]]) -> Dict[int, int]:
    """
    此函数接收一个设备ID列表，然后为每个设备ID分配一个进程ID。
    
    :param device_list: 包含设备ID的列表，ID可以是字符串或整数类型。
    :return: 一个字典，其中键是进程ID（整数），值是对应的设备ID（整数）。
    """
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def set_device(device: torch.device) -> torch.device:
    """
    设置训练设备为GPU、NPU或CPU。
 
    :param device: PyTorch 设备对象.
    :return: 设置的PyTorch设备对象。
    """
    if device.type == "npu":
        torch_npu.npu.set_device(device)
        print(f"set device to {device}")
        return device

    elif device.type == "cuda" and torch.cuda.is_available():
        return device

    else:
        print("warning: CPU is used for training!!!")
        return torch.device("cpu")


def init_distributed_training(args: Namespace, ngpus_per_node: int,
                              gpu: Optional[Union[str, int]]) -> None:
    """
    初始化分布式训练环境。
    此函数用于设置分布式训练的环境参数。在多进程分布式训练中，计算每个进程的独特排名。

    :param args: 包含分布式训练设置的命名空间对象。
    :param ngpus_per_node: 每个节点上的GPU数量。
    :param gpu: 指定当前进程应使用的GPU索引，可以是字符串或整数.
    """

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )


def save_checkpoint(state: dict,
                    is_best: bool,
                    folder: str,
                    filename: str = "checkpoint.pth") -> None:
    """
    保存训练过程中的模型检查点。
    如果当前检查点是最佳模型，检查点保存为“model_best.pth”。

    :param state: 要保存的模型状态(参数和其他信息)
    :param is_best: 布尔值，指示当前检查点是否是迄今为止最佳的模型。
    :param folder: 保存检查点的文件夹路径。
    :param filename: 检查点文件的名称，默认为"checkpoint.pth"。
    """
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_path = os.path.join(folder, filename)
        torch.save(state, file_path)

        if is_best:
            shutil.copyfile(file_path, os.path.join(folder, "model_best.pth"))
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """
    计算并存储平均值和当前值。

    此类用于在训练过程中跟踪和计算任何数值（如损失、准确率等）的平均值、总和和计数。
    支持分布式环境下的all-reduce。

    Attributes:
        name (str): 计量器的名称。
        fmt (str): 输出格式。
        summary_type (Summary): 摘要类型，决定输出摘要的形式。
        val (float): 最新的数值。
        avg (float): 平均值。
        sum (float): 总和。
        count (int): 计数。
    """

    def __init__(self,
                 name: str,
                 fmt: str = ":f",
                 summary_type: Summary = Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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
        self.avg = self.sum / self.count

    def all_reduce(self):
        """
        当使用分布式训练时，这个方法可以帮助同步不同进程间的统计数据。
        """
        device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
        total = torch.tensor([self.sum, self.count],
                             dtype=torch.float32,
                             device=device)

        try:
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        except Exception as e:
            print(f"Error in all_reduce: {e}")
            return

        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        """返回格式化的统计信息字符串。"""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        """根据summary_type返回相应的摘要信息。"""
        fmtstr = {
            Summary.NONE: "",
            Summary.AVERAGE: "{name} {avg:.3f}",
            Summary.SUM: "{name} {sum:.3f}",
            Summary.COUNT: "{name} {count:.3f}"
        }.get(self.summary_type, "")

        if fmtstr == "":
            raise ValueError("Invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    进度度量器，用于显示训练过程中的进度和指标。

    此类提供了方法来格式化并打印训练过程中的进度和不同度量指标的值。可以用于显示当前批次的进度，以及累积的平均值、总和等。

    Attributes:
        num_batches (int): 批次总数。
        meters (list of AverageMeter): 存储各种度量指标的对象列表。
        prefix (str): 打印信息前的前缀字符串。
    """

    def __init__(self, num_batches, meters, prefix=""):
        """
        初始化进度度量器。

        :param num_batches: 批次总数。
        :param meters: 度量器对象的列表。
        :param prefix: 显示信息前的前缀字符串。
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        """
        显示当前批次的进度和度量指标。

        :param batch: 当前批次编号。
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        """显示所有度量指标的摘要信息。"""
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        """生成批次格式化字符串。"""
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


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
        return res
