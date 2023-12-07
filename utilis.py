from enum import Enum
import torch
from typing import List, Dict, Optional, Union
from argparse import Namespace
import torch.distributed as dist
import os
import torch_npu
import shutil


def device_id_to_process_device_map(
        device_list: List[Union[str, int]]) -> Dict[int, int]:
    """
    map device id to process id and device id in the process
    """
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def set_device(args: Namespace) -> torch.device:
    """设置训练设备为GPU或NPU"""
    if args.device == 'npu':

        loc = 'npu:{}'.format(args.gpu)
        torch_npu.npu.set_device(loc)
        print(f'set device to {loc}')
        return torch.device(loc)

    elif args.device == 'gpu' and torch.cuda.is_available():

        return torch.device('cuda:{}'.format(args.gpu) if args.gpu else 'cuda')

    else:

        print('warning: CPU is used for training!!!')

        return torch.device('cpu')


def init_distributed_training(args: Namespace, ngpus_per_node: int,
                              gpu: Optional[Union[str, int]]) -> None:
    """初始化分布式训练环境"""
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)


def save_checkpoint(state,
                    is_best,
                    folder='/home/HW/Pein/DistributedTrain/check_point',
                    filename='checkpoint.pth'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, filename)
    torch.save(state, file_path)

    if is_best:
        shutil.copyfile(file_path, os.path.join(folder, 'model_best.pth'))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.npu.is_available():
            device = torch.device("npu:0")
        else:
            device = torch.device("cpu")

        total = torch.tensor([self.sum, self.count],
                             dtype=torch.float32,
                             device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1, )):
    """
    Computes the accuracy over the k top predictions
    for the specified values of k
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
