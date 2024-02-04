from typing import Any, Callable, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


class OptimizerManager:
    """
    优化器管理器类，用于创建和管理 PyTorch 优化器及学习率调度器。
    """

    def __init__(self, model_parameters: nn.Parameter, optimizer_type: str,
                 optimizer_params: Dict[str, Any]):
        """
        初始化 OptimizerManager。

        :param model_parameters: 模型的参数。
        :param optimizer_type: 优化器类型，如 'SGD', 'Adam', 'AdamW'。
        :param optimizer_params: 优化器参数的字典。
        """
        self.optimizer = self.create_optimizer(model_parameters,
                                               optimizer_type,
                                               optimizer_params)
        self.scheduler: Optional[_LRScheduler] = None

    def create_optimizer(self, parameters: nn.Parameter, optimizer_type: str,
                         optimizer_params: Dict[str, Any]) -> optim.Optimizer:
        """
        根据指定的优化器类型和参数创建优化器。

        :param parameters: 模型的参数。
        :param optimizer_type: 优化器的类型。
        :param optimizer_params: 优化器的参数字典。
        :return: 创建的优化器。
        """

        # 定义 SGD 和 Adam 类优化器支持的参数键
        sgd_supported_keys = {'lr', 'momentum', 'weight_decay'}
        adam_supported_keys = {'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'}

        if optimizer_type == 'SGD':
            # 为 SGD 筛选适用的参数
            sgd_params = {
                k: v
                for k, v in optimizer_params.items() if k in sgd_supported_keys
            }
            return optim.SGD(parameters, **sgd_params)
        elif optimizer_type in ['Adam', 'AdamW']:
            # 为 Adam 和 AdamW 筛选适用的参数
            adam_params = {
                k: v
                for k, v in optimizer_params.items()
                if k in adam_supported_keys
            }
            if optimizer_type == 'Adam':
                return optim.Adam(parameters, **adam_params)
            else:
                return optim.AdamW(parameters, **adam_params)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def create_scheduler(self, scheduler_type, **kwargs):
        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, **kwargs)
        elif scheduler_type == 'StepLR':
            self.scheduler = StepLR(self.optimizer, **kwargs)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def scheduler_step(self, *args: Any, **kwargs: Any) -> None:
        if self.scheduler:
            # 获取当前学习率
            current_lr = self.scheduler.optimizer.param_groups[0]['lr']

            # 调用学习率调度器的 step 函数
            self.scheduler.step(*args, **kwargs)

            # 获取更新后的学习率
            updated_lr = self.scheduler.optimizer.param_groups[0]['lr']

            # # 打印学习率信息
            # if current_lr != updated_lr:
            #     print(f"Learning Rate updated: {current_lr} -> {updated_lr}")
        else:
            raise RuntimeError(
                "Scheduler not initialized. Call create_scheduler() first.")


class CriterionManager:
    """
    损失函数管理器类，用于创建和管理 PyTorch 损失函数。
    """

    def __init__(self, criterion_type: str, **kwargs: Any) -> None:
        self.criterion: nn.Module = self.create_criterion(
            criterion_type, **kwargs)

    def create_criterion(self, criterion_type: str,
                         **kwargs: Any) -> nn.Module:
        if criterion_type == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss(**kwargs)
        elif criterion_type == 'MSELoss':
            return nn.MSELoss(**kwargs)
        else:
            raise ValueError(f"Unsupported criterion type: {criterion_type}")

    def to_device(self, device: torch.device) -> None:
        self.criterion = self.criterion.to(device)
