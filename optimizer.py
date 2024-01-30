from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class OptimizerManager:
    """
    优化器管理器类，用于创建和管理 PyTorch 优化器及学习率调度器。
    """

    def __init__(self, model_parameters: torch.nn.parameter.Parameter,
                 optimizer_type: str, lr: float, **kwargs: Any) -> None:
        self.optimizer: optim.Optimizer = self.create_optimizer(
            model_parameters, optimizer_type, lr, **kwargs)
        self.scheduler: Optional[Union[lr_scheduler.ReduceLROnPlateau,
                                       lr_scheduler.StepLR]] = None

    def create_optimizer(self, parameters: torch.nn.parameter.Parameter,
                         optimizer_type: str, lr: float,
                         **kwargs: Any) -> optim.Optimizer:
        if optimizer_type == 'SGD':
            momentum = kwargs.get('momentum', 0)
            return optim.SGD(parameters, lr, momentum=momentum, **kwargs)
        elif optimizer_type in ['Adam', 'AdamW']:
            # 提取 Adam 和 AdamW 共同使用的参数
            betas = kwargs.get('betas', (0.9, 0.999))
            eps = kwargs.get('eps', 1e-8)
            weight_decay = kwargs.get('weight_decay', 0)
            amsgrad = kwargs.get('amsgrad', False)

            common_args = {
                'lr': lr,
                'betas': betas,
                'eps': eps,
                'weight_decay': weight_decay,
                'amsgrad': amsgrad
            }
            if optimizer_type == 'Adam':
                return optim.Adam(parameters, **common_args)
            else:
                return optim.AdamW(parameters, **common_args)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def create_scheduler(self, scheduler_type, **kwargs):
        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **kwargs)
        elif scheduler_type == 'StepLR':
            self.scheduler = lr_scheduler.StepLR(self.optimizer, **kwargs)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def scheduler_step(self, *args: Any, **kwargs: Any) -> None:
        if self.scheduler:
            self.scheduler.step(*args, **kwargs)
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
