import torch
import torch.nn as nn
from typing import Dict, Tuple, List


class ModelStatsTracker:

    def __init__(self, model: nn.Module) -> None:
        """
        初始化模型统计追踪器。

        :param model: 要追踪统计信息的神经网络模型。
        """
        self.model: nn.Module = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.layer_stats: Dict[str, Dict[str, float]] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """为模型的每一层注册前向和后向钩子，以便于在数据流经网络时收集统计信息。"""
        for name, module in self.model.named_modules():
            if name:  # 跳过根模块
                # 注册前向钩子，使用默认参数确保正确捕获`name`的当前值
                self.hooks.append(
                    module.register_forward_hook(
                        (lambda name: lambda module, input, output: self.
                         _forward_hook(module, input, output, name))(name)))
                # 注册后向钩子，同样使用默认参数技巧
                self.hooks.append(
                    module.register_full_backward_hook(
                        (lambda name: lambda module, grad_input,
                         grad_output: self._backward_hook(
                             module, grad_input, grad_output, name))(name)))

    def _forward_hook(self, module: nn.Module, input, output,
                      name: str) -> None:
        """
        前向传播钩子函数，用于计算和保存模块权重的统计信息，
        包括均值、标准差，以及对特定类型层的额外计算（如最大特征值）。

        :param module: 当前层的模块。
        :param input: 当前层的输入数据。
        :param output: 当前层的输出数据。
        :param name: 当前层的名称。
        """
        # 初始化或更新层的统计信息存储结构
        if name not in self.layer_stats:
            self.layer_stats[name] = {}

        # 对有权重参数的模块计算均值和标准差
        if hasattr(module, 'weight') and module.weight is not None:
            weight_mean = module.weight.data.mean().item()
            weight_std = module.weight.data.std().item()
            self.layer_stats[name].update({
                "weight_mean": weight_mean,
                "weight_std": weight_std
            })

        # 示例：对线性层计算权重的最大特征值
        if isinstance(module, nn.Linear):
            # 注意：此处应替换为实际计算最大特征值的代码
            # 假设max_eigenvalue是计算得到的最大特征值
            max_eigenvalue = 0  # 示例值，应替换为实际计算结果
            self.layer_stats[name].update({"max_eigenvalue": max_eigenvalue})

    def _backward_hook(self, module: nn.Module,
                       grad_output: Tuple[torch.Tensor], name: str) -> None:
        """
        后向传播钩子函数，用于追踪和保存梯度信息。

        :param module: 当前层的模块。
        :param grad_output: 当前层梯度输出。
        :param name: 当前层的名称。
        """
        if grad_output[0] is not None:
            grad_norm = torch.norm(grad_output[0], 2).item()
            if name in self.layer_stats:
                self.layer_stats[name].update({"grad_norm": grad_norm})
            else:
                self.layer_stats[name] = {"grad_norm": grad_norm}

    def remove_hooks(self) -> None:
        """移除所有注册的钩子，以避免内存泄漏。"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def print_stats(self) -> None:
        """打印每层的统计信息。"""
        for name, stats in self.layer_stats.items():
            print(f"Layer: {name}")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value}")
