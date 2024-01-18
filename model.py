import torch
import torch.nn as nn  # noqa: F401
import torch_npu  # noqa: F401
from typing import Callable
import torchvision.models as models
from argparse import Namespace


class CIFAR10Net(nn.Module):

    def __init__(
        self,
        device: str,
        model_func: Callable,
        num_classes: int = 10,
        pretrained: bool = False,
    ):
        super(CIFAR10Net, self).__init__()
        self.model = model_func(pretrained=pretrained)

        # 需要检查模型是否有 'fc' 属性
        if hasattr(self.model, 'fc'):
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            # 对于没有 'fc' 的模型，需要适当的处理
            raise NotImplementedError("Model does not have a 'fc' layer")

        self.device = device

    def forward(self, x):
        return self.model(x)

    def to_device(self):
        self.model = self.model.to(self.device)


def load_or_create_model(args: Namespace) -> CIFAR10Net:
    """加载或创建模型"""
    # 通过传递 model_func 和相关参数来创建 CIFAR10Net 实例
    model_func = models.__dict__[args.arch]
    model = CIFAR10Net(model_func=model_func,
                       num_classes=10,
                       pretrained=args.pretrained,
                       device=args.device)
    return model


if __name__ == '__main__':
    # 创建 CIFAR-10 的模型实例
    device = 'npu' if torch.npu.is_available() else 'cpu'  # 示例中使用 npu 如果可用
    model_name = 'resnet50'
    model = CIFAR10Net(model_func=models.__dict__[model_name],
                       num_classes=10,
                       pretrained=True,
                       device=device)
    # model.to_device()
