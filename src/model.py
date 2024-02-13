from typing import Callable

import torch
import torch.nn as nn
import torchvision.models as models


class CIFARNet(nn.Module):

    def __init__(self,
                 model_func: Callable,
                 num_classes: int = 10,
                 pretrained: bool = False,
                 device: torch.device = None):
        super(CIFARNet, self).__init__()
        self.model = model_func(pretrained=pretrained)
        self.num_classes = num_classes
        self.device = device

        # 修改ResNet的第一层卷积和去掉maxpooling
        if isinstance(self.model, models.ResNet):
            # 修改第一层卷积
            self.model.conv1 = nn.Conv2d(3,
                                         64,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         bias=False)
            # 去掉maxpooling
            self.model.maxpool = nn.Identity()
            print('Model adjusted for CIFAR-10/100 dataset')

        # 替换最后的全连接层以适应 CIFAR10/100 的类别数
        if hasattr(self.model, 'fc'):
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif hasattr(self.model, 'classifier'):  # 对于 VGG 等模型
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_features, num_classes)
        else:
            raise NotImplementedError(
                "Model does not have 'fc' or 'classifier' layer")

    def forward(self, x):
        return self.model(x)

    def to_device(self):
        if not self.device:
            raise ValueError(
                'Device is not set! Please set device before calling to_device()'
            )
        self.to(self.device)

    def unfreeze(self):
        """解冻模型的所有参数"""
        for param in self.parameters():
            param.requires_grad = True

        # 将torch.device对象转换为字符串表示形式
        device_str = str(self.device)

        # 检查设备是否为编号为0的主设备或CPU
        if device_str == 'cpu' or device_str.endswith(':0'):
            print('All layers are unfrozen for training')


def load_or_create_model(arch: str,
                         dataset_name: str = 'cifar100',
                         pretrained: bool = True,
                         device: torch.device = None) -> CIFARNet:
    """加载或创建模型，兼容 CIFAR10 和 CIFAR100 数据集"""
    # 通过传递 model_func 和相关参数来创建 CIFARNet 实例
    model_func = models.__dict__[arch]
    num_classes = 100 if dataset_name == 'cifar100' else 10
    model = CIFARNet(model_func=model_func,
                     num_classes=num_classes,
                     pretrained=pretrained,
                     device=device)
    return model


if __name__ == '__main__':
    # 创建 CIFAR-10 的模型实例
    device = 'npu' if torch.npu.is_available() else 'cpu'  # 示例中使用 npu 如果可用
    model_name = 'resnet50'
    model = CIFARNet(model_func=models.__dict__[model_name],
                     num_classes=10,
                     pretrained=True,
                     device=device)
