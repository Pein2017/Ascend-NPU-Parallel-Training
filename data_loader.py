import os
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def get_dataloaders(
    data_path: str,
    dataset_name: str,
    batch_size: int,
    num_workers: int = 4,
    split_ratio: float = 0.9,
    distributed: bool = True,
    download: bool = False,
    transform: Optional[Tuple[transforms.Compose, transforms.Compose]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DistributedSampler],
           Optional[DistributedSampler]]:
    """
    创建并返回 CIFAR10 或 CIFAR100 数据集的数据加载器。

    :param data_path: 数据集的路径。
    :param dataset_name: 'cifar10' 或 'cifar100'。
    :param batch_size: 批次大小。
    :param split_ratio: 训练集划分比例。
    :return: 训练、验证和测试数据加载器的元组。
    """
    # mean = {
    #     'cifar10': (0.4914, 0.4822, 0.4465),
    #     'cifar100': (0.5071, 0.4867, 0.4408),
    # }

    # std = {
    #     'cifar10': [0.2470, 0.2435, 0.2616],
    #     'cifar100': (0.2675, 0.2565, 0.2761),
    # }

    # 选择数据集

    # 选择数据集
    if dataset_name == 'cifar10':
        Dataset = datasets.CIFAR10
    elif dataset_name == 'cifar100':
        Dataset = datasets.CIFAR100
    else:
        raise ValueError(f"{dataset_name}不支持，请选择 'cifar10' 或 'cifar100'。")

    # 临时下载数据集以计算均值和标准差
    temp_dataset = Dataset(root=data_path,
                           train=True,
                           download=download,
                           transform=None)
    data = np.array(temp_dataset.data) / 255.0  # 转换到0-1范围
    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    del temp_dataset, data

    if transform is None:
        # 使用计算得到的均值和标准差
        normalize = transforms.Normalize(mean=mean, std=std)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # 使用自定义转换
        train_transform, test_transform = transform

    train_dataset = Dataset(root=data_path,
                            train=True,
                            download=download,
                            transform=train_transform)

    test_dataset = Dataset(root=data_path,
                           train=False,
                           download=download,
                           transform=test_transform)

    # 划分训练和验证集
    train_size = int(split_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset,
                                              [train_size, val_size])

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset,
                                         shuffle=False,
                                         drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              drop_last=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=val_sampler,
                            drop_last=True,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_sampler, val_sampler


if __name__ == '__main__':
    dataset_name = 'cifar100'
    current_script_path = os.path.abspath(__file__)
    # 获取当前脚本所在的目录
    current_script_dir = os.path.dirname(current_script_path)
    # 设置数据集存储路径为该目录下的子目录 'cifar10_data'
    data_path = os.path.join(current_script_dir, f'{dataset_name}_data')

    # 检查数据路径是否存在且不为空
    if os.path.exists(data_path) and os.listdir(data_path):
        print(
            f"Dataset '{data_path}' already exists and is not empty. Skipping download."
        )
    else:
        # 确保数据路径存在
        os.makedirs(data_path, exist_ok=True)
        # 执行get_dataloaders
        _ = get_dataloaders(data_path=data_path,
                            dataset_name=dataset_name,
                            batch_size=1,
                            distributed=False,
                            download=True)
        print(f"Data downloaded to '{data_path}'.")
