import torch  # noqa: F401
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import os


def get_dataloaders(data_path: str,
                    batch_size: int,
                    num_workers: int = 4,
                    split_ratio: float = 0.9,
                    distributed: bool = True,
                    download: bool = True) -> tuple:
    """
    创建并返回CIFAR10数据集的数据加载器。

    :param data_path: 数据集的路径。
    :param batch_size: 批次大小。
    :param split_ratio: 训练集划分比例。
    :return: 训练、验证和测试数据加载器的元组。
    """
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    # 训练数据集的转换
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # 测试/验证数据集的转换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # 创建训练数据集
    train_dataset = datasets.CIFAR10(root=data_path,
                                     train=True,
                                     download=download,
                                     transform=train_transform)

    # 创建测试数据集
    test_dataset = datasets.CIFAR10(root=data_path,
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

    # 创建数据加载器
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

    current_script_path = os.path.abspath(__file__)
    # 获取当前脚本所在的目录
    current_script_dir = os.path.dirname(current_script_path)
    # 设置数据集存储路径为该目录下的子目录 'cifar10_data'
    data_path = os.path.join(current_script_dir, 'cifar10_data')

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
                            batch_size=1,
                            distributed=False,
                            download=True)
        print(f"Data downloaded to '{data_path}'.")
