import logging
import os
from typing import Optional, Tuple

from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def verify_and_download_dataset(
    dataset_name: str, dataset_path: str, logger: logging.Logger
) -> None:
    """
    Verify if the dataset exists and download it if not. Ensures that the data
    path is set correctly and initiates the download process if the dataset is not found.

    Args:
    - dataset_name (str): The name of the dataset (e.g., 'cifar100').
    - dataset_path (str): The file path where the dataset should be stored. If None,
                       the path will default to a sub-directory named '{dataset_name}-data'
                       in the parent directory of this script.
    - logger (Logger): Logger object for logging status messages.

    Raises:
    - FileNotFoundError: If the dataset directory does not exist or is empty after checking.
    """
    if not dataset_path:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        dataset_path = os.path.join(base_dir, f"{dataset_name}-data")
        os.makedirs(dataset_path, exist_ok=True)

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        logger.info(
            f"Data for {dataset_name} is not found. Downloading to '{dataset_path}'."
        )
        _ = get_dataloaders(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            batch_size=1,
            distributed=False,
            download=True,
        )
        logger.info(f"Data downloaded and available at '{dataset_path}'.")
    else:
        logger.info(
            f"Dataset {dataset_name} already exists at '{dataset_path}'. No download needed."
        )


def calculate_mean_std(loader: DataLoader) -> Tuple[float, float]:
    """
    Helper function for get the mean and standard deviation of the dataset.
    """
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        # Reshape [B, C, W, H] -> [B, C, W * H]
        images = images.view(images.size(0), images.size(1), -1)
        # Update total number of images
        total_images_count += images.size(0)
        # Compute mean and std for this batch
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    # Normalize by the total number of images
    mean /= total_images_count
    std /= total_images_count
    return mean, std


def get_dataloaders(
    dataset_path: str,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    train_ratio: float,
    test_ratio: float,
    distributed: bool = False,
    download: bool = True,
    transform: Optional[Tuple[transforms.Compose, transforms.Compose]] = None,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    Optional[DistributedSampler],
    Optional[DistributedSampler],
]:
    """
    Creates and returns dataloaders for CIFAR10 or CIFAR100 datasets with configurable transforms.

    Args:
        dataset_path (str): Path to the dataset.
        dataset_name (str): Either 'cifar10' or 'cifar100'.
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of worker processes for loading data.
        train_ratio (float): Proportion of the dataset to include in the train split.
        test_ratio (float): Proportion of the dataset to include in the test split.
        distributed (bool): Whether to use distributed training.
        download (bool): Whether to download the datasets.
        transform (Optional[Tuple[transforms.Compose, transforms.Compose]]): Tuple of train and test transforms.

    Returns:
        Tuple containing three DataLoaders for training, validation, and testing, and two optional DistributedSamplers for train and validation sets if distributed training is enabled.
    """

    if dataset_name not in ["cifar10", "cifar100"]:
        raise ValueError(
            f"Unsupported dataset name: {dataset_name}. Choose 'cifar10' or 'cifar100'."
        )
    # dataset mapping
    dataset_classes = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}
    # Get the dataset class from the mapping.
    Dataset = dataset_classes[dataset_name]

    # Define default transformations if none provided
    if transform is None:
        # Calculate mean and standard deviation for normalization
        temp_dataset = Dataset(
            root=dataset_path,
            train=True,
            download=download,
            transform=transforms.ToTensor(),
        )
        data_loader = DataLoader(
            temp_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Calculate mean and std deviation using the DataLoader
        mean, std = calculate_mean_std(data_loader)
        normalize = transforms.Normalize(mean=mean, std=std)

        del temp_dataset, data_loader

        # Train transformation with data augmentation
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        transform = (train_transform, test_transform)

    assert len(transform) == 2, "Transform should be a tuple of two transforms."

    # Load entire dataset for splitting
    full_dataset = Dataset(
        root=dataset_path, train=True, download=download, transform=transform[0]
    )
    total_size: int = len(full_dataset)
    train_size: int = int(total_size * train_ratio)
    test_size: int = int(total_size * test_ratio)
    val_size: int = total_size - train_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Set up samplers for distributed training if enabled
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset) if distributed else None

    # Create data loaders for each set
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_sampler, val_sampler


if __name__ == "__main__":
    dataset_name = "cifar100"
    current_script_path = os.path.abspath(__file__)
    # Get the directory where the current script resides
    current_script_dir = os.path.dirname(os.path.dirname(current_script_path))
    # Set the dataset storage path to a subdirectory 'cifar100_data' within the script's directory
    dataset_path = os.path.join(current_script_dir, f"{dataset_name}_data")

    # Check if the data path exists and is not empty
    if os.path.exists(dataset_path) and os.listdir(dataset_path):
        print(f"Dataset directory '{dataset_path}' already exists and is not empty.")
    else:
        # Ensure the data directory exists
        os.makedirs(dataset_path, exist_ok=True)
        print(f"Data directory '{dataset_path}' created.")

    # Define transformations for training and testing
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    # Execute get_dataloaders to initialize data loaders with the specified transformations
    train_loader, val_loader, test_loader, train_sampler, val_sampler = get_dataloaders(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        batch_size=64,
        num_workers=2,
        distributed=False,
        download=True,
        transform=(train_transform, test_transform),
    )  # (train_transform, test_transform)

    print("Data loaders for training, validation, and testing are ready.")
