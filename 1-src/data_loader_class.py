import logging
import os
from typing import Optional, Tuple

import torch.distributed as dist
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torchvision import transforms


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


class DataLoaderManager:
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        logger: logging.Logger,
        use_fake_data: bool = False,
        download: bool = True,
        fake_data_size: Tuple[int, int, int] = (1000, 100, 256),
        image_shape: Tuple[int, int, int] = (3, 32, 32),
    ):
        """
        Initialize the DataLoaderManager with dataset details.
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", f"{dataset_name}_data"
        )
        os.makedirs(self.dataset_path, exist_ok=True)
        self.logger = logger
        self.use_fake_data = use_fake_data
        self.download = download
        self.train_fake_size, self.val_fake_size, self.test_fake_size = fake_data_size
        self.image_shape = image_shape

    def verify_and_download_dataset(self):
        """
        Verify if the dataset exists and initialize dataset download if necessary using `get_dataset`.
        """
        # Here, simply calling get_dataset with a basic setup will ensure that the dataset is downloaded if not present.
        _, _, _ = self.get_dataset(train_ratio=0.9)  # Adjust train_ratio as needed
        self.logger.debug(f"Dataset verified and ready at '{self.dataset_path}'.")

    def get_dataset(
        self,
        train_ratio: float,
        transform: Optional[Tuple[transforms.Compose, transforms.Compose]] = None,
    ):
        """
        Retrieve datasets for training, validation, and testing based on whether fake data or real data is used.
        """
        if self.use_fake_data:
            num_classes = {"cifar100": 100, "cifar10": 10}.get(self.dataset_name)

            if not num_classes:
                raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

            self.logger.info("Using fake data for training.")

            # Create train, validation, and test fake datasets
            fake_data_params = {
                "image_size": self.image_shape,
                "num_classes": num_classes,
                "transform": transforms.ToTensor(),
            }

            train_dataset = datasets.FakeData(
                size=self.train_fake_size, **fake_data_params
            )
            val_dataset = datasets.FakeData(size=self.val_fake_size, **fake_data_params)
            test_dataset = datasets.FakeData(
                size=self.test_fake_size, **fake_data_params
            )

        else:
            dataset_classes = {
                "cifar10": datasets.CIFAR10,
                "cifar100": datasets.CIFAR100,
            }
            Dataset = dataset_classes[self.dataset_name]

            # Define default transformations if none provided
            if transform is None:
                temp_dataset = Dataset(
                    root=self.dataset_path,
                    train=True,
                    download=self.download,
                    transform=transforms.ToTensor(),
                )
                data_loader = DataLoader(
                    dataset=temp_dataset,
                    batch_size=256,
                    shuffle=False,
                    num_workers=0,
                )

                mean, std = calculate_mean_std(data_loader)
                normalize = transforms.Normalize(mean=mean, std=std)

                del temp_dataset, data_loader

                train_transform = transforms.Compose(
                    transforms=[
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
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

            # Split the real dataset into training and validation sets
            full_dataset = Dataset(
                root=self.dataset_path,
                train=True,
                download=self.download,
                transform=transform[0],
            )

            total_size = len(full_dataset)
            train_size = int(total_size * train_ratio)
            val_size = total_size - train_size

            train_dataset, val_dataset = random_split(
                dataset=full_dataset, lengths=[train_size, val_size]
            )

            # Create the test dataset
            test_dataset = Dataset(
                root=self.dataset_path,
                train=False,
                download=self.download,
                transform=transform[1],
            )
        # wrong when calling verify_and_download_dataset
        # self.logger.debug(
        #     f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}"
        # )
        return train_dataset, val_dataset, test_dataset

    def get_dataloaders(
        self,
        batch_size: int,
        adjusted_batch_size: int,
        num_workers: int,
        train_ratio: float,
        distributed: bool = False,
        transform: Optional[Tuple[transforms.Compose, transforms.Compose]] = None,
    ) -> Tuple[
        DataLoader,
        DataLoader,
        DataLoader,
        Optional[DistributedSampler],
        Optional[DistributedSampler],
        Optional[DistributedSampler],
    ]:
        """
        # NOTE:
        The `sampler` attribute in the DataLoader determines the type of sampling:

        - When `sampler` is `None`, PyTorch assigns a default sampler:
        - `SequentialSampler` if `shuffle=False`
        - `RandomSampler` if `shuffle=True`

        - When `distributed=True` and a `DistributedSampler` is provided, the DataLoader uses `DistributedSampler`:
        - Ensures each process gets a unique subset of the data.

        This function returns data loaders for training, validation, and testing datasets with appropriate samplers.
        """
        train_dataset, val_dataset, test_dataset = self.get_dataset(
            train_ratio, transform
        )

        if distributed:
            if not dist.is_initialized():
                raise RuntimeError(
                    "Distributed mode is enabled, but the torch.distributed backend is not initialized."
                )
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
            )
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
            )

        else:
            train_sampler, val_sampler, test_sampler = (None, None, None)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=adjusted_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            drop_last=True,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            drop_last=False,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=adjusted_batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers,
            drop_last=False,
        )

        return (
            train_loader,
            val_loader,
            test_loader,
            train_sampler,
            val_sampler,
            test_sampler,
        )


def print_loader_info(train_loader, val_loader, test_loader):
    """Print the sizes of the datasets loaded via the given data loaders."""
    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)
    test_dataset_size = len(test_loader.dataset)
    print(f"Training dataset size: {train_dataset_size}")
    print(f"Validation dataset size: {val_dataset_size}")
    print(f"Test dataset size: {test_dataset_size}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("DataLoaderManagerLogger")

    # Define dataset information and paths
    dataset_name = "cifar100"
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(os.path.dirname(current_script_path))
    dataset_path = os.path.join(current_script_dir, f"{dataset_name}_data")
    dataset_path = "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/cifar100_data"

    # Real dataset test
    print("\nTesting with real dataset...")
    data_loader_manager_real = DataLoaderManager(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        logger=logger,
        download=True,
    )
    data_loader_manager_real.verify_and_download_dataset()

    # Define transformations for real training and testing
    train_transform_real = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )
    test_transform_real = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    # Get dataloaders for real data
    train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = (
        data_loader_manager_real.get_dataloaders(
            batch_size=256,
            adjusted_batch_size=256,
            num_workers=0,
            train_ratio=0.9,
            distributed=False,
            transform=(train_transform_real, test_transform_real),
        )
    )
    isinstance(train_loader, DistributedSampler)

    print_loader_info(train_loader, val_loader, test_loader)

    # Fake dataset test
    print("\nTesting with fake dataset...")
    data_loader_manager_fake = DataLoaderManager(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        logger=logger,
        use_fake_data=True,
        fake_data_size=(1000, 100, 100),  # Example sizes for train, validation, test
        image_shape=(3, 32, 32),
    )

    # Get dataloaders for fake data
    train_loader, val_loader, test_loader, train_sampler, val_sampler = (
        data_loader_manager_fake.get_dataloaders(
            batch_size=256,
            adjusted_batch_size=256,
            num_workers=0,
            train_ratio=0.9,
            distributed=False,
        )
    )
    print_loader_info(train_loader, val_loader, test_loader)
