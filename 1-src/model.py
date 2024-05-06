import logging
from typing import Callable, Dict

import torch
import torch.nn as nn
import torchvision.models as models
from setup_utilis import setup_logger

model_logger = setup_logger(
    name="ModelProcess",
    log_file_name="model_process.log",
    level=logging.DEBUG,
    console=True,
)


class CIFARNet(nn.Module):
    def __init__(
        self,
        num_classes,
        model_func: Callable,
        pretrained: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super(CIFARNet, self).__init__()
        self.model = model_func(pretrained=pretrained)
        self.num_classes = num_classes
        self.device = device
        # Modify the first convolution layer and remove max pooling for ResNet
        if isinstance(self.model, models.ResNet):
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.model.maxpool = nn.Identity()

        # Replace the final layer to adapt to CIFAR-10/100 class numbers
        final_layer = (
            self.model.fc if hasattr(self.model, "fc") else self.model.classifier[-1]
        )
        num_features = final_layer.in_features
        new_final_layer = nn.Linear(num_features, num_classes)
        if hasattr(self.model, "fc"):
            self.model.fc = new_final_layer
        elif hasattr(self.model, "classifier"):
            self.model.classifier[-1] = new_final_layer
        else:
            raise NotImplementedError(
                "Model must have a 'fc' or 'classifier[-1]' attribute."
            )

        if self.is_primary_device():
            model_logger.debug("CIFARNet adjusted for CIFAR-10/100 dataset")

    def forward(self, x):
        return self.model(x)

    def to_device(self):
        if not self.device:
            raise ValueError(
                "Device is not set! Please set device before calling to_device()"
            )
        self.to(self.device)

    def unfreeze(self):
        """Unfreeze all model parameters for training."""
        for param in self.parameters():
            param.requires_grad = True
        if self.is_primary_device():
            model_logger.debug("All layers are unfrozen for training")

    def is_primary_device(self, primary_id: int = 0) -> bool:
        """
        Check if the current device is the primary device or CPU.

        Args:
        primary_id (int): Device index that is considered the primary device. Defaults to 0.

        Returns:
        bool: True if the current device is the primary device, False otherwise.
        """
        if not self.device:  # Ensure there is a device attribute set
            raise ValueError("Device not set for this model instance.")
        device_str = str(self.device)
        return device_str == "cpu" or device_str.endswith(f":{primary_id}")


def load_or_create_model(config: Dict) -> CIFARNet:
    """Load or create a model, compatible with CIFAR10 and CIFAR100 datasets."""
    try:
        arch = config["model"]["arch"]
        dataset_name = config["data"].get("dataset_name", "cifar100")
        pretrained = config["model"].get("pretrained", False)
        device_str = config["distributed_training"].get("device", "cpu")
        device = torch.device(device_str)

        model_func = models.__dict__[arch]
        num_classes = 100 if dataset_name == "cifar100" else 10

        model = CIFARNet(
            model_func=model_func,
            num_classes=num_classes,
            pretrained=pretrained,
            device=device,
        )
        return model
    except KeyError as e:
        raise ValueError(f"Missing configuration for {e.args[0]}")


if __name__ == "__main__":
    # Define configuration for the model
    config = {
        "model": {
            "arch": "resnet18",
            "pretrained": True,  # Assuming you might want to check with pretrained weights
        },
        "data": {"dataset_name": "cifar10"},
        "distributed_training": {"device": "cpu"},
    }

    model = load_or_create_model(config)
    model.to_device()  # Ensure the model is on the correct device

    # Print model information and check the requires_grad status of each parameter
    print(f"Model loaded on {model.device} with architecture {config['model']['arch']}")
    for name, param in model.named_parameters():
        print(f"{name} - requires_grad: {param.requires_grad}")
