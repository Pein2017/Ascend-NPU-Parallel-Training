import logging
import os
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist

# warnings.filterwarnings("ignore")#
from setup_utilis import setup_logger

utilis_worker: logging.Logger = setup_logger(
    name="UtilisProcess",
    log_file_name="utilis_process.log",
    level=logging.DEBUG,
    console=False,
)


def load_checkpoint(checkpoint_path: str) -> Tuple[Dict, Dict, int, Optional[float]]:
    """
    Simplified function to load a training checkpoint.

    :param checkpoint_path: Path to the checkpoint file, as a string.
    :return: A tuple containing the model's state_dict, the optimizer's state dictionary,
             the best training epoch (best_epoch), and the best get_topk_acc (best_acc1).
    """

    # Correct the condition to check for the file's existence
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    utilis_worker.info(f"=> Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)

    model_state_dict = checkpoint.get("state_dict")
    optimizer_state_dict = checkpoint.get("optimizer")
    best_epoch = checkpoint.get("best_epoch", 0)
    best_acc1 = checkpoint.get("best_acc1", None)

    utilis_worker.debug(
        f"=> Loaded checkpoint '{checkpoint_path}' (epoch {best_epoch}, best_acc1={best_acc1})"
    )

    return model_state_dict, optimizer_state_dict, best_epoch, best_acc1


def device_id_to_process_device_map(
    device_list: Union[str, List[Union[str, int]]],
) -> Dict[int, int]:
    """
    Converts a device ID list into a mapping of process IDs to device IDs.

    Args:
    device_list (Union[str, List[Union[str, int]]]): A comma-separated string of device IDs, or a list of device IDs that can be either strings or integers.

    Returns:
    Dict[int, int]: A dictionary where keys are process IDs (integers),
                    and values are the corresponding device IDs (integers).

    Example:
        >>> device_id_to_process_device_map("0, 1, 2, 3")
        {0: 0, 1: 1, 2: 2, 3: 3}
        >>> device_id_to_process_device_map([0, 1, 2, 3])
        {0: 0, 1: 1, 2: 2, 3: 3}
    """

    # Detect input type and convert to a list of integers
    if isinstance(device_list, str):
        devices = [int(x.strip()) for x in device_list.split(",")]
    elif isinstance(device_list, list):
        devices = [int(x) for x in device_list]
    else:
        raise TypeError(
            "device_list must be either a comma-separated string or a list of integers or strings."
        )

    # Sort the device IDs to maintain consistent mapping
    devices.sort()

    # Create a dictionary mapping process IDs to device IDs
    process_device_map = {
        process_id: device_id for process_id, device_id in enumerate(devices)
    }

    return process_device_map


def set_device(device: str, gpu: Optional[Union[str, int]]) -> torch.device:
    """
    Set the training device to GPU or NPU.

    Args:
    device (str): The type of device ('gpu', 'npu', or 'cpu').
    gpu (Optional[Union[str, int]]): The specific device identifier for GPU or NPU.

    Returns:
    torch.device: The device object representing the device to be used for training.
    """
    loc = "cpu"  # Default to CPU if no valid device type is provided

    if device == "npu":
        loc = f"npu:{gpu}"
        import torch_npu  # Import locally to avoid errors if the module isn't available

        torch_npu.npu.set_device(loc)
    elif device == "gpu":
        loc = f"cuda:{gpu}"
        torch.cuda.set_device(loc)

    utilis_worker.debug(f"Set device {loc} for training.")
    return torch.device(loc)


def init_distributed_training(
    distributed: bool,
    dist_url: str,
    rank: int,
    dist_backend: str,
    world_size: int,
    multiprocessing_distributed: bool,
    ngpus_per_node: int,
    gpu: Optional[Union[str, int]],
) -> None:
    """
    Initialize the distributed training environment with explicit parameters.

    Args:
    distributed (bool): Whether to enable distributed training.
    dist_url (str): URL for inter-process communication in distributed training.
    rank (int): Global rank of the current process.
    dist_backend (str): Backend to use for distributed training.
    world_size (int): Total number of processes in the distributed training.
    multiprocessing_distributed (bool): Whether to use multiprocessing in distributed training.
    ngpus_per_node (int): Number of GPUs per node.
    gpu (Optional[Union[str, int]]): Index of the GPU to be used by the current process.
    """
    if distributed:
        # Correctly handle rank setting in multiprocessing environments
        if dist_url == "env://":
            if rank == -1:
                rank = int(os.environ.get("RANK", 0))  # Use default rank 0 if not set

        # Adjust rank based on the number of GPUs per node if multiprocessing is enabled
        if multiprocessing_distributed and gpu is not None:
            rank = rank * ngpus_per_node + int(gpu)

        # Initialize the process group for distributed training
        utilis_worker.debug(
            f"Initializing distributed training initialized successfully: Rank={rank}, World Size={world_size}"
        )
        dist.init_process_group(
            backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank
        )

    else:
        utilis_worker.warning(
            "Distributed training is not enabled. This initialization is skipped."
        )


# TODO Fix the logic here
def save_checkpoint(
    state: Dict,
    is_best: bool,
    checkpoint_folder: str,
    arch: str,
    current_epoch: int,
    check_point_suffix: str,
    ckpt_save_interval: int = 200,
) -> None:
    """
    Save the model checkpoint during training. If the current checkpoint is the best model,
    it's saved under a special name. Regular checkpoints are saved at specified intervals.

    Args:
        state (Dict): Model state to be saved (parameters and other information).
        is_best (bool): Boolean indicating whether the current checkpoint is the best so far.
        checkpoint_folder (str): Folder path where the checkpoint is to be saved.
        arch (str): Architecture name, used in the directory structure.
        current_epoch (int): Current epoch number, used in the filename.
        check_point_suffix (str): Custom suffix for filename differentiation.
        ckpt_save_interval (int): Interval at which regular checkpoints are saved.
    """
    try:
        # Construct the path for the checkpoint directory
        arch_path: str = os.path.join(checkpoint_folder, arch)
        if not os.path.exists(arch_path):
            os.makedirs(name=arch_path)

        # Base file name includes architecture, custom suffix, and epoch
        base_filename: str = f"epoch{current_epoch}-{check_point_suffix}"

        # Determine the file path for the checkpoint
        if current_epoch % ckpt_save_interval == 0 or is_best:
            suffix: Literal["best"] | Literal["regular"] = (
                "best" if is_best else "regular"
            )
            filename: str = f"{base_filename}-{suffix}.pth"
            file_path: str = os.path.join(arch_path, filename)
            torch.save(obj=state, f=file_path)
            utilis_worker.debug(f"Checkpoint saved at {file_path}")

    except Exception as e:
        utilis_worker.error(f"Error saving checkpoint: {e}", exc_info=True)
