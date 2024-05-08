"""逐行测试时启用"""
# import sys

# sys.path.append('/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/src')

import logging
import os
import time
from multiprocessing import Array
from multiprocessing.sharedctypes import SynchronizedArray
from typing import Any, Dict, List, Tuple

import torch.multiprocessing as mp
import torch.utils.data.distributed
from config import config_from_yaml as config
from data_loader import verify_and_download_dataset
from model import CIFARNet, load_or_create_model
from setup_utilis import setup_environment, setup_logger
from utilis import device_id_to_process_device_map
from worker import main_worker

#! Clean up all the previs logs
logs_path = "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/4-logger/"
if os.path.exists(logs_path):
    # List all files in the directory
    files = os.listdir(logs_path)
    # Filter for files that end with .log
    log_files = [file for file in files if file.endswith(".log")]
    # Remove each found log file
    for log_file in log_files:
        file_path = os.path.join(logs_path, log_file)
        os.remove(file_path)
        print(f"Removing {file_path}")


main_logger: logging.Logger = setup_logger(
    name="MainProcess",
    log_file_name="main_process.log",
    level=logging.DEBUG,
    console=True,
)
main_logger.info(msg="Logger initialized.")


def start_worker(config: Dict) -> Tuple[float, int, float]:
    """
    Execute the training process based on provided configuration, managing device
    assignment and initialization of distributed training.

    Args:
        config (Dict): Configuration settings including device and distributed training specifics.

    Returns:
        Tuple[float, int, float]: Tuple containing the best accuracy, corresponding epoch,
        and top-1 accuracy average.
    """
    dist_training: Dict = config["distributed_training"]
    dist_url: str = dist_training["master_addr"]
    world_size: int = dist_training["world_size"]
    multiprocessing_distributed: bool = dist_training.get(
        "multiprocessing_distributed", False
    )

    # Automatically configure world size from environment if using environment setup for distributed training
    if dist_url == "env://" and world_size == -1:
        world_size = int(x=os.environ["WORLD_SIZE"])
    distributed: bool = world_size > 1 or multiprocessing_distributed

    # Map device IDs to physical device names
    device: str = dist_training["device"]
    device_list: List[str] | List[int] = dist_training["device_list"]
    process_device_map: Dict[int, int] = device_id_to_process_device_map(
        device_list=device_list
    )
    config["distributed_training"]["process_device_map"] = process_device_map
    config["distributed_training"]["distributed"] = distributed

    ngpus_per_node: int = (
        len(process_device_map) if device == "npu" else torch.cuda.device_count()
    )

    if distributed:
        # Adjust world size based on the number of GPUs/NPUs per node
        world_size *= ngpus_per_node
        dist_training["world_size"] = world_size

        # Log model loading
        main_logger.debug(msg="Loading model...")
        model: CIFARNet = load_or_create_model(config=config)
        main_logger.debug(msg=f'Model {config["model"]["arch"]} downloaded.')
        del model

        result: SynchronizedArray[Any] = Array(
            "d", [-1.0, -1, -1.0], lock=False
        )  # Initilize by a fake result to avoid error
        main_logger.debug(
            msg="Worker processes spawned successfully. Waiting for results..."
        )

        mp.spawn(
            main_worker,
            args=(ngpus_per_node, config, result),
            nprocs=ngpus_per_node,
            join=True,
        )

        main_logger.debug(msg=f"Results received. {result}")
        return result[0], int(result[1]), result[2]
    else:
        raise Exception("Single process training is not supported currently!")


def main(config: Dict) -> None:
    """Run the main application workflow."""

    master_addr: str = config["distributed_training"]["master_addr"]
    master_port: int = config["distributed_training"]["master_port"]
    seed: int = config["training"].get("seed", None)
    setup_environment(master_addr=master_addr, master_port=master_port, seed=seed)

    dataset_path: str = config["data"]["path"]
    dataset_name: str = config["data"]["dataset_name"]
    verify_and_download_dataset(
        dataset_name=dataset_name, dataset_path=dataset_path, logger=main_logger
    )

    result: None | Tuple[float, int, float] = start_worker(config=config)
    best_acc1: float
    best_epoch: int
    top1: float

    best_acc1, best_epoch, top1 = result

    main_logger.info(
        f"Finished! Best validation accuracy: {best_acc1} at epoch {best_epoch}. Test top1 accuracy: {top1}"
    )


if __name__ == "__main__":
    start_time: float = time.time()
    main(config=config)
    end_time: float = time.time()

    elapsed_time_seconds: float = end_time - start_time
    hours: float
    remainder: float
    minutes: float
    seconds: float

    hours, remainder = divmod(elapsed_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    main_logger.info(
        f"Total time cost: {int(hours):02d}h:{int(minutes):02d}mins:{int(seconds):02d}s"
    )
    main_logger.info(
        f'lr: {config["training"]["lr"]} , batch_size: {config["training"]["batch_size"]}'
    )
