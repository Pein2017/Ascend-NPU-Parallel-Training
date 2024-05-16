import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from multiprocessing import Array
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch_npu  # noqa
from config import ExperimentManager
from config import config_from_yaml as config
from data_loader_class import DataLoaderManager
from setup_utilis import setup_logger
from torch.utils.tensorboard import SummaryWriter
from worker_class import Worker


def initialize_distributed_environment(backend="hccl", init_method="env://"):
    """
    Initialize the distributed environment.
    Args:
        backend (str): The backend to use for distributed processing.
        init_method (str): The method to initialize the process group.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method=init_method)

    print(f"Distributed Environment initialized with backend {backend}.")
    # Print out rank and world size for verification
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} reporting for duty.")
    torch.npu.set_device(rank)


class MainManager:
    def __init__(
        self,
        config: Dict,
        writer: Optional[SummaryWriter],
        custom_suffix: Optional[str],
    ):
        self.config = config
        self.writer = writer
        self.custom_suffix = custom_suffix

        self.logger_dir = config["logging"]["logger_dir"]

        self.start_time = None
        self.main_logger = None

        self.gpu = int(os.getenv("LOCAL_RANK", "0"))
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Initialize the result array with a length of 5, setting default values to -1.0
        # The indices correspond to: best_epoch, best_train_acc, best_val_acc, lr_at_best, final_test_acc
        self.result = Array("d", [-1.0] * 6, lock=False)

    def setup_main_logger(self):
        """Setup specific configurations for the manager based on the rank."""
        self.clean_logs(self.logger_dir)
        self._setup_logger()
        self.main_logger.info(f"Main logger is set up on node {self.rank}.")
        self.main_logger.info(
            f"Distributed training initialized with backend: {self.config['distributed_training']['dist_backend']}, init_method: {self.config['distributed_training']['dist_url']}"
        )

    @staticmethod
    def clean_logs(directory):
        """Remove previous log files in the given directory."""
        if os.path.exists(directory):
            files = [file for file in os.listdir(directory) if file.endswith(".log")]
            for log_file in files:
                os.remove(os.path.join(directory, log_file))
                print(f"Removed {log_file}")

    def _setup_logger(self):
        """Initialize the main logger. Should be called on the master node only."""
        if dist.is_initialized():
            self.main_logger = setup_logger(
                name="MainProcess",
                log_file_name="main_process.log",
                level=logging.DEBUG,
                console=True,
            )
        else:
            raise RuntimeError(
                "Distributed training is not initialized when setting up logger."
            )

        self.main_logger.debug("Main Logger initialized.")
        self.start_time = time.time()

    def setup_deterministic_mode(self):
        """Set deterministic behavior based on the specified seed in the configuration."""
        seed = self.config["training"].get("seed", None)
        is_deteriminstic = self.config["training"].get("is_deteriminstic", False)
        if seed is not None:
            import random

            import numpy as np
            import torch.backends.cudnn as cudnn

            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            if is_deteriminstic:
                cudnn.deterministic = True
                cudnn.benchmark = False  # TODO: change this
            else:
                cudnn.deterministic = False
                cudnn.benchmark = True

            # Ensure only the main process logs the deterministic setting
            if self.is_master_node():
                print(f"Seed set to {seed}. Training will be deterministic.")
                Warning("Deterministic mode can slow down training.")
        else:
            if self.is_master_node():
                print("Seed is not set. Training will not be deterministic.")

    def verify_and_download_data(self):
        """Perform dataset verification and downloading on master node only."""
        if self.is_master_node():
            # dataset_path = self.config["data"]["path"]
            dataset_name = self.config["data"]["dataset_name"]
            if dataset_name == "cifar10":
                dataset_path = (
                    "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/cifar100_data"
                )
            elif dataset_name == "cifar100":
                dataset_path = (
                    "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/cifar100_data"
                )
            else:
                raise ValueError(f"dataset_name: {dataset_name} is not supported.")

            data_loader_manager = DataLoaderManager(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                logger=self.main_logger,
                use_fake_data=config["data"]["use_dummy"],
            )
            data_loader_manager.verify_and_download_dataset()
            del data_loader_manager
            self.main_logger.info("Dataset verified and downloaded successfully.")

    def run_training(self):
        """Execute the distributed training and final logging."""

        if self.is_master_node():
            self.main_logger.info("Attempting to synchronize all processes...")

        # Synchronize all processes at the start
        dist.barrier()

        if self.is_master_node():
            self.main_logger.info("All processes synchronized successfully.")

        # Record start time
        self.start_time = time.time()

        self.start_worker_and_get_result()

        (
            best_epoch,
            best_train_acc1,
            best_val_acc1,
            best_test_acc1,
            lr_at_best,
            test_acc1,
        ) = self.result

        # Synchronize all processes at the end
        dist.barrier()

        # Counting the total time taken for training
        if self.is_master_node():
            end_time = time.time()
            elapsed_time_seconds = end_time - self.start_time
            hours, remainder = divmod(elapsed_time_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            self.main_logger.info("\nTraining Finished!.")
            self.main_logger.info(
                f"Total time cost: {int(hours)}h:{int(minutes)}mins:{int(seconds)}s"
            )
            self.main_logger.info(
                f"Best training accuracy: {best_train_acc1:.4f}, Best validation accuracy: {best_val_acc1:.4f}, Best epoch: {best_epoch}, LR at best: {lr_at_best:.6f}, Final test accuracy: {test_acc1:.4f}"
            )
            self.main_logger.info(
                f'lr: {self.config["training"]["lr"]}, batch_size: {self.config["training"]["batch_size"]}, optimizer: {self.config["optimizer"]["name"]}'
            )

    def start_worker_and_get_result(self):
        """Start a Worker instance to handle the main training tasks."""

        worker = Worker(
            config=self.config,
            result=self.result,
            writer=self.writer,
            custom_suffix=self.custom_suffix,
        )
        worker.execute_main_task()

    def is_master_node(
        self,
    ) -> bool:
        if dist.is_initialized() and self.rank == 0:
            return True
        else:
            return False


def main(default_yaml_path: str, experiment_yaml_path: str):
    initialize_distributed_environment(
        backend="hccl", init_method="env://"
    )  #! here is a hardcore setting

    if dist.is_initialized():
        print("Distributed environment initialized.")
    else:
        raise RuntimeError("Distributed environment not initialized.")

    exp_manager = ExperimentManager(
        default_yaml_path=default_yaml_path,
        experiment_yaml_path=experiment_yaml_path,
    )

    if dist.get_rank() == 0:
        custom_suffix = exp_manager.config_suffix
        writer = exp_manager.setup_tensorboard_writer(exp_manager.differences)

        exp_manager.setup_logging_csv()
        print("Experiment data logged successfully.")

        exp_manager.copy_experiment_yaml_to_main_folder()
        print("Experiment YAML copied successfully.")
    else:
        custom_suffix = None
        writer = None

    main_manager = MainManager(
        exp_manager.experiment_config, writer=writer, custom_suffix=custom_suffix
    )
    main_manager.setup_deterministic_mode()  # Called on every process

    if dist.get_rank() == 0:
        main_manager.setup_main_logger()
        main_manager.verify_and_download_data()

    main_manager.run_training()

    if dist.get_rank() == 0:
        result = main_manager.result
        updated_metrics = {
            "best_epoch": int(result[0]),
            "best_train_acc1": result[1],
            "best_val_acc1": result[2],
            "best_test_acc1": result[3],
            "lr_at_best": result[4],
            "final_test_acc1": result[5],
        }
        exp_manager.update_experiment_metrics(
            timestamp=exp_manager.event_timestamp, metrics=updated_metrics
        )


def cleanup():
    if dist.is_initialized():
        print("Cleaning up distributed process group...")
        dist.destroy_process_group()
        print("Distributed process group cleaned up!")

    # Kill all related Python processes to ensure clean termination
    current_pid = os.getpid()
    for proc in mp.active_children():
        if proc.pid != current_pid:
            print(f"Terminating process {proc.pid}")
            proc.terminate()


def signal_handler(sig, frame):
    print("Received signal to terminate. Cleaning up...")
    cleanup()


if __name__ == "__main__":
    # Register signal handler for clean termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        default_yaml_path = (
            "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/1-src/config.yaml"
        )
        experiment_yaml_path = (
            "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/1-src/yamls/config_2.yaml"
        )
        main(default_yaml_path, experiment_yaml_path)
        print("\n")
        print("*" * 50)
        print("Training completed!")
        print("*" * 50)
        print("\n")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception:
        print("Error during main:")
        traceback.print_exc()  # This will print the detailed traceback
        # Forcefully terminate the program (mimicking Ctrl+C behavior)
        print("Forcefully terminating the distributed training...")
        cleanup()
        sys.exit(1)  # Exit with error status
    finally:
        cleanup()

# def main(config):
#     try:
#         manager = DistributedTrainingManager(config)
#         manager.initialize_distributed_training()
#         manager.setup_deterministic_mode()
#         manager.verify_and_download_data()
#         manager.run_training()
#     except Exception as e:
#         print(f"Errror occurred: \n{e}")
#     finally:
#         if dist.is_initialized():
#             print("Cleaning up distributed process group...")
#             dist.destroy_process_group()


# if __name__ == "__main__":
#     try:
#         main(config)
#     except KeyboardInterrupt:
#         print("Training interrupted by user.")
#     except Exception as e:
#         print(f"Unhandled exception during training setup or execution: {e}")
