import argparse
import datetime
import logging
import os
import shutil
import sys
import time
import traceback
from multiprocessing import Array
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch_npu  # noqa
from torch.utils.tensorboard import SummaryWriter


def initialize_distributed_environment(backend="hccl", init_method="env://"):
    """
    Initialize the distributed environment.
    Args:
        backend (str): The backend to use for distributed processing.
        init_method (str): The method to initialize the process group.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method=init_method)

    # Print out rank and world size for verification
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if dist.get_rank() == 0:
        print(f"Distributed Environment initialized with backend {backend}.")
    print(f"Rank {rank}/{world_size} reporting for duty.")
    torch.npu.set_device(rank)


# Initialize the distributed environment before importing self-defined packages
initialize_distributed_environment()


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

        self.logger_dir = None
        self.update_logger_dir()

        self.start_time = None
        self.main_logger = None

        self.gpu = int(os.getenv("LOCAL_RANK", "0"))
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Initialize the result array with a length of 5, setting default values to -1.0
        # The indices correspond to: best_epoch, best_train_acc, best_val_acc, lr_at_best, final_test_acc
        self.result = Array("d", [-1.0] * 6, lock=False)

    def update_logger_dir(self):
        """Update the logger directory based on existing folders and check the date."""
        base_logger_dir = self.config["logging"]["logger_dir"]
        timestamp = base_logger_dir.split("/")[-2]  # Extract the date part

        # Check if the date in the folder matches today's date
        try:
            folder_date = datetime.datetime.strptime(timestamp, "%y-%m-%d").date()
            current_date = datetime.date.today()
            expected_date = datetime.date(
                current_date.year, current_date.month, current_date.day
            )
            if folder_date != expected_date:
                print(
                    f"Warning: The folder date {folder_date} does not match today's date {expected_date}."
                )
        except ValueError:
            raise ValueError(f"The folder date format {timestamp} is incorrect.")

        # Base directory for experiments
        exp_base_dir = os.path.dirname(base_logger_dir)

        # Count existing Exp folders
        if not os.path.exists(exp_base_dir):
            os.makedirs(exp_base_dir)

        existing_folders = [
            d
            for d in os.listdir(exp_base_dir)
            if os.path.isdir(os.path.join(exp_base_dir, d))
        ]
        exp_numbers = [
            int(folder.replace("Exp", ""))
            for folder in existing_folders
            if folder.startswith("Exp") and folder.replace("Exp", "").isdigit()
        ]

        if exp_numbers:
            next_exp_num = max(exp_numbers) + 1
        else:
            next_exp_num = 1
            print("Starting new experiment folder. \n ")

        new_logger_dir = os.path.join(exp_base_dir, f"Exp{next_exp_num}")
        self.config["logging"]["logger_dir"] = new_logger_dir
        self.logger_dir = new_logger_dir
        if dist.get_rank() == 0:
            print(f"Logger directory updated to '{new_logger_dir}'")

    def setup_main_logger(self):
        """Setup specific configurations for the manager based on the rank."""
        from global_settings import DEFAULT_LOGGER_DIR

        self.DEFAULT_LOGGER_DIR = DEFAULT_LOGGER_DIR

        self.clean_logs(DEFAULT_LOGGER_DIR)

        self._setup_logger()
        self.main_logger.debug(f"Main logger is set up on node {self.rank}.")
        self.main_logger.debug(
            f"Distributed training initialized with backend: {self.config['distributed_training']['dist_backend']}, init_method: {self.config['distributed_training']['dist_url']}"
        )

    @staticmethod
    def clean_logs(directory):
        """Remove previous log files in the given directory."""
        if os.path.exists(directory):
            files = [file for file in os.listdir(directory) if file.endswith(".log")]
            for log_file in files:
                os.remove(os.path.join(directory, log_file))
                # print(f"Removed {log_file} by MainManager.")

    def _setup_logger(self):
        """Initialize the main logger. Should be called on the master node only."""
        from setup_utilis import setup_logger

        if dist.is_initialized():
            self.main_logger = setup_logger(
                name="MainProcess",
                log_file_name="main_process.log",
                level=logging.INFO,
                console=True,
            )
        else:
            raise RuntimeError(
                "Distributed training is not initialized when setting up logger."
            )

        self.main_logger.debug("Main Logger initialized by MainManager.")
        # self.start_time = time.time()

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
            if self.rank == 0:
                print(f"Seed set to {seed}. Training will be deterministic.")
                Warning("Deterministic mode can slow down training.")
        else:
            if self.rank == 0:
                print("Seed is not set. Training will not be deterministic.")

    def verify_and_download_data(self):
        """Perform dataset verification and downloading on master node only."""
        from data_loader_class import DataLoaderManager

        if self.rank == 0:
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
                use_fake_data=self.config["data"]["use_dummy"],
            )
            data_loader_manager.verify_and_download_dataset()
            del data_loader_manager
            self.main_logger.info("Dataset verified and downloaded successfully.")
        else:
            pass

    def run_training(self):
        """Execute the distributed training and final logging."""

        if self.rank == 0:
            pass
            # self.main_logger.info("Attempting to synchronize all processes...")

        # Synchronize all processes at the start
        dist.barrier()

        if self.rank == 0:
            pass
            # self.main_logger.info("All processes synchronized successfully.")

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
        if self.rank == 0:
            end_time = time.time()
            elapsed_time_seconds = end_time - self.start_time
            hours, remainder = divmod(elapsed_time_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            self.main_logger.info("\n \nTraining Finished! \n")
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
        from worker import Worker

        worker = Worker(
            config=self.config,
            result=self.result,
            writer=self.writer,
            custom_suffix=self.custom_suffix,
        )
        # print("here to worker!")
        worker.execute_main_task()

    def _copy_loggers(self, src=None, dest=None):
        src = src or self.DEFAULT_LOGGER_DIR
        dest = dest or self.logger_dir

        # Ensure logger directories exist
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source logger directory '{src}' not found.")

        if not os.path.exists(dest):
            os.makedirs(dest)

        # Copy .log files from src to dest, excluding subdirectories
        for filename in os.listdir(src):
            if filename.endswith(".log"):
                src_file = os.path.join(src, filename)
                dest_file = os.path.join(dest, filename)
                shutil.copy2(src_file, dest_file)
                # print(f"Copied '{src_file}' to '{dest_file}'")

        # Optionally, confirm that the files were copied
        print(f"All .log files copied from '{src}' to '{dest}'.")


def main(default_yaml_path: str, experiment_yaml_path: str):
    from config import ExperimentManager

    # Since I initialize this outside the main
    if not dist.is_initialized():
        initialize_distributed_environment(
            backend="hccl", init_method="env://"
        )  #! here is a hardcore setting

    if dist.is_initialized():
        if dist.get_rank() == 0:
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
            "best_train_acc1": round(result[1], 4),
            "best_val_acc1": round(result[2], 4),
            "best_test_acc1": round(result[3], 4),
            "lr_at_best": result[4],
            "final_test_acc1": round(result[5], 4),
        }
        exp_manager.update_experiment_metrics(
            timestamp=exp_manager.event_timestamp, metrics=updated_metrics
        )

        main_manager._copy_loggers()

        print("main finished! \n")


def cleanup():
    if dist.is_initialized():
        print("Cleaning up distributed process group...")
        dist.destroy_process_group()
        print("Distributed process group cleaned up!")

    # # Kill all related Python processes to ensure clean termination
    # current_pid = os.getpid()
    # for proc in mp.active_children():
    #     if proc.pid != current_pid:
    #         print(f"Terminating process {proc.pid}")
    #         proc.terminate()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiments with different YAML configurations."
    )
    parser.add_argument(
        "--default_yaml_path",
        type=str,
        default="/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/1-src/config.yaml",
        help="Path to the default YAML configuration file.",
    )
    parser.add_argument(
        "--experiment_yaml_folder",
        type=str,
        required=True,
        help="Path to the folder containing experiment YAML files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    start_time = time.time()
    try:
        finished_files = set()
        total_finished = 0

        while True:
            # Get list of YAML files in the experiment YAML folder using os
            yaml_files = [
                f
                for f in os.listdir(args.experiment_yaml_folder)
                if f.endswith(".yaml")
            ]
            yaml_files.sort()

            # Filter out files that have already been finished
            to_be_finished_files = [f for f in yaml_files if f not in finished_files]

            # If there are no more files to be processed, break the loop
            if not to_be_finished_files:
                print("All YAML files have been processed. Exiting.")
                break

            # Calculate total_experiments each iteration
            total_experiments = len(finished_files) + len(to_be_finished_files)

            for index, yaml_file in enumerate(to_be_finished_files):
                experiment_yaml_path = os.path.join(
                    args.experiment_yaml_folder, yaml_file
                )
                if dist.get_rank() == 0:
                    print("\n")
                    print("~" * 50)
                    print(
                        f"\nStart running experiment {total_finished + 1}/{total_experiments} \nwith config: {experiment_yaml_path} \n"
                    )
                    print("~" * 50)
                    print("\n")

                main(args.default_yaml_path, experiment_yaml_path)
                if dist.get_rank() == 0:
                    print("\n")
                    print("*" * 50)
                    print(
                        f"Experiment {total_finished + 1}/{total_experiments} completed!"
                    )
                    print("*" * 50)
                    print("\n")

                # Mark the current file as finished and increment the finished counter
                finished_files.add(yaml_file)
                total_finished += 1

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
        if dist.get_rank() == 0:
            end_time = time.time()
            elapsed_time_seconds = end_time - start_time
            hours, remainder = divmod(elapsed_time_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(
                f"\n\nTotal runtime for all experiments: {int(hours)}h:{int(minutes)}m:{int(seconds)}s"
            )
            print(f"Total number of experiments completed: {total_finished}")

        cleanup()
