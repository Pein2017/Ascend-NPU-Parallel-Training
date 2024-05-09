import logging
import os
import random
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch_npu  # noqa: F401
from global_settings import commit_file_path, commit_message, logger_dir
from torch.utils.tensorboard import SummaryWriter


def setup_logger(
    name: str,
    log_file_name: str,
    logger_dir: str = logger_dir,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with a specified name, log level, and optional file handler.

    Args:
        name (str): Name of the logger, to distinguish between different parts of the application.
        log_file_name (str): Name of the log file.
        logger_dir (str): Directory to store the log file. Defaults to './logger'.
        level (int): Logging level, e.g., logging.INFO, logging.DEBUG.
        console (bool): Whether to output logs to the console.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.propagate = False
        logger.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

        if console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir, exist_ok=True)
        full_path = os.path.join(logger_dir, log_file_name)
        file_handler = logging.FileHandler(full_path, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


setup_utilis_logger: logging.Logger = setup_logger(
    name="SetupProcess",
    log_file_name="setup_process.log",
    level=logging.DEBUG,
    console=False,
)


def setup_deterministic_mode(seed: int = None) -> None:
    """
    Set deterministic training mode using a specified seed.

    Args:
    - seed (int, optional): The seed value for random number generators. If None, training will not be deterministic.

    Effects:
    - If a seed is provided, this function sets the seed for random number generation in Python's `random` module,
      PyTorch, and CUDA's cuDNN backend to ensure deterministic operations. Note that deterministic mode can
      potentially slow down training due to the disabling of certain optimizations.
    - If no seed is provided, training will proceed in non-deterministic mode.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        setup_utilis_logger.info(f"Seed set to {seed}. Training will be deterministic.")
        setup_utilis_logger.warning("Deterministic mode can slow down training.")
    else:
        setup_utilis_logger.info("Seed is not set. Training will not be deterministic.")


def setup_environment(master_addr: str, master_port: int, seed: int = None) -> None:
    """
    Configure environment variables for distributed training and set random seed for deterministic training.

    Args:
    - master_addr (str): Address of the master node for distributed training.
    - master_port (int): Port number on which the master node will communicate.
    - seed (int, optional): Seed for random number generation to ensure deterministic operations. If None, training may not be deterministic.

    Effects:
    - Sets environment variables for the distributed training master address and port.
    - Configures deterministic training if a seed is provided.
    """
    # Setting up distributed training parameters
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    setup_utilis_logger.info(
        f"Set MASTER_ADDR to {master_addr} and MASTER_PORT to {master_port}"
    )

    # Setup deterministic training mode using the explicitly passed seed
    setup_deterministic_mode(seed)


def log_commit_data(commit_file_path: str, commit_data: dict) -> None:
    """
    Log the setup details to a CSV file for record-keeping using a dictionary of commit data.

    Args:
        commit_file_path (str): The path to the CSV file where commit logs are saved.
        commit_data (dict): Dictionary containing 'Architecture', 'Configuration', 'Event File Dir', and 'Commit' information.
    """
    setup_utilis_logger.warning(
        "This function is deprecated, try to use TrainSetUpManager class instead"
    )
    # Specify column order and create the DataFrame with a single row of commit_data
    column_order = ["Architecture", "Configuration", "Event File Dir", "Commit"]
    df = pd.DataFrame([commit_data], columns=column_order)

    # Check if the CSV file exists, prepend the new data, and save it
    if os.path.exists(commit_file_path):
        # Load existing data
        existing_df = pd.read_csv(commit_file_path, index_col="Index")

        # Concatenate the new DataFrame at the top of the existing DataFrame
        combined_df = pd.concat([df, existing_df]).reset_index(drop=True)

        # Reindex to keep a continuous index and save to CSV
        combined_df.index.name = "Index"
        combined_df.to_csv(commit_file_path)
    else:
        # If the file does not exist, create new, set the index name, and save
        df.index.name = "Index"
        df.to_csv(commit_file_path)


def setup_tensorboard_and_commit(
    train_logger: logging.Logger,
    gpu: int,
    batch_size: int,
    arch: str,
    lr: float,
    optimizer_name: str,
    tb_log_dir: str,
    debug_mode: bool,
) -> Tuple[Optional[SummaryWriter], Optional[str]]:
    """
    Set up TensorBoard and log the configuration details to a CSV file. This function is intended
    to initialize TensorBoard on GPU 0 and log the session setup parameters for detailed tracking
    and reproducibility of experimental setups.

    Args:
        train_logger (logging.Logger): Logger for setup progress and events.
        gpu (int): GPU index, used to determine whether the setup should proceed.
        # NOTE: the batch size is already overall and not per GPU
        batch_size (int): Batch size per GPU, influencing the overall workload.
        arch (str): Model architecture designation.
        lr (float): Learning rate used in the optimizer.
        optimizer_name (str): Name of the optimizer in use.
        tb_log_dir (str): Base directory for saving TensorBoard log files.
        debug_mode (bool): Toggle to include additional debug outputs in the logs.

    Returns:
        Tuple[Optional[SummaryWriter], Optional[str]]:
            - SummaryWriter object if initialized, otherwise None.
            - Custom suffix for file naming based on configuration details and commit message.
    """
    setup_utilis_logger.warning(
        "This function is deprecated, try to use TrainSetUpManager class instead"
    )
    writer = None
    custom_suffix = None

    if gpu == 0:
        configuration_suffix = f"batch-{batch_size}-lr-{lr}-{optimizer_name}"
        if debug_mode:
            configuration_suffix = f"debug-{configuration_suffix}"
        custom_suffix = configuration_suffix

        save_event_path = os.path.join(tb_log_dir, "events", arch, configuration_suffix)
        os.makedirs(save_event_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_event_dir = os.path.join(save_event_path, timestamp)
        writer = SummaryWriter(log_dir=full_event_dir, filename_suffix=custom_suffix)
        train_logger.debug(
            f"TensorBoard set up at {full_event_dir} by (setup_tensorboard_and_commit)"
        )

        # Prepare commit data
        commit_data = {
            "Architecture": arch,
            "Configuration": custom_suffix,
            "Event File Dir": writer.log_dir,
            "Commit": commit_message,
        }
        # Log commit data to CSV
        log_commit_data(commit_file_path=commit_file_path, commit_data=commit_data)

    return writer, custom_suffix


class TrainingSetupManager:
    def __init__(
        self,
        train_logger: logging.Logger,
        gpu: int,
        tb_log_dir: str,
        commit_file_path: str,
        debug_mode: bool = False,
        commit_message: str = "Setup completed",
    ) -> None:
        self.train_logger = train_logger
        self.gpu = gpu
        self.tb_log_dir = tb_log_dir
        self.commit_file_path = commit_file_path
        self.debug_mode = debug_mode
        self.commit_message = commit_message

        self.event_timestamp = None

    def setup_tensorboard_and_commit(
        self,
        batch_size: int,
        arch: str,
        lr: float,
        optimizer_name: str,
        commit_message: str,
    ) -> Tuple[Optional[SummaryWriter], Optional[str], Optional[str]]:
        """
        Setup TensorBoard logging and commit the training configuration.

        Args:
            batch_size (int): The batch size used in training.
            arch (str): The architecture name of the model.
            lr (float): Learning rate.
            optimizer_name (str): The name of the optimizer used.
            commit_message (str): Message to log as part of the training commit.

        Returns:
            Tuple[Optional[SummaryWriter], Optional[str], Optional[str]]:
            A tuple containing a SummaryWriter instance or None, a custom suffix string or None,
            and a timestamp string or None.
        """
        writer = None
        custom_suffix = None
        timestamp = None

        if self.gpu == 0:
            configuration_suffix = f"batch-{batch_size}-lr-{lr}-{optimizer_name}"
            if self.debug_mode:
                configuration_suffix = f"debug-{configuration_suffix}"
            custom_suffix = configuration_suffix

            save_event_path = os.path.join(
                self.tb_log_dir, "events", arch, configuration_suffix
            )
            os.makedirs(save_event_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.event_timestamp = timestamp
            full_event_dir = os.path.join(save_event_path, timestamp)
            writer = SummaryWriter(
                log_dir=full_event_dir, filename_suffix=custom_suffix
            )
            self.train_logger.debug(f"TensorBoard set up at {full_event_dir}")

            # Prepare and log commit data
            commit_data = {
                "Architecture": arch,
                "Configuration": custom_suffix,
                "Event File Dir": writer.log_dir,
                "Commit": self.commit_message,
            }
            self.log_commit_data(commit_data)

        return writer, custom_suffix, timestamp

    def log_commit_data(self, commit_data: dict) -> None:
        """
        Log setup details to a CSV file for record-keeping using a dictionary of commit data.
        This method is enhanced to support arbitrary keys for logging additional metrics and ensures
        the most recent data is at the top of the file.

        Args:
            commit_data (dict): A dictionary containing training setup data and possibly new metrics.
        """
        if self.gpu != 0:
            return  # Only log from the primary GPU

        # Specify column order, starting with expected columns
        default_columns = ["Architecture", "Configuration", "Event File Dir", "Commit"]
        additional_columns = [key for key in commit_data if key not in default_columns]
        column_order = default_columns + additional_columns

        # Create a DataFrame with a single row of commit data
        df = pd.DataFrame([commit_data], columns=column_order)

        if os.path.exists(self.commit_file_path):
            # Load existing data
            existing_df = pd.read_csv(self.commit_file_path)
            # Prepend the new data by concatenating it before the existing data
            combined_df = pd.concat([df, existing_df], ignore_index=True, sort=False)
        else:
            # If the file does not exist, create new and log creation
            combined_df = df
            self.train_logger.info(
                f"Log file does not exist, creating new one at {self.commit_file_path}"
            )

        # Save to CSV without an additional index column
        combined_df.to_csv(self.commit_file_path, index=False)
        self.train_logger.info("Log data committed successfully.")

    def update_commit_log(self, event_file_dir, metrics_data):
        """
        Update the commit log with additional metrics data.

        Args:
            event_file_dir (str): The directory related to the event for which data is being logged.
            metrics_data (dict): Metrics to log, e.g., best accuracy or test scores. This dictionary
                                 can contain any number of additional metrics.

        Description:
            This method checks if the log file exists, updates it with new metrics or creates it if it doesn't exist.
        """
        self.train_logger.info("Attempting to update commit log with final results.")

        # Check if the log file exists
        if os.path.exists(self.commit_file_path):
            df = pd.read_csv(self.commit_file_path)

            # Check if the specific event directory already has a record
            row_index = df[df["Event File Dir"] == event_file_dir].index
            if not row_index.empty:
                # Update existing rows with new metrics
                for key, value in metrics_data.items():
                    if key not in df.columns:
                        df[key] = (
                            None  # Create new columns for metrics not previously logged
                        )
                    df.loc[row_index, key] = value
                df.to_csv(self.commit_file_path, index=False)
                self.train_logger.info("Commit log updated successfully.")
            else:
                # Append new row if no matching record is found
                new_data = {"Event File Dir": event_file_dir, **metrics_data}
                new_row = pd.DataFrame([new_data])
                df = pd.concat([df, new_row], ignore_index=True, sort=False)
                df.to_csv(self.commit_file_path, index=False)
                self.train_logger.info("New commit log entry created.")
        else:
            # Create a new log file with the specified metrics if it does not exist
            new_data = {"Event File Dir": event_file_dir, **metrics_data}
            df = pd.DataFrame([new_data])
            df.to_csv(self.commit_file_path, index=False)
            self.train_logger.info(
                "Commit log file created as it did not previously exist."
            )


if __name__ == "__main__":
    # Setup logger
    train_logger = setup_utilis_logger

    # Initialize the training setup manager
    manager = TrainingSetupManager(
        train_logger=train_logger,
        gpu=0,
        tb_log_dir="/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/3-tb_logs/",
        commit_file_path="/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/3-tb_logs/commit_log.csv",
        debug_mode=True,
        commit_message="Test for TrainingSetupManager class",
    )

    # Simulate setting up TensorBoard and initial commit
    writer, custom_suffix, timestamp = manager.setup_tensorboard_and_commit(
        batch_size=32,
        arch="test_arch",
        lr=0.001,
        optimizer_name="Adam",
        commit_message="Test for TrainingSetupManager class",
    )

    # Simulate updating the commit log with additional data after a hypothetical test
    additional_metrics = {"best_acc1": -50, "best_epoch": -10, "test_top1": -0.84}
    manager.update_commit_log(
        event_file_dir=writer.log_dir, metrics_data=additional_metrics
    )
    print("test finished!")
