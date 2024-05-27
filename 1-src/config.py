import os
from collections import OrderedDict
from datetime import datetime

import pandas as pd
import yaml
from torch.utils.tensorboard import SummaryWriter


class ExperimentManager:
    def __init__(
        self,
        default_yaml_path: str,
        experiment_yaml_path: str,
    ):
        self.default_yaml_path = default_yaml_path
        self.experiment_yaml_path = experiment_yaml_path
        self.default_config = self.load_yaml(default_yaml_path)
        self.experiment_yaml_path = experiment_yaml_path
        self.experiment_config = self.load_yaml(experiment_yaml_path)

        self.interested_keys = {
            "training": ["lr", "batch_size"],
            "model": ["arch", "pretrained"],
            "optimizer": ["name", "momentum", "weight_decay", "betas", "criterion"],
            "scheduler": ["type", "mode", "factor", "patience", "warmup_steps"],
        }

        self.debug_mode = self.experiment_config["training"]["debug_mode"]

        self.differences = self.compare_configs(
            self.default_config, self.experiment_config
        )
        self.config_suffix = self.get_configuration_suffix(self.differences)

        self.commit_message = self.experiment_config.get("commit", {}).get(
            "commit_message", ""
        )

        self.event_timestamp = datetime.now().strftime("%Y-%m-%d||%H-%M-%S")

        self.experiment_log_dir = self.experiment_config["logging"][
            "experiment_log_dir"
        ]

        self.logger_dir = self.experiment_config["logging"]["logger_dir"]

    def load_yaml(self, path: str) -> OrderedDict:
        """
        Load a YAML file and return an OrderedDict of the contents.
        """
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def compare_configs(self, default_config: dict, experiment_config: dict):
        """
        Recursively compare two configuration dictionaries and return an OrderedDict of differences,
        including only the keys specified in interested_keys.

        Args:
            default_config (dict): The default configuration dictionary.
            experiment_config (dict): The experiment-specific configuration dictionary.

        Returns:
            OrderedDict: Differences between the default and experiment configurations.
        """
        differences = OrderedDict()

        def recurse(default, experiment, path=[]):
            for key, exp_value in experiment.items():
                new_path = path + [key]  # Build path to this item

                # Check if the current path is in the list of interested keys
                if len(new_path) == 1 and new_path[0] in self.interested_keys:
                    if key not in default:
                        # If key is not in default, add the new key (last key in new_path) and value to differences
                        differences[new_path[-1]] = exp_value
                    elif isinstance(exp_value, dict) and isinstance(default[key], dict):
                        # Recursively call for nested dictionaries
                        recurse(default[key], exp_value, new_path)
                    elif default[key] != exp_value:
                        # If values differ and are not dictionaries, add them to differences
                        differences[new_path[-1]] = exp_value
                elif (
                    len(new_path) == 2
                    and new_path[0] in self.interested_keys
                    and new_path[1] in self.interested_keys[new_path[0]]
                ):
                    if key not in default:
                        # If key is not in default, add the new key (last key in new_path) and value to differences
                        differences[new_path[-1]] = exp_value
                    elif default[key] != exp_value:
                        # If values differ and are not dictionaries, add them to differences
                        differences[new_path[-1]] = exp_value

        recurse(default_config, experiment_config)
        return differences

    def setup_logging_csv(self):
        """
        Setup a CSV log file to record experiment configurations and results, initializing
        columns for best epoch statistics and final test accuracy, with the 'time' column as a key.
        The 'commit_message' column is also added and populated with the value from self.commit_message.
        Columns for configuration differences are populated with actual data, while the new metric columns are initialized as empty.
        """
        log_path = self.experiment_config["log_csv_path"]

        # Define initial columns, including time and metrics columns
        initial_columns = (
            ["time", "commit_message"]
            + [
                "best_epoch",
                "best_train_acc1",
                "best_val_acc1",
                "best_test_acc1",
                "lr_at_best",
                "final_test_acc1",
            ]
            + list(self.differences.keys())
        )

        if not os.path.exists(log_path):
            # Initialize DataFrame with the correct order of columns, all columns empty initially except for 'time'
            log_df = pd.DataFrame(columns=initial_columns)
            print(f'Creating new log csv file at "{log_path}"...')
        else:
            log_df = pd.read_csv(log_path)
            # Ensure all expected columns exist, preserving order, adding new ones if necessary
            for column in initial_columns:
                if column not in log_df.columns:
                    log_df[column] = None
            # Reorder columns to the specified order
            log_df = log_df.reindex(columns=initial_columns)

        # Prepare new data row
        new_data = {
            key: None for key in initial_columns
        }  # Initialize all columns as None
        new_data.update(self.differences)  # Update with actual differences

        new_data["time"] = self.event_timestamp  # Set the current timestamp for 'time'
        new_data["commit_message"] = self.commit_message  # Set the commit message

        # Create a DataFrame for the new row and use pd.concat to append it
        new_row_df = pd.DataFrame([new_data])  # insertion first at the top
        log_df = pd.concat(
            [
                new_row_df,
                log_df,
            ],
            ignore_index=True,
        )

        log_df.to_csv(log_path, index=False)
        print("Experiment data logged successfully.")

    def update_experiment_metrics(self, timestamp, metrics):
        """
        Update the metrics for an experiment entry in the CSV log file based on the provided timestamp and metrics dictionary.

        Args:
            timestamp (str): The timestamp used to identify the specific experiment entry.
            metrics (dict): A dictionary containing metric values to update, which may include:
                            'best_epoch', 'best_train_acc', 'best_val_acc', 'lr_at_best', 'final_test_acc'.
        """
        log_path = self.experiment_config["log_csv_path"]

        # Load the existing log data
        log_df = pd.read_csv(log_path)

        # Check for new keys and add them as new columns if necessary
        for key in metrics.keys():
            if key not in log_df.columns:
                log_df[key] = None  # Initialize new column with None
                print(f"Added new column '{key}' to the CSV.")

        # Find the index of the row with the matching timestamp
        row_indices = log_df[log_df["time"] == timestamp].index
        if not row_indices.empty:
            # Ensure there is only one match; otherwise, raise an error
            if len(row_indices) > 1:
                raise ValueError(
                    f"Multiple entries found for timestamp '{timestamp}'. Expecting only one."
                )
            row_index = row_indices[0]  # Single index as scalar

            # Update the columns with values provided in metrics dictionary
            for key, value in metrics.items():
                if value is not None:
                    log_df.at[row_index, key] = value

            # Save the updated DataFrame back to the CSV
            log_df.to_csv(log_path, index=False)
            print("Experiment metrics updated successfully.")
        else:
            raise ValueError(f"Timestamp '{timestamp}' not found in the log CSV.")

    def get_configuration_suffix(self, config):
        """
        Generate a configuration suffix based on the provided configuration dictionary.

        Args:
            config (dict): Configuration dictionary with keys and values that define the experiment settings.

        Returns:
            str: The configuration suffix formed by joining key-value pairs.
        """
        configuration_suffix = "-".join(
            [f"{key}-{value}" for key, value in config.items()]
        )
        if self.debug_mode:
            configuration_suffix = f"debug-{configuration_suffix}"
        return configuration_suffix

    def setup_tensorboard_writer(self, config):
        """
        Set up a TensorBoard writer using the provided configuration dictionary.
        The writer will use a directory structure based on the configuration values.

        Args:
            config (dict): Configuration dictionary with keys and values that define the experiment settings.
        """
        # Get configuration suffix
        configuration_suffix = self.get_configuration_suffix(config)

        # Use the predefined event timestamp from the log
        timestamp = self.event_timestamp

        # Create a hierarchical folder structure based on configuration values, formatted as 'key-value'
        if self.debug_mode:
            folder_hierarchy = [self.experiment_log_dir, "debug"] + [
                f"{key}-{value}" for key, value in config.items()
            ]
        else:
            folder_hierarchy = [self.experiment_log_dir] + [
                f"{key}-{value}" for key, value in config.items()
            ]

        # Append the predefined timestamp under an 'event' subfolder to maintain consistent event logging
        folder_hierarchy += [
            timestamp,
            "event",
        ]

        # Construct the full directory path and create the directory if it doesn't exist
        full_event_dir = os.path.join(*folder_hierarchy)
        os.makedirs(full_event_dir, exist_ok=True)

        self.main_folder = os.path.dirname(full_event_dir)

        # Initialize the TensorBoard writer with the specified directory
        writer = SummaryWriter(
            log_dir=full_event_dir, filename_suffix=configuration_suffix
        )
        print(f"TensorBoard set up at {full_event_dir}")
        return writer

    def copy_experiment_yaml_to_main_folder(self):
        """Copies the experiment YAML file to the main folder."""
        import shutil

        if not self.main_folder:
            raise ValueError(
                "Main folder not set. Call the setup_tensorboard_writer first."
            )

        destination_path = os.path.join(
            self.main_folder, os.path.basename(self.experiment_yaml_path)
        )

        # Copy the file
        shutil.copy(self.experiment_yaml_path, destination_path)
        print(f"Yaml file copied to {destination_path} by ExperimentManager.")


def load_config_from_yaml():
    # 获取当前脚本文件的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建config.yaml文件的绝对路径
    yaml_file_path = os.path.join(script_dir, "config.yaml")
    with open(yaml_file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


config_from_yaml = load_config_from_yaml()

if __name__ == "__main__":
    # Define example YAML configurations as strings
    default_yaml_path = (
        "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/1-src/config.yaml"
    )

    experiment_yaml_path = (
        "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/1-src/yamls/config1.yaml"
    )

    # Instantiate the ExperimentManager with the example configurations
    manager = ExperimentManager(default_yaml_path, experiment_yaml_path)

    # Access and print the differences between the default and experiment configurations
    print("Differences between default and experiment configurations:")
    for key, value in manager.differences.items():
        print(f"{key}: {value}")

    # Print the configuration suffix to verify it is being generated correctly
    print("Configuration Suffix:", manager.config_suffix)

    # Simulate setting up the TensorBoard writer and print the expected directory setup
    writer = manager.setup_tensorboard_writer(manager.differences)
    print("TensorBoard writer setup completed. Check the log directory specified.")

    manager.setup_logging_csv()

    test_updated_results = {
        "best_epoch": 100,
        "best_train_acc1": 0.99,
        "best_val_acc1": 0.98,
        "best_test_acc1": 0.97,
        "lr_at_best": 3e-3,
        "final_test_acc1": 0.97,
    }
    manager.update_experiment_metrics(
        timestamp=manager.event_timestamp, metrics=test_updated_results
    )
