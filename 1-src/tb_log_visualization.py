import logging
import os

import matplotlib.pyplot as plt
from setup_utilis import setup_logger
from tensorboard.backend.event_processing import event_accumulator


class TBLogExporter:
    def __init__(
        self,
        event_folder_path: str,
        custom_suffix: str,
        tb_logger: logging.Logger,
        event_prefix: str = "events.out.tfevents.",
        fig_exported_dir: str = None,
    ) -> None:
        """
        Initialize the TBLogExporter class to manage and export data from TensorBoard log files.

        Args:
            event_folder_path (str): The base directory where TensorBoard event files are stored.
            custom_suffix (str): Suffix to identify specific event files.
            tb_logger (logging.Logger): Logger for logging information and warnings.
            event_prefix (str): Prefix that TensorBoard event files start with.
        """
        if not os.path.exists(event_folder_path):
            tb_logger.error(f"Event log path does not exist: {event_folder_path}")
            raise FileNotFoundError(
                f"Event log path does not exist: {event_folder_path}"
            )

        self.event_folder_path = event_folder_path
        self.custom_suffix = custom_suffix
        self.tb_logger = tb_logger
        self.event_prefix = event_prefix
        self.arch = os.path.basename(os.path.dirname(event_folder_path))

        if not fig_exported_dir:
            self.fig_exported_dir = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(self.event_folder_path))
                )
            )
            print(f"before: {self.fig_exported_dir}")
            self.fig_exported_dir = os.path.join(self.fig_exported_dir, "figs")
        else:
            # recommended format {dir_path/figs}
            pass
        self.event_file_name, self.experiment_number = self.find_latest_log_event()
        print("************")
        print(self.event_file_name, self.experiment_number)
        print("************")

    def find_latest_log_event(self):
        """
        Find the latest TensorBoard log event file from the most recent timestamp directory
        and count the total number of experiments based on the number of timestamp directories.

        Returns:
            Tuple[str, int]: The latest log file name and the experiment count.
        """
        parent_dir = os.path.dirname(self.event_folder_path)
        # List all timestamp directories in the parent directory of the specified event folder path
        timestamp_dirs = [
            d
            for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d))
        ]

        if not timestamp_dirs:
            self.tb_logger.error(
                f"No timestamp directories found in {self.event_folder_path}"
            )
            raise FileNotFoundError(
                f"No timestamp directories found in {self.event_folder_path}"
            )

        # Sort directories to find the latest
        timestamp_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(parent_dir, x)))
        latest_dir = timestamp_dirs[-1]
        full_path_to_latest_dir = os.path.join(parent_dir, latest_dir)

        # Listing event files within the latest directory
        event_files = [
            f
            for f in os.listdir(full_path_to_latest_dir)
            if f.startswith(self.event_prefix) and f.endswith(self.custom_suffix)
        ]
        event_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(full_path_to_latest_dir, f))
        )
        latest_file = event_files[-1] if event_files else None

        experiment_number = len(timestamp_dirs)  # Count of timestamp directories

        return latest_file, experiment_number

    def load_tb_event(self, event_file_name: str):
        """
        Load TensorBoard events from a specified log file.

        Args:
            event_file_name (str): Name of the log file to load events from.

        Returns:
            EventAccumulator: Loaded events or None if the file cannot be found or loaded.
        """
        event_file_path = os.path.join(self.event_folder_path, event_file_name)
        if not os.path.exists(event_file_path):
            self.tb_logger.error(f"Event file does not exist: {event_file_path}")
            return None

        tb_event = event_accumulator.EventAccumulator(event_file_path)
        tb_event.Reload()
        return tb_event

    def plot_metrics(
        self,
        tb_event: event_accumulator.EventAccumulator,
        grouped_metrics: dict,
        fig_name: str,
    ) -> None:
        """
        Plot metrics from the TensorBoard event file.

        Args:
            tb_event (EventAccumulator): Loaded TensorBoard event data.
            grouped_metrics (dict): Group names as keys, list of metric names as values.
            fig_name (str): The file name for the saved plot.
        """
        if tb_event is None:
            self.tb_logger.error("TensorBoard event data is not loaded.")
            return

        num_subplots = len(grouped_metrics)
        fig, axes = plt.subplots(
            num_subplots, 1, figsize=(10, 5 * num_subplots), squeeze=False
        )
        axes = axes.flatten()  # Flatten the axes array for easy indexing

        # Plot each group of metrics
        for i, (group_name, metrics) in enumerate(grouped_metrics.items()):
            ax = axes[i]
            for metric in metrics:
                if metric not in tb_event.Tags().get("scalars", []):
                    self.tb_logger.warning(
                        f"Metric {metric} not found in TensorBoard data."
                    )
                    continue
                data = tb_event.scalars.Items(metric)
                if not data:
                    self.tb_logger.warning(f"No data found for metric {metric}.")
                    continue

                ax.plot(
                    [int(item.step) for item in data],
                    [item.value for item in data],
                    label=metric.split("/")[-1],
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel(group_name)
            ax.legend()
            ax.set_title(f"{group_name} Over Epochs")

        plt.tight_layout()
        save_path = self.construct_save_path(fig_name=fig_name)
        plt.savefig(save_path)
        plt.close(fig)
        self.tb_logger.info(f"Figure exported successfully to {save_path}")

    def construct_save_path(self, fig_name: str) -> str:
        # Construct the final save path for the plot
        save_path = os.path.join(
            self.fig_exported_dir,
            self.arch,
            f"{self.custom_suffix}-exp{self.experiment_number}-{fig_name}",
        )

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        return save_path

    def export(self, grouped_metrics: dict, fig_name: str) -> None:
        """
        Export metrics from TensorBoard to a plot.

        Args:
            grouped_metrics: Metrics to plot, grouped by category.
            fig_name: Name for the output plot file.
        """
        tb_event = self.load_tb_event(event_file_name=self.event_file_name)
        self.plot_metrics(
            tb_event=tb_event, grouped_metrics=grouped_metrics, fig_name=fig_name
        )


if __name__ == "__main__":
    tb_logger = setup_logger(
        name="TBLogExporter",
        log_file_name="tb_log_exporter.log",
        level=logging.DEBUG,
        console=True,
    )

    # Define the directory and custom suffix for the TensorBoard logs
    custom_suffix = "debug-batch-1024-lr-1e-1-SGD"
    fig_name = "metrics.png"
    event_folder_path = "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/3-tb_logs/events/resnet18/debug-batch-1024-lr-1e-1-SGD/2024-05-06 11:17:38"

    # Ensure the log directory exists
    if not os.path.exists(event_folder_path):
        tb_logger.error(f"Log directory does not exist: {event_folder_path}")
        raise FileNotFoundError(f"Log directory does not exist: {event_folder_path}")

    # Instantiate the TBLogExporter class
    exporter = TBLogExporter(
        event_folder_path=event_folder_path,
        custom_suffix=custom_suffix,
        tb_logger=tb_logger,
    )

    # Define the metrics to be plotted
    grouped_metrics = {
        "Loss": ["Loss/train", "Loss/val"],
        "Top1": ["Top1/train", "Top1/val"],
        "Top5": ["Top5/train", "Top5/val"],
        "Learning Rate": ["Learning Rate"],
    }

    # Export the metrics to a specified figure file
    exporter.export(grouped_metrics=grouped_metrics, fig_name=fig_name)
