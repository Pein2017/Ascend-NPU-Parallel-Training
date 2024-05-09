import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from metric_utilis import MetricTracker
from optimizer import OptimizerManager, SchedulerManager
from setup_utilis import TrainingSetupManager, setup_logger
from stats_tracker import ModelStatsTracker
from tb_log_visualization import TBLogExporter
from torch.nn import Module
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from train_utilis import create_meters, process_batch, update_meters


class Trainer:
    def __init__(
        self,
        model: Module,
        criterion: Module,
        optimizer_manager: OptimizerManager,
        scheduler_manager: SchedulerManager,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader],
        device: torch.device,
        epochs: int,
        start_epoch: int,
        print_freq: int,
        arch: str,
        batch_size: int,
        lr: float,
        optimizer_name: str,
        verbose: bool,
        is_validation_enabled: bool,
        is_distributed: bool,
        is_amp_enabled: bool,
        amp: Optional[Any],
        hist_record_num: int,
        train_logger: Optional[logging.Logger] = None,
        model_stats_tracker: Optional[ModelStatsTracker] = None,
        train_sampler: Optional[Sampler] = None,
        val_sampler: Optional[Sampler] = None,
        tb_log_dir: Optional[str] = None,
        debug_mode: bool = False,
        ckpt_save_interval: int = 300,
        checkpoint_folder: Optional[str] = "./checkpoints",
        commit_message: Optional[str] = "No commit message",
        commit_file_path: Optional[str] = "./commit.csv",
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer_manager = optimizer_manager
        self.scheduler_manager = scheduler_manager
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.print_freq = print_freq
        self.arch = arch
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer_name = optimizer_name

        self.verbose = verbose
        self.is_validation_enabled = is_validation_enabled
        self.is_distributed = is_distributed
        self.is_amp_enabled = is_amp_enabled
        self.amp = amp
        self.hist_record_num = hist_record_num

        self.train_logger = train_logger
        self.model_stats_tracker = model_stats_tracker
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.tb_log_dir = tb_log_dir
        self.debug_mode = debug_mode
        self.ckpt_save_interval = ckpt_save_interval
        self.gpu = self.device.index

        self.checkpoint_folder = (
            "./checkpoints" if checkpoint_folder is None else checkpoint_folder
        )
        self.commit_message = (
            "No commit message" if commit_message is None else commit_message
        )
        self.commit_file_path = (
            "./commit.csv" if commit_file_path is None else commit_file_path
        )

        self.event_timestamp = None
        self.times = []
        self.trainer_setup()
        self.histogram_record_interval: int = max(
            1,
            self.epochs
            // max(self.hist_record_num, self.epochs),  # avoid division by zero
        )

    def trainer_setup(self) -> None:
        """
        Perform any necessary setup before training.
        """
        # Setup logging
        if self.train_logger is None:
            log_level = logging.DEBUG if self.gpu == 0 else logging.INFO
            self.train_logger = setup_logger(
                name=f"Train:{self.gpu}",
                log_file_name=f"train_{self.gpu}.log",
                level=log_level,
                console=False,
            )

        self.writer: SummaryWriter | None
        self.custom_suffix: str | None

        self.training_setup_manager = TrainingSetupManager(
            self.train_logger,
            self.gpu,
            self.tb_log_dir,
            self.commit_file_path,
            self.debug_mode,
            commit_message=self.commit_message,
        )
        # Set up TensorBoard and obtain the writer and custom suffix for file naming
        """
        custom_suffix: str = f"{self.arch}_{self.lr}_{self.optimizer_name}
                            eg:batch-256-lr-2e-3-SGD
        """
        self.writer, self.custom_suffix, self.event_timestamp = (
            self.training_setup_manager.setup_tensorboard_and_commit(
                self.batch_size,
                self.arch,
                self.lr,
                self.optimizer_name,
                commit_message=self.commit_message,
            )
        )  # NOTE: if gpu !=0, writer and custom_suffix will be None

    def train_one_epoch(
        self,
        current_epoch: int,
        data_loader: DataLoader,
        prefix: str = "Training Epoch",
        verbose: bool = False,
        print_freq: int = 100,
    ) -> Tuple[
        Optional[ModelStatsTracker], MetricTracker, MetricTracker, MetricTracker
    ]:
        """
        Train model for one epoch and handle all associated metrics and logging.

        Args:
            current_epoch (int): The current epoch number for logging purposes.
            data_loader (DataLoader): The data loader used for training.
            prefix (str): A string prefix for logging, identifying this as a training epoch.
            verbose (bool): If True, provides detailed logging information.
            print_freq (int): How frequently to log progress within the epoch.

        Returns:
            Tuple[Optional[ModelStatsTracker], MetricTracker, MetricTracker, MetricTracker]:
            ModelStatsTracker and metrics trackers for loss, top1, and top5 accuracy.
        """
        # Initialize progress meters with the given prefix and data loader size
        meters, progress = create_meters(
            batch_size=len(data_loader),
            prefix=f"{prefix}:[{current_epoch}]",
        )
        losses_meter, top1, top5, batch_processing_time, data_loading_time = meters

        # Set the model to training mode
        self.model.train()
        end = time.time()

        # Training loop using the provided data loader
        for i, batch in enumerate(data_loader):
            data_loading_time.update(time.time() - end)
            loss, acc1, acc5 = process_batch(
                batch=batch,
                model=self.model,
                criterion=self.criterion,
                device=self.device,
                is_training=True,
            )

            # Update the meters
            update_meters(
                meters=[losses_meter, top1, top5],
                loss=loss,
                acc1=acc1,
                acc5=acc5,
                batch_size=batch[0].size(0),
            )

            # Zero gradients, backward pass, and optimizer step
            self.optimizer_manager.zero_grad()
            loss.backward()
            self.optimizer_manager.step()

            batch_processing_time.update(time.time() - end)
            end = time.time()

            # Verbose logging per specified print frequency
            if self.gpu == 0 and verbose and i % print_freq == 0:
                self.train_logger.debug(progress.display(i + 1))
                self.train_logger.debug("\n")

        # Synchronize the metrics across all GPUs if using distributed training
        losses_meter.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

        # Summary logging of the epoch progress
        if self.gpu == 0 and verbose:
            self.train_logger.debug(progress.display_summary())

        # Optional tracking of model statistics
        if self.gpu == 0 and self.model_stats_tracker is not None:
            pass

        return self.model_stats_tracker, losses_meter, top1, top5

    def evaluate_one_epoch(
        self,
        current_epoch: int,
        data_loader: DataLoader,
        prefix: str = "Val",
        verbose: bool = False,
        print_freq: int = 100,
    ) -> Tuple[
        Optional[ModelStatsTracker], MetricTracker, MetricTracker, MetricTracker
    ]:
        """
        Perform validation or testing on the specified dataset to evaluate the performance of the model.

        Args:
            current_epoch (int): The current epoch number, used for logging.
            data_loader (DataLoade): DataLoader to use for validation or testing.
            prefix (str): A prefix for identifying the purpose of the validation (e.g., "Val" or "Test").
            verbose (bool): If True, provides detailed logging information.
            print_freq (int): How frequently to log progress within the epoch.

        Returns:
            Tuple[Optional[ModelStatsTracker], MetricTracker, MetricTracker, MetricTracker]:
            ModelStatsTracker and metrics trackers for loss, top1, and top5 accuracy.
        """

        meters, progress = create_meters(
            batch_size=len(data_loader),
            prefix=f"{prefix}:[{current_epoch}]",
        )
        losses_meter, top1, top5, batch_processing_time, data_loading_time = meters

        self.model.eval()

        end = time.time()
        for i, batch in enumerate(data_loader):
            loss, acc1, acc5 = process_batch(
                batch=batch,
                model=self.model,
                criterion=self.criterion,
                device=self.device,
                is_training=False,
            )

            update_meters(
                meters=[losses_meter, top1, top5],
                loss=loss,
                acc1=acc1,
                acc5=acc5,
                batch_size=batch[0].size(0),
            )

            batch_processing_time.update(val=time.time() - end)
            end = time.time()

            if verbose and i % print_freq == 0:
                self.train_logger.debug(progress.display(i + 1))

        if verbose:
            self.train_logger.debug(progress.display_summary())

        if self.model_stats_tracker is not None:
            pass

        return self.model_stats_tracker, losses_meter, top1, top5

    def train_multiple_epochs(
        self,
    ) -> Tuple[float, int]:
        best_acc1 = 0.0
        best_epoch = self.start_epoch

        if self.gpu == 0:
            input_tensor = torch.randn(1, 3, 32, 32).to(self.device)
            if not self.writer:
                self.train_logger.error("TensorBoard writer not initialized.")
                raise ValueError("TensorBoard writer not initialized.")
            self.writer.add_graph(self.model, input_tensor)
            self.train_logger.debug("Model graph added to TensorBoard.")

        for current_epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            if self.is_distributed:
                self.train_sampler.set_epoch(current_epoch)
            tracker, train_losses_meter, train_top1, train_top5 = self.train_one_epoch(
                current_epoch,
                data_loader=self.train_loader,
                verbose=self.verbose,
                print_freq=self.print_freq,
            )
            if self.is_validation_enabled:
                val_tracker, val_losses_meter, val_top1, val_top5 = (
                    self.evaluate_one_epoch(
                        current_epoch,
                        data_loader=self.val_loader,
                        verbose=self.verbose,
                        print_freq=self.print_freq,
                    )
                )
            # Evaluate the early stopping decision and whether it's the best model so far
            is_best = False

            if self.gpu == 0:
                self.record_parameter_histograms(current_epoch)
                self.process_epoch_metrics(
                    current_epoch,
                    train_losses_meter,
                    train_top1,
                    train_top5,
                    val_losses_meter,
                    val_top1,
                    val_top5,
                )

                is_best: bool = val_top1.avg > best_acc1
                best_acc1: float = (
                    max(val_top1.avg, best_acc1) if is_best else best_acc1
                )
                best_epoch: int = current_epoch if is_best else best_epoch

            # Update scheduler and early stopping based on validation results
            if self.scheduler_manager:
                self.scheduler_manager.scheduler_step(
                    current_epoch, val_losses_meter.avg
                )

            # Handle model checkpointing
            checkpoint_state = self.create_checkpoint_state(best_epoch, best_acc1)
            if self.is_amp_enabled and self.amp:
                checkpoint_state["amp"] = self.amp.state_dict()
            if current_epoch % self.ckpt_save_interval == 0 or is_best:
                self.save_checkpoint(
                    checkpoint_state, is_best, current_epoch, check_point_suffix=None
                )

            self.log_epoch_timing(current_epoch, epoch_start)
            if self.early_stop(
                val_loss=val_losses_meter.avg, current_epoch=current_epoch
            ):
                break
        # Finalize training at the end of all loops
        self.finalize_training()
        return (best_acc1, best_epoch)

    def log_epoch_timing(self, current_epoch, epoch_start):
        """
        Logs the duration of the current epoch and estimates the time left for the training.

        Args:
            current_epoch (int): The current epoch number.
            epoch_start (float): The start time of the current epoch.
        """
        epoch_duration = time.time() - epoch_start
        self.times.append(epoch_duration)
        average_epoch_duration = sum(self.times) / len(self.times)
        estimated_time_left = (self.epochs - current_epoch - 1) * average_epoch_duration

        hours_left, minutes_left = divmod(estimated_time_left, 3600)
        self.train_logger.info(
            f"Epoch {current_epoch + 1}/{self.epochs} completed in {epoch_duration:.2f} seconds."
        )
        self.train_logger.info(
            f"Approximate time left: {int(hours_left)} hours, {int(minutes_left // 60)} minutes"
        )

    def log_epoch_completion(self, current_epoch: int, epoch_start: float):
        epoch_duration = time.time() - epoch_start
        self.times.append(epoch_duration)
        average_epoch_duration = sum(self.times) / len(self.times)
        estimated_time_left = (self.epochs - current_epoch - 1) * average_epoch_duration
        hours_left, minutes_left = divmod(estimated_time_left, 3600)
        self.train_logger.info(
            f"Epoch {current_epoch + 1}/{self.epochs} completed in {epoch_duration:.2f} seconds."
        )
        self.train_logger.info(
            f"Approximate time left: {int(hours_left)} hours, {int(minutes_left // 60)} minutes"
        )

    def early_stop(self, val_loss: float, current_epoch: int):
        """Determine if early stopping is triggered and broadcast the decision."""

        early_stop_decision = self.optimizer_manager.check_early_stopping(val_loss)

        # Create a tensor from the early stop decision and broadcast it
        early_stop_tensor = torch.tensor(
            [int(early_stop_decision)], dtype=torch.int, device=self.device
        )
        dist.broadcast(tensor=early_stop_tensor, src=0)

        # Retrieve the updated early stop decision after the broadcast
        early_stop_decision = bool(early_stop_tensor.item())

        # Log and return the decision
        if early_stop_decision:
            self.train_logger.info(
                f"Early stopping triggered across all processes at epoch:{current_epoch}"
            )
        return early_stop_decision

    def finalize_training(self) -> None:
        """
        Finalize the training by closing the writer and exporting metrics.

        Args:
            best_acc1 (float): Best top-1 accuracy achieved during training.
            best_epoch (int): Epoch number corresponding to the best accuracy.
        """
        if self.writer:
            self.writer.close()

            if not self.custom_suffix:
                self.train_logger.error(
                    "custom_suffix must be specified when using TensorBoard"
                )
                raise ValueError(
                    "custom_suffix must be specified when using TensorBoard"
                )

            exporter = TBLogExporter(
                event_folder_path=self.writer.log_dir,
                custom_suffix=self.custom_suffix,
                tb_logger=self.train_logger,
            )

            grouped_metrics = {
                "Loss": ["Loss/train", "Loss/val"],
                "Top1": ["Top1/train", "Top1/val"],
                "Top5": ["Top5/train", "Top5/val"],
                "Learning Rate": ["Learning Rate"],
            }

            fig_name = f"metrics-{self.custom_suffix}.png"
            exporter.export(grouped_metrics=grouped_metrics, fig_name=fig_name)

    def record_parameter_histograms(self, current_epoch: int):
        """Log parameter histograms to TensorBoard."""
        if (
            current_epoch % self.histogram_record_interval == 0
            or current_epoch == self.epochs - 1
        ):
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(
                    tag=name.replace(".", "/"),
                    values=param.cpu().data.numpy(),
                    global_step=current_epoch,
                )

    def record_metrics(self, current_epoch, train_metrics, val_metrics):
        """Utility to log metrics to TensorBoard."""
        if train_metrics:
            for metric_name, value in train_metrics.items():
                self.writer.add_scalar(metric_name, value, current_epoch)
        else:
            self.train_logger.error("No training metrics to log.")
        if val_metrics:
            for metric_name, value in val_metrics.items():
                self.writer.add_scalar(metric_name, value, current_epoch)
        else:
            self.train_logger.error("No validation metrics to log.")

    def process_epoch_metrics(
        self,
        current_epoch,
        train_losses_meter,
        train_top1,
        train_top5,
        val_losses_meter,
        val_top1,
        val_top5,
    ):
        """Record training and validation metrics."""
        train_metrics = {
            "Loss/train": train_losses_meter.avg,
            "Top1/train": train_top1.avg,
            "Top5/train": train_top5.avg,
        }
        if self.is_validation_enabled:
            val_metrics = {
                "Loss/val": val_losses_meter.avg,
                "Top1/val": val_top1.avg,
                "Top5/val": val_top5.avg,
            }
        else:
            val_metrics = {}
            self.train_logger.warning("Evaluation for validation set is disabled.")
        self.writer.add_scalar(
            "Learning Rate",
            self.optimizer_manager.optimizer.param_groups[0]["lr"],
            global_step=current_epoch,
        )
        self.record_metrics(current_epoch, train_metrics, val_metrics)

    def save_checkpoint(
        self,
        state: Dict,
        is_best: bool,
        current_epoch: int,
        check_point_suffix: Optional[str] = None,
    ) -> None:
        """
        Save the model checkpoint during training. If the current checkpoint is the best model,
        it's saved under a special name. Regular checkpoints are saved at specified intervals.

        Args:
            state (Dict): Model state to be saved (parameters and other information).
            is_best (bool): Indicates whether the current checkpoint is the best so far.
            current_epoch (int): Current epoch number, used in the filename.
            check_point_suffix (Optional[str]): Custom suffix for filename differentiation.
        """
        try:
            # Ensure the checkpoint folder for the architecture exists
            save_checkpoint_folder = os.path.join(
                self.checkpoint_folder,
                self.arch,
                self.custom_suffix,
                self.event_timestamp,
            )
            if not os.path.exists(save_checkpoint_folder):
                os.makedirs(save_checkpoint_folder)

            # Define base filename with architecture, custom suffix, and epoch
            if check_point_suffix:
                base_filename = f"epoch{current_epoch}-{check_point_suffix}"
            else:
                base_filename = f"epoch{current_epoch}"

            # Determine file path for the checkpoint
            if current_epoch % self.ckpt_save_interval == 0 or is_best:
                suffix = "best" if is_best else "regular"
                filename = f"{base_filename}-{suffix}.pth"
                file_path = os.path.join(save_checkpoint_folder, filename)

                # Save the checkpoint
                torch.save(state, file_path)
                self.train_logger.debug(f"Checkpoint saved at {file_path}")

        except Exception as e:
            self.train_logger.error(f"Error saving checkpoint: {e}", exc_info=True)

    def create_checkpoint_state(
        self,
        best_epoch: int,
        best_acc1: float,
    ) -> Dict:
        """Create and return the checkpoint state to be saved."""
        state = {
            "best_epoch": best_epoch,
            "best_acc1": best_acc1,
            "arch": self.arch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer_manager.optimizer.state_dict(),
            "scheduler": self.scheduler_manager.scheduler.state_dict()
            if self.scheduler_manager
            else None,
        }
        return state
