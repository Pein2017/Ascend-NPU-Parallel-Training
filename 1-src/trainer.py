import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch_npu  # noqa
from metric_utilis import MetricProgressTracker
from optimizer import OptimizerManager, SchedulerManager
from setup_utilis import setup_logger
from stats_tracker import ModelStatsTracker
from tb_log_visualization import TBLogExporter
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler, Sampler
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
        verbose_print_interval: int,
        arch: str,
        batch_size: int,
        lr: float,
        optimizer_name: str,
        verbose: bool,
        val_enabled: bool,
        is_distributed: bool,
        amp_enabled: bool,
        amp: Optional[Any],
        hist_save_interval: int,
        eval_interval: int,
        writer: Optional[SummaryWriter],
        custom_suffix: Optional[str],
        # event_timestamp: str,
        accum_steps: int = 5,
        train_logger: Optional[logging.Logger] = None,
        model_stats_tracker: Optional[ModelStatsTracker] = None,
        train_sampler: Optional[Sampler] = None,
        val_sampler: Optional[Sampler] = None,
        test_sampler: Optional[Sampler] = None,
        debug_mode: bool = False,
        ckpt_save_interval: int = 300,
        ckpt_dir: Optional[str] = "./checkpoints",
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
        self.verbose_print_interval = verbose_print_interval
        self.arch = arch
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer_name = optimizer_name

        self.verbose = verbose
        self.val_enabled = val_enabled
        self.is_distributed = is_distributed
        self.amp_enabled = amp_enabled
        self.amp = amp
        self.hist_save_interval = hist_save_interval
        self.eval_interval = eval_interval

        # TODO: pass in
        self.writer = writer
        self.custom_suffix = custom_suffix
        self.accum_steps = accum_steps

        self.train_logger = train_logger
        self.model_stats_tracker = model_stats_tracker
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
        self.debug_mode = debug_mode
        self.ckpt_save_interval = ckpt_save_interval
        self.gpu = self.device.index
        self.rank = dist.get_rank() if is_distributed else 0
        self.world_size = dist.get_world_size() if is_distributed else 1

        self.ckpt_dir = "./checkpoints" if ckpt_dir is None else ckpt_dir
        self.commit_message = (
            "No commit message" if commit_message is None else commit_message
        )
        self.commit_file_path = (
            "./commit.csv" if commit_file_path is None else commit_file_path
        )

        self.times = []
        self.setup_trainer_logger()
        # self.histogram_record_interval: int = max(
        #     1,
        #     self.epochs
        #     // max(self.hist_save_interval, self.epochs),  # avoid division by zero
        # ) Depreciated

    def setup_trainer_logger(self) -> None:
        """
        Perform any necessary setup before training.
        """
        # Setup logging
        if self.train_logger is None:
            log_level = logging.DEBUG if self.rank == 0 else logging.INFO
            self.train_logger = setup_logger(
                name=f"Trainer:{self.rank}",
                log_file_name=f"train_{self.rank}.log",
                level=log_level,
                console=False,
            )

    def train_one_epoch(
        self,
        current_epoch: int,
        data_loader: DataLoader,
        prefix: str = "Train_one_epoch",
        verbose: bool = False,
        verbose_print_interval: int = 100,
        accum_steps: int = 5,
    ) -> Tuple[
        Optional[ModelStatsTracker],
        MetricProgressTracker,
        MetricProgressTracker,
        MetricProgressTracker,
    ]:
        """
        Train model for one epoch and handle all associated metrics and logging.

        Args:
            current_epoch (int): The current epoch number for logging purposes.
            data_loader (DataLoader): The data loader used for training.
            prefix (str): A string prefix for logging, identifying this as a training epoch.
            verbose (bool): If True, provides detailed logging information.
            verbose_print_interval (int): How frequently to log progress within the epoch.
            accum_steps (int): Number of batches to accumulate before performing a backward pass and optimizer step.

        Returns:
            Tuple[Optional[ModelStatsTracker], MetricProgressTracker, MetricProgressTracker, MetricProgressTracker]:
            ModelStatsTracker and metrics trackers for loss, top1, and top5 accuracy.
        """
        torch.autograd.set_detect_anomaly(True)

        # Initialize progress meters with the given prefix and data loader size
        meters = create_meters(
            batch_size=len(data_loader),
            total_batches=len(data_loader),
            prefix=f"{prefix}-Epoch:[{current_epoch}]",
            device=self.device,
        )
        (
            losses_meter,
            top1,
            top5,
            process_one_batch_time,
            data_loading_time,
            total_batch_time,
            backward_time,
        ) = meters

        end = time.time()

        # Zero gradients at the start
        self.optimizer_manager.zero_grad()

        # Training loop using the provided data loader
        for i, batch in enumerate(iterable=data_loader):
            data_loading_time.update(time.time() - end)

            batch_start = time.time()

            loss, acc1, acc5 = process_batch(
                batch=batch,
                model=self.model,
                criterion=self.criterion,
                device=self.device,
                is_training=True,
            )

            loss.backward()

            # Update metrics that need to be updated per batch
            batch_metric_values = {
                "Top-1 accuracy": (acc1, batch[0].size(0)),
                "Top-5 accuracy": (acc5, batch[0].size(0)),
                "Data Loading": (data_loading_time.val, batch[0].size(0)),
                "Loss": (loss.item(), batch[0].size(0)),
            }
            update_meters(
                [top1, top5, data_loading_time, losses_meter], batch_metric_values
            )

            # Perform optimizer step every accum_steps
            if (i + 1) % accum_steps == 0 or (i + 1) == len(data_loader):
                # Measure backward pass time
                backward_start = time.time()
                self.optimizer_manager.step()
                self.optimizer_manager.zero_grad()
                backward_time.update(time.time() - backward_start)

                # Update backward time metric after the optimizer step
                metric_values = {
                    "Backward Time": (backward_time.val, batch[0].size(0)),
                }
                update_meters([backward_time], metric_values)

            total_batch_time.update(time.time() - batch_start)

            total_batch_metric_values = {
                "Total Batch Time": (total_batch_time.val, batch[0].size(0)),
                "Batch Training": (process_one_batch_time.val, batch[0].size(0)),
            }

            update_meters(
                [total_batch_time, process_one_batch_time], total_batch_metric_values
            )

            process_one_batch_time.update(time.time() - end)
            end = time.time()

            # Verbose logging per specified print frequency
            if self.rank == 0 and verbose and (i + 1) % verbose_print_interval == 0:
                self.train_logger.debug(f"Local metrics at batch {i + 1}:")
                for meter in meters:
                    self.train_logger.debug(meter.display(i + 1))
                self.train_logger.debug("\n")

        # Perform all_reduce to synchronize metrics across processes
        for meter in meters:
            meter.all_reduce()

        # Display summary at the end of the epoch (only on rank 0)
        if self.rank == 0:
            self.train_logger.debug(
                f"Aggregated metrics after epoch {current_epoch + 1}:"
            )
            for meter in meters:
                self.train_logger.debug(meter.display_summary())
            self.train_logger.debug("\n")

        # Optional tracking of model statistics
        if self.rank == 0 and self.model_stats_tracker is not None:
            pass

        return self.model_stats_tracker, losses_meter, top1, top5

    def evaluate_one_epoch(
        self,
        current_epoch: int,
        data_loader: DataLoader,
        prefix: str = "Val",
        verbose: bool = False,
        verbose_print_interval: int = 100,
    ) -> Tuple[
        Optional[ModelStatsTracker],
        MetricProgressTracker,
        MetricProgressTracker,
        MetricProgressTracker,
    ]:
        """
        Perform validation or testing on the specified dataset to evaluate the performance of the model.
        #NOTE:
        No processing time record in evaluation
        Args:
            current_epoch (int): The current epoch number, used for logging.
            data_loader (DataLoade): DataLoader to use for validation or testing.
            prefix (str): A prefix for identifying the purpose of the validation (e.g., "Val" or "Test").
            verbose (bool): If True, provides detailed logging information.
            verbose_print_interval (int): How frequently to log progress within the epoch.

        Returns:
            Tuple[Optional[ModelStatsTracker], MetricTracker, MetricTracker, MetricTracker]:
            ModelStatsTracker and metrics trackers for loss, top1, and top5 accuracy.
        """

        # Check if the DataLoader uses a DistributedSampler
        if isinstance(data_loader.sampler, DistributedSampler):
            # self.train_logger.debug(f"{prefix} DataLoader is using DistributedSampler")
            pass
        else:
            raise ValueError(f"{prefix} DataLoader must use DistributedSampler")

        meters = create_meters(
            batch_size=len(data_loader),
            total_batches=len(data_loader),
            prefix=f"{prefix}-Epoch:[{current_epoch}]",
            device=self.device,
        )
        losses_meter, top1, top5 = meters[:3]

        self.model.eval()

        for i, batch in enumerate(data_loader):
            loss, acc1, acc5 = process_batch(
                batch=batch,
                model=self.model,
                criterion=self.criterion,
                device=self.device,
                is_training=False,
            )

            # Update the meters
            metric_values = {
                "Loss": (loss.item(), batch[0].size(0)),
                "Top-1 accuracy": (acc1, batch[0].size(0)),
                "Top-5 accuracy": (acc5, batch[0].size(0)),
            }
            update_meters(meters[:3], metric_values)

        for meter in meters[:3]:
            meter.all_reduce()

        if self.model_stats_tracker is not None:
            pass

        return self.model_stats_tracker, losses_meter, top1, top5

    def train_multiple_epochs(self) -> Tuple[int, float, float, float, float]:
        """
        Train the model over multiple epochs and return the training statistics.

        This function trains the model for a set number of epochs and monitors the accuracy on the validation set
        to determine the best performing epoch. The best epoch is defined as the one where the model achieves
        the highest top-1 accuracy on the validation set.

        Returns:
            Tuple[int, float, float, float]:
                best_epoch (int): The epoch number where the best_val_acc1 was achieved, indicative of the model's optimal performance.
                best_train_acc1 (float): The highest training accuracy achieved across all epochs.
                best_val_acc1 (float): The highest validation accuracy achieved, used to determine the best epoch.
                best_test_acc1 (float): The highest test accuracy achieved, providing context for the model's generalization.
                lr_at_best (float): The learning rate at the best epoch, providing context for the training conditions at optimal performance.
        """
        best_train_acc1 = 0.0
        best_val_acc1 = 0.0
        best_test_acc1 = 0.0
        best_epoch = self.start_epoch
        lr_at_best = 0.0

        if self.rank == 0:
            # Infer input tensor shape from the train_loader
            input_tensor = next(iter(self.train_loader))[0].to(self.device)

            if not self.writer:
                self.train_logger.error("TensorBoard writer not initialized.")
                raise ValueError("TensorBoard writer not initialized.")

            # Use self.model.module to refer to the underlying model when using DDP
            original_model = (
                self.model.module
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                else self.model
            )

            self.writer.add_graph(original_model, input_tensor)
            self.train_logger.debug("Model graph added to TensorBoard.")

        for current_epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()

            if self.is_distributed:
                self.train_sampler.set_epoch(current_epoch)

            tracker, train_losses_meter, train_top1, train_top5 = self.train_one_epoch(
                current_epoch,
                data_loader=self.train_loader,
                prefix=f"Trainer[{self.rank}]",
                verbose=self.verbose,
                verbose_print_interval=self.verbose_print_interval,
            )
            train_metrics = {
                "Loss/train": train_losses_meter.avg,
                "Top1/train": train_top1.avg,
                "Top5/train": train_top5.avg,
            }

            # Perform validation at specified intervals
            if self.val_enabled and (current_epoch + 1) % self.eval_interval == 0:
                # All ranks will perform validation
                val_tracker, val_losses_meter, val_top1, val_top5 = (
                    self.evaluate_one_epoch(
                        current_epoch=current_epoch,
                        data_loader=self.val_loader,
                        prefix="Validation",
                        verbose=self.verbose,
                        verbose_print_interval=self.verbose_print_interval,
                    )
                )
                val_metrics = {
                    "Loss/val": val_losses_meter.avg,
                    "Top1/val": val_top1.avg,
                    "Top5/val": val_top5.avg,
                }

                if self.rank == 0 and self.verbose:
                    self.train_logger.debug(
                        f"Aggregated validation metrics after epoch {current_epoch + 1}:"
                    )
                    self.train_logger.debug(f"Validation Loss: {val_losses_meter.avg}")
                    self.train_logger.debug(
                        f"Validation Top-1 Accuracy: {val_top1.avg}"
                    )
                    self.train_logger.debug(
                        f"Validation Top-5 Accuracy: {val_top5.avg}"
                    )

                test_tracker, test_losses_meter, test_top1, test_top5 = (
                    self.evaluate_one_epoch(
                        current_epoch=current_epoch,
                        data_loader=self.test_loader,
                        prefix="Test",
                        verbose=self.verbose,
                        verbose_print_interval=self.verbose_print_interval,
                    )
                )
                test_metrics = {
                    "Loss/test": test_losses_meter.avg,
                    "Top1/test": test_top1.avg,
                    "Top5/test": test_top5.avg,
                }

                if self.rank == 0 and self.verbose:
                    self.train_logger.debug(
                        f"Aggregated test metrics after epoch {current_epoch + 1}:"
                    )
                    self.train_logger.debug(f"Test Loss: {test_losses_meter.avg}")
                    self.train_logger.debug(f"Test Top-1 Accuracy: {test_top1.avg}")
                    self.train_logger.debug(f"Test Top-5 Accuracy: {test_top5.avg}")

            else:
                val_metrics = {}
                test_metrics = {}
                val_losses_meter, val_top1, val_top5 = None, None, None
                test_losses_meter, test_top1, test_top5 = None, None, None

            # Evaluate the early stopping decision and whether it's the best model so far
            is_best = False
            if self.rank == 0:
                self.record_parameter_histograms(current_epoch)
                self.process_epoch_metrics(
                    current_epoch,
                    train_metrics,
                    val_metrics,
                    test_metrics,
                )

                if val_top1 is not None and val_top1.avg > best_val_acc1:
                    is_best = True
                    best_train_acc1 = train_top1.avg
                    best_val_acc1 = val_top1.avg
                    best_test_acc1 = test_top1.avg
                    best_epoch = current_epoch
                    lr_at_best = self.optimizer_manager.optimizer.param_groups[0]["lr"]

            if self.check_early_stop(
                val_loss=float(val_losses_meter.avg)
                if val_losses_meter is not None
                else float("inf"),
                current_epoch=current_epoch,
            ):
                break

            # Update scheduler on all devices using the broadcasted validation loss
            if self.scheduler_manager:
                if val_losses_meter is not None:
                    self.scheduler_manager.scheduler_step(
                        current_epoch, float(val_losses_meter.avg)
                    )
                else:
                    # Only call scheduler_step without a metric if the scheduler is not ReduceLROnPlateau
                    if not isinstance(
                        self.scheduler_manager.scheduler, ReduceLROnPlateau
                    ):
                        self.scheduler_manager.scheduler_step(current_epoch)

            if self.rank == 0:
                self.save_checkpoint(
                    is_best,
                    current_epoch,
                    best_epoch,
                    best_val_acc1,
                )

            dist.barrier()

            self.log_epoch_duration_and_estimate_remaining_time(
                current_epoch, epoch_start
            )

        # Finalize training at the end of all loops
        if self.rank == 0:
            self.finalize_training()
            self.train_logger.info(
                f"Trainer 0 finalized_training, best_acc1={best_val_acc1}, best_epoch={best_epoch}"
            )

        # NOTE: only the rank0 is responsible for return, other ranks will return all zeros

        return (best_epoch, best_train_acc1, best_val_acc1, best_test_acc1, lr_at_best)

    def log_epoch_duration_and_estimate_remaining_time(
        self, current_epoch: int, epoch_start_time: float, log_interval: int = 20
    ) -> None:
        """
        Logs the duration of the current epoch and estimates the remaining training time at specified intervals.

        Args:
            current_epoch (int): The current epoch number.
            epoch_start_time (float): The start time of the current epoch (in seconds since the epoch).
            log_interval (int): Interval at which to log the estimated remaining time (default is 20 epochs).
        """
        # Calculate the duration of the current epoch
        epoch_duration = time.time() - epoch_start_time
        self.times.append(epoch_duration)

        # Calculate the total running time so far
        total_running_time = sum(self.times)

        # Calculate the average duration per epoch
        average_epoch_duration = np.mean(self.times)

        # Estimate the remaining time
        remaining_epochs = self.epochs - current_epoch - 1
        estimated_time_left = remaining_epochs * average_epoch_duration

        # Convert total running time and estimated remaining time to hours, minutes, and seconds
        total_hours, total_remainder = divmod(total_running_time, 3600)
        total_minutes, total_seconds = divmod(total_remainder, 60)

        remaining_hours, remaining_remainder = divmod(estimated_time_left, 3600)
        remaining_minutes, remaining_seconds = divmod(remaining_remainder, 60)

        # Log the information at the specified intervals or at the last epoch
        if (current_epoch % log_interval == 0) or (current_epoch + 1 == self.epochs):
            self.train_logger.info(
                f"Epoch {current_epoch + 1}/{self.epochs} completed in {epoch_duration:.2f} seconds."
            )
            self.train_logger.info(
                f"Total running time so far: {int(total_hours)}h:{int(total_minutes)}m:{int(total_seconds)}s."
            )
            self.train_logger.info(
                f"Average time per epoch: {average_epoch_duration:.2f} seconds."
            )
            self.train_logger.info(
                f"Estimated remaining time: {int(remaining_hours)}h:{int(remaining_minutes)}m:{int(remaining_seconds)}s."
            )

    def check_early_stop(self, val_loss: float, current_epoch: int):
        """Determine if early stopping is triggered."""
        early_stop_decision = self.optimizer_manager.check_early_stopping(val_loss)
        early_stop_decision = bool(early_stop_decision)

        if early_stop_decision:
            self.train_logger.info(
                f"Early stopping triggered across all processes at epoch:{current_epoch}"
            )
        return early_stop_decision

    def finalize_training(self) -> None:
        """
        Finalizes the training process by closing the TensorBoard writer and exporting metrics.

        This method should be called at the end of training to ensure all resources are properly released
        and metrics are exported for analysis.

        Raises:
            ValueError: If custom_suffix is not specified when using TensorBoard.
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

            # Define the metrics to be grouped and exported
            grouped_metrics = {
                "Loss": ["Loss/train", "Loss/val", "Loss/test"],
                "Top1": ["Top1/train", "Top1/val", "Top1/test"],
                "Top5": ["Top5/train", "Top5/val", "Top5/test"],
                "Learning Rate": ["Learning Rate"],
            }

            fig_name = f"{self.custom_suffix}.png"
            exporter.export(grouped_metrics=grouped_metrics, fig_name=fig_name)

    def record_parameter_histograms(self, current_epoch: int):
        """Log parameter histograms to TensorBoard."""
        if (
            current_epoch % self.hist_save_interval == 0
            or current_epoch == self.epochs - 1
        ):
            original_model = (
                self.model.module
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                else self.model
            )
            for name, param in original_model.named_parameters():
                self.writer.add_histogram(
                    tag=name.replace(".", "/"),
                    values=param.cpu().data.numpy(),
                    global_step=current_epoch,
                )

    def process_epoch_metrics(
        self,
        current_epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        test_metrics: dict,
    ) -> None:
        """Record training and validation metrics."""

        self.record_metrics(current_epoch, train_metrics, val_metrics, test_metrics)

        self.writer.add_scalar(
            "Learning Rate",
            self.optimizer_manager.optimizer.param_groups[0]["lr"],
            global_step=current_epoch,
        )

    def record_metrics(
        self,
        current_epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        test_metrics: dict,
    ) -> None:
        """Utility to log metrics to TensorBoard."""

        def log_metrics(metrics: dict, prefix: str):
            if metrics:
                for metric_name, value in metrics.items():
                    if value is not None:
                        self.writer.add_scalar(metric_name, value, current_epoch)
            else:
                self.train_logger.warning(f"No {prefix} metrics to log.")

        log_metrics(train_metrics, "training")
        self.writer.flush()

        if (current_epoch + 1) % self.eval_interval == 0:
            log_metrics(val_metrics, "validation")
            log_metrics(test_metrics, "test")
            self.writer.flush()

    def save_checkpoint(
        self,
        is_best: bool,
        current_epoch: int,
        best_epoch: int,
        best_val_acc1: float,
    ) -> None:
        """
        Save the model checkpoint during training. Saves 'regular' checkpoints at specified intervals
        and 'best' checkpoints whenever a new best model is found.

        Args:
            is_best (bool): Indicates whether the current checkpoint is the best so far.
            current_epoch (int): Current epoch number, used in the filename.
            best_epoch (int): Epoch number where the best validation accuracy was achieved.
            best_val_acc1 (float): Best validation accuracy achieved so far.
        """
        try:
            # Determine if a regular checkpoint needs to be saved
            should_save_regular = (
                current_epoch > 0 and current_epoch % self.ckpt_save_interval == 0
            )

            # If no checkpoint needs to be saved, return early
            if not should_save_regular and not is_best:
                return

            # Ensure the checkpoint folder for the architecture exists
            main_folder = os.path.dirname(self.writer.log_dir)
            save_checkpoint_folder = os.path.join(main_folder, "checkpoints")
            os.makedirs(save_checkpoint_folder, exist_ok=True)

            # Create checkpoint state
            checkpoint_state = self.create_checkpoint_state(best_epoch, best_val_acc1)

            if self.amp_enabled and self.amp:
                checkpoint_state["amp"] = self.amp.state_dict()

            if should_save_regular:
                regular_filename = f"regular-epoch:{current_epoch}.pth"
                regular_file_path = os.path.join(
                    save_checkpoint_folder, regular_filename
                )
                torch.save(checkpoint_state, regular_file_path)
                self.train_logger.debug(
                    f"Regular checkpoint saved at {regular_file_path}"
                )

            if is_best:
                best_filename = f"best-epoch:{current_epoch}.pth"
                best_file_path = os.path.join(save_checkpoint_folder, best_filename)

                # Find and delete the previous best checkpoint
                previous_best_checkpoints = [
                    os.path.join(save_checkpoint_folder, file)
                    for file in os.listdir(save_checkpoint_folder)
                    if "best-epoch" in file and file.endswith(".pth")
                ]
                for file in previous_best_checkpoints:
                    os.remove(file)
                    # self.train_logger.debug(
                    #     f"Deleted previous best checkpoint at {file}"
                    # )

                # Save the new best checkpoint
                torch.save(checkpoint_state, best_file_path)
                # self.train_logger.debug(f"Best checkpoint saved at {best_file_path}")

        except Exception as e:
            self.train_logger.error(f"Error saving checkpoint: {e}", exc_info=True)

    def create_checkpoint_state(
        self,
        best_epoch: int,
        best_acc1: float,
    ) -> Dict:
        """Create and return the checkpoint state to be saved."""

        # Access the underlying model if it's wrapped by DDP
        model_state_dict = (
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict()
        )
        state = {
            "best_epoch": best_epoch,
            "best_acc1": best_acc1,
            "arch": self.arch,
            "state_dict": model_state_dict,
            "optimizer": self.optimizer_manager.optimizer.state_dict(),
            "scheduler": self.scheduler_manager.scheduler.state_dict()
            if self.scheduler_manager
            else None,
        }
        return state
