import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch_npu  # noqa
from metric_utilis import MetricTracker
from optimizer import OptimizerManager, SchedulerManager
from setup_utilis import setup_logger
from stats_tracker import ModelStatsTracker
from tb_log_visualization import TBLogExporter
from torch.nn import Module
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
        hist_save_interval: int,
        writer: Optional[SummaryWriter],
        custom_suffix: Optional[str],
        # event_timestamp: str,
        accumulation_steps: int = 5,
        train_logger: Optional[logging.Logger] = None,
        model_stats_tracker: Optional[ModelStatsTracker] = None,
        train_sampler: Optional[Sampler] = None,
        val_sampler: Optional[Sampler] = None,
        test_sampler: Optional[Sampler] = None,
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
        self.hist_save_interval = hist_save_interval

        # TODO: pass in
        self.writer = writer
        self.custom_suffix = custom_suffix
        self.accumulation_steps = accumulation_steps

        self.train_logger = train_logger
        self.model_stats_tracker = model_stats_tracker
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
        self.tb_log_dir = tb_log_dir
        self.debug_mode = debug_mode
        self.ckpt_save_interval = ckpt_save_interval
        self.gpu = self.device.index
        self.rank = dist.get_rank() if is_distributed else 0
        self.world_size = dist.get_world_size() if is_distributed else 1

        self.checkpoint_folder = (
            "./checkpoints" if checkpoint_folder is None else checkpoint_folder
        )
        self.commit_message = (
            "No commit message" if commit_message is None else commit_message
        )
        self.commit_file_path = (
            "./commit.csv" if commit_file_path is None else commit_file_path
        )

        self.times = []
        self.train_logger_setup()
        # self.histogram_record_interval: int = max(
        #     1,
        #     self.epochs
        #     // max(self.hist_save_interval, self.epochs),  # avoid division by zero
        # ) Depreciated

    def train_logger_setup(self) -> None:
        """
        Perform any necessary setup before training.
        """
        # Setup logging
        if self.train_logger is None:
            log_level = logging.DEBUG if self.debug_mode else logging.INFO
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
        print_freq: int = 100,
        accumulation_steps: int = 5,
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
            accumulation_steps (int): Number of batches to accumulate before performing a backward pass and optimizer step.

        Returns:
            Tuple[Optional[ModelStatsTracker], MetricTracker, MetricTracker, MetricTracker]:
            ModelStatsTracker and metrics trackers for loss, top1, and top5 accuracy.
        """
        torch.autograd.set_detect_anomaly(True)

        # Initialize progress meters with the given prefix and data loader size
        meters, progress = create_meters(
            batch_size=len(data_loader),
            prefix=f"{prefix}-Epoch:[{current_epoch}]",
            device=self.device,
        )
        (
            losses_meter,
            top1,
            top5,
            process_one_batch_time,
            data_loading_time,
            sync_time,
        ) = meters

        end = time.time()

        # Zero gradients at the start
        self.optimizer_manager.zero_grad()

        # Initialize list to accumulate losses
        accumulated_losses = []

        # Training loop using the provided data loader
        for i, batch in enumerate(iterable=data_loader):
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

            # Accumulate loss
            accumulated_losses.append(loss / accumulation_steps)

            # Perform backward pass and optimizer step every accumulation_steps
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data_loader):
                # Sum the accumulated losses
                accumulated_loss = sum(accumulated_losses)

                sync_start = time.time()
                accumulated_loss.backward()

                sync_time.update(time.time() - sync_start)

                # Optimizer step and zero gradients
                self.optimizer_manager.step()
                self.optimizer_manager.zero_grad()

            # Reset accumulated losses
            accumulated_losses = []

            process_one_batch_time.update(time.time() - end)
            end = time.time()

            # Verbose logging per specified print frequency
            if self.rank == 0 and verbose and (i + 1) % print_freq == 0:
                self.train_logger.debug(progress.display(i + 1))
                self.train_logger.debug("\n")

        losses_meter.all_reduce()
        top1.all_reduce()
        top5.all_reduce()
        process_one_batch_time.all_reduce()
        data_loading_time.all_reduce()
        sync_time.all_reduce()

        # Summary logging of the epoch progress
        if self.rank == 0 and verbose:
            self.train_logger.debug(progress.display_summary())

        # Optional tracking of model statistics
        if self.rank == 0 and self.model_stats_tracker is not None:
            pass

        return self.model_stats_tracker, losses_meter, top1, top5

    def train_one_epoch2(
        self,
        current_epoch: int,
        data_loader: DataLoader,
        prefix: str = "Train_one_epoch",
        verbose: bool = False,
        print_freq: int = 100,
        accumulation_steps: int = 1,
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
            accumulation_steps (int): Number of batches to accumulate before performing a optimizer step.

        Returns:
            Tuple[Optional[ModelStatsTracker], MetricTracker, MetricTracker, MetricTracker]:
            ModelStatsTracker and metrics trackers for loss, top1, and top5 accuracy.
        """

        # Initialize progress meters with the given prefix and data loader size
        meters, progress = create_meters(
            batch_size=len(data_loader),
            prefix=f"{prefix}-Epoch:[{current_epoch}]",
            device=self.device,
        )
        (
            losses_meter,
            top1,
            top5,
            process_one_batch_time,
            data_loading_time,
            sync_time,
        ) = meters

        end = time.time()

        # Zero gradients at the start
        self.optimizer_manager.zero_grad()

        # Training loop using the provided data loader
        for i, batch in enumerate(iterable=data_loader):
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

            sync_start = time.time()
            loss.backward()
            sync_time.update(time.time() - sync_start)

            # Perform optimizer step every accumulation_steps
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data_loader):
                self.optimizer_manager.step()
                self.optimizer_manager.zero_grad()

            process_one_batch_time.update(time.time() - end)
            end = time.time()

            # Verbose logging per specified print frequency
            if self.rank == 0 and verbose and (i + 1) % print_freq == 0:
                self.train_logger.debug(progress.display(i + 1))
                self.train_logger.debug("\n")

        losses_meter.all_reduce()
        top1.all_reduce()
        top5.all_reduce()
        process_one_batch_time.all_reduce()
        data_loading_time.all_reduce()
        sync_time.all_reduce()

        # Summary logging of the epoch progress
        if self.rank == 0 and verbose:
            self.train_logger.debug(progress.display_summary())

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
        print_freq: int = 100,
    ) -> Tuple[
        Optional[ModelStatsTracker], MetricTracker, MetricTracker, MetricTracker
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
            print_freq (int): How frequently to log progress within the epoch.

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

        meters, progress = create_meters(
            batch_size=len(data_loader),
            prefix=f"{prefix}-Epoch:[{current_epoch}]",
            device=self.device,
        )
        (
            losses_meter,
            top1,
            top5,
            _,
            _,
            _,
        ) = meters

        self.model.eval()

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

            if self.rank == 0 and verbose and (i + 1) % print_freq == 0:
                self.train_logger.debug(progress.display(i + 1))

        losses_meter.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

        if verbose:
            self.train_logger.debug(progress.display_summary())

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
                print_freq=self.print_freq,
            )

            if self.is_validation_enabled:
                # All ranks will perform validation
                val_tracker, val_losses_meter, val_top1, val_top5 = (
                    self.evaluate_one_epoch(
                        current_epoch=current_epoch,
                        data_loader=self.val_loader,
                        prefix="Validation",
                        verbose=self.verbose,
                        print_freq=self.print_freq,
                    )
                )

                test_tracker, test_losses_meter, test_top1, test_top5 = (
                    self.evaluate_one_epoch(
                        current_epoch=current_epoch,
                        data_loader=self.test_loader,
                        prefix="Test",
                        verbose=self.verbose,
                        print_freq=self.print_freq,
                    )
                )

            # Evaluate the early stopping decision and whether it's the best model so far
            is_best = False
            if self.rank == 0:
                self.record_parameter_histograms(current_epoch)
                self.process_epoch_metrics(
                    current_epoch,
                    train_losses_meter,
                    train_top1,
                    train_top5,
                    val_losses_meter,
                    val_top1,
                    val_top5,
                    test_losses_meter,
                    test_top1,
                    test_top5,
                )

                is_best = val_top1.avg > best_val_acc1
                if is_best:
                    best_train_acc1 = train_top1.avg
                    best_val_acc1 = val_top1.avg
                    best_test_acc1 = test_top1.avg
                    best_epoch = current_epoch
                    lr_at_best = self.optimizer_manager.optimizer.param_groups[0]["lr"]

            if self.check_early_stop(
                val_loss=float(val_losses_meter.avg), current_epoch=current_epoch
            ):
                break

            # Update scheduler on all devices using the broadcasted validation loss
            if self.scheduler_manager:
                self.scheduler_manager.scheduler_step(
                    current_epoch, float(val_losses_meter.avg)
                )

            # Handle model checkpointing
            checkpoint_state = self.create_checkpoint_state(best_epoch, best_val_acc1)
            if self.is_amp_enabled and self.amp:
                checkpoint_state["amp"] = self.amp.state_dict()
            if current_epoch % self.ckpt_save_interval == 0 or is_best:
                if self.rank == 0:
                    self.save_checkpoint(
                        checkpoint_state,
                        is_best,
                        current_epoch,
                    )

            self.log_epoch_duration_and_estimate_remaining_time(
                current_epoch, epoch_start
            )
            # self.train_logger.debug(f"{self.rank}: mark 7")
        # Finalize training at the end of all loops
        if self.rank == 0:
            self.finalize_training()
            self.train_logger.info(
                f"Trainer 0 finalized_training, best_acc1={best_val_acc1}, best_epoch={best_epoch}"
            )

        # NOTE: only the rank0 is responsible for return, other ranks will return all zeros

        return (best_epoch, best_train_acc1, best_val_acc1, best_test_acc1, lr_at_best)

    def log_epoch_duration_and_estimate_remaining_time(
        self, current_epoch, epoch_start, print_interval=20
    ):
        """
        Logs the duration of the current epoch and estimates the time left for the training at specified intervals.

        Args:
            current_epoch (int): The current epoch number.
            epoch_start (float): The start time of the current epoch.
            print_interval (int): Interval at which to print logging information.
        """
        epoch_duration = time.time() - epoch_start
        self.times.append(epoch_duration)

        # Total running time so far
        total_running_time = sum(self.times)

        # Average time for one epoch
        average_epoch_duration = np.mean(self.times)

        # Approximate remaining time
        estimated_time_left = (self.epochs - current_epoch - 1) * average_epoch_duration

        hours_total, remainder_total = divmod(total_running_time, 3600)
        minutes_total, seconds_total = divmod(remainder_total, 60)

        hours_left, remainder_left = divmod(estimated_time_left, 3600)
        minutes_left, seconds_left = divmod(remainder_left, 60)

        # Log only at specified intervals or the last epoch
        if current_epoch % print_interval == 0 or (current_epoch + 1) == self.epochs:
            self.train_logger.info(
                f"Epoch {current_epoch + 1}/{self.epochs} completed in {epoch_duration:.2f} seconds."
            )
            self.train_logger.info(
                f"Total running time so far: {int(hours_total)}h:{int(minutes_total)}m:{int(seconds_total)}s."
            )
            self.train_logger.info(
                f"Average time per epoch: {average_epoch_duration:.2f} seconds."
            )
            self.train_logger.info(
                f"Approximate time left: {int(hours_left)}h:{int(minutes_left)}m:{int(seconds_left)}s."
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

    def record_metrics(self, current_epoch, train_metrics, val_metrics, test_metrics):
        """Utility to log metrics to TensorBoard."""
        if train_metrics:
            for metric_name, value in train_metrics.items():
                self.writer.add_scalar(f"Train/{metric_name}", value, current_epoch)
        else:
            self.train_logger.error("No training metrics to log.")

        self.writer.flush()

        if val_metrics:
            for metric_name, value in val_metrics.items():
                self.writer.add_scalar(f"Val/{metric_name}", value, current_epoch)
        else:
            self.train_logger.warning("No validation metrics to log.")

        self.writer.flush()

        if test_metrics:
            for metric_name, value in test_metrics.items():
                self.writer.add_scalar(f"Test/{metric_name}", value, current_epoch)
        else:
            self.train_logger.warning("No test metrics to log.")

        self.writer.flush()

    def process_epoch_metrics(
        self,
        current_epoch,
        train_losses_meter,
        train_top1,
        train_top5,
        val_losses_meter,
        val_top1,
        val_top5,
        test_losses_meter,
        test_top1,
        test_top5,
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
            test_metrics = {
                "Loss/test": test_losses_meter.avg,
                "Top1/test": test_top1.avg,
                "Top5/test": test_top5.avg,
            }
        else:
            val_metrics = {}
            test_metrics = {}
            self.train_logger.warning("Evaluation for validation set is disabled.")

        self.writer.add_scalar(
            "Learning Rate",
            self.optimizer_manager.optimizer.param_groups[0]["lr"],
            global_step=current_epoch,
        )
        self.record_metrics(
            current_epoch, train_metrics, val_metrics, test_metrics=test_metrics
        )

    def save_checkpoint(
        self,
        state: dict,
        is_best: bool,
        current_epoch: int,
    ) -> None:
        """
        Save the model checkpoint during training. Saves 'regular' checkpoints at specified intervals
        and 'best' checkpoints whenever a new best model is found.

        Args:
            state (dict): Model state to be saved (parameters and other information).
            is_best (bool): Indicates whether the current checkpoint is the best so far.
            current_epoch (int): Current epoch number, used in the filename.
        """
        try:
            # Ensure the checkpoint folder for the architecture exists
            main_folder = os.path.dirname(self.writer.log_dir)

            save_checkpoint_folder = os.path.join(main_folder, "checkpoints")
            os.makedirs(save_checkpoint_folder, exist_ok=True)

            # Regular checkpoint saving
            regular_filename = f"regular-epoch:{current_epoch}.pth"
            regular_file_path = os.path.join(save_checkpoint_folder, regular_filename)
            torch.save(state, regular_file_path)

            self.train_logger.debug(f"Regular checkpoint saved at {regular_file_path}")

            # Best checkpoint logic
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
                torch.save(state, best_file_path)
                self.train_logger.debug(f"Best checkpoint saved at {best_file_path}")

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
