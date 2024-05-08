import logging
import os
import time
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from metric_utilis import MetricTracker
from optimizer import OptimizerManager, SchedulerManager
from setup_utilis import setup_logger, setup_tensorboard_and_commit
from stats_tracker import ModelStatsTracker
from tb_log_visualization import TBLogExporter
from torch import nn
from torch.utils.data import DataLoader, Sampler
from train_utilis import create_meters, process_batch, record_metrics, update_meters
from utilis import save_checkpoint


def train(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    train_logger: logging.Logger,
    optimizer_manager: OptimizerManager,
    current_epoch: int,
    device: torch.device,
    tracker: Optional[ModelStatsTracker],
    gpu: int,
    verbose: bool = False,
    print_freq: int = 100,
) -> Tuple[Optional[ModelStatsTracker], MetricTracker, MetricTracker, MetricTracker]:
    """
    Train model for one epoch and handle all associated metrics and logging.

    :param train_loader: DataLoader for the training data.
    :param model: The model to train.
    :param criterion: The loss function.
    :param train_logger: Logger for training related logging.
    :param optimizer_manager: Manages the optimizer.
    :param current_epoch: The current epoch number for training.
    :param device: The device tensors are located on.
    :param tracker: ModelStatsTracker object, for tracking the model's statistics.
    :param gpu: GPU index if using multiple GPUs.
    :param verbose: If true, print detailed logging information.
    :param print_freq: Frequency of logging within the epoch.

    :return: A tuple containing the updated tracker, losses_meter, top1 get_topk_acc meter,
             and top5 get_topk_acc meter.
    """

    meters, progress = create_meters(
        batch_size=len(train_loader), prefix=f"Epoch:[{current_epoch}]"
    )

    losses_meter, top1, top5, batch_processing_time, data_loading_time = meters

    model.train()  # Ensure model is in training mode

    end = time.time()
    for i, batch in enumerate(train_loader):
        data_loading_time.update(time.time() - end)
        loss, acc1, acc5 = process_batch(
            batch=batch,
            model=model,
            criterion=criterion,
            device=device,
            is_training=True,
        )
        update_meters(
            meters=[losses_meter, top1, top5],
            loss=loss,
            acc1=acc1,
            acc5=acc5,
            batch_size=batch[0].size(0),
        )

        optimizer_manager.zero_grad()
        loss.backward()
        optimizer_manager.step()

        batch_processing_time.update(time.time() - end)
        end = time.time()

        if verbose and i % print_freq == 0 and gpu == 0:
            train_logger.debug(progress.display(i + 1))
            train_logger.debug("\n")

    # Synchronize metrics across devices if training in a distributed environment
    losses_meter.all_reduce()
    top1.all_reduce()
    top5.all_reduce()

    # Display training summary for this epoch
    if gpu == 0:  # Ensure only the primary GPU logs detailed output
        if verbose:
            train_logger.debug(progress.display_summary())

    return tracker, losses_meter, top1, top5


def validate(
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    train_logger: logging.Logger,
    current_epoch: int,
    device: torch.device,
    tracker: Optional[ModelStatsTracker],
    prefix: str = "Test",
    verbose: bool = False,
    print_freq: int = 100,
) -> Tuple[Optional[ModelStatsTracker], MetricTracker, MetricTracker, MetricTracker]:
    """
    Perform validation on the validation dataset to evaluate the performance of the model.

    :param val_loader: DataLoader for the validation data.
    :param model: The model to be evaluated.
    :param criterion: The loss function used for evaluation.
    :param train_logger: Logger for validation related logging.
    :param current_epoch: Current epoch of training for context.
    :param device: The device computations will be performed on.
    :param tracker: ModelStatsTracker object, for tracking the model's statistics.
    :param prefix: Prefix to add to the progress bar.
    :param verbose: Flag to print detailed information during validation.
    :param print_freq: Frequency of printing detailed information.
    :return: Tuple containing the tracker, losses_meter, top1 get_topk_acc meter, and top5 get_topk_acc meter.
    """

    meters, progress = create_meters(batch_size=len(val_loader), prefix=f"{prefix}:")
    losses_meter, top1, top5, batch_processing_time, data_loading_time = meters

    model.eval()
    end = time.time()

    for i, batch in enumerate(val_loader):
        loss, acc1, acc5 = process_batch(
            batch=batch,
            model=model,
            criterion=criterion,
            device=device,
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
        end: float = time.time()

        if verbose and i % print_freq == 0:
            train_logger.debug(msg=progress.display(current_batch=i + 1))

    if verbose:
        train_logger.debug(msg=progress.display_summary())

    return tracker, losses_meter, top1, top5


def run_training_loop(
    model: nn.Module,
    criterion: nn.Module,
    optimizer_manager: OptimizerManager,
    scheduler_manager: SchedulerManager,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    hist_record_num=20,
    start_epoch: int = 0,
    distributed: bool = False,
    world_size: int = 1,
    ngpus_per_node: int = 1,
    batch_size: int = 128,
    verbose: bool = False,
    print_freq: int = 10,
    arch: str = "resnet18",
    lr: float = 0.01,
    optimizer_name: str = "SGD",
    amp_enabled: bool = False,
    amp: Optional[object] = None,
    tb_log_dir: Optional[str] = None,
    checkpoint_folder: str = "./checkpoints",
    debug_mode: bool = False,
    tracker: Optional[ModelStatsTracker] = None,
    train_sampler: Optional[Sampler] = None,
    val_sampler: Optional[Sampler] = None,
    ckpt_save_interval: int = 300,
) -> Tuple[float, int]:
    """
    Run training and validation loop.
    """
    gpu = device.index
    log_level: int = logging.DEBUG if gpu == 0 else logging.INFO
    train_logger: logging.Logger = setup_logger(
        name=f"Train:{gpu}",
        log_file_name=f"train_{gpu}.log",
        level=log_level,
        console=False,
    )

    # Setup logging and TensorBoard
    writer, custom_suffix = setup_tensorboard_and_commit(
        train_logger=train_logger,
        gpu=gpu,
        batch_size=batch_size,
        arch=arch,
        lr=lr,
        optimizer_name=optimizer_name,
        tb_log_dir=tb_log_dir,
        debug_mode=debug_mode,
    )

    best_acc1 = 0.0
    best_epoch = start_epoch
    # Add the model graph to TensorBoard
    if gpu == 0:
        input_tensor = torch.randn(1, 3, 32, 32)
        input_tensor = input_tensor.to(f"npu:{gpu}")
        if not writer:
            train_logger.error("TensorBoard writer not initialized.")
            raise ValueError("TensorBoard writer not initialized.")
        # Add the model graph to TensorBoard
        writer.add_graph(model=model, input_to_model=input_tensor)
        train_logger.debug("Model graph added to TensorBoard.")
    # start_time = time.time()
    times = []
    histogram_record_interval: int = max(1, epochs // hist_record_num)
    for current_epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        if distributed:
            train_sampler.set_epoch(current_epoch)
        else:
            train_logger.error("Distributed training is not enabled.")
            raise ValueError("Distributed training is not enabled.")

        tracker, train_losses_meter, train_top1, train_top5 = train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            train_logger=train_logger,
            optimizer_manager=optimizer_manager,
            current_epoch=current_epoch,
            device=device,
            tracker=tracker,
            gpu=gpu,
            verbose=verbose,
            print_freq=print_freq,
        )

        early_stop_decision = False
        is_best = False

        if gpu == 0:
            if (
                current_epoch % histogram_record_interval == 0
                or current_epoch == epochs - 1
            ):  # Also record at the last epoch
                for name, param in model.named_parameters():
                    writer.add_histogram(
                        tag=name.replace(".", "/"),
                        values=param.cpu().data.numpy(),
                        global_step=current_epoch,
                    )

            val_tracker, val_losses_meter, val_top1, val_top5 = validate(
                val_loader=val_loader,
                model=model,
                criterion=criterion,
                train_logger=train_logger,
                current_epoch=current_epoch,
                device=device,
                tracker=tracker,
                prefix="Val:",
                verbose=verbose,
                print_freq=print_freq,
            )

            # Record training metrics
            train_metrics = {
                "Loss/train": train_losses_meter.avg,
                "Top1/train": train_top1.avg,
                "Top5/train": train_top5.avg,
            }
            val_metrics = {
                "Loss/val": val_losses_meter.avg,
                "Top1/val": val_top1.avg,
                "Top5/val": val_top5.avg,
            }
            # Record training and validation metrics
            record_metrics(
                writer=writer, metrics_dict=train_metrics, global_step=current_epoch
            )
            record_metrics(
                writer=writer, metrics_dict=val_metrics, global_step=current_epoch
            )
            # Record learning rate separately
            writer.add_scalar(
                tag="Learning Rate",
                scalar_value=optimizer_manager.optimizer.param_groups[0]["lr"],
                global_step=current_epoch,
            )

            if scheduler_manager:
                scheduler_manager.scheduler_step(
                    current_epoch=current_epoch, metric=val_losses_meter.avg
                )

            is_best: bool = val_top1.avg > best_acc1
            best_acc1: float = max(val_top1.avg, best_acc1) if is_best else best_acc1
            best_epoch: int = current_epoch if is_best else best_epoch

            optimizer_manager.check_early_stopping(val_loss=val_losses_meter.avg)
            if optimizer_manager.early_stop:
                early_stop_decision = True

            checkpoint = {
                "best_epoch": best_epoch,
                "best_acc1": best_acc1,
                "arch": arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer_manager.optimizer.state_dict(),
                "scheduler": scheduler_manager.scheduler.state_dict()
                if scheduler_manager
                else None,
            }

            if amp_enabled and amp:
                checkpoint["amp"] = amp.state_dict()

            save_checkpoint(
                state=checkpoint,
                is_best=is_best,
                checkpoint_folder=checkpoint_folder,
                arch=arch,
                current_epoch=current_epoch,
                check_point_suffix=custom_suffix,
                ckpt_save_interval=ckpt_save_interval,
            )

        # Broadcast early stopping decision to all processes
        early_stop_tensor: torch.Tensor = torch.tensor(
            data=[int(early_stop_decision)], dtype=torch.int, device=device
        )
        dist.broadcast(tensor=early_stop_tensor, src=0)
        early_stop_decision = bool(early_stop_tensor.item())

        if early_stop_decision:
            train_logger.info(
                f"Early stopping triggered at {current_epoch} across all processes."
            )
            break

        """time left calculation"""
        epoch_duration = time.time() - epoch_start
        times.append(epoch_duration)
        average_epoch_duration = sum(times) / len(times)
        estimated_time_left = (epochs - current_epoch - 1) * average_epoch_duration

        # Convert seconds to hours and minutes for a more readable format
        hours_left = int(estimated_time_left // 3600)
        minutes_left = int((estimated_time_left % 3600) // 60)
        train_logger.info(
            f"Epoch {current_epoch + 1}/{epochs} completed in {epoch_duration:.2f} seconds."
        )
        train_logger.info(
            f"Approximate time left: {hours_left} hours, {minutes_left} minutes"
        )

    # Close the writer at the end of training
    if writer:
        writer.close()

        # NOTE: {custom_suffix}-{fig_name}
        if not custom_suffix:
            train_logger.error("custom_suffix must be specified when using TensorBoard")
            raise ValueError("custom_suffix must be specified when using TensorBoard")
        # Instantiate TBLogExporter class
        #! writer.log_dir = save_event_path = os.path.join(tb_log_dir, 'events', arch, configuration_suffix)
        exporter = TBLogExporter(
            event_folder_path=writer.log_dir,
            custom_suffix=custom_suffix,
            tb_logger=train_logger,
        )

        # Define the metrics
        grouped_metrics = {
            "Loss": ["Loss/train", "Loss/val"],
            "Top1": ["Top1/train", "Top1/val"],
            "Top5": ["Top5/train", "Top5/val"],
            "Learning Rate": ["Learning Rate"],
        }
        # Export the metrics to a file named according to the custom_suffix
        fig_name: str = f"metrics-{custom_suffix}.png"
        exporter.export(grouped_metrics=grouped_metrics, fig_name=fig_name)

    return (best_acc1, best_epoch)


# Run the test function
if __name__ == "__main__":
    # Create a logger
    logging.basicConfig(level=logging.DEBUG)
    train_logger: logging.Logger = logging.getLogger("TestLogger")

    # Define test parameters
    gpu = 0  # Simulate as if running on GPU index 0
    world_size = 1  # Number of GPUs
    batch_size = 64  # Batch size per GPU
    arch = "resnet18"  # Model architecture
    lr = 0.01  # Learning rate
    optimizer_name = "adam"  # Optimizer
    tb_log_dir = "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/2-tb_logs/"
    debug_mode = True  # Include debug information

    # Call the function
    writer, custom_suffix = setup_tensorboard_and_commit(
        train_logger,
        gpu,
        world_size,
        batch_size,
        arch,
        lr,
        optimizer_name,
        tb_log_dir,
        debug_mode,
    )
    # Check if TensorBoard and logger setup is successful
    if writer and os.path.exists(path=writer.log_dir):
        train_logger.debug(
            f"TensorBoard setup successful. Logs are being saved in directoy: {writer.log_dir}"
        )
    else:
        raise ValueError("TensorBoard setup failed.")

    # Print the custom suffix to see if it's set correctly
    print(f"Custom suffix used: {custom_suffix}")
