import logging
from typing import Dict, Literal, Optional, Union

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from apex import amp
from data_loader import get_dataloaders
from metric_utilis import MetricTracker
from model import CIFARNet, load_or_create_model
from optimizer import CriterionManager, SchedulerManager, initialize_optimizer_manager
from stats_tracker import ModelStatsTracker
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from trainer import Trainer
from utilis import init_distributed_training, load_checkpoint, set_device, setup_logger


def main_worker(
    gpu: Optional[Union[str, int]], ngpus_per_node: int, config: Dict, result
) -> None:
    """
    Execute the main worker process for distributed training on a specified GPU.

    This function initializes the training environment, trains the model, and, if
    running on the primary GPU (typically GPU 0), conducts evaluation and returns the results.

    Args:
        gpu (Optional[Union[str, int]]): Identifier for the specific GPU/NPU to be used
            for training. Can be either the device number or a string name.
        ngpus_per_node (int): Total number of GPUs per node involved in the training process.
        config (Dict): Configuration settings for the training process, including model details,
            dataset information, and other parameters relevant to the training and evaluation.
        result (Array): A multiprocessing array used for returning results from the primary GPU
            to the main process. It holds the best accuracy, corresponding epoch, and the average
            of top-1 accuracies over the validation set.

    Returns:
        None: This function does not return directly. Instead, results from the primary GPU (GPU 0)
        are put into the provided multiprocessing array.
    """
    # Configuration unpacking
    arch: str = config["model"]["arch"]
    pretrained: bool = config["model"].get("pretrained", False)
    dataset_name: str = config["data"]["dataset_name"]

    # Determine the physical GPU from the logical mapping
    process_device_map: Dict[int, int] = config["distributed_training"][
        "process_device_map"
    ]
    physical_gpu: int = process_device_map.get(str(object=gpu), gpu)

    # Extract and validate the device type
    device_type: str = config["distributed_training"].get("device", "cpu")

    device: torch.device = set_device(device=device_type, gpu=physical_gpu)

    # Logger setup for the worker
    worker_log_file: str = f"worker_{device.index}.log"
    worker_logger: logging.Logger = setup_logger(
        name=f"Worker:{device.index}",
        log_file_name=worker_log_file,
        level=logging.DEBUG if device.index == 0 else logging.INFO,
        console=True,
    )

    if device_type not in ["gpu", "npu", "cpu"]:
        worker_logger.error(f"Unsupported device type: {device_type}.")

    worker_logger.debug(f"Initialized logger for {device_type}{device.index}.")

    # Model loading and preparation
    model: CIFARNet = load_or_create_model(config=config)
    model.device = device
    model.unfreeze()
    model.to_device()
    worker_logger.debug(
        f"Model {arch} loaded with pretrained={pretrained} and moved to {device}."
    )

    # Adjust training parameters based on NPU availability
    batch_size = int(config["training"]["batch_size"])
    adjusted_batch_size: int = int(batch_size / ngpus_per_node)
    workers: int = int(
        (config["training"]["workers"] + ngpus_per_node - 1) / ngpus_per_node
    )
    worker_logger.debug(
        f"Adjusted batch size to {adjusted_batch_size} and worker count to {workers}."
    )

    # Extract distributed training parameters
    distributed: bool = config["distributed_training"]["distributed"]
    dist_url: str = config["distributed_training"]["dist_url"]
    rank: int = config["distributed_training"]["rank"]
    dist_backend: str = config["distributed_training"]["dist_backend"]
    world_size: int = config["distributed_training"].get("world_size", 1)

    # Initialize distributed training if enabled
    if distributed:
        init_distributed_training(
            distributed=distributed,
            dist_url=dist_url,
            rank=rank,
            dist_backend=dist_backend,
            world_size=world_size,
            multiprocessing_distributed=distributed,
            ngpus_per_node=ngpus_per_node,
            gpu=gpu,
        )
        worker_logger.info(f"Distributed training initialized for {world_size} nodes.")

    # Data handling for training and validation
    dataset_name: str = config["data"]["dataset_name"]
    num_workers: int = config["training"]["workers"]
    use_dummy_data: bool = config["data"].get("dummy", False)
    if use_dummy_data:
        worker_logger.info(msg="Using dummy data for training.")
        num_classes: int | None = (
            100
            if dataset_name == "cifar100"
            else 10
            if dataset_name == "cifar10"
            else None
        )
        if not num_classes:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

        image_shape: tuple[Literal[3], Literal[32], Literal[32]] = (
            3,
            32,
            32,
        )  # Default shape for CIFAR images

        # Create fake datasets to simulate training and validation data
        train_dataset = datasets.FakeData(
            size=1000,
            image_size=image_shape,
            num_classes=num_classes,
            transform=transforms.ToTensor(),
        )
        val_dataset = datasets.FakeData(
            size=100,
            image_size=image_shape,
            num_classes=num_classes,
            transform=transforms.ToTensor(),
        )
        test_dataset = datasets.FakeData(
            size=100,
            image_size=image_shape,
            num_classes=num_classes,
            transform=transforms.ToTensor(),
        )

        # Create samplers based on whether distributed training is enabled
        if distributed:
            train_sampler = DistributedSampler(
                dataset=train_dataset, shuffle=True, drop_last=True
            )
            val_sampler = DistributedSampler(
                dataset=val_dataset, shuffle=False, drop_last=True
            )
        else:
            train_sampler = None
            val_sampler = None

        # Create data loaders
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=adjusted_batch_size if distributed else batch_size,
            shuffle=True,
            sampler=train_sampler,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=adjusted_batch_size if distributed else batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        # Setup for real CIFAR dataset based on configuration details
        dataset_path: str = config["data"]["path"]
        dataset_name: str = config["data"]["dataset_name"]
        train_ratio: float = config["training"]["train_ratio"]

        # Retrieve data loaders for training, validation, and testing datasets
        train_loader, val_loader, test_loader, train_sampler, val_sampler = (
            get_dataloaders(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                batch_size=batch_size,
                adjusted_batch_size=adjusted_batch_size,
                num_workers=num_workers,
                train_ratio=train_ratio,
                distributed=distributed,
                logger=worker_logger,
            )
        )
        train_dataset_size = len(train_loader.dataset)
        val_dataset_size = len(val_loader.dataset)
        test_dataset_size = len(test_loader.dataset)
        worker_logger.debug(
            f"Data loaders for {dataset_name} initialized successfully with split ratio {train_ratio} and batch size {batch_size}."
        )
        # TODO: check whether the following line is correct
        worker_logger.debug(
            msg=f"Training size is {train_dataset_size}, val size is {val_dataset_size}, test size is {test_dataset_size}",
        )

    # Initialize the OptimizerManager
    optimizer_manager = initialize_optimizer_manager(model=model, config=config)

    # Attempt to load model and optimizer from checkpoint
    checkpoint_path: str = config["training"].get("checkpoint_path", None)
    if checkpoint_path is not None:
        model_state_dict, optimizer_state_dict, best_epoch, best_acc1 = load_checkpoint(
            checkpoint_path=checkpoint_path
        )
        # Update the state of the model and optimizer
        if model_state_dict:
            model.load_state_dict(state_dict=model_state_dict)
            worker_logger.debug(
                f"Model state updated from checkpoint at {checkpoint_path}."
            )
        if optimizer_state_dict:
            optimizer_manager.update_optimizer_state(
                optimizer_state_dict=optimizer_state_dict, params_to_restore=["lr"]
            )
            worker_logger.debug("Optimizer state updated from checkpoint.")

    # Scheduler setup if specified in the config
    if "scheduler" in config:
        scheduler_manager = SchedulerManager(optimizer=optimizer_manager.optimizer)
        scheduler_config: Dict = config["scheduler"]
        scheduler_type: str = scheduler_config["type"]
        warmup_iters: int = scheduler_config["warmup_iters"]

        # scheduler type and warmup_iters should be explicitly passed to the scheduler
        scheduler_kwargs: Dict = {
            k: v
            for k, v in scheduler_config.items()
            if k not in ["type", "warmup_iters"]
        }

        # Convert numerical values as necessary
        if "factor" in scheduler_kwargs:
            scheduler_kwargs["factor"] = float(scheduler_kwargs["factor"])
        if "step_size" in scheduler_kwargs:
            scheduler_kwargs["step_size"] = int(scheduler_kwargs["step_size"])
        if "gamma" in scheduler_kwargs:
            scheduler_kwargs["gamma"] = float(scheduler_kwargs["gamma"])
        if "patience" in scheduler_kwargs:
            scheduler_kwargs["patience"] = int(scheduler_kwargs["patience"])

        # Create scheduler using SchedulerManager
        scheduler_manager.create_scheduler(
            scheduler_type=scheduler_type, warmup_iters=warmup_iters, **scheduler_kwargs
        )
        worker_logger.debug(
            f"Scheduler {scheduler_type} set with parameters: {scheduler_kwargs}"
        )
        amp_enabled: bool = config["amp"]["is_amp_enabled"]
        opt_level: str = config["amp"]["opt_level"]
        loss_scale: float | int = config["amp"]["loss_scale"]
    else:
        scheduler_manager = None

    if amp_enabled:
        model, optimizer_manager.optimizer = amp.initialize(
            models=model,
            optimizers=optimizer_manager.optimizer,
            opt_level=opt_level,
            loss_scale=loss_scale,
        )
        worker_logger.debug(
            f"AMP initialized with opt_level {opt_level} and loss_scale {loss_scale}."
        )

    # Check configuration for enabling ModelStatsTracker
    track_flag: bool = config.get("training").get("model_stats_tracker", False)
    if track_flag:
        model_stats_tracker = ModelStatsTracker(model=model)
        worker_logger.debug("ModelStatsTracker is enabled.")
    else:
        model_stats_tracker = None

    # For NPU or any other devices where benchmarking needs to be turned off
    cudnn.benchmark = False

    verbose: bool = config["training"].get("verbose", False)
    print_freq: int = config["training"].get("print_freq", 100)
    start_epoch: int = config["training"].get("start_epoch", 0)
    ckpt_save_interval = config["training"].get("ckpt_save_interval", 300)

    # Setup for criterion using the CriterionManager
    criterion_type: str = config["optimizer"]["criterion"]
    criterion_manager = CriterionManager(criterion_type)
    criterion_manager.to_device(device)
    criterion: torch.nn.Module = criterion_manager.criterion

    debug_mode: bool = config["training"].get("debug_mode", False)

    worker_logger.debug(
        f'Training debug mode is {"enabled" if debug_mode else "disabled"}, '
        f'Verbose mode is {"on" if verbose else "off"}, '
        f'Printing frequency set to every {print_freq} iterations, '
        f'Starting from epoch {start_epoch}.'
    )

    # Initialize the Trainer with configurations
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer_manager=optimizer_manager,
        scheduler_manager=scheduler_manager,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=config["training"]["epochs"],
        start_epoch=config["training"]["start_epoch"],
        print_freq=print_freq,
        arch=config["model"]["arch"],
        batch_size=batch_size,
        lr=config["training"]["lr"],
        optimizer_name=config["optimizer"]["name"],
        verbose=verbose,
        is_validation_enabled=config["evaluation"]["is_validation_enabled"],
        is_distributed=(ngpus_per_node > 1),
        is_amp_enabled=amp_enabled,
        amp=amp,
        hist_save_interval=config["training"].get("hist_save_interval", 20),
        train_logger=None,  # NOTE initilized inside the trainer
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        tb_log_dir=config["logging"]["tb_log_dir"],
        debug_mode=config["training"]["debug_mode"],
        checkpoint_folder=config["training"]["checkpoint_folder"],
        ckpt_save_interval=ckpt_save_interval,
        commit_message=config["commit"]["commit_message"],
        commit_file_path=config["commit"]["commit_file_path"],
    )

    # Run training loop
    best_acc1, best_epoch = trainer.train_multiple_epochs()

    top1 = None
    if gpu == 0 and config["evaluation"]["is_evaluation_enabled"]:
        if len(test_loader) < 10:
            raise ValueError("Test loader is too small, something is wrong.")
        # Evaluate the model on the test set
        model_stats_tracker: ModelStatsTracker | None
        loss_metric: MetricTracker
        top1: MetricTracker
        top5: MetricTracker
        model_stats_tracker, loss_metric, top1, top5 = trainer.evaluate_one_epoch(
            current_epoch=-1,
            data_loader=test_loader,
            prefix="Test",
            verbose=False,
        )
        worker_logger.info(f"Final testing at NPU/GPU: {gpu}")
        worker_logger.info(
            msg=f"For test set, test loss is: {loss_metric.avg}, test top1 is: {top1.avg}, test top5 is: {top5.avg}"
        )
        result[0] = best_acc1
        result[1] = best_epoch
        result[2] = top1.avg if top1 else -1

        # commit the test information and best acc1/epoch
        metrics_data = {
            "best_acc1": best_acc1,
            "best_epoch": int(best_epoch),
            "test_top1": top1.avg if top1 else -1,
        }
        trainer.training_setup_manager.update_commit_log(
            event_file_dir=trainer.writer.log_dir, metrics_data=metrics_data
        )
