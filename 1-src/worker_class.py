import logging
import os
from typing import Any, Dict

import torch
import torch.distributed as dist
from data_loader_class import DataLoaderManager
from git import Optional
from model import load_or_create_model
from optimizer import CriterionManager, SchedulerManager, initialize_optimizer_manager
from setup_utilis import setup_logger
from stats_tracker import ModelStatsTracker
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer
from utilis import load_checkpoint

# import torch_npu


class Worker:
    def __init__(
        self,
        config: Dict,
        result: Any,
        writer: Optional[SummaryWriter],
        custom_suffix: Optional[str],
    ):
        """
        Initialize the Worker class with configuration settings and result storage.

        Args:
            config (Dict): Configuration settings for the training.
            result (Any): Multiprocessing array or other type of storage for returning results.
            writer(SummaryWriter): Created by ExperimentManager in main.
            custom_suffix(str): Custom suffix for tensorboard logs and exported figures.
        """
        # Basic worker settings
        self.config = config
        self.result = result
        self.writer = writer
        self.custom_suffix = custom_suffix

        # self.event_timestamp = os.path.dirname(self.writer.log_dir)

        self.gpu = int(os.getenv("LOCAL_RANK"))
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Automatically set the number of workers based on the number of GPUs
        #! Here, $NPROC_PER_NODE should be manually set in the environment variables before training starts.
        self.ngpus_per_node = int(os.getenv("NPROC_PER_NODE", "1"))
        if self.ngpus_per_node == 1:
            Warning("ngpus_per_node is set to 1, make sure this is intended.")
        print(
            f"Initializing Worker: Global Rank = {self.rank}, Local Rank = {self.gpu}, World Size = {self.world_size}, ngpus_per_node = {self.ngpus_per_node}"
        )

        # Initialize to None, will be set in methods
        self.worker_logger = None
        self.device = None
        self.model = None
        self.optimizer_manager = None
        self.scheduler_manager = None
        self.trainer = None
        self.amp = None

        # Initialize logging, device, and other components
        self.initialize_worker()

    def initialize_worker(self):
        """
        Initialize the worker by unpacking configurations and setting up logging,
        device, distributed training, and data loaders.
        """
        # Extract all relevant configuration parameters directly to class attributes
        self.extract_parameters()

        # Setup logging and device
        self.setup_logger_and_device()

        # Load and prepare the data
        self.prepare_data()

        # Additional initialization steps for model, optimizer, checkpoint, scheduler, etc.
        self.setup_model()
        self.setup_optimizer()
        self.load_checkpoint()
        self.setup_scheduler()
        self.setup_benchmark_and_stats_tracker()
        self.setup_criterion()
        self.initialize_amp()

    def extract_parameters(self):
        """
        Extract the necessary configuration parameters into the corresponding attributes.
        """
        # Training
        training_config = self.config["training"]
        self.lr = training_config["lr"]
        self.batch_size = training_config["batch_size"]
        # NOTE: batch size is the total batch size , adjusted for distributed training in each process
        self.adjusted_batch_size = self.batch_size // self.ngpus_per_node
        self.verbose = training_config["verbose"]
        self.seed = training_config["seed"]
        self.hist_save_interval = training_config["hist_save_interval"]
        self.train_ratio = training_config["train_ratio"]
        self.print_freq = training_config["print_freq"]
        self.workers = training_config["workers"]
        self.start_epoch = training_config["start_epoch"]
        self.epochs = training_config["epochs"]
        self.debug_mode = training_config["debug_mode"]
        self.model_stats_tracker_flag = training_config["model_stats_tracker_flag"]
        self.checkpoint_folder = training_config["checkpoint_folder"]
        self.checkpoint_path = training_config["checkpoint_path"]
        self.ckpt_save_interval = training_config["ckpt_save_interval"]

        # Commit
        commit_config = self.config["commit"]
        self.commit_message = commit_config["commit_message"]
        self.commit_file_path = commit_config["commit_file_path"]

        # Model
        model_config = self.config["model"]
        self.arch = model_config["arch"]
        self.pretrained = model_config["pretrained"]

        # Optimizer
        optimizer_config = self.config["optimizer"]
        self.optimizer_name = optimizer_config["name"]
        self.optimizer_momentum = optimizer_config["momentum"]
        self.optimizer_weight_decay = optimizer_config["weight_decay"]
        self.optimizer_betas = optimizer_config["betas"]
        self.criterion_type = optimizer_config["criterion"]

        # Scheduler
        scheduler_config = self.config["scheduler"]
        self.scheduler_type = scheduler_config["type"]
        self.scheduler_mode = scheduler_config["mode"]
        self.scheduler_factor = scheduler_config["factor"]
        self.scheduler_patience = scheduler_config["patience"]
        self.warmup_iters = scheduler_config["warmup_iters"]

        # Early Stopping
        early_stopping_config = self.config["early_stopping"]
        self.min_loss_improvement = early_stopping_config["min_loss_improvement"]
        self.early_stopping_patience = early_stopping_config["patience"]

        # Data
        data_config = self.config["data"]
        self.dataset_path = data_config["path"]
        self.dataset_name = data_config["dataset_name"]
        self.use_dummy = data_config["use_dummy"]

        # Logging
        logging_config = self.config["logging"]
        self.tb_log_dir = logging_config["tb_log_dir"]
        self.logger_dir = logging_config["logger_dir"]

        # Evaluation
        evaluation_config = self.config["evaluation"]
        self.is_evaluation_enabled = evaluation_config["is_evaluation_enabled"]
        self.is_validation_enabled = evaluation_config["is_validation_enabled"]

        # AMP
        amp_config = self.config["amp"]
        self.is_amp_enabled = amp_config["is_amp_enabled"]
        self.amp_loss_scale = amp_config["loss_scale"]
        self.amp_opt_level = amp_config["opt_level"]

        # Distributed Training
        distributed_training_config = self.config["distributed_training"]
        self.distributed = distributed_training_config["distributed"]
        self.dist_url = distributed_training_config["dist_url"]
        self.dist_backend = distributed_training_config["dist_backend"]
        self.master_addr = distributed_training_config["master_addr"]
        self.master_port = distributed_training_config["master_port"]
        self.device_type = distributed_training_config["device_type"]
        self.device_list = distributed_training_config["device_list"]

    def setup_logger_and_device(self):
        """
        Set up both the device and logger based on the distributed training configuration.
        """
        self.setup_logger()  # Setup logger first to ensure logging is available for device setup

        if self.device_type not in ["gpu", "npu", "cpu"]:
            self.worker_logger.error(
                f"Unsupported device type: {self.device_type}. Defaulting to CPU."
            )
            raise
            self.device_type = "cpu"  # Ensuring a fallback to a valid device type

        self.device = self.set_device(self.device_type)
        self.worker_logger.info(
            f"Initialized {self.device_type} on {self.device}."
        )  #! modify back to .debug

    def setup_logger(self):
        """
        Set up the logger for the Worker instance with a specific log file based on the local rank.
        """
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        self.worker_logger = setup_logger(
            name=f"Worker:{self.rank}",
            log_file_name=f"worker_{self.rank}.log",
            level=log_level,
            console=False,
        )

        self.worker_logger.info("Worker Logger initialized.")

    def set_device(self, device_type: str):
        """
        Set the training device based on the device type and local rank.

        Args:
            device_type (str): The type of device ('gpu', 'npu', or 'cpu').

        Returns:
            torch.device: The configured device object for training.
        """
        if device_type == "npu":
            loc = f"npu:{self.gpu}"
            try:
                import torch_npu  # noqa

                torch.npu.set_device(loc)
            except ImportError as e:
                self.worker_logger.error(f"Failed to import torch_npu: {e}")
                raise
        elif device_type == "gpu":
            loc = f"cuda:{self.gpu}"
            # torch.cuda.set_device(loc)
        else:
            loc = "cpu"
            raise ValueError(f"Unsupported device type: {device_type} in CPU branch")

        self.worker_logger.debug(f"Set device {loc} for training.")
        return torch.device(loc)

    def prepare_data(self):
        """
        Set up the data loaders using the DataLoaderManager class for either real or fake datasets.
        """

        fake_data_size = (1000, 100, 100)

        image_shape = (3, 32, 32)  # CIFAR image shape

        # Initialize DataLoaderManager with appropriate flags for fake data
        data_loader_manager = DataLoaderManager(
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            logger=self.worker_logger,
            use_fake_data=self.use_dummy,
            fake_data_size=fake_data_size,
            image_shape=image_shape,
        )

        # Get the data loaders
        # NOTE:
        """
        Validation set doesn't use distributed sampler and should be evaluated on single GPU.
        Train and Test sets should be evaluated on all GPUs.
        """
        (
            train_loader,
            val_loader,
            test_loader,
            train_sampler,
            val_sampler,
            test_sampler,
        ) = data_loader_manager.get_dataloaders(
            batch_size=self.batch_size,
            adjusted_batch_size=self.adjusted_batch_size,
            num_workers=self.workers,
            train_ratio=self.train_ratio,
            distributed=self.distributed,
            transform=None,
        )

        # Logging dataset sizes and confirmation
        train_dataset_size = len(train_loader.dataset)
        val_dataset_size = len(val_loader.dataset)
        test_dataset_size = len(test_loader.dataset)
        self.worker_logger.debug(
            f"Data loaders for {self.dataset_name} initialized successfully with split ratio {self.train_ratio} and total batch size {self.batch_size}."
        )
        self.worker_logger.debug(
            f"Training size is {train_dataset_size}, val size is {val_dataset_size}, test size is {test_dataset_size}."
        )

        # Store data loaders and samplers in the Worker class for further use
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler

        self.worker_logger.info("Data loaders initialized successfully.")

    def setup_model(self):
        """
        Load and prepare the model based on the model configuration.
        """
        model = load_or_create_model(config=self.config, device=self.device)
        model.unfreeze()
        model.to_device()

        self.model = DDP(model, device_ids=[self.gpu], find_unused_parameters=False)

        self.worker_logger.debug(
            f"Model {self.arch} loaded with pretrained={self.pretrained} and moved to {self.device}."
        )

        self.workers = int(
            (self.workers + self.ngpus_per_node - 1) / self.ngpus_per_node
        )

        self.worker_logger.debug(
            f"Adjusted batch size to {self.adjusted_batch_size} and worker count to {self.workers}."
        )

    def setup_optimizer(self):
        # Todo: Supporat puerly if-elif statement based on the optimizer, don't try to generalize the API usages.
        self.optimizer_manager = initialize_optimizer_manager(
            model=self.model, config=self.config
        )
        self.worker_logger.debug("Optimizer initialized.")

    def load_checkpoint(self):
        checkpoint_path = self.checkpoint_path
        if checkpoint_path:
            # Attempt to load the checkpoint data
            state = load_checkpoint(checkpoint_path=checkpoint_path)
            if state:
                # Todo: Notice the '.module' settings of state_dict for DDP models.
                # Update the model state
                if "model_state_dict" in state:
                    self.model.load_state_dict(state_dict=state["model_state_dict"])
                    self.worker_logger.debug(
                        f"Model state updated from checkpoint at {checkpoint_path}."
                    )

                # Update the optimizer state
                if "optimizer_state_dict" in state:
                    self.optimizer_manager.update_optimizer_state(
                        optimizer_state_dict=state["optimizer_state_dict"],
                        params_to_restore=["lr"],
                    )
                    self.worker_logger.debug("Optimizer state updated from checkpoint.")

    def setup_scheduler(self):
        """
        Prepare and set up the learning rate scheduler using the SchedulerManager.
        """
        # Retrieve the scheduler configuration
        scheduler_config = self.config.get("scheduler")
        if scheduler_config:
            # Initialize SchedulerManager with the current optimizer
            self.scheduler_manager = SchedulerManager(
                optimizer=self.optimizer_manager.optimizer
            )

            # Extract scheduler parameters from the configuration
            scheduler_type = self.scheduler_type
            warmup_iters = self.warmup_iters

            # Filter out specific keys and handle necessary type conversions
            scheduler_kwargs = {
                k: self._convert_scheduler_param(k, v)
                for k, v in scheduler_config.items()
                if k not in ["type", "warmup_iters"]
            }

            # Create the scheduler using the manager
            self.scheduler_manager.create_scheduler(
                scheduler_type=scheduler_type,
                warmup_iters=warmup_iters,
                **scheduler_kwargs,
            )
            self.worker_logger.debug(
                f"Scheduler {scheduler_type} initialized with parameters: {scheduler_kwargs}"
            )
        else:
            self.scheduler_manager = None
            self.worker_logger.debug(
                "No scheduler configuration found. Skipping scheduler setup."
            )

    def _convert_scheduler_param(self, param_name: str, param_value):
        """
        Convert scheduler parameter to the appropriate type based on the parameter name.

        Args:
            param_name (str): The name of the parameter to convert.
            param_value (Any): The value of the parameter before conversion.

        Returns:
            Any: The value after conversion to the correct type.
        """
        conversion_map = {
            "factor": float,
            "step_size": int,
            "gamma": float,
            "patience": int,
        }
        # Apply the conversion if the parameter is in the conversion map
        return conversion_map.get(param_name, lambda x: x)(param_value)

    def setup_benchmark_and_stats_tracker(self):
        # Disable benchmark for specific devices (e.g., NPU)
        torch.backends.cudnn.benchmark = False
        # Check configuration for enabling ModelStatsTracker
        if self.model_stats_tracker_flag:
            self.model_stats_tracker = ModelStatsTracker(model=self.model)
            self.worker_logger.debug("ModelStatsTracker is enabled.")
        else:
            self.model_stats_tracker = None

        # Logging additional configuration details
        self.worker_logger.debug(
            f'Training debug mode is {"enabled" if self.debug_mode else "disabled"}, '
            f'Verbose mode is {"on" if self.verbose else "off"}, '
            f'Printing frequency set to every {self.print_freq} iterations, '
            f'Starting from epoch {self.start_epoch}.'
        )

    def setup_criterion(self):
        criterion_type = self.criterion_type
        self.criterion_manager = CriterionManager(criterion_type=criterion_type)
        self.criterion_manager.to_device(self.device)
        self.criterion = self.criterion_manager.criterion
        self.worker_logger.debug(
            f"Criterion {criterion_type} initialized and set to device."
        )

    def initialize_amp(self):
        if self.is_amp_enabled:
            from apex import amp

            self.amp = amp
            self.model, self.optimizer_manager.optimizer = amp.initialize(
                models=self.model,
                optimizers=self.optimizer_manager.optimizer,
                opt_level=self.amp_opt_level,
                loss_scale=self.amp_loss_scale,
            )
            self.worker_logger.debug("AMP initialized.")

    def execute_main_task(self):
        """
        Main execution method that prepares and runs training.
        """
        self.initialize_trainer()
        self.run_training()

    def initialize_trainer(self):
        """
        Initializes the Trainer with necessary configurations and dependencies.
        """
        if self.is_amp_enabled:
            self.initialize_amp()

        self.trainer = Trainer(
            model=self.model,
            criterion=self.criterion,
            optimizer_manager=self.optimizer_manager,
            scheduler_manager=self.scheduler_manager,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            device=self.device,
            epochs=self.epochs,
            start_epoch=self.start_epoch,
            print_freq=self.print_freq,
            arch=self.arch,
            batch_size=self.batch_size,
            lr=self.lr,
            optimizer_name=self.optimizer_name,
            verbose=self.verbose,
            is_validation_enabled=self.is_validation_enabled,
            is_distributed=self.distributed,
            is_amp_enabled=self.is_amp_enabled,
            amp=self.amp,
            writer=self.writer,
            custom_suffix=self.custom_suffix,
            hist_save_interval=self.hist_save_interval,
            tb_log_dir=self.tb_log_dir,
            debug_mode=self.debug_mode,
            checkpoint_folder=self.checkpoint_folder,
            ckpt_save_interval=self.ckpt_save_interval,
            commit_message=self.commit_message,
            commit_file_path=self.commit_file_path,
            train_sampler=self.train_sampler,
            val_sampler=self.val_sampler,
            test_sampler=self.test_sampler,
        )
        self.worker_logger.info("Trainer initialized successfully.")

    def run_training(self):
        """
        Executes the training loop using the initialized Trainer instance from the Trainer class.

        This method handles the entire training process, including invoking the training over multiple epochs and logging the best training and validation accuracy, best epoch, and learning rate at the best epoch. The training results are obtained from the 'train_multiple_epochs' method of the Trainer class.

        If running on the primary GPU (GPU 0) and evaluation is enabled, it also performs a final evaluation on the test set, logging the test loss and accuracy metrics (top-1 and top-5).
        """

        (best_epoch, best_train_acc1, best_val_acc1, best_test_acc1, lr_at_best) = (
            self.trainer.train_multiple_epochs()
        )

        if self.rank == 0:
            self.worker_logger.info(
                f"Worker[{self.rank}] has finished, the best epoch is {best_epoch}, best train acc1 is {best_train_acc1}, best val acc1 is {best_val_acc1}, best test acc1 is {best_test_acc1}, lr at best is {lr_at_best}"
            )

        if self.is_evaluation_enabled:
            # Perform final evaluation on the test set if enabled on all GPUs
            self.worker_logger.info("Final evaluation on the test set.")
            model_stats_tracker, losses_meter, top1, top5 = (
                self.trainer.evaluate_one_epoch(
                    -1, self.test_loader, "Final test", False
                )
            )

        if self.rank == 0 and self.is_evaluation_enabled:
            self.result[0] = best_epoch
            self.result[1] = best_train_acc1
            self.result[2] = best_val_acc1
            self.result[3] = best_test_acc1
            self.result[4] = lr_at_best
            self.result[5] = float(losses_meter.avg)

    def cleanup(self):
        """Clean up and destroy the distributed process group on deletion."""
        if dist.is_initialized():
            dist.destroy_process_group()
