import glob
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from trajcast.data._keys import (
    DISPLACEMENTS_KEY,
    MODEL_TYPE_KEY,
    UPDATE_VELOCITIES_KEY,
)
from trajcast.data.dataset import AtomicGraphDataset
from trajcast.model.checkpoint import CheckpointHandler, CheckpointState
from trajcast.model.losses import MultiobjectiveLoss
from trajcast.model.models import EfficientTrajCastModel, FlexibleModel, TrajCastModel
from trajcast.model.utils import (
    CustomChainedScheduler,
    TensorBoard,
)
from trajcast.utils.misc import GLOBAL_DEVICE, convert_irreps_to_string

LR_STRATEGIES = {
    "MultiStep": torch.optim.lr_scheduler.MultiStepLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
    "Exponential": torch.optim.lr_scheduler.ExponentialLR,
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "CosineAnnealing": torch.optim.lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "Linear": torch.optim.lr_scheduler.LinearLR,
    "LinearLR": torch.optim.lr_scheduler.LinearLR,
    "Constant": torch.optim.lr_scheduler.LinearLR,
    "ConstantLR": torch.optim.lr_scheduler.LinearLR,
}


class Trainer:
    _allowed_train_attrs = [
        "seed",
        "device",
        "target_field",
        "restart_latest",
        "reference_fields",
        "batch_size",
        "max_grad_norm",
        "num_epochs",
        "criterion",
        "optimizer",
        "optimizer_settings",
        "scheduler",
        "scheduler_settings",
        "chained_scheduler_hp",
        "checkpoint_settings",
        "tensorboard_settings",
        "use_tensorboard",
        "model_type",
        "wandb",
    ]

    def __init__(self, config: Dict):
        self.config = config
        train_config = config.get("training")
        # based on config build:
        # - the variables related to training the model
        # we start with this to get the seed for initialising the model
        for key, value in train_config.items():
            if key not in self._allowed_train_attrs:
                raise ValueError(f"Key '{key}' is not allowed.")

        # Reference and target, batchsize, num_epochs
        self.reference_fields = train_config.get("reference_fields")
        self.target_field = train_config.get("target_field")
        self.batch_size = train_config.get("batch_size")
        self.num_epochs = train_config.get("num_epochs")
        self._tqdm_disable = not sys.stderr.isatty()
        self.wandb_run = None
        self.wandb_enabled = False
        self.use_tensorboard = train_config.get("use_tensorboard", True)

        if "precision" in self.config["model"]:
            assert self.config["model"]["precision"] in [64, 32]
            if self.config["model"]["precision"] == 64:
                torch.set_default_dtype(torch.float64)
            elif self.config["model"]["precision"] == 32:
                torch.set_default_dtype(torch.float32)
        else:
            torch.set_default_dtype(torch.float32)

        if train_config.get("device"):
            GLOBAL_DEVICE.device = train_config["device"]

        # set the seed:
        self.device = GLOBAL_DEVICE.device
        self.seed = train_config.get("seed", 42)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        # we also want to make sure we have the argument for restarting
        self.restart_latest = train_config.get("restart_latest", False)

        # and deal with the required checkpoints
        checkpoint_settings = train_config.get("checkpoint_settings")
        if not checkpoint_settings:
            self.checkpoint_settings = {
                "root": os.path.join(os.getcwd(), "checkpoints"),
            }
        else:
            self.checkpoint_settings = checkpoint_settings
            if "root" not in self.checkpoint_settings.keys():
                self.checkpoint_settings["root"] = os.path.join(
                    os.getcwd(), "checkpoints"
                )

        # - the dataset
        self.dataset = AtomicGraphDataset(**self.config["data"])
        # get output_dimensions for reference_fields
        if isinstance(self.reference_fields, list):
            self.output_dimensions = [
                self.dataset[0][prop].size(-1) for prop in self.reference_fields
            ]

        # - the model
        model_type = train_config.get(MODEL_TYPE_KEY, "Flexible")

        if model_type == "Flexible":
            self.model = FlexibleModel(
                config=self.config["model"], predicted_fields=self.reference_fields
            ).to(self.device)

        elif model_type == "TrajCast":
            # compute RMS for normalising
            rms = []
            means = []
            for field in self.reference_fields:
                data = getattr(self.dataset, field)
                rms.append(data.pow(2).mean().sqrt().item())
                means.append(0.0)

            # avg number of neighbors
            if not self.config["model"].get("avg_num_neighbors"):
                self.config["model"]["avg_num_neighbors"] = (
                    torch.tensor(
                        [data.num_edges / data.num_nodes for data in self.dataset],
                        dtype=torch.float32,
                    )
                    .mean()
                    .item()
                )

            self.model = TrajCastModel(
                config=self.config["model"],
                predicted_fields=self.reference_fields,
                rms_targets=rms,
                mean_targets=means,
            ).to(self.device)

        elif model_type == "EfficientTrajCastModel":
            # compute RMS for normalising
            rms = []
            means = []
            for field in self.reference_fields:
                data = getattr(self.dataset, field)

                rms.append(data.pow(2).mean().sqrt().item())
                means.append(0.0)

            # avg number of neighbors
            if not self.config["model"].get("avg_num_neighbors"):
                self.config["model"]["avg_num_neighbors"] = (
                    torch.tensor(
                        [data.num_edges / data.num_nodes for data in self.dataset],
                        dtype=torch.float32,
                    )
                    .mean()
                    .item()
                )

            self.model = EfficientTrajCastModel(
                config=self.config["model"],
                predicted_fields=self.reference_fields,
                rms_targets=rms,
                mean_targets=means,
            ).to(self.device)

        else:
            raise KeyError(f"The chosen model type: {model_type} is not allowed.")

        # Define Loss Function
        criterion = train_config.get("criterion")
        if isinstance(criterion, str):
            self.loss_function = {
                "mse": torch.nn.MSELoss(),
                "mae": torch.nn.L1Loss(),
            }[criterion]
        elif isinstance(criterion, dict):
            criterion["dimensions"] = self.output_dimensions
            self.loss_function = MultiobjectiveLoss(**criterion)

        # Choose optimiser
        optimizer = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }[train_config.get("optimizer").lower()]
        params = list(self.model.parameters())

        # Setup optimiser
        self.optimizer = optimizer(
            params=params, **train_config.get("optimizer_settings")
        )

        # initialise gradient clipping
        self.max_grad_norm = train_config.get("max_grad_norm", float("inf"))

        # scheduler

        # Check whether tensorboard is available
        self.tensorboard = None
        self.tensorboard_settings = train_config.get("tensorboard_settings")
        if self.use_tensorboard and self.tensorboard_settings:
            self.tensorboard = TensorBoard(settings=self.tensorboard_settings)
            self.tensorboard.loss_function = self.loss_function
            self.tensorboard.target_field = self.target_field
            self.tensorboard.reference_fields = self.reference_fields
            self.tensorboard.dimensions = self.output_dimensions

        self._init_wandb(train_config)

    @classmethod
    def build_from_yaml(cls, filename: str):
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Could not find the file under the path {filename}"
            )
        with open(filename, "r") as file:
            dictionary = yaml.load(file, Loader=yaml.FullLoader)

        return cls(config=dictionary)

    def dump_config_to_yaml(self, filename: Optional[str] = "config.yaml"):
        convert_irreps_to_string(self.config)

        with open(filename, "w") as file:
            yaml.dump(self.config, file, sort_keys=False)

    def create_logger(self, directory: Optional[str] = os.getcwd()):
        """_summary_"""

        logger = logging.getLogger()

        # we are not interested in things below info level
        logger.setLevel(logging.INFO)

        # same format as in MACE: https://github.com/ACEsuit/mace/blob/main/mace/tools/utils.py
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # here we set where to save the log
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(
            directory, f"log_{str(int(datetime.timestamp(datetime.now())))}.txt"
        )

        # now we add this to the logger
        fiha = logging.FileHandler(path)
        fiha.setFormatter(formatter)

        logger.addHandler(fiha)

    def _init_wandb(self, train_config: Dict) -> None:
        wandb_settings = train_config.get("wandb", {}) or {}
        if not wandb_settings.get("enabled", False):
            return

        try:
            import wandb  # type: ignore
        except ImportError as exc:  # pragma: no cover - defensive
            raise ImportError(
                "wandb logging requested but the package is not installed. Install via `pip install wandb`."
            ) from exc

        project_name = wandb_settings.get("project") or self.config["data"].get(
            "name", "trajcast"
        )
        run_name = wandb_settings.get("run_name") or self.config.get("model", {}).get(
            "model_type", "trajcast"
        )
        wandb_dir = wandb_settings.get("dir") or os.path.join(os.getcwd(), "wandb")
        os.makedirs(wandb_dir, exist_ok=True)

        self.wandb_run = wandb.init(
            project=project_name,
            entity=wandb_settings.get("entity"),
            name=run_name,
            dir=wandb_dir,
            config=self._build_wandb_config_snapshot(),
            reinit=True,
        )
        self.wandb_enabled = self.wandb_run is not None

    def _build_wandb_config_snapshot(self) -> Dict[str, Any]:
        # Keep the config logger-friendly; strip WandB-specific keys to avoid recursion
        training_copy = {
            k: v
            for k, v in self.config.get("training", {}).items()
            if k not in {"wandb"}
        }
        training_copy.update(
            {
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "device": str(self.device),
            }
        )

        return {
            "model": self.config.get("model", {}),
            "data": self.config.get("data", {}),
            "training": training_copy,
        }

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().item()
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - safety net
            return None

    def _log_wandb_metrics(
        self,
        *,
        epoch: int,
        loss_train,
        loss_val,
        maes_train: Dict,
        lr,
        rel_maes_train: Optional[Dict] = None,
        maes_val: Optional[Dict] = None,
        rel_maes_val: Optional[Dict] = None,
    ) -> None:
        if not self.wandb_enabled:
            return

        metrics = {
            "epoch": epoch,
            "loss/train": self._to_float(loss_train),
            "lr": self._to_float(lr),
        }

        val_loss_value = self._to_float(loss_val)
        if val_loss_value is not None:
            metrics["loss/val"] = val_loss_value

        if maes_train:
            if DISPLACEMENTS_KEY in maes_train:
                metrics["mae/train/displacements"] = self._to_float(
                    maes_train[DISPLACEMENTS_KEY]
                )
            if UPDATE_VELOCITIES_KEY in maes_train:
                metrics["mae/train/update_velocities"] = self._to_float(
                    maes_train[UPDATE_VELOCITIES_KEY]
                )

        if rel_maes_train:
            if DISPLACEMENTS_KEY in rel_maes_train:
                metrics["mae_pct/train/displacements"] = 100 * self._to_float(
                    rel_maes_train[DISPLACEMENTS_KEY]
                )
            if UPDATE_VELOCITIES_KEY in rel_maes_train:
                metrics["mae_pct/train/update_velocities"] = 100 * self._to_float(
                    rel_maes_train[UPDATE_VELOCITIES_KEY]
                )

        if maes_val:
            if DISPLACEMENTS_KEY in maes_val:
                metrics["mae/val/displacements"] = self._to_float(
                    maes_val[DISPLACEMENTS_KEY]
                )
            if UPDATE_VELOCITIES_KEY in maes_val:
                metrics["mae/val/update_velocities"] = self._to_float(
                    maes_val[UPDATE_VELOCITIES_KEY]
                )

        if rel_maes_val:
            if DISPLACEMENTS_KEY in rel_maes_val:
                metrics["mae_pct/val/displacements"] = 100 * self._to_float(
                    rel_maes_val[DISPLACEMENTS_KEY]
                )
            if UPDATE_VELOCITIES_KEY in rel_maes_val:
                metrics["mae_pct/val/update_velocities"] = 100 * self._to_float(
                    rel_maes_val[UPDATE_VELOCITIES_KEY]
                )

        if self.wandb_run:
            self.wandb_run.log(metrics, step=epoch)

    def _finish_wandb(self, best_loss) -> None:
        if not self.wandb_enabled or self.wandb_run is None:
            return

        best_loss_value = self._to_float(best_loss)
        if best_loss_value is not None:
            self.wandb_run.summary["best_val_loss"] = best_loss_value
        self.wandb_run.finish()

    def _compute_validation_loss(self):
        """Compute validation loss/MAEs without TensorBoard logging for WandB-only runs."""
        if not self.tensorboard_settings:
            return None, None, None

        val_cfg = self.tensorboard_settings.get("loss_validation") or {}
        data_args = val_cfg.get("data")
        if not data_args:
            return None, None, None

        batch_size = val_cfg.get("batch_size", 1)
        validation_set = AtomicGraphDataset(**data_args)
        val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

        loss = 0.0
        total_size = 0
        mae_disp = 0.0
        mae_vel = 0.0
        err_disp_abs_sum = 0.0
        err_vel_abs_sum = 0.0
        ref_disp_abs_sum = 0.0
        ref_vel_abs_sum = 0.0

        self.model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = self.model(val_batch.to(self.device))
                predictions = val_batch[self.target_field]
                reference = (
                    val_batch[self.reference_fields]
                    if isinstance(self.reference_fields, str)
                    else torch.hstack([val_batch[field] for field in self.reference_fields])
                )

                loss_batch = self.loss_function(predictions, reference)
                loss += loss_batch.detach() * val_batch.size(0)

                err_disp, err_vel = torch.split(
                    (predictions - reference).abs(), self.output_dimensions, dim=1
                )
                ref_disp, ref_vel = torch.split(
                    reference, self.output_dimensions, dim=1
                )

                mae_disp += err_disp.mean().detach() * val_batch.num_nodes
                mae_vel += err_vel.mean().detach() * val_batch.num_nodes
                err_disp_abs_sum += err_disp.sum().detach()
                err_vel_abs_sum += err_vel.sum().detach()
                ref_disp_abs_sum += ref_disp.abs().sum().detach()
                ref_vel_abs_sum += ref_vel.abs().sum().detach()
                total_size += val_batch.size(0)

        if total_size == 0:
            return None, None, None

        loss /= total_size
        mae_disp /= total_size
        mae_vel /= total_size

        maes = {
            DISPLACEMENTS_KEY: mae_disp.item(),
            UPDATE_VELOCITIES_KEY: mae_vel.item(),
        }

        rel_maes = {}
        if ref_disp_abs_sum > 0:
            rel_maes[DISPLACEMENTS_KEY] = (err_disp_abs_sum / ref_disp_abs_sum).item()
        if ref_vel_abs_sum > 0:
            rel_maes[UPDATE_VELOCITIES_KEY] = (err_vel_abs_sum / ref_vel_abs_sum).item()

        return loss, maes, rel_maes

    def _train_epoch(self, train_loader: DataLoader, epoch_index: int):
        running_loss = 0.0
        mae_disp = 0.0
        mae_vel = 0.0
        err_disp_abs_sum = 0.0
        err_vel_abs_sum = 0.0
        ref_disp_abs_sum = 0.0
        ref_vel_abs_sum = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch_index + 1}/{self.num_epochs}",
            dynamic_ncols=True,
            leave=False,
            disable=self._tqdm_disable,
        )

        for data_batch in progress:
            # Forward pass

            data_batch = self.model(data_batch.to(self.device))

            predictions = data_batch[self.target_field]
            reference = (
                data_batch[self.reference_fields]
                if isinstance(self.reference_fields, str)
                else torch.hstack(
                    [data_batch[field] for field in self.reference_fields]
                )
            )

            # compute loss
            loss = self.loss_function(predictions, reference)

            # compute mae
            err_disp, err_vel = torch.split(
                (predictions - reference).abs(), self.output_dimensions, dim=1
            )
            ref_disp, ref_vel = torch.split(
                reference, self.output_dimensions, dim=1
            )

            batch_mae_disp = err_disp.mean().detach()
            batch_mae_vel = err_vel.mean().detach()

            mae_disp += batch_mae_disp * data_batch.size(0)
            mae_vel += batch_mae_vel * data_batch.size(0)

            # accumulate absolute sums for relative MAE (percent) computation
            err_disp_abs_sum += err_disp.sum().detach()
            err_vel_abs_sum += err_vel.sum().detach()
            ref_disp_abs_sum += ref_disp.abs().sum().detach()
            ref_vel_abs_sum += ref_vel.abs().sum().detach()

            # Backward pass and update weights
            self.optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm,
                norm_type=2,
                error_if_nonfinite=False,
            )

            self.optimizer.step()

            # Compute and accumluate loss
            running_loss += loss.detach() * data_batch.size(0)

            progress.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                mae_disp=f"{batch_mae_disp.item():.3e}",
                mae_vel=f"{batch_mae_vel.item():.3e}",
            )

        epoch_loss = running_loss / train_loader.dataset.num_nodes
        # store maes in dictionary
        maes = {}
        maes[DISPLACEMENTS_KEY] = mae_disp.item() / train_loader.dataset.num_nodes
        maes[UPDATE_VELOCITIES_KEY] = mae_vel.item() / train_loader.dataset.num_nodes

        rel_maes = {}
        if ref_disp_abs_sum > 0:
            rel_maes[DISPLACEMENTS_KEY] = (err_disp_abs_sum / ref_disp_abs_sum).item()
        if ref_vel_abs_sum > 0:
            rel_maes[UPDATE_VELOCITIES_KEY] = (err_vel_abs_sum / ref_vel_abs_sum).item()

        return epoch_loss, maes, rel_maes

    def train(self):
        # setup the logger
        if self.tensorboard:
            log_dir = os.path.join(os.path.dirname(self.tensorboard.log_dir), "logs")
        else:
            log_dir = os.path.join(os.getcwd(), "logs")

        self.create_logger(directory=log_dir)
        logging.info("You are using TrajCast.")

        # Create DataLoader
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        logging.info(f"Using {len(self.dataset)} configurations for training.")
        logging.info(
            f"Training files are stored under root: {self.config['data']['root']} with the filenames: {self.config['data']['files']}."
        )

        if hasattr(self.model, "layers"):
            logging.info(f"Model architecture: {self.model.layers}")
        else:
            logging.info(f"Model architecture: {self.model.config}")
        logging.info(f"Precision {torch.get_default_dtype()}")
        if self.model.o3_backend == "cueq":
            logging.info("Running with cuequivariance as o3 backend.")

        # Check whether scheduler is available
        lr_scheduler = None
        scheduler = self.config["training"].get("scheduler")
        scheduler_settings = self.config["training"].get("scheduler_settings")
        chained_scheduler_hp = self.config["training"].get("chained_scheduler_hp")

        if scheduler:
            # Create list of the schedulers
            scheduler_list = [scheduler] if isinstance(scheduler, str) else scheduler
            # Check scheduler settings correpond to
            if scheduler_settings:
                if not set(scheduler_list).issubset(set(scheduler_settings.keys())):
                    raise KeyError(
                        "The keys in the scheduler_settings dictionary should correspond with the declared schedulers."
                    )
            else:
                raise TypeError(
                    "Scheduler attribute is present but scheduler_settings attribute is absent."
                )

            if chained_scheduler_hp:
                milestones = chained_scheduler_hp["milestones"]
                per_epoch = chained_scheduler_hp.get("per_epoch", True)
                monitor_lr_scheduler = chained_scheduler_hp.get(
                    "monitor_lr_scheduler", False
                )
            else:
                raise KeyError(
                    "chaned_scheduler_hp is not specified. Please specify it."
                )

            schedulers = []
            for sched in scheduler_list:
                scheduler = LR_STRATEGIES[sched]
                scheduler_params = scheduler_settings.get(sched, {})
                schedulers.append(scheduler(self.optimizer, **scheduler_params))

            lr_scheduler = CustomChainedScheduler(
                per_epoch,
                schedulers,
                milestones,
                self.num_epochs,
                len(data_loader),
                monitor_lr_scheduler,
                scheduler_list,
            )

        logging.info(
            f"We are using the following training parameters: {self.config['training']}"
        )

        # Training loop
        # get params (from MACE: https://github.com/ACEsuit/mace/blob/main/mace/tools/torch_tools.py)
        n_params = int(sum(np.prod(p.shape) for p in self.model.parameters()))
        logging.info(f"Number of model parameters: {n_params}")
        logging.info("Started training.")

        if self.wandb_enabled and self.wandb_run:
            self.wandb_run.config.update(
                {
                    "num_parameters": n_params,
                    "train_dataset_size": len(self.dataset),
                },
                allow_val_change=True,
            )

        # initialise the checkpoint handler
        checkpoint_handler = CheckpointHandler(
            directory=self.checkpoint_settings["root"],
            keep_latest=self.checkpoint_settings.get("keep_latest", False),
        )
        checkpoint_interval = self.checkpoint_settings.get("interval", 1)

        # init the starting epoch
        start_epoch = 0
        best_loss = None

        # restart from latest checkpoint in case this is desired
        if self.restart_latest and os.path.exists(self.checkpoint_settings["root"]):

            restart_epoch, best_loss = checkpoint_handler.load_latest(
                state=CheckpointState(
                    self.model, self.optimizer, lr_scheduler, best_loss
                )
            )

            if restart_epoch is not None:
                start_epoch = restart_epoch + 1
                logging.info(
                    f"Restarting from latest checkpoint in epoch: {start_epoch}. Current best loss is {best_loss}"
                )

        epoch = start_epoch
        loss_val = 0.0

        os.makedirs(self.checkpoint_settings["root"], exist_ok=True)

        while epoch < self.num_epochs:

            loss_train, maes_train, rel_maes_train = self._train_epoch(
                data_loader, epoch
            )

            if lr_scheduler is not None:
                lr_rate = lr_scheduler.return_lr(self.optimizer)
            else:
                lr_rate = self.config["training"].get("optimizer_settings")["lr"]

            if self.use_tensorboard and self.tensorboard:
                (
                    loss_val,
                    val_maes,
                    val_rel_maes,
                ) = self.tensorboard.update(
                    epoch=epoch,
                    loss=loss_train.item(),
                    model=self.model,
                    lr=lr_rate,
                    maes=maes_train,
                )
            else:
                loss_val, val_maes, val_rel_maes = self._compute_validation_loss()

            # if scheduler is set update learning rate
            if lr_scheduler:
                lr_scheduler.step(loss_val)

            self._log_wandb_metrics(
                epoch=epoch,
                loss_train=loss_train,
                loss_val=loss_val,
                maes_train=maes_train,
                lr=lr_rate,
                rel_maes_train=rel_maes_train,
                maes_val=val_maes,
                rel_maes_val=val_rel_maes,
            )

            # report loss
            logging.info(
                f"Epoch {epoch}: train_loss={loss_train}; \t val_loss={loss_val}."
            )

            # In the first epoch and checkpoint has no best_loss
            if epoch == start_epoch and not best_loss:
                best_loss = loss_val

            if loss_val and loss_val <= best_loss:
                best_loss = loss_val

                logging.info("Saving new best model!")

                # Delete old one if exists
                old_best_path = glob.glob(
                    os.path.join(self.checkpoint_settings["root"], "best*")
                )
                if old_best_path:
                    os.remove(old_best_path[0])

                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.checkpoint_settings["root"], f"best_model_epoch-{epoch}.pt"
                    ),
                )

            # generate checkpoints and save loss to logging
            if epoch % checkpoint_interval == 0:

                best_loss = 0 if loss_val is None else best_loss

                # save checkpoints
                checkpoint_handler.save(
                    state=CheckpointState(
                        self.model, self.optimizer, lr_scheduler, best_loss
                    ),
                    epoch=epoch,
                )

            epoch += 1

        logging.info("Training done.")
        path_to_model = os.path.join(os.path.dirname(log_dir), "model_params.pt")
        torch.save(self.model.state_dict(), path_to_model)
        logging.info(f"Final model saved to {path_to_model}.")
        self._finish_wandb(best_loss)
