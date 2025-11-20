import os
import warnings
from typing import Dict, List, Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from trajcast.data._keys import (
    DISPLACEMENTS_KEY,
    TENSORBOARD_GRADIENTS_KEY,
    TENSORBOARD_LOG_ROOT_KEY,
    TENSORBOARD_LOSS_KEY,
    TENSORBOARD_LR_KEY,
    TENSORBOARD_VALIDATION_LOSS_KEY,
    TENSORBOARD_WEIGHT_STATS_KEY,
    TENSORBOARD_WEIGHTS_KEY,
    UPDATE_VELOCITIES_KEY,
)
from trajcast.data.dataset import AtomicGraphDataset

# from trajcast.validation.trajectory_comparison import (
#     compare_instantaneous_temperature,
#     compute_particle_position_error,
# )
from trajcast.model.models import TrajCastModel
from trajcast.utils.misc import GLOBAL_DEVICE


class TensorBoard:
    def __init__(self, settings: Dict):

        if TENSORBOARD_LOG_ROOT_KEY not in settings.keys():
            warnings.warn(
                "The logging directory for TensorBoard was not set by the user. It will now be set automatically.",
                UserWarning,
            )

        self.log_dir = settings.get(
            TENSORBOARD_LOG_ROOT_KEY, os.path.join(os.getcwd(), "tb_log")
        )
        # initialise writer
        self.writer = SummaryWriter(self.log_dir)

        # pop the log dir from the settings which should be metrics only now
        if TENSORBOARD_LOG_ROOT_KEY in settings.keys():
            settings.pop(TENSORBOARD_LOG_ROOT_KEY)

        # few variables
        self.device = GLOBAL_DEVICE.device
        self._dimensions = [3, 3]

        # define what needs to be computed in every epoch
        self._metrics_mapping = {
            TENSORBOARD_LOSS_KEY: self._track_loss,
            TENSORBOARD_VALIDATION_LOSS_KEY: self._track_loss_val,
            TENSORBOARD_GRADIENTS_KEY: self._track_gradients,
            TENSORBOARD_WEIGHTS_KEY: self._track_weights,
            TENSORBOARD_LR_KEY: self._track_lr,
            TENSORBOARD_WEIGHT_STATS_KEY: self._track_weight_stats,
        }

        self._stats_mapping = {
            "MAX_NORM": self._stat_max_norm,
            "MIN_NORM": self._stat_min_norm,
            "MAX": self._stat_max,
            "MIN": self._stat_min,
            "AVERAGE_NORM": self._stat_average,
        }

        # now loop over the settings and collect the metrics and arguments (if given)
        if not all([key in self._metrics_mapping.keys() for key in settings.keys()]):
            raise KeyError("Unknown metrics in Tensorboard settings!")
        self.metrics_to_record = settings

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value: list):
        self._dimensions = value

    def _stat_max_norm(self, params):
        # We compute the maximum weight in absolute value of the given layer.
        return torch.abs(params.detach()).max().item()

    def _stat_min_norm(self, params):
        # We compute the minimum weight in absolute value of the given layer.
        return torch.abs(params.detach()).min().item()

    def _stat_max(self, params):
        # We compute the maximum weight of the given layer.
        return params.detach().max().item()

    def _stat_min(self, params):
        # We compute the minimum weight of the given layer.
        return params.detach().min().item()

    def _stat_average(self, params):
        # We compute the mean of the absolute value of the given layer.
        return torch.abs(params.detach()).mean().item()

    # def

    def _track_weight_stats(
        self,
        epoch: int,
        model,
        every: Optional[int] = 1,
        stats: Optional[List] = {"MAX_NORM", "MIN_NORM", "MAX", "MIN", "AVERAGE_NORM"},
    ):
        compute_weight_stats_every_n_epochs = every
        if not set(stats).issubset(
            {"MAX_NORM", "MIN_NORM", "MAX", "MIN", "AVERAGE_NORM"}
        ):
            raise ValueError(
                'Stats should be "MAX_NORM", "MIN_NORM", "MAX", "MIN", "AVERAGE_NORM"'
            )

        if epoch % compute_weight_stats_every_n_epochs == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    for stat in stats:
                        value = self._stats_mapping[stat](param)
                        self.writer.add_scalar(f"{name}+'___'+{stat}", value, epoch)

    def _track_loss(
        self,
        loss: float,
        epoch: int,
    ):
        self.writer.add_scalars(
            "loss",
            {"training": loss},
            epoch,
        )

    def _track_loss_val(
        self, epoch: int, model: TrajCastModel, data_args: Dict, batch_size: int = 1
    ):
        # start by loading the validation dataset:
        validation_set = AtomicGraphDataset(**data_args)

        val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
        # make predictions
        loss = 0
        total_data_size = 0
        mae_disp = 0.0
        mae_vel = 0.0
        err_disp_abs_sum = 0.0
        err_vel_abs_sum = 0.0
        ref_disp_abs_sum = 0.0
        ref_vel_abs_sum = 0.0

        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = model(val_batch.to(self.device))
                predictions = val_batch[self.target_field]

                # format reference
                reference = (
                    val_batch[self.reference_fields]
                    if isinstance(self.reference_fields, str)
                    else torch.hstack(
                        [val_batch[field] for field in self.reference_fields]
                    )
                )
                # compute loss
                loss_batch = self.loss_function(predictions, reference)
                loss += loss_batch.detach() * val_batch.size(0)

                # compute mae
                err_disp, err_vel = torch.split(
                    (predictions - reference).abs(), self.dimensions, dim=1
                )
                ref_disp, ref_vel = torch.split(reference, self.dimensions, dim=1)
                total_data_size += val_batch.size(0)

                mae_disp += err_disp.mean().detach() * val_batch.num_nodes
                mae_vel += err_vel.mean().detach() * val_batch.num_nodes
                err_disp_abs_sum += err_disp.sum().detach()
                err_vel_abs_sum += err_vel.sum().detach()
                ref_disp_abs_sum += ref_disp.abs().sum().detach()
                ref_vel_abs_sum += ref_vel.abs().sum().detach()

        # save to the ternsorboard
        epoch_loss = loss / total_data_size
        self.writer.add_scalars("loss", {"validation": epoch_loss.item()}, epoch)

        # track maes
        # normalise mae
        maes = {}

        mae_disp /= total_data_size
        mae_vel /= total_data_size
        maes[DISPLACEMENTS_KEY] = mae_disp
        maes[UPDATE_VELOCITIES_KEY] = mae_vel

        rel_maes = {}
        if ref_disp_abs_sum > 0:
            rel_maes[DISPLACEMENTS_KEY] = err_disp_abs_sum / ref_disp_abs_sum
        if ref_vel_abs_sum > 0:
            rel_maes[UPDATE_VELOCITIES_KEY] = err_vel_abs_sum / ref_vel_abs_sum

        for prop, mae in maes.items():
            self.writer.add_scalars(
                f"MAE {prop}",
                {"validation": mae.item()},
                epoch,
            )

        for prop, rel_mae in rel_maes.items():
            self.writer.add_scalars(
                f"MAE% {prop}",
                {"validation": (rel_mae * 100)},
                epoch,
            )

        return epoch_loss, maes, rel_maes

    def _track_weights(self, epoch: int, model, every: int = 1):

        if epoch % every == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f"weights_{name}", param, epoch)

    def _track_gradients(self, epoch: int, model, every: int = 1):
        if epoch % every == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f"grad_{name}", param.grad, epoch)

    def _track_mae_training(self, epoch: int, maes: Dict):
        for prop, mae in maes.items():
            self.writer.add_scalars(
                f"MAE {prop}",
                {"training": mae},
                epoch,
            )

    def _track_lr(self, lr: float, epoch: int):
        self.writer.add_scalar(
            "Learning rate",
            lr,
            epoch,
        )

    def update(
        self,
        epoch: int,
        model: Optional[TrajCastModel] = None,
        loss: Optional[float] = 0.0,
        lr: Optional[float] = 0.0,
        maes: Optional[Dict] = {},
    ):
        validation_loss = None
        validation_maes = None
        validation_rel_maes = None

        # loop over all metrics
        for metric, metric_args in self.metrics_to_record.items():
            # loss
            if metric == TENSORBOARD_LOSS_KEY:
                self._track_loss(epoch=epoch, loss=loss)
                self._track_mae_training(epoch=epoch, maes=maes)

            # loss validation
            if metric == TENSORBOARD_VALIDATION_LOSS_KEY:
                # get information on validation data
                data_args = metric_args["data"]
                val_batch_size = metric_args.get("batch_size", 1)
                validation_loss, validation_maes, validation_rel_maes = self._track_loss_val(
                    epoch=epoch,
                    model=model,
                    data_args=data_args,
                    batch_size=val_batch_size,
                )

            if metric == TENSORBOARD_LR_KEY:
                self._track_lr(lr=lr, epoch=epoch)

            if metric == TENSORBOARD_WEIGHTS_KEY:
                if isinstance(metric_args, Dict):
                    self._track_weights(epoch=epoch, model=model, **metric_args)
                else:
                    self._track_weights(epoch=epoch, model=model)

            if metric == TENSORBOARD_GRADIENTS_KEY:
                if isinstance(metric_args, Dict):
                    self._track_gradients(epoch=epoch, model=model, **metric_args)
                else:
                    self._track_gradients(
                        epoch=epoch,
                        model=model,
                    )

            if metric == TENSORBOARD_WEIGHT_STATS_KEY:
                if isinstance(metric_args, Dict):
                    self._track_weight_stats(epoch=epoch, model=model, **metric_args)
                else:
                    self._track_weight_stats(
                        epoch=epoch,
                        model=model,
                    )

        return validation_loss, validation_maes, validation_rel_maes


# Notice that the resave and continuation has to be changed.
class CustomChainedScheduler:
    """Class which combines several type of scheduler.
    The first scheduler is executed till step_count is smaller than the first milestone.
    The second scheduler is executed till step_count is smaller than the second milestone.
    It continues in this way."""

    def __init__(
        self,
        per_epoch: bool,
        schedulers: List[_LRScheduler],
        milestones: List[int],
        total_num_epochs: int,
        total_num_training_steps: int,
        monitor_lr_scheduler: bool,
        scheduler_list: list,
    ):
        """Args:
        per_epoch: A boolean value that, if set to true, says to interpret the milestones as epoch_milestones. If set to false the milestones are interpretes as per_step_milestones.
        schedulers: (list(torch.optim.lr_scheduler)) a list of instantiated scheduler objectes from a predefined list of torch schedulers.
        milestones: (list(int))an ascending list of milestones which defines till when the ith scheduler is performed. This can represent the step at which we switch the scheduler or the epoch at which we switch the scheduler.
        total_num_epochs: The total number of epochs (useful to compute if the last milestone has been given correctly).
        total_num_training_steps: The total number of training_steps per epoch (useful to compute if the last milestone has been given correctly.
        monitor_lr_scheduler: A boolean value which, if set to true, allows the class to drops the lrs in a list for future reference.
        """

        self.per_epoch = per_epoch
        self.schedulers = schedulers
        self.milestones = milestones
        self.monitor_lr_scheduler = monitor_lr_scheduler
        self.list_of_lrs = []
        self.scheduler_list = scheduler_list

        if self.per_epoch:
            self.construct_epoch_milestones(total_num_training_steps)

        # Check Milestones are in ascending order
        for i in range(len(self.milestones) - 1):
            if self.milestones[i] >= self.milestones[i + 1]:
                raise ValueError("Milestones should be in ascending order")
        # Check the last milestone is smaller than the maximum amount of training steps
        if self.milestones[-1] <= total_num_epochs * total_num_training_steps:
            raise ValueError(
                f"The last milestones must have a value which is equal {total_num_epochs * total_num_training_steps}"
            )

        self.current_milestone = self.milestones.pop(0)
        self.current_scheduler_index = 0
        self.step_count = 0

    def step(self, val_loss):
        # if self.per_epoch set to true, call self.scheduler_step at the end of an epoch. Otherwise call it always.
        self.step_count += 1
        if self.per_epoch and self.step_count % self.steps_per_epoch == 0:
            if self.scheduler_list[self.current_scheduler_index] == "ReduceLROnPlateau":
                self.schedulers[self.current_scheduler_index].step(val_loss)
            else:
                self.scheduler_step()
        elif not self.per_epoch:
            self.scheduler_step()

    def scheduler_step(self):
        """method which perform the scheduler.step() on the current scheduler under use."""

        if self.step_count <= self.current_milestone:
            self.schedulers[self.current_scheduler_index].step()
        else:
            self.current_scheduler_index += 1
            self.schedulers[self.current_scheduler_index].step()
            self.current_milestone = self.milestones.pop(0)

        if self.monitor_lr_scheduler:
            mock_optimizer = None
            self.list_of_lrs.append(self.return_lr(mock_optimizer))

    def return_lr(self, optimizer):
        """method which returns the last scheduler."""
        if isinstance(
            self.schedulers[self.current_scheduler_index],
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = self.schedulers[self.current_scheduler_index].get_last_lr()[0]

        return lr

    def save(self):
        """method which saves the attributes of the class, and the dictionary of the schedulers, in form of dictioanry."""
        schedulers_dicts = [scheduler.state_dict() for scheduler in self.schedulers]
        milestones = self.milestones
        step_count = self.step_count
        current_milestone = self.current_milestone
        current_scheduler_index = self.current_scheduler_index
        list_of_lrs = self.list_of_lrs

        dictionary_values = {
            "schedulers_dicts": schedulers_dicts,
            "milestones": milestones,
            "step_count": step_count,
            "current_milestone": current_milestone,
            "current_scheduler_index": current_scheduler_index,
            "list_of_lrs": list_of_lrs,
        }

        return dictionary_values

    def load(self, dictionary: dict):
        """method that load the paramters into the attributes of this class."""
        for i, scheduler in enumerate(self.schedulers):
            scheduler.load_state_dict(dictionary["schedulers_dicts"][i])
        self.milestones = dictionary["milestones"]
        self.step_count = dictionary["step_count"]
        self.current_milestone = dictionary["current_milestone"]
        self.current_scheduler_index = dictionary["current_scheduler_index"]
        self.list_of_lrs = dictionary["list_of_lrs"]

    def construct_epoch_milestones(self, total_num_training_steps):
        for pos, milestone in enumerate(self.milestones):
            self.milestones[pos] = milestone * total_num_training_steps
        self.steps_per_epoch = total_num_training_steps
