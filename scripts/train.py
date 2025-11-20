#!/usr/bin/env python3
"""Train a TrajCast model using the configuration from the tutorial notebook."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict
import yaml

from trajcast.data._keys import (
    DISPLACEMENTS_KEY,
    TENSORBOARD_LOG_ROOT_KEY,
    UPDATE_VELOCITIES_KEY,
)
from trajcast.model.training import Trainer

SYSTEM_CONFIG = {
    "example": {
        "atom_type_mapper": {1: 0, 6: 1, 7: 2, 8: 3},
        "num_chem_elements": 4,
        "default_cutoff": 4.0,
    },
    "paracetamol": {
        "atom_type_mapper": {1: 0, 6: 1, 7: 2, 8: 3},
        "num_chem_elements": 4,
        "default_cutoff": 4.0,
    },
    "water": {
        "atom_type_mapper": {1: 0, 8: 1},
        "num_chem_elements": 2,
        "default_cutoff": 3.5,
    },
    "quartz": {
        "atom_type_mapper": {8: 0, 14: 1},
        "num_chem_elements": 2,
        "default_cutoff": 5.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TrajCast on a downloaded dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file. If set, values are loaded first and then overridden by CLI flags.",
    )
    parser.add_argument(
        "--system",
        choices=sorted(SYSTEM_CONFIG.keys()),
        default="example",
        help="Which preset (and atom-type mapping) to use.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory that contains train/val split files (default: data/<system>).",
    )
    parser.add_argument(
        "--train-file",
        default="train.extxyz",
        help="Training file name inside the data root.",
    )
    parser.add_argument(
        "--val-file",
        default="val.extxyz",
        help="Validation file name inside the data root.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Root directory for checkpoints, logs, and TensorBoard (default: runs/<system>).",
    )
    parser.add_argument(
        "--run-name",
        default="trajcast",
        help="Name for this training run (used for WandB run name).",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Weights & Biases project name (defaults to the selected system/dataset).",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Weights & Biases entity/organization (optional).",
    )
    parser.add_argument(
        "--wandb",
        dest="use_wandb",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
        help="Disable Weights & Biases logging.",
    )
    parser.set_defaults(use_wandb=True)
    parser.add_argument(
        "--tensorboard",
        dest="use_tensorboard",
        action="store_true",
        help="Enable TensorBoard logging (default: enabled).",
    )
    parser.add_argument(
        "--no-tensorboard",
        dest="use_tensorboard",
        action="store_false",
        help="Disable TensorBoard logging.",
    )
    parser.set_defaults(use_tensorboard=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=1705,
        help="Random seed used for model initialization.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device identifier understood by torch (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        choices=(32, 64),
        default=32,
        help="Floating point precision for training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs to train (increase for full-scale runs).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Gradient clipping threshold.",
    )
    parser.add_argument(
        "--num-hidden-channels",
        type=int,
        default=16,
        help="Hidden channels per irrep (see tutorial).",
    )
    parser.add_argument(
        "--num-mp-layers",
        type=int,
        default=3,
        help="Number of message passing layers.",
    )
    parser.add_argument(
        "--edge-cutoff",
        type=float,
        default=None,
        help="Radial cutoff used for neighbor construction.",
    )
    parser.add_argument(
        "--vel-max",
        type=float,
        default=0.11,
        help="Velocity magnitude upper bound for the Gaussian basis.",
    )
    parser.add_argument(
        "--num-edge-rbf",
        type=int,
        default=8,
        help="Number of radial basis functions for edges.",
    )
    parser.add_argument(
        "--num-edge-poly-cutoff",
        type=int,
        default=6,
        help="Order of the polynomial cutoff.",
    )
    parser.add_argument(
        "--num-vel-rbf",
        type=int,
        default=8,
        help="Number of radial basis functions for velocity magnitudes.",
    )
    parser.add_argument(
        "--max-rotation-order",
        type=int,
        default=1,
        help="Maximum irrep order for equivariant features.",
    )
    parser.add_argument(
        "--model-type",
        choices=("Flexible", "TrajCast", "EfficientTrajCastModel", "PaiNN"),
        default=None,
        help="Override model type (default keeps config or EfficientTrajCastModel for presets).",
    )
    parser.add_argument(
        "--mlp-width",
        type=int,
        default=16,
        help="Hidden width for the edge/velocity MLPs.",
    )
    parser.add_argument(
        "--restart-latest",
        action="store_true",
        help="Resume from the latest checkpoint if available.",
    )
    parser.add_argument(
        "--o3-backend",
        choices=("e3nn", "cueq"),
        default="e3nn",
        help="Equivariance backend to use (cueq recommended on GPUs).",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=None,
        help="Optional path to dump the resolved YAML config (default: <run-dir>/config_used.yaml).",
    )
    return parser.parse_args()


def build_model_config(
    args: argparse.Namespace, atom_type_mapper: Dict[int, int], edge_cutoff: float
) -> Dict:
    mlp_width = args.mlp_width
    mlp_layers = [mlp_width, mlp_width, mlp_width]
    return {
        "precision": args.precision,
        "num_chem_elements": len(atom_type_mapper),
        "edge_cutoff": edge_cutoff,
        "num_edge_rbf": args.num_edge_rbf,
        "num_edge_poly_cutoff": args.num_edge_poly_cutoff,
        "vel_max": args.vel_max,
        "num_vel_rbf": args.num_vel_rbf,
        "max_rotation_order": args.max_rotation_order,
        "num_hidden_channels": args.num_hidden_channels,
        "num_mp_layers": args.num_mp_layers,
        "edge_mlp_kwargs": {
            "n_neurons": mlp_layers,
            "activation": "silu",
        },
        "vel_mlp_kwargs": {
            "n_neurons": mlp_layers,
            "activation": "silu",
        },
        "nl_gate_kwargs": {
            "activation_scalars": {"o": "tanh", "e": "silu"},
            "activation_gates": {"e": "silu"},
        },
        "conserve_ang_mom": True,
        "o3_backend": args.o3_backend,
        "net_lin_mom": [0.0, 0.0, 0.0],
        "net_ang_mom": [0.0, 0.0, 0.0],
    }


def build_data_config(
    *,
    name: str,
    cutoff_radius: float,
    file_path: Path,
    atom_type_mapper: Dict[int, int],
) -> Dict:
    return {
        "root": str(file_path.parent.resolve()),
        "name": name,
        "cutoff_radius": cutoff_radius,
        "files": [file_path.name],
        "rename": True,
        "atom_type_mapper": atom_type_mapper,
    }


def build_training_config(
    args: argparse.Namespace,
    run_dir: Path,
    training_data: Dict,
    validation_data: Dict,
    model_type: str,
) -> Dict:
    tensorboard_settings = {
        "loss": True,
        "lr": True,
        TENSORBOARD_LOG_ROOT_KEY: str(run_dir / "tb_log"),
        "loss_validation": {
            "data": validation_data,
        },
    }

    return {
        "seed": args.seed,
        "model_type": model_type,
        "device": args.device,
        "restart_latest": args.restart_latest,
        "target_field": "target",
        "reference_fields": [DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY],
        "batch_size": args.batch_size,
        "max_grad_norm": args.max_grad_norm,
        "num_epochs": args.num_epochs,
        "criterion": {
            "loss_type": {"main_loss": "mse"},
            "learnable_weights": False,
        },
        "optimizer": "adam",
        "optimizer_settings": {
            "lr": args.learning_rate,
            "amsgrad": True,
        },
        "scheduler": ["ReduceLROnPlateau"],
        "scheduler_settings": {
            "ReduceLROnPlateau": {"factor": 0.8, "patience": 25, "min_lr": 1e-4}
        },
        "chained_scheduler_hp": {
            "milestones": [10_000_000],
            "per_epoch": True,
            "monitor_lr_scheduler": False,
        },
        "checkpoint_settings": {
            "root": str(run_dir / "checkpoints"),
        },
        "use_tensorboard": args.use_tensorboard,
        "wandb": {
            "enabled": args.use_wandb,
            "project": args.wandb_project or args.system,
            "entity": args.wandb_entity,
            "run_name": args.run_name,
            "dir": str(run_dir / "wandb"),
        },
        "tensorboard_settings": tensorboard_settings,
    }


def main() -> None:
    args = parse_args()
    # If a full config is provided, load it and (optionally) override a few fields.
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Optional CLI overrides
        if args.model_type:
            config.setdefault("training", {})["model_type"] = args.model_type

        if args.run_dir:
            run_dir = args.run_dir.resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_root = str(run_dir / "checkpoints")
            tb_root = str(run_dir / "tb_log")
            config.setdefault("training", {}).setdefault("checkpoint_settings", {})[
                "root"
            ] = ckpt_root
            config["training"].setdefault("tensorboard_settings", {})[
                TENSORBOARD_LOG_ROOT_KEY
            ] = tb_root

        trainer = Trainer(config)
        trainer.train()
        config_out = args.config_out or Path("config_used.yaml")
        trainer.dump_config_to_yaml(str(config_out))
        print(f"[done] Training finished. Config saved to {config_out}")
        return

    system_cfg = SYSTEM_CONFIG[args.system]
    model_type = args.model_type or "EfficientTrajCastModel"

    data_root = args.data_root or Path("data") / args.system
    data_root = data_root.resolve()
    train_file = data_root / args.train_file
    val_file = data_root / args.val_file

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    run_dir = args.run_dir or Path("runs") / args.system
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    cutoff = args.edge_cutoff or system_cfg["default_cutoff"]
    atom_type_mapper = system_cfg["atom_type_mapper"]

    model_config = build_model_config(args, atom_type_mapper, cutoff)
    training_data = build_data_config(
        name=f"{args.system}_training",
        cutoff_radius=cutoff,
        file_path=train_file,
        atom_type_mapper=atom_type_mapper,
    )
    validation_data = build_data_config(
        name=f"{args.system}_validation",
        cutoff_radius=cutoff,
        file_path=val_file,
        atom_type_mapper=atom_type_mapper,
    )

    training_config = build_training_config(
        args=args,
        run_dir=run_dir,
        training_data=training_data,
        validation_data=validation_data,
        model_type=model_type,
    )

    config = {
        "model": model_config,
        "data": training_data,
        "training": training_config,
    }

    trainer = Trainer(config)
    trainer.train()

    config_out = args.config_out or run_dir / "config_used.yaml"
    trainer.dump_config_to_yaml(str(config_out))
    print(f"[done] Training finished. Config saved to {config_out}")


if __name__ == "__main__":
    main()
