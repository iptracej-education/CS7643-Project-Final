#!/usr/bin/env python3
"""
Standalone transformer robustness runner using manually pasted fine-tuned configs.

What this script does:
- loads the UJI bundle
- trains clean-data transformer and set-transformer models for joint, multitask, coordinate
- trains augmented/composite-data versions
- evaluates all models on the robustness grid of degraded Wi-Fi conditions
- computes the clean-vs-aug score comparison table
- saves outputs to CSV files

This removes the Jupyter/IPython kernel from the execution path, but it does not
by itself guarantee deterministic PyTorch behavior. Use --seed and --deterministic
if you want to tighten reproducibility.


How to run:

cd <project directory>
mkdir notebooks/logs/transformer_robustness
python notebooks/transformer_robustness.py --output-dir notebooks/logs/transformer_robustness

Then, open transformer_robustness_result.ipynb and run all cells

"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root that contains src/. Defaults to searching upward from cwd.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/logs/transformer"),
        help="Directory for CSV outputs. Default aligns with transformer_robustness_result.ipynb.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help='Force device, e.g. "cpu" or "cuda:0". Default: auto-detect.',
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional global seed to set before model/data creation.",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic torch/cuDNN behavior where possible.",
    )
    p.add_argument(
        "--print-full-cmp",
        action="store_true",
        help="Print full cmp table instead of just a preview.",
    )
    return p.parse_args()


def set_repro(seed: int | None, deterministic: bool) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Best effort only. Some ops may still be nondeterministic depending on environment.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"[warn] Could not enable full deterministic algorithms: {e}")


def find_project_root(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        root = explicit_root.resolve()
        if not (root / "src").exists():
            raise RuntimeError(f"{root} does not contain src/")
        return root

    root = Path.cwd().resolve()
    while not (root / "src").exists():
        if root.parent == root:
            raise RuntimeError("Could not find project root containing 'src'.")
        root = root.parent
    return root


def _detach_cpu_nested(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _detach_cpu_nested(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_detach_cpu_nested(v) for v in obj)
    raise TypeError(f"Unsupported output type: {type(obj)}")


def _to_device_nested(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device_nested(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device_nested(v, device) for v in obj)
    raise TypeError(f"Unsupported output type: {type(obj)}")


def _concat_nested(chunks):
    first = chunks[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(chunks, dim=0)
    if isinstance(first, dict):
        return {k: _concat_nested([chunk[k] for chunk in chunks]) for k in first}
    if isinstance(first, (list, tuple)):
        return type(first)(
            _concat_nested([chunk[i] for chunk in chunks]) for i in range(len(first))
        )
    raise TypeError(f"Unsupported output type: {type(first)}")


def evaluate_on_tensors_local(
    model: torch.nn.Module,
    X: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 512,
) -> dict[str, float]:
    """Transformer-safe local override of src.training.evaluate_on_tensors.

    Keeps the notebook's exact metric path and metric names:
    - model.evaluate_outputs(outputs=..., targets=...)
    - metrics["eval_loss"]

    Only change: support nested model outputs (e.g. dict outputs).
    """
    model.eval()
    if not hasattr(model, "evaluate_outputs") or not callable(model.evaluate_outputs):
        raise TypeError("Model must implement evaluate_outputs(outputs, targets).")

    X_t = torch.as_tensor(X)
    y_t = torch.as_tensor(y)
    n = int(X_t.shape[0])
    if n == 0:
        raise ValueError("X must be non-empty.")

    loss_sum = 0.0
    outputs_list = []
    targets_list = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = X_t[start:end].to(device, non_blocking=True)
            yb = y_t[start:end].to(device, non_blocking=True)
            out = model(xb)
            loss = model.compute_loss(out, yb)
            loss_sum += float(loss.item()) * (end - start)
            outputs_list.append(_detach_cpu_nested(out))
            targets_list.append(yb.detach().cpu())

    all_out = _to_device_nested(_concat_nested(outputs_list), device)
    all_y = torch.cat(targets_list, dim=0).to(device)
    metrics = dict(model.evaluate_outputs(outputs=all_out, targets=all_y))
    metrics["eval_loss"] = loss_sum / max(n, 1)
    return metrics



def main() -> None:
    args = parse_args()
    set_repro(args.seed, args.deterministic)

    project_root = find_project_root(args.project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("PROJECT_ROOT =", project_root)
    print("sys.path[0]  =", sys.path[0])

    from src.data_prep import prepare_uji_data, prepare_uji_data_downgradation as prepare_uji_data_for_robustness
    from src.degradation_experiments import (
        COMPOSITE_SEED_BASE,
        COMPOSITE_TRAIN_SCENARIOS,
        EVAL_DEGRADATION_SCENARIOS as EVAL_DEGRADATION_SCENARIOS,
        eval_scenario_seed,
    )
    from src.models.set_transformer import (
        CoordinateSetTransformerModel,
        JointSetTransformerModel,
        MultiTaskSetTransformerModel,
        SetTransformerConfig,
    )
    from src.models.standard_transformer import (
        CoordinateTransformerModel,
        JointTransformerModel,
        MultiTaskTransformerModel,
        TransformerConfig,
    )
    from src.training import TrainConfig, train_from_tensors

    # -------------------------------------------------------------------------
    # Paste the FINAL_TUNED_CFGS dictionary printed by
    # fair_transformer_hyperparameter_tuning.ipynb here.
    #
    # Expected shape:
    # FINAL_TUNED_CFGS = {
    #     "standard": {
    #         "joint": {
    #             "lr": ..., "weight_decay": ..., "dropout": ...,
    #             "grad_clip_norm": ..., "max_epochs": ..., "patience": ...,
    #             "print_every": 5, "batch_size": 256, "val_batch_size": 512,
    #             "architecture": {"d_model": 128, "nhead": 4, "num_layers": 2, "dim_feedforward": 256},
    #         },
    #         "multitask": {...},
    #         "coordinate": {...},
    #     },
    #     "set": {
    #         "joint": {
    #             "lr": ..., "weight_decay": ..., "dropout": ...,
    #             "architecture": {"d_model": 128, "num_heads": 4, "num_sab_layers": 2, "dim_feedforward": 256, "num_seed_vectors": 1},
    #         },
    #         ...
    #     },
    # }
    # -------------------------------------------------------------------------
    FINAL_TUNED_CFGS = {'set': {'coordinate': {'lr': 0.001,
   'weight_decay': 0.0005,
   'dropout': 0.1,
   'grad_clip_norm': 1.0,
   'max_epochs': 50,
   'patience': 10,
   'print_every': 5,
   'batch_size': 256,
   'val_batch_size': 512,
   'architecture': {'d_model': 128,
    'num_heads': 4,
    'num_sab_layers': 2,
    'dim_feedforward': 256,
    'num_seed_vectors': 1}},
  'joint': {'lr': 0.0005,
   'weight_decay': 0.0005,
   'dropout': 0.1,
   'grad_clip_norm': 1.0,
   'max_epochs': 80,
   'patience': 15,
   'print_every': 5,
   'batch_size': 256,
   'val_batch_size': 512,
   'architecture': {'d_model': 128,
    'num_heads': 4,
    'num_sab_layers': 2,
    'dim_feedforward': 256,
    'num_seed_vectors': 1}},
  'multitask': {'lr': 0.0005,
   'weight_decay': 0.0005,
   'dropout': 0.1,
   'grad_clip_norm': 1.0,
   'max_epochs': 80,
   'patience': 15,
   'print_every': 5,
   'batch_size': 256,
   'val_batch_size': 512,
   'architecture': {'d_model': 128,
    'num_heads': 4,
    'num_sab_layers': 2,
    'dim_feedforward': 256,
    'num_seed_vectors': 1}}},
 'standard': {'coordinate': {'lr': 0.001,
   'weight_decay': 0.0001,
   'dropout': 0.1,
   'grad_clip_norm': 1.0,
   'max_epochs': 50,
   'patience': 10,
   'print_every': 5,
   'batch_size': 256,
   'val_batch_size': 512,
   'architecture': {'d_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256}},
  'joint': {'lr': 0.001,
   'weight_decay': 0.0005,
   'dropout': 0.1,
   'grad_clip_norm': 1.0,
   'max_epochs': 50,
   'patience': 10,
   'print_every': 5,
   'batch_size': 256,
   'val_batch_size': 512,
   'architecture': {'d_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256}},
  'multitask': {'lr': 0.001,
   'weight_decay': 0.0001,
   'dropout': 0.1,
   'grad_clip_norm': 1.0,
   'max_epochs': 50,
   'patience': 10,
   'print_every': 5,
   'batch_size': 256,
   'val_batch_size': 512,
   'architecture': {'d_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256}}}}

    TASKS = ("joint", "multitask", "coordinate")
    FAMILIES = ("standard", "set")
    TRAIN_KEYS = (
        "lr",
        "weight_decay",
        "max_epochs",
        "patience",
        "print_every",
        "batch_size",
        "val_batch_size",
        "grad_clip_norm",
    )

    DEFAULT_STANDARD_ARCH = {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
    }
    DEFAULT_SET_ARCH = {
        "d_model": 128,
        "num_heads": 4,
        "num_sab_layers": 2,
        "dim_feedforward": 256,
        "num_seed_vectors": 1,
        "dropout": 0.1,
    }

    def _entry_for(family: str, task: str) -> dict:
        return dict(FINAL_TUNED_CFGS[family][task])

    def _train_spec(entry: dict) -> dict:
        # Supports both shapes:
        # 1) flat tuning output: {"lr": ..., "architecture": {...}}
        # 2) nested output: {"train": {...}, "architecture": {...}}
        raw = dict(entry.get("train", entry))
        raw.pop("architecture", None)
        raw.pop("dropout", None)
        return raw

    def _architecture_spec(family: str, entry: dict) -> dict:
        if family == "standard":
            arch = dict(DEFAULT_STANDARD_ARCH)
        elif family == "set":
            arch = dict(DEFAULT_SET_ARCH)
        else:
            raise ValueError(f"Unknown transformer family: {family}")

        arch.update(dict(entry.get("architecture", {})))
        if "dropout" in entry:
            arch["dropout"] = entry["dropout"]
        if "train" in entry and isinstance(entry["train"], dict) and "dropout" in entry["train"]:
            arch["dropout"] = entry["train"]["dropout"]
        return arch

    def validate_manual_transformer_configs(final_cfgs: dict) -> pd.DataFrame:
        if not final_cfgs:
            raise RuntimeError(
                "FINAL_TUNED_CFGS is empty. Copy the final dictionary from "
                "fair_transformer_hyperparameter_tuning_v3_original_names.ipynb first."
            )

        missing = []
        rows = []
        for family in FAMILIES:
            if family not in final_cfgs:
                missing.append((family, "<family missing>"))
                continue
            for task in TASKS:
                if task not in final_cfgs[family]:
                    missing.append((family, task))
                    continue

                entry = dict(final_cfgs[family][task])
                train = _train_spec(entry)
                arch = _architecture_spec(family, entry)

                required_train = ("lr", "weight_decay", "max_epochs", "patience")
                missing_keys = [k for k in required_train if k not in train]
                if missing_keys:
                    raise RuntimeError(
                        f"Training config for {(family, task)} is missing keys: {missing_keys}"
                    )

                rows.append(
                    {
                        "family": family,
                        "task": task,
                        **{k: train.get(k) for k in TRAIN_KEYS},
                        **{f"arch_{k}": v for k, v in arch.items()},
                    }
                )

        if missing:
            raise RuntimeError(f"Missing required tuned configs: {missing}")

        return pd.DataFrame(rows).sort_values(["task", "family"]).reset_index(drop=True)

    tuned_hparams_df = validate_manual_transformer_configs(FINAL_TUNED_CFGS)

    def make_train_cfg(family: str, task: str, run_name: str) -> TrainConfig:
        entry = _entry_for(family, task)
        params = _train_spec(entry)
        return TrainConfig(
            run_name=run_name,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            batch_size=params.get("batch_size", 256),
            val_batch_size=params.get("val_batch_size", 512),
            max_epochs=params["max_epochs"],
            patience=params["patience"],
            print_every=params.get("print_every", 5),
            grad_clip_norm=params.get("grad_clip_norm", 1.0),
        )

    def make_standard_cfg(task: str) -> TransformerConfig:
        entry = _entry_for("standard", task)
        params = _architecture_spec("standard", entry)
        return TransformerConfig(
            d_model=int(params["d_model"]),
            nhead=int(params["nhead"]),
            num_layers=int(params["num_layers"]),
            dim_feedforward=int(params["dim_feedforward"]),
            dropout=float(params["dropout"]),
        )

    def make_set_cfg(task: str) -> SetTransformerConfig:
        entry = _entry_for("set", task)
        params = _architecture_spec("set", entry)
        return SetTransformerConfig(
            d_model=int(params["d_model"]),
            num_heads=int(params["num_heads"]),
            num_sab_layers=int(params["num_sab_layers"]),
            dim_feedforward=int(params["dim_feedforward"]),
            dropout=float(params["dropout"]),
            num_seed_vectors=int(params["num_seed_vectors"]),
        )

    bundle = prepare_uji_data()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    joint_y_train, joint_y_val = bundle.get_targets(["joint"])
    mt_y_train, mt_y_val = bundle.get_targets(["building", "floor"])
    coord_y_train, coord_y_val = bundle.get_targets(["longitude", "latitude"])

    X_train_parts = [bundle.X_train]
    for k, spec in enumerate(COMPOSITE_TRAIN_SCENARIOS):
        deg = prepare_uji_data_for_robustness(
            bundle, seed=COMPOSITE_SEED_BASE + k, **spec
        )
        X_train_parts.append(deg.X_train)
    X_train_composite = np.concatenate(X_train_parts, axis=0)
    n_parts = len(X_train_parts)
    joint_y_composite = np.concatenate([joint_y_train] * n_parts, axis=0)
    mt_y_composite = np.concatenate([mt_y_train] * n_parts, axis=0)
    coord_y_composite = np.concatenate([coord_y_train] * n_parts, axis=0)

    print("device:", device)
    print("robustness scenarios:", len(EVAL_DEGRADATION_SCENARIOS))
    print("X_train clean / composite:", bundle.X_train.shape[0], X_train_composite.shape[0])
    print("\nSelected fine-tuned configs:\n")
    print(tuned_hparams_df.to_string(index=False))

    def eval_robustness_grid(
        model: torch.nn.Module,
        y_val: np.ndarray,
        eval_batch_size: int,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for i, (name, dr, bd, ns) in enumerate(EVAL_DEGRADATION_SCENARIOS):
            seed = eval_scenario_seed(i)
            if name == "clean":
                b = bundle
            else:
                b = prepare_uji_data_for_robustness(
                    bundle,
                    dropout_rate=dr,
                    bias_db=bd,
                    noise_std=ns,
                    seed=seed,
                )
            m = evaluate_on_tensors_local(
                model, b.X_val, y_val, device, batch_size=eval_batch_size
            )
            row = {"scenario": name, **dict(m)}
            rows.append(row)
        return pd.DataFrame(rows)

    def stack_model_results(
        models: dict[str, torch.nn.Module],
        y_vals: dict[str, np.ndarray],
        eval_batch_sizes: dict[str, int],
        train_regime: str,
    ) -> pd.DataFrame:
        parts = []
        for mname, model in models.items():
            df = eval_robustness_grid(model, y_vals[mname], eval_batch_sizes[mname])
            df["model"] = mname
            df["train_regime"] = train_regime
            parts.append(df)
        return pd.concat(parts, ignore_index=True)

    # -------------------------------------------------------------------------
    # 1) Train on clean data
    # -------------------------------------------------------------------------
    joint_standard_cfg = make_standard_cfg("joint")
    joint_standard_train_cfg = make_train_cfg(
        "standard", "joint", "transformer_joint_clean"
    )
    joint_clean = JointTransformerModel(
        in_dim=bundle.X_train.shape[1],
        backbone_cfg=joint_standard_cfg,
    )
    joint_clean_result = train_from_tensors(
        joint_clean,
        bundle.X_train,
        joint_y_train,
        bundle.X_val,
        joint_y_val,
        device,
        joint_standard_train_cfg,
    )

    joint_set_cfg = make_set_cfg("joint")
    joint_set_train_cfg = make_train_cfg(
        "set", "joint", "set_transformer_joint_clean"
    )
    joint_set_clean = JointSetTransformerModel(
        in_dim=bundle.X_train.shape[1],
        backbone_cfg=joint_set_cfg,
    )
    joint_set_clean_result = train_from_tensors(
        joint_set_clean,
        bundle.X_train,
        joint_y_train,
        bundle.X_val,
        joint_y_val,
        device,
        joint_set_train_cfg,
    )

    mt_standard_cfg = make_standard_cfg("multitask")
    mt_standard_train_cfg = make_train_cfg(
        "standard", "multitask", "transformer_multitask_clean"
    )
    mt_clean = MultiTaskTransformerModel(
        in_dim=bundle.X_train.shape[1],
        backbone_cfg=mt_standard_cfg,
    )
    mt_clean_result = train_from_tensors(
        mt_clean,
        bundle.X_train,
        mt_y_train,
        bundle.X_val,
        mt_y_val,
        device,
        mt_standard_train_cfg,
    )

    mt_set_cfg = make_set_cfg("multitask")
    mt_set_train_cfg = make_train_cfg(
        "set", "multitask", "set_transformer_multitask_clean"
    )
    mt_set_clean = MultiTaskSetTransformerModel(
        in_dim=bundle.X_train.shape[1],
        backbone_cfg=mt_set_cfg,
    )
    mt_set_clean_result = train_from_tensors(
        mt_set_clean,
        bundle.X_train,
        mt_y_train,
        bundle.X_val,
        mt_y_val,
        device,
        mt_set_train_cfg,
    )

    coord_standard_cfg = make_standard_cfg("coordinate")
    coord_standard_train_cfg = make_train_cfg(
        "standard", "coordinate", "transformer_coordinate_clean"
    )
    coord_clean = CoordinateTransformerModel(
        in_dim=bundle.X_train.shape[1],
        coordinate_std=bundle.coordinate_std,
        backbone_cfg=coord_standard_cfg,
    )
    coord_clean_result = train_from_tensors(
        coord_clean,
        bundle.X_train,
        coord_y_train,
        bundle.X_val,
        coord_y_val,
        device,
        coord_standard_train_cfg,
    )

    coord_set_cfg = make_set_cfg("coordinate")
    coord_set_train_cfg = make_train_cfg(
        "set", "coordinate", "set_transformer_coordinate_clean"
    )
    coord_set_clean = CoordinateSetTransformerModel(
        in_dim=bundle.X_train.shape[1],
        coordinate_std=bundle.coordinate_std,
        backbone_cfg=coord_set_cfg,
    )
    coord_set_clean_result = train_from_tensors(
        coord_set_clean,
        bundle.X_train,
        coord_y_train,
        bundle.X_val,
        coord_y_val,
        device,
        coord_set_train_cfg,
    )

    models_clean = {
        "transformer_joint": joint_clean,
        "set_transformer_joint": joint_set_clean,
        "transformer_multitask": mt_clean,
        "set_transformer_multitask": mt_set_clean,
        "transformer_coordinate": coord_clean,
        "set_transformer_coordinate": coord_set_clean,
    }
    yval_by_model = {
        "transformer_joint": joint_y_val,
        "set_transformer_joint": joint_y_val,
        "transformer_multitask": mt_y_val,
        "set_transformer_multitask": mt_y_val,
        "transformer_coordinate": coord_y_val,
        "set_transformer_coordinate": coord_y_val,
    }
    eval_batch_sizes = {
        "transformer_joint": _train_spec(_entry_for("standard", "joint")).get("val_batch_size", 512),
        "set_transformer_joint": _train_spec(_entry_for("set", "joint")).get("val_batch_size", 512),
        "transformer_multitask": _train_spec(_entry_for("standard", "multitask")).get("val_batch_size", 512),
        "set_transformer_multitask": _train_spec(_entry_for("set", "multitask")).get("val_batch_size", 512),
        "transformer_coordinate": _train_spec(_entry_for("standard", "coordinate")).get("val_batch_size", 512),
        "set_transformer_coordinate": _train_spec(_entry_for("set", "coordinate")).get("val_batch_size", 512),
    }

    results_clean = stack_model_results(
        models_clean,
        yval_by_model,
        eval_batch_sizes,
        "clean_train",
    )

    # -------------------------------------------------------------------------
    # 2) Train on augmented/composite data
    # -------------------------------------------------------------------------
    joint_aug = JointTransformerModel(
        in_dim=bundle.X_train.shape[1],
        backbone_cfg=joint_standard_cfg,
    )
    joint_aug_result = train_from_tensors(
        joint_aug,
        X_train_composite,
        joint_y_composite,
        bundle.X_val,
        joint_y_val,
        device,
        make_train_cfg("standard", "joint", "transformer_joint_aug"),
    )

    joint_set_aug = JointSetTransformerModel(
        in_dim=bundle.X_train.shape[1],
        backbone_cfg=joint_set_cfg,
    )
    joint_set_aug_result = train_from_tensors(
        joint_set_aug,
        X_train_composite,
        joint_y_composite,
        bundle.X_val,
        joint_y_val,
        device,
        make_train_cfg("set", "joint", "set_transformer_joint_aug"),
    )

    mt_aug = MultiTaskTransformerModel(
        in_dim=bundle.X_train.shape[1],
        backbone_cfg=mt_standard_cfg,
    )
    mt_aug_result = train_from_tensors(
        mt_aug,
        X_train_composite,
        mt_y_composite,
        bundle.X_val,
        mt_y_val,
        device,
        make_train_cfg("standard", "multitask", "transformer_multitask_aug"),
    )

    mt_set_aug = MultiTaskSetTransformerModel(
        in_dim=bundle.X_train.shape[1],
        backbone_cfg=mt_set_cfg,
    )
    mt_set_aug_result = train_from_tensors(
        mt_set_aug,
        X_train_composite,
        mt_y_composite,
        bundle.X_val,
        mt_y_val,
        device,
        make_train_cfg("set", "multitask", "set_transformer_multitask_aug"),
    )

    coord_aug = CoordinateTransformerModel(
        in_dim=bundle.X_train.shape[1],
        coordinate_std=bundle.coordinate_std,
        backbone_cfg=coord_standard_cfg,
    )
    coord_aug_result = train_from_tensors(
        coord_aug,
        X_train_composite,
        coord_y_composite,
        bundle.X_val,
        coord_y_val,
        device,
        make_train_cfg("standard", "coordinate", "transformer_coordinate_aug"),
    )

    coord_set_aug = CoordinateSetTransformerModel(
        in_dim=bundle.X_train.shape[1],
        coordinate_std=bundle.coordinate_std,
        backbone_cfg=coord_set_cfg,
    )
    coord_set_aug_result = train_from_tensors(
        coord_set_aug,
        X_train_composite,
        coord_y_composite,
        bundle.X_val,
        coord_y_val,
        device,
        make_train_cfg("set", "coordinate", "set_transformer_coordinate_aug"),
    )

    models_aug = {
        "transformer_joint": joint_aug,
        "set_transformer_joint": joint_set_aug,
        "transformer_multitask": mt_aug,
        "set_transformer_multitask": mt_set_aug,
        "transformer_coordinate": coord_aug,
        "set_transformer_coordinate": coord_set_aug,
    }

    results_aug = stack_model_results(
        models_aug,
        yval_by_model,
        eval_batch_sizes,
        "aug_train",
    )

    # -------------------------------------------------------------------------
    # 3) Compare score clean train vs aug train
    # -------------------------------------------------------------------------
    scenario_order = [
        "clean",
        "dropout_0.05", "dropout_0.10", "dropout_0.15", "dropout_0.20",
        "dropout_0.25", "dropout_0.30", "dropout_0.35", "dropout_0.40",
        "dropout_0.45", "dropout_0.50", "dropout_0.60",
        "drop0.25_bias3_noise1",
        "drop0.35_bias4_noise2",
        "drop0.40_bias5_noise3",
        "drop0.15_bias6_noise0",
        "bias5_only",
        "noise3_only",
    ]

    cmp = results_clean[["scenario", "model", "score"]].merge(
        results_aug[["scenario", "model", "score"]],
        on=["scenario", "model"],
        suffixes=("_clean_train", "_aug_train"),
    )
    cmp["delta_score"] = cmp["score_aug_train"] - cmp["score_clean_train"]
    cmp["scenario"] = pd.Categorical(cmp["scenario"], categories=scenario_order, ordered=True)

    pivot = cmp.pivot_table(index="scenario", columns="model", values="delta_score", aggfunc="first")

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    tuned_hparams_df.to_csv(outdir / "tuned_hparams.csv", index=False)
    results_clean.to_csv(outdir / "results_clean.csv", index=False)
    results_aug.to_csv(outdir / "results_aug.csv", index=False)
    cmp.sort_values(["model", "scenario"]).to_csv(outdir / "cmp_delta_score.csv", index=False)
    pivot.to_csv(outdir / "cmp_delta_score_pivot.csv")

    def _train_result_row(name: str, regime: str, result) -> dict:
        row = {
            "model": name,
            "train_regime": regime,
            "best_epoch": result.best_epoch,
        }
        row.update(result.best_metrics)
        return row

    best_metrics_df = pd.DataFrame([
        _train_result_row("transformer_joint", "clean_train", joint_clean_result),
        _train_result_row("set_transformer_joint", "clean_train", joint_set_clean_result),
        _train_result_row("transformer_multitask", "clean_train", mt_clean_result),
        _train_result_row("set_transformer_multitask", "clean_train", mt_set_clean_result),
        _train_result_row("transformer_coordinate", "clean_train", coord_clean_result),
        _train_result_row("set_transformer_coordinate", "clean_train", coord_set_clean_result),
        _train_result_row("transformer_joint", "aug_train", joint_aug_result),
        _train_result_row("set_transformer_joint", "aug_train", joint_set_aug_result),
        _train_result_row("transformer_multitask", "aug_train", mt_aug_result),
        _train_result_row("set_transformer_multitask", "aug_train", mt_set_aug_result),
        _train_result_row("transformer_coordinate", "aug_train", coord_aug_result),
        _train_result_row("set_transformer_coordinate", "aug_train", coord_set_aug_result),
    ])
    best_metrics_df.to_csv(outdir / "best_metrics_summary.csv", index=False)

    print(f"\nSaved outputs to: {outdir}\n")
    print("best_metrics_summary.csv")
    print(best_metrics_df.to_string(index=False))

    print("\ncmp_delta_score preview:\n")
    cmp_sorted = cmp.sort_values(["model", "scenario"])
    if args.print_full_cmp:
        print(cmp_sorted.to_string(index=False))
    else:
        with pd.option_context("display.max_rows", 20, "display.max_columns", None, "display.width", None):
            print(cmp_sorted.head(20).to_string(index=False))
            print("...")
            print(cmp_sorted.tail(20).to_string(index=False))

    print("\nDelta score pivot:\n")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
