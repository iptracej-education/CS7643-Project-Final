from __future__ import annotations

from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 2e-3
    weight_decay: float = 1e-4
    optimizer_name: str = "adamw"  # "adamw" or "sgd"
    batch_size: int = 256
    val_batch_size: int = 512
    train_shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    max_epochs: int = 200
    patience: int = 20
    print_every: int = 5
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    scheduler_min_lr: float = 1e-5
    grad_clip_norm: float | None = None
    enable_logging: bool = True
    log_root: str = "logs"
    run_name: str | None = None


@dataclass(frozen=True)
class TrainResult:
    best_epoch: int
    best_metrics: dict[str, float]
    history: list[dict[str, float]]


def _build_optimizer(model: nn.Module, cfg: TrainConfig) -> Optimizer:
    if cfg.optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    if cfg.optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9
        )
    raise ValueError("optimizer_name must be 'adamw' or 'sgd'.")


def _make_run_name(model: nn.Module, cfg: TrainConfig) -> str:
    if cfg.run_name is not None and cfg.run_name.strip():
        return cfg.run_name.strip()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model.model_name}_{stamp}"


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig | None = None,
) -> TrainResult:
    """Generic train loop for compatible torch models."""
    cfg = cfg or TrainConfig()
    if not hasattr(model, "model_name"):
        raise TypeError("Model must define a 'model_name' attribute.")
    if not hasattr(model, "evaluate_outputs") or not callable(model.evaluate_outputs):
        raise TypeError("Model must implement evaluate_outputs(outputs, targets, ...).")
    model.to(device)
    optimizer = _build_optimizer(model, cfg)
    run_name = _make_run_name(model, cfg)
    log_root = Path(cfg.log_root)
    epoch_log_path = log_root / "training" / f"{run_name}.jsonl"
    result_log_path = log_root / "result" / f"{run_name}.json"

    scheduler = None
    if cfg.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.scheduler_min_lr,
        )

    best_score = float("-inf")
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("Each batch must be (inputs, targets, ...).")
            xb = torch.as_tensor(batch[0]).to(device, non_blocking=True)
            yb = torch.as_tensor(batch[1]).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = model.compute_loss(out, yb)
            loss.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            n = xb.shape[0]
            train_loss_sum += float(loss.item()) * n
            train_count += int(n)

        train_loss = train_loss_sum / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        metric_sums: dict[str, float] = {}
        with torch.no_grad():
            for batch in val_loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError("Each batch must be (inputs, targets, ...).")
                xb = torch.as_tensor(batch[0]).to(device, non_blocking=True)
                yb = torch.as_tensor(batch[1]).to(device, non_blocking=True)

                out = model(xb)
                loss = model.compute_loss(out, yb)
                batch_metrics = model.evaluate_outputs(
                    outputs=out,
                    targets=yb,
                )

                n = xb.shape[0]
                val_loss_sum += float(loss.item()) * n
                val_count += int(n)
                for k, v in batch_metrics.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v) * n

        val_loss = val_loss_sum / max(val_count, 1)
        val_metrics = {k: v / max(val_count, 1) for k, v in metric_sums.items()}
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        epoch_metrics.update(val_metrics)
        history.append(epoch_metrics)
        if cfg.enable_logging:
            epoch_log_path.parent.mkdir(parents=True, exist_ok=True)
            with epoch_log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "run_name": run_name,
                            "model_name": model.model_name,
                            **epoch_metrics,
                        }
                    )
                    + "\n"
                )

        if "score" not in epoch_metrics:
            raise ValueError(
                "Model evaluation must return a 'score' metric for training selection."
            )
        score = float(epoch_metrics["score"])
        if scheduler is not None:
            scheduler.step(score)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = dict(epoch_metrics)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % cfg.print_every == 0 or epoch == 1:
            msg = (
                f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} score={score:.4f}"
            )
            print(msg)

        if bad_epochs >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError("Training finished without recording a best state.")
    model.load_state_dict(best_state)
    if cfg.enable_logging:
        result_log_path.parent.mkdir(parents=True, exist_ok=True)
        with result_log_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_name": run_name,
                    "model_name": model.model_name,
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                    "history_len": len(history),
                },
                f,
                indent=2,
                sort_keys=True,
            )
    return TrainResult(
        best_epoch=best_epoch, best_metrics=best_metrics, history=history
    )


def evaluate_on_tensors(
    model: nn.Module,
    X: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 512,
) -> dict[str, float]:
    """Run ``model`` on all of ``(X, y)`` and return metrics from ``evaluate_outputs``.

    Aggregates predictions over the full set before calling ``evaluate_outputs`` so
    metrics like coordinate RMSE match the whole split (not a batch-wise average).
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
    outputs_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = X_t[start:end].to(device, non_blocking=True)
            yb = y_t[start:end].to(device, non_blocking=True)
            out = model(xb)
            loss = model.compute_loss(out, yb)
            loss_sum += float(loss.item()) * (end - start)
            outputs_list.append(out.detach().cpu())
            targets_list.append(yb.detach().cpu())

    all_out = torch.cat(outputs_list, dim=0).to(device)
    all_y = torch.cat(targets_list, dim=0).to(device)
    metrics = dict(model.evaluate_outputs(outputs=all_out, targets=all_y))
    metrics["eval_loss"] = loss_sum / max(n, 1)
    return metrics


def train_from_tensors(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    cfg: TrainConfig | None = None,
) -> TrainResult:
    """Convenience wrapper: build loaders/optimizer from one config dataclass."""
    cfg = cfg or TrainConfig()

    train_ds = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train))
    val_ds = TensorDataset(torch.as_tensor(X_val), torch.as_tensor(y_val))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.train_shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
    )
