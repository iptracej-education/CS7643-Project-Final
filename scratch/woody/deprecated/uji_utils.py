from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path("data/UjiIndoorLoc")
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_train_val(DATA_DIR: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train = pd.read_csv(DATA_DIR / "TrainingData.csv")
        val = pd.read_csv(DATA_DIR / "ValidationData.csv")
        return train, val
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Default Data path {DATA_DIR} not found. Please manually pass the path."
        )


def get_wap_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("WAP")]


def make_features(df: pd.DataFrame, wap_columns: Iterable[str]) -> np.ndarray:
    """Feature transform shared across models.

    - Replace UJIIndoorLoc's no-signal sentinel (100) with -110 dBm.
    - Normalize RSSI from roughly [-110, 0] to [0, 1].
    - Concatenate RSSI and a binary detection mask.
    """
    wap_columns = list(wap_columns)
    rssi = df[wap_columns].to_numpy(dtype=np.float32, copy=True)
    detected = (rssi != 100).astype(np.float32)
    rssi[rssi == 100] = -110.0
    rssi = (rssi + 110.0) / 110.0
    rssi = np.clip(rssi, 0.0, 1.0)
    return np.concatenate([rssi, detected], axis=1).astype(np.float32)


def build_joint_labels(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Create joint BUILDINGID_FLOOR labels and a fitted encoder.

    The returned encoder's classes correspond to strings like "1_3" (building 1, floor 3).
    """
    y_train_raw = train["BUILDINGID"].astype(str) + "_" + train["FLOOR"].astype(str)
    y_val_raw = val["BUILDINGID"].astype(str) + "_" + val["FLOOR"].astype(str)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    return y_train, y_val, le


class MLPMultiTask(nn.Module):
    """Backbone + separate heads for building and floor."""

    def __init__(
        self,
        in_dim: int,
        hidden: Tuple[int, ...] = (1024, 512, 256),
        p_drop: float = 0.25,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(p_drop),
                ]
            )
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head_building = nn.Linear(prev, 3)
        self.head_floor = nn.Linear(prev, 5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        return self.head_building(z), self.head_floor(z)


class MLPJoint(nn.Module):
    """Backbone + single head for joint BUILDINGID_FLOOR label."""

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden: Tuple[int, ...] = (1024, 512, 256),
        p_drop: float = 0.25,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(p_drop),
                ]
            )
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head_joint = nn.Linear(prev, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head_joint(z)


@torch.no_grad()
def evaluate_multitask(
    loader: DataLoader, model: nn.Module, device: torch.device
) -> dict[str, float]:
    """Evaluate building/floor accuracies and joint accuracy."""
    model.eval()
    b_true, b_pred = [], []
    f_true, f_pred = [], []
    for xb, yb, yf in loader:
        xb = xb.to(device)
        lb, lf = model(xb)
        pb = lb.argmax(1).cpu().numpy()
        pf = lf.argmax(1).cpu().numpy()
        b_pred.append(pb)
        f_pred.append(pf)
        b_true.append(yb.numpy())
        f_true.append(yf.numpy())

    b_true_arr = np.concatenate(b_true)
    f_true_arr = np.concatenate(f_true)
    b_pred_arr = np.concatenate(b_pred)
    f_pred_arr = np.concatenate(f_pred)
    b_acc = (b_true_arr == b_pred_arr).mean()
    f_acc = (f_true_arr == f_pred_arr).mean()
    joint = ((b_true_arr == b_pred_arr) & (f_true_arr == f_pred_arr)).mean()
    return {"b_acc": float(b_acc), "f_acc": float(f_acc), "joint": float(joint)}


@torch.no_grad()
def evaluate_joint(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    class_labels: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate joint classifier accuracy, and optionally per-building/floor.

    Args:
        loader: yields (inputs, joint_label_index).
        model: joint classifier.
        device: torch device.
        class_labels: optional array of shape (num_classes,) where each entry
            is a string like "BUILDINGID_FLOOR" (e.g., "1_3"). If provided,
            building/floor accuracies are computed by decomposing predictions
            and targets via these labels.
    """
    model.eval()
    y_true, y_pred = [], []
    for xb, y in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
        y_pred.append(pred)
        y_true.append(y.numpy())
    y_true_arr = np.concatenate(y_true)
    y_pred_arr = np.concatenate(y_pred)
    joint_acc = float(accuracy_score(y_true_arr, y_pred_arr))

    if class_labels is None:
        return {"joint": joint_acc}

    # Decompose joint class indices to (building, floor) using provided labels.
    labels = np.asarray(class_labels)
    true_labels = labels[y_true_arr]
    pred_labels = labels[y_pred_arr]

    def _split_bf(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = np.empty(arr.shape[0], dtype=np.int64)
        f = np.empty(arr.shape[0], dtype=np.int64)
        for i, s in enumerate(arr):
            bs, fs = str(s).split("_")
            b[i] = int(bs)
            f[i] = int(fs)
        return b, f

    b_true, f_true = _split_bf(true_labels)
    b_pred, f_pred = _split_bf(pred_labels)
    b_acc = float((b_true == b_pred).mean())
    f_acc = float((f_true == f_pred).mean())
    return {"joint": joint_acc, "b_acc": b_acc, "f_acc": f_acc}


@dataclass
class TrainConfig:
    lr: float = 2e-3
    weight_decay: float = 1e-4
    floor_loss_weight: float = 1.1
    max_epochs: int = 200
    patience: int = 20
    print_every: int = 5


def train_multitask_classifier(
    model: MLPMultiTask,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig | None = None,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Train multi-task building/floor classifier with early stopping.

    Returns:
        metrics: dict with best validation metrics.
        best_state: state_dict of best model.
    """
    if cfg is None:
        cfg = TrainConfig()

    ce_b = nn.CrossEntropyLoss()
    ce_f = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=4, min_lr=1e-5
    )

    best_joint = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    bad_epochs = 0

    train_ds: TensorDataset = train_loader.dataset  # type: ignore[assignment]

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb, yf in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yf = yf.to(device)

            opt.zero_grad(set_to_none=True)
            lb, lf = model(xb)
            loss = ce_b(lb, yb) + cfg.floor_loss_weight * ce_f(lf, yf)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_ds)
        metrics = evaluate_multitask(val_loader, model, device)
        sched.step(metrics["joint"])

        if metrics["joint"] > best_joint:
            best_joint = metrics["joint"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % cfg.print_every == 0 or epoch == 1:
            lr = opt.param_groups[0]["lr"]
            print(
                f"epoch={epoch:02d} "
                f"loss={train_loss:.4f} "
                f"b_acc={metrics['b_acc']:.4f} "
                f"f_acc={metrics['f_acc']:.4f} "
                f"joint={metrics['joint']:.4f} "
                f"lr={lr:.6f}"
            )

        if bad_epochs >= cfg.patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    final_metrics = evaluate_multitask(val_loader, model, device)
    final_metrics["best_epoch"] = float(best_epoch)
    return final_metrics, best_state


def train_joint_classifier(
    model: MLPJoint,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_labels: np.ndarray,
    cfg: TrainConfig | None = None,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Train joint BUILDINGID_FLOOR classifier with early stopping.

    Returns:
        metrics: dict with best validation metrics (joint, b_acc, f_acc).
        best_state: state_dict of best model.
    """
    if cfg is None:
        cfg = TrainConfig()

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=4, min_lr=1e-5
    )

    best_joint = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    bad_epochs = 0

    train_ds: TensorDataset = train_loader.dataset  # type: ignore[assignment]

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, y in train_loader:
            xb = xb.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_ds)
        metrics = evaluate_joint(val_loader, model, device, class_labels)
        sched.step(metrics["joint"])

        if metrics["joint"] > best_joint:
            best_joint = metrics["joint"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % cfg.print_every == 0 or epoch == 1:
            lr = opt.param_groups[0]["lr"]
            print(
                f"epoch={epoch:02d} "
                f"loss={train_loss:.4f} "
                f"joint={metrics['joint']:.4f} "
                f"b_acc={metrics.get('b_acc', float('nan')):.4f} "
                f"f_acc={metrics.get('f_acc', float('nan')):.4f} "
                f"lr={lr:.6f}"
            )

        if bad_epochs >= cfg.patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    final_metrics = evaluate_joint(val_loader, model, device, class_labels)
    final_metrics["best_epoch"] = float(best_epoch)
    return final_metrics, best_state
