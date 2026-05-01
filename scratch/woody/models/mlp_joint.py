from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from scratch.woody.utils.trainconfig import TrainConfig


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
def evaluate_joint(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    class_labels: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate joint classifier accuracy, and optionally per-building/floor."""
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


def train_joint_classifier(
    model: MLPJoint,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_labels: np.ndarray,
    cfg: TrainConfig | None = None,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Train joint BUILDINGID_FLOOR classifier with early stopping."""
    if cfg is None:
        cfg = TrainConfig()

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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

