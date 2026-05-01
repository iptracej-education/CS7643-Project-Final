from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scratch.woody.utils.trainconfig import TrainConfig


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


def train_multitask_classifier(
    model: MLPMultiTask,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig | None = None,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Train multi-task building/floor classifier with early stopping."""
    if cfg is None:
        cfg = TrainConfig()

    ce_b = nn.CrossEntropyLoss()
    ce_f = nn.CrossEntropyLoss()
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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

