from __future__ import annotations

import torch
import torch.nn as nn

from src.constants import JOINT_CLASSES
from src.data_prep import decode_joint_labels


class CNNJointModel(nn.Module):
    """1D CNN for joint BUILDINGID_FLOOR classification.

    The encoder is intentionally larger than the original CNN version so that
    the parameter count is closer to the MLP baseline while preserving the CNN
    inductive bias.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        if in_dim % 2 != 0:
            raise ValueError(f"in_dim must be even, got {in_dim}")

        self.model_name = "cnn_joint"
        self.n_waps = in_dim // 2

        self.conv = nn.Sequential(
            nn.Conv1d(2, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, JOINT_CLASSES),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    # Chcek the comments in cnn_coordinates.py for design
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_waps * 2:
            raise ValueError(
                f"Expected input shape (batch, {self.n_waps * 2}), got {tuple(x.shape)}"
            )
        rssi = x[:, : self.n_waps]
        mask = x[:, self.n_waps :]
        x2d = torch.stack([rssi, mask], dim=1)
        features = self.gap(self.conv(x2d)).squeeze(-1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._encode(x))

    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        y = targets[:, 0].to(dtype=torch.int64)
        return self.loss_fn(outputs, y)

    def evaluate_outputs(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        y_true = torch.round(targets[:, 0]).to(dtype=torch.int64)
        y_pred = torch.argmax(outputs, dim=1).to(dtype=torch.int64)
        joint_acc = float((y_true == y_pred).float().mean().item())

        y_true_np = y_true.detach().cpu().numpy().reshape(-1)
        y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
        b_true_np, f_true_np = decode_joint_labels(y_true_np)
        b_pred_np, f_pred_np = decode_joint_labels(y_pred_np)

        b_true = torch.as_tensor(b_true_np, dtype=torch.int64, device=y_true.device)
        f_true = torch.as_tensor(f_true_np, dtype=torch.int64, device=y_true.device)
        b_pred = torch.as_tensor(b_pred_np, dtype=torch.int64, device=y_true.device)
        f_pred = torch.as_tensor(f_pred_np, dtype=torch.int64, device=y_true.device)

        b_acc = float((b_true == b_pred).float().mean().item())
        f_acc = float((f_true == f_pred).float().mean().item())

        return {
            "score": joint_acc,
            "joint_accuracy": joint_acc,
            "building_accuracy": b_acc,
            "floor_accuracy": f_acc,
        }
