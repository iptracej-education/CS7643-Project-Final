from __future__ import annotations

import torch
import torch.nn as nn

from src.constants import JOINT_CLASSES
from src.data_prep import decode_joint_labels


class CNNJointModel(nn.Module):
    """1D CNN for joint BUILDINGID_FLOOR classification.

    Input shape : (N, 1040)  — 520 RSSI values + 520 detection-mask values
    Reshaped to : (N, 2, 520) — 2 channels over 520 WAP positions
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.model_name = "cnn_joint"

        # Each WAP contributes one RSSI channel and one detection-mask channel.
        # in_dim is expected to be 1040 (520 * 2).
        self.n_waps = in_dim // 2  # 520

        self.conv = nn.Sequential(
            # (N, 2, 520) -> (N, 64, 520)
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # (N, 64, 520) -> (N, 128, 260)
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # (N, 128, 260) -> (N, 256, 130)
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Global average pool collapses the spatial dimension -> (N, 256)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256, JOINT_CLASSES),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1040)
        # Split into RSSI and detection mask, stack as 2 channels
        rssi = x[:, : self.n_waps]          # (N, 520)
        mask = x[:, self.n_waps :]          # (N, 520)
        x2d = torch.stack([rssi, mask], dim=1)  # (N, 2, 520)

        features = self.conv(x2d)           # (N, 256, 130)
        pooled = self.gap(features)         # (N, 256, 1)
        flat = pooled.squeeze(-1)           # (N, 256)
        return self.classifier(flat)        # (N, JOINT_CLASSES)

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