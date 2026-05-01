from __future__ import annotations

import torch
import torch.nn as nn

from src.constants import BUILDING_CLASSES, FLOOR_CLASSES
from src.data_prep import encode_joint_labels


class CNNMultiTaskModel(nn.Module):
    """1D CNN with shared trunk and lightweight task heads.

    This is intentionally aligned with the MLP multitask structure:
    one shared encoder/projection, then separate final heads.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        if in_dim % 2 != 0:
            raise ValueError(f"in_dim must be even, got {in_dim}")

        self.model_name = "cnn_multitask"
        self.n_waps = in_dim // 2
        self.building_classes = BUILDING_CLASSES
        self.floor_classes = FLOOR_CLASSES
        self.floor_loss_weight = 1.0

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

        self.shared_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.head_building = nn.Linear(256, BUILDING_CLASSES)
        self.head_floor = nn.Linear(256, FLOOR_CLASSES)

        self.loss_building = nn.CrossEntropyLoss()
        self.loss_floor = nn.CrossEntropyLoss()

    
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
        z = self.shared_head(self._encode(x))
        building_logits = self.head_building(z)
        floor_logits = self.head_floor(z)
        return torch.cat([building_logits, floor_logits], dim=1)

    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        b_end = self.building_classes
        building_logits = outputs[:, :b_end]
        floor_logits = outputs[:, b_end : b_end + self.floor_classes]
        y_building = targets[:, 0].to(dtype=torch.int64)
        y_floor = targets[:, 1].to(dtype=torch.int64)

        return self.loss_building(
            building_logits, y_building
        ) + self.floor_loss_weight * self.loss_floor(floor_logits, y_floor)

    def evaluate_outputs(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        b_end = self.building_classes
        building_logits = outputs[:, :b_end]
        floor_logits = outputs[:, b_end : b_end + self.floor_classes]

        b_pred = torch.argmax(building_logits, dim=1).to(dtype=torch.int64)
        f_pred = torch.argmax(floor_logits, dim=1).to(dtype=torch.int64)
        true = torch.round(targets).to(dtype=torch.int64)
        b_true, f_true = true[:, 0], true[:, 1]

        b_acc = float((b_true == b_pred).float().mean().item())
        f_acc = float((f_true == f_pred).float().mean().item())

        y_true_joint = encode_joint_labels(
            b_true.detach().cpu().numpy(), f_true.detach().cpu().numpy()
        )
        y_pred_joint = encode_joint_labels(
            b_pred.detach().cpu().numpy(),
            f_pred.detach().cpu().numpy(),
            allow_unknown=True,
        )
        joint_acc = float((y_true_joint == y_pred_joint).mean())

        return {
            "score": joint_acc,
            "building_accuracy": b_acc,
            "floor_accuracy": f_acc,
            "joint_accuracy": joint_acc,
        }
