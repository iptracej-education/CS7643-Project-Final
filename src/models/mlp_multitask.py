from __future__ import annotations

import torch
import torch.nn as nn

from src.constants import BUILDING_CLASSES, FLOOR_CLASSES
from src.data_prep import encode_joint_labels


class MultiTaskMLPModel(nn.Module):
    """MLP with separate heads for building and floor prediction."""

    def __init__(
        self,
        in_dim: int,
    ) -> None:
        super().__init__()
        self.model_name = "mlp_multitask"
        self.building_classes = BUILDING_CLASSES
        self.floor_classes = FLOOR_CLASSES
        hidden_dims = (1024, 512, 256)
        dropout = 0.25
        self.floor_loss_weight = 1.0
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend(
                [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev = h
        self.shared = nn.Sequential(*layers)
        self.head_building = nn.Linear(prev, self.building_classes)
        self.head_floor = nn.Linear(prev, self.floor_classes)
        self.loss_building = nn.CrossEntropyLoss()
        self.loss_floor = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.shared(x)
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
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
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
