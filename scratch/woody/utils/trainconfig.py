from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    lr: float = 2e-3
    weight_decay: float = 1e-4
    floor_loss_weight: float = 1.1
    max_epochs: int = 200
    patience: int = 20
    print_every: int = 5
