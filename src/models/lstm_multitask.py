from __future__ import annotations

import torch
import torch.nn as nn

from src.constants import BUILDING_CLASSES, FLOOR_CLASSES
from src.data_prep import encode_joint_labels


class LSTMMultiTaskModel(nn.Module):
    """BiLSTM multitask model with AP identity and attention pooling."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        if in_dim % 2 != 0:
            raise ValueError(f"in_dim must be even, got {in_dim}")

        self.model_name = "lstm_multitask"
        self.n_waps = in_dim // 2
        self.building_classes = BUILDING_CLASSES
        self.floor_classes = FLOOR_CLASSES
        self.floor_loss_weight = 1.0

        token_dim = 32
        ap_embed_dim = 16
        hidden_size = 384
        dropout = 0.25

        self.token_proj = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU(),
        )
        self.ap_embedding = nn.Embedding(self.n_waps, ap_embed_dim)

        self.lstm = nn.LSTM(
            input_size=token_dim + ap_embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        feat_dim = hidden_size * 2
        self.attn_pool = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.shared_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_building = nn.Linear(256, BUILDING_CLASSES)
        self.head_floor = nn.Linear(256, FLOOR_CLASSES)

        self.loss_building = nn.CrossEntropyLoss()
        self.loss_floor = nn.CrossEntropyLoss()


    # For _to_sequence and _encode functions, please check lstm_coordinates.py 

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_waps * 2:
            raise ValueError(
                f"Expected input shape (batch, {self.n_waps * 2}), got {tuple(x.shape)}"
            )
        rssi = x[:, : self.n_waps]
        mask = x[:, self.n_waps :]
        seq = torch.stack([rssi, mask], dim=2)

        token_feat = self.token_proj(seq)
        ap_ids = torch.arange(self.n_waps, device=x.device)
        ap_feat = self.ap_embedding(ap_ids).unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([token_feat, ap_feat], dim=2)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._to_sequence(x)
        out, _ = self.lstm(seq)
        attn_logits = self.attn_pool(out).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        pooled = torch.sum(attn_weights * out, dim=1)
        return pooled

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
