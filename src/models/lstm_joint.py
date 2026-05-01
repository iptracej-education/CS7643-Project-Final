from __future__ import annotations

import torch
import torch.nn as nn

from src.constants import JOINT_CLASSES
from src.data_prep import decode_joint_labels


class LSTMJointModel(nn.Module):
    """BiLSTM for joint BUILDINGID_FLOOR classification.

    Key changes:
    - Add AP identity embeddings so the model can distinguish AP sources.
    - Use a single larger BiLSTM layer instead of a stacked recurrent tower.
    - Replace mean pooling with learned attention pooling.
    These changes keep the model in the same general parameter range as the MLP
    baseline while reducing the optimization burden of a long 520-step sequence.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        if in_dim % 2 != 0:
            raise ValueError(f"in_dim must be even, got {in_dim}")

        self.model_name = "lstm_joint"
        self.n_waps = in_dim // 2

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

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, JOINT_CLASSES),
        )

        self.loss_fn = nn.CrossEntropyLoss()

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
