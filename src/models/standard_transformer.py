from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.constants import BUILDING_CLASSES, FLOOR_CLASSES, JOINT_CLASSES
from src.data_prep import decode_joint_labels


# This transfomer implementation design choice

# The original transfomer paper uses the 512 as d_model and 2048 as dim_feedforward. Howerver, 
# this might waist spaces for the features that this dataset provide. We only want to understand
# the Access Point (AP) interactions. So this design is intentionally more conservative in
# capacity than the original paper



@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1


class APTransformerBackbone(nn.Module):
    """Standard transformer encoder for UJIIndoorLoc AP tokens.

    Input convention:
        - `make_features(...)` in data_prep returns [rssi features | detection mask]
          concatenated into one flat vector of length 2 * num_aps.
        - This backbone reshapes that vector into `(batch, num_aps, 2)` where each
          token is `[normalized_rssi, detected_flag]` for one AP.
    """

    def __init__(
        self,
        in_dim: int,
        cfg: TransformerConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = cfg or TransformerConfig()
        if in_dim % 2 != 0:
            raise ValueError(
                "Expected even in_dim because features are [rssi | detected] pairs."
            )
        if cfg.d_model % cfg.nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.cfg = cfg
        self.in_dim = in_dim
        self.num_aps = in_dim // 2
        self.token_proj = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        # Learned AP identity embedding is critical here. Without it, AP #7 and AP #301
        # with the same token values would be indistinguishable to the model.
        self.ap_embedding = nn.Embedding(self.num_aps, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.cls_bias = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.num_layers,
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "ap_index",
            torch.arange(self.num_aps, dtype=torch.long),
            persistent=False,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.ap_embedding.weight, mean=0.0, std=0.02)

    # def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
    #     if x.ndim != 2 or x.shape[1] != self.in_dim:
    #         raise ValueError(f"Expected input shape (batch, {self.in_dim}), got {tuple(x.shape)}")
    #     return x.view(x.shape[0], self.num_aps, 2)
    
    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected input shape (batch, {self.in_dim}), got {tuple(x.shape)}"
            )

        # Current data_prep layout:
        # [all normalized RSSI features | all detected-mask features]
        half = self.num_aps
        rssi = x[:, :half]          # (batch, num_aps)
        detected = x[:, half:]      # (batch, num_aps)

        # Convert to per-AP tokens:
        # token i = [AP_i_rssi, AP_i_detected]
        tokens = torch.stack([rssi, detected], dim=-1)   # (batch, num_aps, 2)
        return tokens


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._to_tokens(x)
        h = self.token_proj(tokens)
        h = h + self.ap_embedding(self.ap_index)[None, :, :]
        cls = self.cls_token.expand(x.shape[0], -1, -1) + self.cls_bias
        seq = torch.cat([cls, h], dim=1)
        seq = self.dropout(seq)
        encoded = self.encoder(seq)
        pooled = self.final_norm(encoded[:, 0, :])
        return pooled


class JointTransformerModel(nn.Module):
    """Standard transformer for joint BUILDINGID_FLOOR classification."""

    def __init__(
        self,
        in_dim: int,
        backbone_cfg: TransformerConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_name = "transformer_joint"
        self.backbone = APTransformerBackbone(in_dim=in_dim, cfg=backbone_cfg)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.cfg.d_model, self.backbone.cfg.d_model),
            nn.GELU(),
            nn.Dropout(self.backbone.cfg.dropout),
            nn.Linear(self.backbone.cfg.d_model, JOINT_CLASSES),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y = targets[:, 0].to(dtype=torch.int64)
        return self.loss_fn(outputs, y)

    def evaluate_outputs(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
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


class MultiTaskTransformerModel(nn.Module):
    """Standard transformer for separate building/floor heads."""

    def __init__(
        self,
        in_dim: int,
        backbone_cfg: TransformerConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_name = "transformer_multitask"
        self.backbone = APTransformerBackbone(in_dim=in_dim, cfg=backbone_cfg)
        d = self.backbone.cfg.d_model
        p = self.backbone.cfg.dropout
        self.building_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, BUILDING_CLASSES)
        )
        self.floor_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, FLOOR_CLASSES)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "building_logits": self.building_head(h),
            "floor_logits": self.floor_head(h),
        }

    def compute_loss(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        y_building = targets[:, 0].to(dtype=torch.int64)
        y_floor = targets[:, 1].to(dtype=torch.int64)
        loss_building = self.loss_fn(outputs["building_logits"], y_building)
        loss_floor = self.loss_fn(outputs["floor_logits"], y_floor)
        return loss_building + loss_floor

    def evaluate_outputs(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> dict[str, float]:
        y_building = torch.round(targets[:, 0]).to(dtype=torch.int64)
        y_floor = torch.round(targets[:, 1]).to(dtype=torch.int64)
        pred_building = torch.argmax(outputs["building_logits"], dim=1).to(dtype=torch.int64)
        pred_floor = torch.argmax(outputs["floor_logits"], dim=1).to(dtype=torch.int64)

        building_acc = float((y_building == pred_building).float().mean().item())
        floor_acc = float((y_floor == pred_floor).float().mean().item())
        joint_acc = float(
            ((y_building == pred_building) & (y_floor == pred_floor)).float().mean().item()
        )
        return {
            "score": joint_acc,
            "joint_accuracy": joint_acc,
            "building_accuracy": building_acc,
            "floor_accuracy": floor_acc,
        }


class CoordinateTransformerModel(nn.Module):
    """Standard transformer for normalized (longitude, latitude) regression."""

    def __init__(
        self,
        in_dim: int,
        coordinate_std: torch.Tensor | list[float] | tuple[float, float] | None = None,
        backbone_cfg: TransformerConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_name = "transformer_coordinate"
        self.backbone = APTransformerBackbone(in_dim=in_dim, cfg=backbone_cfg)
        d = self.backbone.cfg.d_model
        p = self.backbone.cfg.dropout
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, 2)
        )
        self.loss_fn = nn.MSELoss()
        self.register_buffer("coordinate_std", None)
        if coordinate_std is not None:
            std = torch.as_tensor(coordinate_std, dtype=torch.float32)
            if std.shape != (2,):
                raise ValueError("coordinate_std must have shape (2,).")
            self.coordinate_std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(outputs, targets.to(dtype=outputs.dtype))

    def evaluate_outputs(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        if self.coordinate_std is None:
            raise ValueError("Coordinate evaluation in meters requires coordinate_std.")
        pred = outputs.to(dtype=torch.float64)
        true = targets.to(dtype=torch.float64)
        diff = pred - true
        lon_diff_norm = diff[:, 0]
        lat_diff_norm = diff[:, 1]
        euclidean_norm = torch.linalg.norm(diff, dim=1)
        coord_euclidean_norm = float(torch.mean(euclidean_norm).item())
        coord_rmse_norm = float(torch.sqrt(torch.mean(euclidean_norm.square())).item())

        std = self.coordinate_std.to(dtype=torch.float64)
        lon_diff_m = lon_diff_norm * std[0]
        lat_diff_m = lat_diff_norm * std[1]
        euclidean_m = torch.sqrt(lon_diff_m.square() + lat_diff_m.square())
        coord_euclidean_m = float(torch.mean(euclidean_m).item())
        coord_rmse_m = float(torch.sqrt(torch.mean(euclidean_m.square())).item())
        return {
            "score": -coord_euclidean_m,
            "coordinate_mean_euclidean": coord_euclidean_norm,
            "coordinate_rmse": coord_rmse_norm,
            "coordinate_mean_euclidean_m": coord_euclidean_m,
            "coordinate_rmse_m": coord_rmse_m,
        }
