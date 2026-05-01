from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.constants import BUILDING_CLASSES, FLOOR_CLASSES, JOINT_CLASSES
from src.data_prep import decode_joint_labels



# Transformer Attention Design

# We convert flat 1024 feature values into a [520,2] token to get better attention mechanism.

'''
We conver the data into [520,2] tokens for 1 batch to learn all of the weights in the model
We repeat all batches to update the weights and better attention.  

The transformer uses (batch, 520, 2) to define one token per AP, projects each AP into a hidden embedding, 
and then learns AP-to-AP relationships through attention over those embeddings.

Here is an example of tokens, token projections, and attention mechanism.


sample row
────────────────────────────────────────
AP1 = [0.63, 1.0]
AP2 = [0.00, 0.0]
AP3 = [0.71, 1.0]
...
AP520 = [rssi, det]

token projection
────────────────────────────────────────
[0.63, 1.0] -> h1
[0.00, 0.0] -> h2
[0.71, 1.0] -> h3
...
[rssi, det] -> h520

attention
────────────────────────────────────────
h1 attends to h17, h203, h411, ...
h2 attends to h5, h88, ...

'''


## SET transformer implementation design choice ##

# The original transfomer paper uses the 512 as d_model and 2048 as dim_feedforward. However, 
# this might waist spaces for the features that this dataset provide. We only want to understand
# the Access Point (AP) interactions. So this design is intentionally more conservative in
# capacity than the original paper

# With 128/256, it is about 0.13 M parameter per block. With 512 / 2048, this is about 3.15 M parameters per block.
# This is roughtly 24× larger per block.

# For AP setting, 128 / 4 heads / 2 SAB / 256 FFN reads like a small, sane baseline. 
# 512 / 8 heads / 2048 FFN reads like an NLP-scale default transplanted into a smaller structured-data problem. 
# That bigger design would only look rational if we had clear signs of under-fitting, enough training data, and 
# evidence that richer cross-AP interactions were being left on the table.


'''
INPUT TO BACKBONE
────────────────────────────────────────────────────────────
x
shape: (batch, 1040)

one row:
[AP1_rssi, AP1_det, AP2_rssi, AP2_det, ..., AP520_rssi, AP520_det]


STEP 1: RESHAPE TO TOKENS
────────────────────────────────────────────────────────────
(batch, 1040)
   -> view(...)
(batch, 520, 2)

token 0 = [AP1_rssi,   AP1_det]
token 1 = [AP2_rssi,   AP2_det]
token 2 = [AP3_rssi,   AP3_det]
...
token519 = [AP520_rssi, AP520_det]


STEP 2: TOKEN PROJECTION
────────────────────────────────────────────────────────────
(batch, 520, 2)
   -> Linear(2, 128) + LayerNorm
(batch, 520, 128)


STEP 3: ADD AP IDENTITY
────────────────────────────────────────────────────────────
(batch, 520, 128)
 + AP embedding (1, 520, 128)
--------------------------------
(batch, 520, 128)


STEP 4: SAB STACK
────────────────────────────────────────────────────────────
(batch, 520, 128)
   -> SAB
(batch, 520, 128)
   -> SAB
(batch, 520, 128)


STEP 5: PMA POOLING
────────────────────────────────────────────────────────────
(batch, 520, 128)
   -> PMA with 1 seed
(batch, 1, 128)
   -> take [:, 0, :]
(batch, 128)

'''



@dataclass(frozen=True)
class SetTransformerConfig:
    d_model: int = 128
    num_heads: int = 4
    num_sab_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    num_seed_vectors: int = 1


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, num_heads * head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q_proj = self._split_heads(self.q_proj(q))
        k_proj = self._split_heads(self.k_proj(k))
        v_proj = self._split_heads(self.v_proj(k))

        attn_scores = torch.matmul(q_proj, k_proj.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attended = torch.matmul(attn_weights, v_proj)
        attended = self._merge_heads(attended)
        attended = self.o_proj(attended)

        h = self.norm1(q + self.dropout(attended))
        out = self.norm2(h + self.ff(h))
        return out


class SetAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mab = MultiHeadAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mab(x, x)


class PoolingByMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_seed_vectors: int = 1,
    ) -> None:
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seed_vectors, d_model) * 0.02)
        self.mab = MultiHeadAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seeds = self.seed_vectors.expand(x.shape[0], -1, -1)
        return self.mab(seeds, x)


class APSetTransformerBackbone(nn.Module):
    """
    
    Set Transformer backbone for AP tokens.
    
    External input layout from current data_prep.py:
        x.shape == (batch, 2 * num_aps)
        x = [all_rssi | all_detected]

    Internal token layout used by this backbone:
        tokens.shape == (batch, num_aps, 2)
        token_i = [AP_i_rssi, AP_i_detected]

    Important design choice:
    A naive set model would lose which AP produced which reading. That is a mistake.
    We therefore add a learned AP identity embedding before the SAB stack so the model
    keeps AP identity while remaining permutation-equivariant over token order.
    
    """

    def __init__(
        self,
        in_dim: int,
        cfg: SetTransformerConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = cfg or SetTransformerConfig()
        if in_dim % 2 != 0:
            raise ValueError(
                "Expected even in_dim because features are [rssi | detected] pairs."
            )
        self.cfg = cfg
        self.in_dim = in_dim
        self.num_aps = in_dim // 2
        self.token_proj = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.ap_embedding = nn.Embedding(self.num_aps, cfg.d_model)
        self.sab_layers = nn.ModuleList(
            [
                SetAttentionBlock(
                    d_model=cfg.d_model,
                    num_heads=cfg.num_heads,
                    dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_sab_layers)
            ]
        )
        self.pma = PoolingByMultiheadAttention(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            num_seed_vectors=cfg.num_seed_vectors,
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "ap_index",
            torch.arange(self.num_aps, dtype=torch.long),
            persistent=False,
        )
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
        h = self.dropout(h)
        for block in self.sab_layers:
            h = block(h)
        pooled = self.pma(h)[:, 0, :]
        return self.final_norm(pooled)


class JointSetTransformerModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        backbone_cfg: SetTransformerConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_name = "set_transformer_joint"
        self.backbone = APSetTransformerBackbone(in_dim=in_dim, cfg=backbone_cfg)
        d = self.backbone.cfg.d_model
        p = self.backbone.cfg.dropout
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, JOINT_CLASSES)
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


class MultiTaskSetTransformerModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        backbone_cfg: SetTransformerConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_name = "set_transformer_multitask"
        self.backbone = APSetTransformerBackbone(in_dim=in_dim, cfg=backbone_cfg)
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


class CoordinateSetTransformerModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        coordinate_std: torch.Tensor | list[float] | tuple[float, float] | None = None,
        backbone_cfg: SetTransformerConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_name = "set_transformer_coordinate"
        self.backbone = APSetTransformerBackbone(in_dim=in_dim, cfg=backbone_cfg)
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
