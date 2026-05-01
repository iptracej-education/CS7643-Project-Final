from __future__ import annotations

import torch
import torch.nn as nn


class CoordinateMLPModel(nn.Module):
    """MLP for coordinate prediction (longitude, latitude)."""

    def __init__(
        self,
        in_dim: int,
        coordinate_std: torch.Tensor | list[float] | tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = "mlp_coordinates"
        out_dim = 2
        hidden_dims = (1024, 512, 256)
        dropout = 0.25
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend(
                [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()
        self.register_buffer("coordinate_std", None)
        if coordinate_std is not None:
            std = torch.as_tensor(coordinate_std, dtype=torch.float32)
            if std.shape != (2,):
                raise ValueError("coordinate_std must have shape (2,).")
            self.coordinate_std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

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
