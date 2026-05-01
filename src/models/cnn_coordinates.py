from __future__ import annotations

import torch
import torch.nn as nn


class CNNCoordinateModel(nn.Module):
    """1D CNN for coordinate prediction (longitude, latitude).

    Design goals:
    - Preserve the MLP task API: forward / compute_loss / evaluate_outputs
    - Keep the RSSI + detection-mask pairing per AP
    - Increase capacity so the learnable-parameter count is in the same
      order of magnitude as the MLP baseline
    """

    def __init__(
        self,
        in_dim: int,
        coordinate_std: torch.Tensor | list[float] | tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        if in_dim % 2 != 0:
            raise ValueError(f"in_dim must be even, got {in_dim}")

        self.model_name = "cnn_coordinates"
        self.n_waps = in_dim // 2

        # Larger CNN trunk so capacity is closer to the MLP baseline.
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
            #nn.MaxPool1d(kernel_size=2),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 2),
        )

        self.loss_fn = nn.MSELoss()
        self.register_buffer("coordinate_std", None)
        if coordinate_std is not None:
            std = torch.as_tensor(coordinate_std, dtype=torch.float32)
            if std.shape != (2,):
                raise ValueError("coordinate_std must have shape (2,).")
            self.coordinate_std = std

   
    # Input layout from the current shared preprocessing pipeline
    # ----------------------------------------------------------
    # x shape: (batch, 2 * n_waps)
    #
    # The flat feature vector is organized as:
    # [all normalized RSSI values | all detected-mask values]
    #
    # Example for one sample:
    # ----------------------------------------------------------
    # x[i] =
    # [AP1_rssi, AP2_rssi, AP3_rssi, ..., APN_rssi,
    #  AP1_mask, AP2_mask, AP3_mask, ..., APN_mask]
    #
    # where:
    #   - APk_rssi is the normalized RSSI feature for access point k
    #   - APk_mask indicates whether AP k was detected
    #
    # Why reshape for CNN
    # ----------------------------------------------------------
    # A flat vector weakens the local relationship between the two
    # features that belong to the same access point. For the CNN,
    # we want each AP position to carry its paired signal:
    #
    #   AP_k -> [RSSI_k, MASK_k]
    #
    # So we split the flat input into:
    #   rssi: (batch, n_waps)
    #   mask: (batch, n_waps)
    #
    # and stack them into a 2-channel 1D signal:
    #
    #   x2d shape: (batch, 2, n_waps)
    #
    # ----------------------------------------------------------
    # Original flat input:
    #
    #   (batch, 2*n_waps)
    #
    #   [ rssi_1  rssi_2  rssi_3  ...  rssi_N | mask_1  mask_2  mask_3  ...  mask_N ]
    #
    # Split into two aligned feature groups:
    #
    #   rssi -> [ rssi_1  rssi_2  rssi_3  ...  rssi_N ]
    #   mask -> [ mask_1  mask_2  mask_3  ...  mask_N ]
    #
    # Stack into CNN input:
    #
    #   x2d = (batch, 2, n_waps)
    #
    #          channel 0: [ rssi_1  rssi_2  rssi_3  ...  rssi_N ]
    #          channel 1: [ mask_1  mask_2  mask_3  ...  mask_N ]
    #
    # This preserves the per-AP pairing while allowing Conv1d filters
    # to scan across AP positions.
    #
    # After that:
    #   x2d -> Conv1d stack -> Global Average Pooling -> feature vector
    #   features shape: (batch, hidden_dim)
    # 
   
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_waps * 2:
            raise ValueError(
                f"Expected input shape (batch, {self.n_waps * 2}), got {tuple(x.shape)}"
            )
        rssi = x[:, : self.n_waps]
        mask = x[:, self.n_waps :]
        x2d = torch.stack([rssi, mask], dim=1)  # (N, 2, num_waps)
        #features = self.gap(self.conv(x2d)).squeeze(-1)  # (N, 512)
        features = self.gap(self.conv(x2d)).flatten(1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._encode(x))

    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(outputs, targets.to(dtype=outputs.dtype))

    def evaluate_outputs(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        if self.coordinate_std is None:
            raise ValueError("Coordinate evaluation in meters requires coordinate_std.")

        pred = outputs.to(dtype=torch.float64)
        true = targets.to(dtype=torch.float64)
        diff = pred - true

        euclidean_norm = torch.linalg.norm(diff, dim=1)
        coord_euclidean_norm = float(torch.mean(euclidean_norm).item())
        coord_rmse_norm = float(torch.sqrt(torch.mean(euclidean_norm.square())).item())

        std = self.coordinate_std.to(dtype=torch.float64)
        lon_diff_m = diff[:, 0] * std[0]
        lat_diff_m = diff[:, 1] * std[1]
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
