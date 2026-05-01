from __future__ import annotations

import torch
import torch.nn as nn


class TemplateModel(nn.Module):
    """Template model for joint BUILDINGID_FLOOR classification."""

    def __init__(
        self,
        in_dim: int,
    ) -> None:
        super().__init__()
        # MAKE SURE that you set the model name, which our training script will call
        self.model_name = "template_model"
        #######################################################
        # TODO: Implement your custom model here
        #######################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #### TODO ####
        pass

    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        #### TODO ####
        pass

    def evaluate_outputs(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        #### TODO ####
        some_metric = 0.0
        return {
            # always return a score, and training script will try to maximize it
            "score": some_metric,
            # other metrics we want to track
        }
