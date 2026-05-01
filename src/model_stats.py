from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd
import torch.nn as nn


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_stats_dict(model: nn.Module, model_name: Optional[str] = None) -> Dict[str, Any]:
    trainable_params = count_trainable_parameters(model)
    all_params = count_all_parameters(model)

    return {
        "model": model_name if model_name is not None else model.__class__.__name__,
        "trainable_params": trainable_params,
        "all_params": all_params,
        "frozen_params": all_params - trainable_params,
    }


def parameter_summary(model: nn.Module, trainable_only: bool = False) -> pd.DataFrame:
    rows = []
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue

        rows.append(
            {
                "name": name,
                "shape": tuple(param.shape),
                "num_params": int(param.numel()),
                "requires_grad": bool(param.requires_grad),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("num_params", ascending=False).reset_index(drop=True)
    return df