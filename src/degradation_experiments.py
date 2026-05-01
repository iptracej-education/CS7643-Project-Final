from __future__ import annotations

from typing import Any

COMPOSITE_TRAIN_SCENARIOS: list[dict[str, Any]] = [
    {"dropout_rate": 0.20, "bias_db": 0.0, "noise_std": 0.0},
    {"dropout_rate": 0.40, "bias_db": 0.0, "noise_std": 0.0},
    {"dropout_rate": 0.30, "bias_db": 5.0, "noise_std": 2.0},
]

COMPOSITE_SEED_BASE: int = 42


EVAL_DEGRADATION_SCENARIOS: list[tuple[str, float, float, float]] = [
    ("clean", 0.0, 0.0, 0.0),
    ("dropout_0.05", 0.05, 0.0, 0.0),
    ("dropout_0.10", 0.10, 0.0, 0.0),
    ("dropout_0.15", 0.15, 0.0, 0.0),
    ("dropout_0.20", 0.20, 0.0, 0.0),
    ("dropout_0.25", 0.25, 0.0, 0.0),
    ("dropout_0.30", 0.30, 0.0, 0.0),
    ("dropout_0.35", 0.35, 0.0, 0.0),
    ("dropout_0.40", 0.40, 0.0, 0.0),
    ("dropout_0.45", 0.45, 0.0, 0.0),
    ("dropout_0.50", 0.50, 0.0, 0.0),
    ("dropout_0.60", 0.60, 0.0, 0.0),
    ("drop0.25_bias3_noise1", 0.25, 3.0, 1.0),
    ("drop0.35_bias4_noise2", 0.35, 4.0, 2.0),
    ("drop0.40_bias5_noise3", 0.40, 5.0, 3.0),
    ("drop0.15_bias6_noise0", 0.15, 6.0, 0.0),
    ("bias5_only", 0.0, 5.0, 0.0),
    ("noise3_only", 0.0, 0.0, 3.0),
]

EVAL_SEED_BASE: int = 10_000


def eval_scenario_seed(index: int) -> int:
    return EVAL_SEED_BASE + int(index)
