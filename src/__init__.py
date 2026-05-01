"""Core project modules for shared training pipelines."""

from src.data_prep import (
    UJIDataBundle,
    decode_joint_labels,
    prepare_uji_data,
    prepare_uji_data_downgradation,
)
from src.constants import BUILDING_CLASSES, FLOOR_CLASSES, JOINT_CLASSES
from src.models import CoordinateMLPModel, JointMLPModel, MultiTaskMLPModel
from src.training import (
    TrainConfig,
    TrainResult,
    evaluate_on_tensors,
    train_from_tensors,
    train_model,
)

__all__ = [
    "UJIDataBundle",
    "prepare_uji_data",
    "prepare_uji_data_downgradation",
    "JointMLPModel",
    "MultiTaskMLPModel",
    "CoordinateMLPModel",
    "TrainConfig",
    "TrainResult",
    "train_from_tensors",
    "train_model",
    "evaluate_on_tensors",
    "decode_joint_labels",
    "BUILDING_CLASSES",
    "FLOOR_CLASSES",
    "JOINT_CLASSES",
    "CNNJointModel"
]

