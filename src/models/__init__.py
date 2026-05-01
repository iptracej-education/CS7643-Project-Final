"""Concrete model implementations for the shared interface."""

from src.models.cnn_coordinates import CNNCoordinateModel
from src.models.cnn_joint import CNNJointModel
from src.models.cnn_multitask import CNNMultiTaskModel
from src.models.lstm_coordinates import LSTMCoordinateModel
from src.models.lstm_joint import LSTMJointModel
from src.models.lstm_multitask import LSTMMultiTaskModel
from src.models.mlp_coordinates import CoordinateMLPModel
from src.models.mlp_joint import JointMLPModel
from src.models.mlp_multitask import MultiTaskMLPModel

__all__ = [
    "CNNCoordinateModel",
    "CNNJointModel",
    "CNNMultiTaskModel",
    "LSTMCoordinateModel",
    "LSTMJointModel",
    "LSTMMultiTaskModel",
    "CoordinateMLPModel",
    "JointMLPModel",
    "MultiTaskMLPModel",
]

