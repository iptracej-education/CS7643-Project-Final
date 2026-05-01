from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from scratch.woody.utils.config import UJI_TRAIN_CSV, UJI_VAL_CSV


def load_train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(UJI_TRAIN_CSV)
    val = pd.read_csv(UJI_VAL_CSV)
    return train, val


def get_wap_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("WAP")]


def make_features(df: pd.DataFrame, wap_columns: Iterable[str]) -> np.ndarray:
    """Shared feature transform for UJIIndoorLoc."""
    wap_columns = list(wap_columns)
    rssi = df[wap_columns].to_numpy(dtype=np.float32, copy=True)
    detected = (rssi != 100).astype(np.float32)
    rssi[rssi == 100] = -110.0
    rssi = (rssi + 110.0) / 110.0
    rssi = np.clip(rssi, 0.0, 1.0)
    return np.concatenate([rssi, detected], axis=1).astype(np.float32)


def build_joint_labels(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    y_train_raw = train["BUILDINGID"].astype(str) + "_" + train["FLOOR"].astype(str)
    y_val_raw = val["BUILDINGID"].astype(str) + "_" + val["FLOOR"].astype(str)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    return y_train, y_val, le
