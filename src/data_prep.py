from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from importlib.resources import as_file, files

from src.constants import JOINT_CLASS_PAIRS

TargetKey = Literal["joint", "building", "floor", "longitude", "latitude"]
_PAIR_TO_JOINT_ID = {pair: idx for idx, pair in enumerate(JOINT_CLASS_PAIRS)}
_JOINT_ID_TO_PAIR = np.asarray(JOINT_CLASS_PAIRS, dtype=np.int64)


def encode_joint_labels(
    building_ids: np.ndarray,
    floor_ids: np.ndarray,
    *,
    allow_unknown: bool = False,
) -> np.ndarray:
    """Encode (building, floor) to contiguous deterministic joint ids."""
    b = np.asarray(building_ids).astype(np.int64, copy=False)
    f = np.asarray(floor_ids).astype(np.int64, copy=False)
    if b.shape != f.shape:
        raise ValueError("building_ids and floor_ids must have the same shape.")
    out = np.empty(b.shape, dtype=np.int64)
    it = np.ndindex(b.shape)
    for idx in it:
        pair = (int(b[idx]), int(f[idx]))
        joint_id = _PAIR_TO_JOINT_ID.get(pair, -1)
        if joint_id < 0 and not allow_unknown:
            raise ValueError(f"Unsupported (building, floor) pair for encoding: {pair}")
        out[idx] = joint_id
    return out


def decode_joint_labels(joint_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode contiguous deterministic joint ids to (building_ids, floor_ids)."""
    joint = np.asarray(joint_ids).astype(np.int64, copy=False)
    if np.any(joint < 0) or np.any(joint >= len(JOINT_CLASS_PAIRS)):
        raise ValueError("joint ids are out of supported range.")
    pairs = _JOINT_ID_TO_PAIR[joint.reshape(-1)]
    building = pairs[:, 0].reshape(joint.shape)
    floor = pairs[:, 1].reshape(joint.shape)
    return building, floor


@dataclass(frozen=True)
class UJIDataBundle:
    """Shared train/val payload for all downstream tasks."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    wap_columns: list[str]
    X_train: np.ndarray
    X_val: np.ndarray
    y_joint_train: np.ndarray
    y_joint_val: np.ndarray
    y_building_train: np.ndarray
    y_building_val: np.ndarray
    y_floor_train: np.ndarray
    y_floor_val: np.ndarray
    coordinate_mean: np.ndarray
    coordinate_std: np.ndarray
    y_coordinates_train: np.ndarray
    y_coordinates_val: np.ndarray

    def _get_single_target(self, key: TargetKey) -> tuple[np.ndarray, np.ndarray]:
        if key == "joint":
            return self.y_joint_train, self.y_joint_val
        if key == "building":
            return self.y_building_train, self.y_building_val
        if key == "floor":
            return self.y_floor_train, self.y_floor_val
        if key == "longitude":
            return self.y_coordinates_train[:, 0], self.y_coordinates_val[:, 0]
        if key == "latitude":
            return self.y_coordinates_train[:, 1], self.y_coordinates_val[:, 1]
        raise ValueError(f"Unsupported target key: {key}")

    @staticmethod
    def _as_2d(arr: np.ndarray) -> np.ndarray:
        return arr[:, None] if arr.ndim == 1 else arr

    def get_targets(
        self, targets: Sequence[TargetKey]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return train/val targets for one or many target keys.

        Examples:
            - ["joint"] -> (N, 1), (M, 1)
            - ["longitude", "latitude"] -> (N, 2), (M, 2)
            - ["building", "floor"] -> (N, 2), (M, 2)
            - ["longitude", "latitude", "floor"] -> (N, 3), (M, 3)
        """
        if isinstance(targets, str):
            raise TypeError("targets must be a sequence of target keys, not a string.")
        if len(targets) == 0:
            raise ValueError("targets list cannot be empty.")

        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        for key in targets:
            y_train, y_val = self._get_single_target(key)
            train_parts.append(self._as_2d(y_train))
            val_parts.append(self._as_2d(y_val))

        y_train_cat = np.concatenate(train_parts, axis=1)
        y_val_cat = np.concatenate(val_parts, axis=1)
        return y_train_cat, y_val_cat

    def __add__(self, other: object) -> UJIDataBundle:
        """Concatenate two bundles along the sample axis (train with train, val with val).

        Use this to stack full datasets that share the same coordinate normalization and WAP
        layout (e.g. multiple degradation views of the same split). ``coordinate_mean`` and
        ``coordinate_std`` must match exactly; otherwise targets would be inconsistent with
        ``X`` if they came from different ``prepare_uji_data`` runs.
        """
        if not isinstance(other, UJIDataBundle):
            return NotImplemented
        if self.wap_columns != other.wap_columns:
            raise ValueError(
                "Cannot concat bundles: wap_columns differ "
                f"({len(self.wap_columns)} vs {len(other.wap_columns)})."
            )
        if self.X_train.shape[1] != other.X_train.shape[1]:
            raise ValueError(
                "Cannot concat bundles: feature dimension mismatch "
                f"({self.X_train.shape[1]} vs {other.X_train.shape[1]})."
            )
        if not np.allclose(self.coordinate_mean, other.coordinate_mean):
            raise ValueError(
                "Cannot concat bundles: coordinate_mean differs. "
                "Only combine bundles from the same normalization."
            )
        if not np.allclose(self.coordinate_std, other.coordinate_std):
            raise ValueError(
                "Cannot concat bundles: coordinate_std differs. "
                "Only combine bundles from the same normalization."
            )

        return UJIDataBundle(
            train_df=pd.concat(
                [self.train_df, other.train_df], axis=0, ignore_index=True
            ),
            val_df=pd.concat([self.val_df, other.val_df], axis=0, ignore_index=True),
            wap_columns=list(self.wap_columns),
            X_train=np.concatenate([self.X_train, other.X_train], axis=0),
            X_val=np.concatenate([self.X_val, other.X_val], axis=0),
            y_joint_train=np.concatenate(
                [self.y_joint_train, other.y_joint_train], axis=0
            ),
            y_joint_val=np.concatenate([self.y_joint_val, other.y_joint_val], axis=0),
            y_building_train=np.concatenate(
                [self.y_building_train, other.y_building_train], axis=0
            ),
            y_building_val=np.concatenate(
                [self.y_building_val, other.y_building_val], axis=0
            ),
            y_floor_train=np.concatenate(
                [self.y_floor_train, other.y_floor_train], axis=0
            ),
            y_floor_val=np.concatenate([self.y_floor_val, other.y_floor_val], axis=0),
            coordinate_mean=np.asarray(
                self.coordinate_mean, dtype=np.float32, copy=True
            ),
            coordinate_std=np.asarray(self.coordinate_std, dtype=np.float32, copy=True),
            y_coordinates_train=np.concatenate(
                [self.y_coordinates_train, other.y_coordinates_train], axis=0
            ),
            y_coordinates_val=np.concatenate(
                [self.y_coordinates_val, other.y_coordinates_val], axis=0
            ),
        )

    def __iadd__(self, other: object) -> UJIDataBundle:
        """In-place extend is implemented as rebinding to ``self + other`` (bundle is frozen)."""
        combined = self.__add__(other)
        if combined is NotImplemented:
            return NotImplemented
        return combined

    @classmethod
    def concat(cls, *bundles: UJIDataBundle) -> UJIDataBundle:
        """Concatenate one or more bundles in order (same rules as ``__add__``)."""
        if len(bundles) == 0:
            raise ValueError("concat requires at least one UJIDataBundle.")
        out = bundles[0]
        for b in bundles[1:]:
            out = out + b
        return out


def _uji_resource_paths(
    package: str = "data",
    train_filename: str = "TrainingData.csv",
    val_filename: str = "ValidationData.csv",
) -> tuple[Path, Path]:
    base = files(package).joinpath("UjiIndoorLoc")
    train_res = base.joinpath(train_filename)
    val_res = base.joinpath(val_filename)

    with as_file(train_res) as train_path, as_file(val_res) as val_path:
        if not train_path.is_file() or not val_path.is_file():
            raise FileNotFoundError(
                "Could not resolve packaged UJIIndoorLoc CSV resources under "
                f"package '{package}/UjiIndoorLoc'."
            )
        return Path(train_path), Path(val_path)


def load_uji_train_val(
    package: str = "data",
    train_filename: str = "TrainingData.csv",
    val_filename: str = "ValidationData.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load UJIIndoorLoc train/val from package resources."""
    train_path, val_path = _uji_resource_paths(package, train_filename, val_filename)
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    return train, val


def get_wap_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("WAP")]


def rssi_matrix_to_features(rssi: np.ndarray) -> np.ndarray:
    """UJI raw WAP matrix -> model features (normalized RSSI + detection mask).

    Raw values use ``100`` as the missing / not-detected sentinel; detected RSSI is in dBm
    (typically negative, same convention as ``make_features``).
    """
    rssi = np.asarray(rssi, dtype=np.float32, copy=True)
    detected = (rssi != 100).astype(np.float32)
    rssi[rssi == 100] = -110.0
    rssi = (rssi + 110.0) / 110.0
    rssi = np.clip(rssi, 0.0, 1.0)
    return np.concatenate([rssi, detected], axis=1).astype(np.float32)


def make_features(df: pd.DataFrame, wap_columns: Iterable[str]) -> np.ndarray:
    """RSSI + detection mask transform shared across all models."""
    wap_columns = list(wap_columns)
    rssi = df[wap_columns].to_numpy(dtype=np.float32, copy=True)
    return rssi_matrix_to_features(rssi)


def apply_uji_wap_corruptions(
    rssi_raw: np.ndarray,
    rng: np.random.Generator,
    *,
    dropout_rate: float = 0.0,
    bias_db: float = 0.0,
    noise_std: float = 0.0,
    missing_value: float = 100.0,
) -> np.ndarray:
    """Corrupt raw UJI WAP readings (same idea as ``uji_compare_models_updated`` notebook).

    ``rssi_raw`` uses ``missing_value`` (100) for missing; detected cells are dBm RSSI.

    Applied in order: constant bias, Gaussian noise (detected only), clamp to [-110, -30]
    dBm, then independent Bernoulli dropout on originally-detected cells (set to missing).
    """
    Xc = np.asarray(rssi_raw, dtype=np.float32, copy=True)
    if Xc.ndim != 2:
        raise ValueError("rssi_raw must be 2-D (n_samples, n_waps).")
    detected = Xc != float(missing_value)

    if bias_db != 0.0:
        Xc[detected] = Xc[detected] + float(bias_db)

    if noise_std > 0.0:
        noise = rng.normal(0.0, float(noise_std), size=Xc.shape).astype(np.float32)
        Xc[detected] = Xc[detected] + noise[detected]

    Xc[detected] = np.clip(Xc[detected], -110.0, -30.0)

    dr = float(np.clip(float(dropout_rate), 0.0, 1.0))
    if dr > 0.0:
        drop_mask = rng.random(Xc.shape, dtype=np.float64) < dr
        Xc[detected & drop_mask] = float(missing_value)

    return Xc


def build_joint_labels(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic joint BUILDINGID_FLOOR labels."""
    y_joint_train = encode_joint_labels(
        train["BUILDINGID"].to_numpy(dtype=np.int64, copy=True),
        train["FLOOR"].to_numpy(dtype=np.int64, copy=True),
    )
    y_joint_val = encode_joint_labels(
        val["BUILDINGID"].to_numpy(dtype=np.int64, copy=True),
        val["FLOOR"].to_numpy(dtype=np.int64, copy=True),
    )
    return y_joint_train, y_joint_val


def build_coordinate_targets(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build normalized coordinate targets from (LONGITUDE, LATITUDE)."""
    coord_cols = ["LONGITUDE", "LATITUDE"]
    y_train = train[coord_cols].to_numpy(dtype=np.float32, copy=True)
    y_val = val[coord_cols].to_numpy(dtype=np.float32, copy=True)

    coord_mean = y_train.mean(axis=0).astype(np.float32, copy=False)
    coord_std = y_train.std(axis=0).astype(np.float32, copy=False)
    coord_std = np.clip(coord_std, a_min=1e-6, a_max=None)

    y_train_norm = ((y_train - coord_mean) / coord_std).astype(np.float32, copy=False)
    y_val_norm = ((y_val - coord_mean) / coord_std).astype(np.float32, copy=False)
    return y_train_norm, y_val_norm, coord_mean, coord_std


def prepare_uji_data(
    package: str = "data",
    train_filename: str = "TrainingData.csv",
    val_filename: str = "ValidationData.csv",
) -> UJIDataBundle:
    """Canonical dataset prep entrypoint for all teammates."""
    train_df, val_df = load_uji_train_val(package, train_filename, val_filename)
    wap_columns = get_wap_columns(train_df)

    X_train = make_features(train_df, wap_columns)
    X_val = make_features(val_df, wap_columns)
    y_joint_train, y_joint_val = build_joint_labels(train_df, val_df)
    y_coordinates_train, y_coordinates_val, coordinate_mean, coordinate_std = (
        build_coordinate_targets(train_df, val_df)
    )

    y_building_train = train_df["BUILDINGID"].to_numpy(dtype=np.int64, copy=True)
    y_building_val = val_df["BUILDINGID"].to_numpy(dtype=np.int64, copy=True)
    y_floor_train = train_df["FLOOR"].to_numpy(dtype=np.int64, copy=True)
    y_floor_val = val_df["FLOOR"].to_numpy(dtype=np.int64, copy=True)

    return UJIDataBundle(
        train_df=train_df,
        val_df=val_df,
        wap_columns=wap_columns,
        X_train=X_train,
        X_val=X_val,
        y_joint_train=y_joint_train,
        y_joint_val=y_joint_val,
        y_building_train=y_building_train,
        y_building_val=y_building_val,
        y_floor_train=y_floor_train,
        y_floor_val=y_floor_val,
        coordinate_mean=coordinate_mean,
        coordinate_std=coordinate_std,
        y_coordinates_train=y_coordinates_train,
        y_coordinates_val=y_coordinates_val,
    )


def prepare_uji_data_downgradation(
    bundle: UJIDataBundle,
    *,
    dropout_rate: float = 0.0,
    bias_db: float = 0.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> UJIDataBundle:
    """Return a new bundle with degraded WiFi inputs (train and val features rebuilt).


    - ``bias_db``: constant offset added to all *detected* raw RSSI values (dBm).
    - ``noise_std``: Gaussian noise std (dBm) on detected readings.
    - ``dropout_rate``: per detected cell, probability of forcing missing (100).

    Raw WAP columns are read from ``train_df`` / ``val_df``, corrupted with a single
    ``numpy.random.Generator`` (train matrix first, then val). DataFrames and all
    label arrays are unchanged; only ``X_train`` and ``X_val`` are replaced.
    """
    wap_columns = bundle.wap_columns
    rng = np.random.default_rng(seed)
    raw_train = bundle.train_df[wap_columns].to_numpy(dtype=np.float32, copy=True)
    raw_val = bundle.val_df[wap_columns].to_numpy(dtype=np.float32, copy=True)
    raw_train_c = apply_uji_wap_corruptions(
        raw_train,
        rng,
        dropout_rate=dropout_rate,
        bias_db=bias_db,
        noise_std=noise_std,
    )
    raw_val_c = apply_uji_wap_corruptions(
        raw_val,
        rng,
        dropout_rate=dropout_rate,
        bias_db=bias_db,
        noise_std=noise_std,
    )
    X_train_d = rssi_matrix_to_features(raw_train_c)
    X_val_d = rssi_matrix_to_features(raw_val_c)

    return UJIDataBundle(
        train_df=bundle.train_df,
        val_df=bundle.val_df,
        wap_columns=bundle.wap_columns,
        X_train=X_train_d,
        X_val=X_val_d,
        y_joint_train=bundle.y_joint_train,
        y_joint_val=bundle.y_joint_val,
        y_building_train=bundle.y_building_train,
        y_building_val=bundle.y_building_val,
        y_floor_train=bundle.y_floor_train,
        y_floor_val=bundle.y_floor_val,
        coordinate_mean=bundle.coordinate_mean,
        coordinate_std=bundle.coordinate_std,
        y_coordinates_train=bundle.y_coordinates_train,
        y_coordinates_val=bundle.y_coordinates_val,
    )
