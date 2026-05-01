from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from scratch.woody.uji_utils import build_joint_labels


@dataclass
class XGBJointResult:
    model: XGBClassifier
    label_encoder: object
    joint_acc: float
    b_acc: float
    f_acc: float


def train_xgb_joint(
    X_train: np.ndarray,
    X_val: np.ndarray,
    train: pd.DataFrame,
    val: pd.DataFrame,
    n_estimators: int = 300,
    max_depth: int = 8,
    learning_rate: float = 0.08,
) -> XGBJointResult:
    """Train an XGBoost classifier on joint BUILDINGID_FLOOR labels.

    This is the shared implementation used by notebooks so that the
    preprocessing and evaluation logic stays in one place.
    """
    y_train_joint, y_val_joint, le = build_joint_labels(train, val)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_joint)

    pred_joint = model.predict(X_val)
    joint_acc = float(accuracy_score(y_val_joint, pred_joint))

    # Decompose joint labels into building / floor for separate metrics.
    pred_label = le.inverse_transform(pred_joint)
    pred_building = pd.Series(pred_label).str.split("_").str[0].astype(int).to_numpy()
    pred_floor = pd.Series(pred_label).str.split("_").str[1].astype(int).to_numpy()

    yb_val = val["BUILDINGID"].to_numpy()
    yf_val = val["FLOOR"].to_numpy()
    b_acc = float(accuracy_score(yb_val, pred_building))
    f_acc = float(accuracy_score(yf_val, pred_floor))

    return XGBJointResult(
        model=model,
        label_encoder=le,
        joint_acc=joint_acc,
        b_acc=b_acc,
        f_acc=f_acc,
    )
