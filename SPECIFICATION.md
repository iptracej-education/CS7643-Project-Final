# Indoor Localization Project - Project Specification

## 1. Project Objective

This project evaluates deep learning methods for indoor localization on the **UJIIndoorLoc** dataset. The goal is to compare simple and sequence/attention-based models under a shared training and evaluation protocol, and to measure how localization performance changes when the WiFi input is degraded.

The project focuses on two prediction settings:

- **Classification**
  - building prediction
  - floor prediction
  - optional joint building+floor prediction
- **Regression**
  - 2D coordinate prediction

---

## 2. Scope

### In scope
- Dataset: **UJIIndoorLoc**
- Tasks:
  - building classification
  - floor classification
  - optional joint building+floor classification
  - 2D coordinate regression
- Models:
  - **MLP** baseline
  - **1D CNN** baseline
  - **LSTM** 
  - **Transformer** 
- Robustness study:
  - AP dropout
  - RSSI bias
  - RSSI noise

### Out of scope
- classical ML models as main baselines
- uncertainty modeling

---

## 3. Research Questions

This project is guided by the following questions:

1. How well do different deep learning models perform on indoor localization under a shared clean-data protocol?
2. How much does performance degrade when WiFi signals are corrupted or partially removed?
3. Does a more advanced architecture provide better robustness than a simple baseline such as an MLP?

---

## 4. Shared Experimental Protocol

To make comparisons valid, all models must use the same data pipeline and evaluation rules.

### Fixed across all models
- same UJIIndoorLoc data source
- same train/validation/test split protocol
- same WAP column ordering
- same feature construction
- same target definitions
- same loader construction
- same degradation settings
- same evaluation metrics

A new model should change only the model definition, not the preprocessing or evaluation logic.

---

## 5. Dataset and Split

The project uses the official UJIIndoorLoc files:

- `TrainingData.csv`
- `ValidationData.csv`

### Split protocol
- `TrainingData.csv` is split into:
  - internal training set
  - internal validation set
- `ValidationData.csv` is used as the final held-out test set

### Default split settings
- validation ratio: `0.15`
- seed: `42`
- stratification: by `BUILDINGID_FLOOR` when available

This keeps model comparison fair and reproducible.

---

## 6. Features

All models use the same input features built from the WAP columns.

### Shared feature processing
1. select WAP columns
2. build a binary detection mask
3. replace UJI no-signal value `100` with `-110`
4. normalize RSSI values into `[0, 1]`
5. concatenate normalized RSSI and detection mask

This produces a consistent feature vector for all models.

---

## 7. Targets

### Classification targets
- building label
- floor label
- optional joint building+floor label

### Regression target
- 2D coordinate:
  - `LONGITUDE`
  - `LATITUDE`

The classification and regression pipelines must use the same input features but different target construction.

---

## 8. Models

### Required models
- **MLP** baseline
- **1D CNN** baseline
- **one advanced model**:
  - LSTM, or
  - Transformer

This keeps the project ambitious but still manageable.

### Model implementation rule
Each model must be implemented in its own file under `models/`.

Examples:
- `models/mlp.py`
- `models/cnn.py`
- `models/lstm.py`
- `models/transformer.py`

---

## 9. Training Utilities

Shared utilities should handle:
- seeding
- config loading
- dataset loading
- feature creation
- train/validation splitting
- loader construction
- shared training settings

Task-specific utilities should handle:
- classification target generation and evaluation
- regression target generation and evaluation
- degradation application

This avoids duplicated logic across notebooks and scripts.

---

## 10. Evaluation

### Classification metrics
- building accuracy
- floor accuracy
- joint accuracy if joint classification is used

### Regression metrics

Our task is to predict physical location accurately in real space, not minimize the MSE (training loss). 

Mean Squared Error (MSE) is calculated as: 

$$(\hat{x} - x)^2 + (\hat{y} - y)^2$$

However, MSE has major physical flaws: 
- It is not interpretable in meters.
- It is sensitive to scale.

We should covert it to the distance:

$$\text{distance} = \sqrt{(\Delta x)^2 + (\Delta y)^2}$$

Therefore, we prepare the 4 metrics.

- mean distance error (average performance across all samples.)
- median distance error (case performance (50th percentile))
- RMSE distance (Penalizes large errors more heavily: $\text{RMSE} = \sqrt{\mathbb{E}[d^2]}$)
- 90th percentile distance (Worst-case performance capturing tail risk, but not extreme outliers)

The regression evaluation should emphasize physical localization error, not only training loss.

---

## 11. Degradation Study

To test robustness, trained models will also be evaluated under degraded WiFi inputs.

### Degradation types
- **AP dropout**: remove part of the access point signal input
- **RSSI bias**: shift signal values
- **RSSI noise**: add random perturbation

The same degradation settings must be applied across all models.

---

## 12. Code Organization

Recommended project structure:

```text
project/
├── configs/
├── data/
│   └── UjiIndoorLoc/
├── models/
├── notebooks/
├── results/
├── scripts/
├── src/
│   ├── uji_utils.py
│   └── uji_degradation.py
└── README.md