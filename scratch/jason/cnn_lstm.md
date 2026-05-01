# CNN & LSTM Models — Jason Park

## Files Added

| File | Description |
|---|---|
| `src/models/cnn_joint.py` | 1D CNN for joint building+floor classification |
| `src/models/lstm_joint.py` | Bidirectional LSTM for joint building+floor classification |
| `notebooks/cnn_joint_baseline.ipynb` | CNN training and evaluation notebook |
| `notebooks/lstm_joint_baseline.ipynb` | LSTM training and evaluation notebook |
| `notebooks/hyperparam_tuning.ipynb` | Hyperparameter tuning for CNN and LSTM |
| `scratch/jason/` | Personal workspace |

---

## CNN Model (`cnn_joint`)

### Architecture

- **Input**: `(N, 1040)` → reshaped to `(N, 2, 520)` — 2 channels (RSSI + detection mask) over 520 WAPs
- **Conv Block 1**: `Conv1d(2, 64, k=7)` → BatchNorm → ReLU
- **Conv Block 2**: `Conv1d(64, 128, k=5)` → BatchNorm → ReLU → MaxPool1d(2) → 260
- **Conv Block 3**: `Conv1d(128, 256, k=3)` → BatchNorm → ReLU → MaxPool1d(2) → 130
- **Global Average Pool** → `(N, 256)`
- **Classifier**: Dropout(0.25) → Linear(256, 13)
- **Parameters**: 144,845
- **Loss**: CrossEntropyLoss

### Baseline Results (pre-tuning)

| Metric | Value |
|---|---|
| best_epoch | 32 |
| joint_accuracy | 0.7228 |
| building_accuracy | 0.9199 |
| floor_accuracy | 0.7444 |

---

## LSTM Model (`lstm_joint`)

### Architecture

- **Input**: `(N, 1040)` → reshaped to `(N, 520, 2)` — 520 WAP timesteps, 2 features (RSSI + mask)
- **Bidirectional LSTM**: `input=2`, `hidden=256`, `num_layers=2` → output `512`
- **Classifier**: LayerNorm → Dropout(0.25) → Linear(512, 128) → ReLU → Dropout(0.25) → Linear(128, 13)
- **Loss**: CrossEntropyLoss

### Baseline Results (pre-tuning)

| Metric | Value |
|---|---|
| best_epoch | 1 |
| joint_accuracy | 0.0360 |
| building_accuracy | 0.2412 |
| floor_accuracy | 0.1548 |

> **Note**: LSTM baseline failed to learn — LR=2e-3 too high and no gradient clipping.
> Hyperparameter tuning (lower LR + grad clip) is in `hyperparam_tuning.ipynb`.

---

## Comparison vs Baselines (pre-tuning)

| Model | joint_accuracy | building_accuracy | floor_accuracy |
|---|---|---|---|
| transformer_multitask | 0.9433 | 0.9982 | 0.9433 |
| transformer_joint | 0.9352 | 0.9955 | 0.9352 |
| mlp_joint | 0.9019 | 0.9982 | 0.9019 |
| **cnn_joint** | **0.7228** | **0.9199** | **0.7444** |
| **lstm_joint** | **0.0360** | **0.2412** | **0.1548** |

---

## Hyperparameter Tuning Plan

### CNN — configs to try
| Run | LR | Weight Decay |
|---|---|---|
| cnn_lr2e3 | 2e-3 | 1e-4 | baseline |
| cnn_lr1e3 | 1e-3 | 1e-4 | lower LR |
| cnn_lr5e4 | 5e-4 | 1e-4 | even lower LR |
| cnn_wd1e3 | 1e-3 | 1e-3 | stronger weight decay |

### LSTM — configs to try
| Run | LR | Grad Clip |
|---|---|---|
| lstm_lr2e3_noclip | 2e-3 | None | baseline |
| lstm_lr1e3_clip1 | 1e-3 | 1.0 | lower LR + clipping |
| lstm_lr5e4_clip1 | 5e-4 | 1.0 | even lower + clipping |
| lstm_lr1e4_clip1 | 1e-4 | 1.0 | very low + clipping |
