# Indoor Localization Project

This is the final project repository for Indoor Location Project in Class CS7643.

The project members consist of :

- Jason W Park (jasonpark9001)
- Wu, Di (dwu400) 
- Kiyoshi Watanabe (kw53) 

## Setup (required)

- Run `pip install -e .` before running notebooks/tests.
- This is required for `pyproject.toml` package imports (`from src...`) to work correctly.

## Dimension alignment

- `get_targets([...])` defines the target tensor layout used by loss/evaluation.
- Current keys are all width-1: `joint`, `building`, `floor`, `longitude`, `latitude`.
- Common shapes:
  - `["joint"]` -> `(N, 1)`
  - `["building", "floor"]` -> `(N, 2)`
  - `["longitude", "latitude"]` -> `(N, 2)`
- Keep model outputs aligned with the chosen target layout.

## Workflow

- Implement a new model in `src/models/`.
- Write a notebook for that model under `notebooks/`.
- Reuse shared preprocessing (`src/data_prep.py`) and training (`src/training.py`).
- Training logs/results are handled by `src/training.py`.

## Registration + run (Important Drill)

- Add the model to `src/models/__init__.py`.
- Import it in notebooks from `src.models`.
- Run `pip install -e .` before any test/run.

## Model contract
- Implement `forward`, `compute_loss`, `evaluate_outputs`.
- `evaluate_outputs` must return a dict containing `"score"`.
- Trainer maximizes `"score"`; for lower-is-better metrics, return a negated score (e.g. `-mae`, `-distance`).
