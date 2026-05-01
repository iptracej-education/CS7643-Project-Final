from importlib.resources import files
import random
import numpy as np

SEED = 42

UJI_DATA_DIR = files("data").joinpath("UjiIndoorLoc")
UJI_TRAIN_CSV = UJI_DATA_DIR.joinpath("TrainingData.csv")
UJI_VAL_CSV = UJI_DATA_DIR.joinpath("ValidationData.csv")


def seed_everything(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
