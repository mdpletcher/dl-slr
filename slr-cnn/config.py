import torch

from dataclasses import dataclass, field
from typing import Tuple, Optional, List

@dataclass
class TrainConfig:
    """Config file for training the SLR CNN"""

    INPUT_HEIGHT: int = 48
    INPUT_WIDTH: int = 24
    INPUT_CHANNELS: List[str] = field(default_factory = lambda: ["swe_mm_model", "T", "SPD", "Q", "W", "R"])
    CHANNELS_1D: List[str] = field(default_factory = lambda: ["swe_mm_model",])

    CONV1_OUT_CHANNELS: int = 16
    CONV2_OUT_CHANNELS: int = 32
    KERNEL_SIZE: int = 3
    PADDING: int = 1
    POOL_KERNEL: Tuple[int, int] = (2, 2)

    FC_HIDDEN_UNITS: int = 128
    OUTPUT_SIZE: int = 1

    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    NUM_EPOCHS: int = 50
    OPTIMIZER: str = "adam"
    LOSS_FN: str = "mse"
    PATIENCE: int = 7
    TUNE: bool = True

    DEVICE: torch.device = field(default_factory = lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    SEED: int = 42
    LOG_INTERVAL: int = 10
    SAVE_MODEL: bool = False

    BASE_DIR: str = "/uufs/chpc.utah.edu/common/home/steenburgh-group10/mpletcher/DL_SLR/"
    PT_SAVE_DIR: str = "/uufs/chpc.utah.edu/common/home/steenburgh-group10/mpletcher/DL_SLR/data/pt/sandbox/"
    PT_SAVE_STR: str = "with_last_weeks_removed"

    ACC_SAVE_DIR: str = ""
    MODEL_SAVE_DIR: str = ""
    HPARAMS_SEARCH_SAVE_DIR: str = "/uufs/chpc.utah.edu/common/home/steenburgh-group10/mpletcher/DL_SLR/data/hparams_search"
    