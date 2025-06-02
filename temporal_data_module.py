import os
from datetime import datetime

import dask.array as da
import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from hydra.utils import to_absolute_path
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


try:
    import wandb  # Optional, for logging to Weights & Biases
except ImportError:
    wandb = None

from src.models import get_model
from src.utils import (
    Normalizer,
    calculate_weighted_metric,
    convert_predictions_to_kaggle_format,
    create_climate_data_array,
    create_comparison_plots,
    get_lat_weights,
    get_logger,
    get_trainer_config,
)
from main import (
    _load_process_ssp_data,
    ClimateDataset,
    ClimateEmulationDataModule,
)

log = get_logger(__name__)

def create_padded_temporal_windows(data: torch.Tensor, window_size: int):
    """
    Pads the time dimension to preserve full length after rolling windows.
    Input: [T, C, H, W]
    Output: [T, window_size, C, H, W]
    """
    T, C, H, W = data.shape

    # Pad the front with (t - 1) zero frames: [t-1, C, H, W]
    pad = torch.zeros((window_size - 1, C, H, W), dtype=data.dtype, device=data.device)

    padded_data = torch.cat([pad, data], dim=0)  # shape: [T + t - 1, C, H, W]

    # Now extract T windows of size t
    windows = []
    for i in range(T):
        window = padded_data[i:i + window_size]  # [t, C, H, W]
        windows.append(window)

    return torch.stack(windows, dim=0)  # [T, t, C, H, W]

class TemporalDataModule(ClimateEmulationDataModule):
    def __init__(self, context_window=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_window = context_window


    def setup(self, stage: str | None = None):
        super().setup(stage)

        self.train_dataset.input_tensors = create_padded_temporal_windows(self.train_dataset.input_tensors, self.context_window)
        self.val_dataset.input_tensors = create_padded_temporal_windows(self.val_dataset.input_tensors, self.context_window)
        self.test_dataset.input_tensors = create_padded_temporal_windows(self.test_dataset.input_tensors, self.context_window)

