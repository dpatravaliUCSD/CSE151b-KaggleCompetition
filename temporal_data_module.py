import os
from datetime import datetime

import dask.array as da
import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def pad_longitude_4d(input_tensor: torch.Tensor, target_width: int = 74) -> torch.Tensor:
    B, C, H, W = input_tensor.shape
    pad_width = target_width - W
    if pad_width > 0:
        pad_slice = input_tensor[..., :pad_width]  # Copy from the beginning along longitude
        return torch.cat([input_tensor, pad_slice], dim=-1)
    return input_tensor

def create_padded_temporal_windows(data: torch.Tensor, window_size: int):
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

class TemporalClimateDataset(Dataset):
    def __init__(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, output_is_normalized: bool = True):
        assert input_tensor.shape[0] == output_tensor.shape[0], "Mismatched sample size"
        self.input_tensors = input_tensor.float()
        self.output_tensors = output_tensor.float()
        # self.input_tensors = torch.from_numpy(input_tensor).float()
        # self.output_tensors = torch.from_numpy(output_tensor).float()
        self.size = input_tensor.shape[0]
        self.output_is_normalized = output_is_normalized

        log.info(
            f"PaddedClimateDataset: {self.size} samples, input shape: {self.input_tensors.shape}, "
            f"normalized output: {self.output_is_normalized}"
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.output_tensors[idx]


class TemporalDataModule(ClimateEmulationDataModule):
    def __init__(self, context_window=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_window = context_window

    def setup(self, stage: str | None = None):
        log.info(f"Setting up data module with timestep channels and temporal windows for stage: {stage} from {self.hparams.path}")
    
        with xr.open_zarr(self.hparams.path, consolidated=True, chunks={"time": 24}) as ds:
            spatial_template_da = ds["rsdt"].isel(time=0, ssp=0, drop=True)
    
            train_inputs_dask_list, train_outputs_dask_list = [], []
            val_input_dask, val_output_dask = None, None
            val_ssp = "ssp370"
            val_months = 120
    
            log.info(f"Loading data from SSPs: {self.hparams.train_ssps}")
            for ssp in self.hparams.train_ssps:
                ssp_input_dask, ssp_output_dask = _load_process_ssp_data(
                    ds,
                    ssp,
                    self.hparams.input_vars,
                    self.hparams.output_vars,
                    self.hparams.target_member_id,
                    spatial_template_da,
                )
    
                # Add normalized timestep channel per SSP
                num_timesteps = ssp_input_dask.shape[0]
                timestep_channel = da.linspace(0, 1, num_timesteps).reshape((num_timesteps, 1, 1, 1))
                timestep_channel = da.broadcast_to(timestep_channel, (num_timesteps, 1, *ssp_input_dask.shape[2:]))
                ssp_input_dask = da.concatenate([ssp_input_dask, timestep_channel], axis=1)
    
                if ssp == val_ssp:
                    val_input_dask = ssp_input_dask[-val_months:]
                    val_output_dask = ssp_output_dask[-val_months:]
                    train_inputs_dask_list.append(ssp_input_dask[:-val_months])
                    train_outputs_dask_list.append(ssp_output_dask[:-val_months])
                else:
                    train_inputs_dask_list.append(ssp_input_dask)
                    train_outputs_dask_list.append(ssp_output_dask)
    
            train_input_dask = da.concatenate(train_inputs_dask_list, axis=0)
            train_output_dask = da.concatenate(train_outputs_dask_list, axis=0)
    
            input_mean = da.nanmean(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            input_std = da.nanstd(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            output_mean = da.nanmean(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()
            output_std = da.nanstd(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()
    
            self.normalizer.set_input_statistics(mean=input_mean, std=input_std)
            self.normalizer.set_output_statistics(mean=output_mean, std=output_std)
    
            train_input_norm = self.normalizer.normalize(train_input_dask, data_type="input").compute()
            train_output_norm = self.normalizer.normalize(train_output_dask, data_type="output").compute()
            val_input_norm = self.normalizer.normalize(val_input_dask, data_type="input").compute()
            val_output_norm = self.normalizer.normalize(val_output_dask, data_type="output").compute()
    
            test_input_dask, test_output_dask = _load_process_ssp_data(
                ds,
                self.hparams.test_ssp,
                self.hparams.input_vars,
                self.hparams.output_vars,
                self.hparams.target_member_id,
                spatial_template_da,
            )
    
            # Add timestep channel for test SSP
            num_timesteps_test = test_input_dask.shape[0]
            timestep_channel = da.linspace(0, 1, num_timesteps_test).reshape((num_timesteps_test, 1, 1, 1))
            timestep_channel = da.broadcast_to(timestep_channel, (num_timesteps_test, 1, *test_input_dask.shape[2:]))
            test_input_dask = da.concatenate([test_input_dask, timestep_channel], axis=1)
    
            test_slice = slice(-self.hparams.test_months, None)
            test_input_norm = self.normalizer.normalize(test_input_dask[test_slice], data_type="input").compute()
            test_output_raw = test_output_dask[test_slice].compute()

        train_input_tensor = torch.from_numpy(train_input_norm)
        val_input_tensor = torch.from_numpy(val_input_norm)
        test_input_tensor = torch.from_numpy(test_input_norm)
        
        # Apply circular padding to longitude (W) dimension only
        train_input_tensor = pad_longitude_4d(train_input_tensor, target_width=74)
        val_input_tensor = pad_longitude_4d(val_input_tensor, target_width=74)
        test_input_tensor = pad_longitude_4d(test_input_tensor, target_width=74)
        
        # Create temporal windows
        train_input_win = create_padded_temporal_windows(train_input_tensor, self.context_window)
        val_input_win = create_padded_temporal_windows(val_input_tensor, self.context_window)
        test_input_win = create_padded_temporal_windows(test_input_tensor, self.context_window)
        
        # Final dataset creation
        self.train_dataset = TemporalClimateDataset(train_input_win, torch.from_numpy(train_output_norm), output_is_normalized=True)
        self.val_dataset = TemporalClimateDataset(val_input_win, torch.from_numpy(val_output_norm), output_is_normalized=True)
        self.test_dataset = TemporalClimateDataset(test_input_win, torch.from_numpy(test_output_raw), output_is_normalized=False)
    
        log.info(
            f"Datasets created with timestep and temporal augmentation. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}"
        )
