import os
from typing import Dict, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from _climate_kaggle_metric import score as kaggle_score
from src.utils import (
    calculate_weighted_metric,
    create_climate_data_array,
    get_lat_weights,
    get_logger,
    convert_predictions_to_kaggle_format,
)

log = get_logger(__name__)

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Skip connection if dimensions change
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(residual)
        return F.relu(out)

class ClimateEmulator3D(pl.LightningModule):
    def __init__(
        self,
        n_input_channels: int = 5,
        n_output_channels: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.encoder = nn.Sequential(
            # Initial feature extraction
            nn.Conv3d(n_input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            # Residual blocks
            ResidualBlock3D(64, 128),
            ResidualBlock3D(128, 256),
            ResidualBlock3D(256, 128),
            
            # Final conv layer
            nn.Conv3d(128, n_output_channels, kernel_size=1)
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Store validation outputs for epoch end calculations
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Store normalizer from datamodule
        self.normalizer = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def on_fit_start(self) -> None:
        # Get normalizer from datamodule
        self.normalizer = self.trainer.datamodule.normalizer
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y_true = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y_true)
        
        # Log training loss
        self.log("train/loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        x, y_true = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y_true)
        
        # Log validation loss
        self.log("val/loss", loss, prog_bar=True, batch_size=x.size(0), sync_dist=True)
        
        # Store predictions and targets for epoch end calculations
        self.validation_step_outputs.append({
            "y_pred": self.normalizer.inverse_transform_output(y_pred.detach().cpu()),
            "y_true": self.normalizer.inverse_transform_output(y_true.detach().cpu())
        })
        
        return {"loss": loss}
    
    def on_validation_epoch_end(self) -> None:
        # Stack all predictions and targets
        all_preds = torch.cat([x["y_pred"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["y_true"] for x in self.validation_step_outputs])
        
        # Get coordinates
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        time_coords = np.arange(all_preds.shape[0])
        output_vars = ["tas", "pr"]  # Temperature and precipitation
        
        # Convert predictions and targets to Kaggle format
        predictions = all_preds.numpy()
        targets = all_targets.numpy()
        
        submission_df = convert_predictions_to_kaggle_format(
            predictions, 
            time_coords, 
            lat_coords, 
            lon_coords, 
            output_vars
        )
        solution_df = convert_predictions_to_kaggle_format(
            targets,
            time_coords, 
            lat_coords, 
            lon_coords, 
            output_vars
        )
        
        # Calculate Kaggle score
        kaggle_val_score = kaggle_score(solution_df, submission_df, "ID")
        self.log("val/kaggle_score", kaggle_val_score)
        
        # Also calculate individual metrics for monitoring
        area_weights = self.trainer.datamodule.get_lat_weights()
        
        # Variable-specific metric weights as specified in _test_kaggle_metric.py
        metric_weights = {
            "tas": {
                "monthly_rmse": 0.1,
                "time_mean": 1.0,
                "time_std": 1.0,
            },
            "pr": {
                "monthly_rmse": 0.1,
                "time_mean": 1.0,
                "time_std": 0.75,
            },
        }
        var_weights = {"tas": 0.5, "pr": 0.5}
        
        var_scores = {}
        # Calculate metrics for each variable
        for i, var_name in enumerate(output_vars):
            # Extract predictions and targets for this variable
            preds_var = all_preds[:, i].numpy()
            targets_var = all_targets[:, i].numpy()
            
            # Create xarray objects for weighted calculations
            preds_xr = create_climate_data_array(
                preds_var,
                time_coords=time_coords,
                lat_coords=lat_coords,
                lon_coords=lon_coords,
                var_name=var_name,
                var_unit="K" if var_name == "tas" else "mm/day"
            )
            targets_xr = create_climate_data_array(
                targets_var,
                time_coords=time_coords,
                lat_coords=lat_coords,
                lon_coords=lon_coords,
                var_name=var_name,
                var_unit="K" if var_name == "tas" else "mm/day"
            )
            
            # Calculate metrics
            # 1. Monthly RMSE
            diff_squared = (preds_xr - targets_xr) ** 2
            monthly_rmse = calculate_weighted_metric(diff_squared, area_weights, ("time", "y", "x"), "rmse")
            self.log(f"val/{var_name}/monthly_rmse", float(monthly_rmse))
            
            # 2. Time-mean RMSE
            pred_time_mean = preds_xr.mean(dim="time")
            target_time_mean = targets_xr.mean(dim="time")
            mean_diff_squared = (pred_time_mean - target_time_mean) ** 2
            time_mean_rmse = calculate_weighted_metric(mean_diff_squared, area_weights, ("y", "x"), "rmse")
            self.log(f"val/{var_name}/time_mean_rmse", float(time_mean_rmse))
            
            # 3. Time-stddev MAE
            pred_time_std = preds_xr.std(dim="time")
            target_time_std = targets_xr.std(dim="time")
            std_abs_diff = np.abs(pred_time_std - target_time_std)
            time_std_mae = calculate_weighted_metric(std_abs_diff, area_weights, ("y", "x"), "mae")
            self.log(f"val/{var_name}/time_std_mae", float(time_std_mae))
            
            # Calculate combined score for this variable using competition weights
            weights = metric_weights[var_name]
            var_score = (
                weights["monthly_rmse"] * monthly_rmse +
                weights["time_mean"] * time_mean_rmse +
                weights["time_std"] * time_std_mae
            )
            self.log(f"val/{var_name}/score", float(var_score))
            var_scores[var_name] = var_score
        
        # Clear saved outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        """
        Test step is similar to validation step but for final evaluation
        """
        x, y_true = batch
        y_pred = self(x)
        
        # Store predictions for later processing
        self.test_step_outputs.append({
            "y_pred": self.normalizer.inverse_transform_output(y_pred.detach().cpu()),
            "y_true": self.normalizer.inverse_transform_output(y_true.detach().cpu())
        })
        
        return {}

    def on_test_epoch_end(self) -> None:
        """
        Process test predictions and create Kaggle submission
        """
        # Stack all predictions
        all_preds = torch.cat([x["y_pred"] for x in self.test_step_outputs])
        
        # Get coordinates
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        time_coords = np.arange(all_preds.shape[0])
        output_vars = ["tas", "pr"]
        
        # Convert to numpy for submission format
        predictions = all_preds.numpy()
        
        # Create submission DataFrame
        submission_df = convert_predictions_to_kaggle_format(
            predictions, 
            time_coords, 
            lat_coords, 
            lon_coords, 
            output_vars
        )
        
        # Save submission
        output_dir = self.trainer.log_dir if self.trainer.log_dir else "outputs"
        submission_path = os.path.join(output_dir, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        log.info(f"Saved submission to {submission_path}")
        
        # Clear saved outputs
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": 1
            }
        }

def train_model(cfg: DictConfig) -> None:
    """
    Main training function
    
    Args:
        cfg: Hydra configuration object
    """
    # Set random seed
    pl.seed_everything(cfg.seed)
    
    # Create data module
    data_module = ClimateEmulationDataModule(**cfg.data)
    
    # Create model
    model = ClimateEmulator3D(
        n_input_channels=len(cfg.data.input_vars),
        n_output_channels=len(cfg.data.output_vars),
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    
    # Setup logging
    if cfg.use_wandb:
        logger = WandbLogger(
            project=cfg.wandb_project,
            name=cfg.run_name,
            entity=cfg.wandb_entity
        )
        logger.watch(model)
    else:
        logger = None
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.training.early_stopping_patience,
            mode="min"
        ),
        ModelCheckpoint(
            monitor="val/loss",
            dirpath=os.path.join(cfg.output_dir, "checkpoints"),
            filename="climate_model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min"
        )
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)

if __name__ == "__main__":
    # This will be called by Hydra
    pass 