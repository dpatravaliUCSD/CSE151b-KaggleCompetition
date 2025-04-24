import os
import sys
from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import OmegaConf

# Add the project root to the path
project_root = Path.cwd()
sys.path.append(str(project_root))

from src.climate_3d_cnn import ClimateEmulator3D
from src.utils import create_comparison_plots
from _climate_kaggle_metric import score as kaggle_score

# 1. Configuration
# Load configuration
config = OmegaConf.load('configs/main_config.yaml')

# Update data path if needed
config.data.path = 'data/processed_data_cse151b_v2_corrupted_ssp245'

# Set training parameters
config.training.lr = 1e-3
config.training.max_epochs = 100
config.training.early_stopping_patience = 10

# Print configuration
print(OmegaConf.to_yaml(config))

# 2. Data Module Setup
# Create data module
data_module = ClimateEmulationDataModule(**config.data)
data_module.setup()

# 3. Model Creation and Training
# Create model
model = ClimateEmulator3D(
    n_input_channels=len(config.data.input_vars),
    n_output_channels=len(config.data.output_vars),
    learning_rate=config.training.lr,
    weight_decay=config.training.weight_decay
)

# Setup callbacks
callbacks = [
    pl.callbacks.EarlyStopping(
        monitor="val/kaggle_score",
        patience=config.training.early_stopping_patience,
        mode="min"
    ),
    pl.callbacks.ModelCheckpoint(
        monitor="val/kaggle_score",
        dirpath="checkpoints",
        filename="climate_model-{epoch:02d}-{val_kaggle_score:.2f}",
        save_top_k=3,
        mode="min"
    )
]

# Create trainer
trainer = pl.Trainer(
    max_epochs=config.training.max_epochs,
    accelerator="auto",
    devices="auto",
    callbacks=callbacks,
    gradient_clip_val=config.training.gradient_clip_val,
    accumulate_grad_batches=config.training.accumulate_grad_batches,
)

# Train model
trainer.fit(model, data_module)

# 4. Model Evaluation
# Load best model
best_model_path = trainer.checkpoint_callback.best_model_path
model = ClimateEmulator3D.load_from_checkpoint(best_model_path)

# Run validation to get predictions
trainer.validate(model, data_module)

# Get validation predictions
val_dataloader = data_module.val_dataloader()
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in val_dataloader:
        x, y = batch
        y_pred = model(x)
        # Denormalize predictions and targets
        y_pred = model.normalizer.inverse_transform_output(y_pred)
        y = model.normalizer.inverse_transform_output(y)
        all_preds.append(y_pred)
        all_targets.append(y)

all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

# 5. Visualization
# Get coordinates
lat_coords, lon_coords = data_module.get_coords()
time_coords = np.arange(all_preds.shape[0])

# Create visualization for temperature
tas_preds = all_preds[:, 0].numpy()
tas_targets = all_targets[:, 0].numpy()

# Create xarray DataArrays
tas_pred_xr = create_climate_data_array(
    tas_preds,
    time_coords=time_coords,
    lat_coords=lat_coords,
    lon_coords=lon_coords,
    var_name="tas",
    var_unit="K"
)
tas_true_xr = create_climate_data_array(
    tas_targets,
    time_coords=time_coords,
    lat_coords=lat_coords,
    lon_coords=lon_coords,
    var_name="tas",
    var_unit="K"
)

# Plot comparison
fig = create_comparison_plots(
    tas_true_xr.isel(time=0),
    tas_pred_xr.isel(time=0),
    "Temperature (K)"
)
plt.show()

# 6. Generate Submission
# Run test predictions
trainer.test(model, data_module)

# Load and display submission
submission_path = os.path.join(trainer.log_dir, "submission.csv")
submission_df = pd.read_csv(submission_path)
print("Submission shape:", submission_df.shape)
submission_df.head() 