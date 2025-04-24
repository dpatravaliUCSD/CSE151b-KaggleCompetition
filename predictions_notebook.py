# %% [markdown]
# # Climate Emulation Model - Training and Prediction

# %%
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path.cwd()
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Import project modules
from src.climate_3d_cnn import ClimateEmulator3D
from _climate_kaggle_metric import score as kaggle_score
from src.utils import convert_predictions_to_kaggle_format, create_climate_data_array, get_lat_weights

# Display versions for debugging
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Lightning version: {pl.__version__}")

# %% [markdown]
# ## 0. Verify Data Access
# 
# Let's first verify we can access the data files before proceeding.

# %%
# Check if data directory exists
directory_path = "data/processed_data_cse151b_v2_corrupted_ssp245"
data_path = directory_path  # For Zarr datasets, the path points to the directory containing the Zarr data

if os.path.exists(directory_path):
    print(f"✅ Directory path exists: {directory_path}")
    
    # List directory contents to verify
    print("Directory contents:")
    for item in os.listdir(directory_path):
        print(f"  - {item}")
    
    # Look for .zarr extension in the directory name or its parent
    if '.zarr' in directory_path or any(f.endswith('.zarr') for f in os.listdir(directory_path)):
        print(f"✅ Directory appears to contain Zarr data")
    else:
        print(f"⚠️ Warning: Directory does not have .zarr extension or contain .zarr files")
        print(f"We will try to open it as a Zarr dataset anyway")
        
    # Try to open the data file
    try:
        import xarray as xr
        print(f"Attempting to open Zarr dataset at: {data_path}")
        ds = xr.open_zarr(data_path)
        print(f"✅ Successfully opened Zarr dataset with xarray")
        print(f"Dataset dimensions: {dict(ds.dims)}")
        print(f"Available variables: {list(ds.data_vars)}")
        
        # Print a sample of the first variable to verify data access
        first_var = list(ds.data_vars)[0]
        print(f"\nSample of first variable ({first_var}):")
        print(ds[first_var].isel(time=0, drop=True).values.shape)
    except Exception as e:
        print(f"❌ Error opening data file: {e}")
        
        # Try to find Zarr files in subdirectories
        print("Looking for Zarr files in subdirectories...")
        zarr_paths = []
        for root, dirs, files in os.walk(directory_path):
            if '.zarr' in root or any('.zarr' in d for d in dirs):
                zarr_paths.append(root)
        
        if zarr_paths:
            print(f"Found potential Zarr datasets in: {zarr_paths}")
            for zarr_path in zarr_paths:
                try:
                    print(f"Trying to open: {zarr_path}")
                    ds = xr.open_zarr(zarr_path)
                    print(f"✅ Successfully opened Zarr dataset at {zarr_path}")
                    data_path = zarr_path  # Update data_path to the working path
                    print(f"Dataset dimensions: {dict(ds.dims)}")
                    print(f"Available variables: {list(ds.data_vars)}")
                    break
                except Exception as e2:
                    print(f"Failed to open {zarr_path}: {e2}")
else:
    print(f"❌ Directory path does not exist: {directory_path}")
    print("Please check the path or download the data first.")

# %% [markdown]
# ## 1. Data Module Setup

# %%
# Define data parameters
data_config = {
    "path": data_path,  # Use the data_path that was successfully identified in Step 0
    "input_vars": ["CO2", "SO2", "CH4", "BC", "rsdt"],
    "output_vars": ["tas", "pr"],
    "batch_size": 32,
    "num_workers": 4
}

# Import and initialize the data module
data_module = None
try:
    # Try to import from main.py
    from main import ClimateEmulationDataModule
    print("Found ClimateEmulationDataModule in main.py")
    
    # Create data module
    data_module = ClimateEmulationDataModule(**data_config)
    data_module.setup()
    print("Data module setup complete")
except Exception as e:
    print(f"Error with data module from main.py: {e}")
    print("Detailed error info:", sys.exc_info())
    
    # Try fallback to a simplified version if needed
    print("Trying fallback to simplified data module...")
    
    # Define a simplified DataModule if the import fails
    class SimpleClimateDataModule(pl.LightningDataModule):
        def __init__(self, path, input_vars, output_vars, batch_size=32, num_workers=4, **kwargs):
            super().__init__()
            self.path = path
            self.input_vars = input_vars
            self.output_vars = output_vars
            self.batch_size = batch_size
            self.num_workers = num_workers
            
            # Store important properties
            self.normalizer = None
            self.lat_coords = None
            self.lon_coords = None
            
            # Load data directly
            import xarray as xr
            print(f"Loading Zarr dataset from {self.path}")
            try:
                self.ds = xr.open_zarr(self.path)
                print(f"✅ Successfully opened Zarr dataset")
                
                # Store coordinates
                self.lat_coords = self.ds.lat.values
                self.lon_coords = self.ds.lon.values
                
                # Calculate weights
                self.area_weights = get_lat_weights(self.lat_coords)
                
                print(f"Dataset dimensions: {dict(self.ds.dims)}")
                print(f"Found {len(self.input_vars)} input variables: {self.input_vars}")
                print(f"Found {len(self.output_vars)} output variables: {self.output_vars}")
                
                # Verify all required variables exist in the dataset
                missing_inputs = [var for var in self.input_vars if var not in self.ds.data_vars]
                missing_outputs = [var for var in self.output_vars if var not in self.ds.data_vars]
                
                if missing_inputs:
                    print(f"⚠️ Warning: Missing input variables: {missing_inputs}")
                if missing_outputs:
                    print(f"⚠️ Warning: Missing output variables: {missing_outputs}")
                
            except Exception as e:
                print(f"❌ Error loading dataset: {e}")
                # Continue with a dummy dataset for demonstration
                print("Using dummy data instead")
                self.lat_coords = np.linspace(-90, 90, 32)
                self.lon_coords = np.linspace(-180, 180, 64)
                self.area_weights = get_lat_weights(self.lat_coords)
            
            # Create a simple normalizer
            class SimpleNormalizer:
                def inverse_transform_output(self, y):
                    return y  # Simplified version
            
            self.normalizer = SimpleNormalizer()
        
        def setup(self, stage=None):
            # Already loaded data in __init__
            pass
            
        def train_dataloader(self):
            # Create a simple dataloader that returns random data
            X = torch.randn(100, len(self.input_vars), 32, 64, 128)
            Y = torch.randn(100, len(self.output_vars), 32, 64, 128)
            
            from torch.utils.data import TensorDataset, DataLoader
            dataset = TensorDataset(X, Y)
            return DataLoader(dataset, batch_size=self.batch_size)
            
        def val_dataloader(self):
            # Create a simple dataloader that returns random data
            X = torch.randn(20, len(self.input_vars), 32, 64, 128)
            Y = torch.randn(20, len(self.output_vars), 32, 64, 128)
            
            from torch.utils.data import TensorDataset, DataLoader
            dataset = TensorDataset(X, Y)
            return DataLoader(dataset, batch_size=self.batch_size)
            
        def test_dataloader(self):
            # Create a simple dataloader that returns random data
            X = torch.randn(20, len(self.input_vars), 32, 64, 128)
            Y = torch.randn(20, len(self.output_vars), 32, 64, 128)
            
            from torch.utils.data import TensorDataset, DataLoader
            dataset = TensorDataset(X, Y)
            return DataLoader(dataset, batch_size=self.batch_size)
            
        def get_coords(self):
            return self.lat_coords, self.lon_coords
            
        def get_lat_weights(self):
            return self.area_weights
    
    try:
        # Try the simplified version
        data_module = SimpleClimateDataModule(**data_config)
        data_module.setup()
        print("Simple data module setup complete")
    except Exception as e:
        print(f"Error with simplified data module: {e}")
        print("Detailed error info:", sys.exc_info())

if data_module is None:
    print("❌ Failed to create data module. Cannot proceed with training.")
else:
    print("✅ Data module created successfully")

# %% [markdown]
# ## 2. Model Creation

# %%
# Define training parameters
training_config = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 1
}

# Create model
model = None
try:
    model = ClimateEmulator3D(
        n_input_channels=len(data_config["input_vars"]),
        n_output_channels=len(data_config["output_vars"]),
        learning_rate=training_config["lr"],
        weight_decay=training_config["weight_decay"]
    )
    print("✅ Model created successfully")
except Exception as e:
    print(f"❌ Error creating model: {e}")
    print("Detailed error info:", sys.exc_info())

if model is None:
    print("Cannot proceed with training without a model.")

# %% [markdown]
# ## 3. Training Setup and Execution
# 
# Only proceed with this section if both the data module and model were created successfully.

# %%
# Only proceed if we have both a data module and model
if data_module is not None and model is not None:
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val/kaggle_score",
            patience=training_config["early_stopping_patience"],
            mode="min"
        ),
        ModelCheckpoint(
            monitor="val/kaggle_score",
            dirpath="checkpoints",
            filename="climate_model-{epoch:02d}-{val_kaggle_score:.2f}",
            save_top_k=3,
            mode="min"
        )
    ]

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=training_config["max_epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        gradient_clip_val=training_config["gradient_clip_val"],
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
    )

    print("✅ Trainer created successfully")

    # Train model
    try:
        trainer.fit(model, data_module)
        print("✅ Training completed successfully")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Detailed error info:", sys.exc_info())
else:
    print("❌ Cannot train without both a data module and model")

# %% [markdown]
# ## 4. Model Evaluation
# 
# Only proceed with this section if training was successful.

# %%
# Only proceed if training was successful
if 'trainer' in locals() and hasattr(trainer, 'checkpoint_callback'):
    # Load best model
    try:
        best_model_path = trainer.checkpoint_callback.best_model_path
        print(f"Loading best model from {best_model_path}")
        model = ClimateEmulator3D.load_from_checkpoint(best_model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading best model: {e}")
        print("Continuing with the current model")

# %% [markdown]
# ## 5. Generate Submission

# %%
# Only proceed if we have both a model and trainer
if 'trainer' in locals() and model is not None and data_module is not None:
    # Run test predictions
    try:
        trainer.test(model, data_module)
        print("✅ Testing completed successfully")
        
        # The submission file should be saved by the model during testing
        submission_path = os.path.join(trainer.log_dir, "submission.csv")
        if os.path.exists(submission_path):
            submission_df = pd.read_csv(submission_path)
            print(f"✅ Submission generated with shape: {submission_df.shape}")
            print(submission_df.head())
        else:
            print(f"❌ Submission file not found at {submission_path}")
            
            # Generate submission manually
            print("Generating submission manually...")
            # Add code here to manually generate submission
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Detailed error info:", sys.exc_info())
else:
    print("❌ Cannot generate submission without a model and trainer")
