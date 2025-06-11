# CSE 151B Competition Spring 2025 - Climate Emulation

This repository is a continuation of the original [baseline repository](https://github.com/salvaRC/cse151b-spring2025-competition) provided for the CSE 151B Climate Emulation Competition. While the original repo offered a PyTorch Lightning training script and a basic CNN model, our team has extended it with multiple modeling strategies, temporal data augmentations, and experimental architectures. Each branch consists of self-contained experiments with different approaches, all working from the shared data processing pipeline and configuration structure. 

## | Kaggle Competition Website
    
[CSE 151B Competition - Climate Emulation](https://www.kaggle.com/t/6f53c429d53099dc7cc590f9bf390b10)

## | Overview

This competition challenges participants to develop machine learning models that can accurately emulate a physics-based climate model to project future climate patterns under varying emissions scenarios. Your models will be evaluated on their ability to capture both spatial patterns and temporal variability - key requirements for actionable climate predictions.

  ### Description
  Climate models are essential tools for understanding Earth's future climate, but they are computationally expensive to run. Machine learning approaches offer a promising alternative that 
  could dramatically reduce computational costs while maintaining prediction accuracy. In this competition, you'll work with data from CMIP6 climate model simulations under different Shared 
  Socioeconomic Pathway (SSP) scenarios.

  The training data consists of monthly climate variables (precipitation and temperature) from multiple SSP scenarios. 
  Your task is to develop models that can predict these variables given various input variables including greenhouse gas concentrations and aerosols under new SSP scenarios. 
  Success in this competition requires models that can:

  1. Capture complex spatial patterns of climate variables across the globe
  2. Accurately represent both mean climate states and temporal variability
  3. Learn the physical relationships between input climate forcings and climate responses

  This challenge simulates a real-world problem in climate science: using data from existing scenarios to predict climate under new scenarios, thereby reducing the need for expensive 
  simulation runs.

  ### Evaluation
  Submissions are evaluated using a combination of area-weighted metrics that account for the different sizes of grid cells at different latitudes (cells near the equator cover more area than those near the poles):

  1. **Monthly Area-Weighted RMSE**: Measures the accuracy of your model's monthly predictions. Calculated as: √(weighted_mean((prediction - actual)²))

  2. **Decadal Mean Area-Weighted RMSE**: Specifically evaluates how well your model captures the spatial patterns in the time-averaged climate. This metric is particularly important for 
  capturing long-term climate change signals. This metric is calculated as: √(weighted_mean((time_mean(predictions) - time_mean(actuals))²)), where time_mean is the mean over a 10-year period

  3. **Decadal Standard Deviation Area-Weighted MAE**: Assesses how well your model represents the temporal variability at each location. This metric ensures models don't just predict the mean 
  state correctly but also capture climate variability. This metric is calculated as: weighted_mean(abs(time_std(predictions) - time_std(actuals))), where time_std is the standard deviation over a 10-year period.

  The final score is a weighted combination of these metrics across precipitation and temperature variables. Note that an important consideration for climate emulators is their computational efficiency (i.e., how quickly they can make predictions at inference time). We encourage you to consider this when designing your models, although this competition does not explicitly evaluate that.

  ## | Dataset Details

  For computational efficiency, the data have been coarsened to a (48, 72) lat-lon grid. 

  Input Variables (also called Forcings):
  - ``CO2`` - Carbon dioxide concentrations
  - ``SO2`` - Sulfur dioxide emissions
  - ``CH4`` - Methane concentrations
  - ``BC`` - Black carbon emissions
  - ``rsdt`` - Incoming solar radiation at top of atmosphere (can be useful to inject knowledge of the season/time of year)

  Output Variables to Predict:
  - ``tas`` - Surface air temperature (in Kelvin)
  - ``pr`` - Precipitation rate (in mm/day)
   
   **Note:** You are free to use any or all of the input variables to make your predictions. 
   Similarly, it is up to you how to predict the output variables (e.g. predict both tas and pr together, or predict them separately).

  ### Data Structure

  The dataset is stored in Zarr format, which efficiently handles large multidimensional arrays. The data includes:

  - Spatial dimensions: Global grid with latitude (y) and longitude (x) coordinates
  - Time dimension: Monthly climate data
  - Member ID dimension: Each scenario was simulated three times (i.e. a 3-member ensemble). This is done to account for the internal variability of the climate system (i.e. the fact that the climate system can evolve differently even under the same external forcings). Thus, given any snapshot of monthly forcings, any of the corresponding monthly climate responses from any of the three ensemble members is a valid target.
  - Multiple scenarios: Data from different Shared Socioeconomic Pathways (SSPs)
    - Training: SSP126 (low emissions), SSP370 (high emissions), SSP585 (very high emissions)
    - Validation: Last 10 years of SSP370
    - Testing: SSP245 (intermediate emissions)
  

## Branch Overview

Each branch includes a notebook summarizing the experiment:

| Branch     | Model/Approach Description | Notebook |
|------------|-----------------------------|----------|
| `main`     | **3D U-Net** with time dimension augmentation. This was our best-performing model on both validation and public leaderboard. | `temporal_experiment.ipynb` |
| `rnn`      | **ClimateRNNNet**, an underperforming RNN-based model from our project milestone experiments. | `rnn_experiment.ipynb` |
| `timestep` | 3D U-Net with both **time dimension augmentation** and an **extra timestep channel**. Improved validation loss but worsened Kaggle score. | `timestep_experiment.ipynb` |
| `separate` | **Separate 3D U-Net** models per output variable (`tas` and `pr`). Slightly worse leaderboard score, but informative for future extensions. | `separate_experiment.ipynb` |
