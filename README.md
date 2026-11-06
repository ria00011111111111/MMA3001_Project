# MMA3001 – Occupancy State Forecasting

## Project Overview

This project investigates the use of machine learning to forecast occupancy states in smart building infrastructure.

Using historical occupancy sensor observations, the project develops a forecasting model to predict the recorded occupancy state of a floor-space at the next consecutive five-minute interval. The aim is to investigate whether temporal and calendar-based information can provide useful predictions of future occupancy and how the resulting model performs against simpler forecasting approaches.

Developed as part of **MMA3001 – Numerical Methods and Machine Learning** at Monash University.

## Engineering Problem

Building occupancy information can support decisions relating to building operation, resource allocation and infrastructure management. However, occupancy sensor datasets may contain irregular observation intervals, imbalanced occupancy classes and periods with missing consecutive observations.

This project addresses the problem by developing a supervised classification approach that uses historical sensor observations to forecast the next recorded occupancy state.

**Forecasting objective:** Predict the occupancy state at the next consecutive five-minute interval using information available at the current forecast origin.

The model predicts one of three recorded occupancy states:

- `CurrentlyOccupied`
- `NotOccupied`
- `RecentlyOccupied`

The target represents the sensor's recorded classification rather than independently verified physical occupancy.

## Dataset

The project uses an occupancy sensor dataset containing approximately 9.1 million raw observations.

The original dataset includes fields relating to:

- Floor-space identifiers
- Occupancy classifications
- Observation timestamps
- Occupancy state change timestamps
- Previous occupancy classifications
- Headcount information

### Data Preprocessing

The preprocessing workflow:

1. Loads the compressed CSV dataset in chunks to manage memory usage.
2. Converts observation timestamps to timezone-aware UTC values.
3. Removes records missing required forecasting information.
4. Assigns observations to five-minute intervals.
5. Retains the latest observation for each floor-space and interval.
6. Constructs forecasting samples using consecutive five-minute observations.

After preprocessing, the dataset contains approximately **482,538 valid aggregated intervals**.

Only consecutive five-minute observation pairs are retained for the forecasting task. Observations separated by longer gaps are excluded rather than treated as consecutive forecasts.

## Forecasting Methodology

### Input Features

The Random Forest model uses six input features:

| Feature | Description |
|---|---|
| `floorspaceid` | Identifies the floor-space being forecast |
| `current_state` | Occupancy state at the forecast origin |
| `hour` | Hour of the day |
| `day_of_week` | Day of the week |
| `month` | Month of the year |
| `is_weekend` | Indicates whether the observation falls on a weekend |

The target variable is the recorded occupancy state at the next consecutive five-minute interval.

### Random Forest Classifier

A Random Forest classifier is implemented using Scikit-learn.

| Parameter | Value |
|---|---:|
| Number of trees | 100 |
| Maximum tree depth | 12 |
| Minimum samples per leaf | 10 |
| Class weighting | `balanced_subsample` |
| Random state | 42 |
| Parallel processing | All available processors |

Categorical features are one-hot encoded, while numerical calendar features are passed through the preprocessing pipeline.

### Baseline Models

Two baseline approaches are used to assess whether the Random Forest provides an improvement over simpler forecasting strategies.

**1. Majority-class baseline**

Always predicts the most frequently occurring occupancy state in the training dataset.

**2. Persistence baseline**

Assumes that the next occupancy state will remain unchanged from the current recorded occupancy state.

## Model Evaluation and Validation

A chronological training and testing split is used to evaluate forecasting performance.

The dataset is split using unique forecast-origin timestamps rather than randomly assigning individual observations. Training samples are further restricted so that their target timestamps occur before the beginning of the testing forecast period.

This prevents future target information from crossing the training/testing boundary.

The main evaluation uses:

- **Accuracy:** Proportion of all predictions that are correct.
- **Balanced accuracy:** Average recall across occupancy classes.
- **Macro F1:** Unweighted average F1 score across occupancy classes.
- **Weighted F1:** Average F1 score weighted by class support.

Additional validation and analysis include:

- Confusion matrices to examine classification errors.
- Class-level F1 comparisons to investigate minority-class performance.
- Rolling chronological evaluation across multiple testing windows.
- Sensitivity analysis of Random Forest hyperparameters.
- Forecast timing analysis to investigate observation interval irregularities.
- Automated tests and forecast output audits.

The evaluation considers both overall predictive performance and the model's ability to identify less frequently occurring occupancy states.

## Project Structure

```text
MMA3001_Project/
│
├── data/
│   └── Occupancy dataset ZIP
│
├── scripts/
│   ├── 01_*.py
│   ├── 02_occupancy_forecast_visualisation.py
│   ├── 03_*.py
│   ├── 04_*.py
│   ├── 05_*.py
│   ├── 06_*.py
│   └── 07_*.py
│
├── test/
│   └── test_run.py
│
├── outputs/
│   ├── occupancy_forecast_v2/
│   ├── occupancy_forecast_visualisations/
│   ├── occupancy_rolling_evaluation/
│   └── occupancy_sensitivity/
│
├── README.md
├── LICENSE
└── requirements.txt
