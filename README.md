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

**Dataset download:** [Download the occupancy sensor dataset (Google Drive)](https://drive.google.com/file/d/1NDrTZZYlD7QnbkZbh4w5BwsIyO8lAXIh/view?usp=sharing)

The downloaded archive must be placed in the project's `data/` directory with the following filename:

`5occupancySensor_MayToDec2024_9MRows.zip`

The dataset is not stored in this Git repository because of its size. It must be downloaded separately before running the complete modelling pipeline.

The original dataset includes fields relating to:

- Floor-space identifiers
- Occupancy classifications
- Observation timestamps
- Occupancy state change timestamps
- Previous occupancy classifications
- Headcount information

### Data Preprocessing

The preprocessing workflow:

1. Loads the compressed CSV dataset in chunks of 250,000 rows.
2. Converts observation timestamps to timezone-aware UTC values.
3. Removes records missing required forecasting information.
4. Assigns observations to five-minute intervals.
5. Retains the latest observation for each floor-space and interval.
6. Constructs forecasting samples using consecutive five-minute observations.

Chunked CSV reading limits the size of each individual read operation. However, retained chunks are subsequently combined for downstream processing, so the complete modelling workflow is not strictly memory-bounded.

After preprocessing, the dataset contains approximately **482,538 aggregated intervals**.

Only consecutive five-minute observation pairs are retained for the forecasting task. Observations separated by longer gaps are excluded rather than treated as consecutive forecasts.

## Forecasting Methodology

### Input Features

The main Random Forest model uses six input features:

| Feature | Description |
|---|---|
| `floorspaceid` | Identifies the floor-space being forecast |
| `current_state` | Occupancy state at the forecast origin |
| `hour` | Hour of the day |
| `day_of_week` | Day of the week |
| `month` | Month of the year |
| `is_weekend` | Indicates whether the observation falls on a weekend |

The target variable is the recorded occupancy state at the next consecutive five-minute interval.

The rolling evaluation and sensitivity analysis use additional temporal predictors, including the previous recorded occupancy state and an observation-gap feature. Their feature construction is described in the project report.

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
│   └── .gitkeep
│       └── 5occupancySensor_MayToDec2024_9MRows.zip  (download separately)
│
├── docs/
│   ├── index.html
│   ├── occupancy_modelling.html
│   └── occupancy_modelling/
│       └── HTML documentation for the seven analysis scripts
│
├── outputs/
│   ├── occupancy_timing_analysis/
│   ├── occupancy_forecast_v2/
│   ├── occupancy_forecast_visualisations/
│   ├── occupancy_rolling_evaluation/
│   └── occupancy_sensitivity/
│
├── src/
│   └── occupancy_modelling/
│       ├── 01_occupancy_timing_analysis.py
│       ├── 02_occupancy_forecast_visualisation.py
│       ├── 03_occupancy_forecast_validation_audit.py
│       ├── 04_actual_forecasting.py
│       ├── 05_rolling_occupancy_modelling.py
│       ├── 06_rolling_class_analysis.py
│       └── 07_occupancy_sensitivity.py
│
├── test/
│   ├── test_class_analysis.py
│   ├── test_forecasting.py
│   ├── test_pipeline.py
│   ├── test_rolling_evaluation.py
│   ├── test_run.py
│   ├── test_sensitivity.py
│   ├── test_timing_analysis.py
│   ├── test_validation.py
│   └── test_report.txt
│
├── run.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── .gitattributes
```

## Installation and Setup

The project uses **Python 3.10 or later** and the following main dependencies:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

All required packages are specified in `requirements.txt`.

### 1. Clone the repository

```powershell
git clone https://github.com/ria00011111111111/MMA3001_Project.git
cd MMA3001_Project
```

### 2. Create a virtual environment

```powershell
py -3 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Upgrade pip and install the required Python packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the dataset

Download the [occupancy sensor dataset](https://drive.google.com/file/d/1NDrTZZYlD7QnbkZbh4w5BwsIyO8lAXIh/view?usp=sharing).

Create the data directory if it does not already exist:

```powershell
New-Item -ItemType Directory -Force -Path .\data
```

Place the downloaded ZIP archive directly inside `data/` and ensure its filename is:

```text
5occupancySensor_MayToDec2024_9MRows.zip
```

**Do not extract the ZIP archive.** The modelling scripts read the compressed CSV directly.

## Running the Project

### Run the complete modelling pipeline

From the project root, execute:

```powershell
python -B .\run.py
```

The main runner executes the seven analysis scripts sequentially:

| Order | Script | Purpose |
|---:|---|---|
| 1 | `01_occupancy_timing_analysis.py` | Occupancy observation timing and state-transition analysis |
| 2 | `02_occupancy_forecast_visualisation.py` | Forecast model training, evaluation and visualisation |
| 3 | `03_occupancy_forecast_validation_audit.py` | Forecast output validation and audit |
| 4 | `04_actual_forecasting.py` | Actual elapsed-time analysis of forecasting pairs |
| 5 | `05_rolling_occupancy_modelling.py` | Rolling chronological model evaluation |
| 6 | `06_rolling_class_analysis.py` | Class-level rolling evaluation analysis |
| 7 | `07_occupancy_sensitivity.py` | Random Forest hyperparameter sensitivity analysis |

Scripts execute in the documented order. If a script fails, the pipeline stops and reports the failure rather than continuing with incomplete outputs.

The `-B` option prevents Python from writing bytecode cache files.

### Run the automated tests

To execute the complete test suite:

```powershell
python -B .\test\test_run.py
```

The test runner discovers the test modules in `test/`, executes the suite and writes a report to:

```text
test/test_report.txt
```

The command returns a non-zero exit status if the test suite fails.

## Outputs

The repository includes generated outputs from the project run so that the results can be inspected without rerunning the full pipeline. Running the scripts regenerates the corresponding output files in `outputs/`.

| Directory | Contents |
|---|---|
| `occupancy_timing_analysis/` | Observation timing, state transitions and aggregated occupancy intervals |
| `occupancy_forecast_v2/` | Forecast predictions, model comparisons and validation audit results |
| `occupancy_forecast_visualisations/` | Forecast figures, feature importance and visualisation summaries |
| `occupancy_rolling_evaluation/` | Rolling evaluation metrics, predictions and class-level results |
| `occupancy_sensitivity/` | Hyperparameter sensitivity results and validation predictions |

## Documentation

HTML documentation for the seven analysis scripts is available in the `docs/` directory.

Open `docs/index.html` in a web browser to access the project documentation.
"""

out = "/mnt/data/README_MMA3001_updated.md"
pypandoc.convert_text(
    readme,
    "md",
    format="md",
    outputfile=out,
    extra_args=["--standalone"]
)
print(f"Created: {out}\nSize: {Path(out).stat().st_size:,} bytes")
