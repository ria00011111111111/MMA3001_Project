
"""Rolling chronological evaluation for occupancy forecasting.

This script does not modify the existing forecast model or its outputs.
Each test window is later in time than its corresponding training data.
"""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ------------------------------------------------------------
# 1. Paths and settings
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "5occupancySensor_MayToDec2024_9MRows (1).zip"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_rolling_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 250_000
FORECAST_MINUTES = 5

TARGET = "occupancystatus"

CATEGORICAL_FEATURES = [
    "floorspaceid",
    "current_state",
    "previous_state",
]

NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "log_gap_minutes",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

STATE_ORDER = [
    "CurrentlyOccupied",
    "NotOccupied",
    "RecentlyOccupied",
]

RANDOM_FOREST_SETTINGS = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_leaf": 10,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}


# ------------------------------------------------------------
# 2. Load and aggregate occupancy observations
# ------------------------------------------------------------

print("=" * 70)
print("ROLLING CHRONOLOGICAL OCCUPANCY FORECAST EVALUATION")
print("=" * 70)

if not ZIP_FILE.exists():
    raise FileNotFoundError(f"Input ZIP file not found: {ZIP_FILE}")

chunks = []
rows_read = 0

with ZipFile(ZIP_FILE) as archive:
    csv_files = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    ]

    if len(csv_files) != 1:
        raise ValueError(
            "Expected exactly one CSV in the ZIP archive, "
            f"found {len(csv_files)}."
        )

    with archive.open(csv_files[0]) as file:
        for chunk in pd.read_csv(
            file,
            usecols=[
                "floorspaceid",
                TARGET,
                "collecteddate",
            ],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            rows_read += len(chunk)

            chunk["collecteddate"] = pd.to_datetime(
                chunk["collecteddate"],
                utc=True,
                errors="coerce",
            )

            chunk = chunk.dropna(
                subset=[
                    "floorspaceid",
                    TARGET,
                    "collecteddate",
                ]
            )

            chunk["interval"] = chunk["collecteddate"].dt.floor("5min")
            chunks.append(chunk)

            print(f"Read {rows_read:,} raw rows...")

data = pd.concat(chunks, ignore_index=True)
del chunks

data = data.sort_values(
    ["floorspaceid", "interval", "collecteddate"]
)

data = data.drop_duplicates(
    subset=["floorspaceid", "interval"],
    keep="last",
)

data = data.sort_values(
    ["floorspaceid", "interval"]
).reset_index(drop=True)

print(f"\nAggregated intervals: {len(data):,}")


# ------------------------------------------------------------
# 3. Construct forecasting features and targets
# ------------------------------------------------------------

data["current_state"] = data[TARGET]

grouped = data.groupby("floorspaceid", sort=False)

data["previous_state"] = grouped[TARGET].shift(1)
data["previous_timestamp"] = grouped["collecteddate"].shift(1)

data["gap_minutes"] = (
    data["collecteddate"] - data["previous_timestamp"]
).dt.total_seconds() / 60

data.loc[data["gap_minutes"] <= 0, "gap_minutes"] = np.nan

data["log_gap_minutes"] = np.log1p(
    data["gap_minutes"].clip(upper=1440)
)

data["hour"] = data["interval"].dt.hour
data["day_of_week"] = data["interval"].dt.dayofweek
data["month"] = data["interval"].dt.month
data["is_weekend"] = (
    data["day_of_week"] >= 5
).astype(int)

future = data[
    ["floorspaceid", "interval", TARGET]
].copy()

future = future.rename(
    columns={
        "interval": "target_interval",
        TARGET: "future_state",
    }
)

data["target_interval"] = (
    data["interval"] + pd.Timedelta(minutes=FORECAST_MINUTES)
)

data = data.merge(
    future,
    on=["floorspaceid", "target_interval"],
    how="left",
    validate="many_to_one",
)

data = data.dropna(
    subset=[
        "future_state",
        "previous_state",
        "log_gap_minutes",
    ]
).copy()

data[TARGET] = data["future_state"]

data = data.sort_values(
    ["target_interval", "floorspaceid"]
).reset_index(drop=True)

if data.empty:
    raise ValueError("No eligible forecasting rows were generated.")

print(f"Eligible forecast pairs: {len(data):,}")
print(
    "Target period:",
    data["target_interval"].min(),
    "to",
    data["target_interval"].max(),
)


# ------------------------------------------------------------
# 4. Create chronological test windows
# ------------------------------------------------------------

# Use the same unique target timestamps to define window boundaries.
# The first 60% of times is available for the first training window.
# Then evaluate successive 10% windows: 60-70%, 70-80%, 80-90%.
# The final 10% remains unused for this rolling experiment.

unique_times = np.array(
    sorted(data["target_interval"].dropna().unique())
)

if len(unique_times) < 100:
    raise ValueError("Not enough unique timestamps for rolling evaluation.")

n_times = len(unique_times)

cutoff_60 = int(n_times * 0.60)
cutoff_70 = int(n_times * 0.70)
cutoff_80 = int(n_times * 0.80)
cutoff_90 = int(n_times * 0.90)

windows = [
    {
        "window": "Window 1",
        "train_end": unique_times[cutoff_60 - 1],
        "test_start": unique_times[cutoff_60],
        "test_end": unique_times[cutoff_70 - 1],
    },
    {
        "window": "Window 2",
        "train_end": unique_times[cutoff_70 - 1],
        "test_start": unique_times[cutoff_70],
        "test_end": unique_times[cutoff_80 - 1],
    },
    {
        "window": "Window 3",
        "train_end": unique_times[cutoff_80 - 1],
        "test_start": unique_times[cutoff_80],
        "test_end": unique_times[cutoff_90 - 1],
    },
]


# ------------------------------------------------------------
# 5. Evaluate each window
# ------------------------------------------------------------

results = []
prediction_tables = []

for window in windows:
    name = window["window"]
    train_end = pd.Timestamp(window["train_end"])
    test_start = pd.Timestamp(window["test_start"])
    test_end = pd.Timestamp(window["test_end"])

    # Training rows must have targets no later than train_end.
    train = data[
        data["target_interval"] <= train_end
    ].copy()

    # Test rows belong only to this future chronological window.
    test = data[
        (data["target_interval"] >= test_start)
        & (data["target_interval"] <= test_end)
    ].copy()

    if train.empty or test.empty:
        print(f"Skipping {name}: empty train or test set.")
        continue

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    if y_train.nunique() < 2:
        print(f"Skipping {name}: fewer than two training classes.")
        continue

    print("\n" + "-" * 70)
    print(name)
    print(f"Training targets through: {train_end}")
    print(f"Testing targets: {test_start} to {test_end}")
    print(f"Training rows: {len(train):,}")
    print(f"Testing rows:  {len(test):,}")

    # Baseline 1: majority class learned from training data.
    majority_class = y_train.value_counts().idxmax()
    majority_predictions = np.repeat(
        majority_class, len(test)
    )

    # Baseline 2: persistence, using the current recorded state.
    persistence_predictions = X_test["current_state"].to_numpy()

    # Random Forest pipeline, retaining original model settings.
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        ),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(**RANDOM_FOREST_SETTINGS),
            ),
        ]
    )

    model.fit(X_train, y_train)
    rf_predictions = model.predict(X_test)

    model_predictions = {
        "Majority baseline": majority_predictions,
        "Persistence baseline": persistence_predictions,
        "Random Forest": rf_predictions,
    }

    for model_name, predictions in model_predictions.items():
        results.append({
            "window": name,
            "model": model_name,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "macro_f1": f1_score(
                y_test,
                predictions,
                labels=STATE_ORDER,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                y_test,
                predictions,
                labels=STATE_ORDER,
                average="weighted",
                zero_division=0,
            ),
        })

    prediction_tables.append(
        pd.DataFrame({
            "window": name,
            "target_interval": test["target_interval"].to_numpy(),
            "floorspaceid": test["floorspaceid"].to_numpy(),
            "actual_state": y_test.to_numpy(),
            "majority_prediction": majority_predictions,
            "persistence_prediction": persistence_predictions,
            "random_forest_prediction": rf_predictions,
        })
    )


# ------------------------------------------------------------
# 6. Save results
# ------------------------------------------------------------

if not results:
    raise RuntimeError("No rolling evaluation windows completed.")

results_df = pd.DataFrame(results)

results_path = OUTPUT_DIR / "rolling_evaluation_metrics.csv"
results_df.to_csv(results_path, index=False)

predictions_df = pd.concat(
    prediction_tables,
    ignore_index=True,
)

predictions_path = OUTPUT_DIR / "rolling_evaluation_predictions.csv"
predictions_df.to_csv(predictions_path, index=False)

print("\n" + "=" * 70)
print("ROLLING EVALUATION RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "window",
            "model",
            "train_rows",
            "test_rows",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ]
    ].to_string(
        index=False,
        formatters={
            "accuracy": "{:.4f}".format,
            "balanced_accuracy": "{:.4f}".format,
            "macro_f1": "{:.4f}".format,
            "weighted_f1": "{:.4f}".format,
        },
    )
)

print("\nSaved metrics to:")
print(results_path)

print("\nSaved predictions to:")
print(predictions_path)

print("\nRolling evaluation complete.")
