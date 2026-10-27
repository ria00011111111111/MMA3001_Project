
"""
06_14_occupancysensitivity.py

Sensitivity analysis for the occupancy forecasting Random Forest.

Purpose:
- Compare max_depth = 8, 12, 16
- Compare min_samples_leaf = 5, 10, 20
- Preserve chronological train/validation order
- Exclude target times on or after 2025-10-21 04:00 UTC
- Compare against majority-class and persistence baselines
- Save results separately without overwriting existing outputs

Target:
Recorded occupancy state in the next five-minute bin.
This is not necessarily exactly five minutes after the current
sensor observation.
"""

from pathlib import Path
from zipfile import ZipFile
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# 1. Paths and settings
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "5occupancySensor_MayToDec2024_9MRows (1).zip"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 250_000
BIN_SIZE = "5min"

# Do not use the later period that was used for the original test.
TEST_PERIOD_START = pd.Timestamp("2025-10-21 04:00:00", tz="UTC")

DEPTH_VALUES = [8, 12, 16]
LEAF_VALUES = [5, 10, 20]

RANDOM_STATE = 42
N_ESTIMATORS = 100

FEATURES = [
    "floorspaceid",
    "current_state",
    "previous_state",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "log_gap_minutes",
]

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

print("=" * 70)
print("OCCUPANCY FORECAST SENSITIVITY ANALYSIS")
print("=" * 70)
print(f"Input ZIP: {ZIP_PATH}")
print(f"Output directory: {OUTPUT_DIR}")

if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"Could not find the input ZIP file:\n{ZIP_PATH}\n"
        "Check ZIP_PATH near the top of this script."
    )


# ------------------------------------------------------------
# 2. Read and aggregate the occupancy data
# ------------------------------------------------------------

print("\n[1/6] Reading occupancy data in chunks...")

required_columns = [
    "floorspaceid",
    "occupancystatus",
    "collecteddate",
]

aggregated_chunks = []
rows_read = 0

with ZipFile(ZIP_PATH) as archive:
    csv_files = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    ]

    if not csv_files:
        raise ValueError("No CSV file was found inside the ZIP archive.")

    if len(csv_files) > 1:
        print(
            f"Found {len(csv_files)} CSV files. "
            f"Using the first: {csv_files[0]}"
        )

    csv_name = csv_files[0]

    with archive.open(csv_name) as csv_file:
        for chunk_number, chunk in enumerate(
            pd.read_csv(
                csv_file,
                usecols=lambda column: column in required_columns,
                chunksize=CHUNK_SIZE,
                low_memory=False,
            ),
            start=1,
        ):
            rows_read += len(chunk)

            missing_columns = set(required_columns) - set(chunk.columns)
            if missing_columns:
                raise ValueError(
                    f"Required columns are missing: {missing_columns}"
                )

            chunk["collecteddate"] = pd.to_datetime(
                chunk["collecteddate"],
                errors="coerce",
                utc=True,
            )

            chunk = chunk.dropna(
                subset=[
                    "floorspaceid",
                    "occupancystatus",
                    "collecteddate",
                ]
            )

            chunk["interval"] = chunk["collecteddate"].dt.floor(BIN_SIZE)

            # Keep the last recorded observation within each
            # floor-space and five-minute interval.
            chunk = chunk.sort_values("collecteddate")

            chunk = chunk.drop_duplicates(
                subset=["floorspaceid", "interval"],
                keep="last",
            )

            aggregated_chunks.append(
                chunk[
                    [
                        "floorspaceid",
                        "occupancystatus",
                        "collecteddate",
                        "interval",
                    ]
                ]
            )

            if chunk_number % 5 == 0:
                print(
                    f"  Read {rows_read:,} raw rows..."
                )

print(f"Raw rows read: {rows_read:,}")

# Different chunks can contain records for the same interval.
# Recombine and apply the same last-observation rule globally.
print("\nAggregating intervals across chunks...")

intervals = pd.concat(aggregated_chunks, ignore_index=True)
del aggregated_chunks

intervals = intervals.sort_values("collecteddate")

intervals = intervals.drop_duplicates(
    subset=["floorspaceid", "interval"],
    keep="last",
)

intervals = intervals.sort_values(
    ["floorspaceid", "interval"]
).reset_index(drop=True)

print(f"Aggregated intervals: {len(intervals):,}")
print(
    "Coverage: "
    f"{intervals['interval'].min()} to "
    f"{intervals['interval'].max()}"
)


# ------------------------------------------------------------
# 3. Construct forecasting features and target
# ------------------------------------------------------------

print("\n[2/6] Building features and next-bin targets...")

intervals["previous_state"] = (
    intervals.groupby("floorspaceid")["occupancystatus"].shift(1)
)

intervals["previous_interval"] = (
    intervals.groupby("floorspaceid")["interval"].shift(1)
)

intervals["gap_minutes"] = (
    intervals["interval"] - intervals["previous_interval"]
).dt.total_seconds() / 60.0

# Match the original model's log-gap feature.
intervals["log_gap_minutes"] = np.log1p(
    intervals["gap_minutes"].clip(lower=0)
)

intervals["hour"] = intervals["interval"].dt.hour
intervals["day_of_week"] = intervals["interval"].dt.dayofweek
intervals["month"] = intervals["interval"].dt.month
intervals["is_weekend"] = (
    intervals["day_of_week"] >= 5
).astype(int)

# A target at the next five-minute interval label.
targets = intervals[
    ["floorspaceid", "interval", "occupancystatus"]
].copy()

targets = targets.rename(
    columns={
        "interval": "target_interval",
        "occupancystatus": "target_state",
    }
)

targets["origin_interval"] = (
    targets["target_interval"] - pd.Timedelta(minutes=5)
)

data = intervals.merge(
    targets,
    left_on=["floorspaceid", "interval"],
    right_on=["floorspaceid", "origin_interval"],
    how="inner",
    validate="one_to_one",
)

data = data.rename(
    columns={
        "occupancystatus": "current_state",
    }
)

data = data.dropna(
    subset=FEATURES + ["target_state", "target_interval"]
).copy()

# Only use the historical period before the original test begins.
data = data.loc[
    data["target_interval"] < TEST_PERIOD_START
].copy()

data = data.sort_values("target_interval").reset_index(drop=True)

if data.empty:
    raise ValueError(
        "No eligible forecast pairs remain before the cutoff. "
        "Check the data timestamps and TEST_PERIOD_START."
    )

print(f"Eligible historical forecast pairs: {len(data):,}")
print(
    "Eligible target period: "
    f"{data['target_interval'].min()} to "
    f"{data['target_interval'].max()}"
)

# Keep every floor-space row from a target interval in the same split.
unique_times = np.sort(data["target_interval"].unique())

if len(unique_times) < 10:
    raise ValueError("Not enough unique target intervals to create a split.")

split_index = int(len(unique_times) * 0.80)
split_index = min(max(split_index, 1), len(unique_times) - 1)

validation_start = pd.Timestamp(unique_times[split_index])

train = data.loc[
    data["target_interval"] < validation_start
].copy()

validation = data.loc[
    data["target_interval"] >= validation_start
].copy()

if train.empty or validation.empty:
    raise ValueError("Training or validation split is empty.")

X_train = train[FEATURES]
y_train = train["target_state"]

X_val = validation[FEATURES]
y_val = validation["target_state"]

print(f"Training rows: {len(train):,}")
print(f"Validation rows: {len(validation):,}")
print(f"Training ends before: {validation_start}")
print(
    "Validation target period: "
    f"{validation['target_interval'].min()} to "
    f"{validation['target_interval'].max()}"
)
print(f"Original test cutoff: {TEST_PERIOD_START}")


# ------------------------------------------------------------
# 4. Baselines
# ------------------------------------------------------------

print("\n[3/6] Evaluating baselines...")

baseline_predictions = {}

majority_model = DummyClassifier(strategy="most_frequent")
majority_model.fit(X_train, y_train)
baseline_predictions["Majority baseline"] = majority_model.predict(X_val)

# Persistence predicts that the state stays the same.
baseline_predictions["Persistence baseline"] = (
    X_val["current_state"].to_numpy()
)


# ------------------------------------------------------------
# 5. Evaluate the nine Random Forest configurations
# ------------------------------------------------------------

print("\n[4/6] Running the sensitivity grid...")

all_predictions = {}
summary_rows = []
class_rows = []

for model_name, predictions in baseline_predictions.items():
    all_predictions[model_name] = predictions

    summary_rows.append(
        {
            "model": model_name,
            "max_depth": np.nan,
            "min_samples_leaf": np.nan,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "accuracy": accuracy_score(y_val, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_val, predictions
            ),
            "macro_f1": f1_score(
                y_val, predictions, average="macro", zero_division=0
            ),
            "weighted_f1": f1_score(
                y_val, predictions, average="weighted", zero_division=0
            ),
        }
    )

    report = classification_report(
        y_val,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    for label in sorted(y_val.unique()):
        metrics = report.get(label, {})
        class_rows.append(
            {
                "model": model_name,
                "max_depth": np.nan,
                "min_samples_leaf": np.nan,
                "class": label,
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1_score": metrics.get("f1-score", 0.0),
                "support": metrics.get("support", 0),
            }
        )

for depth in DEPTH_VALUES:
    for leaf_size in LEAF_VALUES:
        model_name = (
            f"Random Forest depth={depth}, leaf={leaf_size}"
        )

        print(f"\nTraining {model_name}...")

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    CATEGORICAL_FEATURES,
                ),
                ("numeric", "passthrough", NUMERIC_FEATURES),
            ],
            remainder="drop",
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=N_ESTIMATORS,
                        max_depth=depth,
                        min_samples_leaf=leaf_size,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        all_predictions[model_name] = predictions

        accuracy = accuracy_score(y_val, predictions)
        balanced_accuracy = balanced_accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val, predictions, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            y_val, predictions, average="weighted", zero_division=0
        )

        summary_rows.append(
            {
                "model": model_name,
                "max_depth": depth,
                "min_samples_leaf": leaf_size,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            }
        )

        report = classification_report(
            y_val,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        for label in sorted(y_val.unique()):
            metrics = report.get(label, {})
            class_rows.append(
                {
                    "model": model_name,
                    "max_depth": depth,
                    "min_samples_leaf": leaf_size,
                    "class": label,
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1_score": metrics.get("f1-score", 0.0),
                    "support": metrics.get("support", 0),
                }
            )

        print(
            f"  Accuracy={accuracy:.4f} | "
            f"Balanced accuracy={balanced_accuracy:.4f} | "
            f"Macro F1={macro_f1:.4f} | "
            f"Weighted F1={weighted_f1:.4f}"
        )


# ------------------------------------------------------------
# 6. Save results separately
# ------------------------------------------------------------

print("\n[5/6] Saving sensitivity results...")

summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    ["macro_f1", "balanced_accuracy"],
    ascending=False,
).reset_index(drop=True)

class_metrics = pd.DataFrame(class_rows)

summary_path = OUTPUT_DIR / "sensitivity_summary.csv"
class_path = OUTPUT_DIR / "sensitivity_class_metrics.csv"
predictions_path = OUTPUT_DIR / "sensitivity_validation_predictions.csv"
report_path = OUTPUT_DIR / "sensitivity_report.txt"

summary.to_csv(summary_path, index=False)
class_metrics.to_csv(class_path, index=False)

prediction_output = validation[
    [
        "floorspaceid",
        "interval",
        "target_interval",
        "current_state",
        "target_state",
    ]
].copy()

for model_name, predictions in all_predictions.items():
    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("=", "")
        .replace(",", "")
    )
    prediction_output[f"pred_{safe_name}"] = predictions

prediction_output.to_csv(predictions_path, index=False)

# Human-readable report
with open(report_path, "w", encoding="utf-8") as report_file:
    report_file.write("OCCUPANCY FORECAST SENSITIVITY ANALYSIS\n")
    report_file.write("=" * 60 + "\n\n")
    report_file.write(
        "Validation data ends before the original test period cutoff.\n"
    )
    report_file.write(
        f"Original test cutoff: {TEST_PERIOD_START}\n"
    )
    report_file.write(
        f"Validation starts: {validation['target_interval'].min()}\n"
    )
    report_file.write(
        f"Validation ends: {validation['target_interval'].max()}\n\n"
    )

    report_file.write("MODEL SUMMARY, SORTED BY MACRO F1\n")
    report_file.write("-" * 60 + "\n")
    report_file.write(summary.to_string(index=False))
    report_file.write("\n\n")

    best = summary.loc[summary["model"].str.startswith("Random Forest")].iloc[0]
    report_file.write("BEST VALIDATION CONFIGURATION BY MACRO F1\n")
    report_file.write("-" * 60 + "\n")
    report_file.write(f"Model: {best['model']}\n")
    report_file.write(f"Macro F1: {best['macro_f1']:.4f}\n")
    report_file.write(
        f"Balanced accuracy: {best['balanced_accuracy']:.4f}\n"
    )
    report_file.write(f"Accuracy: {best['accuracy']:.4f}\n")
    report_file.write(f"Weighted F1: {best['weighted_f1']:.4f}\n\n")

    report_file.write(
        "Interpretation caution:\n"
        "The best configuration is selected using validation performance, "
        "not the original test period. RecentlyOccupied performance should "
        "be checked separately because overall metrics can hide weak "
        "minority-class predictions. A better validation score does not "
        "guarantee better performance on future data.\n"
    )

print("\n[6/6] Sensitivity analysis complete.")
print(f"Summary: {summary_path}")
print(f"Class metrics: {class_path}")
print(f"Validation predictions: {predictions_path}")
print(f"Text report: {report_path}")

print("\nTOP RESULTS")
print(
    summary[
        [
            "model",
            "max_depth",
            "min_samples_leaf",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ]
    ].head(12).to_string(index=False)
)

print("\nCheck sensitivity_class_metrics.csv for class-level results.")
