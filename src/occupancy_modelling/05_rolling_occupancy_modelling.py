from pathlib import Path

"""Evaluate occupancy forecasting with expanding chronological windows.

The script aggregates sensor readings into five-minute bins, predicts the
state in the next bin, and compares a Random Forest with two simple baselines.
It writes separate evaluation outputs and does not overwrite the main forecast
pipeline's files.

Important assumptions:
    - The latest observation in each floor-space/bin represents that bin.
    - A target exists only when the exact next five-minute bin is present.
    - Features are calculated from the current and earlier observations only.
    - Windows are split by target timestamp, so training targets precede tests.
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "5occupancySensor_MayToDec2024_9MRows (1).zip"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_rolling_evaluation"

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

# Keep this list aligned with the labels used by the source dataset.
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


def load_occupancy_data(zip_path: Path) -> pd.DataFrame:
    """Read valid occupancy rows from the single CSV in the ZIP archive."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"Input ZIP file not found: {zip_path}")

    chunks = []
    rows_read = 0

    with ZipFile(zip_path) as archive:
        csv_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]
        if len(csv_files) != 1:
            raise ValueError(
                "Expected exactly one CSV in the ZIP archive, "
                f"found {len(csv_files)}."
            )

        with archive.open(csv_files[0]) as source:
            for chunk in pd.read_csv(
                source,
                usecols=["floorspaceid", TARGET, "collecteddate"],
                chunksize=CHUNK_SIZE,
                low_memory=False,
            ):
                rows_read += len(chunk)
                chunk["collecteddate"] = pd.to_datetime(
                    chunk["collecteddate"], utc=True, errors="coerce"
                )
                chunk = chunk.dropna(
                    subset=["floorspaceid", TARGET, "collecteddate"]
                )
                chunk["interval"] = chunk["collecteddate"].dt.floor("5min")
                chunks.append(chunk)
                print(f"Read {rows_read:,} raw rows...")

    if not chunks:
        raise ValueError("The source CSV contained no data rows.")

    data = pd.concat(chunks, ignore_index=True)
    del chunks

    if data.empty:
        raise ValueError("No valid occupancy rows remained after cleaning.")

    # Retain the latest source observation in each floor-space/time bin.
    data = data.sort_values(
        ["floorspaceid", "interval", "collecteddate"]
    ).drop_duplicates(
        subset=["floorspaceid", "interval"], keep="last"
    )
    return data.sort_values(
        ["floorspaceid", "interval"]
    ).reset_index(drop=True)


def build_forecast_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create features and match each row to its exact next five-minute bin.

    Returns:
        A table of eligible forecast rows and counts describing target matching.
    """
    data = data.copy()
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
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["target_interval"] = (
        data["interval"] + pd.Timedelta(minutes=FORECAST_MINUTES)
    )

    # Lookup table of observed states in each target interval.
    future = data[["floorspaceid", "interval", TARGET]].rename(
        columns={
            "interval": "target_interval",
            TARGET: "future_state",
        }
    )

    candidates = data.merge(
        future,
        on=["floorspaceid", "target_interval"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    total_rows = len(candidates)
    missing_target_rows = int((candidates["_merge"] == "left_only").sum())

    eligible = candidates.drop(columns="_merge").dropna(
        subset=["future_state", "previous_state", "log_gap_minutes"]
    ).copy()
    eligible[TARGET] = eligible["future_state"]
    eligible = eligible.sort_values(
        ["target_interval", "floorspaceid"]
    ).reset_index(drop=True)

    counts = {
        "aggregated_intervals": total_rows,
        "rows_without_exact_next_bin": missing_target_rows,
        "rows_with_exact_next_bin": total_rows - missing_target_rows,
        "eligible_forecast_rows": len(eligible),
        "rows_excluded_after_target_matching": (
            total_rows - len(eligible)
        ),
    }
    return eligible, counts


def make_model() -> Pipeline:
    """Build the preprocessing and Random Forest pipeline."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="median"))]
                ),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(**RANDOM_FOREST_SETTINGS),
            ),
        ]
    )


def create_windows(data: pd.DataFrame) -> list[dict]:
    """Create three expanding-window tests using chronological target times."""
    unique_times = np.sort(data["target_interval"].dropna().unique())
    if len(unique_times) < 100:
        raise ValueError(
            "Not enough unique target timestamps for rolling evaluation."
        )

    n_times = len(unique_times)
    cutoffs = [int(n_times * fraction) for fraction in (0.60, 0.70, 0.80, 0.90)]
    if any(cutoffs[i] >= cutoffs[i + 1] for i in range(len(cutoffs) - 1)):
        raise ValueError("Not enough timestamps to create non-empty windows.")

    return [
        {
            "window": f"Window {index + 1}",
            "train_end": unique_times[cutoffs[index] - 1],
            "test_start": unique_times[cutoffs[index]],
            "test_end": unique_times[cutoffs[index + 1] - 1],
        }
        for index in range(3)
    ]


def evaluate_windows(data: pd.DataFrame, windows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and score each chronological window against two baselines."""
    results = []
    prediction_tables = []

    for window in windows:
        name = window["window"]
        train_end = pd.Timestamp(window["train_end"])
        test_start = pd.Timestamp(window["test_start"])
        test_end = pd.Timestamp(window["test_end"])

        train = data[data["target_interval"] <= train_end].copy()
        test = data[
            (data["target_interval"] >= test_start)
            & (data["target_interval"] <= test_end)
        ].copy()

        # Defensive checks make the intended chronology explicit.
        if train.empty or test.empty:
            print(f"Skipping {name}: empty training or test set.")
            continue
        if train["target_interval"].max() >= test["target_interval"].min():
            raise ValueError(f"Chronological overlap detected in {name}.")
        if train[TARGET].nunique() < 2:
            print(f"Skipping {name}: fewer than two training classes.")
            continue

        X_train, y_train = train[FEATURES], train[TARGET]
        X_test, y_test = test[FEATURES], test[TARGET]

        majority_class = y_train.value_counts().idxmax()
        majority_predictions = np.repeat(majority_class, len(test))
        persistence_predictions = X_test["current_state"].to_numpy()

        model = make_model()
        model.fit(X_train, y_train)
        rf_predictions = model.predict(X_test)

        predictions_by_model = {
            "Majority baseline": majority_predictions,
            "Persistence baseline": persistence_predictions,
            "Random Forest": rf_predictions,
        }

        print("\n" + "-" * 70)
        print(name)
        print(f"Training targets through: {train_end}")
        print(f"Testing targets: {test_start} to {test_end}")
        print(f"Training rows: {len(train):,}")
        print(f"Testing rows:  {len(test):,}")

        for model_name, predictions in predictions_by_model.items():
            results.append({
                "window": name,
                "model": model_name,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "accuracy": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                "macro_f1": f1_score(
                    y_test, predictions, labels=STATE_ORDER,
                    average="macro", zero_division=0,
                ),
                "weighted_f1": f1_score(
                    y_test, predictions, labels=STATE_ORDER,
                    average="weighted", zero_division=0,
                ),
            })

        prediction_tables.append(pd.DataFrame({
            "window": name,
            "target_interval": test["target_interval"].to_numpy(),
            "floorspaceid": test["floorspaceid"].to_numpy(),
            "actual_state": y_test.to_numpy(),
            "majority_prediction": majority_predictions,
            "persistence_prediction": persistence_predictions,
            "random_forest_prediction": rf_predictions,
        }))

    if not results or not prediction_tables:
        raise RuntimeError("No rolling evaluation windows completed.")

    return (
        pd.DataFrame(results),
        pd.concat(prediction_tables, ignore_index=True),
    )


def main() -> None:
    """Run the rolling evaluation and save metrics, predictions, and audit counts."""
    print("=" * 70)
    print("ROLLING CHRONOLOGICAL OCCUPANCY FORECAST EVALUATION")
    print("=" * 70)

    data = load_occupancy_data(ZIP_FILE)
    print(f"\nAggregated intervals: {len(data):,}")

    forecast_rows, counts = build_forecast_rows(data)
    print("\nFORECAST ROW COUNTS")
    for label, value in counts.items():
        print(f"{label.replace('_', ' ').capitalize()}: {value:,}")

    if forecast_rows.empty:
        raise ValueError("No eligible forecasting rows were generated.")

    print(
        "\nTarget period:",
        forecast_rows["target_interval"].min(),
        "to",
        forecast_rows["target_interval"].max(),
    )

    windows = create_windows(forecast_rows)
    results, predictions = evaluate_windows(forecast_rows, windows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "rolling_evaluation_metrics.csv"
    predictions_path = OUTPUT_DIR / "rolling_evaluation_predictions.csv"
    counts_path = OUTPUT_DIR / "rolling_evaluation_row_counts.csv"

    results.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    pd.DataFrame([counts]).to_csv(counts_path, index=False)

    print("\n" + "=" * 70)
    print("ROLLING EVALUATION RESULTS")
    print("=" * 70)
    print(
        results[
            [
                "window", "model", "train_rows", "test_rows",
                "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
            ]
        ].to_string(
            index=False,
            formatters={
                metric: "{:.4f}".format
                for metric in (
                    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"
                )
            },
        )
    )
    print("\nSaved metrics to:", metrics_path)
    print("Saved predictions to:", predictions_path)
    print("Saved row counts to:", counts_path)
    print("\nRolling evaluation complete.")


if __name__ == "__main__":
    main()
