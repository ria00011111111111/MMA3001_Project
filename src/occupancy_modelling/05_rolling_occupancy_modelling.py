"""Evaluate occupancy forecasting using expanding chronological windows.

The script aggregates occupancy sensor readings into five-minute intervals,
constructs features using current and previous observations, and predicts
the occupancy state in the exact next five-minute interval.

A Random Forest classifier is evaluated against two simple baselines:
a majority-class baseline and a persistence baseline. Evaluation uses three
expanding chronological windows, with training targets preceding the
corresponding test period.

The script saves evaluation metrics, individual predictions, and forecast
row counts to a dedicated output directory. It does not overwrite the main
forecast pipeline's outputs.

Important assumptions:
    - The latest observation in each floor-space/time bin represents that bin.
    - A target exists only when the exact next five-minute bin is present.
    - Features are calculated using current and earlier observations only.
    - Windows are split by target timestamp, ensuring training targets precede
      test targets.
    - The persistence baseline predicts that the current state will continue
      into the next interval.
    - The majority baseline always predicts the most common training class.

Outputs:
    - rolling_evaluation_metrics.csv:
        Evaluation metrics for each model and chronological window.
    - rolling_evaluation_predictions.csv:
        Actual and predicted occupancy states for each evaluated row.
    - rolling_evaluation_row_counts.csv:
        Counts describing target matching and forecast-row eligibility.

Notes:
    This script evaluates recorded occupancy sensor states rather than
    independently verified physical occupancy. A nominal five-minute target
    interval does not necessarily represent exactly five minutes of elapsed
    time between the original sensor observations.
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
    """Load and aggregate valid occupancy observations from a ZIP archive.

    Reads the source CSV in chunks to limit the amount of data loaded at once.
    Invalid rows with missing floor-space identifiers, occupancy states, or
    timestamps are removed. Timestamps are converted to UTC and assigned to
    five-minute intervals.

    When multiple observations exist for the same floor space and interval,
    only the latest source observation is retained.

    Args:
        zip_path: Path to the ZIP archive containing the occupancy CSV.

    Returns:
        A DataFrame containing one observation per floor space and
        five-minute interval, sorted chronologically within each floor space.
        The DataFrame includes the original observation timestamp and the
        corresponding interval timestamp.

    Raises:
        FileNotFoundError: If the specified ZIP archive does not exist.
        ValueError: If the archive does not contain exactly one CSV file,
            contains no data rows, or contains no valid observations.

    Note:
        The interval timestamp is calculated by flooring the original
        observation timestamp to a five-minute boundary. It is not a
        replacement for the original sensor timestamp.
    """
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
    """Construct forecasting features and match exact next-interval targets.

    Each aggregated observation becomes a candidate forecast origin. Features
    are constructed from the current observation, the previous observation
    for the same floor space, and the timestamp of the current interval.

    The target is obtained by matching the current row to an observation
    belonging to the exact next five-minute interval for the same floor
    space. Rows without a matching target, previous state, or valid
    observation-gap feature are excluded from the eligible dataset.

    Args:
        data: Aggregated occupancy observations containing floor-space IDs,
            occupancy states, original timestamps, and interval timestamps.

    Returns:
        A tuple containing:

        - A DataFrame of eligible forecast rows, including the engineered
          features and the matched future occupancy state in ``TARGET``.
        - A dictionary of row counts describing the target-matching and
          eligibility process.

    Note:
        The target is matched using the interval timestamp, not by selecting
        the next available observation. Consequently, an observation is
        excluded if the exact next five-minute bin is absent.

        The observation-gap feature uses the original ``collecteddate``
        timestamps. It is therefore possible for the elapsed time between
        sensor observations to differ from five minutes, even when their
        interval timestamps are consecutive.
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
    """Create the preprocessing and Random Forest classification pipeline.

    Categorical features are imputed using their most frequent values and
    one-hot encoded. Numeric features are imputed using their median values.
    The transformed features are then passed to a Random Forest classifier.

    Returns:
        A scikit-learn Pipeline containing the preprocessing transformer
        and configured Random Forest classifier.

    Note:
        The one-hot encoder ignores previously unseen categories during
        prediction. Imputation and encoding are fitted as part of the
        pipeline, so their parameters are learned from the training data
        when ``fit`` is called.
    """
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
    """Generate three expanding chronological evaluation windows.

    Unique target timestamps are sorted chronologically and divided using
    cutoffs at 60%, 70%, 80%, and 90% of the available timestamps.

    Each window uses all eligible observations up to the preceding cutoff
    for training, followed by the next chronological segment for testing.
    This produces three expanding training sets and three non-overlapping
    test periods.

    Args:
        data: Eligible forecast rows containing a ``target_interval``
            column.

    Returns:
        A list of three dictionaries. Each dictionary contains the window
        name and its training-end, test-start, and test-end timestamps.

    Raises:
        ValueError: If fewer than 100 unique target timestamps are available
            or if the cutoffs cannot produce non-empty chronological windows.

    Note:
        Approximately the first 60% of unique target timestamps are used
        for the first training set. The three test periods cover approximately
        60–70%, 70–80%, and 80–90% of the timestamps. The final 10% is not
        included in these evaluation windows.
    """
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


def evaluate_windows(
    data: pd.DataFrame,
    windows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and evaluate forecasting models across chronological windows.

    For each window, the function selects training and test rows according
    to their target timestamps. It then evaluates three prediction
    approaches:

    - Majority baseline: predicts the most common training target class.
    - Persistence baseline: predicts that the current state continues
      into the target interval.
    - Random Forest: fits the configured classification pipeline using
      the training data and predicts the test data.

    Accuracy, balanced accuracy, macro F1, and weighted F1 are calculated
    for each model and window. Individual predictions are also retained
    for later inspection.

    Args:
        data: Eligible forecast rows containing the engineered features,
            target labels, floor-space IDs, and target timestamps.
        windows: Chronological window definitions generated by
            ``create_windows``.

    Returns:
        A tuple containing:

        - A DataFrame of evaluation metrics, with one row per model
          and completed window.
        - A DataFrame containing the actual and predicted states for
          every evaluated test row.

    Raises:
        ValueError: If chronological overlap is detected between a
            training and test set.
        RuntimeError: If no evaluation windows complete successfully.

    Note:
        Windows with empty training or test sets, or fewer than two
        training classes, are skipped. Training data expands between
        windows, while the test periods remain separate.

        The F1 metrics use ``STATE_ORDER`` as the fixed set of class
        labels. This ensures that the reported macro and weighted F1
        scores use a consistent class ordering across windows.
    """
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
    """Run the rolling evaluation and save all generated outputs.

    Loads and aggregates the occupancy sensor data, constructs eligible
    forecasting rows, creates chronological evaluation windows, and
    evaluates the majority baseline, persistence baseline, and Random
    Forest classifier.

    The function saves the evaluation metrics, individual predictions,
    and row-count summary to ``OUTPUT_DIR`` before printing a summary
    of the completed evaluation.

    Raises:
        FileNotFoundError: If the configured input ZIP file is missing.
        ValueError: If the input data is invalid, no eligible forecast
            rows are generated, or chronological windows cannot be created.
        RuntimeError: If no evaluation windows complete successfully.
    """
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