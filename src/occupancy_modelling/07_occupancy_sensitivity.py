"""Evaluate the sensitivity of occupancy forecasting to Random Forest settings.

The script investigates how different Random Forest hyperparameter
configurations affect occupancy forecasting performance.

Sensor observations are aggregated into five-minute intervals and matched
to observations in the exact next five-minute interval to construct
forecasting pairs. Features are generated using current and previous
observations, and the resulting dataset is split chronologically into
training and validation periods.

The evaluation compares:
    - A majority-class baseline.
    - A persistence baseline.
    - Nine Random Forest configurations using combinations of maximum
      tree depth and minimum samples per leaf.

The Random Forest configurations are evaluated using a fixed validation
period. Performance is measured using accuracy, balanced accuracy,
macro F1, weighted F1, and per-class precision, recall, and F1-score.

Outputs:
    outputs/occupancy_sensitivity/sensitivity_summary.csv:
        Overall evaluation metrics for each model configuration.

    outputs/occupancy_sensitivity/sensitivity_class_metrics.csv:
        Per-class evaluation metrics for each model configuration.

    outputs/occupancy_sensitivity/sensitivity_validation_predictions.csv:
        Actual and predicted occupancy states for validation observations.

    outputs/occupancy_sensitivity/sensitivity_report.txt:
        Text summary of the evaluation, including the best-performing
        Random Forest configuration according to validation macro F1.

Important assumptions:
    - The latest observation in each floor-space/five-minute bin represents
      that interval.
    - A forecasting target exists only when the exact next five-minute
      interval is present.
    - Features are calculated from current and previous observations only.
    - Training and validation sets are separated chronologically by
      target timestamp.
    - The original test period is excluded from both training and validation.
    - Model selection is based on validation performance, not independent
      test performance.

Notes:
    The best Random Forest configuration is selected using validation
    macro F1. Its validation score is not an independent final test result.

    The script evaluates recorded occupancy sensor states rather than
    independently verified physical occupancy.
"""

from pathlib import Path
from zipfile import ZipFile
import sys
sys.dont_write_bytecode = True
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


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "5occupancySensor_MayToDec2024_9MRows.zip"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_sensitivity"

CHUNK_SIZE = 250_000
BIN_SIZE = "5min"

# Keep the original test period separate from model selection.
TEST_PERIOD_START = pd.Timestamp(
    "2025-10-21 04:00:00",
    tz="UTC",
)

DEPTH_VALUES = [8, 12, 16]
LEAF_VALUES = [5, 10, 20]

RANDOM_STATE = 42
N_ESTIMATORS = 100
VALIDATION_FRACTION = 0.20

STATE_ORDER = [
    "CurrentlyOccupied",
    "NotOccupied",
    "RecentlyOccupied",
]

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

MODEL_COLUMNS = [
    "floorspaceid",
    "interval",
    "target_interval",
    "current_state",
    "target_state",
]


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def load_occupancy_data(zip_path: Path) -> pd.DataFrame:
    """Load and aggregate occupancy sensor records from a ZIP archive.

    Reads the occupancy CSV in chunks to limit memory usage. Each chunk
    is cleaned by removing records with missing floor-space identifiers,
    occupancy states, or timestamps. Timestamps are converted to UTC
    and assigned to five-minute intervals.

    Within each chunk, duplicate floor-space/interval records are resolved
    by retaining the latest observation. A second deduplication step
    resolves duplicates that occur across different chunks.

    Args:
        zip_path: Path to the ZIP archive containing the occupancy
            sensor CSV file.

    Returns:
        A DataFrame containing one observation per floor space and
        five-minute interval. The output includes the original observation
        timestamp, interval timestamp, occupancy state, and source-order
        index. Records are sorted by floor space and interval.

    Raises:
        ValueError: If the ZIP archive contains no CSV files, if required
            columns are missing, or if no usable occupancy records remain.

    Note:
        If multiple CSV files are present, the first CSV listed in the
        archive is used. The source-order index provides a consistent
        way to resolve records with identical timestamps.
    """
    required_columns = {
        "floorspaceid",
        "occupancystatus",
        "collecteddate",
    }

    aggregated_chunks = []
    rows_read = 0

    with ZipFile(zip_path) as archive:
        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError("No CSV file was found inside the ZIP archive.")

        if len(csv_files) > 1:
            print(
                f"Found {len(csv_files)} CSV files. "
                f"Using the first: {csv_files[0]}"
            )

        with archive.open(csv_files[0]) as csv_file:
            for chunk_number, chunk in enumerate(
                pd.read_csv(
                    csv_file,
                    usecols=lambda column: column in required_columns,
                    chunksize=CHUNK_SIZE,
                    low_memory=False,
                ),
                start=1,
            ):
                missing = required_columns - set(chunk.columns)
                if missing:
                    raise ValueError(
                        f"Missing required columns: {sorted(missing)}"
                    )

                # Preserve source order to resolve equal timestamps
                # consistently across chunks.
                chunk["_source_order"] = np.arange(
                    rows_read,
                    rows_read + len(chunk),
                )
                rows_read += len(chunk)

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
                ).copy()

                chunk["interval"] = chunk["collecteddate"].dt.floor(
                    BIN_SIZE
                )

                chunk = chunk.sort_values(
                    ["collecteddate", "_source_order"]
                )

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
                            "_source_order",
                        ]
                    ]
                )

                if chunk_number % 5 == 0:
                    print(f"  Read {rows_read:,} raw rows...")

    print(f"Raw rows read: {rows_read:,}")

    if not aggregated_chunks:
        raise ValueError("No usable occupancy records were found.")

    intervals = pd.concat(aggregated_chunks, ignore_index=True)
    del aggregated_chunks

    # Resolve duplicate bins that occur across different chunks.
    intervals = intervals.sort_values(
        ["collecteddate", "_source_order"]
    )

    intervals = intervals.drop_duplicates(
        subset=["floorspaceid", "interval"],
        keep="last",
    )

    intervals = intervals.sort_values(
        ["floorspaceid", "interval"]
    ).reset_index(drop=True)

    print(f"Aggregated intervals: {len(intervals):,}")
    print(
        f"Coverage: {intervals['interval'].min()} to "
        f"{intervals['interval'].max()}"
    )

    return intervals


def build_forecast_pairs(intervals: pd.DataFrame) -> pd.DataFrame:
    """Construct forecasting features and match exact next-interval targets.

    For each floor space, the function identifies the previous occupancy
    state and previous interval. It then calculates the observation gap
    and derives calendar-based features from the current interval.

    Each observation is matched to the occupancy state for the exact
    next five-minute interval. Rows without a matching target or valid
    feature values are excluded.

    The original test period is also excluded so that it remains separate
    from model training and validation.

    Args:
        intervals: Aggregated occupancy observations containing floor-space
            identifiers, occupancy states, original timestamps, and
            interval timestamps.

    Returns:
        A DataFrame of eligible forecasting pairs containing the engineered
        features, current occupancy state, target occupancy state, and
        target interval.

    Raises:
        ValueError: If no eligible forecasting pairs remain before the
            test cutoff or if unexpected occupancy labels are encountered.

    Note:
        Target matching is performed using the exact next five-minute
        interval, rather than the next available observation.

        The observation-gap feature is calculated using interval timestamps,
        not the original sensor collection timestamps.
    """
    intervals = intervals.copy()

    grouped = intervals.groupby("floorspaceid")

    intervals["previous_state"] = grouped["occupancystatus"].shift(1)
    intervals["previous_interval"] = grouped["interval"].shift(1)

    intervals["gap_minutes"] = (
        intervals["interval"] - intervals["previous_interval"]
    ).dt.total_seconds() / 60.0

    intervals["log_gap_minutes"] = np.log1p(
        intervals["gap_minutes"].clip(lower=0)
    )

    intervals["hour"] = intervals["interval"].dt.hour
    intervals["day_of_week"] = intervals["interval"].dt.dayofweek
    intervals["month"] = intervals["interval"].dt.month
    intervals["is_weekend"] = (
        intervals["day_of_week"] >= 5
    ).astype(int)

    targets = intervals[
        ["floorspaceid", "interval", "occupancystatus"]
    ].rename(
        columns={
            "interval": "target_interval",
            "occupancystatus": "target_state",
        }
    )

    # An observation is paired only with a target exactly five minutes later.
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
        columns={"occupancystatus": "current_state"}
    )

    data = data.dropna(
        subset=FEATURES + ["target_state", "target_interval"]
    ).copy()

    # Exclude the original test period from both training and validation.
    data = data.loc[
        data["target_interval"] < TEST_PERIOD_START
    ].copy()

    data = data.sort_values("target_interval").reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "No eligible forecast pairs remain before the test cutoff."
        )

    unknown_states = set(data["target_state"].unique()) - set(STATE_ORDER)
    if unknown_states:
        raise ValueError(
            f"Unexpected occupancy labels: {sorted(unknown_states)}. "
            "Update STATE_ORDER to match the dataset."
        )

    print(f"Eligible historical forecast pairs: {len(data):,}")
    print(
        f"Target period: {data['target_interval'].min()} to "
        f"{data['target_interval'].max()}"
    )

    return data


def chronological_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split forecasting pairs into chronological training and validation sets.

    Unique target timestamps are sorted and divided according to
    ``VALIDATION_FRACTION``. All rows with target timestamps before the
    validation start are assigned to training. Rows at or after the
    validation start are assigned to validation.

    Splitting by target timestamp ensures that observations belonging
    to the same target time are not divided between the two sets.

    Args:
        data: Eligible forecasting pairs containing a ``target_interval``
            column.

    Returns:
        A tuple containing:

        - The chronological training DataFrame.
        - The subsequent validation DataFrame.

    Raises:
        ValueError: If there are too few unique target timestamps or if
            either resulting dataset is empty or chronological overlap
            is detected.

    Note:
        The original test cutoff is reported but is not used to construct
        the validation split. The cutoff has already been applied during
        forecast-pair construction.
    """
    unique_times = np.sort(data["target_interval"].unique())

    if len(unique_times) < 10:
        raise ValueError(
            "Not enough unique target timestamps to create a reliable split."
        )

    split_index = int(
        len(unique_times) * (1 - VALIDATION_FRACTION)
    )
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

    if train["target_interval"].max() >= validation["target_interval"].min():
        raise ValueError("Training and validation target times overlap.")

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Training ends at: {train['target_interval'].max()}")
    print(f"Validation starts at: {validation['target_interval'].min()}")
    print(f"Validation ends at: {validation['target_interval'].max()}")
    print(f"Original test cutoff: {TEST_PERIOD_START}")

    return train, validation


# ---------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------

def evaluate_predictions(
    model_name: str,
    predictions: np.ndarray,
    y_val: pd.Series,
    depth: int | None,
    leaf_size: int | None,
    train_rows: int,
    validation_rows: int,
) -> tuple[dict, list[dict]]:
    """Calculate overall and per-class metrics for a set of predictions.

    Computes accuracy, balanced accuracy, macro F1, and weighted F1
    for the supplied predictions. A classification report is also
    generated to obtain precision, recall, F1-score, and support
    for each occupancy class.

    Args:
        model_name: Descriptive name of the model or baseline.
        predictions: Predicted occupancy states for the validation rows.
        y_val: Actual occupancy states for the validation rows.
        depth: Maximum tree depth for a Random Forest configuration.
            Use ``None`` for baseline models.
        leaf_size: Minimum samples per leaf for a Random Forest
            configuration. Use ``None`` for baseline models.
        train_rows: Number of observations used for model training.
        validation_rows: Number of observations used for validation.

    Returns:
        A tuple containing:

        - A dictionary of overall evaluation metrics and model settings.
        - A list of dictionaries containing per-class metrics.

    Note:
        The fixed ``STATE_ORDER`` ensures that F1 metrics are calculated
        consistently across model configurations. The
        ``zero_division=0`` setting assigns zero when a metric would
        otherwise be undefined.
    """
    summary_row = {
        "model": model_name,
        "max_depth": depth,
        "min_samples_leaf": leaf_size,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "accuracy": accuracy_score(y_val, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_val, predictions),
        "macro_f1": f1_score(
            y_val,
            predictions,
            labels=STATE_ORDER,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_val,
            predictions,
            labels=STATE_ORDER,
            average="weighted",
            zero_division=0,
        ),
    }

    report = classification_report(
        y_val,
        predictions,
        labels=STATE_ORDER,
        output_dict=True,
        zero_division=0,
    )

    class_rows = []

    for label in STATE_ORDER:
        metrics = report[label]

        class_rows.append({
            "model": model_name,
            "max_depth": depth,
            "min_samples_leaf": leaf_size,
            "class": label,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1-score"],
            "support": int(metrics["support"]),
        })

    print(
        f"  Accuracy={summary_row['accuracy']:.4f} | "
        f"Balanced accuracy={summary_row['balanced_accuracy']:.4f} | "
        f"Macro F1={summary_row['macro_f1']:.4f} | "
        f"Weighted F1={summary_row['weighted_f1']:.4f}"
    )

    return summary_row, class_rows


def build_model(depth: int, leaf_size: int) -> Pipeline:
    """Create a preprocessing and Random Forest classification pipeline.

    Categorical features are one-hot encoded, with previously unseen
    categories ignored during transformation. Numeric features are passed
    directly to the classifier without additional transformation.

    The Random Forest is configured using the supplied maximum tree depth
    and minimum samples per leaf, alongside the shared estimator count,
    class weighting, and random state.

    Args:
        depth: Maximum depth of each decision tree in the Random Forest.
        leaf_size: Minimum number of training samples required at a leaf
            node.

    Returns:
        A scikit-learn Pipeline containing the feature preprocessor and
        configured Random Forest classifier.

    Note:
        The preprocessing transformer is fitted as part of the pipeline,
        ensuring that categorical encoding is learned from the training
        data when ``fit`` is called.
    """
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

    return Pipeline(
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


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

def save_results(
    summary_rows: list[dict],
    class_rows: list[dict],
    predictions_by_model: dict[str, np.ndarray],
    validation: pd.DataFrame,
) -> None:
    """Save sensitivity metrics, validation predictions, and a text report.

    Converts the overall and per-class metric records into DataFrames,
    sorts the overall results by macro F1 and balanced accuracy, and
    writes the resulting CSV files to ``OUTPUT_DIR``.

    Validation predictions from each model are added to a copy of the
    validation data before being saved. A text report summarises the
    evaluation period, model metrics, and best Random Forest configuration.

    Args:
        summary_rows: Overall evaluation metric dictionaries for all
            baselines and Random Forest configurations.
        class_rows: Per-class metric dictionaries for all evaluated models.
        predictions_by_model: Dictionary mapping model names to their
            validation predictions.
        validation: Validation DataFrame containing the original
            identifiers, timestamps, current states, and target states.

    Returns:
        None. Results are written to CSV and text files.

    Raises:
        ValueError: If no Random Forest results are available for selecting
            the best configuration.

    Note:
        The best Random Forest configuration is selected using validation
        macro F1. This is a model-selection result and should not be
        interpreted as an independent final test result.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["macro_f1", "balanced_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    class_metrics = pd.DataFrame(class_rows)

    summary_path = OUTPUT_DIR / "sensitivity_summary.csv"
    class_path = OUTPUT_DIR / "sensitivity_class_metrics.csv"
    predictions_path = (
        OUTPUT_DIR / "sensitivity_validation_predictions.csv"
    )
    report_path = OUTPUT_DIR / "sensitivity_report.txt"

    summary.to_csv(summary_path, index=False)
    class_metrics.to_csv(class_path, index=False)

    prediction_output = validation[MODEL_COLUMNS].copy()

    for model_name, predictions in predictions_by_model.items():
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("=", "")
            .replace(",", "")
        )
        prediction_output[f"pred_{safe_name}"] = predictions

    prediction_output.to_csv(predictions_path, index=False)

    random_forest_results = summary[
        summary["model"].str.startswith("Random Forest")
    ]

    if random_forest_results.empty:
        raise ValueError("No Random Forest results were generated.")

    best = random_forest_results.iloc[0]

    with report_path.open("w", encoding="utf-8") as report_file:
        report_file.write("OCCUPANCY FORECAST SENSITIVITY ANALYSIS\n")
        report_file.write("=" * 60 + "\n\n")
        report_file.write(
            "Validation targets are strictly before the original test cutoff.\n"
        )
        report_file.write(f"Original test cutoff: {TEST_PERIOD_START}\n")
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

        report_file.write("BEST RANDOM FOREST CONFIGURATION\n")
        report_file.write("-" * 60 + "\n")
        report_file.write(f"Model: {best['model']}\n")
        report_file.write(f"Macro F1: {best['macro_f1']:.4f}\n")
        report_file.write(
            f"Balanced accuracy: {best['balanced_accuracy']:.4f}\n"
        )
        report_file.write(f"Accuracy: {best['accuracy']:.4f}\n")
        report_file.write(f"Weighted F1: {best['weighted_f1']:.4f}\n\n")
        report_file.write(
            "The best configuration is selected using validation results. "
            "Its validation score is not an independent final test result. "
            "Inspect class-level metrics, especially RecentlyOccupied, "
            "before drawing conclusions."
        )

    print("\nSensitivity analysis complete.")
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


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    """Run the occupancy forecasting sensitivity analysis.

    Loads and aggregates the occupancy sensor data, constructs eligible
    forecasting pairs, and creates a chronological training/validation
    split.

    The function first evaluates the majority and persistence baselines.
    It then trains and evaluates each Random Forest configuration defined
    by ``DEPTH_VALUES`` and ``LEAF_VALUES``.

    Finally, it saves the overall metrics, per-class metrics, validation
    predictions, and text report.

    Raises:
        FileNotFoundError: If the configured input ZIP archive does not
            exist.
        ValueError: If the input data cannot be processed, no eligible
            forecast pairs remain, an invalid occupancy state is found,
            or the chronological split cannot be created.
    """
    print("=" * 70)
    print("OCCUPANCY FORECAST SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Input ZIP: {ZIP_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"Input ZIP not found: {ZIP_PATH}")

    print("\n[1/4] Loading and aggregating sensor data...")
    intervals = load_occupancy_data(ZIP_PATH)

    print("\n[2/4] Building forecasting pairs and chronological split...")
    data = build_forecast_pairs(intervals)
    train, validation = chronological_split(data)

    X_train = train[FEATURES]
    y_train = train["target_state"]
    X_val = validation[FEATURES]
    y_val = validation["target_state"]

    summary_rows = []
    class_rows = []
    predictions_by_model = {}

    print("\n[3/4] Evaluating baselines...")

    majority_model = DummyClassifier(strategy="most_frequent")
    majority_model.fit(X_train, y_train)

    baseline_predictions = {
        "Majority baseline": majority_model.predict(X_val),
        "Persistence baseline": X_val["current_state"].to_numpy(),
    }

    for model_name, predictions in baseline_predictions.items():
        predictions_by_model[model_name] = predictions

        summary_row, per_class_rows = evaluate_predictions(
            model_name,
            predictions,
            y_val,
            depth=None,
            leaf_size=None,
            train_rows=len(train),
            validation_rows=len(validation),
        )

        summary_rows.append(summary_row)
        class_rows.extend(per_class_rows)

    print("\n[4/4] Evaluating Random Forest configurations...")

    for depth in DEPTH_VALUES:
        for leaf_size in LEAF_VALUES:
            model_name = (
                f"Random Forest depth={depth}, leaf={leaf_size}"
            )
            print(f"\nTraining {model_name}...")

            model = build_model(depth, leaf_size)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)

            predictions_by_model[model_name] = predictions

            summary_row, per_class_rows = evaluate_predictions(
                model_name,
                predictions,
                y_val,
                depth=depth,
                leaf_size=leaf_size,
                train_rows=len(train),
                validation_rows=len(validation),
            )

            summary_rows.append(summary_row)
            class_rows.extend(per_class_rows)

    save_results(
        summary_rows,
        class_rows,
        predictions_by_model,
        validation,
    )


if __name__ == "__main__":
    main()