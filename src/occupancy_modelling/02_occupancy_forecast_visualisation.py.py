"""Create diagnostic plots for the occupancy-state forecasting models.

The script reads the held-out forecast predictions and model summary CSV files
created by the forecasting pipeline. It writes confusion matrices, model
comparison plots, feature-importance plots, class-level F1 scores, a sample
time-series plot, and a text summary to ``outputs/occupancy_forecast_visualisations``.

Run this script from any working directory using the Python environment that
contains pandas, NumPy, Matplotlib, and scikit-learn. Paths are resolved relative
to this file, not the current working directory.

Important interpretation notes:
    - The target values are recorded sensor states, not independently verified
      physical occupancy.
    - Random Forest feature importance describes how the fitted model uses
      features; it does not establish causation.
    - The representative time series is a selected example, not an overall
      performance estimate.
"""

from pathlib import Path

import matplotlib

# Use a non-interactive backend so plots can be generated in scripts, terminals,
# and automated environments without opening a window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_forecast_v2"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_forecast_visualisations"

PREDICTIONS_FILE = INPUT_DIR / "test_forecast_predictions.csv"
COMPARISON_FILE = INPUT_DIR / "forecast_model_comparison.csv"
IMPORTANCE_FILE = INPUT_DIR / "forecast_feature_importance.csv"

# Keep the state order consistent across confusion matrices and time-series
# plots so the same state always occupies the same position.
STATE_ORDER = [
    "CurrentlyOccupied",
    "NotOccupied",
    "RecentlyOccupied",
]

METRIC_COLUMNS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]

PREDICTION_COLUMNS = [
    "forecast_origin",
    "forecast_target_time",
    "actual_future_state",
    "random_forest_prediction",
    "persistence_prediction",
    "floorspaceid",
]

FIGURE_FILENAMES = [
    "01_random_forest_confusion_matrix.png",
    "02_normalised_confusion_matrix.png",
    "03_model_comparison.png",
    "04_feature_importance.png",
    "05_class_level_f1_comparison.png",
    "06_actual_vs_predicted_time_series.png",
]


# ---------------------------------------------------------------------------
# 2. INPUT VALIDATION AND LOADING
# ---------------------------------------------------------------------------

def require_columns(dataframe, required_columns, file_path):
    """Raise a helpful error if a CSV is missing required columns.

    Args:
        dataframe: DataFrame loaded from the CSV.
        required_columns: Column names needed by the analysis.
        file_path: Source file, used to make the error message informative.
    """
    missing = sorted(set(required_columns) - set(dataframe.columns))
    if missing:
        raise ValueError(
            f"{file_path} is missing required column(s): {', '.join(missing)}"
        )


def load_inputs():
    """Load and validate the forecast outputs needed to create the plots.

    Returns:
        Tuple containing cleaned predictions, model comparison, and feature
        importance DataFrames.
    """
    for path in (PREDICTIONS_FILE, COMPARISON_FILE, IMPORTANCE_FILE):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required input file was not found: {path}\n"
                "Run the occupancy forecasting script before this script."
            )

    predictions = pd.read_csv(PREDICTIONS_FILE)
    comparison = pd.read_csv(COMPARISON_FILE)
    importance = pd.read_csv(IMPORTANCE_FILE)

    require_columns(predictions, PREDICTION_COLUMNS, PREDICTIONS_FILE)
    require_columns(comparison, ["model", *METRIC_COLUMNS], COMPARISON_FILE)
    require_columns(importance, ["feature", "importance"], IMPORTANCE_FILE)

    # Parse timestamps as UTC to keep comparisons and plot labels unambiguous.
    for column in ("forecast_origin", "forecast_target_time"):
        predictions[column] = pd.to_datetime(
            predictions[column], utc=True, errors="coerce"
        )

    # Invalid timestamps or missing labels cannot be evaluated meaningfully.
    predictions = predictions.dropna(
        subset=[
            "forecast_origin",
            "forecast_target_time",
            "actual_future_state",
            "random_forest_prediction",
            "persistence_prediction",
            "floorspaceid",
        ]
    ).copy()

    if predictions.empty:
        raise ValueError(
            "No valid prediction rows remain after removing missing or invalid values."
        )

    # State labels are treated as strings consistently throughout the script.
    for column in (
        "actual_future_state",
        "random_forest_prediction",
        "persistence_prediction",
    ):
        predictions[column] = predictions[column].astype(str)

    return predictions, comparison, importance


def get_state_labels(actual, *predicted_series):
    """Return the preferred state order plus any unexpected observed labels.

    Including labels seen only in predictions prevents those predictions from
    silently disappearing from a confusion matrix.
    """
    observed = set(actual.dropna().astype(str))
    for series in predicted_series:
        observed.update(series.dropna().astype(str))

    return [state for state in STATE_ORDER if state in observed] + sorted(
        observed - set(STATE_ORDER)
    )


# ---------------------------------------------------------------------------
# 3. CONFUSION MATRICES
# ---------------------------------------------------------------------------

def plot_confusion_matrices(actual, predicted, labels):
    """Save raw-count and row-normalised Random Forest confusion matrices.

    Rows represent actual classes and columns represent predicted classes.
    Row normalisation makes the proportion correctly or incorrectly classified
    within each actual class easier to compare despite class imbalance.
    """
    matrix = confusion_matrix(actual, predicted, labels=labels)

    # Raw confusion matrix: each cell is the number of examples in that pair.
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Number of predictions")
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted future state",
        ylabel="Actual future state",
        title="Random Forest Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / FIGURE_FILENAMES[0], dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    # Divide each row by its actual-class total. Empty rows remain zero rather
    # than producing divide-by-zero warnings or NaN values.
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(normalised, vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Proportion of actual class")
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted future state",
        ylabel="Actual future state",
        title="Normalised Random Forest Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    for row in range(normalised.shape[0]):
        for column in range(normalised.shape[1]):
            value = normalised[row, column]
            ax.text(
                column,
                row,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
            )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / FIGURE_FILENAMES[1], dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. MODEL COMPARISON
# ---------------------------------------------------------------------------

def plot_model_comparison(comparison):
    """Plot the saved evaluation metrics for each model."""
    if comparison.empty:
        raise ValueError(f"Model comparison file is empty: {COMPARISON_FILE}")

    comparison_long = comparison.melt(
        id_vars="model",
        value_vars=METRIC_COLUMNS,
        var_name="metric",
        value_name="score",
    )

    # A bar chart supports direct comparison of the same metric across models.
    pivoted = comparison_long.pivot(
        index="metric", columns="model", values="score"
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    pivoted.plot(kind="bar", ax=ax)
    ax.set_title("Occupancy Forecast Model Comparison")
    ax.set_xlabel("Evaluation metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / FIGURE_FILENAMES[2], dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def plot_feature_importance(importance):
    """Plot the 15 highest-importance features from the saved model output.

    Feature importance is a model diagnostic: it should not be interpreted as
    evidence that a feature causes a particular occupancy state.
    """
    if importance.empty:
        print("Skipping feature-importance plot: no rows are available.")
        return False

    # Coerce invalid values to NaN and exclude them before sorting.
    importance = importance.copy()
    importance["importance"] = pd.to_numeric(
        importance["importance"], errors="coerce"
    )
    importance = importance.dropna(subset=["feature", "importance"])

    if importance.empty:
        print("Skipping feature-importance plot: no numeric importance values.")
        return False

    top_features = (
        importance.sort_values("importance", ascending=False)
        .head(15)
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(top_features["feature"].astype(str), top_features["importance"])
    ax.set_title("Random Forest: Top 15 Feature Importances")
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / FIGURE_FILENAMES[3], dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# 6. CLASS-LEVEL F1 SCORES
# ---------------------------------------------------------------------------

def calculate_class_f1(actual, predicted, state):
    """Calculate the F1 score for one class using one-vs-rest counts.

    Args:
        actual: Actual state labels.
        predicted: Predicted state labels.
        state: Class whose F1 score is being calculated.

    Returns:
        F1 score between 0 and 1. Returns zero when precision and recall are
        both undefined because the denominator is zero.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    true_positive = np.sum((actual == state) & (predicted == state))
    false_positive = np.sum((actual != state) & (predicted == state))
    false_negative = np.sum((actual == state) & (predicted != state))

    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def plot_class_level_f1(actual, rf_pred, persistence_pred, labels):
    """Save per-state F1 scores to CSV and compare the two forecasting models."""
    results = []
    for state in labels:
        results.append(
            {
                "state": state,
                "Persistence": calculate_class_f1(actual, persistence_pred, state),
                "Random Forest": calculate_class_f1(actual, rf_pred, state),
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "class_level_f1_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.set_index("state").plot(kind="bar", ax=ax)
    ax.set_title("F1 Score by Occupancy State")
    ax.set_xlabel("Actual occupancy state")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / FIGURE_FILENAMES[4], dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. ACTUAL VERSUS PREDICTED STATES OVER TIME
# ---------------------------------------------------------------------------

def plot_representative_time_series(predictions):
    """Plot and export a representative 48-hour window for one floor space.

    The floor space with the most test examples is selected for convenience.
    If multiple forecast origins exist for one target timestamp, the last row
    after sorting is retained to avoid plotting multiple values at one time.
    The resulting plot is illustrative and is not a substitute for aggregate
    model metrics.
    
    Returns:
        A tuple containing the selected floor-space ID and number of plotted
        rows, or (None, 0) if no usable time-series rows are available.
    """
    print("Creating time-series comparison...")

    floor_counts = predictions["floorspaceid"].value_counts()
    if floor_counts.empty:
        print("Skipping time-series plot: no floor-space IDs are available.")
        return None, 0

    selected_floor = floor_counts.index[0]
    floor_data = predictions.loc[
        predictions["floorspaceid"] == selected_floor
    ].copy()
    floor_data = floor_data.sort_values("forecast_target_time")

    # A single point per target timestamp makes the time series readable.
    floor_data = floor_data.drop_duplicates(
        subset=["forecast_target_time"], keep="last"
    )
    if floor_data.empty:
        print("Skipping time-series plot: no rows remain for the selected floor.")
        return selected_floor, 0

    # Centre the window on the middle timestamp to show a typical segment of
    # this floor space's available test period rather than its first few rows.
    centre_time = floor_data.iloc[len(floor_data) // 2]["forecast_target_time"]
    window_start = centre_time - pd.Timedelta(hours=24)
    window_end = centre_time + pd.Timedelta(hours=24)
    plot_data = floor_data.loc[
        floor_data["forecast_target_time"].between(window_start, window_end)
    ].copy()

    state_codes = {state: index for index, state in enumerate(STATE_ORDER)}
    for column, output_column in (
        ("actual_future_state", "actual_code"),
        ("random_forest_prediction", "predicted_code"),
        ("persistence_prediction", "persistence_code"),
    ):
        plot_data[output_column] = plot_data[column].map(state_codes)

    # If the input contains unexpected labels, the mapping above produces NaN.
    # Warn rather than silently pretending those values have a known numeric code.
    if plot_data[["actual_code", "predicted_code", "persistence_code"]].isna().any().any():
        print(
            "Warning: unexpected state labels found in the time-series data; "
            "those values will appear as gaps in the plot."
        )

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.step(
        plot_data["forecast_target_time"],
        plot_data["actual_code"],
        where="post",
        label="Actual future state",
    )
    ax.step(
        plot_data["forecast_target_time"],
        plot_data["predicted_code"],
        where="post",
        label="Random Forest prediction",
        alpha=0.85,
    )
    ax.step(
        plot_data["forecast_target_time"],
        plot_data["persistence_code"],
        where="post",
        label="Persistence prediction",
        alpha=0.7,
    )

    # Use the fixed order to keep state positions consistent between plots.
    ax.set_yticks(range(len(STATE_ORDER)), STATE_ORDER)
    ax.set_title(
        "Actual vs Predicted Occupancy States (Representative 48-Hour Window)"
    )
    ax.set_xlabel("Forecast target time (UTC)")
    ax.set_ylabel("Recorded occupancy state")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / FIGURE_FILENAMES[5], dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    plot_data.to_csv(OUTPUT_DIR / "representative_time_series.csv", index=False)
    print(f"Selected floor space: {selected_floor}")
    print(f"Time-series points plotted: {len(plot_data):,}")
    return selected_floor, len(plot_data)


# ---------------------------------------------------------------------------
# 8. SUMMARY AND MAIN WORKFLOW
# ---------------------------------------------------------------------------

def write_summary(predictions, generated_figures):
    """Write a text summary of the input size and outputs generated."""
    summary_lines = [
        "OCCUPANCY FORECAST VISUALISATION SUMMARY",
        "=" * 45,
        f"Test predictions used: {len(predictions):,}",
        f"Floor spaces represented: {predictions['floorspaceid'].nunique()}",
        f"Visualisation output directory: {OUTPUT_DIR}",
        "",
        "Generated figures:",
        *generated_figures,
        "",
        "Interpretation notes:",
        "- Recorded sensor states are not independently verified physical occupancy.",
        "- Feature importance describes model behaviour, not causation.",
        "- The time-series chart is a representative example, not an aggregate metric.",
        "- All timestamps are shown in UTC.",
    ]
    (OUTPUT_DIR / "visualisation_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


def main():
    """Run validation, generate all available visualisations, and save a summary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OCCUPANCY FORECAST VISUALISATIONS")
    print("=" * 70)

    predictions, comparison, importance = load_inputs()
    print(f"Test predictions loaded: {len(predictions):,}")
    print(f"Output directory: {OUTPUT_DIR}")

    actual = predictions["actual_future_state"]
    rf_pred = predictions["random_forest_prediction"]
    persistence_pred = predictions["persistence_prediction"]
    labels = get_state_labels(actual, rf_pred, persistence_pred)

    if set(labels) - set(STATE_ORDER):
        print(
            "Warning: unexpected state labels found. They are included in "
            "confusion matrices and F1 results, but not assigned a position "
            "in the representative time-series plot."
        )

    print("\nCreating confusion matrices...")
    plot_confusion_matrices(actual, rf_pred, labels)

    print("Creating model comparison chart...")
    plot_model_comparison(comparison)

    print("Creating feature-importance chart...")
    feature_plot_created = plot_feature_importance(importance)

    print("Comparing class-level F1 scores...")
    plot_class_level_f1(actual, rf_pred, persistence_pred, labels)

    _, time_series_count = plot_representative_time_series(predictions)

    # Only list figures that were actually written during this run.
    generated_figures = FIGURE_FILENAMES[:3]
    if feature_plot_created:
        generated_figures.append(FIGURE_FILENAMES[3])
    generated_figures.append(FIGURE_FILENAMES[4])
    if time_series_count:
        generated_figures.append(FIGURE_FILENAMES[5])

    write_summary(predictions, generated_figures)

    print("\nAll available visualisations saved to:")
    print(OUTPUT_DIR)
    print("\nVisualisation generation complete.")


if __name__ == "__main__":
    main()

