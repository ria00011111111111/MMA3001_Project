
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_forecast_v2"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "occupancy_forecast_visualisations"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = INPUT_DIR / "test_forecast_predictions.csv"
COMPARISON_FILE = INPUT_DIR / "forecast_model_comparison.csv"
IMPORTANCE_FILE = INPUT_DIR / "forecast_feature_importance.csv"

STATE_ORDER = [
    "CurrentlyOccupied",
    "NotOccupied",
    "RecentlyOccupied",
]

STATE_COLOURS = {
    "CurrentlyOccupied": 0,
    "NotOccupied": 1,
    "RecentlyOccupied": 2,
}

print("=" * 70)
print("OCCUPANCY FORECAST VISUALISATIONS")
print("=" * 70)


# ============================================================
# 2. CHECK AND LOAD OUTPUTS
# ============================================================

for path in [
    PREDICTIONS_FILE,
    COMPARISON_FILE,
    IMPORTANCE_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the corrected forecast script first."
        )

predictions = pd.read_csv(PREDICTIONS_FILE)

comparison = pd.read_csv(COMPARISON_FILE)

importance = pd.read_csv(IMPORTANCE_FILE)

predictions["forecast_origin"] = pd.to_datetime(
    predictions["forecast_origin"],
    utc=True,
    errors="coerce",
)

predictions["forecast_target_time"] = pd.to_datetime(
    predictions["forecast_target_time"],
    utc=True,
    errors="coerce",
)

predictions = predictions.dropna(
    subset=[
        "forecast_origin",
        "forecast_target_time",
        "actual_future_state",
        "random_forest_prediction",
        "persistence_prediction",
    ]
).copy()

print(f"Test predictions loaded: {len(predictions):,}")
print(f"Output directory: {OUTPUT_DIR}")


# ============================================================
# 3. CONFUSION MATRIX
# ============================================================

print("\nCreating confusion matrices...")

actual = predictions["actual_future_state"].astype(str)
rf_pred = predictions["random_forest_prediction"].astype(str)
persistence_pred = predictions["persistence_prediction"].astype(str)

observed_states = set(actual.unique())

labels = [
    state for state in STATE_ORDER
    if state in observed_states
]

matrix = confusion_matrix(
    actual,
    rf_pred,
    labels=labels,
)

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

plt.setp(
    ax.get_xticklabels(),
    rotation=25,
    ha="right",
)

threshold = matrix.max() / 2 if matrix.size else 0

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(
            j,
            i,
            f"{matrix[i, j]:,}",
            ha="center",
            va="center",
            color="white" if matrix[i, j] > threshold else "black",
        )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "01_random_forest_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 4. NORMALISED CONFUSION MATRIX
# ============================================================

row_totals = matrix.sum(axis=1, keepdims=True)

normalised_matrix = np.divide(
    matrix,
    row_totals,
    out=np.zeros_like(matrix, dtype=float),
    where=row_totals != 0,
)

fig, ax = plt.subplots(figsize=(9, 7))

image = ax.imshow(
    normalised_matrix,
    vmin=0,
    vmax=1,
    interpolation="nearest",
)

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

plt.setp(
    ax.get_xticklabels(),
    rotation=25,
    ha="right",
)

for i in range(normalised_matrix.shape[0]):
    for j in range(normalised_matrix.shape[1]):
        value = normalised_matrix[i, j]

        ax.text(
            j,
            i,
            f"{value:.1%}",
            ha="center",
            va="center",
            color="white" if value > 0.5 else "black",
        )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "02_normalised_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 5. MODEL COMPARISON
# ============================================================

print("Creating model comparison chart...")

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
]

comparison_long = comparison.melt(
    id_vars="model",
    value_vars=metric_columns,
    var_name="metric",
    value_name="score",
)

fig, ax = plt.subplots(figsize=(11, 6))

comparison_long.pivot(
    index="metric",
    columns="model",
    values="score",
).plot(
    kind="bar",
    ax=ax,
)

ax.set_title("Occupancy Forecast Model Comparison")
ax.set_xlabel("Evaluation metric")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=20)
ax.legend(title="Model")

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "03_model_comparison.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================

print("Creating feature importance chart...")

top_features = (
    importance.sort_values("importance", ascending=False)
    .head(15)
    .sort_values("importance", ascending=True)
)

fig, ax = plt.subplots(figsize=(11, 8))

ax.barh(
    top_features["feature"],
    top_features["importance"],
)

ax.set_title("Random Forest: Top 15 Feature Importances")
ax.set_xlabel("Feature importance")
ax.set_ylabel("Feature")

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "04_feature_importance.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 7. CLASS-LEVEL F1 COMPARISON
# ============================================================

print("Comparing class-level F1 scores...")

def class_f1(actual_values, predicted_values, state):
    actual_values = np.asarray(actual_values)
    predicted_values = np.asarray(predicted_values)

    true_positive = np.sum(
        (actual_values == state)
        & (predicted_values == state)
    )

    false_positive = np.sum(
        (actual_values != state)
        & (predicted_values == state)
    )

    false_negative = np.sum(
        (actual_values == state)
        & (predicted_values != state)
    )

    denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    return (
        2 * true_positive / denominator
        if denominator > 0
        else 0.0
    )


class_results = []

for state in labels:
    class_results.append({
        "state": state,
        "Persistence": class_f1(
            actual,
            persistence_pred,
            state,
        ),
        "Random Forest": class_f1(
            actual,
            rf_pred,
            state,
        ),
    })

class_results_df = pd.DataFrame(class_results)

class_results_df.to_csv(
    OUTPUT_DIR / "class_level_f1_scores.csv",
    index=False,
)

class_results_plot = class_results_df.set_index("state")

fig, ax = plt.subplots(figsize=(10, 6))

class_results_plot.plot(
    kind="bar",
    ax=ax,
)

ax.set_title("F1 Score by Occupancy State")
ax.set_xlabel("Actual occupancy state")
ax.set_ylabel("F1 score")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=20)
ax.legend(title="Model")

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "05_class_level_f1_comparison.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 8. ACTUAL VERSUS PREDICTED STATES OVER TIME
# ============================================================

print("Creating time-series comparison...")

# Select one floor space with the most test examples.
floor_counts = predictions["floorspaceid"].value_counts()
selected_floor = floor_counts.index[0]

floor_data = predictions.loc[
    predictions["floorspaceid"] == selected_floor
].copy()

floor_data = floor_data.sort_values(
    "forecast_target_time"
)

# Plot a representative 48-hour window, starting near the
# middle of this floor space's test period.
floor_data = floor_data.drop_duplicates(
    subset=["forecast_target_time"],
    keep="last",
)

if len(floor_data) > 0:
    middle_index = len(floor_data) // 2

    centre_time = floor_data.iloc[
        middle_index
    ]["forecast_target_time"]

    window_start = centre_time - pd.Timedelta(hours=24)
    window_end = centre_time + pd.Timedelta(hours=24)

    plot_data = floor_data.loc[
        floor_data["forecast_target_time"].between(
            window_start,
            window_end,
        )
    ].copy()

    # Numeric codes are for visualisation only.
    state_codes = {
        state: index
        for index, state in enumerate(STATE_ORDER)
    }

    plot_data["actual_code"] = (
        plot_data["actual_future_state"]
        .map(state_codes)
    )

    plot_data["predicted_code"] = (
        plot_data["random_forest_prediction"]
        .map(state_codes)
    )

    plot_data["persistence_code"] = (
        plot_data["persistence_prediction"]
        .map(state_codes)
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

    ax.set_yticks(
        range(len(STATE_ORDER)),
        STATE_ORDER,
    )

    ax.set_title(
        "Actual vs Predicted Occupancy States "
        "(Representative 48-Hour Window)"
    )

    ax.set_xlabel("Forecast target time (UTC)")
    ax.set_ylabel("Occupancy state")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "06_actual_vs_predicted_time_series.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    plot_data.to_csv(
        OUTPUT_DIR / "representative_time_series.csv",
        index=False,
    )

    print(f"Selected floor space: {selected_floor}")
    print(f"Time-series points plotted: {len(plot_data):,}")

else:
    print("No rows available for the time-series plot.")


# ============================================================
# 9. SAVE SUMMARY
# ============================================================

summary_lines = [
    "OCCUPANCY FORECAST VISUALISATION SUMMARY",
    "=" * 45,
    f"Test predictions: {len(predictions):,}",
    f"Floor spaces represented: {predictions['floorspaceid'].nunique()}",
    f"Visualisation output directory: {OUTPUT_DIR}",
    "",
    "Generated figures:",
    "01_random_forest_confusion_matrix.png",
    "02_normalised_confusion_matrix.png",
    "03_model_comparison.png",
    "04_feature_importance.png",
    "05_class_level_f1_comparison.png",
    "06_actual_vs_predicted_time_series.png",
    "",
    "Note: the time-series chart shows recorded states and model",
    "predictions, not independently verified physical occupancy.",
    "All timestamps are shown in UTC.",
]

(OUTPUT_DIR / "visualisation_summary.txt").write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

print("\nAll visualisations saved to:")
print(OUTPUT_DIR)

print("\nVisualisation generation complete.")
