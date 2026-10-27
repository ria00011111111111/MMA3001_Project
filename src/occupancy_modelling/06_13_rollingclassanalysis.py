
"""Compare per-class forecasting performance across rolling windows."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "occupancy_rolling_evaluation"
    / "rolling_evaluation_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "occupancy_rolling_evaluation"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_ORDER = [
    "CurrentlyOccupied",
    "NotOccupied",
    "RecentlyOccupied",
]

MODEL_COLUMNS = {
    "Majority baseline": "majority_prediction",
    "Persistence baseline": "persistence_prediction",
    "Random Forest": "random_forest_prediction",
}

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Predictions file not found: {INPUT_FILE}")

data = pd.read_csv(INPUT_FILE)

required = {
    "window",
    "actual_state",
    *MODEL_COLUMNS.values(),
}
missing = required - set(data.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

results = []

for window in sorted(data["window"].unique()):
    window_data = data[data["window"] == window]

    print("\n" + "=" * 70)
    print(window)
    print("=" * 70)

    for model_name, prediction_column in MODEL_COLUMNS.items():
        report = classification_report(
            window_data["actual_state"],
            window_data[prediction_column],
            labels=STATE_ORDER,
            target_names=STATE_ORDER,
            output_dict=True,
            zero_division=0,
        )

        print(f"\n{model_name}")
        print(
            f"{'Class':<22} {'Precision':>10} "
            f"{'Recall':>10} {'F1':>10} {'Support':>10}"
        )

        for state in STATE_ORDER:
            metrics = report[state]

            print(
                f"{state:<22} "
                f"{metrics['precision']:>10.3f} "
                f"{metrics['recall']:>10.3f} "
                f"{metrics['f1-score']:>10.3f} "
                f"{int(metrics['support']):>10,}"
            )

            results.append({
                "window": window,
                "model": model_name,
                "class": state,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1-score"],
                "support": int(metrics["support"]),
            })

results_df = pd.DataFrame(results)

output_file = OUTPUT_DIR / "rolling_class_level_metrics.csv"
results_df.to_csv(output_file, index=False)

print("\nSaved class-level results to:")
print(output_file)
