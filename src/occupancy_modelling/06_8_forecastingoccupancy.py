
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "5occupancySensor_MayToDec2024_9MRows (1).zip"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "occupancy_forecast_v2"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 250_000
TEST_FRACTION = 0.20
RANDOM_STATE = 42
MAX_GAP_MINUTES = 24 * 60

TARGET = "occupancystatus"
FORECAST_MINUTES = 5

print("=" * 70)
print("OCCUPANCY FORECAST MODEL - CORRECTED FEATURE CONSTRUCTION")
print("=" * 70)


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

print("\nReading occupancy data from ZIP...")

chunks = []
total_rows = 0

with ZipFile(ZIP_FILE) as archive:
    csv_files = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found inside the occupancy ZIP."
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
            total_rows += len(chunk)

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

            chunk["interval"] = (
                chunk["collecteddate"].dt.floor("5min")
            )

            chunks.append(
                chunk[
                    [
                        "floorspaceid",
                        TARGET,
                        "collecteddate",
                        "interval",
                    ]
                ]
            )

            print(f"Processed {total_rows:,} records...")

print("\nCombining records...")

data = pd.concat(chunks, ignore_index=True)
del chunks

print(f"Valid raw records: {len(data):,}")


# ============================================================
# 3. AGGREGATE TO FIVE-MINUTE INTERVALS
# ============================================================

print("\nAggregating observations...")

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

print(f"Aggregated intervals: {len(data):,}")
print(f"Floor spaces: {data['floorspaceid'].nunique()}")


# ============================================================
# 4. BUILD FEATURES ON THE COMPLETE TIME SERIES
# ============================================================

print("\nCreating historical features before target filtering...")

# Current state is the state observed at forecast origin t.
data["current_state"] = data[TARGET]

# Calculate historical features on the COMPLETE series.
# Do this before removing rows without a future observation.
grouped = data.groupby(
    "floorspaceid",
    sort=False,
)

data["previous_state"] = (
    grouped["current_state"].shift(1)
)

data["previous_timestamp"] = (
    grouped["collecteddate"].shift(1)
)

data["gap_minutes"] = (
    data["collecteddate"]
    - data["previous_timestamp"]
).dt.total_seconds() / 60

# Invalid or nonpositive observation gaps are not usable.
data.loc[
    data["gap_minutes"] <= 0,
    "gap_minutes",
] = np.nan

data["log_gap_minutes"] = np.log1p(
    data["gap_minutes"].clip(
        upper=MAX_GAP_MINUTES
    )
)

# Calendar features known at the forecast origin.
data["hour"] = data["interval"].dt.hour
data["day_of_week"] = data["interval"].dt.dayofweek
data["month"] = data["interval"].dt.month
data["is_weekend"] = (
    data["day_of_week"] >= 5
).astype(int)

# Preserve the complete feature-engineered data for target matching.
# No future information is used to calculate these features.


# ============================================================
# 5. CREATE THE NEXT-FIVE-MINUTE TARGET
# ============================================================

print("\nCreating next-five-minute forecast target...")

# Create a separate lookup of observed states at future intervals.
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
    data["interval"]
    + pd.Timedelta(minutes=FORECAST_MINUTES)
)

# Match the current row to the observed state exactly five minutes later.
forecast_data = data.merge(
    future,
    on=["floorspaceid", "target_interval"],
    how="left",
    validate="many_to_one",
)

eligible_count = int(
    forecast_data["future_state"].notna().sum()
)

excluded_count = int(
    forecast_data["future_state"].isna().sum()
)

print(f"Source intervals: {len(data):,}")
print(f"Eligible forecast examples: {eligible_count:,}")
print(
    "Excluded because no next-interval reading: "
    f"{excluded_count:,}"
)
print(
    "Eligible percentage: "
    f"{100 * eligible_count / len(data):.2f}%"
)

# Do not assume the state of an unobserved interval.
forecast_data = forecast_data.dropna(
    subset=["future_state"]
).copy()

forecast_data["forecast_target"] = (
    forecast_data["future_state"]
)


# ============================================================
# 6. FILTER INCOMPLETE FEATURES
# ============================================================

print("\nFiltering rows with unavailable historical features...")

forecast_data = forecast_data.dropna(
    subset=[
        "previous_state",
        "log_gap_minutes",
    ]
).copy()

categorical_features = [
    "floorspaceid",
    "current_state",
    "previous_state",
]

numeric_features = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "log_gap_minutes",
]

feature_columns = (
    categorical_features + numeric_features
)

X = forecast_data[feature_columns].copy()
y = forecast_data["forecast_target"].copy()

# Explicitly prevent target leakage.
forbidden_features = {
    "future_state",
    "forecast_target",
    "target_interval",
    TARGET,
    "previous_timestamp",
    "collecteddate",
    "interval",
}

assert not (
    set(X.columns) & forbidden_features
), "Potential target leakage detected."

assert (
    forecast_data["current_state"]
    == forecast_data[TARGET]
).all(), "Current state does not match the observed state."

assert (
    forecast_data["target_interval"]
    == forecast_data["interval"]
    + pd.Timedelta(minutes=FORECAST_MINUTES)
).all(), "Incorrect forecast horizon."

print(f"Final forecast examples: {len(X):,}")
print(f"Predictor columns: {feature_columns}")

print("\nFuture target distribution:")
print(y.value_counts().to_string())


# ============================================================
# 7. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

print("\nCreating chronological train/test split...")

unique_target_times = np.sort(
    forecast_data["target_interval"].unique()
)

split_index = int(
    len(unique_target_times) * (1 - TEST_FRACTION)
)

split_index = min(
    max(split_index, 1),
    len(unique_target_times) - 1,
)

cutoff_time = unique_target_times[split_index]

train_mask = (
    forecast_data["target_interval"] < cutoff_time
)

test_mask = (
    forecast_data["target_interval"] >= cutoff_time
)

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

print(f"Training examples: {len(X_train):,}")
print(f"Testing examples: {len(X_test):,}")
print(f"Test target period starts: {cutoff_time}")

assert len(X_train) > 0 and len(X_test) > 0
assert (
    forecast_data.loc[train_mask, "target_interval"].max()
    < forecast_data.loc[test_mask, "target_interval"].min()
), "Training and testing target periods overlap."

labels = sorted(y.unique())

print("\nTraining class distribution:")
print(y_train.value_counts().to_string())

print("\nTesting class distribution:")
print(y_test.value_counts().to_string())

if y_train.nunique() < 2:
    raise ValueError(
        "Training data contains fewer than two target classes."
    )


# ============================================================
# 8. BASELINE MODELS
# ============================================================

print("\nCreating baseline predictions...")

# Baseline 1: predict the most common training class.
majority_model = DummyClassifier(
    strategy="most_frequent"
)
majority_model.fit(X_train, y_train)

majority_predictions = majority_model.predict(X_test)

# Baseline 2: persistence.
# Predict that the state remains the same for five minutes.
persistence_predictions = (
    X_test["current_state"].to_numpy()
)


# ============================================================
# 9. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numeric",
            "passthrough",
            numeric_features,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

model.fit(X_train, y_train)
rf_predictions = model.predict(X_test)


# ============================================================
# 10. EVALUATION
# ============================================================

def evaluate_model(name, actual, predicted):
    return {
        "model": name,
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(
            actual,
            predicted,
        ),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            actual,
            predicted,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
    }


results = pd.DataFrame([
    evaluate_model(
        "Majority baseline",
        y_test,
        majority_predictions,
    ),
    evaluate_model(
        "Persistence baseline",
        y_test,
        persistence_predictions,
    ),
    evaluate_model(
        "Random Forest forecast",
        y_test,
        rf_predictions,
    ),
])

results.to_csv(
    OUTPUT_DIR / "forecast_model_comparison.csv",
    index=False,
)

print("\nFORECAST MODEL COMPARISON")
print(
    results.to_string(
        index=False,
        float_format="%.4f",
    )
)

print("\nRANDOM FOREST CLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        rf_predictions,
        labels=labels,
        zero_division=0,
    )
)

print("\nPERSISTENCE BASELINE CLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        persistence_predictions,
        labels=labels,
        zero_division=0,
    )
)


# ============================================================
# 11. CONFUSION MATRICES
# ============================================================

for name, predictions in [
    ("random_forest", rf_predictions),
    ("persistence", persistence_predictions),
]:
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    matrix_df.to_csv(
        OUTPUT_DIR / f"confusion_matrix_{name}.csv"
    )


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

print("\nSaving feature importance...")

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = model.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
}).sort_values(
    "importance",
    ascending=False,
)

importance_df.to_csv(
    OUTPUT_DIR / "forecast_feature_importance.csv",
    index=False,
)

print("\nTOP 15 FEATURES")
print(
    importance_df.head(15).to_string(index=False)
)


# ============================================================
# 13. SAVE TEST PREDICTIONS
# ============================================================

prediction_output = forecast_data.loc[
    test_mask,
    [
        "interval",
        "target_interval",
        "collecteddate",
        "floorspaceid",
        "current_state",
        "previous_state",
        "gap_minutes",
        "forecast_target",
    ],
].copy()

prediction_output = prediction_output.rename(
    columns={
        "interval": "forecast_origin",
        "target_interval": "forecast_target_time",
        "forecast_target": "actual_future_state",
    }
)

prediction_output["majority_prediction"] = (
    majority_predictions
)

prediction_output["persistence_prediction"] = (
    persistence_predictions
)

prediction_output["random_forest_prediction"] = (
    rf_predictions
)

prediction_output.to_csv(
    OUTPUT_DIR / "test_forecast_predictions.csv",
    index=False,
)


# ============================================================
# 14. SAVE AUDIT INFORMATION
# ============================================================

audit = pd.DataFrame([{
    "raw_records_read": total_rows,
    "aggregated_intervals": len(data),
    "eligible_next_interval_pairs": eligible_count,
    "excluded_no_next_interval": excluded_count,
    "final_forecast_examples": len(forecast_data),
    "training_examples": len(X_train),
    "testing_examples": len(X_test),
    "forecast_horizon_minutes": FORECAST_MINUTES,
    "test_target_start_utc": str(cutoff_time),
    "historical_features_calculated_before_target_filter": True,
    "future_target_in_features": False,
    "missing_future_intervals_filled": False,
}])

audit.to_csv(
    OUTPUT_DIR / "forecast_audit.csv",
    index=False,
)

print("\nOutputs saved to:")
print(OUTPUT_DIR)

print("\nLIMITATIONS")
print("- Forecasts require an observed reading exactly five minutes later.")
print("- Missing future readings are not imputed.")
print("- The target is recorded sensor state, not verified occupancy.")
print("- Environmental variables are not included.")
print("- Sensor-to-room mapping remains unresolved.")

print("\nCorrected occupancy forecast complete.")
