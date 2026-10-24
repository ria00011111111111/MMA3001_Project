
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_model_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 250_000
TEST_FRACTION = 0.20
RANDOM_STATE = 42
MAX_GAP_MINUTES = 24 * 60

TARGET = "occupancystatus"

print("=" * 70)
print("OCCUPANCY MODEL V2 - TIMESTAMP AND LEAKAGE AUDIT")
print("=" * 70)


# ============================================================
# 2. LOAD RAW OCCUPANCY DATA
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

            # Keep all valid rows here. Deduplicate globally below
            # so chunk boundaries cannot affect the selected record.
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

print("\nCombining raw records...")

data = pd.concat(chunks, ignore_index=True)
del chunks

print(f"Valid raw records: {len(data):,}")


# ============================================================
# 3. GLOBAL DEDUPLICATION AND TIMESTAMP AUDIT
# ============================================================

print("\nAuditing timestamps and selecting interval labels...")

data = data.sort_values(
    ["floorspaceid", "interval", "collecteddate"]
)

# Select the last collected observation within each five-minute
# interval for each floor space.
data = data.drop_duplicates(
    subset=["floorspaceid", "interval"],
    keep="last",
)

data = data.sort_values(
    ["floorspaceid", "interval"]
).reset_index(drop=True)

# Save an audit summary.
timestamp_audit = {
    "valid_raw_records": int(total_rows),
    "valid_records_after_timestamp_filter": int(
        len(data)
    ),
    "aggregated_intervals": int(len(data)),
    "unique_floor_spaces": int(
        data["floorspaceid"].nunique()
    ),
    "earliest_interval_utc": str(
        data["interval"].min()
    ),
    "latest_interval_utc": str(
        data["interval"].max()
    ),
    "duplicate_floor_space_intervals": int(
        data.duplicated(
            ["floorspaceid", "interval"]
        ).sum()
    ),
}

pd.Series(timestamp_audit).to_csv(
    OUTPUT_DIR / "timestamp_audit.csv",
    header=["value"],
)

print(f"Aggregated intervals: {len(data):,}")
print(f"Floor spaces: {data['floorspaceid'].nunique()}")
print(f"Earliest interval: {data['interval'].min()}")
print(f"Latest interval: {data['interval'].max()}")
print(
    "Duplicate floor-space intervals remaining:",
    data.duplicated(
        ["floorspaceid", "interval"]
    ).sum(),
)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("\nCreating features from past observations...")

grouped = data.groupby("floorspaceid", sort=False)

# Previous recorded state within the same floor space.
data["previous_state"] = grouped[TARGET].shift(1)

# Previous actual observation timestamp within that floor space.
data["previous_timestamp"] = (
    grouped["collecteddate"].shift(1)
)

data["gap_minutes"] = (
    data["collecteddate"] - data["previous_timestamp"]
).dt.total_seconds() / 60

# Diagnose invalid or unusual observation gaps.
invalid_gap_count = int(
    (data["gap_minutes"] <= 0).sum()
)

negative_gap_count = int(
    (data["gap_minutes"] < 0).sum()
)

print(f"Non-positive gaps before filtering: {invalid_gap_count}")
print(f"Negative gaps before filtering: {negative_gap_count}")

# Remove records where a valid previous observation is unavailable.
data.loc[
    data["gap_minutes"] <= 0,
    "gap_minutes",
] = np.nan

# This is a past-to-current observation gap, not a future gap.
data["gap_minutes_capped"] = data["gap_minutes"].clip(
    upper=MAX_GAP_MINUTES
)

data["log_gap_minutes"] = np.log1p(
    data["gap_minutes_capped"]
)

# Calendar features use the current interval timestamp.
data["hour"] = data["interval"].dt.hour
data["day_of_week"] = data["interval"].dt.dayofweek
data["month"] = data["interval"].dt.month
data["is_weekend"] = (
    data["day_of_week"] >= 5
).astype(int)

# Do not use rows lacking a previous observation.
data = data.dropna(
    subset=[
        "previous_state",
        "previous_timestamp",
        "log_gap_minutes",
    ]
).copy()

categorical_features = [
    "floorspaceid",
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

# The target is the last recorded state in the current
# five-minute interval. No target-derived features are included.
X = data[feature_columns].copy()
y = data[TARGET].copy()

print(f"Usable observations: {len(data):,}")
print(f"Feature columns: {feature_columns}")


# ============================================================
# 5. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

print("\nCreating chronological train/test split...")

unique_times = np.sort(data["interval"].unique())

split_index = int(
    len(unique_times) * (1 - TEST_FRACTION)
)
split_index = min(
    max(split_index, 1),
    len(unique_times) - 1,
)

cutoff_time = unique_times[split_index]

# Ensure every floor-space record in a given interval belongs
# to only one side of the split.
train_mask = data["interval"] < cutoff_time
test_mask = data["interval"] >= cutoff_time

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

print(f"Training observations: {len(X_train):,}")
print(f"Testing observations: {len(X_test):,}")
print(f"Training ends before: {cutoff_time}")
print(f"Testing starts at: {cutoff_time}")

assert (
    data.loc[train_mask, "interval"].max()
    < data.loc[test_mask, "interval"].min()
), "Train/test temporal overlap detected."

print("\nTraining class distribution:")
print(y_train.value_counts().to_string())

print("\nTesting class distribution:")
print(y_test.value_counts().to_string())

labels = sorted(data[TARGET].unique())

if y_train.nunique() < 2:
    raise ValueError(
        "Training data contains fewer than two classes."
    )


# ============================================================
# 6. BASELINES
# ============================================================

print("\nCreating baseline predictions...")

# Baseline 1: always predict the most common training class.
majority_model = DummyClassifier(
    strategy="most_frequent"
)
majority_model.fit(X_train, y_train)

majority_predictions = majority_model.predict(X_test)

# Baseline 2: predict that the current state equals the
# previous observed state.
transition_predictions = (
    X_test["previous_state"].to_numpy()
)


# ============================================================
# 7. RANDOM FOREST WITH TIME-GAP FEATURE
# ============================================================

print("\nTraining Random Forest with time-gap feature...")

def make_model(numeric_columns):
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
                numeric_columns,
            ),
        ]
    )

    return Pipeline(
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


model = make_model(numeric_features)
model.fit(X_train, y_train)

rf_predictions = model.predict(X_test)


# ============================================================
# 8. ABLATION TEST WITHOUT TIME GAP
# ============================================================

print("\nTraining Random Forest without time-gap feature...")

reduced_numeric_features = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

reduced_feature_columns = (
    categorical_features + reduced_numeric_features
)

model_without_gap = make_model(
    reduced_numeric_features
)

model_without_gap.fit(
    X_train[reduced_feature_columns],
    y_train,
)

predictions_without_gap = model_without_gap.predict(
    X_test[reduced_feature_columns]
)


# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate(name, actual, predicted):
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
    evaluate(
        "Majority-class baseline",
        y_test,
        majority_predictions,
    ),
    evaluate(
        "Previous-state baseline",
        y_test,
        transition_predictions,
    ),
    evaluate(
        "Random Forest with gap",
        y_test,
        rf_predictions,
    ),
    evaluate(
        "Random Forest without gap",
        y_test,
        predictions_without_gap,
    ),
])

results.to_csv(
    OUTPUT_DIR / "ablation_model_comparison.csv",
    index=False,
)

print("\nMODEL COMPARISON")
print(
    results.to_string(
        index=False,
        float_format="%.4f",
    )
)

print("\nCLASSIFICATION REPORT: RANDOM FOREST WITH GAP")
print(
    classification_report(
        y_test,
        rf_predictions,
        labels=labels,
        zero_division=0,
    )
)

print("\nCLASSIFICATION REPORT: RANDOM FOREST WITHOUT GAP")
print(
    classification_report(
        y_test,
        predictions_without_gap,
        labels=labels,
        zero_division=0,
    )
)


# ============================================================
# 10. CONFUSION MATRICES
# ============================================================

for name, predictions in [
    ("with_gap", rf_predictions),
    ("without_gap", predictions_without_gap),
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
# 11. FEATURE IMPORTANCE
# ============================================================

print("\nSaving Random Forest feature importance...")

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
    OUTPUT_DIR / "feature_importance_v2.csv",
    index=False,
)

print("\nTOP 15 FEATURES")
print(
    importance_df.head(15).to_string(index=False)
)


# ============================================================
# 12. SAVE TEST PREDICTIONS AND DIAGNOSTICS
# ============================================================

prediction_output = data.loc[
    test_mask,
    [
        "interval",
        "collecteddate",
        "floorspaceid",
        TARGET,
        "previous_state",
        "previous_timestamp",
        "gap_minutes",
        "log_gap_minutes",
    ],
].copy()

prediction_output["majority_prediction"] = (
    majority_predictions
)
prediction_output["previous_state_prediction"] = (
    transition_predictions
)
prediction_output["random_forest_prediction"] = (
    rf_predictions
)
prediction_output["random_forest_without_gap_prediction"] = (
    predictions_without_gap
)

prediction_output.to_csv(
    OUTPUT_DIR / "test_predictions_v2.csv",
    index=False,
)

# Save a small, explicit audit of the prediction timeline.
timeline_audit = pd.DataFrame([{
    "cutoff_time_utc": str(cutoff_time),
    "train_start_utc": str(
        data.loc[train_mask, "interval"].min()
    ),
    "train_end_utc": str(
        data.loc[train_mask, "interval"].max()
    ),
    "test_start_utc": str(
        data.loc[test_mask, "interval"].min()
    ),
    "test_end_utc": str(
        data.loc[test_mask, "interval"].max()
    ),
    "train_test_overlap": bool(
        data.loc[train_mask, "interval"].max()
        >= data.loc[test_mask, "interval"].min()
    ),
    "gap_uses_future_observation": False,
    "target": TARGET,
    "target_definition": (
        "Last recorded state in each five-minute interval"
    ),
}])

timeline_audit.to_csv(
    OUTPUT_DIR / "timeline_audit.csv",
    index=False,
)

print("\nOutputs saved to:")
print(OUTPUT_DIR)

print("\nIMPORTANT INTERPRETATION NOTES")
print("- Timestamps are parsed as UTC.")
print("- Each interval retains its latest collected observation.")
print("- Previous state and gap use earlier observations only.")
print("- The model predicts the recorded interval label, not")
print("  continuous physical occupancy throughout the interval.")
print("- Temporal ordering does not prove physical room matching.")
print("- Environmental data is not used in this model.")
print("- The time-gap feature may reflect sensor reporting behaviour.")
print("- This audit cannot establish whether the source sensor's")
print("  labels themselves accurately represent physical occupancy.")

print("\nOccupancy model audit complete.")
