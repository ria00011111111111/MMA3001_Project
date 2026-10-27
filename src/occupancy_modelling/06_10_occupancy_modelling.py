
"""Audit the timing and integrity of the occupancy forecast."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_forecast_v2"
PREDICTIONS_FILE = OUTPUT_DIR / "test_forecast_predictions.csv"

FORECAST_MINUTES = 5
TOLERANCE_SECONDS = 60

print("=" * 65)
print("OCCUPANCY FORECAST VALIDATION AUDIT")
print("=" * 65)

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Forecast predictions not found: {PREDICTIONS_FILE}"
    )

data = pd.read_csv(PREDICTIONS_FILE)

required_columns = [
    "forecast_origin",
    "forecast_target_time",
    "collecteddate",
    "floorspaceid",
    "current_state",
    "actual_future_state",
    "random_forest_prediction",
    "persistence_prediction",
]

missing = set(required_columns) - set(data.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Parse timestamps as UTC.
for column in [
    "forecast_origin",
    "forecast_target_time",
    "collecteddate",
]:
    data[column] = pd.to_datetime(
        data[column], utc=True, errors="coerce"
    )

if data[required_columns].isna().any().any():
    print("WARNING: Required columns contain missing values.")

data = data.dropna(subset=required_columns).copy()

# 1. Verify nominal forecast horizon.
data["nominal_horizon_minutes"] = (
    data["forecast_target_time"] - data["forecast_origin"]
).dt.total_seconds() / 60

print("\n1. NOMINAL FORECAST HORIZON")
print(data["nominal_horizon_minutes"].describe().to_string())

assert np.allclose(
    data["nominal_horizon_minutes"], FORECAST_MINUTES
), "Some forecast target timestamps are not five minutes after origin."

print("PASS: All nominal target timestamps are five minutes ahead.")

# 2. Compare actual observation timestamps.
# collecteddate is the timestamp of the current observation.
# The actual future observation timestamp is not currently saved
# in the predictions file, so it cannot be verified directly here.
data["current_observation_offset_seconds"] = (
    data["collecteddate"] - data["forecast_origin"]
).dt.total_seconds()

print("\n2. CURRENT OBSERVATION TIMING WITHIN ITS BIN")
print(data["current_observation_offset_seconds"].describe().to_string())

# 3. Check for duplicate floor-space / forecast-origin pairs.
duplicates = data.duplicated(
    subset=["floorspaceid", "forecast_origin"]
).sum()

print("\n3. DUPLICATE FORECAST ORIGINS")
print(f"Duplicate floor-space/origin pairs: {duplicates:,}")

if duplicates:
    print("WARNING: Duplicate forecast origins were found.")
else:
    print("PASS: No duplicate floor-space/origin pairs.")

# 4. Check target and predictions use the expected state labels.
valid_states = {
    "CurrentlyOccupied",
    "RecentlyOccupied",
    "NotOccupied",
}

for column in [
    "current_state",
    "actual_future_state",
    "random_forest_prediction",
    "persistence_prediction",
]:
    unexpected = set(data[column].unique()) - valid_states
    if unexpected:
        print(f"WARNING: Unexpected values in {column}: {unexpected}")

# 5. Verify persistence predictions equal current state.
persistence_mismatches = (
    data["persistence_prediction"] != data["current_state"]
).sum()

print("\n4. PERSISTENCE BASELINE CHECK")
print(f"Mismatches: {persistence_mismatches:,}")

assert persistence_mismatches == 0, (
    "Persistence predictions do not always equal current state."
)
print("PASS: Persistence predicts the current state.")

# 6. Summarise test coverage by floor space.
coverage = (
    data.groupby("floorspaceid")
    .agg(
        predictions=("actual_future_state", "size"),
        first_target=("forecast_target_time", "min"),
        last_target=("forecast_target_time", "max"),
    )
    .reset_index()
)

coverage.to_csv(
    OUTPUT_DIR / "forecast_validation_coverage.csv",
    index=False,
)

# Save a short audit report.
report = [
    "OCCUPANCY FORECAST VALIDATION AUDIT",
    f"Rows checked: {len(data):,}",
    f"Nominal horizon minutes: {FORECAST_MINUTES}",
    f"Duplicate origins: {duplicates:,}",
    f"Persistence mismatches: {persistence_mismatches:,}",
    "",
    "IMPORTANT LIMITATION:",
    "The predictions file does not contain the actual timestamp",
    "of the future observation used to create each target.",
    "Therefore, actual elapsed time between the current and future",
    "sensor observations cannot be verified using this file alone.",
    "The source data must be reprocessed to audit that interval.",
]

(OUTPUT_DIR / "forecast_validation_audit.txt").write_text(
    "\n".join(report),
    encoding="utf-8",
)

print("\nSaved validation audit to:")
print(OUTPUT_DIR)
print("\nAudit complete.")
