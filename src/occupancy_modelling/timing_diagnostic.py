
"""Check the timing represented in the saved forecast predictions."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "occupancy_forecast_v2"
    / "test_forecast_predictions.csv"
)

data = pd.read_csv(INPUT_FILE)

required = [
    "forecast_origin",
    "forecast_target_time",
    "collecteddate",
    "floorspaceid",
    "current_state",
    "actual_future_state",
]

missing = [column for column in required if column not in data.columns]

if missing:
    print("Missing columns:", missing)
    print("Available columns:", data.columns.tolist())
    raise SystemExit("Cannot complete the timing diagnostic.")

for column in [
    "forecast_origin",
    "forecast_target_time",
    "collecteddate",
]:
    data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")

data = data.dropna(subset=required).copy()

data["origin_to_target_seconds"] = (
    data["forecast_target_time"] - data["forecast_origin"]
).dt.total_seconds()

data["observation_to_target_seconds"] = (
    data["forecast_target_time"] - data["collecteddate"]
).dt.total_seconds()

print("\nFORECAST TIMING DIAGNOSTIC")
print("=" * 45)

print("\nOrigin to target (seconds):")
print(data["origin_to_target_seconds"].describe())

print("\nActual observation to target (seconds):")
print(data["observation_to_target_seconds"].describe())

print("\nTarget occurs before observation:",
      (data["observation_to_target_seconds"] < 0).sum())

print("\nTime from observation to target, in minutes:")
print(
    (data["observation_to_target_seconds"] / 60)
    .describe()
    .round(2)
)

print("\nSample rows:")
print(
    data[
        [
            "forecast_origin",
            "collecteddate",
            "forecast_target_time",
            "origin_to_target_seconds",
            "observation_to_target_seconds",
        ]
    ].head(10).to_string(index=False)
)
