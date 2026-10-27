
"""Check the actual time between occupancy readings used as forecast pairs."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "5occupancySensor_MayToDec2024_9MRows (1).zip"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_forecast_v2"

CHUNK_SIZE = 250_000
FORECAST_MINUTES = 5

print("=" * 65)
print("ACTUAL OCCUPANCY FORECAST TIMING AUDIT")
print("=" * 65)

# Read only the fields required for the timing audit.
chunks = []
total_rows = 0

with ZipFile(ZIP_FILE) as archive:
    csv_file = next(
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    )

    with archive.open(csv_file) as file:
        for chunk in pd.read_csv(
            file,
            usecols=["floorspaceid", "occupancystatus", "collecteddate"],
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
                subset=["floorspaceid", "occupancystatus", "collecteddate"]
            )
            chunk["interval"] = chunk["collecteddate"].dt.floor("5min")
            chunks.append(chunk)

            print(f"Read {total_rows:,} raw rows...")

data = pd.concat(chunks, ignore_index=True)
del chunks

# Reproduce the original model's aggregation rule.
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

print(f"\nAggregated intervals: {len(data):,}")

# Keep the actual timestamp and state from each selected observation.
current = data[
    ["floorspaceid", "interval", "collecteddate", "occupancystatus"]
].rename(
    columns={
        "collecteddate": "current_observation_time",
        "occupancystatus": "current_state",
    }
)

future = data[
    ["floorspaceid", "interval", "collecteddate", "occupancystatus"]
].rename(
    columns={
        "interval": "future_interval",
        "collecteddate": "future_observation_time",
        "occupancystatus": "actual_future_state",
    }
)

current["future_interval"] = (
    current["interval"] + pd.Timedelta(minutes=FORECAST_MINUTES)
)

pairs = current.merge(
    future,
    on=["floorspaceid", "future_interval"],
    how="inner",
    validate="one_to_one",
)

pairs["actual_elapsed_minutes"] = (
    pairs["future_observation_time"]
    - pairs["current_observation_time"]
).dt.total_seconds() / 60

pairs["elapsed_error_minutes"] = (
    pairs["actual_elapsed_minutes"] - FORECAST_MINUTES
).abs()

print(f"Matched forecast pairs: {len(pairs):,}")

print("\nACTUAL ELAPSED TIME BETWEEN READINGS (MINUTES)")
print(pairs["actual_elapsed_minutes"].describe().to_string())

print("\nPAIRS WITHIN DIFFERENT TIMING TOLERANCES")
for tolerance_seconds in [30, 60, 120]:
    count = (
        pairs["elapsed_error_minutes"] * 60 <= tolerance_seconds
    ).sum()
    percentage = 100 * count / len(pairs)
    print(
        f"Within {tolerance_seconds:>3} seconds of 5 minutes: "
        f"{count:,} ({percentage:.2f}%)"
    )

print("\nELAPSED-TIME QUANTILES (MINUTES)")
print(
    pairs["actual_elapsed_minutes"]
    .quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    .to_string()
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pairs.to_csv(
    OUTPUT_DIR / "actual_forecast_timing_pairs.csv",
    index=False,
)

summary = pd.DataFrame([{
    "matched_pairs": len(pairs),
    "mean_elapsed_minutes": pairs["actual_elapsed_minutes"].mean(),
    "median_elapsed_minutes": pairs["actual_elapsed_minutes"].median(),
    "minimum_elapsed_minutes": pairs["actual_elapsed_minutes"].min(),
    "maximum_elapsed_minutes": pairs["actual_elapsed_minutes"].max(),
    "within_30_seconds_percent": 100 * (
        pairs["elapsed_error_minutes"] * 60 <= 30
    ).mean(),
    "within_60_seconds_percent": 100 * (
        pairs["elapsed_error_minutes"] * 60 <= 60
    ).mean(),
    "within_120_seconds_percent": 100 * (
        pairs["elapsed_error_minutes"] * 60 <= 120
    ).mean(),
}])

summary.to_csv(
    OUTPUT_DIR / "actual_forecast_timing_summary.csv",
    index=False,
)

print("\nSaved timing audit to:")
print(OUTPUT_DIR)
print("\nTiming audit complete.")
