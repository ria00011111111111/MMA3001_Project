
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "5occupancySensor_MayToDec2024_9MRows (1).zip"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_timing_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 250_000

print("=" * 65)
print("OCCUPANCY TIMING AND TRANSITION ANALYSIS")
print("=" * 65)

# ============================================================
# 2. LOAD DATA AND AGGREGATE TO FIVE-MINUTE INTERVALS
# ============================================================

interval_chunks = []
total_rows = 0

print("\nReading occupancy data...")

with ZipFile(ZIP_FILE) as archive:
    csv_files = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError("No CSV found in occupancy ZIP.")

    with archive.open(csv_files[0]) as file:
        for chunk in pd.read_csv(
            file,
            usecols=[
                "floorspaceid",
                "occupancystatus",
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
                    "occupancystatus",
                    "collecteddate",
                ]
            )

            chunk["interval"] = chunk["collecteddate"].dt.floor("5min")

            # Retain the last observed state in each interval
            # within this chunk.
            chunk = chunk.sort_values("collecteddate")
            chunk = chunk.drop_duplicates(
                subset=["floorspaceid", "interval"],
                keep="last",
            )

            interval_chunks.append(
                chunk[
                    [
                        "floorspaceid",
                        "interval",
                        "occupancystatus",
                        "collecteddate",
                    ]
                ]
            )

            print(f"Processed {total_rows:,} records...")

print("\nCombining interval summaries...")

data = pd.concat(interval_chunks, ignore_index=True)
del interval_chunks

# Resolve intervals that appeared in different chunks.
data = data.sort_values("collecteddate")
data = data.drop_duplicates(
    subset=["floorspaceid", "interval"],
    keep="last",
)

data = data.sort_values(
    ["floorspaceid", "interval"]
).reset_index(drop=True)

print(f"Source records: {total_rows:,}")
print(f"Aggregated observations: {len(data):,}")

# ============================================================
# 3. TIME GAPS BETWEEN OBSERVATIONS
# ============================================================

print("\nAnalysing observation time gaps...")

data["previous_interval"] = (
    data.groupby("floorspaceid")["interval"].shift(1)
)

data["gap_minutes"] = (
    data["interval"] - data["previous_interval"]
).dt.total_seconds() / 60

gap_data = data.dropna(subset=["gap_minutes"])

gap_summary = (
    gap_data.groupby("floorspaceid")["gap_minutes"]
    .agg(
        observations="count",
        median_gap_minutes="median",
        mean_gap_minutes="mean",
        p90_gap_minutes=lambda x: x.quantile(0.90),
        p95_gap_minutes=lambda x: x.quantile(0.95),
        maximum_gap_minutes="max",
    )
    .reset_index()
)

gap_summary.to_csv(
    OUTPUT_DIR / "observation_gap_summary.csv",
    index=False,
)

print("\nOBSERVATION GAP SUMMARY (minutes)")
print(gap_summary.to_string(index=False, float_format="%.2f"))

# ============================================================
# 4. STATE TRANSITIONS
# ============================================================

print("\nAnalysing occupancy state transitions...")

data["previous_state"] = (
    data.groupby("floorspaceid")["occupancystatus"].shift(1)
)

transitions = data.dropna(subset=["previous_state"]).copy()

transition_counts = (
    transitions.groupby(
        ["previous_state", "occupancystatus"]
    )
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

transition_counts.to_csv(
    OUTPUT_DIR / "state_transition_counts.csv",
    index=False,
)

print("\nMOST COMMON STATE TRANSITIONS")
print(transition_counts.head(20).to_string(index=False))

# Transition probabilities: what follows each previous state?
transition_counts["probability"] = (
    transition_counts["count"]
    / transition_counts.groupby("previous_state")["count"]
    .transform("sum")
)

transition_counts.to_csv(
    OUTPUT_DIR / "state_transition_probabilities.csv",
    index=False,
)

# ============================================================
# 5. HOW LONG DOES EACH OBSERVED STATE PERSIST?
# ============================================================

print("\nEstimating observed state-run durations...")

data["previous_state"] = (
    data.groupby("floorspaceid")["occupancystatus"].shift(1)
)

# Start a new run whenever the state changes or there is a
# gap greater than five minutes between aggregated observations.
data["new_run"] = (
    data["previous_state"].isna()
    | data["occupancystatus"].ne(data["previous_state"])
    | data["gap_minutes"].gt(5)
)

data["run_id"] = (
    data.groupby("floorspaceid")["new_run"].cumsum()
)

runs = (
    data.groupby(
        ["floorspaceid", "run_id", "occupancystatus"],
        as_index=False,
    )
    .agg(
        observed_intervals=("interval", "size"),
        first_interval=("interval", "min"),
        last_interval=("interval", "max"),
    )
)

# This measures elapsed time between the first and last
# observations in a run, not the true physical state duration.
runs["observed_span_minutes"] = (
    runs["last_interval"] - runs["first_interval"]
).dt.total_seconds() / 60

duration_summary = (
    runs.groupby("occupancystatus")
    .agg(
        observed_runs=("run_id", "count"),
        median_observed_span_minutes=("observed_span_minutes", "median"),
        mean_observed_span_minutes=("observed_span_minutes", "mean"),
        p90_observed_span_minutes=(
            "observed_span_minutes",
            lambda x: x.quantile(0.90),
        ),
    )
    .reset_index()
)

duration_summary.to_csv(
    OUTPUT_DIR / "observed_state_run_summary.csv",
    index=False,
)

print("\nOBSERVED STATE-RUN SUMMARY")
print(
    duration_summary.to_string(
        index=False,
        float_format="%.2f",
    )
)

runs.to_csv(
    OUTPUT_DIR / "observed_state_runs.csv",
    index=False,
)

# ============================================================
# 6. SAVE AGGREGATED DATA
# ============================================================

data.to_csv(
    OUTPUT_DIR / "occupancy_5min_intervals.csv",
    index=False,
)

print("\nSaved outputs to:")
print(OUTPUT_DIR)

print("\nImportant:")
print("- Five-minute aggregation simplifies irregular source records.")
print("- State-run durations are observed spans, not true durations.")
print("- Long gaps are not treated as continuous state persistence.")
print("- No environmental room mapping is assumed.")
print("\nAnalysis complete.")
