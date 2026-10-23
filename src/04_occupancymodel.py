
"""
File 04: Occupancy Model
MMA3001 Room Occupancy Project

This is an initial rule-based occupancy estimator.
It does not establish the true number of occupants.

Run from the project root:
    python src/04_occupancy_model.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "engineered_sensor_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("MMA3001 - OCCUPANCY ESTIMATION")
print("=" * 60)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE}. "
        "Run File 03 first."
    )

df = pd.read_csv(INPUT_FILE)

if "timestamp" not in df.columns:
    raise ValueError("Missing timestamp column.")

if "sensorid" not in df.columns:
    raise ValueError("Missing sensorid column.")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    utc=True
)

df = df.dropna(subset=["timestamp"]).copy()

df = df.sort_values(
    ["sensorid", "timestamp"]
).reset_index(drop=True)

print(f"\nRows loaded: {len(df)}")
print(f"Sensors found: {df['sensorid'].nunique()}")


# ============================================================
# 3. IDENTIFY CO2 COLUMN
# ============================================================

co2_candidates = [
    column for column in df.columns
    if "carbon dioxide" in column.lower()
    or "co2" in column.lower()
    or "co₂" in column.lower()
]

if not co2_candidates:
    raise ValueError(
        "Could not identify the CO2 column. "
        "Check the measurement column names in the CSV."
    )

co2_column = co2_candidates[0]

df[co2_column] = pd.to_numeric(
    df[co2_column],
    errors="coerce"
)

print(f"CO2 column: {co2_column}")


# ============================================================
# 4. DEFINE INITIAL CO2 THRESHOLDS
# ============================================================

# These are provisional heuristic thresholds, not validated
# occupancy thresholds for this particular room.
#
# CO2 concentration depends on outdoor CO2, ventilation,
# room size, sensor location and occupant activity.

LOW_CO2 = 600
MODERATE_CO2 = 800
HIGH_CO2 = 1000


def classify_occupancy(co2):
    """Estimate an occupancy state from a CO2 measurement."""

    if pd.isna(co2):
        return "Unknown"

    if co2 < LOW_CO2:
        return "Low CO2"

    if co2 < MODERATE_CO2:
        return "Possible occupancy"

    if co2 < HIGH_CO2:
        return "Elevated CO2"

    return "High CO2"


df["occupancy_estimate"] = df[co2_column].apply(
    classify_occupancy
)

# Numeric state for plotting only.
state_mapping = {
    "Unknown": -1,
    "Low CO2": 0,
    "Possible occupancy": 1,
    "Elevated CO2": 2,
    "High CO2": 3,
}

df["occupancy_state"] = (
    df["occupancy_estimate"].map(state_mapping)
)


# ============================================================
# 5. SUMMARISE ESTIMATED STATES
# ============================================================

print("\nEstimated state counts:")

state_counts = (
    df["occupancy_estimate"]
    .value_counts(dropna=False)
    .reindex(
        [
            "Unknown",
            "Low CO2",
            "Possible occupancy",
            "Elevated CO2",
            "High CO2",
        ],
        fill_value=0
    )
)

print(state_counts.to_string())

state_percentages = (
    state_counts / max(len(df), 1) * 100
).round(2)

summary = pd.DataFrame({
    "reading_count": state_counts,
    "percentage": state_percentages,
})

summary.to_csv(
    OUTPUT_DIR / "occupancy_state_summary.csv"
)


# ============================================================
# 6. INSPECT OTHER ENVIRONMENTAL MEASUREMENTS
# ============================================================

print("\nEnvironmental measurements by estimated state:")

context_candidates = [
    column for column in df.columns
    if any(
        term in column.lower()
        for term in [
            "temperature",
            "humidity",
            "volatile organic",
            "voc",
            "formaldehyde",
            "light",
        ]
    )
    and not any(
        term in column.lower()
        for term in [
            "change",
            "rolling",
            "previous",
            "per_minute",
            "difference",
        ]
    )
]

context_columns = []

for column in context_candidates:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    if df[column].notna().any():
        context_columns.append(column)

if context_columns:
    context_summary = (
        df.groupby("occupancy_estimate")[context_columns]
        .mean(numeric_only=True)
        .round(3)
    )

    print(context_summary.to_string())

    context_summary.to_csv(
        OUTPUT_DIR / "environment_by_estimated_state.csv"
    )
else:
    print("No additional environmental measurements were identified.")


# ============================================================
# 7. PLOT CO2 AND ESTIMATED STATES
# ============================================================

for sensor_id, sensor_df in df.groupby("sensorid"):
    sensor_df = sensor_df.sort_values("timestamp")

    if sensor_df[co2_column].notna().sum() == 0:
        continue

    fig, axes = plt.subplots(
        2, 1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    axes[0].plot(
        sensor_df["timestamp"],
        sensor_df[co2_column],
        marker=".",
        linestyle="-",
        label="Measured CO2"
    )

    axes[0].axhline(
        LOW_CO2,
        linestyle="--",
        label=f"{LOW_CO2} ppm threshold"
    )

    axes[0].axhline(
        MODERATE_CO2,
        linestyle="--",
        label=f"{MODERATE_CO2} ppm threshold"
    )

    axes[0].axhline(
        HIGH_CO2,
        linestyle="--",
        label=f"{HIGH_CO2} ppm threshold"
    )

    axes[0].set_ylabel("CO2 concentration (ppm)")
    axes[0].set_title(f"CO2 and Estimated Occupancy States: Sensor {sensor_id}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].step(
        sensor_df["timestamp"],
        sensor_df["occupancy_state"],
        where="post"
    )

    axes[1].set_yticks(
        [-1, 0, 1, 2, 3],
        [
            "Unknown",
            "Low CO2",
            "Possible occupancy",
            "Elevated CO2",
            "High CO2",
        ]
    )

    axes[1].set_xlabel("Timestamp (UTC)")
    axes[1].set_ylabel("Estimated state")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    safe_sensor_id = str(sensor_id).replace("/", "_")

    fig.savefig(
        OUTPUT_DIR / f"occupancy_sensor_{safe_sensor_id}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 8. SAVE ESTIMATES
# ============================================================

output_file = OUTPUT_DIR / "occupancy_estimates.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("OCCUPANCY ESTIMATION COMPLETE")
print("=" * 60)

print(f"\nEstimates saved to:\n{output_file}")

print(
    "\nImportant: These are CO2-based heuristic states, "
    "not confirmed occupancy labels or a count of people."
)
