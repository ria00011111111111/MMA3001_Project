
"""
File 03: Feature Engineering
MMA3001 Room Occupancy Project

Purpose:
- Load cleaned sensor data
- Create time-based features
- Calculate changes in environmental readings
- Calculate rolling statistics for each sensor
- Save the engineered dataset for modelling

Run from the project root:
    python src/03_feature_engineering.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "cleaned_sensor_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "engineered_sensor_data.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD THE CLEANED DATA
# ============================================================

print("\n" + "=" * 65)
print("MMA3001 - FEATURE ENGINEERING")
print("=" * 65)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find the cleaned dataset:\n{INPUT_FILE}\n"
        "Run 01_datacleaning.py first."
    )

df = pd.read_csv(INPUT_FILE)

print("\nOriginal dataset:")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 3. CHECK AND PREPARE THE TIMESTAMP
# ============================================================

if "timestamp" not in df.columns:
    raise ValueError(
        "The dataset has no 'timestamp' column. "
        "Check the column names in cleaned_sensor_data.csv."
    )

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    utc=True
)

invalid_timestamps = df["timestamp"].isna().sum()

print(f"\nRows with invalid timestamps: {invalid_timestamps}")

# Invalid timestamps cannot be used for time-based features.
df = df.dropna(subset=["timestamp"]).copy()

# Ensure the data is in chronological order within each sensor.
if "sensorid" in df.columns:
    df = df.sort_values(
        ["sensorid", "timestamp"]
    ).reset_index(drop=True)
else:
    raise ValueError(
        "The dataset has no 'sensorid' column. "
        "Sensor-specific features require sensor IDs."
    )

print(f"Rows remaining: {len(df)}")


# ============================================================
# 4. IDENTIFY ENVIRONMENTAL MEASUREMENTS
# ============================================================

# Only use columns that actually exist in the dataset.
# IDs, timestamps and descriptive metadata are not measurements.

excluded_columns = {
    "id",
    "sensorid",
    "timestamp",
    "createdate",
    "processdate",
    "devicetype",
    "status",
    "processing_errors",
    "jsondata",
}

measurement_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
    and pd.api.types.is_numeric_dtype(df[column])
]

print("\nEnvironmental measurement columns:")
for column in measurement_columns:
    print(f"- {column}")

if not measurement_columns:
    raise ValueError(
        "No numeric measurement columns were found. "
        "Check the cleaned dataset and its data types."
    )


# ============================================================
# 5. CREATE TIME-BASED FEATURES
# ============================================================

print("\nCreating time-based features...")

# These features describe when each measurement was recorded.
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["day_of_month"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

# Monday = 0, Sunday = 6.
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# Broad time-of-day categories.
df["time_of_day"] = pd.cut(
    df["hour"],
    bins=[-1, 5, 11, 16, 20, 23],
    labels=[
        "night",
        "morning",
        "afternoon",
        "evening",
        "late_night",
    ],
    ordered=False,
)

# Cyclical features preserve the fact that 23:00 and 00:00
# are close together in time.
df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

df["day_of_week_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["day_of_week_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)


# ============================================================
# 6. CALCULATE TIME BETWEEN READINGS
# ============================================================

print("Calculating time intervals...")

df["time_since_previous_minutes"] = (
    df.groupby("sensorid")["timestamp"]
    .diff()
    .dt.total_seconds()
    / 60
)

# Negative or zero intervals indicate repeated timestamps
# or a data-quality issue, so mark them as missing.
df.loc[
    df["time_since_previous_minutes"] <= 0,
    "time_since_previous_minutes"
] = np.nan


# ============================================================
# 7. CREATE LAG AND CHANGE FEATURES
# ============================================================

print("Creating previous-reading and change features...")

# Grouping by sensor prevents one sensor's reading from
# being used as another sensor's previous reading.

for column in measurement_columns:
    previous_column = f"{column}_previous"

    df[previous_column] = (
        df.groupby("sensorid")[column].shift(1)
    )

    # Absolute change from the previous recorded reading.
    df[f"{column}_change"] = (
        df[column] - df[previous_column]
    )

    # Rate of change per minute.
    # This is useful when readings have irregular time intervals.
    df[f"{column}_change_per_minute"] = (
        df[f"{column}_change"]
        / df["time_since_previous_minutes"].replace(0, np.nan)
    )


# ============================================================
# 8. CREATE ROLLING STATISTICS
# ============================================================

print("Calculating rolling statistics...")

# Use the previous three readings, excluding the current reading.
# This avoids including the current measurement in its own
# historical average.
#
# These are reading-based windows, not fixed-duration windows.
# If sampling intervals vary significantly, interpret them carefully.

for column in measurement_columns:
    grouped_measurements = df.groupby("sensorid")[column]

    df[f"{column}_rolling_mean_3"] = (
        grouped_measurements
        .transform(
            lambda series: series.shift(1).rolling(
                window=3,
                min_periods=2
            ).mean()
        )
    )

    df[f"{column}_rolling_std_3"] = (
        grouped_measurements
        .transform(
            lambda series: series.shift(1).rolling(
                window=3,
                min_periods=2
            ).std()
        )
    )

    # Difference from the recent historical average.
    df[f"{column}_difference_from_rolling_mean"] = (
        df[column] - df[f"{column}_rolling_mean_3"]
    )


# ============================================================
# 9. CREATE CO2-SPECIFIC FEATURES IF AVAILABLE
# ============================================================

# Identify a carbon dioxide column without assuming its exact name.
co2_candidates = [
    column
    for column in measurement_columns
    if "carbon dioxide" in column.lower()
    or "co2" in column.lower()
    or "co₂" in column.lower()
]

if co2_candidates:
    co2_column = co2_candidates[0]

    print(f"\nCreating additional CO2 features using: {co2_column}")

    # Positive values indicate increasing CO2.
    df["co2_increasing"] = (
        df[f"{co2_column}_change"] > 0
    ).astype("Int64")

    # Indicates a relatively large increase between consecutive readings.
    # This is a data-driven flag, not proof of room occupancy.
    change_threshold = (
        df[f"{co2_column}_change"].abs().quantile(0.95)
    )

    df["co2_large_change"] = (
        df[f"{co2_column}_change"].abs()
        >= change_threshold
    ).astype("Int64")

    print(
        "95th percentile of absolute CO2 change: "
        f"{change_threshold:.3f}"
    )

else:
    print(
        "\nNo CO2 column was identified. "
        "The general environmental features were still created."
    )


# ============================================================
# 10. CHECK THE ENGINEERED DATASET
# ============================================================

print("\n" + "=" * 65)
print("FEATURE ENGINEERING CHECKS")
print("=" * 65)

print(f"\nFinal number of rows: {len(df)}")
print(f"Final number of columns: {len(df.columns)}")

print("\nFirst five rows:")
print(df.head().to_string())

print("\nMissing values in newly created features:")

original_columns = set(
    pd.read_csv(INPUT_FILE, nrows=0).columns
)

new_feature_columns = [
    column
    for column in df.columns
    if column not in original_columns
]

if new_feature_columns:
    missing_features = (
        df[new_feature_columns]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(missing_features.to_string())
else:
    print("No new features were created.")

print("\nRepeated sensor/timestamp combinations:")

repeated = df.duplicated(
    subset=["sensorid", "timestamp"],
    keep=False
)

print(f"Rows involved in repeated combinations: {repeated.sum()}")


# ============================================================
# 11. SAVE THE ENGINEERED DATASET
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 65)

print(f"\nSaved engineered data to:\n{OUTPUT_FILE}")

print(
    "\nNext step: use this dataset in "
    "04_occupancy_model.py."
)
