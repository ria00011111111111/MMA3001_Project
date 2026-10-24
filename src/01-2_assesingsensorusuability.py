
import json
import zipfile
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = (
    ROOT / "data" / "5EnvSensor_MayToDec2024_180kRows.zip"
)

OUTPUT_DIR = ROOT / "outputs" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the original dataset
# --------------------------------------------------

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    csv_files = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError("No CSV file found inside the ZIP.")

    with archive.open(csv_files[0]) as file:
        raw = pd.read_csv(file)

print("Original rows:", len(raw))


# --------------------------------------------------
# 3. Extract measurement values and timestamps
# --------------------------------------------------

def extract_measurements(json_text):
    result = {}

    if pd.isna(json_text):
        return result

    try:
        measurements = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        return result

    if not isinstance(measurements, list):
        return result

    for measurement in measurements:
        name = measurement.get("variable", {}).get("name")
        values = measurement.get("values", [])

        if not name or not values:
            continue

        first = values[0]

        result[name] = first.get("value")
        result[f"{name}_time"] = first.get("time")

    return result


expanded = raw["jsondata"].apply(
    extract_measurements
).apply(pd.Series)

df = pd.concat(
    [raw[["id", "sensorid", "createdate"]], expanded],
    axis=1
)


# --------------------------------------------------
# 4. Convert measurement timestamps
# --------------------------------------------------

df["timestamp"] = pd.to_datetime(
    pd.to_numeric(
        df.get("Carbon dioxide_time"),
        errors="coerce"
    ),
    unit="s",
    errors="coerce",
    utc=True
)

variables = [
    "Carbon dioxide",
    "Temperature",
    "Humidity",
    "Pressure",
    "Battery",
]

variables = [
    column for column in variables
    if column in df.columns
]

for column in variables:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# --------------------------------------------------
# 5. Check whether readings change
# --------------------------------------------------

variation_results = []

for sensor_id, group in df.groupby("sensorid"):
    # Preserve the order in which records were stored.
    group = group.reset_index(drop=True)

    for variable in variables:
        values = group[variable]
        valid = values.dropna()

        # Compare adjacent valid records in original row order.
        changes = valid.ne(valid.shift()).iloc[1:]

        variation_results.append({
            "sensorid": sensor_id,
            "variable": variable,
            "valid_readings": len(valid),
            "unique_readings": valid.nunique(),
            "fraction_of_adjacent_readings_that_change": (
                changes.mean() if len(changes) else None
            ),
        })

variation = pd.DataFrame(variation_results)

variation.to_csv(
    OUTPUT_DIR / "sensor_value_variation.csv",
    index=False
)


# --------------------------------------------------
# 6. Examine timestamp gaps
# --------------------------------------------------

gap_results = []

for sensor_id, group in df.groupby("sensorid"):
    times = (
        group["timestamp"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    gaps = (
        times.diff()
        .dt.total_seconds()
        .div(60)
        .dropna()
    )

    gap_results.append({
        "sensorid": sensor_id,
        "original_rows": len(group),
        "unique_timestamps": times.nunique(),
        "median_gap_minutes": gaps.median(),
        "maximum_gap_minutes": gaps.max(),
        "gaps_over_1_hour": int((gaps > 60).sum()),
        "gaps_over_1_day": int((gaps > 1440).sum()),
    })

gap_summary = pd.DataFrame(gap_results)

gap_summary.to_csv(
    OUTPUT_DIR / "sensor_timestamp_gaps.csv",
    index=False
)


# --------------------------------------------------
# 7. Count records sharing timestamps
# --------------------------------------------------

timestamp_counts = (
    df.groupby("sensorid")
    .agg(
        original_rows=("id", "size"),
        unique_timestamps=("timestamp", "nunique"),
    )
    .reset_index()
)

timestamp_counts["rows_per_unique_timestamp"] = (
    timestamp_counts["original_rows"]
    / timestamp_counts["unique_timestamps"].replace(0, pd.NA)
)

timestamp_counts.to_csv(
    OUTPUT_DIR / "sensor_row_reduction.csv",
    index=False
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\nREADING VARIATION")
print(variation.to_string(index=False))

print("\nTIMESTAMP GAPS")
print(gap_summary.to_string(index=False))

print("\nREPEATED TIMESTAMP SUMMARY")
print(timestamp_counts.to_string(index=False))

print("\nSaved diagnostic files:")
print(OUTPUT_DIR / "sensor_value_variation.csv")
print(OUTPUT_DIR / "sensor_timestamp_gaps.csv")
print(OUTPUT_DIR / "sensor_row_reduction.csv")
