
import pandas as pd
import json
from pathlib import Path

# ==================================================
# 1. LOAD DATA
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "5EnvSensor_MayToDec2024_180kRows.zip"
OUTPUT_FILE = ROOT / "data" / "cleaned_sensor_data.csv"
DIAGNOSTICS_DIR = ROOT / "outputs" / "data_quality"

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Input data not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, low_memory=False)

print("=" * 65)
print("MMA3001 - DATA CLEANING")
print("=" * 65)
print("Original dataset shape:", df.shape)
print("Original columns:", df.columns.tolist())


# ==================================================
# 2. INSPECT RAW JSON
# ==================================================

print("\n" + "=" * 65)
print("RAW JSON INSPECTION")
print("=" * 65)

sensor_rows = df[
    df["sensorid"].astype(str) == "6012002000227"
]

for row_index in sensor_rows.index[:2]:
    row = df.loc[row_index]

    print(f"\nRecord ID: {row.get('id')}")
    print(f"Sensor ID: {row.get('sensorid')}")
    print(f"Database creation time: {row.get('createdate')}")
    print(f"Processing time: {row.get('processdate')}")

    try:
        raw_json = json.loads(row["jsondata"])
        print(json.dumps(raw_json, indent=2))
    except (json.JSONDecodeError, TypeError):
        print("Could not parse raw JSON.")


# ==================================================
# 3. REMOVE EXACT DUPLICATE ROWS
# ==================================================

before = len(df)
df = df.drop_duplicates().copy()

print("\nExact duplicate rows removed:", before - len(df))


# ==================================================
# 4. EXTRACT SENSOR READINGS
# ==================================================

def extract_sensor_data(json_string):
    if pd.isna(json_string):
        return {}

    try:
        readings = json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return {}

    if not isinstance(readings, list):
        return {}

    extracted = {}

    for item in readings:
        if not isinstance(item, dict):
            continue

        variable = item.get("variable", {})
        values = item.get("values", [])

        if not isinstance(variable, dict):
            continue

        name = variable.get("name")

        if not name or not isinstance(values, list):
            continue

        extracted[name + " number_of_values"] = len(values)

        if not values:
            extracted[name] = None
            extracted[name + " unit"] = variable.get("unit")
            extracted[name + " time"] = None
            continue

        reading = values[0]

        if not isinstance(reading, dict):
            extracted[name] = None
            extracted[name + " unit"] = variable.get("unit")
            extracted[name + " time"] = None
            continue

        extracted[name] = (
            None
            if reading.get("errorCode") is not None
            else reading.get("value")
        )

        extracted[name + " unit"] = variable.get("unit")
        extracted[name + " time"] = reading.get("time")

    return extracted


sensor_data = df["jsondata"].apply(extract_sensor_data)
sensor_data = pd.DataFrame(sensor_data.tolist(), index=df.index)

df = pd.concat(
    [df.drop(columns=["jsondata"]), sensor_data],
    axis=1
)


# ==================================================
# 5. CONVERT TIMESTAMPS
# ==================================================

time_columns = [
    column for column in df.columns
    if column.endswith(" time")
]

for column in time_columns:
    raw = df[column]
    numeric = pd.to_numeric(raw, errors="coerce")

    numeric_mask = raw.notna() & numeric.notna()

    converted = pd.Series(
        pd.NaT,
        index=df.index,
        dtype="datetime64[ns, UTC]"
    )

    # Assumes numeric sensor timestamps are Unix seconds.
    converted.loc[numeric_mask] = pd.to_datetime(
        numeric.loc[numeric_mask],
        unit="s",
        errors="coerce",
        utc=True
    )

    string_mask = raw.notna() & ~numeric_mask

    if string_mask.any():
        converted.loc[string_mask] = pd.to_datetime(
            raw.loc[string_mask],
            errors="coerce",
            utc=True,
            format="mixed"
        )

    df[column] = converted


# ==================================================
# 6. TIMESTAMP DIAGNOSTICS
# ==================================================

print("\n" + "=" * 65)
print("TIMESTAMP DIAGNOSTICS")
print("=" * 65)

count_columns = [
    column for column in df.columns
    if column.endswith(" number_of_values")
]

if count_columns:
    count_summary = df[count_columns].describe()

    print("\nNumber of readings per variable:")
    print(count_summary.to_string())

    count_summary.to_csv(
        DIAGNOSTICS_DIR / "json_reading_count_summary.csv"
    )

timestamp_summary = []

for sensor_id, group in df.groupby("sensorid", dropna=False):
    times = group["Carbon dioxide time"].dropna()
    unique_times = times.nunique()

    timestamp_summary.append({
        "sensorid": sensor_id,
        "rows": len(group),
        "missing_timestamps": group["Carbon dioxide time"].isna().sum(),
        "unique_timestamps": unique_times,
        "repeated_timestamps": len(times) - unique_times,
        "first_timestamp": times.min() if len(times) else pd.NaT,
        "last_timestamp": times.max() if len(times) else pd.NaT,
        "timestamp_span_hours": (
            (times.max() - times.min()).total_seconds() / 3600
            if unique_times > 1
            else 0
        )
    })

timestamp_summary = pd.DataFrame(timestamp_summary)

print("\nPer-sensor timestamp summary:")
print(timestamp_summary.to_string(index=False))

timestamp_summary.to_csv(
    DIAGNOSTICS_DIR / "timestamp_summary.csv",
    index=False
)


# ==================================================
# 7. COMPARE SENSOR AND DATABASE TIMESTAMPS
# ==================================================

df["createdate_parsed"] = pd.to_datetime(
    df["createdate"],
    errors="coerce",
    utc=True,
    format="mixed"
)

df["co2_timestamp_minus_createdate_minutes"] = (
    df["Carbon dioxide time"] - df["createdate_parsed"]
).dt.total_seconds() / 60

print("\nCO2 timestamp minus createdate, in minutes:")
print(
    df["co2_timestamp_minus_createdate_minutes"]
    .describe()
    .to_string()
)

df["timestamp"] = df["Carbon dioxide time"]

timestamp_issues = df[
    df["timestamp"].isna()
    | df["createdate_parsed"].isna()
    | df["co2_timestamp_minus_createdate_minutes"].abs().gt(60)
].copy()

timestamp_issues.to_csv(
    DIAGNOSTICS_DIR / "timestamp_issues.csv",
    index=False
)

print("\nMissing or mismatched timestamp records:", len(timestamp_issues))


# ==================================================
# 8. CONVERT MEASUREMENTS TO NUMERIC
# ==================================================

measurement_columns = [
    "Battery",
    "Carbon dioxide",
    "Humidity",
    "Formaldehyde",
    "Pressure",
    "Temperature",
    "Light Volatile Organic Compounds",
    "Particulate matter 1",
    "Particulate matter 4",
    "Particulate matter 2.5",
    "Particulate matter 10"
]

for column in measurement_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ==================================================
# 9. FLAG OUT-OF-RANGE MEASUREMENTS
# ==================================================

valid_ranges = {
    "Battery": (0, 6),
    "Carbon dioxide": (0, 10000),
    "Humidity": (0, 100),
    "Formaldehyde": (0, 10000),
    "Pressure": (800, 1200),
    "Temperature": (-30, 60),
    "Light Volatile Organic Compounds": (0, 100000),
    "Particulate matter 1": (0, 2000),
    "Particulate matter 4": (0, 2000),
    "Particulate matter 2.5": (0, 2000),
    "Particulate matter 10": (0, 2000)
}

range_diagnostics = []

for column, (minimum, maximum) in valid_ranges.items():
    if column not in df.columns:
        continue

    invalid = df[column].notna() & (
        (df[column] < minimum)
        | (df[column] > maximum)
    )

    count = int(invalid.sum())

    print(f"{column}: {count} out-of-range readings")

    range_diagnostics.append({
        "variable": column,
        "minimum_allowed": minimum,
        "maximum_allowed": maximum,
        "out_of_range_count": count
    })

    df.loc[invalid, column] = float("nan")

pd.DataFrame(range_diagnostics).to_csv(
    DIAGNOSTICS_DIR / "range_diagnostics.csv",
    index=False
)


# ==================================================
# 10. MISSING VALUE SUMMARY
# ==================================================

missing_summary = pd.DataFrame({
    "column": df.columns,
    "missing_count": [
        int(df[column].isna().sum()) for column in df.columns
    ],
    "missing_percent": [
        df[column].isna().mean() * 100 for column in df.columns
    ]
})

print("\nMissing values:")
print(
    missing_summary[missing_summary["missing_count"] > 0]
    .to_string(index=False)
)

missing_summary.to_csv(
    DIAGNOSTICS_DIR / "missing_values_summary.csv",
    index=False
)


# ==================================================
# 11. SORT AND FIND REPEATED TIMESTAMPS
# ==================================================

df = df.sort_values(
    ["sensorid", "timestamp", "createdate_parsed"],
    na_position="last"
).reset_index(drop=True)

display_columns = [
    column for column in [
        "sensorid",
        "createdate",
        "timestamp",
        "Carbon dioxide",
        "Temperature",
        "Humidity"
    ]
    if column in df.columns
]

print("\nFirst 10 sorted readings:")
print(df[display_columns].head(10).to_string(index=False))

repeated = df.duplicated(
    subset=["sensorid", "timestamp"],
    keep=False
)

print("\nRows sharing a sensor ID and timestamp:", int(repeated.sum()))

df.loc[repeated].to_csv(
    DIAGNOSTICS_DIR / "repeated_sensor_timestamps.csv",
    index=False
)


# ==================================================
# 12. EXPORT CLEANED DATA
# ==================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("DATA CLEANING COMPLETE")
print("=" * 65)
print("Final dataset shape:", df.shape)
print("Cleaned dataset:", OUTPUT_FILE)
print("Diagnostics saved in:", DIAGNOSTICS_DIR)
