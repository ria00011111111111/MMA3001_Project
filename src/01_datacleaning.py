
import json
import zipfile
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT / "data" / "5EnvSensor_MayToDec2024_180kRows.zip"
)

OUTPUT_DIR = PROJECT_ROOT / "data"
DIAGNOSTICS_DIR = PROJECT_ROOT / "outputs" / "diagnostics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print("STEP 1: LOADING DATA")
print("=" * 65)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Check that the ZIP file is inside the data folder."
    )

print("Input file:", DATA_FILE)

with zipfile.ZipFile(DATA_FILE, "r") as archive:
    csv_files = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    ]

    if not csv_files:
        raise ValueError("No CSV file found inside the ZIP archive.")

    if len(csv_files) > 1:
        print("CSV files found:", csv_files)
        print("Using:", csv_files[0])

    with archive.open(csv_files[0]) as file:
        df = pd.read_csv(file)

print("Rows loaded:", len(df))
print("Columns:", df.columns.tolist())

required_columns = {"sensorid", "jsondata", "createdate"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

df["source_row"] = range(1, len(df) + 1)

# Parse record creation times as UTC.
df["createdate_parsed"] = pd.to_datetime(
    df["createdate"],
    errors="coerce",
    utc=True,
)


# ============================================================
# 3. EXTRACT SENSOR MEASUREMENTS
# ============================================================

print("\n" + "=" * 65)
print("STEP 2: EXTRACTING SENSOR MEASUREMENTS")
print("=" * 65)


def extract_measurements(json_value):
    """
    Extract measurements from the dataset's JSON list.

    Each list item contains:
        variable: {"name": "..."}
        values: [{"time": ..., "value": ...}]
    """

    if pd.isna(json_value):
        return {"total_json_readings": 0}

    try:
        data = json.loads(json_value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"total_json_readings": 0}

    if not isinstance(data, list):
        return {"total_json_readings": 0}

    result = {}
    total_readings = 0

    for measurement in data:
        if not isinstance(measurement, dict):
            continue

        variable = measurement.get("variable", {})

        if not isinstance(variable, dict):
            continue

        name = variable.get("name")

        if not name:
            continue

        values = measurement.get("values", [])

        if not isinstance(values, list):
            continue

        total_readings += len(values)

        if not values:
            result[name] = None
            result[f"{name} time"] = None
            result[f"{name} number_of_values"] = 0
            continue

        first = values[0]

        if not isinstance(first, dict):
            continue

        result[name] = first.get("value")
        result[f"{name} time"] = first.get("time")
        result[f"{name} number_of_values"] = len(values)

    result["total_json_readings"] = total_readings

    return result


extracted = df["jsondata"].apply(extract_measurements)

measurements_df = pd.DataFrame(
    extracted.tolist(),
    index=df.index,
)

df = pd.concat(
    [df.drop(columns=["jsondata"]), measurements_df],
    axis=1,
)

print("Measurement fields extracted:", len(measurements_df.columns))
print("Measurement columns:", [
    column for column in measurements_df.columns
    if not column.endswith(" time")
    and not column.endswith(" number_of_values")
    and column != "total_json_readings"
])
print("Rows after extraction:", len(df))


# ============================================================
# 4. CONVERT TIMESTAMPS TO UTC
# ============================================================

print("\n" + "=" * 65)
print("STEP 3: CONVERTING TIMESTAMPS")
print("=" * 65)


def convert_timestamp(series):
    """Convert Unix timestamps in seconds or date strings to UTC."""

    numeric = pd.to_numeric(series, errors="coerce")

    result = pd.to_datetime(
        numeric,
        unit="s",
        errors="coerce",
        utc=True,
    )

    non_numeric = numeric.isna() & series.notna()

    if non_numeric.any():
        result.loc[non_numeric] = pd.to_datetime(
            series.loc[non_numeric],
            errors="coerce",
            utc=True,
            format="mixed",
        )

    return pd.to_datetime(result, errors="coerce", utc=True)


time_columns = [
    column for column in df.columns
    if column.endswith(" time")
]

for column in time_columns:
    df[column] = convert_timestamp(df[column])

if "Carbon dioxide time" not in df.columns:
    raise ValueError(
        "Carbon dioxide time was not extracted. "
        "Check the JSON structure and measurement names."
    )

# Create the primary timestamp before using it elsewhere.
df["timestamp"] = df["Carbon dioxide time"]

print("Timestamp columns converted:", len(time_columns))
print("Valid CO2 timestamps:", int(df["timestamp"].notna().sum()))
print("Missing CO2 timestamps:", int(df["timestamp"].isna().sum()))


# ============================================================
# 5. CONVERT MEASUREMENTS TO NUMERIC
# ============================================================

print("\n" + "=" * 65)
print("STEP 4: CONVERTING MEASUREMENTS")
print("=" * 65)

metadata_columns = {
    "id",
    "sensorid",
    "devicetype",
    "status",
    "processing_errors",
    "createdate",
    "processdate",
    "source_row",
    "createdate_parsed",
    "timestamp",
}

for column in df.columns:
    if (
        column not in metadata_columns
        and not column.endswith(" time")
        and not column.endswith(" number_of_values")
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce")

if "total_json_readings" not in df.columns:
    df["total_json_readings"] = 0

df["json_has_no_readings"] = (
    pd.to_numeric(df["total_json_readings"], errors="coerce")
    .fillna(0)
    .eq(0)
)


# ============================================================
# 6. CHECK MISSING AND IMPLAUSIBLE VALUES
# ============================================================

print("\n" + "=" * 65)
print("STEP 5: CHECKING MEASUREMENT QUALITY")
print("=" * 65)

# Broad screening ranges, not manufacturer-specific specifications.
# Flag values instead of deleting records automatically.
valid_ranges = {
    "Carbon dioxide": (250, 10000),
    "Temperature": (-20, 60),
    "Humidity": (0, 100),
    "Pressure": (300, 1200),
    "Battery": (0, 6),
    "Formaldehyde": (0, 1000),
    "Light Volatile Organic Compounds": (0, 10000),
    "Particulate matter 1": (0, 2000),
    "Particulate matter 2.5": (0, 2000),
    "Particulate matter 4": (0, 2000),
    "Particulate matter 10": (0, 2000),
}

measurement_columns = [
    column for column in valid_ranges
    if column in df.columns
]

missing_rows = []

for column in measurement_columns:
    lower, upper = valid_ranges[column]

    df[f"{column}_missing"] = df[column].isna()

    df[f"{column}_out_of_range"] = (
        df[column].notna()
        & ~df[column].between(lower, upper)
    )

    # Retain the original measurement; mask only in the screened column.
    df[f"{column}_clean"] = df[column].mask(
        df[f"{column}_out_of_range"]
    )

    missing_rows.append({
        "measurement": column,
        "missing_count": int(df[column].isna().sum()),
        "missing_percentage": float(df[column].isna().mean() * 100),
        "out_of_range_count": int(
            df[f"{column}_out_of_range"].sum()
        ),
    })

missing_summary = pd.DataFrame(missing_rows)

missing_summary.to_csv(
    DIAGNOSTICS_DIR / "missing_and_invalid_values.csv",
    index=False,
)

print(missing_summary.to_string(index=False))


# ============================================================
# 7. CHECK TIMESTAMP RELIABILITY
# ============================================================

print("\n" + "=" * 65)
print("STEP 6: CHECKING TIMESTAMP RELIABILITY")
print("=" * 65)

# Both timestamp columns are timezone-aware UTC.
df["timestamp_difference_minutes"] = (
    df["timestamp"] - df["createdate_parsed"]
).dt.total_seconds() / 60

df["timestamp_missing"] = df["timestamp"].isna()
df["createdate_missing"] = df["createdate_parsed"].isna()

df["timestamp_mismatch_over_60min"] = (
    df["timestamp_difference_minutes"].abs() > 60
)

timestamp_summary = (
    df.groupby("sensorid", dropna=False)
    .agg(
        rows=("source_row", "size"),
        unique_measurement_timestamps=("timestamp", "nunique"),
        missing_measurement_timestamps=("timestamp_missing", "sum"),
        missing_creation_timestamps=("createdate_missing", "sum"),
        mismatches_over_60min=("timestamp_mismatch_over_60min", "sum"),
        median_difference_minutes=("timestamp_difference_minutes", "median"),
    )
    .reset_index()
)

timestamp_summary.to_csv(
    DIAGNOSTICS_DIR / "timestamp_quality_summary.csv",
    index=False,
)

timestamp_issues = df.loc[
    df["timestamp_missing"]
    | df["createdate_missing"]
    | df["timestamp_mismatch_over_60min"],
    [
        "source_row",
        "sensorid",
        "createdate",
        "createdate_parsed",
        "timestamp",
        "timestamp_difference_minutes",
        "timestamp_missing",
        "createdate_missing",
        "timestamp_mismatch_over_60min",
    ],
]

timestamp_issues.to_csv(
    DIAGNOSTICS_DIR / "timestamp_issues.csv",
    index=False,
)

print(timestamp_summary.to_string(index=False))
print("Rows flagged for timestamp review:", len(timestamp_issues))


# ============================================================
# 8. FLAG EXACT DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 65)
print("STEP 7: CHECKING EXACT DUPLICATES")
print("=" * 65)

duplicate_check_columns = [
    column for column in df.columns
    if column != "source_row"
]

df["exact_duplicate_record"] = df.duplicated(
    subset=duplicate_check_columns,
    keep="first",
)

exact_duplicates = df.loc[df["exact_duplicate_record"]].copy()

exact_duplicates.to_csv(
    DIAGNOSTICS_DIR / "exact_duplicate_records.csv",
    index=False,
)

print("Exact duplicate records:", int(df["exact_duplicate_record"].sum()))


# ============================================================
# 9. FLAG REPEATED TIMESTAMPS AND READINGS
# ============================================================

print("\n" + "=" * 65)
print("STEP 8: CHECKING REPEATED SENSOR READINGS")
print("=" * 65)

df["repeated_sensor_timestamp"] = (
    df["sensorid"].notna()
    & df["timestamp"].notna()
    & df.duplicated(
        subset=["sensorid", "timestamp"],
        keep=False,
    )
)

comparison_columns = [
    column for column in [
        "Carbon dioxide",
        "Temperature",
        "Humidity",
        "Pressure",
        "Battery",
    ]
    if column in df.columns
]

df = df.sort_values(
    ["sensorid", "createdate_parsed", "source_row"],
    na_position="last",
).reset_index(drop=True)

df["adjacent_identical_readings"] = False

if comparison_columns:
    previous_values = (
        df.groupby("sensorid", dropna=False)[comparison_columns]
        .shift(1)
    )

    current_values = df[comparison_columns]

    comparable = (
        current_values.notna() & previous_values.notna()
    ).any(axis=1)

    equal_values = (
        current_values.eq(previous_values)
        | (current_values.isna() & previous_values.isna())
    )

    identical = equal_values.all(axis=1)

    df["adjacent_identical_readings"] = comparable & identical

repeat_summary = (
    df.groupby("sensorid", dropna=False)
    .agg(
        rows=("source_row", "size"),
        unique_CO2_timestamps=("timestamp", "nunique"),
        unique_CO2_values=("Carbon dioxide", "nunique"),
        repeated_timestamp_rows=("repeated_sensor_timestamp", "sum"),
        adjacent_identical_readings=("adjacent_identical_readings", "sum"),
    )
    .reset_index()
)

repeat_summary.to_csv(
    DIAGNOSTICS_DIR / "repeated_reading_summary.csv",
    index=False,
)

repeated_rows = df.loc[
    df["repeated_sensor_timestamp"]
    | df["adjacent_identical_readings"]
].copy()

repeated_rows.to_csv(
    DIAGNOSTICS_DIR / "repeated_reading_records.csv",
    index=False,
)

print(repeat_summary.to_string(index=False))


# ============================================================
# 10. SAVE CLEANED DATA AND SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("STEP 9: SAVING CLEANED DATA")
print("=" * 65)

df = df.sort_values(
    ["sensorid", "timestamp", "createdate_parsed", "source_row"],
    na_position="last",
).reset_index(drop=True)

cleaned_file = OUTPUT_DIR / "cleaned_sensor_data.csv"
df.to_csv(cleaned_file, index=False)

run_summary = pd.DataFrame([{
    "input_file": DATA_FILE.name,
    "rows_loaded": len(df),
    "number_of_sensors": df["sensorid"].nunique(),
    "missing_CO2_timestamps": int(df["timestamp_missing"].sum()),
    "timestamp_mismatches_over_60min": int(
        df["timestamp_mismatch_over_60min"].sum()
    ),
    "exact_duplicate_records": int(df["exact_duplicate_record"].sum()),
    "repeated_timestamp_rows": int(df["repeated_sensor_timestamp"].sum()),
    "adjacent_identical_readings": int(
        df["adjacent_identical_readings"].sum()
    ),
    "cleaned_file": str(cleaned_file),
}])

run_summary.to_csv(
    DIAGNOSTICS_DIR / "cleaning_run_summary.csv",
    index=False,
)

print("Cleaned dataset saved to:", cleaned_file)
print("Diagnostic reports saved to:", DIAGNOSTICS_DIR)
print("\nRun summary:")
print(run_summary.to_string(index=False))

print("\nDATA CLEANING COMPLETE")
