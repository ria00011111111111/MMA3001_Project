
"""
MMA3001 - Environmental Sensor Data Cleaning
--------------------------------------------
Extracts sensor measurements from the source ZIP, checks data quality,
removes unreliable records, identifies continuous time segments, and
produces a provisional dataset for later modelling.

IMPORTANT:
The current dataset does not contain verified occupancy labels.
This script prepares environmental sensor data; it does not prove
that room occupancy can already be predicted.
"""

from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DIAGNOSTICS_DIR = DATA_DIR / "diagnostics"

ZIP_PATH = DATA_DIR / "5EnvSensor_MayToDec2024_180kRows.zip"

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

MAX_TIMESTAMP_OFFSET_MIN = 10
MAX_TIME_GAP_MIN = 30

# A segment needs at least this many readings to be considered
# a potentially useful starting point for time-window modelling.
MIN_SEGMENT_RECORDS = 30

# Sensor periods identified as potentially reliable from the
# previous data-quality investigation. These are provisional.
ALLOWED_PERIODS = {
    "6012002000326": ("2025-09-01", "2026-05-01"),
    "6012002000869": ("2024-02-01", "2024-06-01"),
}

CORE_COLUMNS = [
    "Carbon dioxide",
    "Temperature",
    "Humidity",
    "Pressure",
]

# Broad physical plausibility ranges.
# These are data-quality checks, not official sensor specifications.
VALUE_RANGES = {
    "Battery": (0, 10),
    "Carbon dioxide": (250, 10000),
    "Humidity": (0, 100),
    "Pressure": (800, 1200),
    "Temperature": (-20, 80),
    "Formaldehyde": (0, 1000),
    "Light Volatile Organic Compounds": (0, 100000),
    "Particulate matter 1": (0, 10000),
    "Particulate matter 2.5": (0, 10000),
    "Particulate matter 4": (0, 10000),
    "Particulate matter 10": (0, 10000),
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def parse_jsondata(json_text):
    """
    Convert a JSON measurement list into:
        measurement name -> value
        measurement name -> Unix timestamp

    Each measurement uses the first value in its values list.
    Invalid JSON or missing measurements are handled safely.
    """

    measurements = {}
    timestamps = {}

    if pd.isna(json_text):
        return measurements, timestamps

    try:
        payload = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        return measurements, timestamps

    if isinstance(payload, dict):
        payload = [payload]

    if not isinstance(payload, list):
        return measurements, timestamps

    for item in payload:
        if not isinstance(item, dict):
            continue

        variable = item.get("variable", {})
        if not isinstance(variable, dict):
            continue

        name = variable.get("name")
        values = item.get("values", [])

        if not name or not isinstance(values, list) or not values:
            continue

        # Select the first available measurement in the JSON entry.
        observation = next(
            (
                value for value in values
                if isinstance(value, dict)
                and value.get("value") is not None
            ),
            None,
        )

        if observation is None:
            continue

        try:
            measurements[name] = float(observation["value"])
        except (TypeError, ValueError):
            measurements[name] = np.nan

        timestamps[name] = observation.get("time")

    return measurements, timestamps


def safe_filename(text):
    """Make a string suitable for use in a filename."""
    return "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(text)
    )


def write_csv(dataframe, path):
    """Save a CSV file and report its location."""
    dataframe.to_csv(path, index=False)
    print(f"Saved: {path}")


# ============================================================
# 3. LOAD SOURCE CSV FROM ZIP
# ============================================================

if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"Could not find the source ZIP:\n{ZIP_PATH}\n\n"
        "Check the ZIP filename and make sure it is inside the data folder."
    )

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    csv_files = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError("No CSV file was found inside the ZIP.")

    # Use the first CSV in the archive.
    source_csv = csv_files[0]

    with archive.open(source_csv) as file:
        raw = pd.read_csv(file, low_memory=False)

print("\n" + "=" * 65)
print("MMA3001 DATA CLEANING")
print("=" * 65)
print(f"Source ZIP: {ZIP_PATH.name}")
print(f"Source CSV: {source_csv}")
print(f"Original records: {len(raw):,}")


# ============================================================
# 4. CHECK REQUIRED SOURCE COLUMNS
# ============================================================

required_columns = {
    "sensorid",
    "jsondata",
    "createdate",
}

missing_columns = required_columns.difference(raw.columns)

if missing_columns:
    raise ValueError(
        f"The source CSV is missing required columns: {sorted(missing_columns)}"
    )

# Preserve a unique identifier for tracing records back to the source.
raw.insert(0, "source_row_id", np.arange(len(raw)))

# Standardise sensor identifiers as strings.
raw["sensorid"] = raw["sensorid"].astype("string").str.strip()


# ============================================================
# 5. EXTRACT MEASUREMENTS FROM JSON
# ============================================================

measurement_rows = []
measurement_time_rows = []

for json_text in raw["jsondata"]:
    values, timestamps = parse_jsondata(json_text)
    measurement_rows.append(values)
    measurement_time_rows.append(timestamps)

measurements_df = pd.DataFrame(measurement_rows)
timestamps_df = pd.DataFrame(measurement_time_rows)

# Join extracted values to source metadata.
data = pd.concat(
    [
        raw.drop(columns=["jsondata"], errors="ignore").reset_index(drop=True),
        measurements_df.reset_index(drop=True),
    ],
    axis=1,
)

# Prefer the CO2 measurement timestamp as the record's main timestamp.
# This matches the previous investigation, where CO2 was present
# in every source row.
if "Carbon dioxide" not in timestamps_df.columns:
    raise ValueError(
        "No Carbon dioxide timestamps were extracted. "
        "Inspect the JSON structure before continuing."
    )

data["measurement_timestamp_unix"] = pd.to_numeric(
    timestamps_df["Carbon dioxide"],
    errors="coerce",
)

data["created_at_utc"] = pd.to_datetime(
    data["createdate"],
    utc=True,
    errors="coerce",
)

data["measurement_time_utc"] = pd.to_datetime(
    data["measurement_timestamp_unix"],
    unit="s",
    utc=True,
    errors="coerce",
)

# Keep the original measurement timestamp for auditability.
data["timestamp_offset_minutes"] = (
    data["measurement_time_utc"] - data["created_at_utc"]
).dt.total_seconds().abs() / 60

data["timestamp_offset_known"] = (
    data["measurement_time_utc"].notna()
    & data["created_at_utc"].notna()
)

data["timestamp_offset_ok"] = (
    data["timestamp_offset_known"]
    & (data["timestamp_offset_minutes"] <= MAX_TIMESTAMP_OFFSET_MIN)
)


# ============================================================
# 6. NUMERIC CONVERSION AND MEASUREMENT QUALITY FLAGS
# ============================================================

for column in VALUE_RANGES:
    if column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    else:
        data[column] = np.nan

# Store a clean version of each measurement.
# Values outside the plausible range become NaN in *_clean columns.
for column, (lower, upper) in VALUE_RANGES.items():
    clean_column = f"{column}_clean"

    valid = data[column].notna() & data[column].between(lower, upper)

    data[clean_column] = data[column].where(valid)

    data[f"{safe_filename(column)}_out_of_range"] = (
        data[column].notna() & ~valid
    )

# Record missingness and quality flags for the core variables.
for column in CORE_COLUMNS:
    data[f"{safe_filename(column)}_missing"] = data[column].isna()

# A record must have the core variables needed for the initial analysis.
data["core_measurements_valid"] = data[
    [f"{column}_clean" for column in CORE_COLUMNS]
].notna().all(axis=1)


# ============================================================
# 7. EXACT DUPLICATE REMOVAL
# ============================================================

# Ignore source_row_id when testing for exact duplicates.
comparison_columns = [
    column for column in data.columns
    if column != "source_row_id"
]

exact_duplicate_mask = data.duplicated(
    subset=comparison_columns,
    keep="first",
)

exact_duplicates = data.loc[exact_duplicate_mask].copy()
data = data.loc[~exact_duplicate_mask].copy()

print(f"Exact duplicate records removed: {len(exact_duplicates):,}")


# ============================================================
# 8. SENSOR RELIABILITY PERIOD FILTER
# ============================================================

data["sensor_period_allowed"] = False

for sensor_id, (start_date, end_date) in ALLOWED_PERIODS.items():
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    mask = (
        data["sensorid"].astype(str).eq(sensor_id)
        & data["measurement_time_utc"].ge(start)
        & data["measurement_time_utc"].lt(end)
    )

    data.loc[mask, "sensor_period_allowed"] = True

# Do not use unknown sensors or measurements outside the approved periods
# in the provisional modelling data.
data["sensor_period_reason"] = np.select(
    [
        data["sensorid"].astype(str).isin(ALLOWED_PERIODS.keys())
        & data["sensor_period_allowed"],
        data["sensorid"].astype(str).isin(ALLOWED_PERIODS.keys())
        & ~data["sensor_period_allowed"],
    ],
    [
        "within_provisional_reliable_period",
        "outside_provisional_reliable_period",
    ],
    default="sensor_not_approved",
)


# ============================================================
# 9. DEFINE PROVISIONAL MODELLING CANDIDATES
# ============================================================

data["exclusion_reason"] = ""

def add_reason(mask, reason):
    """Append a reason without overwriting any previous reason."""
    existing = data.loc[mask, "exclusion_reason"]
    data.loc[mask, "exclusion_reason"] = np.where(
        existing.eq(""),
        reason,
        existing + "; " + reason,
    )


add_reason(
    data["created_at_utc"].isna(),
    "missing_creation_timestamp",
)

add_reason(
    data["measurement_time_utc"].isna(),
    "missing_measurement_timestamp",
)

add_reason(
    ~data["timestamp_offset_ok"],
    "timestamp_offset_missing_or_exceeds_10_minutes",
)

add_reason(
    ~data["sensor_period_allowed"],
    "outside_provisional_reliable_sensor_period",
)

add_reason(
    ~data["core_measurements_valid"],
    "missing_or_invalid_core_measurement",
)

candidate_mask = (
    data["created_at_utc"].notna()
    & data["measurement_time_utc"].notna()
    & data["timestamp_offset_ok"]
    & data["sensor_period_allowed"]
    & data["core_measurements_valid"]
)

candidates = data.loc[candidate_mask].copy()

print(
    "Candidate records before repeated-timestamp handling: "
    f"{len(candidates):,}"
)


# ============================================================
# 10. HANDLE REPEATED SENSOR TIMESTAMPS
# ============================================================

# Multiple source records can refer to the same sensor and measurement
# time. Keep the record with the smallest timestamp offset; use the
# latest creation time as a tie-breaker.
candidates = candidates.sort_values(
    [
        "sensorid",
        "measurement_time_utc",
        "timestamp_offset_minutes",
        "created_at_utc",
        "source_row_id",
    ],
    ascending=[True, True, True, False, True],
)

repeated_mask = candidates.duplicated(
    subset=["sensorid", "measurement_time_utc"],
    keep="first",
)

repeated_records = candidates.loc[repeated_mask].copy()
repeated_records["exclusion_reason"] = "repeated_sensor_measurement_timestamp"

modelling = candidates.loc[~repeated_mask].copy()

modelling = modelling.sort_values(
    ["sensorid", "measurement_time_utc"]
).reset_index(drop=True)

print(f"Repeated timestamp records excluded: {len(repeated_records):,}")
print(f"Modelling records before segmentation: {len(modelling):,}")


# ============================================================
# 11. IDENTIFY CONTINUOUS TIME SEGMENTS
# ============================================================

# A new segment begins when:
# - the sensor changes; or
# - the time gap between readings exceeds MAX_TIME_GAP_MIN.
modelling["time_gap_minutes"] = (
    modelling.groupby("sensorid")["measurement_time_utc"]
    .diff()
    .dt.total_seconds()
    .div(60)
)

new_segment = (
    modelling["time_gap_minutes"].isna()
    | (modelling["time_gap_minutes"] > MAX_TIME_GAP_MIN)
)

modelling["segment_number"] = (
    new_segment.groupby(modelling["sensorid"]).cumsum().astype(int)
)

modelling["time_segment_id"] = (
    modelling["sensorid"].astype(str)
    + "_"
    + modelling["segment_number"].astype(str).str.zfill(4)
)

segment_records = []

for (sensor_id, segment_id), group in modelling.groupby(
    ["sensorid", "time_segment_id"], sort=True
):
    group = group.sort_values("measurement_time_utc")

    times = group["measurement_time_utc"]
    intervals = times.diff().dt.total_seconds().div(60).dropna()

    segment_records.append(
        {
            "sensorid": sensor_id,
            "time_segment_id": segment_id,
            "start_time_utc": times.min(),
            "end_time_utc": times.max(),
            "record_count": len(group),
            "unique_timestamps": times.nunique(),
            "duration_minutes": (
                times.max() - times.min()
            ).total_seconds() / 60,
            "median_interval_minutes": (
                intervals.median() if len(intervals) else np.nan
            ),
            "mean_interval_minutes": (
                intervals.mean() if len(intervals) else np.nan
            ),
            "max_internal_gap_minutes": (
                intervals.max() if len(intervals) else np.nan
            ),
            "percent_intervals_8_to_12_minutes": (
                intervals.between(8, 12).mean() * 100
                if len(intervals) else np.nan
            ),
            "co2_unique_values": group["Carbon dioxide_clean"].nunique(),
            "temperature_unique_values": group["Temperature_clean"].nunique(),
            "humidity_unique_values": group["Humidity_clean"].nunique(),
            "co2_standard_deviation": (
                group["Carbon dioxide_clean"].std()
            ),
            "temperature_standard_deviation": (
                group["Temperature_clean"].std()
            ),
            "humidity_standard_deviation": (
                group["Humidity_clean"].std()
            ),
            "potentially_useful_segment": (
                len(group) >= MIN_SEGMENT_RECORDS
            ),
        }
    )

segment_report = pd.DataFrame(segment_records)

if not segment_report.empty:
    segment_report = segment_report.sort_values(
        ["sensorid", "start_time_utc"]
    )


# ============================================================
# 12. FLAG SEGMENTS TOO SHORT FOR MODELLING
# ============================================================

if not segment_report.empty:
    useful_segments = set(
        segment_report.loc[
            segment_report["potentially_useful_segment"],
            "time_segment_id",
        ]
    )

    modelling["segment_long_enough"] = (
        modelling["time_segment_id"].isin(useful_segments)
    )
else:
    modelling["segment_long_enough"] = False

# Keep all valid readings in modelling_sensor_data.csv, but flag short
# segments. Later scripts must only create windows within one segment.
# This preserves useful observations for inspection without hiding gaps.


# ============================================================
# 13. MEASUREMENT QUALITY SUMMARY
# ============================================================

quality_rows = []

for column in VALUE_RANGES:
    if column not in data.columns:
        continue

    values = data[column]
    clean_values = data[f"{column}_clean"]

    quality_rows.append(
        {
            "measurement": column,
            "total_records": len(data),
            "missing_count": int(values.isna().sum()),
            "missing_percent": (
                values.isna().mean() * 100 if len(data) else np.nan
            ),
            "out_of_range_count": int(
                data[f"{safe_filename(column)}_out_of_range"].sum()
            ),
            "valid_count": int(clean_values.notna().sum()),
            "minimum_valid": clean_values.min(),
            "maximum_valid": clean_values.max(),
            "mean_valid": clean_values.mean(),
            "standard_deviation_valid": clean_values.std(),
        }
    )

measurement_quality = pd.DataFrame(quality_rows)


# ============================================================
# 14. SENSOR RELIABILITY SUMMARY
# ============================================================

reliability_rows = []

for sensor_id, group in data.groupby("sensorid", dropna=False):
    sensor_candidates = candidates[
        candidates["sensorid"].eq(sensor_id)
    ]
    sensor_model = modelling[
        modelling["sensorid"].eq(sensor_id)
    ]

    reliability_rows.append(
        {
            "sensorid": sensor_id,
            "source_records_after_exact_deduplication": len(group),
            "unique_measurement_timestamps": (
                group["measurement_time_utc"].nunique()
            ),
            "records_in_approved_period": int(
                group["sensor_period_allowed"].sum()
            ),
            "records_passing_timestamp_check": int(
                group["timestamp_offset_ok"].sum()
            ),
            "candidate_records": len(sensor_candidates),
            "modelling_records": len(sensor_model),
            "number_of_segments": (
                sensor_model["time_segment_id"].nunique()
            ),
            "co2_unique_values_in_model": (
                sensor_model["Carbon dioxide_clean"].nunique()
            ),
            "temperature_unique_values_in_model": (
                sensor_model["Temperature_clean"].nunique()
            ),
            "humidity_unique_values_in_model": (
                sensor_model["Humidity_clean"].nunique()
            ),
        }
    )

sensor_reliability = pd.DataFrame(reliability_rows)


# ============================================================
# 15. MONTHLY RELIABILITY SUMMARY
# ============================================================

monthly_data = data.loc[
    data["measurement_time_utc"].notna()
].copy()

monthly_data["measurement_month_utc"] = (
    monthly_data["measurement_time_utc"].dt.to_period("M").astype(str)
)

monthly_rows = []

for (sensor_id, month), group in monthly_data.groupby(
    ["sensorid", "measurement_month_utc"], dropna=False
):
    monthly_rows.append(
        {
            "sensorid": sensor_id,
            "measurement_month_utc": month,
            "source_records": len(group),
            "unique_measurement_timestamps": (
                group["measurement_time_utc"].nunique()
            ),
            "approved_period_records": int(
                group["sensor_period_allowed"].sum()
            ),
            "timestamp_check_pass_records": int(
                group["timestamp_offset_ok"].sum()
            ),
            "co2_unique_values": group["Carbon dioxide"].nunique(),
            "temperature_unique_values": group["Temperature"].nunique(),
        }
    )

monthly_reliability = pd.DataFrame(monthly_rows)


# ============================================================
# 16. EXCLUDED RECORDS AND REASONS
# ============================================================

excluded = data.loc[~candidate_mask].copy()

# Candidate records removed due to repeated timestamps are also excluded.
# Combine these for a complete audit trail.
excluded = pd.concat(
    [excluded, repeated_records],
    ignore_index=True,
    sort=False,
)

# A short summary of exclusion reasons.
reason_rows = []

for reason in [
    "missing_creation_timestamp",
    "missing_measurement_timestamp",
    "timestamp_offset_missing_or_exceeds_10_minutes",
    "outside_provisional_reliable_sensor_period",
    "missing_or_invalid_core_measurement",
    "repeated_sensor_measurement_timestamp",
]:
    if reason == "repeated_sensor_measurement_timestamp":
        count = len(repeated_records)
    else:
        count = int(
            data["exclusion_reason"].str.contains(
                reason, regex=False, na=False
            ).sum()
        )

    reason_rows.append(
        {
            "exclusion_reason": reason,
            "record_count": count,
        }
    )

exclusion_summary = pd.DataFrame(reason_rows)


# ============================================================
# 17. SAVE OUTPUT FILES
# ============================================================

write_csv(
    data,
    DATA_DIR / "extracted_sensor_data.csv",
)

write_csv(
    data,
    DATA_DIR / "cleaned_sensor_data.csv",
)

write_csv(
    modelling,
    DATA_DIR / "modelling_sensor_data.csv",
)

write_csv(
    excluded,
    DATA_DIR / "excluded_sensor_records.csv",
)

write_csv(
    exact_duplicates,
    DIAGNOSTICS_DIR / "exact_duplicate_records.csv",
)

write_csv(
    repeated_records,
    DIAGNOSTICS_DIR / "repeated_timestamp_records_excluded.csv",
)

write_csv(
    measurement_quality,
    DIAGNOSTICS_DIR / "measurement_quality_summary.csv",
)

write_csv(
    sensor_reliability,
    DIAGNOSTICS_DIR / "sensor_reliability_summary.csv",
)

write_csv(
    monthly_reliability,
    DIAGNOSTICS_DIR / "sensor_monthly_reliability.csv",
)

write_csv(
    segment_report,
    DIAGNOSTICS_DIR / "continuous_time_segments.csv",
)

write_csv(
    exclusion_summary,
    DIAGNOSTICS_DIR / "exclusion_reason_summary.csv",
)


# ============================================================
# 18. RUN SUMMARY
# ============================================================

run_summary = pd.DataFrame(
    [
        {"metric": "original_records", "value": len(raw)},
        {"metric": "records_after_exact_deduplication", "value": len(data)},
        {"metric": "exact_duplicates_removed", "value": len(exact_duplicates)},
        {"metric": "candidate_records", "value": len(candidates)},
        {"metric": "repeated_timestamps_excluded", "value": len(repeated_records)},
        {"metric": "modelling_records", "value": len(modelling)},
        {"metric": "excluded_records", "value": len(excluded)},
        {
            "metric": "segments",
            "value": modelling["time_segment_id"].nunique(),
        },
        {
            "metric": "segments_at_least_minimum_length",
            "value": int(
                segment_report["potentially_useful_segment"].sum()
            ) if not segment_report.empty else 0,
        },
        {
            "metric": "modelling_rows_with_unknown_timestamp_offset",
            "value": int((~modelling["timestamp_offset_known"]).sum()),
        },
        {
            "metric": "verified_occupancy_labels_available",
            "value": "NO - not identified in this source dataset",
        },
    ]
)

write_csv(
    run_summary,
    DIAGNOSTICS_DIR / "cleaning_run_summary.csv",
)


# ============================================================
# 19. PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("CLEANING COMPLETE")
print("=" * 65)

print(f"Original records:                         {len(raw):,}")
print(f"Records after exact duplicate removal:    {len(data):,}")
print(f"Exact duplicates removed:                 {len(exact_duplicates):,}")
print(f"Candidate records:                        {len(candidates):,}")
print(f"Repeated timestamps excluded:             {len(repeated_records):,}")
print(f"Modelling records:                        {len(modelling):,}")
print(f"Excluded records:                         {len(excluded):,}")
print(
    "Continuous time segments:                "
    f"{modelling['time_segment_id'].nunique():,}"
)

if not segment_report.empty:
    print(
        "Segments with at least "
        f"{MIN_SEGMENT_RECORDS} readings:             "
        f"{int(segment_report['potentially_useful_segment'].sum()):,}"
    )

print("\nRecords per sensor:")
if not modelling.empty:
    print(
        modelling.groupby("sensorid").size().to_string()
    )
else:
    print("No modelling records were produced.")

print("\nOutput files saved in:")
print(f"  {DATA_DIR}")
print(f"  {DIAGNOSTICS_DIR}")

print("\nIMPORTANT:")
print("- The modelling dataset is provisional.")
print("- Segment IDs must be respected when creating time windows.")
print("- No verified occupancy labels have been identified.")
print("- Environmental readings, including CO2, are not ground-truth occupancy.")

print("\nNext step: inspect continuous_time_segments.csv and the reliability reports.")
