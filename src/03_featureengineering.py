from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "modelling_sensor_data.csv"
OUTPUT_DIR = ROOT / "outputs" / "features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_TIMEZONE = "Australia/Melbourne"
MIN_SEGMENT_RECORDS = 30

CORE_VARIABLES = [
    "Carbon dioxide",
    "Temperature",
    "Humidity",
    "Pressure",
]

OPTIONAL_VARIABLES = [
    "Battery",
    "Formaldehyde",
    "Light Volatile Organic Compounds",
    "Particulate matter 1",
    "Particulate matter 2.5",
    "Particulate matter 4",
    "Particulate matter 10",
]

FEATURE_NAMES = {
    "Carbon dioxide": "co2",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Pressure": "pressure",
    "Battery": "battery",
    "Formaldehyde": "formaldehyde",
    "Light Volatile Organic Compounds": "lvoc",
    "Particulate matter 1": "pm1",
    "Particulate matter 2.5": "pm25",
    "Particulate matter 4": "pm4",
    "Particulate matter 10": "pm10",
}

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def get_measurement_column(dataframe, variable):
    """Prefer cleaned measurement columns when available."""
    cleaned_column = f"{variable}_clean"

    if cleaned_column in dataframe.columns:
        return cleaned_column

    if variable in dataframe.columns:
        return variable

    return None


def parse_boolean(series):
    """Safely interpret Boolean columns loaded from CSV."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


# ============================================================
# 3. LOAD AND VALIDATE DATA
# ============================================================

print_section("LOADING DATA")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Run src/01_datacleaning.py first."
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)

required_columns = [
    "sensorid",
    "measurement_time_utc",
    "time_segment_id",
]

missing_required = [
    column for column in required_columns
    if column not in df.columns
]

if missing_required:
    raise KeyError(
        f"Required columns are missing: {missing_required}"
    )

original_row_count = len(df)

df["measurement_time_utc"] = pd.to_datetime(
    df["measurement_time_utc"],
    utc=True,
    errors="coerce",
)

invalid_timestamps = int(
    df["measurement_time_utc"].isna().sum()
)

df = df.dropna(
    subset=["measurement_time_utc", "time_segment_id"]
).copy()

df["sensorid"] = df["sensorid"].astype(str)
df["time_segment_id"] = df["time_segment_id"].astype(str)

df = df.sort_values(
    ["sensorid", "time_segment_id", "measurement_time_utc"]
).reset_index(drop=True)

print(f"Original records: {original_row_count:,}")
print(f"Invalid timestamps removed: {invalid_timestamps:,}")
print(f"Records remaining: {len(df):,}")
print(f"Unique sensors: {df['sensorid'].nunique():,}")
print(f"Original segments: {df['time_segment_id'].nunique():,}")

# ============================================================
# 4. SELECT AND STANDARDISE MEASUREMENTS
# ============================================================

print_section("SELECTING MEASUREMENTS")

measurement_columns = {}

for variable in CORE_VARIABLES + OPTIONAL_VARIABLES:
    column = get_measurement_column(df, variable)

    if column is None:
        print(f"{variable}: unavailable; skipping")
        continue

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    measurement_columns[variable] = column

    feature_name = FEATURE_NAMES[variable]
    df[feature_name] = df[column]

    print(f"{variable}: using {column} -> {feature_name}")

for variable in CORE_VARIABLES:
    if variable not in measurement_columns:
        raise KeyError(
            f"Required environmental measurement missing: {variable}"
        )

# ============================================================
# 5. IDENTIFY USABLE SEGMENTS
# ============================================================

print_section("IDENTIFYING USABLE SEGMENTS")

# Preserve File 01's segment identifiers. Never calculate
# lags or rolling statistics across different segments.

if "segment_long_enough" in df.columns:
    df["segment_long_enough"] = parse_boolean(
        df["segment_long_enough"]
    )
else:
    segment_sizes = df.groupby("time_segment_id")[
        "measurement_time_utc"
    ].transform("size")

    df["segment_long_enough"] = (
        segment_sizes >= MIN_SEGMENT_RECORDS
    )

usable_df = df.loc[
    df["segment_long_enough"]
].copy()

print(f"All cleaned records: {len(df):,}")
print(f"Records in usable segments: {len(usable_df):,}")
print(
    "Usable segments:",
    usable_df["time_segment_id"].nunique(),
)

# Use only sufficiently long segments for feature generation.
# Retain the full dataset separately for traceability.

df = usable_df.sort_values(
    ["sensorid", "time_segment_id", "measurement_time_utc"]
).reset_index(drop=True)

GROUP_COLUMNS = ["sensorid", "time_segment_id"]

# ============================================================
# 6. CREATE TIME FEATURES
# ============================================================

print_section("CREATING TIME FEATURES")

df["measurement_time_melbourne"] = (
    df["measurement_time_utc"]
    .dt.tz_convert(LOCAL_TIMEZONE)
)

df["local_hour"] = (
    df["measurement_time_melbourne"].dt.hour
)

df["local_minute"] = (
    df["measurement_time_melbourne"].dt.minute
)

df["local_day_of_week"] = (
    df["measurement_time_melbourne"].dt.dayofweek
)

df["local_month"] = (
    df["measurement_time_melbourne"].dt.month
)

df["is_weekend"] = (
    df["local_day_of_week"] >= 5
).astype(int)

df["is_business_hours"] = (
    df["local_hour"].between(9, 17)
    & (df["local_day_of_week"] < 5)
).astype(int)

# Cyclical encoding avoids treating midnight as far away
# from 11 pm, or Sunday as far away from Monday.

minute_of_day = (
    df["local_hour"] * 60 + df["local_minute"]
)

df["hour_sin"] = np.sin(
    2 * np.pi * minute_of_day / 1440
)

df["hour_cos"] = np.cos(
    2 * np.pi * minute_of_day / 1440
)

df["day_of_week_sin"] = np.sin(
    2 * np.pi * df["local_day_of_week"] / 7
)

df["day_of_week_cos"] = np.cos(
    2 * np.pi * df["local_day_of_week"] / 7
)

# ============================================================
# 7. SAMPLING INTERVAL FEATURES
# ============================================================

print_section("CREATING SAMPLING INTERVAL FEATURES")

df["minutes_since_previous_reading"] = (
    df.groupby(GROUP_COLUMNS)["measurement_time_utc"]
    .diff()
    .dt.total_seconds()
    .div(60)
)

df["sampling_interval_is_regular"] = (
    df["minutes_since_previous_reading"]
    .between(8, 12)
    .astype(int)
)

# ============================================================
# 8. LAG FEATURES
# ============================================================

print_section("CREATING LAG FEATURES")

# The nominal sampling interval is approximately 10 minutes.
# Only use lagged readings when their actual timestamp spacing
# is close to the expected lag.

LAG_SETTINGS = {
    1: (10, 3),
    3: (30, 5),
    6: (60, 10),
}

BASE_FEATURES = [
    FEATURE_NAMES[variable]
    for variable in CORE_VARIABLES
]

for feature in BASE_FEATURES:
    grouped = df.groupby(GROUP_COLUMNS, sort=False)

    for lag_rows, (expected_minutes, tolerance_minutes) in (
        LAG_SETTINGS.items()
    ):
        lag_value = grouped[feature].shift(lag_rows)

        lag_timestamp = grouped[
            "measurement_time_utc"
        ].shift(lag_rows)

        elapsed_minutes = (
            df["measurement_time_utc"] - lag_timestamp
        ).dt.total_seconds() / 60

        valid_lag = (
            elapsed_minutes.sub(expected_minutes)
            .abs()
            .le(tolerance_minutes)
        )

        df[f"{feature}_lag_{lag_rows}"] = (
            lag_value.where(valid_lag)
        )

        # Change relative to the earlier measurement.
        df[f"{feature}_change_{lag_rows}"] = (
            df[feature] - lag_value
        ).where(valid_lag)

# ============================================================
# 9. ROLLING ENVIRONMENTAL FEATURES
# ============================================================

print_section("CREATING ROLLING FEATURES")

# Rolling statistics include the current measurement and
# earlier measurements within the same segment.
#
# They describe current environmental conditions. If the
# project later predicts future occupancy, features must be
# shifted appropriately to avoid using future information.

for feature in BASE_FEATURES:
    for window in ["30min", "60min"]:

        rolling_mean = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

        rolling_std = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

        rolling_max = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

        rolling_min = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

        for _, group in df.groupby(
            GROUP_COLUMNS,
            sort=False,
        ):
            group = group.sort_values(
                "measurement_time_utc"
            )

            indexed_values = pd.Series(
                group[feature].to_numpy(),
                index=pd.DatetimeIndex(
                    group["measurement_time_utc"]
                ),
            )

            mean_values = indexed_values.rolling(
                window,
                min_periods=2,
            ).mean()

            std_values = indexed_values.rolling(
                window,
                min_periods=2,
            ).std()

            max_values = indexed_values.rolling(
                window,
                min_periods=2,
            ).max()

            min_values = indexed_values.rolling(
                window,
                min_periods=2,
            ).min()

            rolling_mean.loc[group.index] = mean_values.to_numpy()
            rolling_std.loc[group.index] = std_values.to_numpy()
            rolling_max.loc[group.index] = max_values.to_numpy()
            rolling_min.loc[group.index] = min_values.to_numpy()

        window_name = window.replace("min", "m")

        df[f"{feature}_rolling_mean_{window_name}"] = rolling_mean
        df[f"{feature}_rolling_std_{window_name}"] = rolling_std
        df[f"{feature}_rolling_max_{window_name}"] = rolling_max
        df[f"{feature}_rolling_min_{window_name}"] = rolling_min

# ============================================================
# 10. MISSINGNESS FEATURES
# ============================================================

print_section("CREATING MISSINGNESS FEATURES")

for variable, column in measurement_columns.items():
    feature = FEATURE_NAMES[variable]

    df[f"{feature}_missing"] = (
        df[feature].isna().astype(int)
    )

# ============================================================
# 11. SAVE FEATURE-ENGINEERED DATA
# ============================================================

print_section("SAVING FEATURE-ENGINEERED DATA")

# Preserve identifiers, original measurement columns, timestamps,
# segment information, and generated features.

OUTPUT_FILE = OUTPUT_DIR / "feature_engineered_data.csv"

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved feature dataset: {OUTPUT_FILE}")
print(f"Output rows: {len(df):,}")
print(f"Output columns: {len(df.columns):,}")

# ============================================================
# 12. FEATURE SUMMARY
# ============================================================

print_section("CREATING FEATURE SUMMARY")

metadata_columns = {
    "source_row_id",
    "id",
    "sensorid",
    "devicetype",
    "status",
    "processing_errors",
    "createdate",
    "processdate",
    "measurement_timestamp_unix",
    "measurement_time_utc",
    "measurement_time_melbourne",
    "time_segment_id",
    "segment_number",
    "segment_long_enough",
    "segment_long_enough",
    "sensor_period_allowed",
    "sensor_period_reason",
    "exclusion_reason",
    "timestamp_offset_known",
    "timestamp_offset_ok",
    "timestamp_offset_minutes",
    "core_measurements_valid",
    "eda_segment_id",
    "eda_segment_number",
}

feature_columns = [
    column for column in df.columns
    if column not in metadata_columns
    and column not in measurement_columns.values()
]

summary_rows = []

for column in feature_columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        summary_rows.append({
            "feature": column,
            "dtype": str(df[column].dtype),
            "missing_count": int(df[column].isna().sum()),
            "missing_percent": (
                df[column].isna().mean() * 100
            ),
            "unique_values": int(df[column].nunique()),
            "mean": df[column].mean(),
            "standard_deviation": df[column].std(),
            "minimum": df[column].min(),
            "maximum": df[column].max(),
        })
    else:
        summary_rows.append({
            "feature": column,
            "dtype": str(df[column].dtype),
            "missing_count": int(df[column].isna().sum()),
            "missing_percent": (
                df[column].isna().mean() * 100
            ),
            "unique_values": int(df[column].nunique()),
            "mean": np.nan,
            "standard_deviation": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
        })

feature_summary = pd.DataFrame(summary_rows)

feature_summary.to_csv(
    OUTPUT_DIR / "feature_summary.csv",
    index=False,
)

print(
    f"Saved feature summary: "
    f"{OUTPUT_DIR / 'feature_summary.csv'}"
)

# ============================================================
# 13. QUALITY CHECKS
# ============================================================

print_section("QUALITY CHECKS")

quality_summary = pd.DataFrame([{
    "input_rows": original_row_count,
    "invalid_timestamps_removed": invalid_timestamps,
    "output_rows": len(df),
    "output_columns": len(df.columns),
    "unique_sensors": df["sensorid"].nunique(),
    "unique_segments": df["time_segment_id"].nunique(),
    "feature_columns": len(feature_columns),
    "duplicate_sensor_timestamps": int(
        df.duplicated(
            subset=[
                "sensorid",
                "measurement_time_utc",
            ]
        ).sum()
    ),
    "output_file": str(OUTPUT_FILE),
}])

quality_summary.to_csv(
    OUTPUT_DIR / "feature_engineering_quality_summary.csv",
    index=False,
)

print(quality_summary.to_string(index=False))

print_section("FEATURE ENGINEERING COMPLETED")

print(f"Output directory: {OUTPUT_DIR}")
print("Feature dataset: feature_engineered_data.csv")
print("Feature summary: feature_summary.csv")
print(
    "Quality report: feature_engineering_quality_summary.csv"
)

print(
    "\nNote: No occupancy labels have been generated. "
    "The output is ready for further analysis, but supervised "
    "occupancy model training requires a verified target label."
)
