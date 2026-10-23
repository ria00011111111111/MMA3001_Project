
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "modelling_sensor_data.csv"
OUTPUT_DIR = ROOT / "outputs" / "eda_v5"
PLOT_DIR = OUTPUT_DIR / "plots"
SUMMARY_DIR = OUTPUT_DIR / "summaries"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

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


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_csv(dataframe, filename, directory=SUMMARY_DIR):
    path = directory / filename
    dataframe.to_csv(path, index=False)
    print(f"Saved: {path}")


def get_measurement_column(dataframe, variable):
    """Prefer cleaned measurements when available."""
    cleaned_name = f"{variable}_clean"

    if cleaned_name in dataframe.columns:
        return cleaned_name

    if variable in dataframe.columns:
        return variable

    return None


def save_plot(filename):
    """Save and close the current Matplotlib figure."""
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def parse_boolean(series):
    """Safely interpret booleans loaded from CSV."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


# ============================================================
# 3. LOAD DATA
# ============================================================

print_section("LOADING DATA")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find input file:\n{INPUT_FILE}\n"
        "Run src/01_datacleaning.py first."
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Input file: {INPUT_FILE}")
print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")

if "measurement_time_utc" not in df.columns:
    raise KeyError(
        "The input file has no 'measurement_time_utc' column. "
        "Check the output of File 01."
    )

if "sensorid" not in df.columns:
    raise KeyError("The input file has no 'sensorid' column.")

df["measurement_time_utc"] = pd.to_datetime(
    df["measurement_time_utc"],
    utc=True,
    errors="coerce",
)

invalid_timestamp_count = int(
    df["measurement_time_utc"].isna().sum()
)

df = df.dropna(subset=["measurement_time_utc"]).copy()

df["sensorid"] = df["sensorid"].astype(str)

df = df.sort_values(
    ["sensorid", "measurement_time_utc"]
).reset_index(drop=True)

# Keep UTC for calculations and reproducibility.
# Add Melbourne local time for interpreting daily patterns.
df["measurement_time_melbourne"] = (
    df["measurement_time_utc"]
    .dt.tz_convert(LOCAL_TIMEZONE)
)

df["local_hour"] = df["measurement_time_melbourne"].dt.hour
df["local_day_of_week"] = (
    df["measurement_time_melbourne"].dt.day_name()
)
df["local_date"] = (
    df["measurement_time_melbourne"].dt.strftime("%Y-%m-%d")
)
df["local_month"] = (
    df["measurement_time_melbourne"].dt.strftime("%Y-%m")
)
df["is_weekend"] = (
    df["measurement_time_melbourne"].dt.dayofweek >= 5
)

print(f"Invalid timestamps removed: {invalid_timestamp_count:,}")
print(f"Rows remaining: {len(df):,}")
print(f"Unique sensors: {df['sensorid'].nunique():,}")

if not df.empty:
    print(
        "UTC date range:",
        df["measurement_time_utc"].min(),
        "to",
        df["measurement_time_utc"].max(),
    )

    print(
        "Melbourne date range:",
        df["measurement_time_melbourne"].min(),
        "to",
        df["measurement_time_melbourne"].max(),
    )


# ============================================================
# 4. IDENTIFY MEASUREMENT COLUMNS
# ============================================================

print_section("IDENTIFYING MEASUREMENTS")

available_variables = {}

for variable in CORE_VARIABLES + OPTIONAL_VARIABLES:
    column = get_measurement_column(df, variable)

    if column is not None:
        available_variables[variable] = column
        df[column] = pd.to_numeric(df[column], errors="coerce")
        print(f"{variable}: using '{column}'")
    else:
        print(f"{variable}: not present; skipping")

for variable in CORE_VARIABLES:
    if variable not in available_variables:
        raise KeyError(
            f"Required measurement '{variable}' is missing. "
            "Check the File 01 output."
        )


# ============================================================
# 5. DATA QUALITY SUMMARY
# ============================================================

print_section("DATA QUALITY")

quality_rows = []

for variable, column in available_variables.items():
    values = df[column]

    quality_rows.append({
        "variable": variable,
        "source_column": column,
        "total_rows": len(df),
        "valid_values": int(values.notna().sum()),
        "missing_values": int(values.isna().sum()),
        "missing_percent": (
            values.isna().mean() * 100 if len(df) else np.nan
        ),
        "unique_values": int(values.nunique()),
        "minimum": values.min(),
        "median": values.median(),
        "mean": values.mean(),
        "maximum": values.max(),
        "standard_deviation": values.std(),
    })

quality_df = pd.DataFrame(quality_rows)
save_csv(quality_df, "measurement_quality_summary.csv")

print(quality_df.to_string(index=False))


# ============================================================
# 6. SENSOR VARIABILITY
# ============================================================

print_section("SENSOR VARIABILITY")

variability_rows = []

if "segment_long_enough" in df.columns:
    long_mask = parse_boolean(df["segment_long_enough"])
else:
    long_mask = pd.Series(True, index=df.index)

long_df = df.loc[long_mask].copy()

for sensor_id, sensor_data in df.groupby("sensorid"):
    sensor_long = long_df.loc[long_df["sensorid"] == sensor_id]

    for variable, column in available_variables.items():
        values = pd.to_numeric(
            sensor_long[column], errors="coerce"
        ).dropna()

        variability_rows.append({
            "sensorid": sensor_id,
            "measurement_variable": variable,
            "source_column": column,
            "rows_in_long_segments": len(sensor_long),
            "valid_measurements": len(values),
            "unique_values": int(values.nunique()),
            "mean": values.mean(),
            "standard_deviation": values.std(),
            "minimum": values.min(),
            "maximum": values.max(),
        })

variability_df = pd.DataFrame(variability_rows)

save_csv(variability_df, "sensor_variability_summary.csv")

print(variability_df.to_string(index=False))


# ============================================================
# 7. SAMPLING INTERVALS AND CONTINUOUS SEGMENTS
# ============================================================

print_section("SAMPLING INTERVALS")

df["interval_minutes"] = (
    df.groupby("sensorid")["measurement_time_utc"]
    .diff()
    .dt.total_seconds()
    .div(60)
)

intervals = df.loc[
    df["interval_minutes"].gt(0), "interval_minutes"
].dropna()

if not intervals.empty:
    interval_summary = pd.DataFrame([{
        "number_of_intervals": len(intervals),
        "mean_minutes": intervals.mean(),
        "median_minutes": intervals.median(),
        "standard_deviation_minutes": intervals.std(),
        "minimum_minutes": intervals.min(),
        "maximum_minutes": intervals.max(),
        "intervals_8_to_12_minutes_percent": (
            intervals.between(8, 12).mean() * 100
        ),
        "intervals_over_30_minutes": int(
            intervals.gt(30).sum()
        ),
    }])

    save_csv(interval_summary, "sampling_interval_summary.csv")
    print(interval_summary.to_string(index=False))
else:
    print("No positive sampling intervals available.")

# Recalculate segments independently within each sensor.
# A gap greater than 30 minutes starts a new segment.
df["segment_gap_minutes"] = (
    df.groupby("sensorid")["measurement_time_utc"]
    .diff()
    .dt.total_seconds()
    .div(60)
)

df["eda_segment_number"] = (
    df["segment_gap_minutes"].isna()
    | df["segment_gap_minutes"].gt(30)
).groupby(df["sensorid"]).cumsum().astype(int)

df["eda_segment_id"] = (
    df["sensorid"]
    + "_EDA_"
    + df["eda_segment_number"].astype(str).str.zfill(4)
)

segment_rows = []

for (sensor_id, segment_id), group in df.groupby(
    ["sensorid", "eda_segment_id"], sort=False
):
    group = group.sort_values("measurement_time_utc")

    segment_intervals = (
        group["measurement_time_utc"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .dropna()
    )

    segment_rows.append({
        "sensorid": sensor_id,
        "eda_segment_id": segment_id,
        "number_of_readings": len(group),
        "start_utc": group["measurement_time_utc"].min(),
        "end_utc": group["measurement_time_utc"].max(),
        "start_melbourne": (
            group["measurement_time_melbourne"].min()
        ),
        "end_melbourne": (
            group["measurement_time_melbourne"].max()
        ),
        "duration_hours": (
            (
                group["measurement_time_utc"].max()
                - group["measurement_time_utc"].min()
            ).total_seconds() / 3600
        ),
        "median_interval_minutes": (
            segment_intervals.median()
            if not segment_intervals.empty else np.nan
        ),
        "maximum_interval_minutes": (
            segment_intervals.max()
            if not segment_intervals.empty else np.nan
        ),
        "percent_intervals_8_to_12_minutes": (
            segment_intervals.between(8, 12).mean() * 100
            if not segment_intervals.empty else np.nan
        ),
        "long_enough": len(group) >= MIN_SEGMENT_RECORDS,
    })

segments_df = pd.DataFrame(segment_rows)

save_csv(segments_df, "continuous_segment_summary.csv")

print(f"Continuous segments: {len(segments_df):,}")
print(
    "Segments with at least "
    f"{MIN_SEGMENT_RECORDS} readings: "
    f"{int(segments_df['long_enough'].sum()):,}"
)

if not segments_df.empty:
    print("\nLargest segments:")
    print(
        segments_df.sort_values(
            "number_of_readings", ascending=False
        ).head(15).to_string(index=False)
    )

# Calculate time between consecutive segments for each sensor.
between_segment_rows = []

for sensor_id, sensor_segments in segments_df.groupby("sensorid"):
    sensor_segments = sensor_segments.sort_values("start_utc")
    previous_end = None

    for _, segment in sensor_segments.iterrows():
        if previous_end is not None:
            gap_hours = (
                segment["start_utc"] - previous_end
            ).total_seconds() / 3600

            between_segment_rows.append({
                "sensorid": sensor_id,
                "next_segment_id": segment["eda_segment_id"],
                "gap_hours": gap_hours,
                "gap_over_24_hours": gap_hours > 24,
            })

        previous_end = segment["end_utc"]

between_segments_df = pd.DataFrame(between_segment_rows)

if not between_segments_df.empty:
    save_csv(
        between_segments_df,
        "between_segment_gaps.csv",
    )

    print(
        "Gaps over 24 hours:",
        int(between_segments_df["gap_over_24_hours"].sum()),
    )


# ============================================================
# 8. DESCRIPTIVE STATISTICS
# ============================================================

print_section("DESCRIPTIVE STATISTICS")

statistics_rows = []

for subset_name, subset in [
    ("all_selected_rows", df),
    ("long_segments_only", long_df),
]:
    for variable, column in available_variables.items():
        values = pd.to_numeric(
            subset[column], errors="coerce"
        ).dropna()

        statistics_rows.append({
            "dataset_subset": subset_name,
            "measurement_variable": variable,
            "source_column": column,
            "count": len(values),
            "missing": int(subset[column].isna().sum()),
            "mean": values.mean(),
            "median": values.median(),
            "standard_deviation": values.std(),
            "minimum": values.min(),
            "25th_percentile": values.quantile(0.25),
            "75th_percentile": values.quantile(0.75),
            "maximum": values.max(),
            "unique_values": int(values.nunique()),
        })

statistics_df = pd.DataFrame(statistics_rows)

save_csv(statistics_df, "descriptive_statistics.csv")
print(statistics_df.to_string(index=False))


# ============================================================
# 9. DISTRIBUTION PLOTS
# ============================================================

print_section("GENERATING DISTRIBUTION PLOTS")

for variable, column in available_variables.items():
    values = pd.to_numeric(
        long_df[column], errors="coerce"
    ).dropna()

    if values.empty:
        continue

    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=40, edgecolor="black")
    plt.title(f"{variable} Distribution")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    save_plot(f"distribution_{column.replace(' ', '_')}.png")

print("Distribution plots saved.")


# ============================================================
# 10. TIME-SERIES PLOTS BY SENSOR
# ============================================================

print_section("GENERATING TIME-SERIES PLOTS")

for sensor_id, sensor_data in long_df.groupby("sensorid"):
    sensor_data = sensor_data.sort_values("measurement_time_utc")

    for variable in CORE_VARIABLES:
        column = available_variables[variable]

        if sensor_data[column].notna().sum() == 0:
            continue

        plt.figure(figsize=(12, 5))
        plt.plot(
            sensor_data["measurement_time_utc"],
            sensor_data[column],
            linewidth=0.8,
        )
        plt.title(f"{variable} Over Time: Sensor {sensor_id}")
        plt.xlabel("Measurement time (UTC)")
        plt.ylabel(variable)
        plt.gcf().autofmt_xdate()

        save_plot(
            f"timeseries_{sensor_id}_{column.replace(' ', '_')}.png"
        )

print("Time-series plots saved.")


# ============================================================
# 11. CO2 PEAK INVESTIGATION
# ============================================================

print_section("CO2 PEAK INVESTIGATION")

co2_column = available_variables["Carbon dioxide"]

co2_data = long_df.dropna(subset=[co2_column]).copy()

if not co2_data.empty:
    peak_index = co2_data[co2_column].idxmax()
    peak = co2_data.loc[peak_index]

    peak_summary = pd.DataFrame([{
        "sensorid": peak["sensorid"],
        "measurement_time_utc": peak["measurement_time_utc"],
        "measurement_time_melbourne": (
            peak["measurement_time_melbourne"]
        ),
        "co2_ppm": peak[co2_column],
        "dataset_mean_co2_ppm": co2_data[co2_column].mean(),
        "dataset_median_co2_ppm": co2_data[co2_column].median(),
        "readings_over_600_ppm": int(
            co2_data[co2_column].gt(600).sum()
        ),
        "readings_over_800_ppm": int(
            co2_data[co2_column].gt(800).sum()
        ),
        "readings_over_1000_ppm": int(
            co2_data[co2_column].gt(1000).sum()
        ),
        "percent_over_600_ppm": (
            co2_data[co2_column].gt(600).mean() * 100
        ),
        "percent_over_800_ppm": (
            co2_data[co2_column].gt(800).mean() * 100
        ),
        "percent_over_1000_ppm": (
            co2_data[co2_column].gt(1000).mean() * 100
        ),
    }])

    save_csv(peak_summary, "co2_peak_summary.csv")
    print(peak_summary.to_string(index=False))

    # Save the 20 highest readings for manual inspection.
    top_readings = co2_data.nlargest(20, co2_column).copy()

    context_columns = [
        "sensorid",
        "measurement_time_utc",
        "measurement_time_melbourne",
        co2_column,
    ]

    if "Temperature" in available_variables:
        context_columns.append(available_variables["Temperature"])

    if "Humidity" in available_variables:
        context_columns.append(available_variables["Humidity"])

    if "Pressure" in available_variables:
        context_columns.append(available_variables["Pressure"])

    context_columns = [
        column for column in context_columns
        if column in top_readings.columns
    ]

    save_csv(
        top_readings[context_columns],
        "top_20_co2_readings.csv",
    )

    # Save a 2-hour window around the highest reading.
    peak_time = peak["measurement_time_utc"]
    peak_sensor = peak["sensorid"]

    window = co2_data[
        (co2_data["sensorid"] == peak_sensor)
        & (
            co2_data["measurement_time_utc"]
            >= peak_time - pd.Timedelta(hours=1)
        )
        & (
            co2_data["measurement_time_utc"]
            <= peak_time + pd.Timedelta(hours=1)
        )
    ].copy()

    save_csv(window, "co2_peak_context_window.csv")

    plt.figure(figsize=(12, 5))
    plt.plot(
        co2_data["measurement_time_utc"],
        co2_data[co2_column],
        linewidth=0.7,
    )
    plt.axhline(600, linestyle="--", label="600 ppm")
    plt.axhline(800, linestyle="--", label="800 ppm")
    plt.axhline(1000, linestyle="--", label="1000 ppm")
    plt.title("CO₂ Readings Across Selected Data")
    plt.xlabel("Measurement time (UTC)")
    plt.ylabel("CO₂ (ppm)")
    plt.legend()
    plt.gcf().autofmt_xdate()
    save_plot("co2_over_time_with_reference_levels.png")

    plt.figure(figsize=(9, 5))
    plt.hist(co2_data[co2_column], bins=50, edgecolor="black")
    plt.title("CO₂ Distribution in Long Segments")
    plt.xlabel("CO₂ (ppm)")
    plt.ylabel("Frequency")
    save_plot("co2_distribution_long_segments.png")

    print(
        "\nImportant: elevated CO₂ readings have been flagged for "
        "inspection, not automatically removed or classified as "
        "occupied-room labels."
    )
else:
    print("No valid CO₂ measurements available for peak analysis.")


# ============================================================
# 12. HOURLY CO2 PATTERNS IN MELBOURNE TIME
# ============================================================

print_section("HOURLY CO2 ANALYSIS")

if not co2_data.empty:
    hourly_summary = (
        co2_data.groupby(
            ["sensorid", "local_hour"], as_index=False
        )
        .agg(
            reading_count=(co2_column, "count"),
            mean_co2_ppm=(co2_column, "mean"),
            median_co2_ppm=(co2_column, "median"),
            standard_deviation_ppm=(co2_column, "std"),
            maximum_co2_ppm=(co2_column, "max"),
        )
    )

    save_csv(hourly_summary, "hourly_co2_summary_melbourne.csv")

    plt.figure(figsize=(10, 5))

    for sensor_id, sensor_hourly in hourly_summary.groupby("sensorid"):
        plt.plot(
            sensor_hourly["local_hour"],
            sensor_hourly["mean_co2_ppm"],
            marker="o",
            label=sensor_id,
        )

    plt.title("Mean CO₂ by Hour of Day (Melbourne Local Time)")
    plt.xlabel("Hour of day (0–23)")
    plt.ylabel("Mean CO₂ (ppm)")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    save_plot("hourly_mean_co2_melbourne.png")

    # Day-of-week summaries use Melbourne local dates.
    daily_pattern = (
        co2_data.groupby(
            ["sensorid", "local_day_of_week"], as_index=False
        )
        .agg(
            reading_count=(co2_column, "count"),
            mean_co2_ppm=(co2_column, "mean"),
            median_co2_ppm=(co2_column, "median"),
        )
    )

    save_csv(daily_pattern, "day_of_week_co2_summary.csv")


# ============================================================
# 13. CORRELATION ANALYSIS
# ============================================================

print_section("CORRELATION ANALYSIS")

correlation_variables = [
    variable for variable in CORE_VARIABLES
    if variable in available_variables
]

correlation_columns = [
    available_variables[variable]
    for variable in correlation_variables
]

correlation_data = long_df[correlation_columns].copy()
correlation_data.columns = correlation_variables

correlation_matrix = correlation_data.corr(numeric_only=True)

correlation_matrix.to_csv(
    SUMMARY_DIR / "environmental_correlation_matrix.csv"
)

print(correlation_matrix)

plt.figure(figsize=(7, 6))
plt.imshow(correlation_matrix, aspect="auto")
plt.colorbar(label="Correlation coefficient")
plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right",
)
plt.yticks(
    range(len(correlation_matrix.index)),
    correlation_matrix.index,
)
plt.title("Environmental Measurement Correlations")
save_plot("environmental_correlation_matrix.png")


# ============================================================
# 14. SUMMARY AND COMPLETION
# ============================================================

print_section("EDA V5 SUMMARY")

summary = pd.DataFrame([{
    "input_file": str(INPUT_FILE),
    "rows_after_timestamp_validation": len(df),
    "invalid_timestamps_removed": invalid_timestamp_count,
    "number_of_sensors": df["sensorid"].nunique(),
    "number_of_eda_segments": len(segments_df),
    "segments_at_least_30_readings": (
        int(segments_df["long_enough"].sum())
        if not segments_df.empty else 0
    ),
    "rows_in_long_segments": len(long_df),
    "first_timestamp_utc": (
        df["measurement_time_utc"].min()
        if not df.empty else pd.NaT
    ),
    "last_timestamp_utc": (
        df["measurement_time_utc"].max()
        if not df.empty else pd.NaT
    ),
    "output_directory": str(OUTPUT_DIR),
}])

save_csv(summary, "eda_v5_run_summary.csv")

print(summary.to_string(index=False))

print("\nEDA VERSION UPDATE COMPLETED SUCCESSFULLY.")
print(f"CSV summaries: {SUMMARY_DIR}")
print(f"Plots: {PLOT_DIR}")
print(f"All outputs: {OUTPUT_DIR}")
