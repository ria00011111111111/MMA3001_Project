
"""
File 04: Occupancy Analysis and Data Verification
MMA3001 Room Occupancy Project

Purpose:
    1. Audit the feature-engineered dataset.
    2. Check time-series and sensor-data consistency.
    3. Investigate CO2 patterns without claiming confirmed occupancy.
    4. Check particulate-matter measurement relationships.
    5. Identify potential occupancy labels if present.
    6. Save a verification report for the next modelling stage.

Run from the project root:
    python src/04_occupancyanalysis.py

Important:
    Passing these checks does not prove that the sensors are accurate
    or that occupancy can be predicted. Ground-truth occupancy data
    is required to validate occupancy predictions.
"""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "outputs" / "features" / "feature_engineered_data.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "occupancy"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Provisional CO2 indicators, not validated occupancy thresholds.
CO2_THRESHOLDS = [600, 800, 1000]

# Expected approximate sensor sampling interval.
EXPECTED_INTERVAL_MIN = 10

# Large gaps within a segment may indicate a segmentation problem.
MAX_SEGMENT_GAP_MIN = 30

# Possible occupancy-label column names.
LABEL_PATTERN = re.compile(
    r"occup|presence|people_count|person_count|"
    r"ground.?truth|room.?state|target_label",
    re.IGNORECASE,
)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def save_table(data, filename, index=False):
    """Save a DataFrame to the output directory."""
    path = OUTPUT_DIR / filename
    data.to_csv(path, index=index)
    print(f"Saved: {path.name}")


def numeric_column(df, name):
    """Return a numeric column, or None if it does not exist."""
    if name not in df.columns:
        return None

    return pd.to_numeric(df[name], errors="coerce")


def safe_percentage(numerator, denominator):
    """Calculate a percentage without dividing by zero."""
    if denominator == 0:
        return np.nan

    return 100.0 * numerator / denominator


def report_check(checks, name, passed, detail):
    """Record a check without implying scientific validation."""
    checks.append({
        "check": name,
        "status": "PASS" if passed else "REVIEW",
        "details": str(detail),
    })


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("MMA3001 - OCCUPANCY ANALYSIS AND VERIFICATION")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}\n"
        "Run File 03 first."
    )

df = pd.read_csv(INPUT_FILE)
input_rows = len(df)

required_columns = [
    "sensorid",
    "measurement_time_utc",
    "time_segment_id",
]

missing_required = [
    col for col in required_columns if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}"
    )

df["measurement_time_utc"] = pd.to_datetime(
    df["measurement_time_utc"],
    errors="coerce",
    utc=True,
)

invalid_timestamps = int(
    df["measurement_time_utc"].isna().sum()
)

df = df.dropna(
    subset=[
        "sensorid",
        "measurement_time_utc",
        "time_segment_id",
    ]
).copy()

df = df.sort_values(
    ["sensorid", "time_segment_id", "measurement_time_utc"]
).reset_index(drop=True)

group_cols = ["sensorid", "time_segment_id"]

print(f"\nInput file: {INPUT_FILE}")
print(f"Input rows: {input_rows}")
print(f"Invalid timestamps removed: {invalid_timestamps}")
print(f"Rows retained: {len(df)}")
print(f"Sensors: {df['sensorid'].nunique()}")
print(f"Segments: {df.groupby(group_cols).ngroups}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 4. DATA INTEGRITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("DATA INTEGRITY CHECKS")
print("=" * 70)

checks = []

duplicate_mask = df.duplicated(
    subset=["sensorid", "measurement_time_utc"],
    keep=False,
)

duplicate_timestamp_rows = int(duplicate_mask.sum())

# Duplicate timestamps within a sensor can make time-series
# interpretation ambiguous. They are reported, not silently removed.
duplicate_details = (
    df.loc[duplicate_mask]
    .groupby("sensorid")
    .size()
    .rename("duplicate_timestamp_rows")
    .reset_index()
)

save_table(
    duplicate_details,
    "duplicate_timestamp_audit.csv",
)

report_check(
    checks,
    "No duplicate sensor timestamps",
    duplicate_timestamp_rows == 0,
    f"{duplicate_timestamp_rows} rows involved in duplicate timestamps.",
)

report_check(
    checks,
    "No invalid timestamps remain",
    df["measurement_time_utc"].notna().all(),
    f"{invalid_timestamps} invalid timestamps were removed.",
)

# Check the number of records and duration in each segment.
segment_rows = []

for (sensor_id, segment_id), group in df.groupby(
    group_cols, sort=False
):
    times = group["measurement_time_utc"].sort_values()
    intervals = times.diff().dt.total_seconds().div(60).dropna()

    segment_rows.append({
        "sensorid": sensor_id,
        "time_segment_id": segment_id,
        "record_count": len(group),
        "start_utc": times.min(),
        "end_utc": times.max(),
        "duration_hours": (
            (times.max() - times.min()).total_seconds() / 3600
        ),
        "median_interval_minutes": (
            intervals.median() if len(intervals) else np.nan
        ),
        "maximum_interval_minutes": (
            intervals.max() if len(intervals) else np.nan
        ),
        "intervals_over_30_minutes": int(
            (intervals > MAX_SEGMENT_GAP_MIN).sum()
        ),
        "intervals_within_8_to_12_minutes": int(
            intervals.between(8, 12).sum()
        ),
        "interval_count": len(intervals),
    })

segment_audit = pd.DataFrame(segment_rows)

save_table(
    segment_audit,
    "segment_integrity_audit.csv",
)

segments_with_large_gaps = int(
    (
        segment_audit["intervals_over_30_minutes"] > 0
    ).sum()
)

report_check(
    checks,
    "No gaps over 30 minutes within segments",
    segments_with_large_gaps == 0,
    (
        f"{segments_with_large_gaps} segments contain at least "
        "one within-segment interval over 30 minutes."
    ),
)

if "segment_long_enough" in df.columns:
    long_flag = df["segment_long_enough"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    short_segment_rows = int((~long_flag).sum())

    report_check(
        checks,
        "All retained rows are flagged as long segments",
        short_segment_rows == 0,
        f"{short_segment_rows} rows have a false/unknown long-segment flag.",
    )

print(f"Duplicate timestamp rows: {duplicate_timestamp_rows}")
print(f"Segments with gaps over 30 minutes: {segments_with_large_gaps}")


# ============================================================
# 5. FIND MEASUREMENT COLUMNS
# ============================================================

measurement_names = [
    "co2",
    "temperature",
    "humidity",
    "pressure",
    "lvoc",
    "formaldehyde",
    "pm1",
    "pm25",
    "pm4",
    "pm10",
]

available_measurements = []

for name in measurement_names:
    if name in df.columns:
        df[name] = pd.to_numeric(df[name], errors="coerce")
        available_measurements.append(name)

if "co2" not in available_measurements:
    raise ValueError(
        "The standardised 'co2' column is missing. "
        "Check File 03's output."
    )

measurement_rows = []

for name in available_measurements:
    values = df[name]
    valid = values.dropna()

    measurement_rows.append({
        "measurement": name,
        "total_rows": len(values),
        "valid_readings": int(values.notna().sum()),
        "missing_readings": int(values.isna().sum()),
        "missing_percentage": round(values.isna().mean() * 100, 3),
        "minimum": valid.min() if len(valid) else np.nan,
        "median": valid.median() if len(valid) else np.nan,
        "mean": valid.mean() if len(valid) else np.nan,
        "maximum": valid.max() if len(valid) else np.nan,
        "standard_deviation": valid.std() if len(valid) else np.nan,
    })

measurement_audit = pd.DataFrame(measurement_rows)

save_table(
    measurement_audit,
    "measurement_quality_audit.csv",
)


# ============================================================
# 6. CO2 THRESHOLD AND OUTLIER ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CO2 ANALYSIS")
print("=" * 70)

co2 = df["co2"]
valid_co2 = co2.dropna()

# Retain all positive CO2 readings, including high values.
# Flag them for review instead of automatically deleting them.
nonpositive_co2 = int((co2.notna() & (co2 <= 0)).sum())

df["co2_analysis_valid"] = co2.gt(0)
df.loc[~df["co2_analysis_valid"], "co2_analysis_valid"] = False

valid_co2 = co2.where(df["co2_analysis_valid"]).dropna()

co2_counts = []

for threshold in CO2_THRESHOLDS:
    count = int((valid_co2 > threshold).sum())

    co2_counts.append({
        "threshold_ppm": threshold,
        "readings_above_threshold": count,
        "percentage_of_valid_co2_readings": round(
            safe_percentage(count, len(valid_co2)), 4
        ),
    })

threshold_summary = pd.DataFrame(co2_counts)

save_table(
    threshold_summary,
    "co2_threshold_audit.csv",
)

# IQR outlier flag: this is a statistical screening method,
# not proof that a reading is erroneous.
if len(valid_co2) >= 4:
    q1 = valid_co2.quantile(0.25)
    q3 = valid_co2.quantile(0.75)
    iqr = q3 - q1
    upper_iqr = q3 + 1.5 * iqr
    lower_iqr = q1 - 1.5 * iqr

    df["co2_iqr_outlier"] = (
        df["co2_analysis_valid"]
        & ((co2 < lower_iqr) | (co2 > upper_iqr))
    )
else:
    lower_iqr = np.nan
    upper_iqr = np.nan
    df["co2_iqr_outlier"] = False

outliers = df.loc[
    df["co2_iqr_outlier"],
    [
        "sensorid",
        "time_segment_id",
        "measurement_time_utc",
        "co2",
    ],
].copy()

save_table(
    outliers,
    "co2_statistical_outlier_candidates.csv",
)

# Consecutive high readings are counted within each segment.
# A single high reading is not treated as a confirmed event.
df["co2_above_800"] = (
    df["co2_analysis_valid"] & (df["co2"] > 800)
)

df["co2_above_1000"] = (
    df["co2_analysis_valid"] & (df["co2"] > 1000)
)

persistence_rows = []

for (sensor_id, segment_id), group in df.groupby(
    group_cols, sort=False
):
    group = group.sort_values("measurement_time_utc")

    for threshold in [800, 1000]:
        above = group["co2"] > threshold
        above = above.fillna(False)

        # Each transition from False to True starts a run.
        run_id = above.ne(above.shift(fill_value=False)).cumsum()

        run_lengths = (
            group.loc[above]
            .assign(_run=run_id.loc[above].values)
            .groupby("_run")
            .size()
        )

        persistence_rows.append({
            "sensorid": sensor_id,
            "time_segment_id": segment_id,
            "threshold_ppm": threshold,
            "readings_above_threshold": int(above.sum()),
            "number_of_consecutive_runs": len(run_lengths),
            "longest_run_readings": (
                int(run_lengths.max()) if len(run_lengths) else 0
            ),
        })

persistence_summary = pd.DataFrame(persistence_rows)

save_table(
    persistence_summary,
    "co2_persistence_audit.csv",
)

print(f"Valid positive CO2 readings: {len(valid_co2)}")
print(f"Non-positive CO2 readings: {nonpositive_co2}")
print(f"IQR outlier candidates: {len(outliers)}")
print("\nCO2 threshold results:")
print(threshold_summary.to_string(index=False))

report_check(
    checks,
    "CO2 measurements are positive",
    nonpositive_co2 == 0,
    f"{nonpositive_co2} non-positive readings found.",
)

# Provisional descriptive indicators only.
def classify_co2(value):
    if pd.isna(value) or value <= 0:
        return "Unknown"
    if value < 600:
        return "Low CO2"
    if value < 800:
        return "Moderate CO2"
    if value < 1000:
        return "Elevated CO2"
    return "High CO2"


df["co2_indicator"] = df["co2"].apply(classify_co2)

indicator_order = [
    "Unknown",
    "Low CO2",
    "Moderate CO2",
    "Elevated CO2",
    "High CO2",
]

indicator_summary = (
    df["co2_indicator"]
    .value_counts()
    .reindex(indicator_order, fill_value=0)
    .rename_axis("co2_indicator")
    .reset_index(name="reading_count")
)

indicator_summary["percentage_of_all_records"] = (
    indicator_summary["reading_count"] / max(len(df), 1) * 100
).round(3)

save_table(
    indicator_summary,
    "co2_indicator_summary.csv",
)


# ============================================================
# 7. PARTICULATE MATTER RELATIONSHIPS
# ============================================================

print("\n" + "=" * 70)
print("PARTICULATE MATTER VERIFICATION")
print("=" * 70)

pm_columns = ["pm1", "pm25", "pm4", "pm10"]
available_pm = [name for name in pm_columns if name in df.columns]

pm_pair_rows = []

if len(available_pm) >= 2:
    pm_corr = df[available_pm].corr().round(4)

    save_table(
        pm_corr.reset_index(names="measurement"),
        "particulate_matter_correlations.csv",
    )

    for i in range(len(available_pm)):
        for j in range(i + 1, len(available_pm)):
            a = available_pm[i]
            b = available_pm[j]

            pair = df[[a, b]].dropna()
            equal_count = int((pair[a] == pair[b]).sum())

            pm_pair_rows.append({
                "measurement_a": a,
                "measurement_b": b,
                "paired_readings": len(pair),
                "identical_readings": equal_count,
                "percentage_identical": round(
                    safe_percentage(equal_count, len(pair)), 3
                ),
                "correlation": pair[a].corr(pair[b]),
            })

    pm_pair_summary = pd.DataFrame(pm_pair_rows)

    save_table(
        pm_pair_summary,
        "particulate_matter_pair_audit.csv",
    )

    # Check the expected order only if these variables represent
    # cumulative mass concentrations at increasing size cut-offs.
    # Confirm the device's measurement definitions before treating
    # a violation as a sensor error.
    expected_pairs = [
        ("pm1", "pm25"),
        ("pm25", "pm4"),
        ("pm4", "pm10"),
    ]

    ordering_rows = []

    for smaller, larger in expected_pairs:
        if smaller not in df.columns or larger not in df.columns:
            continue

        pair = df[[smaller, larger]].dropna()

        violations = pair[smaller] > pair[larger]

        ordering_rows.append({
            "smaller_fraction": smaller,
            "larger_fraction": larger,
            "paired_readings": len(pair),
            "ordering_violations": int(violations.sum()),
            "violation_percentage": round(
                safe_percentage(int(violations.sum()), len(pair)), 3
            ),
            "interpretation": (
                "Review measurement definitions and sensor behaviour; "
                "not automatically a fault."
            ),
        })

    ordering_summary = pd.DataFrame(ordering_rows)

    save_table(
        ordering_summary,
        "particulate_matter_ordering_audit.csv",
    )

    print("\nParticulate matter pair comparisons:")
    print(pm_pair_summary.to_string(index=False))

    if not ordering_summary.empty:
        print("\nParticulate matter ordering checks:")
        print(ordering_summary.to_string(index=False))

else:
    print("Insufficient particulate matter columns for pairwise analysis.")


# ============================================================
# 8. SENSOR OVERLAP AND COMPARABILITY
# ============================================================

print("\n" + "=" * 70)
print("SENSOR COMPARABILITY")
print("=" * 70)

sensor_rows = []

for sensor_id, group in df.groupby("sensorid"):
    sensor_rows.append({
        "sensorid": sensor_id,
        "records": len(group),
        "start_utc": group["measurement_time_utc"].min(),
        "end_utc": group["measurement_time_utc"].max(),
        "segments": group["time_segment_id"].nunique(),
        "co2_mean": group["co2"].mean(),
        "co2_median": group["co2"].median(),
        "co2_standard_deviation": group["co2"].std(),
        "temperature_mean": (
            group["temperature"].mean()
            if "temperature" in df.columns else np.nan
        ),
        "humidity_mean": (
            group["humidity"].mean()
            if "humidity" in df.columns else np.nan
        ),
    })

sensor_summary = pd.DataFrame(sensor_rows)

save_table(
    sensor_summary,
    "sensor_comparability_audit.csv",
)

# Check whether the overall time ranges overlap.
if len(sensor_summary) >= 2:
    overall_start = sensor_summary["start_utc"].max()
    overall_end = sensor_summary["end_utc"].min()
    overlap_exists = overall_start <= overall_end

    overlap_hours = (
        (overall_end - overall_start).total_seconds() / 3600
        if overlap_exists else 0
    )
else:
    overlap_exists = False
    overlap_hours = 0

print(sensor_summary.to_string(index=False))
print(f"\nOverall time ranges overlap: {overlap_exists}")
print(f"Overall overlap duration (hours): {overlap_hours:.2f}")

report_check(
    checks,
    "At least two sensors have overlapping overall time ranges",
    overlap_exists,
    (
        f"Overlap exists: {overlap_exists}; "
        f"overlap duration: {overlap_hours:.2f} hours. "
        "This does not establish that sensors measure the same room."
    ),
)


# ============================================================
# 9. CHECK FOR POSSIBLE OCCUPANCY LABELS
# ============================================================

print("\n" + "=" * 70)
print("OCCUPANCY LABEL AUDIT")
print("=" * 70)

possible_label_columns = [
    column for column in df.columns
    if LABEL_PATTERN.search(column)
]

label_report = pd.DataFrame({
    "possible_label_column": possible_label_columns
})

if possible_label_columns:
    label_report["non_missing_values"] = [
        int(df[column].notna().sum())
        for column in possible_label_columns
    ]

    label_report["unique_values"] = [
        int(df[column].nunique(dropna=True))
        for column in possible_label_columns
    ]

    label_report["sample_values"] = [
        " | ".join(
            df[column].dropna().astype(str).unique()[:10]
        )
        for column in possible_label_columns
    ]

    print("Possible label columns found:")
    print(label_report.to_string(index=False))
else:
    print(
        "No obvious occupancy-label columns were found "
        "in the feature-engineered dataset."
    )

save_table(
    label_report,
    "possible_occupancy_label_audit.csv",
)

report_check(
    checks,
    "Verified occupancy labels available",
    False,
    (
        "No verified occupancy labels have been established. "
        "Column-name matches, if any, must be manually checked."
    ),
)


# ============================================================
# 10. ENVIRONMENTAL CORRELATIONS
# ============================================================

correlation_columns = [
    name for name in available_measurements
    if df[name].notna().sum() > 1
]

if len(correlation_columns) >= 2:
    correlation_matrix = df[correlation_columns].corr().round(4)

    save_table(
        correlation_matrix.reset_index(names="measurement"),
        "environmental_correlations.csv",
    )


# ============================================================
# 11. PLOT CO2 BY SENSOR AND SEGMENT
# ============================================================

print("\nCreating plots...")

for sensor_id, sensor_df in df.groupby("sensorid", sort=False):
    fig, ax = plt.subplots(figsize=(14, 6))

    for segment_id, group in sensor_df.groupby(
        "time_segment_id", sort=False
    ):
        group = group.sort_values("measurement_time_utc")

        ax.plot(
            group["measurement_time_utc"],
            group["co2"],
            marker=".",
            markersize=2,
            linewidth=0.8,
        )

    for threshold in CO2_THRESHOLDS:
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1,
            label=f"{threshold} ppm",
        )

    ax.set_title(
        f"CO2 Measurements by Continuous Segment: Sensor {sensor_id}"
    )
    ax.set_xlabel("Measurement time (UTC)")
    ax.set_ylabel("CO2 concentration (ppm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(sensor_id))

    fig.savefig(
        PLOT_DIR / f"co2_verification_{safe_id}.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 12. SAVE ANALYSED DATA AND VERIFICATION REPORT
# ============================================================

save_table(
    df,
    "occupancy_analysis.csv",
)

checks_df = pd.DataFrame(checks)

save_table(
    checks_df,
    "verification_report.csv",
)

summary = pd.DataFrame([{
    "input_file": str(INPUT_FILE),
    "input_rows": input_rows,
    "output_rows": len(df),
    "invalid_timestamps_removed": invalid_timestamps,
    "unique_sensors": df["sensorid"].nunique(),
    "unique_segments": df.groupby(group_cols).ngroups,
    "duplicate_timestamp_rows": duplicate_timestamp_rows,
    "segments_with_gaps_over_30_minutes": segments_with_large_gaps,
    "valid_positive_co2_readings": len(valid_co2),
    "co2_iqr_outlier_candidates": len(outliers),
    "possible_label_columns_found": len(possible_label_columns),
    "verified_occupancy_labels_available": False,
    "supervised_occupancy_training_validated": False,
}])

save_table(
    summary,
    "occupancy_analysis_quality_summary.csv",
)

print("\n" + "=" * 70)
print("VERIFICATION REPORT")
print("=" * 70)
print(checks_df.to_string(index=False))

print("\n" + "=" * 70)
print("FILE 04 COMPLETED")
print("=" * 70)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Verification report: {OUTPUT_DIR / 'verification_report.csv'}")
print(f"Segment audit: {OUTPUT_DIR / 'segment_integrity_audit.csv'}")
print(f"PM audit: {OUTPUT_DIR / 'particulate_matter_pair_audit.csv'}")
print(f"Label audit: {OUTPUT_DIR / 'possible_occupancy_label_audit.csv'}")
print(f"Plots: {PLOT_DIR}")

print(
    "\nINTERPRETATION: This script audits data consistency and "
    "environmental patterns. It does not validate actual occupancy. "
    "Check the device's particulate-matter definitions and obtain "
    "ground-truth occupancy labels before claiming occupancy prediction."
)
