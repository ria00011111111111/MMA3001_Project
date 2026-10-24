
"""
MMA3001 - File 05: Model Development and Anomaly Validation

Purpose
-------
1. Prepare cleaned sensor data for modelling.
2. Audit missing measurements and missing time intervals.
3. Interpolate only short, bracketed gaps in optional measurements.
4. Detect environmental anomaly candidates using Isolation Forest.
5. Compare anomalies with CO2 thresholds and sudden environmental changes.
6. Investigate persistent anomaly runs.
7. Preserve all original observations and avoid claiming verified occupancy.

Important:
- An anomaly is not automatically an error.
- CO2 is an indirect environmental indicator, not verified occupancy.
- No supervised occupancy model is trained without confirmed occupancy labels.
- All interpolation and analysis are restricted to sensor-time segments.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "outputs" / "features" / "feature_engineered_data.csv"
OUTPUT_DIR = ROOT / "outputs" / "modeldevelopment"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_INTERVAL_MIN = 10
MAX_INTERPOLATION_GAP_MIN = 20
MIN_SEGMENT_ROWS = 30

RANDOM_STATE = 42

CORE_VARIABLES = [
    "co2",
    "temperature",
    "humidity",
    "pressure",
]

OPTIONAL_VARIABLES = [
    "lvoc",
    "formaldehyde",
    "pm1",
    "pm25",
    "pm4",
    "pm10",
]

ALL_VARIABLES = CORE_VARIABLES + OPTIONAL_VARIABLES

# These are exploratory indicators, not occupancy labels.
CO2_THRESHOLDS = [600, 800, 1000]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find the input file:\n{INPUT_FILE}\n"
        "Run File 03 first to generate the feature-engineered data."
    )

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("The input dataset is empty.")

print("=" * 65)
print("MMA3001 FILE 05: MODEL DEVELOPMENT")
print("=" * 65)
print(f"Input file: {INPUT_FILE}")
print(f"Original rows: {len(df):,}")
print(f"Original columns: {len(df.columns):,}")


# ============================================================
# 3. IDENTIFY TIMESTAMP, SENSOR AND SEGMENT COLUMNS
# ============================================================

def find_column(data, candidates, description, required=True):
    """Find the first available column from a list of alternatives."""
    for candidate in candidates:
        if candidate in data.columns:
            return candidate

    if required:
        raise KeyError(
            f"Could not find the {description} column.\n"
            f"Tried: {candidates}\n"
            f"Available columns: {list(data.columns)}"
        )

    return None


time_col = find_column(
    df,
    [
        "measurement_time_utc",
        "measurement_time",
        "timestamp",
        "time",
    ],
    "measurement timestamp",
)

sensor_col = find_column(
    df,
    ["sensorid", "sensor_id", "sensor"],
    "sensor identifier",
)

segment_col = find_column(
    df,
    ["time_segment_id", "segment_id"],
    "time segment identifier",
)

df[time_col] = pd.to_datetime(
    df[time_col],
    errors="coerce",
    utc=True,
)

invalid_timestamps = int(df[time_col].isna().sum())

df = df.loc[df[time_col].notna()].copy()

for variable in ALL_VARIABLES:
    if variable in df.columns:
        df[variable] = pd.to_numeric(
            df[variable],
            errors="coerce",
        )

missing_core_columns = [
    variable
    for variable in CORE_VARIABLES
    if variable not in df.columns
]

if missing_core_columns:
    raise KeyError(
        "Required core measurements are missing: "
        + ", ".join(missing_core_columns)
    )

# Ensure each sensor's segments remain independent.
df = df.sort_values(
    [sensor_col, segment_col, time_col]
).reset_index(drop=True)

print(f"Rows after invalid timestamps removed: {len(df):,}")
print(f"Invalid timestamps excluded: {invalid_timestamps:,}")
print(f"Sensors: {df[sensor_col].nunique():,}")
print(f"Segments: {df[segment_col].nunique():,}")


# ============================================================
# 4. AUDIT DUPLICATE TIMESTAMPS AND MISSING MEASUREMENTS
# ============================================================

duplicate_mask = df.duplicated(
    subset=[sensor_col, time_col],
    keep=False,
)

duplicate_audit = (
    df.loc[duplicate_mask, [sensor_col, segment_col, time_col]]
    .groupby([sensor_col, time_col], dropna=False)
    .agg(
        number_of_records=(time_col, "size"),
        segments=(segment_col, "nunique"),
    )
    .reset_index()
)

duplicate_audit.to_csv(
    OUTPUT_DIR / "duplicate_timestamp_audit.csv",
    index=False,
)

missing_rows = []

for variable in ALL_VARIABLES:
    if variable not in df.columns:
        continue

    missing_rows.append(
        {
            "variable": variable,
            "total_rows": len(df),
            "missing_values": int(df[variable].isna().sum()),
            "missing_percentage": round(
                100 * df[variable].isna().mean(), 3
            ),
            "available_values": int(df[variable].notna().sum()),
        }
    )

missing_audit = pd.DataFrame(missing_rows)

missing_audit.to_csv(
    OUTPUT_DIR / "missing_measurement_audit.csv",
    index=False,
)

print("\nMissing measurement audit:")
print(missing_audit.to_string(index=False))


# ============================================================
# 5. AUDIT TIME INTERVALS WITHIN EACH SEGMENT
# ============================================================

time_gap_rows = []

for (sensor, segment), group in df.groupby(
    [sensor_col, segment_col],
    sort=False,
):
    times = group[time_col].sort_values()

    intervals = times.diff().dt.total_seconds().div(60)
    intervals = intervals.dropna()

    if intervals.empty:
        continue

    time_gap_rows.append(
        {
            "sensorid": sensor,
            "time_segment_id": segment,
            "number_of_readings": len(group),
            "median_interval_min": intervals.median(),
            "mean_interval_min": intervals.mean(),
            "minimum_interval_min": intervals.min(),
            "maximum_interval_min": intervals.max(),
            "intervals_over_expected": int(
                (intervals > EXPECTED_INTERVAL_MIN * 1.5).sum()
            ),
            "intervals_over_30_min": int(
                (intervals > 30).sum()
            ),
            "estimated_missing_steps": int(
                np.maximum(
                    np.round(
                        intervals / EXPECTED_INTERVAL_MIN
                    ).fillna(1) - 1,
                    0,
                ).sum()
            ),
        }
    )

time_gap_audit = pd.DataFrame(time_gap_rows)

time_gap_audit.to_csv(
    OUTPUT_DIR / "missing_time_step_audit.csv",
    index=False,
)


# ============================================================
# 6. INTERPOLATE ONLY SHORT, BRACKETED OPTIONAL GAPS
# ============================================================

# Original values are preserved in separate columns.
# No new timestamps or synthetic records are created.
interpolation_rows = []

for variable in ALL_VARIABLES:
    if variable not in df.columns:
        continue

    df[f"{variable}_original"] = df[variable]
    df[f"{variable}_imputed"] = False

for (sensor, segment), group in df.groupby(
    [sensor_col, segment_col],
    sort=False,
):
    group_indices = group.index

    group_times = (
        df.loc[group_indices, time_col]
        .astype("int64")
        .to_numpy()
        / 60_000_000_000
    )

    for variable in ALL_VARIABLES:
        if variable not in df.columns:
            continue

        original = df.loc[group_indices, variable].copy()

        # Do not interpolate core variables unnecessarily.
        # They have already been checked for missingness.
        if variable in CORE_VARIABLES:
            imputed_count = 0
        else:
            values = original.to_numpy(dtype=float, copy=True)
            missing_positions = np.flatnonzero(np.isnan(values))
            imputed_count = 0

            valid_positions = np.flatnonzero(~np.isnan(values))

            for position in missing_positions:
                previous = valid_positions[
                    valid_positions < position
                ]
                following = valid_positions[
                    valid_positions > position
                ]

                # Both sides must have real measurements.
                if len(previous) == 0 or len(following) == 0:
                    continue

                left = previous[-1]
                right = following[0]

                gap_minutes = group_times[right] - group_times[left]

                # Interpolate only when the missing observation is
                # bracketed by nearby real readings.
                if (
                    gap_minutes <= MAX_INTERPOLATION_GAP_MIN
                    and gap_minutes > 0
                ):
                    fraction = (
                        (group_times[position] - group_times[left])
                        / gap_minutes
                    )

                    values[position] = (
                        values[left]
                        + fraction * (values[right] - values[left])
                    )

                    df.at[group_indices[position], variable] = (
                        values[position]
                    )
                    df.at[
                        group_indices[position],
                        f"{variable}_imputed",
                    ] = True

                    imputed_count += 1

        interpolation_rows.append(
            {
                "sensorid": sensor,
                "time_segment_id": segment,
                "variable": variable,
                "original_missing": int(original.isna().sum()),
                "imputed_values": imputed_count,
                "remaining_missing": int(
                    df.loc[group_indices, variable].isna().sum()
                ),
            }
        )

interpolation_audit = pd.DataFrame(interpolation_rows)

interpolation_audit.to_csv(
    OUTPUT_DIR / "interpolation_audit.csv",
    index=False,
)

interpolation_summary = (
    interpolation_audit.groupby("variable", as_index=False)
    .agg(
        original_missing=("original_missing", "sum"),
        imputed_values=("imputed_values", "sum"),
        remaining_missing=("remaining_missing", "sum"),
    )
)

interpolation_summary.to_csv(
    OUTPUT_DIR / "interpolation_summary.csv",
    index=False,
)

print("\nInterpolation summary:")
print(interpolation_summary.to_string(index=False))


# ============================================================
# 7. CREATE MODELLING FEATURES
# ============================================================

# Cyclical time features use UTC, keeping the timezone explicit.
hour_utc = (
    df[time_col].dt.hour
    + df[time_col].dt.minute / 60
)

day_of_week = df[time_col].dt.dayofweek

df["hour_sin"] = np.sin(2 * np.pi * hour_utc / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour_utc / 24)

df["day_of_week_sin"] = np.sin(
    2 * np.pi * day_of_week / 7
)
df["day_of_week_cos"] = np.cos(
    2 * np.pi * day_of_week / 7
)

df["interval_minutes"] = (
    df.groupby([sensor_col, segment_col])[time_col]
    .diff()
    .dt.total_seconds()
    .div(60)
)

# Create lag and change features without crossing segment boundaries.
for variable in CORE_VARIABLES:
    df[f"{variable}_lag1"] = (
        df.groupby([sensor_col, segment_col])[variable]
        .shift(1)
    )

    df[f"{variable}_change"] = (
        df[variable] - df[f"{variable}_lag1"]
    )

    df[f"{variable}_missing_original"] = (
        df[f"{variable}_original"].isna()
        if f"{variable}_original" in df.columns
        else df[variable].isna()
    )


# ============================================================
# 8. DETECT ENVIRONMENTAL ANOMALY CANDIDATES
# ============================================================

# Standardising variables prevents the units of CO2 or pressure
# from dominating the distance-based anomaly model.
#
# Isolation Forest is fitted separately to each sensor segment.
# Its flags are candidates for investigation, not errors to delete.

df["environmental_anomaly_flag"] = False
df["environmental_anomaly_score"] = np.nan

anomaly_summary_rows = []

for (sensor, segment), group in df.groupby(
    [sensor_col, segment_col],
    sort=False,
):
    valid_mask = group[CORE_VARIABLES].notna().all(axis=1)
    valid_indices = group.index[valid_mask]

    if len(valid_indices) < MIN_SEGMENT_ROWS:
        anomaly_summary_rows.append(
            {
                "sensorid": sensor,
                "time_segment_id": segment,
                "number_of_rows": len(group),
                "rows_modelled": len(valid_indices),
                "anomaly_count": 0,
                "anomaly_rate_percent": 0.0,
                "status": "Too few complete rows",
            }
        )
        continue

    X = df.loc[valid_indices, CORE_VARIABLES].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_STATE,
    )

    model.fit(X_scaled)

    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    flags = predictions == -1

    df.loc[valid_indices, "environmental_anomaly_flag"] = flags
    df.loc[valid_indices, "environmental_anomaly_score"] = scores

    anomaly_count = int(flags.sum())

    anomaly_summary_rows.append(
        {
            "sensorid": sensor,
            "time_segment_id": segment,
            "number_of_rows": len(group),
            "rows_modelled": len(valid_indices),
            "anomaly_count": anomaly_count,
            "anomaly_rate_percent": round(
                100 * anomaly_count / len(valid_indices), 3
            ),
            "status": "Model fitted",
        }
    )

anomaly_summary = pd.DataFrame(anomaly_summary_rows)

anomaly_summary.to_csv(
    OUTPUT_DIR / "anomaly_detection_summary.csv",
    index=False,
)

df.loc[df["environmental_anomaly_flag"]].to_csv(
    OUTPUT_DIR / "environmental_anomaly_candidates.csv",
    index=False,
)

print("\nAnomaly detection summary:")
print(
    f"Flagged readings: "
    f"{int(df['environmental_anomaly_flag'].sum()):,}"
)
print(
    f"Flagged percentage: "
    f"{100 * df['environmental_anomaly_flag'].mean():.2f}%"
)


# ============================================================
# 8B. COMPARATIVE ISOLATION FOREST: MODEL PER SENSOR
# ============================================================

# Compare the existing segment-level detector with a model
# fitted across eligible segments for each sensor.
#
# This is a comparative experiment, not an assumed improvement.
# Original measurements and baseline flags remain unchanged.

COMPARATIVE_MIN_ROWS = 30

df["environmental_anomaly_sensor_model_flag"] = False
df["environmental_anomaly_sensor_model_score"] = np.nan

sensor_model_rows = []

for sensor, group in df.groupby(sensor_col, sort=False):
    # Use the same four core variables as the baseline model.
    # Only complete readings are eligible.
    valid_mask = group[CORE_VARIABLES].notna().all(axis=1)
    valid_indices = group.index[valid_mask]

    if len(valid_indices) < COMPARATIVE_MIN_ROWS:
        sensor_model_rows.append({
            "sensorid": sensor,
            "rows_available": len(group),
            "rows_modelled": len(valid_indices),
            "flagged_readings": 0,
            "flag_rate_pct": np.nan,
            "status": "Too few complete rows",
        })
        continue

    X = df.loc[valid_indices, CORE_VARIABLES].astype(float)

    # Standardise across this sensor's eligible readings.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_STATE,
    )

    model.fit(X_scaled)

    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)
    flags = predictions == -1

    # Store results separately from the baseline model.
    df.loc[
        valid_indices,
        "environmental_anomaly_sensor_model_flag",
    ] = flags

    df.loc[
        valid_indices,
        "environmental_anomaly_sensor_model_score",
    ] = scores

    sensor_model_rows.append({
        "sensorid": sensor,
        "rows_available": len(group),
        "rows_modelled": len(valid_indices),
        "flagged_readings": int(flags.sum()),
        "flag_rate_pct": round(
            100 * flags.mean(), 3
        ),
        "status": "Model fitted",
    })

sensor_model_summary = pd.DataFrame(sensor_model_rows)

sensor_model_summary.to_csv(
    OUTPUT_DIR / "sensor_level_model_summary.csv",
    index=False,
)

# ------------------------------------------------------------
# Compare baseline and comparative model on identical rows
# ------------------------------------------------------------

comparison_mask = (
    df[CORE_VARIABLES].notna().all(axis=1)
    & df["environmental_anomaly_score"].notna()
    & df["environmental_anomaly_sensor_model_score"].notna()
)

comparison_df = df.loc[comparison_mask].copy()

baseline_flags = comparison_df[
    "environmental_anomaly_flag"
].astype(bool)

sensor_flags = comparison_df[
    "environmental_anomaly_sensor_model_flag"
].astype(bool)

n_compared = len(comparison_df)

both = int((baseline_flags & sensor_flags).sum())
baseline_only = int((baseline_flags & ~sensor_flags).sum())
sensor_only = int((~baseline_flags & sensor_flags).sum())
neither = int((~baseline_flags & ~sensor_flags).sum())

model_comparison_summary = pd.DataFrame([
    {
        "metric": "Readings compared",
        "value": n_compared,
    },
    {
        "metric": "Baseline segment-level flags",
        "value": int(baseline_flags.sum()),
    },
    {
        "metric": "Comparative sensor-level flags",
        "value": int(sensor_flags.sum()),
    },
    {
        "metric": "Flagged by both models",
        "value": both,
    },
    {
        "metric": "Baseline model only",
        "value": baseline_only,
    },
    {
        "metric": "Sensor-level model only",
        "value": sensor_only,
    },
    {
        "metric": "Flagged by neither model",
        "value": neither,
    },
    {
        "metric": "Agreement percentage",
        "value": (
            100 * (both + neither) / n_compared
            if n_compared else np.nan
        ),
    },
])

model_comparison_summary.to_csv(
    OUTPUT_DIR / "isolation_forest_model_comparison.csv",
    index=False,
)

# Export individual readings where the models disagree.
comparison_df["model_disagreement"] = (
    baseline_flags != sensor_flags
)

disagreements = comparison_df.loc[
    comparison_df["model_disagreement"]
].copy()

disagreement_columns = [
    sensor_col,
    segment_col,
    time_col,
    *CORE_VARIABLES,
    "environmental_anomaly_flag",
    "environmental_anomaly_score",
    "environmental_anomaly_sensor_model_flag",
    "environmental_anomaly_sensor_model_score",
]

disagreement_columns = [
    c for c in disagreement_columns
    if c in disagreements.columns
]

disagreements.to_csv(
    OUTPUT_DIR / "isolation_forest_model_disagreements.csv",
    index=False,
)

print("\nComparative Isolation Forest model complete.")
print("\nSensor-level model summary:")
print(sensor_model_summary.to_string(index=False))

print("\nModel comparison:")
print(model_comparison_summary.to_string(index=False))

print(
    "\nModel disagreement readings:",
    len(disagreements),
)

# ============================================================
# 9. COMPARE ANOMALIES AGAINST CO2 THRESHOLDS
# ============================================================

threshold_rows = []

for threshold in CO2_THRESHOLDS:
    above = df["co2"] >= threshold
    above_count = int(above.sum())

    flagged_above = int(
        (
            above
            & df["environmental_anomaly_flag"]
        ).sum()
    )

    threshold_rows.append(
        {
            "co2_threshold_ppm": threshold,
            "readings_at_or_above_threshold": above_count,
            "percentage_of_all_readings": round(
                100 * above_count / len(df), 4
            ),
            "threshold_readings_flagged_as_anomalies": flagged_above,
            "percentage_threshold_readings_flagged": (
                round(100 * flagged_above / above_count, 2)
                if above_count
                else np.nan
            ),
        }
    )

threshold_audit = pd.DataFrame(threshold_rows)

threshold_audit.to_csv(
    OUTPUT_DIR / "high_co2_anomaly_audit.csv",
    index=False,
)

print("\nCO2 threshold comparison:")
print(threshold_audit.to_string(index=False))


# ============================================================
# 10. COMPARE ENVIRONMENTAL VARIABLES FOR FLAGGED VS UNFLAGGED
# ============================================================

comparison_rows = []

for variable in ALL_VARIABLES:
    if variable not in df.columns:
        continue

    for flagged, label in [
        (False, "Not flagged"),
        (True, "Flagged"),
    ]:
        values = df.loc[
            df["environmental_anomaly_flag"] == flagged,
            variable,
        ].dropna()

        comparison_rows.append(
            {
                "variable": variable,
                "group": label,
                "valid_readings": len(values),
                "mean": values.mean() if len(values) else np.nan,
                "median": values.median() if len(values) else np.nan,
                "standard_deviation": (
                    values.std() if len(values) > 1 else np.nan
                ),
                "minimum": values.min() if len(values) else np.nan,
                "maximum": values.max() if len(values) else np.nan,
            }
        )

variable_comparison = pd.DataFrame(comparison_rows)

variable_comparison.to_csv(
    OUTPUT_DIR / "anomaly_variable_comparison.csv",
    index=False,
)


# ============================================================
# 11. AUDIT SUDDEN ENVIRONMENTAL CHANGES
# ============================================================

change_rows = []

for variable in CORE_VARIABLES:
    change_col = f"{variable}_change"

    if change_col not in df.columns:
        continue

    absolute_change = df[change_col].abs().dropna()

    change_rows.append(
        {
            "variable": variable,
            "valid_change_count": len(absolute_change),
            "median_absolute_change": (
                absolute_change.median()
                if len(absolute_change)
                else np.nan
            ),
            "95th_percentile_absolute_change": (
                absolute_change.quantile(0.95)
                if len(absolute_change)
                else np.nan
            ),
            "99th_percentile_absolute_change": (
                absolute_change.quantile(0.99)
                if len(absolute_change)
                else np.nan
            ),
            "maximum_absolute_change": (
                absolute_change.max()
                if len(absolute_change)
                else np.nan
            ),
        }
    )

change_audit = pd.DataFrame(change_rows)

change_audit.to_csv(
    OUTPUT_DIR / "environmental_change_audit.csv",
    index=False,
)

# Export the largest CO2 changes for manual review.
df["absolute_co2_change"] = df["co2_change"].abs()

largest_changes = (
    df.loc[df["co2_change"].notna()]
    .nlargest(100, "absolute_co2_change")
    .copy()
)

largest_changes.to_csv(
    OUTPUT_DIR / "largest_co2_changes.csv",
    index=False,
)


# ============================================================
# 11B. ROBUST CO2-CHANGE ANOMALY VALIDATION
# ============================================================

# Independent detector for unusually large CO2 changes.
# Only valid within-segment intervals of <=20 minutes are used.
# These are candidates for review, not confirmed errors or occupancy.

MAX_CHANGE_INTERVAL_MIN = MAX_INTERPOLATION_GAP_MIN
ROBUST_Z_THRESHOLD = 6.0

df["co2_change_anomaly_flag"] = False
df["co2_change_robust_z"] = np.nan
df["co2_change_median_segment"] = np.nan
df["co2_change_mad_segment"] = np.nan

robust_change_rows = []

for (sensor, segment), group in df.groupby(
    [sensor_col, segment_col],
    sort=False,
):
    changes = pd.to_numeric(
        group["co2_change"], errors="coerce"
    )
    intervals = pd.to_numeric(
        group["interval_minutes"], errors="coerce"
    )

    eligible = (
        changes.notna()
        & intervals.gt(0)
        & intervals.le(MAX_CHANGE_INTERVAL_MIN)
    )

    valid_changes = changes.loc[eligible]

    if len(valid_changes) < MIN_SEGMENT_ROWS:
        robust_change_rows.append({
            "sensorid": sensor,
            "time_segment_id": segment,
            "eligible_changes": len(valid_changes),
            "median_change_ppm": np.nan,
            "mad_change_ppm": np.nan,
            "robust_scale_ppm": np.nan,
            "flagged_changes": 0,
            "status": "Too few eligible changes",
        })
        continue

    median_change = valid_changes.median()
    mad_change = (valid_changes - median_change).abs().median()

    flags = pd.Series(False, index=valid_changes.index)
    robust_z = pd.Series(np.nan, index=valid_changes.index)

    if np.isfinite(mad_change) and mad_change > 0:
        # Equivalent to (change - median) / (1.4826 * MAD).
        robust_z = (
            0.67448975
            * (valid_changes - median_change)
            / mad_change
        )

        flags = robust_z.abs() > ROBUST_Z_THRESHOLD
        status = "Evaluated"

        df.loc[valid_changes.index, "co2_change_robust_z"] = robust_z
    else:
        # Zero MAD does not provide a usable robust scale.
        status = "Not evaluated: zero MAD"

    df.loc[
        valid_changes.index,
        "co2_change_median_segment",
    ] = median_change

    df.loc[
        valid_changes.index,
        "co2_change_mad_segment",
    ] = mad_change

    df.loc[
        flags.index[flags],
        "co2_change_anomaly_flag",
    ] = True

    robust_scale = (
        1.4826 * mad_change
        if np.isfinite(mad_change)
        else np.nan
    )

    robust_change_rows.append({
        "sensorid": sensor,
        "time_segment_id": segment,
        "eligible_changes": len(valid_changes),
        "median_change_ppm": median_change,
        "mad_change_ppm": mad_change,
        "robust_scale_ppm": robust_scale,
        "flagged_changes": int(flags.sum()),
        "status": status,
    })

robust_change_summary = pd.DataFrame(robust_change_rows)

robust_change_summary.to_csv(
    OUTPUT_DIR / "robust_co2_change_summary.csv",
    index=False,
)

# Compare both detectors on identical eligible rows.
comparison_mask = (
    df["co2_change"].notna()
    & df["interval_minutes"].gt(0)
    & df["interval_minutes"].le(MAX_CHANGE_INTERVAL_MIN)
)

comparison_data = df.loc[comparison_mask]

isolation_flags = comparison_data[
    "environmental_anomaly_flag"
].astype(bool)

change_flags = comparison_data[
    "co2_change_anomaly_flag"
].astype(bool)

method_comparison = pd.DataFrame([
    {
        "comparison": "Flagged by both methods",
        "readings": int((isolation_flags & change_flags).sum()),
    },
    {
        "comparison": "Isolation Forest only",
        "readings": int((isolation_flags & ~change_flags).sum()),
    },
    {
        "comparison": "Robust CO2-change detector only",
        "readings": int((~isolation_flags & change_flags).sum()),
    },
    {
        "comparison": "Flagged by neither method",
        "readings": int((~isolation_flags & ~change_flags).sum()),
    },
])

method_comparison.to_csv(
    OUTPUT_DIR / "anomaly_method_comparison.csv",
    index=False,
)

co2_change_candidates = df.loc[
    df["co2_change_anomaly_flag"]
].copy()

co2_change_candidates.to_csv(
    OUTPUT_DIR / "robust_co2_change_candidates.csv",
    index=False,
)


# ============================================================
# 11C. CO2 CHANGE REVERSAL ANALYSIS
# ============================================================

# Example: 446 -> 1387 -> 770 ppm.
# A large change followed by an opposite-direction change is
# worth investigating, but does not prove a sensor fault.
#
# The follow-up must be within the same sensor and segment,
# and each interval must be positive and <=20 minutes.

MIN_REVERSAL_CHANGE_PPM = 200
MAX_REVERSAL_STEPS = 2

df["co2_reversal_candidate"] = False
df["co2_reversal_followup_change"] = np.nan
df["co2_reversal_steps"] = np.nan

for (sensor, segment), group in df.groupby(
    [sensor_col, segment_col],
    sort=False,
):
    group = group.sort_values(time_col)

    indices = group.index.to_list()
    values = pd.to_numeric(
        group["co2_original"]
        if "co2_original" in group.columns
        else group["co2"],
        errors="coerce",
    ).to_numpy()

    times = group[time_col].to_list()

    if len(indices) < 3:
        continue

    changes = np.full(len(indices), np.nan)
    intervals = np.full(len(indices), np.nan)

    for i in range(1, len(indices)):
        if pd.isna(values[i]) or pd.isna(values[i - 1]):
            continue

        changes[i] = values[i] - values[i - 1]
        intervals[i] = (
            times[i] - times[i - 1]
        ).total_seconds() / 60

    for i in range(1, len(indices)):
        first_change = changes[i]
        first_interval = intervals[i]

        if not np.isfinite(first_change):
            continue

        if (
            not np.isfinite(first_interval)
            or first_interval <= 0
            or first_interval > MAX_CHANGE_INTERVAL_MIN
            or abs(first_change) < MIN_REVERSAL_CHANGE_PPM
        ):
            continue

        for steps in range(1, MAX_REVERSAL_STEPS + 1):
            j = i + steps

            if j >= len(indices):
                break

            # Check every interval between the spike and reversal.
            path_intervals = intervals[i + 1:j + 1]

            if (
                len(path_intervals) != steps
                or not np.isfinite(path_intervals).all()
                or (path_intervals <= 0).any()
                or (path_intervals > MAX_CHANGE_INTERVAL_MIN).any()
            ):
                break

            followup_change = changes[j]

            if not np.isfinite(followup_change):
                break

            # Opposite signs indicate movement back in the other direction.
            if first_change * followup_change < 0:
                row_index = indices[i]

                df.at[row_index, "co2_reversal_candidate"] = True
                df.at[
                    row_index,
                    "co2_reversal_followup_change",
                ] = followup_change
                df.at[
                    row_index,
                    "co2_reversal_steps",
                ] = steps

                break

reversal_candidates = df.loc[
    df["co2_reversal_candidate"]
].copy()

reversal_candidates["absolute_co2_change"] = (
    reversal_candidates["co2_change"].abs()
)

reversal_candidates = reversal_candidates.sort_values(
    "absolute_co2_change",
    ascending=False,
)

reversal_candidates.to_csv(
    OUTPUT_DIR / "co2_change_reversal_candidates.csv",
    index=False,
)


# ============================================================
# 11D. COMPACT REVIEW TABLE
# ============================================================

# Add Melbourne local timestamps for easier interpretation.
df["measurement_time_melbourne"] = (
    df[time_col].dt.tz_convert("Australia/Melbourne")
)

review_columns = [
    sensor_col,
    segment_col,
    time_col,
    "measurement_time_melbourne",
    "interval_minutes",
    "co2_lag1",
    "co2",
    "co2_change",
    "co2_change_robust_z",
    "co2_change_anomaly_flag",
    "co2_reversal_candidate",
    "co2_reversal_followup_change",
    "environmental_anomaly_flag",
    "environmental_anomaly_score",
    "temperature",
    "humidity",
    "pressure",
]

review_columns = [
    c for c in review_columns if c in df.columns
]

candidate_review = df.loc[
    df["co2_change_anomaly_flag"]
    | df["co2_reversal_candidate"]
    | df["environmental_anomaly_flag"]
].copy()

candidate_review["absolute_co2_change"] = (
    candidate_review["co2_change"].abs()
)

candidate_review = candidate_review.sort_values(
    "absolute_co2_change",
    ascending=False,
)

candidate_review[review_columns].to_csv(
    OUTPUT_DIR / "co2_anomaly_review.csv",
    index=False,
)

print("\nRobust CO2-change validation:")
print(
    "Robust change candidates:",
    int(df["co2_change_anomaly_flag"].sum()),
)
print(
    "Reversal candidates:",
    int(df["co2_reversal_candidate"].sum()),
)

print("\nAnomaly method comparison:")
print(method_comparison.to_string(index=False))

print("\nLargest CO2 candidates for review:")
print(
    candidate_review[review_columns]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 12. SUMMARISE ANOMALIES BY SENSOR
# ============================================================

sensor_summary = (
    df.groupby(sensor_col, as_index=False)
    .agg(
        number_of_readings=("co2", "size"),
        flagged_readings=("environmental_anomaly_flag", "sum"),
        co2_mean=("co2", "mean"),
        co2_median=("co2", "median"),
        co2_standard_deviation=("co2", "std"),
        co2_maximum=("co2", "max"),
        temperature_mean=("temperature", "mean"),
        humidity_mean=("humidity", "mean"),
        pressure_mean=("pressure", "mean"),
    )
)

sensor_summary["anomaly_rate_percent"] = (
    100
    * sensor_summary["flagged_readings"]
    / sensor_summary["number_of_readings"]
)

sensor_summary.to_csv(
    OUTPUT_DIR / "anomaly_sensor_summary.csv",
    index=False,
)


# ============================================================
# 13. VALIDATE SEGMENT INTEGRITY AND ANOMALY COUNTS
# ============================================================

segment_validation_rows = []

for (sensor, segment), group in df.groupby(
    [sensor_col, segment_col],
    sort=False,
):
    intervals = (
        group[time_col]
        .sort_values()
        .diff()
        .dt.total_seconds()
        .div(60)
        .dropna()
    )

    segment_validation_rows.append(
        {
            "sensorid": sensor,
            "time_segment_id": segment,
            "number_of_readings": len(group),
            "start_time_utc": group[time_col].min(),
            "end_time_utc": group[time_col].max(),
            "anomaly_count": int(
                group["environmental_anomaly_flag"].sum()
            ),
            "anomaly_rate_percent": round(
                100 * group["environmental_anomaly_flag"].mean(),
                3,
            ),
            "duplicate_timestamps": int(
                group[time_col].duplicated().sum()
            ),
            "maximum_interval_min": (
                intervals.max() if len(intervals) else np.nan
            ),
            "intervals_over_30_min": int(
                (intervals > 30).sum()
            ),
            "segment_has_minimum_rows": (
                len(group) >= MIN_SEGMENT_ROWS
            ),
        }
    )

segment_validation = pd.DataFrame(segment_validation_rows)

segment_validation.to_csv(
    OUTPUT_DIR / "anomaly_segment_validation.csv",
    index=False,
)


# ============================================================
# 14. IDENTIFY CONSECUTIVE ANOMALY RUNS
# ============================================================

# A run means consecutive rows in the same segment are flagged.
# It does not automatically mean the environment was occupied.
# Duration is calculated from timestamps, not by multiplying
# row counts by the nominal sampling interval.

run_rows = []

for (sensor, segment), group in df.groupby(
    [sensor_col, segment_col],
    sort=False,
):
    group = group.sort_values(time_col).copy()

    flags = group["environmental_anomaly_flag"].to_numpy()
    indices = group.index.to_numpy()

    run_start = None

    for position, is_flagged in enumerate(flags):
        if is_flagged and run_start is None:
            run_start = position

        is_last = position == len(flags) - 1

        if run_start is not None and (
            not is_flagged or is_last
        ):
            run_end = (
                position
                if is_flagged and is_last
                else position - 1
            )

            run_indices = indices[run_start:run_end + 1]
            run = df.loc[run_indices]

            run_rows.append(
                {
                    "sensorid": sensor,
                    "time_segment_id": segment,
                    "start_time_utc": run[time_col].min(),
                    "end_time_utc": run[time_col].max(),
                    "number_of_consecutive_flagged_readings": len(run),
                    "duration_minutes": (
                        (
                            run[time_col].max()
                            - run[time_col].min()
                        ).total_seconds() / 60
                    ),
                    "mean_co2": run["co2"].mean(),
                    "maximum_co2": run["co2"].max(),
                    "mean_temperature": run["temperature"].mean(),
                    "mean_humidity": run["humidity"].mean(),
                }
            )

            run_start = None

persistent_runs = pd.DataFrame(run_rows)

if not persistent_runs.empty:
    persistent_runs = persistent_runs.sort_values(
        [
            "number_of_consecutive_flagged_readings",
            "duration_minutes",
        ],
        ascending=False,
    )

persistent_runs.to_csv(
    OUTPUT_DIR / "persistent_anomaly_runs.csv",
    index=False,
)


# ============================================================
# 15. CREATE A MODEL DEVELOPMENT QUALITY REPORT
# ============================================================

quality_rows = [
    {
        "check": "Input rows before timestamp validation",
        "result": "PASS",
        "value": len(pd.read_csv(INPUT_FILE, usecols=[time_col])),
        "interpretation": "Rows present in the input feature dataset.",
    },
    {
        "check": "Invalid timestamps",
        "result": "PASS" if invalid_timestamps == 0 else "REVIEW",
        "value": invalid_timestamps,
        "interpretation": "Invalid timestamps were excluded before modelling.",
    },
    {
        "check": "Duplicate sensor timestamps",
        "result": (
            "PASS"
            if not duplicate_mask.any()
            else "REVIEW"
        ),
        "value": int(duplicate_mask.sum()),
        "interpretation": (
            "Audit identifies repeated timestamps. "
            "This script does not automatically delete them."
        ),
    },
    {
        "check": "Rows retained for modelling",
        "result": "INFO",
        "value": len(df),
        "interpretation": "Rows retained after timestamp validation.",
    },
    {
        "check": "Number of sensors",
        "result": "INFO",
        "value": df[sensor_col].nunique(),
        "interpretation": "Sensors may cover different time periods.",
    },
    {
        "check": "Number of time segments",
        "result": "INFO",
        "value": df[segment_col].nunique(),
        "interpretation": "Features and anomaly detection are segment-local.",
    },
    {
        "check": "Flagged environmental anomalies",
        "result": "REVIEW",
        "value": int(df["environmental_anomaly_flag"].sum()),
        "interpretation": (
            "Candidates for investigation, not automatically bad data."
        ),
    },
    {
        "check": "Verified occupancy labels",
        "result": "NOT ESTABLISHED",
        "value": 0,
        "interpretation": (
            "No verified occupancy labels have been established. "
            "Do not claim supervised occupancy prediction or accuracy."
        ),
    },
]

quality_report = pd.DataFrame(quality_rows)

quality_report.to_csv(
    OUTPUT_DIR / "model_development_quality_report.csv",
    index=False,
)


# ============================================================
# 16. SAVE MODEL-READY DATA
# ============================================================

output_file = OUTPUT_DIR / "model_ready_sensor_data.csv"

df.to_csv(output_file, index=False)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("MODEL DEVELOPMENT COMPLETE")
print("=" * 65)

print(f"Rows saved: {len(df):,}")
print(f"Sensors: {df[sensor_col].nunique():,}")
print(f"Segments: {df[segment_col].nunique():,}")

print(
    "Flagged anomalies: "
    f"{int(df['environmental_anomaly_flag'].sum()):,} "
    f"({100 * df['environmental_anomaly_flag'].mean():.2f}%)"
)

print(
    "Duplicate sensor timestamps identified: "
    f"{int(duplicate_mask.sum()):,}"
)

print(
    "CO2 readings at or above 600 ppm: "
    f"{int((df['co2'] >= 600).sum()):,}"
)

print(
    "CO2 readings at or above 800 ppm: "
    f"{int((df['co2'] >= 800).sum()):,}"
)

print(
    "CO2 readings at or above 1000 ppm: "
    f"{int((df['co2'] >= 1000).sum()):,}"
)

print(f"\nOutput folder: {OUTPUT_DIR}")
print(f"Model-ready dataset: {output_file}")

print("\nGenerated audit files:")
for filename in [
    "missing_measurement_audit.csv",
    "missing_time_step_audit.csv",
    "interpolation_audit.csv",
    "interpolation_summary.csv",
    "anomaly_detection_summary.csv",
    "environmental_anomaly_candidates.csv",
    "duplicate_timestamp_audit.csv",
    "high_co2_anomaly_audit.csv",
    "anomaly_variable_comparison.csv",
    "environmental_change_audit.csv",
    "largest_co2_changes.csv",
    "anomaly_sensor_summary.csv",
    "anomaly_segment_validation.csv",
    "persistent_anomaly_runs.csv",
    "model_development_quality_report.csv",
    "model_ready_sensor_data.csv",
    "robust_co2_change_summary.csv",
    "anomaly_method_comparison.csv",
    "robust_co2_change_candidates.csv",
    "co2_change_reversal_candidates.csv",
    "co2_anomaly_review.csv",
]:
    print(f"  - {filename}")

print(
    "\nIMPORTANT: Environmental anomaly flags are not occupancy labels. "
    "Do not remove flagged readings without reviewing the evidence."
)



# ============================================================
# ADDITIONAL MODEL VALIDATION
# Investigate Isolation Forest flag rates and data quality
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


def run_model_diagnostics(output_dir):
    output_dir = Path(output_dir)
    model_file = output_dir / "model_ready_sensor_data.csv"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model-ready file not found: {model_file}"
        )

    df = pd.read_csv(model_file).copy()

    required = [
        "sensorid",
        "time_segment_id",
        "environmental_anomaly_flag",
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert flags safely, including CSV strings such as "True".
    def to_bool(series):
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    df["if_flag"] = to_bool(df["environmental_anomaly_flag"])

    if "co2_change_anomaly_flag" in df.columns:
        df["co2_change_flag"] = to_bool(
            df["co2_change_anomaly_flag"]
        )
    else:
        df["co2_change_flag"] = False

    # --------------------------------------------------------
    # 1. Overall anomaly summary
    # --------------------------------------------------------
    summary = pd.DataFrame([{
        "total_readings": len(df),
        "isolation_forest_flags": int(df["if_flag"].sum()),
        "isolation_forest_flag_rate_pct": (
            df["if_flag"].mean() * 100 if len(df) else np.nan
        ),
        "robust_co2_change_flags": int(
            df["co2_change_flag"].sum()
        ),
        "robust_co2_change_flag_rate_pct": (
            df["co2_change_flag"].mean() * 100
            if len(df) else np.nan
        ),
        "both_methods_flagged": int(
            (df["if_flag"] & df["co2_change_flag"]).sum()
        ),
        "isolation_forest_only": int(
            (df["if_flag"] & ~df["co2_change_flag"]).sum()
        ),
        "co2_change_only": int(
            (~df["if_flag"] & df["co2_change_flag"]).sum()
        ),
        "neither_method_flagged": int(
            (~df["if_flag"] & ~df["co2_change_flag"]).sum()
        ),
    }])

    summary.to_csv(
        output_dir / "validation_overall_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. Flag rate by sensor
    # --------------------------------------------------------
    sensor_summary = (
        df.groupby("sensorid", dropna=False)
        .agg(
            readings=("if_flag", "size"),
            if_flagged=("if_flag", "sum"),
            co2_change_flagged=("co2_change_flag", "sum"),
        )
        .reset_index()
    )

    sensor_summary["if_flag_rate_pct"] = (
        100 * sensor_summary["if_flagged"]
        / sensor_summary["readings"]
    )

    sensor_summary["co2_change_flag_rate_pct"] = (
        100 * sensor_summary["co2_change_flagged"]
        / sensor_summary["readings"]
    )

    sensor_summary.to_csv(
        output_dir / "validation_by_sensor.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Flag rate by sensor and segment
    # --------------------------------------------------------
    segment_summary = (
        df.groupby(
            ["sensorid", "time_segment_id"],
            dropna=False,
        )
        .agg(
            readings=("if_flag", "size"),
            if_flagged=("if_flag", "sum"),
            co2_change_flagged=("co2_change_flag", "sum"),
        )
        .reset_index()
    )

    segment_summary["if_flag_rate_pct"] = (
        100 * segment_summary["if_flagged"]
        / segment_summary["readings"]
    )

    segment_summary["co2_change_flag_rate_pct"] = (
        100 * segment_summary["co2_change_flagged"]
        / segment_summary["readings"]
    )

    segment_summary["high_if_flag_rate"] = (
        segment_summary["if_flag_rate_pct"] >= 50
    )

    segment_summary = segment_summary.sort_values(
        ["if_flag_rate_pct", "readings"],
        ascending=[False, False],
    )

    segment_summary.to_csv(
        output_dir / "validation_by_segment.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. Missingness and imputation audit
    # --------------------------------------------------------
    audit_rows = []

    variables = [
        "co2", "temperature", "humidity", "pressure",
        "formaldehyde", "lvoc", "pm1", "pm25", "pm4", "pm10",
    ]

    for variable in variables:
        if variable not in df.columns:
            continue

        missing_col = f"{variable}_missing_original"
        imputed_col = f"{variable}_imputed"
        original_col = f"{variable}_original"

        if missing_col in df.columns:
            originally_missing = to_bool(df[missing_col])
        elif original_col in df.columns:
            originally_missing = df[original_col].isna()
        else:
            originally_missing = pd.Series(
                False, index=df.index
            )

        if imputed_col in df.columns:
            imputed = to_bool(df[imputed_col])
        else:
            imputed = pd.Series(False, index=df.index)

        audit_rows.append({
            "variable": variable,
            "rows": len(df),
            "missing_in_model_ready": int(df[variable].isna().sum()),
            "originally_missing": int(originally_missing.sum()),
            "imputed": int(imputed.sum()),
            "original_missing_pct": (
                100 * originally_missing.mean()
                if len(df) else np.nan
            ),
            "model_ready_missing_pct": (
                100 * df[variable].isna().mean()
                if len(df) else np.nan
            ),
        })

    pd.DataFrame(audit_rows).to_csv(
        output_dir / "validation_missingness.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Most unusual CO2 readings
    # --------------------------------------------------------
    if "co2_change" in df.columns:
        df["absolute_co2_change_diagnostic"] = (
            pd.to_numeric(df["co2_change"], errors="coerce").abs()
        )
    elif "absolute_co2_change" in df.columns:
        df["absolute_co2_change_diagnostic"] = pd.to_numeric(
            df["absolute_co2_change"], errors="coerce"
        )
    else:
        df["absolute_co2_change_diagnostic"] = np.nan

    review_columns = [
        c for c in [
            "sensorid",
            "time_segment_id",
            "measurement_time_melbourne",
            "co2",
            "co2_lag1",
            "co2_change",
            "co2_change_robust_z",
            "co2_change_anomaly_flag",
            "co2_reversal_candidate",
            "environmental_anomaly_flag",
            "environmental_anomaly_score",
            "temperature",
            "humidity",
            "pressure",
            "formaldehyde_missing",
            "pm1",
            "pm25",
            "pm4",
            "pm10",
        ]
        if c in df.columns
    ]

    review = df.sort_values(
        "absolute_co2_change_diagnostic",
        ascending=False,
        na_position="last",
    )

    review[review_columns].head(100).to_csv(
        output_dir / "validation_top_100_co2_changes.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. Consecutive Isolation Forest anomaly runs
    # --------------------------------------------------------
    sort_columns = [
        c for c in [
            "sensorid",
            "time_segment_id",
            "measurement_time_melbourne",
        ]
        if c in df.columns
    ]

    ordered = df.sort_values(sort_columns).copy()

    group_columns = ["sensorid", "time_segment_id"]

    # Start a new run whenever the flag changes or the group changes.
    flag_change = (
        ordered["if_flag"]
        .ne(ordered.groupby(group_columns, dropna=False)["if_flag"].shift())
    )

    group_change = pd.Series(False, index=ordered.index)
    for col in group_columns:
        group_change |= ordered[col].ne(ordered[col].shift())

    ordered["_run_id"] = (flag_change | group_change).cumsum()

    anomaly_rows = ordered[ordered["if_flag"]].copy()

    if not anomaly_rows.empty:
        runs = (
            anomaly_rows.groupby(
                ["sensorid", "time_segment_id", "_run_id"],
                dropna=False,
            )
            .agg(
                consecutive_flagged_readings=("if_flag", "size"),
                first_time=("measurement_time_melbourne", "min"),
                last_time=("measurement_time_melbourne", "max"),
                mean_co2=("co2", "mean"),
                maximum_co2=("co2", "max"),
                mean_temperature=("temperature", "mean"),
                mean_humidity=("humidity", "mean"),
            )
            .reset_index()
        )

        runs = runs.sort_values(
            "consecutive_flagged_readings",
            ascending=False,
        )

        runs.to_csv(
            output_dir / "validation_if_anomaly_runs.csv",
            index=False,
        )
    else:
        pd.DataFrame(columns=[
            "sensorid",
            "time_segment_id",
            "consecutive_flagged_readings",
            "first_time",
            "last_time",
            "mean_co2",
            "maximum_co2",
            "mean_temperature",
            "mean_humidity",
        ]).to_csv(
            output_dir / "validation_if_anomaly_runs.csv",
            index=False,
        )

    print("\nAdditional model validation complete.")
    print(f"Rows analysed: {len(df):,}")
    print(
        "Isolation Forest flag rate: "
        f"{df['if_flag'].mean() * 100:.2f}%"
    )
    print(
        "Robust CO2-change flag rate: "
        f"{df['co2_change_flag'].mean() * 100:.2f}%"
    )
    print(f"Diagnostic files saved to: {output_dir}")


# Run this AFTER your existing File 05 pipeline has saved
# model_ready_sensor_data.csv.
if __name__ == "__main__":
    run_model_diagnostics(
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "modeldevelopment"
    )

# ============================================================
# FINAL VALIDATION: CONSOLIDATED QUALITY AND MODEL REPORT
# ============================================================

def create_final_validation_report(output_dir):
    """
    Validate the saved model-ready dataset and summarise:
    - dataset integrity
    - missing measurements
    - sensor and segment coverage
    - time gaps within segments
    - anomaly model flag rates and agreement
    - CO2-change candidates
    - known modelling limitations

    This is diagnostic only. It does not delete readings or
    claim that unlabelled anomalies are confirmed errors.
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path

    output_dir = Path(output_dir)
    data_path = output_dir / "model_ready_sensor_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Model-ready dataset not found: {data_path}"
        )

    data = pd.read_csv(data_path)

    # Identify expected columns without changing the dataset.
    timestamp_col = next(
        (c for c in [
            "measurement_time_utc",
            "timestamp_utc",
            "timestamp",
            "measurement_time",
        ] if c in data.columns),
        None,
    )

    sensor_col = next(
        (c for c in ["sensorid", "sensor_id"] if c in data.columns),
        None,
    )

    segment_col = next(
        (c for c in ["time_segment_id", "segment_id"] if c in data.columns),
        None,
    )

    if timestamp_col is None:
        raise KeyError(
            "Could not identify the timestamp column in "
            "model_ready_sensor_data.csv."
        )

    if sensor_col is None:
        raise KeyError(
            "Could not identify the sensor ID column."
        )

    if segment_col is None:
        raise KeyError(
            "Could not identify the time-segment column."
        )

    data[timestamp_col] = pd.to_datetime(
        data[timestamp_col], errors="coerce", utc=True
    )

    report_rows = []

    def add_check(category, metric, value, interpretation):
        report_rows.append({
            "category": category,
            "metric": metric,
            "value": value,
            "interpretation": interpretation,
        })

    # --------------------------------------------------------
    # A. DATASET INTEGRITY
    # --------------------------------------------------------
    add_check(
        "Dataset",
        "Total rows",
        len(data),
        "Number of model-ready readings.",
    )

    add_check(
        "Dataset",
        "Total columns",
        len(data.columns),
        "Number of available variables and features.",
    )

    add_check(
        "Dataset",
        "Unique sensors",
        data[sensor_col].nunique(dropna=True),
        "Number of sensors represented.",
    )

    add_check(
        "Dataset",
        "Unique segments",
        data[segment_col].nunique(dropna=True),
        "Number of segments represented.",
    )

    invalid_timestamps = int(data[timestamp_col].isna().sum())

    add_check(
        "Integrity",
        "Invalid timestamps",
        invalid_timestamps,
        "Ideally zero; investigate invalid timestamps.",
    )

    duplicate_sensor_times = int(
        data.duplicated(
            subset=[sensor_col, timestamp_col], keep=False
        ).sum()
    )

    add_check(
        "Integrity",
        "Rows involved in duplicate sensor timestamps",
        duplicate_sensor_times,
        "Ideally zero. Duplicate rows are counted, not removed.",
    )

    missing_sensor_ids = int(data[sensor_col].isna().sum())
    missing_segment_ids = int(data[segment_col].isna().sum())

    add_check(
        "Integrity",
        "Missing sensor IDs",
        missing_sensor_ids,
        "Ideally zero.",
    )

    add_check(
        "Integrity",
        "Missing segment IDs",
        missing_segment_ids,
        "Ideally zero.",
    )

    # --------------------------------------------------------
    # B. MISSING MEASUREMENTS
    # --------------------------------------------------------
    measurement_columns = [
        c for c in [
            "co2", "temperature", "humidity", "pressure",
            "lvoc", "formaldehyde", "pm1", "pm25", "pm4", "pm10",
        ]
        if c in data.columns
    ]

    for column in measurement_columns:
        missing_count = int(data[column].isna().sum())
        missing_pct = (
            100 * missing_count / len(data) if len(data) else np.nan
        )

        add_check(
            "Missing data",
            f"{column}: missing readings",
            missing_count,
            f"{missing_pct:.2f}% missing. Review before using this variable.",
        )

    # Report imputed values where corresponding flags exist.
    imputation_flags = [
        c for c in data.columns
        if c.endswith("_imputed")
    ]

    for flag_col in imputation_flags:
        flag_values = data[flag_col].fillna(False).astype(bool)

        add_check(
            "Imputation",
            f"{flag_col}: flagged readings",
            int(flag_values.sum()),
            "Interpolated readings; distinguish from original measurements.",
        )

    # --------------------------------------------------------
    # C. TIME-INTERVAL AND SEGMENT CHECKS
    # --------------------------------------------------------
    valid_time = data.dropna(
        subset=[timestamp_col, sensor_col, segment_col]
    ).copy()

    valid_time = valid_time.sort_values(
        [sensor_col, segment_col, timestamp_col]
    )

    valid_time["_interval_minutes_validation"] = (
        valid_time.groupby(
            [sensor_col, segment_col], sort=False
        )[timestamp_col]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    intervals = valid_time["_interval_minutes_validation"].dropna()

    add_check(
        "Time intervals",
        "Intervals checked",
        len(intervals),
        "Intervals are calculated only within the same sensor and segment.",
    )

    add_check(
        "Time intervals",
        "Non-positive intervals",
        int((intervals <= 0).sum()),
        "Investigate duplicates or timestamps that are not strictly increasing.",
    )

    add_check(
        "Time intervals",
        "Intervals over 20 minutes",
        int((intervals > 20).sum()),
        "Potential gaps. Do not interpret them as continuous observations.",
    )

    add_check(
        "Time intervals",
        "Intervals of 0 to 8 minutes",
        int(((intervals > 0) & (intervals < 8)).sum()),
        "Short intervals; inspect if unexpected for the sensor sampling rate.",
    )

    add_check(
        "Time intervals",
        "Intervals from 8 to 12 minutes",
        int(((intervals >= 8) & (intervals <= 12)).sum()),
        "Intervals in the expected approximate 10-minute sampling window.",
    )

    # Check whether each segment contains more than one sensor.
    sensors_per_segment = (
        valid_time.groupby(segment_col)[sensor_col].nunique()
    )

    add_check(
        "Segments",
        "Segments containing multiple sensors",
        int((sensors_per_segment > 1).sum()),
        "Review if segments are intended to belong to a single sensor.",
    )

    # --------------------------------------------------------
    # D. ANOMALY MODEL COMPARISON
    # --------------------------------------------------------
    baseline_flag_col = "environmental_anomaly_flag"
    sensor_flag_col = "environmental_anomaly_sensor_model_flag"

    if baseline_flag_col in data.columns:
        baseline_flags = (
            data[baseline_flag_col].fillna(False).astype(bool)
        )

        add_check(
            "Anomaly models",
            "Segment-level flagged readings",
            int(baseline_flags.sum()),
            "Candidate anomalies, not confirmed sensor faults.",
        )

        add_check(
            "Anomaly models",
            "Segment-level flag rate (%)",
            round(100 * baseline_flags.mean(), 3) if len(data) else np.nan,
            "Descriptive rate; not an accuracy score.",
        )

    if sensor_flag_col in data.columns:
        sensor_flags = (
            data[sensor_flag_col].fillna(False).astype(bool)
        )

        add_check(
            "Anomaly models",
            "Sensor-level flagged readings",
            int(sensor_flags.sum()),
            "Candidate anomalies, not confirmed sensor faults.",
        )

        add_check(
            "Anomaly models",
            "Sensor-level flag rate (%)",
            round(100 * sensor_flags.mean(), 3) if len(data) else np.nan,
            "Descriptive rate; not an accuracy score.",
        )

    if (
        baseline_flag_col in data.columns
        and sensor_flag_col in data.columns
    ):
        # Compare rows for which both model outputs are available.
        comparable = data[
            baseline_flag_col
        ].notna() & data[sensor_flag_col].notna()

        base = data.loc[
            comparable, baseline_flag_col
        ].astype(bool)

        sensor = data.loc[
            comparable, sensor_flag_col
        ].astype(bool)

        n = len(base)
        agreement = (
            100 * (base == sensor).mean() if n else np.nan
        )

        add_check(
            "Anomaly models",
            "Rows compared between models",
            n,
            "Rows with both model flag outputs available.",
        )

        add_check(
            "Anomaly models",
            "Model agreement (%)",
            round(agreement, 3) if n else np.nan,
            "Agreement measures consistency, not correctness.",
        )

        add_check(
            "Anomaly models",
            "Model disagreement readings",
            int((base != sensor).sum()),
            "Useful candidates for visual inspection in File 06.",
        )

    # --------------------------------------------------------
    # E. CO2 CHANGE CANDIDATES
    # --------------------------------------------------------
    change_flag_col = "co2_change_anomaly_flag"
    reversal_col = "co2_reversal_candidate"

    if change_flag_col in data.columns:
        change_flags = (
            data[change_flag_col].fillna(False).astype(bool)
        )

        add_check(
            "CO2 changes",
            "Robust CO2-change candidates",
            int(change_flags.sum()),
            "Sudden CO2 changes requiring contextual review.",
        )

    if reversal_col in data.columns:
        reversal_flags = (
            data[reversal_col].fillna(False).astype(bool)
        )

        add_check(
            "CO2 changes",
            "Reversal candidates",
            int(reversal_flags.sum()),
            "Potential transient patterns; not proof of sensor artefacts.",
        )

    # --------------------------------------------------------
    # F. EXPLICIT LIMITATIONS
    # --------------------------------------------------------
    add_check(
        "Limitations",
        "Verified anomaly labels",
        "Unavailable",
        "Model accuracy cannot be directly measured against verified anomalies.",
    )

    add_check(
        "Limitations",
        "Verified occupancy labels",
        "Unavailable",
        "Do not report occupancy prediction accuracy or treat anomaly flags as occupancy.",
    )

    add_check(
        "Limitations",
        "Automatic deletion of anomaly candidates",
        "Not performed",
        "Preserve observations unless independent evidence justifies exclusion.",
    )

    report = pd.DataFrame(report_rows)

    report_path = output_dir / "final_validation_report.csv"
    report.to_csv(report_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL MODEL VALIDATION REPORT")
    print("=" * 65)
    print(report.to_string(index=False))
    print(f"\nConsolidated report saved to: {report_path}")


# Run after the model-ready dataset has been saved.
if __name__ == "__main__":
    create_final_validation_report(OUTPUT_DIR)