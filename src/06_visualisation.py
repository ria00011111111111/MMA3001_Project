
"""
MMA3001 FILE 06: VISUALISATION
================================

Input:
    outputs/modeldevelopment/model_ready_sensor_data.csv

Outputs:
    outputs/visualisation/

Purpose:
    Visualise CO2 trends, anomaly flags, model disagreements,
    and relationships between environmental variables.

Important:
    Anomaly flags are not verified sensor faults or occupancy labels.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "modeldevelopment"
    / "model_ready_sensor_data.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualisation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "measurement_time_utc"
SENSOR_COL = "sensorid"

BASELINE_FLAG = "environmental_anomaly_flag"
SENSOR_FLAG = "environmental_anomaly_sensor_model_flag"

CO2_THRESHOLDS = [600, 800, 1000]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run src/05_modeldevelopment.py first."
        )

    data = pd.read_csv(INPUT_FILE)

    required = [TIME_COL, SENSOR_COL, "co2"]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise KeyError(f"Required columns missing: {missing}")

    data[TIME_COL] = pd.to_datetime(
        data[TIME_COL],
        errors="coerce",
        utc=True,
    )

    data["co2"] = pd.to_numeric(data["co2"], errors="coerce")

    data = data.dropna(
        subset=[TIME_COL, SENSOR_COL, "co2"]
    ).copy()

    data = data.sort_values(
        [SENSOR_COL, TIME_COL]
    ).reset_index(drop=True)

    for flag in [BASELINE_FLAG, SENSOR_FLAG]:
        if flag in data.columns:
            data[flag] = data[flag].fillna(False).astype(bool)

    # Melbourne local time, including daylight-saving changes.
    data["measurement_time_melbourne"] = (
        data[TIME_COL].dt.tz_convert("Australia/Melbourne")
    )

    print("=" * 60)
    print("MMA3001 FILE 06: VISUALISATION")
    print("=" * 60)
    print(f"Rows loaded: {len(data):,}")
    print(f"Sensors: {data[SENSOR_COL].nunique()}")
    print(f"Output directory: {OUTPUT_DIR}")

    return data


# ============================================================
# 3. PLOT CO2 OVER TIME
# ============================================================

def plot_co2_trends(data):
    for sensor, group in data.groupby(SENSOR_COL):
        group = group.sort_values("measurement_time_melbourne")

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(
            group["measurement_time_melbourne"],
            group["co2"],
            linewidth=0.8,
            label="CO2",
        )

        for threshold in CO2_THRESHOLDS:
            ax.axhline(
                threshold,
                linestyle="--",
                linewidth=1,
                label=f"{threshold} ppm",
            )

        ax.set_title(f"CO2 Over Time: Sensor {sensor}")
        ax.set_xlabel("Melbourne local time")
        ax.set_ylabel("CO2 (ppm)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"co2_trend_sensor_{sensor}.png",
            dpi=200,
        )
        plt.close(fig)


# ============================================================
# 4. PLOT ANOMALY FLAGS OVER TIME
# ============================================================

def plot_anomaly_flags(data):
    if BASELINE_FLAG not in data.columns:
        print(f"Skipping plot: {BASELINE_FLAG} not found.")
        return

    for sensor, group in data.groupby(SENSOR_COL):
        group = group.sort_values("measurement_time_melbourne")

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(
            group["measurement_time_melbourne"],
            group["co2"],
            linewidth=0.7,
            label="CO2",
        )

        flagged = group[group[BASELINE_FLAG]]

        ax.scatter(
            flagged["measurement_time_melbourne"],
            flagged["co2"],
            s=12,
            marker="x",
            label="Segment-level anomaly candidate",
        )

        ax.set_title(f"CO2 and Anomaly Flags: Sensor {sensor}")
        ax.set_xlabel("Melbourne local time")
        ax.set_ylabel("CO2 (ppm)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"co2_anomalies_sensor_{sensor}.png",
            dpi=200,
        )
        plt.close(fig)


# ============================================================
# 5. COMPARE THE TWO MODELS
# ============================================================

def plot_model_disagreements(data):
    if (
        BASELINE_FLAG not in data.columns
        or SENSOR_FLAG not in data.columns
    ):
        print("Skipping model comparison plot: flags unavailable.")
        return

    for sensor, group in data.groupby(SENSOR_COL):
        group = group.sort_values("measurement_time_melbourne")

        disagreement = (
            group[BASELINE_FLAG] != group[SENSOR_FLAG]
        )

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(
            group["measurement_time_melbourne"],
            group["co2"],
            linewidth=0.7,
            label="CO2",
        )

        ax.scatter(
            group.loc[disagreement, "measurement_time_melbourne"],
            group.loc[disagreement, "co2"],
            s=16,
            marker="x",
            label="Models disagree",
        )

        ax.set_title(f"Model Disagreements: Sensor {sensor}")
        ax.set_xlabel("Melbourne local time")
        ax.set_ylabel("CO2 (ppm)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"model_disagreements_sensor_{sensor}.png",
            dpi=200,
        )
        plt.close(fig)


# ============================================================
# 6. PLOT ENVIRONMENTAL RELATIONSHIPS
# ============================================================

def plot_environmental_relationships(data):
    variables = [
        column for column in
        ["co2", "temperature", "humidity", "pressure"]
        if column in data.columns
    ]

    for sensor, group in data.groupby(SENSOR_COL):
        if len(variables) < 2:
            continue

        correlation = group[variables].corr()

        fig, ax = plt.subplots(figsize=(7, 6))

        image = ax.imshow(
            correlation,
            vmin=-1,
            vmax=1,
            aspect="auto",
        )

        ax.set_xticks(range(len(variables)))
        ax.set_xticklabels(variables, rotation=45, ha="right")
        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels(variables)

        for row in range(len(variables)):
            for col in range(len(variables)):
                value = correlation.iloc[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                )

        ax.set_title(f"Environmental Correlations: Sensor {sensor}")
        fig.colorbar(image, ax=ax, label="Pearson correlation")

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"environmental_correlations_sensor_{sensor}.png",
            dpi=200,
        )
        plt.close(fig)

def export_numerical_summaries(data):
    """Export sensor-level CO2, anomaly, change and time-gap summaries."""

    import numpy as np
    import pandas as pd

    summary_dir = OUTPUT_DIR
    summary_dir.mkdir(parents=True, exist_ok=True)

    time_col = "measurement_time_utc"
    sensor_col = "sensorid"
    co2_col = "co2"
    segment_col = "time_segment_id"

    # Work on a copy so the original data is unchanged.
    df = data.copy()

    df[time_col] = pd.to_datetime(
        df[time_col], errors="coerce", utc=True
    )
    df[co2_col] = pd.to_numeric(df[co2_col], errors="coerce")

    df = df.dropna(subset=[time_col, sensor_col, co2_col])
    df = df.sort_values([sensor_col, time_col])

    # Identify available anomaly flag columns.
    anomaly_columns = [
        col for col in [
            "environmental_anomaly_flag",
            "environmental_anomaly_sensor_model_flag"
        ]
        if col in df.columns
    ]

    def to_boolean(series):
        """Safely interpret boolean values stored in CSV columns."""
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)

        return (
            series.astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes"])
        )

    for col in anomaly_columns:
        df[col] = to_boolean(df[col])

    # --------------------------------------------------
    # A. SENSOR-LEVEL CO2 SUMMARY
    # --------------------------------------------------
    sensor_rows = []

    for sensor, group in df.groupby(sensor_col):
        co2 = group[co2_col]
        row = {
            "sensor_id": sensor,
            "readings": len(group),
            "first_reading_utc": group[time_col].min(),
            "last_reading_utc": group[time_col].max(),
            "co2_mean_ppm": co2.mean(),
            "co2_median_ppm": co2.median(),
            "co2_std_ppm": co2.std(),
            "co2_min_ppm": co2.min(),
            "co2_max_ppm": co2.max(),
            "co2_p95_ppm": co2.quantile(0.95),
            "readings_at_least_600_ppm": int((co2 >= 600).sum()),
            "readings_at_least_800_ppm": int((co2 >= 800).sum()),
            "readings_at_least_1000_ppm": int((co2 >= 1000).sum()),
        }

        for col in anomaly_columns:
            row[f"{col}_count"] = int(group[col].sum())
            row[f"{col}_rate_percent"] = (
                group[col].mean() * 100
            )

        sensor_rows.append(row)

    pd.DataFrame(sensor_rows).to_csv(
        summary_dir / "sensor_co2_summary.csv",
        index=False
    )

    # --------------------------------------------------
    # B. LARGEST CO2 CHANGES
    # Respect sensor and segment boundaries.
    # --------------------------------------------------
    grouping = [sensor_col]

    if segment_col in df.columns:
        grouping.append(segment_col)

    df["interval_minutes"] = (
        df.groupby(grouping, dropna=False)[time_col]
        .diff()
        .dt.total_seconds() / 60
    )

    df["co2_change_ppm"] = (
        df.groupby(grouping, dropna=False)[co2_col].diff()
    )

    df["absolute_co2_change_ppm"] = df["co2_change_ppm"].abs()

    valid_changes = df[
        df["interval_minutes"].gt(0)
        & df["interval_minutes"].le(20)
        & df["co2_change_ppm"].notna()
    ].copy()

    largest_changes = valid_changes.nlargest(
        100, "absolute_co2_change_ppm"
    )

    change_columns = [
        time_col, sensor_col, co2_col,
        "interval_minutes", "co2_change_ppm",
        "absolute_co2_change_ppm"
    ]

    if segment_col in largest_changes.columns:
        change_columns.insert(2, segment_col)

    change_columns += [
        col for col in anomaly_columns
        if col in largest_changes.columns
    ]

    largest_changes[change_columns].to_csv(
        summary_dir / "top_100_co2_changes.csv",
        index=False
    )

    # --------------------------------------------------
    # C. LARGEST CO2 READINGS
    # --------------------------------------------------
    peak_columns = [time_col, sensor_col, co2_col]

    if segment_col in df.columns:
        peak_columns.insert(2, segment_col)

    peak_columns += [
        col for col in anomaly_columns if col in df.columns
    ]

    df.nlargest(100, co2_col)[peak_columns].to_csv(
        summary_dir / "top_100_co2_readings.csv",
        index=False
    )

    # --------------------------------------------------
    # D. TIME GAPS OVER 20 MINUTES
    # --------------------------------------------------
    gaps = df[
        df["interval_minutes"].gt(20)
    ].copy()

    gap_columns = [
        time_col, sensor_col, "interval_minutes"
    ]

    if segment_col in gaps.columns:
        gap_columns.insert(2, segment_col)

    gaps[gap_columns].sort_values(
        "interval_minutes", ascending=False
    ).to_csv(
        summary_dir / "time_gaps_over_20_minutes.csv",
        index=False
    )

    # --------------------------------------------------
    # E. ENVIRONMENTAL ANOMALY FLAG RATES BY SENSOR
    # --------------------------------------------------
    if anomaly_columns:
        anomaly_summary = (
            df.groupby(sensor_col)[anomaly_columns]
            .agg(["sum", "mean", "count"])
        )

        anomaly_summary.columns = [
            f"{col}_{stat}" for col, stat in anomaly_summary.columns
        ]

        anomaly_summary = anomaly_summary.reset_index()

        anomaly_summary.to_csv(
            summary_dir / "anomaly_rates_by_sensor.csv",
            index=False
        )

    print("\nNUMERICAL SUMMARIES EXPORTED")
    print(f"Output directory: {summary_dir}")
    print(" - sensor_co2_summary.csv")
    print(" - top_100_co2_changes.csv")
    print(" - top_100_co2_readings.csv")
    print(" - time_gaps_over_20_minutes.csv")

    if anomaly_columns:
        print(" - anomaly_rates_by_sensor.csv")

# ============================================================
# 7. RUN VISUALISATION PIPELINE
# ============================================================
def main():
    data = load_data()

    print("\n" + "=" * 60)
    print("MMA3001 FILE 06: VISUALISATION")
    print("=" * 60)
    print(f"Rows loaded: {len(data):,}")
    print(f"Sensors: {data['sensorid'].nunique()}")
    print(f"Output directory: {OUTPUT_DIR}")

    plot_co2_trends(data)
    plot_anomaly_flags(data)
    plot_model_disagreements(data)
    plot_environmental_relationships(data)

    # NEW: Export numerical summaries.
    export_numerical_summaries(data)

    print("\nVISUALISATION COMPLETE")
    print(f"Plots and summaries saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()