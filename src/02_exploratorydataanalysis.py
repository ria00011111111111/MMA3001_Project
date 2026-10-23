
# ============================================================
# FILE 02: EXPLORATORY DATA ANALYSIS
# MMA3001 - Room Occupancy Estimation
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. FILE PATHS AND OUTPUT DIRECTORY
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "cleaned_sensor_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned data file not found: {DATA_FILE}\n"
        "Run File 01 (data cleaning) first."
    )

# ============================================================
# 2. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\n" + "=" * 70)
print("MMA3001: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nDATASET OVERVIEW")
print("-" * 70)

print(f"Number of rows: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")
print(
    f"Memory usage: "
    f"{df.memory_usage(deep=True).sum() / 1e6:.2f} MB"
)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes.to_string())

# ============================================================
# 3. PROCESS AND INSPECT TIMESTAMPS
# ============================================================

if "timestamp" not in df.columns:
    raise ValueError(
        "Missing 'timestamp' column. Check the output from File 01."
    )

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    utc=True
)

invalid_timestamps = df["timestamp"].isna().sum()

print("\nTIMESTAMP INFORMATION")
print("-" * 70)
print(f"Invalid or missing timestamps: {invalid_timestamps}")

df = df.dropna(subset=["timestamp"]).copy()

if not df.empty:
    print(f"First timestamp (UTC): {df['timestamp'].min()}")
    print(f"Last timestamp (UTC):  {df['timestamp'].max()}")
    print(f"Unique timestamps: {df['timestamp'].nunique():,}")

df = df.sort_values("timestamp").reset_index(drop=True)

if "sensorid" in df.columns:
    print(f"Unique sensors: {df['sensorid'].nunique():,}")

    print("\nNumber of rows per sensor:")
    print(
        df.groupby("sensorid")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

# ============================================================
# 4. IDENTIFY NUMERIC MEASUREMENT COLUMNS
# ============================================================

excluded_columns = {
    "id",
    "sensorid",
    "timestamp",
    "createdate",
    "processdate",
    "status",
    "processing_errors",
}

for column in df.columns:
    if column not in excluded_columns:
        converted = pd.to_numeric(df[column], errors="coerce")

        if converted.notna().any() or df[column].isna().all():
            df[column] = converted

measurement_columns = [
    column
    for column in df.select_dtypes(include=np.number).columns
    if column not in excluded_columns
]

print("\nMEASUREMENT COLUMNS")
print("-" * 70)

if measurement_columns:
    for column in measurement_columns:
        print(f"- {column}")
else:
    print("No numeric measurement columns were identified.")

# ============================================================
# 5. MISSING VALUE ANALYSIS
# ============================================================

print("\nMISSING VALUE ANALYSIS")
print("-" * 70)

missing_summary = pd.DataFrame({
    "Missing Count": df.isna().sum(),
    "Missing Percentage": (df.isna().mean() * 100).round(2),
})

missing_summary = missing_summary.sort_values(
    "Missing Percentage",
    ascending=False
)

print(missing_summary.to_string())

missing_summary.to_csv(
    OUTPUT_DIR / "missing_value_summary.csv"
)

missing_to_plot = missing_summary[
    missing_summary["Missing Count"] > 0
]

if not missing_to_plot.empty:
    plt.figure(figsize=(12, 7))

    missing_to_plot["Missing Percentage"].sort_values().plot(
        kind="barh"
    )

    plt.xlabel("Missing values (%)")
    plt.ylabel("Column")
    plt.title("Missing Values by Column")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "missing_values.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

else:
    print("No missing values found.")

# ============================================================
# 6. DUPLICATE ANALYSIS
# ============================================================

print("\nDUPLICATE ANALYSIS")
print("-" * 70)

exact_duplicates = df.duplicated().sum()

print(f"Exact duplicate rows: {exact_duplicates}")

if "sensorid" in df.columns:
    duplicate_mask = df.duplicated(
        subset=["sensorid", "timestamp"],
        keep=False
    )

    print(
        "Rows sharing a sensor ID and timestamp: "
        f"{duplicate_mask.sum()}"
    )

    if duplicate_mask.any():
        print("\nExamples of repeated sensor/timestamp combinations:")

        print(
            df.loc[
                duplicate_mask,
                ["sensorid", "timestamp"]
            ]
            .sort_values(["sensorid", "timestamp"])
            .head(20)
            .to_string(index=False)
        )

        print(
            "\nRepeated timestamps are flagged for investigation. "
            "They are not automatically deleted."
        )

# ============================================================
# 7. SAMPLING INTERVALS AND DATA GAPS
# ============================================================

print("\nSAMPLING INTERVAL ANALYSIS")
print("-" * 70)

if "sensorid" in df.columns:
    sampling_data = df.dropna(
        subset=["sensorid", "timestamp"]
    ).copy()

    sampling_data = sampling_data.sort_values(
        ["sensorid", "timestamp"]
    )

    sampling_data["interval_minutes"] = (
        sampling_data.groupby("sensorid")["timestamp"]
        .diff()
        .dt.total_seconds() / 60
    )

    interval_summary = (
        sampling_data.groupby("sensorid")["interval_minutes"]
        .agg(["count", "median", "mean", "min", "max"])
        .round(2)
    )

    print(interval_summary.to_string())

    interval_summary.to_csv(
        OUTPUT_DIR / "sampling_interval_summary.csv"
    )

    long_gaps = sampling_data[
        sampling_data["interval_minutes"] > 60
    ]

    print(
        "\nIntervals longer than 60 minutes: "
        f"{len(long_gaps)}"
    )

    if not long_gaps.empty:
        print("\nExamples of long gaps:")
        print(
            long_gaps[
                ["sensorid", "timestamp", "interval_minutes"]
            ]
            .head(20)
            .to_string(index=False)
        )

else:
    intervals = (
        df["timestamp"]
        .sort_values()
        .diff()
        .dt.total_seconds() / 60
    )

    print(intervals.describe().round(2).to_string())

# ============================================================
# 8. DESCRIPTIVE STATISTICS
# ============================================================

print("\nDESCRIPTIVE STATISTICS")
print("-" * 70)

if measurement_columns:
    descriptive_stats = df[measurement_columns].describe().T

    descriptive_stats["missing_count"] = (
        df[measurement_columns].isna().sum()
    )

    print(descriptive_stats.round(3).to_string())

    descriptive_stats.to_csv(
        OUTPUT_DIR / "descriptive_statistics.csv"
    )

# ============================================================
# 9. CONSTANT AND LOW-VARIABILITY MEASUREMENTS
# ============================================================

print("\nLOW-VARIABILITY MEASUREMENTS")
print("-" * 70)

for column in measurement_columns:
    unique_count = df[column].nunique(dropna=True)

    if unique_count <= 1:
        print(
            f"{column}: {unique_count} unique non-missing value(s)"
        )
    elif unique_count <= 5:
        print(
            f"{column}: only {unique_count} unique non-missing values"
        )

# ============================================================
# 10. DISTRIBUTION PLOTS
# ============================================================

print("\nGENERATING DISTRIBUTION PLOTS")
print("-" * 70)

for column in measurement_columns:
    values = df[column].dropna()

    if values.empty or values.nunique() <= 1:
        print(f"Skipping {column}: insufficient variation.")
        continue

    plt.figure(figsize=(10, 5))

    plt.hist(
        values,
        bins=40,
        edgecolor="black",
        alpha=0.75
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column}")
    plt.tight_layout()

    safe_name = "".join(
        character if character.isalnum() or character in "-_"
        else "_"
        for character in column
    )

    plt.savefig(
        OUTPUT_DIR / f"distribution_{safe_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# ============================================================
# 11. TIME-SERIES PLOTS
# ============================================================

print("\nGENERATING TIME-SERIES PLOTS")
print("-" * 70)

preferred_variables = [
    "Carbon dioxide",
    "Temperature",
    "Humidity",
    "Light Volatile Organic Compounds",
]

for preferred_name in preferred_variables:
    matches = [
        column for column in measurement_columns
        if preferred_name.lower() in column.lower()
    ]

    if not matches:
        print(f"Skipping {preferred_name}: column not found.")
        continue

    column = matches[0]

    plot_data = (
        df[["timestamp", column]]
        .dropna()
        .sort_values("timestamp")
    )

    if plot_data.empty:
        continue

    plt.figure(figsize=(13, 5))

    plt.plot(
        plot_data["timestamp"],
        plot_data[column],
        linewidth=0.8
    )

    plt.xlabel("Timestamp (UTC)")
    plt.ylabel(column)
    plt.title(f"{column} Over Time")
    plt.xticks(rotation=30)
    plt.tight_layout()

    safe_name = "".join(
        character if character.isalnum() or character in "-_"
        else "_"
        for character in column
    )

    plt.savefig(
        OUTPUT_DIR / f"time_series_{safe_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# ============================================================
# 12. CO2 ANALYSIS AND PEAK ON 3 APRIL 2026
# ============================================================

print("\nCO2 ANALYSIS")
print("-" * 70)

co2_columns = [
    column for column in measurement_columns
    if "carbon dioxide" in column.lower()
    or column.lower() in {"co2", "co₂"}
]

if not co2_columns:
    print(
        "No CO2 column found. Check the measurement column names."
    )

else:
    co2_column = co2_columns[0]

    co2_data = (
        df[["timestamp", co2_column]]
        .dropna()
        .sort_values("timestamp")
        .copy()
    )

    if not co2_data.empty:
        overall_peak_index = co2_data[co2_column].idxmax()
        overall_peak = df.loc[overall_peak_index]

        print(f"CO2 column: {co2_column}")
        print(f"Valid readings: {len(co2_data):,}")
        print(f"Mean CO2: {co2_data[co2_column].mean():.2f} ppm")
        print(f"Median CO2: {co2_data[co2_column].median():.2f} ppm")
        print(f"Overall maximum: {overall_peak[co2_column]:.2f} ppm")
        print(f"Overall peak timestamp (UTC): {overall_peak['timestamp']}")

        # Select readings on 3 April 2026 in UTC.
        date_start = pd.Timestamp("2026-04-03", tz="UTC")
        date_end = pd.Timestamp("2026-04-04", tz="UTC")

        peak_date_data = co2_data[
            (co2_data["timestamp"] >= date_start)
            & (co2_data["timestamp"] < date_end)
        ].copy()

        print("\nCO2 READINGS ON 3 APRIL 2026 (UTC)")
        print("-" * 70)

        if peak_date_data.empty:
            print(
                "No readings found for 3 April 2026 UTC. "
                "Check whether the graph uses Melbourne local time."
            )

        else:
            peak_index = peak_date_data[co2_column].idxmax()
            peak_row = peak_date_data.loc[peak_index]

            peak_time = peak_row["timestamp"]
            peak_value = peak_row[co2_column]

            print(f"Number of readings: {len(peak_date_data):,}")
            print(f"Mean CO2: {peak_date_data[co2_column].mean():.2f} ppm")
            print(f"Peak CO2: {peak_value:.2f} ppm")
            print(f"Peak timestamp (UTC): {peak_time}")

            # Inspect three hours before and after the peak.
            window_start = peak_time - pd.Timedelta(hours=3)
            window_end = peak_time + pd.Timedelta(hours=3)

            peak_window = co2_data[
                (co2_data["timestamp"] >= window_start)
                & (co2_data["timestamp"] <= window_end)
            ].copy()

            peak_window = peak_window.sort_values("timestamp")

            print("\nCO2 READINGS WITHIN THREE HOURS OF THE PEAK")
            print("-" * 70)
            print(peak_window.to_string(index=False))

            peak_window.to_csv(
                OUTPUT_DIR / "2026-04-03_co2_peak_window.csv",
                index=False
            )

            # Plot the peak and surrounding readings.
            plt.figure(figsize=(13, 5))

            plt.plot(
                peak_window["timestamp"],
                peak_window[co2_column],
                marker="o",
                markersize=3,
                linewidth=1
            )

            plt.axhline(
                1000,
                linestyle="--",
                label="1000 ppm reference"
            )

            plt.scatter(
                [peak_time],
                [peak_value],
                marker="x",
                s=100,
                label="Peak"
            )

            plt.xlabel("Timestamp (UTC)")
            plt.ylabel("CO2 (ppm)")
            plt.title("CO2 Around the Peak on 3 April 2026")
            plt.legend()
            plt.xticks(rotation=30)
            plt.tight_layout()

            plt.savefig(
                OUTPUT_DIR / "2026-04-03_co2_peak.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.show()
            plt.close()

            # Inspect other sensor measurements near the peak.
            print("\nOTHER SENSOR READINGS NEAR THE CO2 PEAK")
            print("-" * 70)

            nearby_start = peak_time - pd.Timedelta(minutes=30)
            nearby_end = peak_time + pd.Timedelta(minutes=30)

            nearby_data = df[
                (df["timestamp"] >= nearby_start)
                & (df["timestamp"] <= nearby_end)
            ].copy()

            if "sensorid" in nearby_data.columns:
                nearby_data = nearby_data.sort_values(
                    ["timestamp", "sensorid"]
                )

            print(nearby_data.to_string(index=False))

            nearby_data.to_csv(
                OUTPUT_DIR / "measurements_near_co2_peak.csv",
                index=False
            )

# ============================================================
# 13. HOURLY CO2 SUMMARY
# ============================================================

if co2_columns:
    hourly_source = (
        df[["timestamp", co2_columns[0]]]
        .dropna()
        .set_index("timestamp")
        .sort_index()
    )

    if not hourly_source.empty:
        hourly_co2 = hourly_source.resample("1h").agg(
            ["mean", "median", "count"]
        )

        hourly_co2.columns = [
            f"{column}_{statistic}"
            for column, statistic in hourly_co2.columns
        ]

        print("\nHOURLY CO2 SUMMARY")
        print("-" * 70)
        print(hourly_co2.head(24).round(2).to_string())

        hourly_co2.to_csv(
            OUTPUT_DIR / "hourly_co2_summary.csv"
        )

        mean_column = f"{co2_columns[0]}_mean"

        plt.figure(figsize=(13, 5))

        plt.plot(
            hourly_co2.index,
            hourly_co2[mean_column],
            linewidth=0.9
        )

        plt.xlabel("Timestamp (UTC)")
        plt.ylabel("Hourly mean CO2 (ppm)")
        plt.title("Hourly Mean CO2 Over Time")
        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "hourly_mean_co2.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

# ============================================================
# 14. CORRELATION ANALYSIS
# ============================================================

print("\nCORRELATION ANALYSIS")
print("-" * 70)

if len(measurement_columns) >= 2:
    correlation_matrix = df[measurement_columns].corr(
        method="pearson",
        min_periods=3
    )

    print(correlation_matrix.round(3).to_string())

    correlation_matrix.to_csv(
        OUTPUT_DIR / "correlation_matrix.csv"
    )

    plt.figure(figsize=(12, 9))

    image = plt.imshow(
        correlation_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    plt.colorbar(image, label="Pearson correlation")

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation_matrix.index)),
        correlation_matrix.index
    )

    plt.title("Correlation Between Sensor Measurements")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    if co2_columns:
        co2_column = co2_columns[0]

        print("\nCORRELATION WITH CO2")
        print("-" * 70)

        print(
            correlation_matrix[co2_column]
            .drop(co2_column)
            .sort_values(ascending=False)
            .round(3)
            .to_string()
        )

else:
    print("Not enough numeric measurements for correlation analysis.")

# ============================================================
# 15. FINAL FINDINGS SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA SUMMARY")
print("=" * 70)

print(f"Rows analysed: {len(df):,}")
print(f"Measurement columns: {len(measurement_columns)}")
print(f"Exact duplicate rows: {df.duplicated().sum()}")

print("\nColumns with the highest missing percentages:")
print(missing_summary.head(5).to_string())

if co2_columns:
    co2_values = df[co2_columns[0]].dropna()

    if not co2_values.empty:
        print(f"\nCO2 median: {co2_values.median():.2f} ppm")
        print(f"CO2 maximum: {co2_values.max():.2f} ppm")

print("\nIMPORTANT LIMITATIONS")
print("- Correlation does not establish causation.")
print("- CO2 is a potential occupancy indicator, not occupancy ground truth.")
print("- Missing data, sensor errors and irregular sampling may affect results.")
print("- Timestamps are interpreted and analysed in UTC.")
print("- Cleaning thresholds should be validated against sensor specifications.")

print(f"\nEDA outputs saved to: {OUTPUT_DIR}")
print("\nEXPLORATORY DATA ANALYSIS COMPLETE")
