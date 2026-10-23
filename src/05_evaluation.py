
"""
File 05: Evaluation
MMA3001 Room Occupancy Project

Evaluates:
- Data coverage and missingness
- Sensitivity to different CO2 thresholds
- Stability of estimated occupancy classifications
- Environmental conditions across estimated states

These checks do not establish true occupancy prediction accuracy.

Run from the project root:
    python src/05_evaluation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "outputs" / "occupancy" / "occupancy_estimates.csv"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print("MMA3001 - OCCUPANCY MODEL EVALUATION")
print("=" * 65)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {INPUT_FILE}. Run File 04 first."
    )

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    utc=True
)

if "sensorid" not in df.columns:
    raise ValueError("The dataset does not contain sensorid.")

co2_candidates = [
    c for c in df.columns
    if "carbon dioxide" in c.lower()
    or "co2" in c.lower()
    or "co₂" in c.lower()
]

if not co2_candidates:
    raise ValueError("Could not identify a CO2 measurement column.")

co2_column = co2_candidates[0]
df[co2_column] = pd.to_numeric(df[co2_column], errors="coerce")

df = df.dropna(subset=["timestamp"]).copy()

print(f"\nRows evaluated: {len(df)}")
print(f"Sensors evaluated: {df['sensorid'].nunique()}")
print(f"CO2 column: {co2_column}")


# ============================================================
# 3. DATA COVERAGE
# ============================================================

print("\n--- DATA COVERAGE ---")

coverage = pd.DataFrame({
    "non_missing": df.notna().sum(),
    "missing": df.isna().sum(),
    "missing_percent": (
        df.isna().mean() * 100
    ).round(2),
})

coverage.to_csv(OUTPUT_DIR / "data_coverage.csv")

print(coverage.to_string())

# Timestamp coverage for each sensor.
sensor_coverage = (
    df.groupby("sensorid")
    .agg(
        readings=("timestamp", "count"),
        first_reading=("timestamp", "min"),
        last_reading=("timestamp", "max"),
        valid_co2_readings=(co2_column, "count"),
    )
    .reset_index()
)

sensor_coverage["span_hours"] = (
    sensor_coverage["last_reading"]
    - sensor_coverage["first_reading"]
).dt.total_seconds() / 3600

sensor_coverage.to_csv(
    OUTPUT_DIR / "sensor_coverage.csv",
    index=False
)

print("\nSensor coverage:")
print(sensor_coverage.to_string(index=False))


# ============================================================
# 4. THRESHOLD SENSITIVITY ANALYSIS
# ============================================================

print("\n--- THRESHOLD SENSITIVITY ---")

# Each pair defines a different provisional threshold scenario.
# These are test settings, not validated occupancy boundaries.
scenarios = {
    "lower": (500, 700, 900),
    "baseline": (600, 800, 1000),
    "higher": (700, 900, 1100),
}


def classify_co2(value, thresholds):
    """Assign an ordinal CO2 state using three thresholds."""

    low, moderate, high = thresholds

    if pd.isna(value):
        return "Unknown"
    if value < low:
        return "Low CO2"
    if value < moderate:
        return "Possible occupancy"
    if value < high:
        return "Elevated CO2"
    return "High CO2"


scenario_columns = []

for scenario_name, thresholds in scenarios.items():
    column = f"state_{scenario_name}"

    df[column] = df[co2_column].apply(
        lambda value: classify_co2(value, thresholds)
    )

    scenario_columns.append(column)

    print(f"{scenario_name}: thresholds = {thresholds}")

    print(
        df[column]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_string()
    )

# Calculate how often each scenario agrees with the baseline.
baseline_column = "state_baseline"

agreement_results = []

for scenario_name in scenarios:
    column = f"state_{scenario_name}"

    valid = (
        df[column].ne("Unknown")
        & df[baseline_column].ne("Unknown")
    )

    if valid.any():
        agreement = (
            df.loc[valid, column]
            == df.loc[valid, baseline_column]
        ).mean() * 100
    else:
        agreement = float("nan")

    agreement_results.append({
        "scenario": scenario_name,
        "thresholds": str(scenarios[scenario_name]),
        "agreement_with_baseline_percent": round(agreement, 2),
    })

agreement_df = pd.DataFrame(agreement_results)

agreement_df.to_csv(
    OUTPUT_DIR / "threshold_sensitivity.csv",
    index=False
)

print("\nAgreement with baseline:")
print(agreement_df.to_string(index=False))

# Plot the proportion assigned to each state for each scenario.
state_order = [
    "Low CO2",
    "Possible occupancy",
    "Elevated CO2",
    "High CO2",
    "Unknown",
]

proportions = pd.DataFrame({
    name: df[f"state_{name}"].value_counts(normalize=True)
    for name in scenarios
}).T

proportions = proportions.reindex(
    columns=state_order,
    fill_value=0
) * 100

ax = proportions.plot(
    kind="bar",
    stacked=True,
    figsize=(11, 6)
)

ax.set_title("Estimated CO2 State Distribution by Threshold Scenario")
ax.set_xlabel("Threshold scenario")
ax.set_ylabel("Percentage of readings")
ax.legend(title="CO2 state", bbox_to_anchor=(1.02, 1))
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "threshold_sensitivity.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
plt.close()


# ============================================================
# 5. ENVIRONMENTAL PATTERNS
# ============================================================

print("\n--- ENVIRONMENTAL PATTERNS ---")

# Summarise actual sensor measurements by the baseline CO2 state.
# This can reveal whether other environmental variables change
# alongside elevated CO2. It cannot prove occupancy.

context_columns = [
    c for c in df.columns
    if any(
        term in c.lower()
        for term in [
            "temperature",
            "humidity",
            "volatile organic",
            "formaldehyde",
            "pressure",
            "particulate matter",
            "light",
        ]
    )
    and not any(
        term in c.lower()
        for term in [
            "change",
            "rolling",
            "previous",
            "per_minute",
            "difference",
        ]
    )
]

# Keep only numeric measurements with at least one valid value.
context_columns = [
    c for c in context_columns
    if pd.api.types.is_numeric_dtype(df[c])
    and df[c].notna().any()
]

if context_columns:
    environment_summary = (
        df.groupby(baseline_column)[context_columns]
        .agg(["count", "mean", "median", "std"])
        .round(3)
    )

    environment_summary.to_csv(
        OUTPUT_DIR / "environmental_patterns.csv"
    )

    print(environment_summary.to_string())
else:
    print("No additional numeric environmental measurements found.")


# ============================================================
# 6. HIGH CO2 EVENTS
# ============================================================

print("\n--- HIGH CO2 EVENTS ---")

HIGH_CO2 = scenarios["baseline"][2]

high_co2 = df.loc[
    df[co2_column] >= HIGH_CO2
].copy()

high_co2.to_csv(
    OUTPUT_DIR / "high_co2_readings.csv",
    index=False
)

print(f"Readings at or above {HIGH_CO2} ppm: {len(high_co2)}")

if not high_co2.empty:
    print("\nHighest CO2 readings:")
    print(
        high_co2[
            ["sensorid", "timestamp", co2_column]
        ]
        .sort_values(co2_column, ascending=False)
        .head(10)
        .to_string(index=False)
    )

# Examine the known date of interest if it exists in the data.
peak_date = pd.Timestamp("2026-04-03", tz="UTC")

peak_day = df.loc[
    df["timestamp"].dt.date == peak_date.date()
].copy()

if not peak_day.empty:
    print("\nReadings recorded on 3 April 2026 (UTC):")
    print(f"Number of readings: {len(peak_day)}")
    print(f"Maximum CO2: {peak_day[co2_column].max():.2f} ppm")

    peak_day.to_csv(
        OUTPUT_DIR / "2026-04-03_readings.csv",
        index=False
    )
else:
    print(
        "\nNo readings found for 3 April 2026 UTC. "
        "Check whether the original timestamps use local time."
    )


# ============================================================
# 7. FINAL REPORT
# ============================================================

report_path = OUTPUT_DIR / "evaluation_summary.txt"

valid_co2 = df[co2_column].dropna()

with report_path.open("w", encoding="utf-8") as report:
    report.write("MMA3001 OCCUPANCY ESTIMATION EVALUATION\n")
    report.write("=" * 45 + "\n\n")

    report.write(f"Rows evaluated: {len(df)}\n")
    report.write(f"Number of sensors: {df['sensorid'].nunique()}\n")
    report.write(f"CO2 measurement column: {co2_column}\n")
    report.write(f"Valid CO2 readings: {len(valid_co2)}\n")

    if not valid_co2.empty:
        report.write(f"Mean CO2: {valid_co2.mean():.2f} ppm\n")
        report.write(f"Median CO2: {valid_co2.median():.2f} ppm\n")
        report.write(f"Maximum CO2: {valid_co2.max():.2f} ppm\n")

    report.write("\nThreshold scenarios:\n")
    report.write(agreement_df.to_string(index=False))
    report.write("\n\nLimitations:\n")
    report.write(
        "- CO2-based states are heuristic estimates.\n"
        "- No true occupancy accuracy can be calculated "
        "without independent occupancy labels.\n"
        "- CO2 is affected by ventilation, room conditions "
        "and sensor placement.\n"
        "- Thresholds require justification and sensitivity analysis.\n"
    )

print("\n" + "=" * 65)
print("EVALUATION COMPLETE")
print("=" * 65)

print(f"\nResults saved to:\n{OUTPUT_DIR}")
print(f"Summary report:\n{report_path}")
print(
    "\nThis evaluates data and threshold sensitivity, "
    "not verified occupancy prediction accuracy."
)
