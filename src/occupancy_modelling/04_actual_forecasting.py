from pathlib import Path

"""Audit actual elapsed times between occupancy observations used as forecast pairs.

Reads occupancy records from a ZIP archive in chunks, retains the last
observation per floor space and five-minute interval, and pairs each interval
with the next interval.

Outputs:
    outputs/occupancy_forecast_v2/actual_forecast_timing_pairs.csv
    outputs/occupancy_forecast_v2/actual_forecast_timing_summary.csv

Note:
    This reproduces a last-observation-per-five-minute-bin rule. Results match
    the original forecasting pipeline only if it uses the same source data and
    aggregation rule.
"""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "5occupancySensor_MayToDec2024_9MRows (1).zip"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_forecast_v2"

CHUNK_SIZE = 250_000
FORECAST_MINUTES = 5
TIMING_TOLERANCES_SECONDS = (30, 60, 120)
REQUIRED_COLUMNS = ["floorspaceid", "occupancystatus", "collecteddate"]


def load_occupancy_data(zip_path: Path) -> pd.DataFrame:
    """Load valid occupancy records from the first CSV in a ZIP archive.

    Reads in chunks to reduce memory use. Rows with missing required values
    or invalid timestamps are excluded.

    Args:
        zip_path: Path to the source ZIP archive.

    Returns:
        Valid records with timestamps parsed as UTC and a five-minute interval.

    Raises:
        FileNotFoundError: If the ZIP archive is missing.
        ValueError: If the archive contains no CSV or no valid records.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"Occupancy ZIP file not found: {zip_path}")

    chunks = []
    total_rows = 0

    with ZipFile(zip_path) as archive:
        csv_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]
        if not csv_files:
            raise ValueError(f"No CSV file found in ZIP archive: {zip_path}")

        with archive.open(csv_files[0]) as csv_file:
            for chunk in pd.read_csv(
                csv_file,
                usecols=REQUIRED_COLUMNS,
                chunksize=CHUNK_SIZE,
                low_memory=False,
            ):
                total_rows += len(chunk)
                chunk["collecteddate"] = pd.to_datetime(
                    chunk["collecteddate"], utc=True, errors="coerce"
                )
                chunk = chunk.dropna(subset=REQUIRED_COLUMNS)
                chunk["interval"] = chunk["collecteddate"].dt.floor("5min")
                chunks.append(chunk)
                print(f"Read {total_rows:,} raw rows...")

    if not chunks:
        raise ValueError("The source CSV contained no data rows.")

    data = pd.concat(chunks, ignore_index=True)
    del chunks

    if data.empty:
        raise ValueError("No valid occupancy rows remained after cleaning.")

    return data


def aggregate_five_minute_intervals(data: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest observation per floor space and five-minute interval.

    Args:
        data: Valid sensor records containing an ``interval`` column.

    Returns:
        A DataFrame with at most one row per floor space and interval.
    """
    data = data.sort_values(
        ["floorspaceid", "interval", "collecteddate"]
    )
    data = data.drop_duplicates(
        subset=["floorspaceid", "interval"], keep="last"
    )
    return data.sort_values(
        ["floorspaceid", "interval"]
    ).reset_index(drop=True)


def build_forecast_pairs(data: pd.DataFrame) -> pd.DataFrame:
    """Pair observations in consecutive five-minute bins for each floor space.

    The elapsed time is measured using the original sensor timestamps, which
    may differ from five minutes because observations are irregular.

    Args:
        data: Aggregated five-minute occupancy records.

    Returns:
        Matched pairs with elapsed time and absolute error from five minutes.
    """
    current = data[
        ["floorspaceid", "interval", "collecteddate", "occupancystatus"]
    ].rename(
        columns={
            "collecteddate": "current_observation_time",
            "occupancystatus": "current_state",
        }
    )
    current["future_interval"] = (
        current["interval"] + pd.Timedelta(minutes=FORECAST_MINUTES)
    )

    future = data[
        ["floorspaceid", "interval", "collecteddate", "occupancystatus"]
    ].rename(
        columns={
            "interval": "future_interval",
            "collecteddate": "future_observation_time",
            "occupancystatus": "actual_future_state",
        }
    )

    pairs = current.merge(
        future,
        on=["floorspaceid", "future_interval"],
        how="inner",
        validate="one_to_one",
    )
    pairs["actual_elapsed_minutes"] = (
        pairs["future_observation_time"] - pairs["current_observation_time"]
    ).dt.total_seconds() / 60
    pairs["elapsed_error_minutes"] = (
        pairs["actual_elapsed_minutes"] - FORECAST_MINUTES
    ).abs()

    return pairs


def summarise_timing(pairs: pd.DataFrame) -> pd.DataFrame:
    """Print timing statistics and return a one-row summary DataFrame.

    Args:
        pairs: Matched pairs created by ``build_forecast_pairs``.

    Returns:
        Summary statistics for export to CSV.

    Raises:
        ValueError: If no consecutive-bin pairs were found.
    """
    if pairs.empty:
        raise ValueError(
            "No matching observations were found in consecutive five-minute bins."
        )

    elapsed = pairs["actual_elapsed_minutes"]
    print(f"Matched forecast pairs: {len(pairs):,}")

    print("\\nACTUAL ELAPSED TIME BETWEEN READINGS (MINUTES)")
    print(elapsed.describe().to_string())

    percentages = {}
    print("\\nPAIRS WITHIN DIFFERENT TIMING TOLERANCES")
    for seconds in TIMING_TOLERANCES_SECONDS:
        within_tolerance = pairs["elapsed_error_minutes"] * 60 <= seconds
        count = int(within_tolerance.sum())
        percentage = 100 * count / len(pairs)
        percentages[seconds] = percentage
        print(
            f"Within {seconds:>3} seconds of 5 minutes: "
            f"{count:,} ({percentage:.2f}%)"
        )

    print("\\nELAPSED-TIME QUANTILES (MINUTES)")
    print(elapsed.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_string())

    return pd.DataFrame([{
        "matched_pairs": len(pairs),
        "mean_elapsed_minutes": elapsed.mean(),
        "median_elapsed_minutes": elapsed.median(),
        "minimum_elapsed_minutes": elapsed.min(),
        "maximum_elapsed_minutes": elapsed.max(),
        "within_30_seconds_percent": percentages[30],
        "within_60_seconds_percent": percentages[60],
        "within_120_seconds_percent": percentages[120],
    }])


def main() -> None:
    """Run the timing audit and save detailed and summary CSV outputs."""
    print("=" * 65)
    print("ACTUAL OCCUPANCY FORECAST TIMING AUDIT")
    print("=" * 65)

    data = load_occupancy_data(ZIP_FILE)
    data = aggregate_five_minute_intervals(data)
    print(f"\\nAggregated intervals: {len(data):,}")

    pairs = build_forecast_pairs(data)
    summary = summarise_timing(pairs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUTPUT_DIR / "actual_forecast_timing_pairs.csv", index=False)
    summary.to_csv(
        OUTPUT_DIR / "actual_forecast_timing_summary.csv", index=False
    )

    print("\\nSaved timing audit to:")
    print(OUTPUT_DIR)
    print("\\nTiming audit complete.")


if __name__ == "__main__":
    main()