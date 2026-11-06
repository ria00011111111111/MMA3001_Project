
"""Audit actual elapsed times between occupancy observations used as forecast pairs.

The script reads occupancy sensor records from a ZIP archive in chunks,
retains the latest observation for each floor space and five-minute interval,
and pairs each interval with the next consecutive interval.

The original sensor timestamps are retained so that the actual elapsed time
between paired observations can be calculated and compared with the nominal
five-minute forecast horizon.

The script generates:
    - A detailed CSV containing the matched current and future observations.
    - A summary CSV containing elapsed-time statistics and timing tolerance
      percentages.

Outputs:
    outputs/occupancy_forecast_v2/actual_forecast_timing_pairs.csv
    outputs/occupancy_forecast_v2/actual_forecast_timing_summary.csv

Paths are resolved relative to this file rather than the current working
directory.

Important interpretation notes:
    - This script reproduces a last-observation-per-five-minute-bin rule.
    - Results match the original forecasting pipeline only if it uses the
      same source data and aggregation rule.
    - The nominal forecast horizon is five minutes, but actual elapsed time
      between original sensor observations may differ because observations
      are irregular.
"""

from pathlib import Path
from zipfile import ZipFile
import sys
sys.dont_write_bytecode = True
import pandas as pd


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "5occupancySensor_MayToDec2024_9MRows.zip"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_forecast_v2"

CHUNK_SIZE = 250_000
FORECAST_MINUTES = 5
TIMING_TOLERANCES_SECONDS = (30, 60, 120)
REQUIRED_COLUMNS = ["floorspaceid", "occupancystatus", "collecteddate"]


# ---------------------------------------------------------------------------
# 2. OCCUPANCY DATA LOADING
# ---------------------------------------------------------------------------

def load_occupancy_data(zip_path: Path) -> pd.DataFrame:
    """Load valid occupancy records from the first CSV in a ZIP archive.

    Reads the source CSV in chunks to limit memory consumption when processing
    the large occupancy dataset. Each chunk is converted to UTC timestamps,
    cleaned of missing required values, and assigned to a five-minute interval
    using the original observation timestamp.

    Args:
        zip_path (Path): Path to the ZIP archive containing the occupancy
            sensor dataset.

    Returns:
        pd.DataFrame: Valid occupancy records containing the original sensor
        timestamps and an additional ``interval`` column representing the
        corresponding five-minute time bin.

    Raises:
        FileNotFoundError: If the specified ZIP archive does not exist.
        ValueError: If the archive contains no CSV files, the source CSV
            contains no data rows, or no valid occupancy records remain
            after cleaning.

    Note:
        Only the first CSV file found in the archive is processed. The
        function reports cumulative raw rows read during chunk processing.
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


# ---------------------------------------------------------------------------
# 3. FIVE-MINUTE INTERVAL AGGREGATION
# ---------------------------------------------------------------------------

def aggregate_five_minute_intervals(data: pd.DataFrame) -> pd.DataFrame:
    """Retain the latest observation per floor space and five-minute interval.

    Sorts the occupancy records by floor-space identifier, interval, and
    original observation timestamp. When multiple observations belong to
    the same floor space and five-minute interval, only the observation
    with the latest original timestamp is retained.

    Args:
        data (pd.DataFrame): Valid occupancy records containing
            ``floorspaceid``, ``interval``, and ``collecteddate`` columns.

    Returns:
        pd.DataFrame: Aggregated occupancy records containing at most one
        observation for each floor-space and five-minute interval
        combination. The returned DataFrame is sorted by floor space and
        interval, with a reset integer index.

    Note:
        The aggregation retains the original ``collecteddate`` timestamp.
        This allows subsequent calculations to measure actual elapsed time
        between sensor observations rather than simply comparing the
        five-minute interval labels.
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


# ---------------------------------------------------------------------------
# 4. FORECAST OBSERVATION PAIRING
# ---------------------------------------------------------------------------

def build_forecast_pairs(data: pd.DataFrame) -> pd.DataFrame:
    """Match observations in consecutive five-minute bins for each floor space.

    Creates two versions of the aggregated occupancy data:
        - Current observations, with their interval advanced by the nominal
          forecast horizon.
        - Future observations, retaining their original interval and
          observation timestamp.

    An inner join matches current observations with future observations
    belonging to the next consecutive five-minute interval for the same
    floor space.

    The actual elapsed time is calculated using the original sensor
    timestamps rather than the nominal interval labels.

    Args:
        data (pd.DataFrame): Aggregated occupancy records containing
            ``floorspaceid``, ``interval``, ``collecteddate``, and
            ``occupancystatus`` columns.

    Returns:
        pd.DataFrame: Matched current and future observation pairs containing:
            - floorspaceid: Floor-space identifier.
            - interval: Current five-minute interval.
            - current_observation_time: Original current observation timestamp.
            - current_state: Current recorded occupancy state.
            - future_interval: Target five-minute interval.
            - future_observation_time: Original future observation timestamp.
            - actual_future_state: Future recorded occupancy state.
            - actual_elapsed_minutes: Actual elapsed time between observations.
            - elapsed_error_minutes: Absolute difference between actual
              elapsed time and the nominal forecast horizon.

    Note:
        Only observations in consecutive five-minute bins are paired.
        Missing intermediate bins are not bridged. Actual elapsed time may
        therefore differ from five minutes even when a pair is successfully
        matched.
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


# ---------------------------------------------------------------------------
# 5. ACTUAL TIMING ANALYSIS
# ---------------------------------------------------------------------------

def summarise_timing(pairs: pd.DataFrame) -> pd.DataFrame:
    """Calculate and report statistics for actual elapsed observation times.

    Calculates descriptive statistics for the elapsed time between matched
    current and future sensor observations. Also determines the percentage
    of pairs whose elapsed time falls within each configured tolerance of
    the nominal five-minute forecast horizon.

    Timing tolerances are defined by TIMING_TOLERANCES_SECONDS.

    Args:
        pairs (pd.DataFrame): Matched observation pairs generated by
            ``build_forecast_pairs()``. Must contain
            ``actual_elapsed_minutes`` and ``elapsed_error_minutes``.

    Returns:
        pd.DataFrame: One-row summary containing:
            - matched_pairs: Number of matched observation pairs.
            - mean_elapsed_minutes: Mean actual elapsed time.
            - median_elapsed_minutes: Median actual elapsed time.
            - minimum_elapsed_minutes: Minimum actual elapsed time.
            - maximum_elapsed_minutes: Maximum actual elapsed time.
            - within_30_seconds_percent: Percentage within 30 seconds of
              the nominal forecast horizon.
            - within_60_seconds_percent: Percentage within 60 seconds.
            - within_120_seconds_percent: Percentage within 120 seconds.

    Raises:
        ValueError: If no consecutive-bin observation pairs are available.

    Note:
        Timing tolerance percentages measure how closely actual observation
        intervals match the nominal five-minute horizon. They do not measure
        forecasting accuracy or classification performance.
    """
    if pairs.empty:
        raise ValueError(
            "No matching observations were found in consecutive five-minute bins."
        )

    elapsed = pairs["actual_elapsed_minutes"]
    print(f"Matched forecast pairs: {len(pairs):,}")

    print("\nACTUAL ELAPSED TIME BETWEEN READINGS (MINUTES)")
    print(elapsed.describe().to_string())

    percentages = {}
    print("\nPAIRS WITHIN DIFFERENT TIMING TOLERANCES")
    for seconds in TIMING_TOLERANCES_SECONDS:
        within_tolerance = pairs["elapsed_error_minutes"] * 60 <= seconds
        count = int(within_tolerance.sum())
        percentage = 100 * count / len(pairs)
        percentages[seconds] = percentage
        print(
            f"Within {seconds:>3} seconds of 5 minutes: "
            f"{count:,} ({percentage:.2f}%)"
        )

    print("\nELAPSED-TIME QUANTILES (MINUTES)")
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


# ---------------------------------------------------------------------------
# 6. MAIN WORKFLOW
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the actual occupancy forecast timing audit.

    Loads and cleans the source occupancy dataset, aggregates observations
    into five-minute intervals, and pairs observations from consecutive
    intervals for each floor space.

    Calculates actual elapsed observation times, reports descriptive
    statistics and timing tolerance percentages, and saves the detailed
    matched pairs and summary results as CSV files.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the source occupancy ZIP archive cannot be found.
        ValueError: If the source archive contains no usable CSV data or
            no consecutive five-minute observation pairs are available.

    Outputs:
        Saves the following files to OUTPUT_DIR:
            - actual_forecast_timing_pairs.csv
            - actual_forecast_timing_summary.csv

    Note:
        The output directory is created if it does not already exist.
        The audit reproduces the five-minute aggregation rule and may not
        exactly reproduce the forecasting pipeline if its preprocessing
        differs.
    """
    print("=" * 65)
    print("ACTUAL OCCUPANCY FORECAST TIMING AUDIT")
    print("=" * 65)

    data = load_occupancy_data(ZIP_FILE)
    data = aggregate_five_minute_intervals(data)
    print(f"\nAggregated intervals: {len(data):,}")

    pairs = build_forecast_pairs(data)
    summary = summarise_timing(pairs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUTPUT_DIR / "actual_forecast_timing_pairs.csv", index=False)
    summary.to_csv(
        OUTPUT_DIR / "actual_forecast_timing_summary.csv", index=False
    )

    print("\nSaved timing audit to:")
    print(OUTPUT_DIR)
    print("\nTiming audit complete.")


if __name__ == "__main__":
    main()
