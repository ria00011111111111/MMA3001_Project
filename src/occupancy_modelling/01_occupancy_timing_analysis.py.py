"""
FILE-01: Occupancy timing and transition analysis.

This script processes raw occupancy sensor records from a ZIP archive
and aggregates them into five-minute intervals for each floor space.

It analyses:
    1. Time gaps between consecutive observations.
    2. Transitions between recorded occupancy states.
    3. The probability of each state following another state.
    4. The observed duration of consecutive runs of the same state.

The analysis describes recorded sensor states rather than independently
verified physical occupancy. Observed run durations are therefore
estimates based on the available records.

Outputs are saved to:
    outputs/occupancy_timing_analysis/
"""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

# Locate the project root relative to this script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input dataset.
ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "5occupancySensor_MayToDec2024_9MRows (1).zip"
)

# Directory for all generated analysis outputs.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_timing_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of source records loaded into memory at a time.
CHUNK_SIZE = 250_000

# Length of each aggregation interval, in minutes.
INTERVAL_MINUTES = 5


def load_and_aggregate_data():
    """Load sensor records and retain the last observation per five-minute interval.

    Records are read in chunks to reduce memory usage when processing
    the large source dataset. Timestamps are converted to UTC, and
    records missing essential fields are removed.

    Within each floor space and interval, the observation with the
    latest timestamp is retained. Duplicate intervals occurring across
    different chunks are resolved after combining the chunk results.

    Returns:
        pd.DataFrame: Aggregated observations sorted by floor space
        and interval.
    """
    interval_chunks = []
    total_rows = 0

    print("\nReading occupancy data...")

    # Open the ZIP archive without extracting the full dataset.
    with ZipFile(ZIP_FILE) as archive:
        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise FileNotFoundError("No CSV found in occupancy ZIP.")

        # Read only the columns required for this analysis.
        with archive.open(csv_files[0]) as file:
            for chunk in pd.read_csv(
                file,
                usecols=[
                    "floorspaceid",
                    "occupancystatus",
                    "collecteddate",
                ],
                chunksize=CHUNK_SIZE,
                low_memory=False,
            ):
                total_rows += len(chunk)

                # Parse timestamps and discard invalid records.
                chunk["collecteddate"] = pd.to_datetime(
                    chunk["collecteddate"],
                    utc=True,
                    errors="coerce",
                )

                chunk = chunk.dropna(
                    subset=[
                        "floorspaceid",
                        "occupancystatus",
                        "collecteddate",
                    ]
                ).copy()

                # Assign each observation to a five-minute interval.
                chunk["interval"] = chunk["collecteddate"].dt.floor(
                    f"{INTERVAL_MINUTES}min"
                )

                # Sort chronologically before selecting the latest
                # observation within each floor-space interval.
                chunk = chunk.sort_values(
                    "collecteddate",
                    kind="stable",
                )

                chunk = chunk.drop_duplicates(
                    subset=["floorspaceid", "interval"],
                    keep="last",
                )

                # Keep only the columns needed for later analysis.
                interval_chunks.append(
                    chunk[
                        [
                            "floorspaceid",
                            "interval",
                            "occupancystatus",
                            "collecteddate",
                        ]
                    ]
                )

                print(f"Processed {total_rows:,} records...")

    if not interval_chunks:
        raise ValueError("No valid occupancy records were found.")

    print("\nCombining interval summaries...")

    # Combine chunk-level summaries and resolve duplicate intervals
    # that may have appeared in different chunks.
    data = pd.concat(interval_chunks, ignore_index=True)
    del interval_chunks

    data = data.sort_values(
        "collecteddate",
        kind="stable",
    )

    data = data.drop_duplicates(
        subset=["floorspaceid", "interval"],
        keep="last",
    )

    # Sort by floor space and time for chronological analysis.
    data = data.sort_values(
        ["floorspaceid", "interval"],
        kind="stable",
    ).reset_index(drop=True)

    print(f"Source records: {total_rows:,}")
    print(f"Aggregated observations: {len(data):,}")

    return data


def analyse_observation_gaps(data):
    """Calculate the time between consecutive observations per floor space.

    The gap is calculated using the difference between consecutive
    aggregated interval timestamps. Summary statistics are then
    calculated separately for each floor space.

    Saves:
        observation_gap_summary.csv
    """
    print("\nAnalysing observation time gaps...")

    # Find the preceding interval for each floor space.
    data["previous_interval"] = (
        data.groupby("floorspaceid")["interval"].shift(1)
    )

    # Calculate elapsed time in minutes between consecutive intervals.
    data["gap_minutes"] = (
        data["interval"] - data["previous_interval"]
    ).dt.total_seconds() / 60

    gap_data = data.dropna(subset=["gap_minutes"])

    # Summarise typical and unusually long observation gaps.
    gap_summary = (
        gap_data.groupby("floorspaceid")["gap_minutes"]
        .agg(
            observations="count",
            median_gap_minutes="median",
            mean_gap_minutes="mean",
            p90_gap_minutes=lambda x: x.quantile(0.90),
            p95_gap_minutes=lambda x: x.quantile(0.95),
            maximum_gap_minutes="max",
        )
        .reset_index()
    )

    gap_summary.to_csv(
        OUTPUT_DIR / "observation_gap_summary.csv",
        index=False,
    )

    print("\nOBSERVATION GAP SUMMARY (minutes)")
    print(
        gap_summary.to_string(
            index=False,
            float_format="%.2f",
        )
    )


def analyse_state_transitions(data):
    """Count state transitions and calculate conditional transition probabilities.

    Each transition represents a pair of consecutive recorded states
    for the same floor space. Probabilities are calculated by dividing
    each transition count by the total number of transitions originating
    from the corresponding previous state.

    Saves:
        state_transition_counts.csv
        state_transition_probabilities.csv
    """
    print("\nAnalysing occupancy state transitions...")

    # Identify the previous recorded state within each floor space.
    data["previous_state"] = (
        data.groupby("floorspaceid")["occupancystatus"].shift(1)
    )

    transitions = data.dropna(subset=["previous_state"])

    # Count how often each previous/current state combination occurs.
    transition_counts = (
        transitions.groupby(
            ["previous_state", "occupancystatus"]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # Save the raw transition counts.
    transition_counts.to_csv(
        OUTPUT_DIR / "state_transition_counts.csv",
        index=False,
    )

    print("\nMOST COMMON STATE TRANSITIONS")
    print(transition_counts.head(20).to_string(index=False))

    # Estimate the probability of the current state given the
    # previous recorded state.
    transition_counts["probability"] = (
        transition_counts["count"]
        / transition_counts.groupby("previous_state")["count"]
        .transform("sum")
    )

    transition_counts.to_csv(
        OUTPUT_DIR / "state_transition_probabilities.csv",
        index=False,
    )


def analyse_state_run_durations(data):
    """Summarise the observed spans of consecutive runs of each state.

    A new run starts when the recorded state changes, the first
    observation for a floor space is reached, or the gap between
    consecutive aggregated observations exceeds one interval.

    The observed span is the elapsed time between the first and last
    observations in a run. It does not establish the true start or end
    of physical occupancy, and a single-observation run has a span of
    zero minutes.

    Saves:
        observed_state_run_summary.csv
        observed_state_runs.csv
    """
    print("\nEstimating observed state-run durations...")

    # Find the previous state within each floor space.
    data["previous_state"] = (
        data.groupby("floorspaceid")["occupancystatus"].shift(1)
    )

    # Mark the start of a new run if the state changes or observations
    # are separated by a gap greater than five minutes.
    data["new_run"] = (
        data["previous_state"].isna()
        | data["occupancystatus"].ne(data["previous_state"])
        | data["gap_minutes"].gt(INTERVAL_MINUTES)
    )

    # Assign a unique sequential run identifier within each floor space.
    data["run_id"] = (
        data.groupby("floorspaceid")["new_run"].cumsum()
    )

    # Summarise the observations belonging to each run.
    runs = (
        data.groupby(
            ["floorspaceid", "run_id", "occupancystatus"],
            as_index=False,
        )
        .agg(
            observed_intervals=("interval", "size"),
            first_interval=("interval", "min"),
            last_interval=("interval", "max"),
        )
    )

    # Calculate the observed elapsed span of each run.
    runs["observed_span_minutes"] = (
        runs["last_interval"] - runs["first_interval"]
    ).dt.total_seconds() / 60

    # Summarise run counts and durations for each recorded state.
    duration_summary = (
        runs.groupby("occupancystatus")
        .agg(
            observed_runs=("run_id", "count"),
            median_observed_span_minutes=(
                "observed_span_minutes",
                "median",
            ),
            mean_observed_span_minutes=(
                "observed_span_minutes",
                "mean",
            ),
            p90_observed_span_minutes=(
                "observed_span_minutes",
                lambda x: x.quantile(0.90),
            ),
        )
        .reset_index()
    )

    duration_summary.to_csv(
        OUTPUT_DIR / "observed_state_run_summary.csv",
        index=False,
    )

    print("\nOBSERVED STATE-RUN SUMMARY")
    print(
        duration_summary.to_string(
            index=False,
            float_format="%.2f",
        )
    )

    # Save the details of every individual observed run.
    runs.to_csv(
        OUTPUT_DIR / "observed_state_runs.csv",
        index=False,
    )


def main():
    """Run the complete occupancy timing and transition analysis.

    Loads and aggregates the source data, performs the three analyses,
    and saves the aggregated observations and analysis outputs.
    """
    print("=" * 65)
    print("OCCUPANCY TIMING AND TRANSITION ANALYSIS")
    print("=" * 65)

    # Prepare the five-minute dataset used by all subsequent analyses.
    data = load_and_aggregate_data()

    # Examine observation frequency and state transition behaviour.
    analyse_observation_gaps(data)
    analyse_state_transitions(data)
    analyse_state_run_durations(data)

    # Save the aggregated dataset for inspection and reuse.
    data.to_csv(
        OUTPUT_DIR / "occupancy_5min_intervals.csv",
        index=False,
    )

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)

    print("\nImportant:")
    print("- Five-minute aggregation simplifies irregular source records.")
    print("- State-run durations are observed spans, not true durations.")
    print("- Long gaps are not treated as continuous state persistence.")
    print("- No environmental room mapping is assumed.")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
