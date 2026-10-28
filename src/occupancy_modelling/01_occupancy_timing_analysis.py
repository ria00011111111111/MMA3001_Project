
"""
Occupancy timing and transition analysis.

This script processes occupancy sensor records from a ZIP archive and
aggregates them into five-minute intervals for each floor space.

The analysis examines:
    1. Time gaps between consecutive observations.
    2. Counts and probabilities of recorded state transitions.
    3. Short-gap transitions where consecutive observations are no more
       than five minutes apart.
    4. Observed durations of consecutive runs of the same state.

The analysis describes recorded sensor states, not independently
verified physical occupancy. Run durations are observed spans between
the first and last observations in each run.

Outputs are saved to:
    outputs/occupancy_timing_analysis/
"""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

# Find the project root relative to this Python file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to the compressed occupancy dataset.
ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "5occupancySensor_MayToDec2024_9MRows (1).zip"
)

# Directory for generated CSV files.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_timing_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of source records loaded into memory at a time.
CHUNK_SIZE = 250_000

# Length of each aggregation interval, in minutes.
INTERVAL_MINUTES = 5

# Maximum gap allowed for the short-gap transition analysis.
MAX_TRANSITION_GAP_MINUTES = 5


def load_and_aggregate_data():
    """Load the ZIP dataset and aggregate records into five-minute bins.

    Records are read in chunks to reduce memory usage. Invalid timestamps
    and records missing essential fields are removed.

    For each floor space and five-minute interval, the observation with
    the latest timestamp is retained. Duplicate intervals across chunks
    are resolved after combining the chunk results.

    Returns:
        pd.DataFrame: Aggregated observations sorted by floor space and
        interval, including the retained observation timestamp.
    """
    if not ZIP_FILE.is_file():
        raise FileNotFoundError(
            f"Occupancy ZIP file not found:\n{ZIP_FILE}\n"
            "Check the filename and location in the SETTINGS section."
        )

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
            raise FileNotFoundError(
                f"No CSV file found inside {ZIP_FILE.name}."
            )

        # Read only the columns needed for this analysis.
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

                # Convert timestamps to UTC and discard invalid records.
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

                # Assign each timestamp to its five-minute interval.
                chunk["interval"] = chunk["collecteddate"].dt.floor(
                    f"{INTERVAL_MINUTES}min"
                )

                # Sort chronologically before retaining the latest record
                # in each floor-space interval.
                chunk = chunk.sort_values(
                    "collecteddate",
                    kind="stable",
                )

                chunk = chunk.drop_duplicates(
                    subset=["floorspaceid", "interval"],
                    keep="last",
                )

                # Store only the columns required by later analyses.
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

    # Combine chunk-level results and resolve duplicate intervals
    # that occurred in different chunks.
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

    # Ensure observations are ordered chronologically within each
    # floor space before calculating gaps and transitions.
    data = data.sort_values(
        ["floorspaceid", "interval"],
        kind="stable",
    ).reset_index(drop=True)

    print(f"Source records: {total_rows:,}")
    print(f"Aggregated observations: {len(data):,}")

    return data


def calculate_observation_gaps(data):
    """Calculate time gaps between consecutive observations per floor space.

    The gap is measured between the timestamps of the aggregated bins.
    Summary statistics describe observation frequency and the size of
    gaps in the recorded dataset.

    Returns:
        pd.DataFrame: Data containing previous intervals and gap lengths.
    """
    # Find the previous aggregated interval for each floor space.
    data["previous_interval"] = (
        data.groupby("floorspaceid")["interval"].shift(1)
    )

    # Calculate the elapsed time between consecutive interval timestamps.
    data["gap_minutes"] = (
        data["interval"] - data["previous_interval"]
    ).dt.total_seconds() / 60

    return data


def analyse_observation_gaps(data):
    """Summarise observation gaps for each floor space.

    Saves:
        observation_gap_summary.csv
    """
    print("\nAnalysing observation time gaps...")

    gap_data = data.dropna(subset=["gap_minutes"])

    # Calculate typical gaps and upper-percentile gap lengths.
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
            col_space=12,
        )
    )


def calculate_transition_counts(data):
    """Count transitions between consecutive recorded states.

    Transitions are calculated separately within each floor space.
    The function includes transitions across all observed time gaps.

    Returns:
        pd.DataFrame: Counts indexed by previous and current state.
    """
    # Identify the previous recorded state within each floor space.
    data["previous_state"] = (
        data.groupby("floorspaceid")["occupancystatus"].shift(1)
    )

    # Keep observations that have a preceding state.
    transitions = data.dropna(subset=["previous_state"]).copy()

    # Count each previous-state/current-state combination.
    transition_counts = (
        transitions.groupby(
            ["previous_state", "occupancystatus"]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return transition_counts


def save_transition_results(transition_counts, prefix=""):
    """Save transition counts, probabilities, and a readable summary.

    Probabilities are conditional on the previous recorded state.
    For each previous state, its outgoing transition probabilities
    sum to one, subject to numerical rounding.

    Args:
        transition_counts: DataFrame containing transition counts.
        prefix: Optional filename prefix for separate analyses.
    """
    # Save the raw transition counts.
    transition_counts.to_csv(
        OUTPUT_DIR / f"{prefix}state_transition_counts.csv",
        index=False,
    )

    print(f"\n{prefix.upper()}MOST COMMON STATE TRANSITIONS")
    print(
        transition_counts.head(20).to_string(
            index=False,
            col_space=12,
        )
    )

    # Calculate the probability of each next state, conditional on
    # the previous state.
    probabilities = transition_counts.copy()

    if not probabilities.empty:
        probabilities["probability"] = (
            probabilities["count"]
            / probabilities.groupby("previous_state")["count"]
            .transform("sum")
        )

    probabilities.to_csv(
        OUTPUT_DIR / f"{prefix}state_transition_probabilities.csv",
        index=False,
    )


def analyse_state_transitions(data):
    """Analyse transitions across all consecutive recorded observations.

    Long observation gaps are included, so these results describe
    transitions between consecutive records rather than exclusively
    short-term changes.

    Saves:
        state_transition_counts.csv
        state_transition_probabilities.csv
    """
    print("\nAnalysing occupancy state transitions...")

    transition_counts = calculate_transition_counts(data)

    save_transition_results(transition_counts)


def analyse_short_gap_transitions(data):
    """Analyse transitions where consecutive observations are close in time.

    Only transitions with a gap of five minutes or less are included.
    This helps distinguish short-gap transitions from transitions
    separated by longer periods without recorded observations.

    Saves:
        short_gap_state_transition_counts.csv
        short_gap_state_transition_probabilities.csv
    """
    print("\nAnalysing short-gap state transitions...")

    # Calculate the previous recorded state for each floor space.
    data["previous_state"] = (
        data.groupby("floorspaceid")["occupancystatus"].shift(1)
    )

    # Keep only observations with a preceding state and a short gap.
    short_gap_data = data.loc[
        data["previous_state"].notna()
        & data["gap_minutes"].le(MAX_TRANSITION_GAP_MINUTES)
    ].copy()

    # Count the transitions remaining after the gap filter.
    short_gap_counts = (
        short_gap_data.groupby(
            ["previous_state", "occupancystatus"]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(
        "Transitions included:",
        f"{len(short_gap_data):,}",
    )

    save_transition_results(
        short_gap_counts,
        prefix="short_gap_",
    )


def analyse_state_run_durations(data):
    """Summarise observed runs of consecutive identical recorded states.

    A new run begins when the recorded state changes, the floor space
    has no previous observation, or the gap between observations
    exceeds the aggregation interval.

    The observed span is the elapsed time between the first and last
    observations in a run. It is not the true physical duration of
    occupancy. A run containing one observation has a span of zero.

    Saves:
        observed_state_run_summary.csv
        observed_state_runs.csv
    """
    print("\nEstimating observed state-run durations...")

    # Identify the previous state within each floor space.
    data["previous_state"] = (
        data.groupby("floorspaceid")["occupancystatus"].shift(1)
    )

    # Start a new run at the first observation, after a state change,
    # or following a gap longer than one aggregation interval.
    data["new_run"] = (
        data["previous_state"].isna()
        | data["occupancystatus"].ne(data["previous_state"])
        | data["gap_minutes"].gt(INTERVAL_MINUTES)
    )

    # Assign a sequential run identifier within each floor space.
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

    # Calculate the elapsed span between the first and last observation.
    runs["observed_span_minutes"] = (
        runs["last_interval"] - runs["first_interval"]
    ).dt.total_seconds() / 60

    # Summarise run counts and observed spans for each recorded state.
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
            col_space=12,
        )
    )

    # Save details of each individual observed run.
    runs.to_csv(
        OUTPUT_DIR / "observed_state_runs.csv",
        index=False,
    )


def main():
    """Run all occupancy timing and transition analyses.

    Loads and aggregates the source data, calculates observation gaps,
    analyses state transitions, compares short-gap transitions, and
    summarises observed state-run durations.
    """
    print("=" * 65)
    print("OCCUPANCY TIMING AND TRANSITION ANALYSIS")
    print("=" * 65)

    # Load the source data and aggregate it into five-minute intervals.
    data = load_and_aggregate_data()

    # Calculate time gaps once and reuse them in subsequent analyses.
    data = calculate_observation_gaps(data)

    # Examine observation frequency and state transition behaviour.
    analyse_observation_gaps(data)
    analyse_state_transitions(data)
    analyse_short_gap_transitions(data)
    analyse_state_run_durations(data)

    # Save the aggregated observations for later analysis.
    data.to_csv(
        OUTPUT_DIR / "occupancy_5min_intervals.csv",
        index=False,
    )

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)

    print("\nImportant:")
    print("- Five-minute aggregation retains existing observations only.")
    print("- State-run durations are observed spans, not true durations.")
    print("- All-gap transitions may span long periods without observations.")
    print("- Short-gap transitions include gaps of five minutes or less.")
    print("- No environmental room mapping is assumed.")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()