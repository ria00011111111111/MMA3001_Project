
from pathlib import Path
from collections import defaultdict
import zipfile
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "occupancy_label_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ZIP_FILE = DATA_DIR / "5occupancySensor_MayToDec2024_9MRows (1).zip"

state_headcount = defaultdict(lambda: defaultdict(int))
state_previous = defaultdict(lambda: defaultdict(int))
total_rows = 0

print("Reading occupancy data...")

with zipfile.ZipFile(ZIP_FILE) as z:
    csv_name = z.namelist()[0]

    with z.open(csv_name) as f:
        for chunk in pd.read_csv(
            f,
            usecols=[
                "occupancystatus",
                "previousoccupancystatus",
                "headcount",
            ],
            chunksize=250_000,
            low_memory=False,
        ):
            total_rows += len(chunk)

            chunk["headcount"] = pd.to_numeric(
                chunk["headcount"], errors="coerce"
            )

            for (status, headcount), count in (
                chunk.groupby(["occupancystatus", "headcount"], dropna=False)
                .size()
                .items()
            ):
                state_headcount[status][headcount] += int(count)

            for (previous, current), count in (
                chunk.groupby(
                    ["previousoccupancystatus", "occupancystatus"],
                    dropna=False,
                )
                .size()
                .items()
            ):
                state_previous[previous][current] += int(count)

            print(f"Processed {total_rows:,} records...")

headcount_rows = []
for status, counts in state_headcount.items():
    for headcount, count in counts.items():
        headcount_rows.append({
            "occupancystatus": status,
            "headcount": headcount,
            "count": count,
        })

pd.DataFrame(headcount_rows).to_csv(
    OUTPUT_DIR / "state_vs_headcount.csv", index=False
)

transition_rows = []
for previous, counts in state_previous.items():
    for current, count in counts.items():
        transition_rows.append({
            "previous_occupancystatus": previous,
            "occupancystatus": current,
            "count": count,
        })

pd.DataFrame(transition_rows).to_csv(
    OUTPUT_DIR / "state_transitions.csv", index=False
)

print("\nHEADCOUNT BY OCCUPANCY STATE")
print(
    pd.DataFrame(headcount_rows)
    .sort_values(["occupancystatus", "count"], ascending=[True, False])
    .groupby("occupancystatus")
    .head(10)
    .to_string(index=False)
)

print("\nMOST COMMON STATE TRANSITIONS")
print(
    pd.DataFrame(transition_rows)
    .sort_values("count", ascending=False)
    .head(20)
    .to_string(index=False)
)

print(f"\nSaved results to: {OUTPUT_DIR}")
