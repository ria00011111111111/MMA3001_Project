
"""
06_03_occupancy_environment_matching.py

Purpose:
1. Summarise occupancy floor spaces.
2. Inspect environmental sensor locations and measurement coverage.
3. Calculate temporal overlap between occupancy and environmental data.
4. Create mapping templates for unresolved room identities.
5. Report whether room-level modelling is currently justified.

This script does not modify the original datasets.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OCCUPANCY_AUDIT_DIR = OUTPUT_DIR / "occupancy_data_audit"
ENVIRONMENTAL_AUDIT_DIR = OUTPUT_DIR / "environmental_data_audit"

MATCHING_OUTPUT_DIR = OUTPUT_DIR / "occupancy_environment_matching"
MATCHING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OCCUPANCY_SUMMARY_FILE = (
    OCCUPANCY_AUDIT_DIR / "occupancy_floorspace_summary.csv"
)

OCCUPANCY_TIMESTAMP_FILE = (
    OCCUPANCY_AUDIT_DIR / "occupancy_timestamp_summary.csv"
)

ENVIRONMENTAL_READINGS_FILE = (
    ENVIRONMENTAL_AUDIT_DIR
    / "5EnvSensor_MayToDec2024_180kRows_readings.csv"
)

ENVIRONMENTAL_SENSOR_SUMMARY_FILE = (
    ENVIRONMENTAL_AUDIT_DIR / "environmental_sensor_summary.csv"
)

# Use the sensor-location mapping supplied in your project notes.
# Unknown locations are deliberately left blank.
ENVIRONMENTAL_SENSOR_IDS = [
    "6012002000227",
    "6012002000241",
    "6012002000326",
    "6012002000777",
    "6012002000869",
]

KNOWN_SENSOR_LOCATIONS = {
    "6012002000869": {
        "building": "60",
        "floor_room": "G.38",
        "area": "Compactus_upper",
        "physical_label": "EMS107",
        "mapping_status": "Known from supplied mapping",
    }
}


# ============================================================
# 2. CHECK REQUIRED INPUT FILES
# ============================================================

required_files = [
    OCCUPANCY_SUMMARY_FILE,
    OCCUPANCY_TIMESTAMP_FILE,
    ENVIRONMENTAL_READINGS_FILE,
    ENVIRONMENTAL_SENSOR_SUMMARY_FILE,
]

missing_files = [p for p in required_files if not p.exists()]

if missing_files:
    print("\nERROR: Required audit files are missing:")
    for path in missing_files:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")

    raise FileNotFoundError(
        "Run 06_2 and 06_3 environmental audit scripts first."
    )


# ============================================================
# 3. LOAD OCCUPANCY AUDIT RESULTS
# ============================================================

occupancy = pd.read_csv(
    OCCUPANCY_SUMMARY_FILE,
    dtype={"floorspaceid": "string"},
)

occupancy["first_timestamp_utc"] = pd.to_datetime(
    occupancy["first_timestamp_utc"],
    utc=True,
    errors="coerce",
)

occupancy["last_timestamp_utc"] = pd.to_datetime(
    occupancy["last_timestamp_utc"],
    utc=True,
    errors="coerce",
)

print("\n" + "=" * 65)
print("OCCUPANCY FLOOR-SPACE SUMMARY")
print("=" * 65)

display_columns = [
    "floorspaceid",
    "rows",
    "first_timestamp_utc",
    "last_timestamp_utc",
    "currently_occupied_rows",
    "recently_occupied_rows",
    "not_occupied_rows",
]

display_columns = [
    c for c in display_columns if c in occupancy.columns
]

print(occupancy[display_columns].to_string(index=False))

occupancy.to_csv(
    MATCHING_OUTPUT_DIR / "occupancy_floorspace_summary.csv",
    index=False,
)


# ============================================================
# 4. LOAD ENVIRONMENTAL READINGS
# ============================================================

print("\nLoading environmental readings...")

environmental = pd.read_csv(
    ENVIRONMENTAL_READINGS_FILE,
    dtype={"sensorid": "string"},
)

environmental["sensorid"] = environmental["sensorid"].str.strip()

environmental["timestamp_utc"] = pd.to_datetime(
    environmental["timestamp_utc"],
    utc=True,
    errors="coerce",
)

environmental["value"] = pd.to_numeric(
    environmental["value"],
    errors="coerce",
)

environmental = environmental.dropna(
    subset=["sensorid", "timestamp_utc", "variable"]
)

# Remove repeated copies of the same sensor-variable-time measurement.
# The earlier audit found no conflicting values for these keys.
measurement_keys = ["sensorid", "timestamp_utc", "variable"]

rows_before = len(environmental)

environmental = environmental.drop_duplicates(
    subset=measurement_keys,
    keep="first",
).copy()

print(f"Environmental rows loaded: {rows_before:,}")
print(f"Unique measurements retained: {len(environmental):,}")
print(f"Repeated rows excluded: {rows_before - len(environmental):,}")


# ============================================================
# 5. ENVIRONMENTAL SENSOR COVERAGE
# ============================================================

sensor_summary = (
    environmental.groupby("sensorid")
    .agg(
        first_measurement_utc=("timestamp_utc", "min"),
        last_measurement_utc=("timestamp_utc", "max"),
        unique_timestamps=("timestamp_utc", "nunique"),
        unique_variables=("variable", "nunique"),
        measurement_rows=("variable", "size"),
    )
    .reset_index()
)

sensor_summary["location_building"] = sensor_summary["sensorid"].map(
    lambda x: KNOWN_SENSOR_LOCATIONS.get(x, {}).get("building", "")
)

sensor_summary["location_floor_room"] = sensor_summary["sensorid"].map(
    lambda x: KNOWN_SENSOR_LOCATIONS.get(x, {}).get("floor_room", "")
)

sensor_summary["location_area"] = sensor_summary["sensorid"].map(
    lambda x: KNOWN_SENSOR_LOCATIONS.get(x, {}).get("area", "")
)

sensor_summary["physical_label"] = sensor_summary["sensorid"].map(
    lambda x: KNOWN_SENSOR_LOCATIONS.get(x, {}).get("physical_label", "")
)

sensor_summary["mapping_status"] = sensor_summary["sensorid"].map(
    lambda x: KNOWN_SENSOR_LOCATIONS.get(
        x, {}
    ).get("mapping_status", "UNRESOLVED - verify physical location")
)

sensor_summary.to_csv(
    MATCHING_OUTPUT_DIR / "environmental_sensor_coverage.csv",
    index=False,
)

print("\n" + "=" * 65)
print("ENVIRONMENTAL SENSOR COVERAGE AND LOCATIONS")
print("=" * 65)

print(sensor_summary.to_string(index=False))


# ============================================================
# 6. CALCULATE TEMPORAL OVERLAP
# ============================================================

# This compares date ranges only.
# It does NOT prove the sensors measure the same physical room.

occupancy_start = occupancy["first_timestamp_utc"].min()
occupancy_end = occupancy["last_timestamp_utc"].max()

overlap_records = []

for _, sensor in sensor_summary.iterrows():
    sensor_start = sensor["first_measurement_utc"]
    sensor_end = sensor["last_measurement_utc"]

    overlap_start = max(occupancy_start, sensor_start)
    overlap_end = min(occupancy_end, sensor_end)

    has_overlap = (
        pd.notna(overlap_start)
        and pd.notna(overlap_end)
        and overlap_start <= overlap_end
    )

    if has_overlap:
        overlap_days = (
            overlap_end - overlap_start
        ).total_seconds() / 86400
    else:
        overlap_start = pd.NaT
        overlap_end = pd.NaT
        overlap_days = 0.0

    overlap_records.append({
        "sensorid": sensor["sensorid"],
        "occupancy_first_utc": occupancy_start,
        "occupancy_last_utc": occupancy_end,
        "environment_first_utc": sensor_start,
        "environment_last_utc": sensor_end,
        "overlap_start_utc": overlap_start,
        "overlap_end_utc": overlap_end,
        "overlap_days": round(overlap_days, 2),
        "temporal_overlap": has_overlap,
        "physical_room_match_confirmed": False,
        "interpretation": (
            "Temporal overlap only; room mapping required"
            if has_overlap
            else "No temporal overlap"
        ),
    })

overlap = pd.DataFrame(overlap_records)

overlap.to_csv(
    MATCHING_OUTPUT_DIR / "environmental_occupancy_time_overlap.csv",
    index=False,
)

print("\n" + "=" * 65)
print("OVERALL TEMPORAL OVERLAP")
print("=" * 65)

print(overlap.to_string(index=False))


# ============================================================
# 7. CHECK EACH FLOOR SPACE AGAINST EACH SENSOR
# ============================================================

# A floor space is a candidate for further investigation if its
# observation period overlaps the sensor's measurement period.
# These are NOT confirmed room-level matches.

pair_records = []

for _, floor in occupancy.iterrows():
    for _, sensor in sensor_summary.iterrows():

        start = max(
            floor["first_timestamp_utc"],
            sensor["first_measurement_utc"],
        )

        end = min(
            floor["last_timestamp_utc"],
            sensor["last_measurement_utc"],
        )

        has_overlap = (
            pd.notna(start)
            and pd.notna(end)
            and start <= end
        )

        pair_records.append({
            "floorspaceid": floor["floorspaceid"],
            "sensorid": sensor["sensorid"],
            "floorspace_first_utc": floor["first_timestamp_utc"],
            "floorspace_last_utc": floor["last_timestamp_utc"],
            "sensor_first_utc": sensor["first_measurement_utc"],
            "sensor_last_utc": sensor["last_measurement_utc"],
            "overlap_start_utc": start if has_overlap else pd.NaT,
            "overlap_end_utc": end if has_overlap else pd.NaT,
            "overlap_days": (
                round((end - start).total_seconds() / 86400, 2)
                if has_overlap else 0.0
            ),
            "temporal_overlap": has_overlap,
            "room_match_confirmed": False,
            "status": (
                "Candidate only - verify room mapping"
                if has_overlap
                else "No temporal overlap"
            ),
        })

pair_overlap = pd.DataFrame(pair_records)

pair_overlap.to_csv(
    MATCHING_OUTPUT_DIR / "floorspace_sensor_overlap_candidates.csv",
    index=False,
)

print("\n" + "=" * 65)
print("FLOOR-SPACE / SENSOR OVERLAP CANDIDATES")
print("=" * 65)

print(
    pair_overlap[
        [
            "floorspaceid",
            "sensorid",
            "overlap_days",
            "temporal_overlap",
            "status",
        ]
    ].to_string(index=False)
)


# ============================================================
# 8. CREATE LOCATION MAPPING TEMPLATES
# ============================================================

# Fill these templates only when authoritative metadata is available.

occupancy_mapping_template = occupancy[
    ["floorspaceid"]
].drop_duplicates().copy()

occupancy_mapping_template["building"] = ""
occupancy_mapping_template["floor"] = ""
occupancy_mapping_template["room_or_area"] = ""
occupancy_mapping_template["mapping_source"] = ""
occupancy_mapping_template["verified"] = "No"

occupancy_mapping_template.to_csv(
    MATCHING_OUTPUT_DIR / "occupancy_floorspace_location_mapping.csv",
    index=False,
)

environmental_mapping_template = sensor_summary[
    ["sensorid"]
].drop_duplicates().copy()

environmental_mapping_template["building"] = ""
environmental_mapping_template["floor"] = ""
environmental_mapping_template["room_or_area"] = ""
environmental_mapping_template["physical_label"] = ""
environmental_mapping_template["mapping_source"] = ""
environmental_mapping_template["verified"] = "No"

# Populate the one known mapping without overwriting other entries.
for idx, row in environmental_mapping_template.iterrows():
    known = KNOWN_SENSOR_LOCATIONS.get(row["sensorid"])

    if known:
        environmental_mapping_template.at[idx, "building"] = (
            known["building"]
        )
        environmental_mapping_template.at[idx, "floor"] = (
            known["floor_room"]
        )
        environmental_mapping_template.at[idx, "room_or_area"] = (
            known["area"]
        )
        environmental_mapping_template.at[idx, "physical_label"] = (
            known["physical_label"]
        )
        environmental_mapping_template.at[idx, "mapping_source"] = (
            "Previously supplied sensor location table"
        )
        environmental_mapping_template.at[idx, "verified"] = "Yes"

environmental_mapping_template.to_csv(
    MATCHING_OUTPUT_DIR / "environmental_sensor_location_mapping.csv",
    index=False,
)


# ============================================================
# 9. WRITE A SUMMARY OF WHAT IS STILL NEEDED
# ============================================================

unmapped_occupancy = len(occupancy_mapping_template)
unmapped_environmental = (
    environmental_mapping_template["verified"] == "No"
).sum()

candidate_pairs = int(pair_overlap["temporal_overlap"].sum())

summary_lines = [
    "OCCUPANCY-ENVIRONMENT MATCHING INVESTIGATION",
    "",
    f"Occupancy floor spaces found: {len(occupancy):,}",
    f"Environmental sensors found: {len(sensor_summary):,}",
    f"Floor-space/sensor pairs with temporal overlap: {candidate_pairs:,}",
    "",
    f"Occupancy floor spaces requiring location verification: "
    f"{unmapped_occupancy:,}",
    f"Environmental sensors requiring location verification: "
    f"{unmapped_environmental:,}",
    "",
    "IMPORTANT:",
    "- Temporal overlap does not prove a physical room match.",
    "- Occupancy floor-space UUIDs must be mapped to real rooms.",
    "- Environmental sensor locations must be verified.",
    "- Do not train a room-level model until compatible locations",
    "  and suitable timestamp alignment have been established.",
    "",
    "NEXT ACTIONS:",
    "1. Find metadata mapping occupancy floorspaceid values to rooms.",
    "2. Verify physical locations of the environmental sensors.",
    "3. Confirm timestamp meanings and occupancy-state semantics.",
    "4. Only then create a matched modelling dataset.",
]

summary_path = MATCHING_OUTPUT_DIR / "matching_investigation_summary.txt"

summary_path.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

print("\n" + "=" * 65)
print("INVESTIGATION COMPLETE")
print("=" * 65)

print("\n".join(summary_lines))

print(f"\nReports saved to:\n{MATCHING_OUTPUT_DIR}")
