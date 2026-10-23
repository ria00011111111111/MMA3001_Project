
from pathlib import Path
import subprocess
import sys
import os

# ==================================================
# MMA3001 PROJECT RUNNER
# Runs all five scripts in sequence
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# --------------------------------------------------
# Create output directories
# --------------------------------------------------

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

for folder in [
    "data_quality",
    "occupancy",
    "evaluation",
    "eda",
]:
    (OUTPUTS_DIR / folder).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Find scripts, allowing for filename variations
# --------------------------------------------------

def find_script(possible_names):
    for name in possible_names:
        candidate = SRC_DIR / name

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"\nCould not find a required script in:\n{SRC_DIR}\n"
        f"Expected one of: {', '.join(possible_names)}"
    )


# --------------------------------------------------
# Scripts in execution order
# --------------------------------------------------

script_definitions = [
    (
        "1/5 - Data cleaning",
        ["01_datacleaning.py"],
    ),
    (
        "2/5 - Exploratory data analysis",
        [
            "02_exploratorydataanalysis.py",
            "02_exploratorydataanalysis",
            "02_exploratory_data_analysis.py",
        ],
    ),
    (
        "3/5 - Feature engineering",
        [
            "03_feature_engineering.py",
            "03_featureengineering.py",
        ],
    ),
    (
        "4/5 - Occupancy model",
        [
            "04_occupancymodel.py",
            "04_occupancy_model.py",
        ],
    ),
    (
        "5/5 - Evaluation",
        ["05_evaluation.py"],
    ),
]


# --------------------------------------------------
# Check project folders
# --------------------------------------------------

print("=" * 65)
print("MMA3001 SENSOR OCCUPANCY PROJECT")
print("=" * 65)

print(f"Project root: {PROJECT_ROOT}")
print(f"Python: {sys.executable}")

if not SRC_DIR.is_dir():
    sys.exit(f"ERROR: src folder not found: {SRC_DIR}")

if not DATA_DIR.is_dir():
    sys.exit(f"ERROR: data folder not found: {DATA_DIR}")

zip_files = list(DATA_DIR.glob("*.zip"))

if not zip_files:
    print(
        "\nWARNING: No ZIP dataset found in the data folder. "
        "Check that your sensor dataset is in data/."
    )


# Locate all scripts before executing anything
try:
    scripts = [
        (description, find_script(names))
        for description, names in script_definitions
    ]
except FileNotFoundError as error:
    sys.exit(str(error))


# --------------------------------------------------
# Force Matplotlib to use Agg
# --------------------------------------------------

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"

# --------------------------------------------------
# Execute all five scripts
# --------------------------------------------------

for index, (description, script_path) in enumerate(scripts, start=1):

    print("\n" + "=" * 65)
    print(description)
    print(f"File: {script_path}")
    print("=" * 65)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            check=False,
        )

    except OSError as error:
        print(f"\nERROR: Could not start {script_path.name}")
        print(error)
        sys.exit(1)

    if result.returncode != 0:
        print("\n" + "=" * 65)
        print(f"FAILED: {script_path.name}")
        print(f"Exit code: {result.returncode}")
        print("=" * 65)
        print(
            "The pipeline has stopped. Fix the error above "
            "before continuing."
        )
        sys.exit(result.returncode)

    print(f"\nCompleted successfully: {description}")


# --------------------------------------------------
# Finish
# --------------------------------------------------

print("\n" + "=" * 65)
print("ALL FIVE SCRIPTS COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nCheck the generated files in:")
print(OUTPUTS_DIR)
