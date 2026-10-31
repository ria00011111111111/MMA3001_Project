"""
MMA3001 Occupancy Forecasting Project
Main pipeline runner.

Executes all analysis and modelling scripts in dependency order,
records execution times, and stops if any script fails.
"""

import subprocess
import sys
import time
from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = PROJECT_ROOT / "src" / "occupancy_modelling"


# Scripts are ordered according to their dependencies.
SCRIPTS = [
    "01_occupancy_timing_analysis.py",
    "04_actual_forecasting.py",
    "02_occupancy_forecast_visualisation.py",
    "03_occupancy_forecast_validation_audit.py",
    "05_rolling_occupancy_modelling.py",
    "06_rolling_class_analysis.py",
    "07_occupancy_sensitivity.py",
]


def main():

    total_start = time.perf_counter()
    execution_times = []

    print("=" * 70)
    print("MMA3001 OCCUPANCY FORECASTING PIPELINE")
    print("=" * 70)

    for index, script_name in enumerate(SCRIPTS, start=1):

        script_path = SCRIPT_DIR / script_name

        print(f"\n[{index}/{len(SCRIPTS)}] STARTING: {script_name}")
        print("=" * 70)

        if not script_path.exists():
            print(f"ERROR: Script not found: {script_path}")
            print("PIPELINE STOPPED.")
            sys.exit(1)

        start_time = time.perf_counter()

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT
        )

        elapsed = time.perf_counter() - start_time
        execution_times.append((script_name, elapsed))

        if result.returncode != 0:
            print(f"\nFAILED: {script_name}")
            print(f"Exit code: {result.returncode}")
            print("PIPELINE STOPPED DUE TO AN ERROR.")
            sys.exit(result.returncode)

        print(f"\nCOMPLETED: {script_name}")
        print(f"Execution time: {elapsed:.2f} seconds")

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nEXECUTION TIME SUMMARY")
    print("-" * 70)

    for script_name, elapsed in execution_times:
        print(f"{script_name:<45} {elapsed:>10.2f} seconds")

    print("-" * 70)
    print(f"{'TOTAL EXECUTION TIME':<45} {total_time:>10.2f} seconds")


if __name__ == "__main__":
    main()