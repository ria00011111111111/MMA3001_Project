
"""Run the seven occupancy analysis scripts in sequence."""

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = PROJECT_ROOT / "src" / "occupancy_modelling"

SCRIPTS = [
    "01_occupancy_timing_analysis.py",
    "02_occupancy_forecast_visualisation.py",
    "03_occupancy_forecast_validation_audit.py",
    "04_actual_forecasting.py",
    "05_rolling_occupancy_modelling.py",
    "06_rolling_class_analysis.py",
    "07_occupancy_sensitivity.py",
]


def main():
    start_time = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    print("=" * 60, flush=True)
    print("OCCUPANCY ANALYSIS PIPELINE STARTED", flush=True)
    print(f"Project: {PROJECT_ROOT}", flush=True)
    print(f"Scripts to run: {len(SCRIPTS)}", flush=True)
    print("=" * 60, flush=True)

    for number, filename in enumerate(SCRIPTS, start=1):
        script_path = SCRIPT_DIR / filename

        if not script_path.is_file():
            print(f"\nERROR: Script not found: {script_path}", flush=True)
            return 1

        print(
            f"\n[{number}/{len(SCRIPTS)}] STARTING: {filename}",
            flush=True,
        )
        script_start = time.perf_counter()

        try:
            subprocess.run(
                [sys.executable, "-u", str(script_path)],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            print(f"\nFAILED: {filename}", flush=True)
            print(f"Exit code: {error.returncode}", flush=True)
            print("PIPELINE STOPPED DUE TO AN ERROR.", flush=True)
            return error.returncode or 1

        elapsed = time.perf_counter() - script_start
        print(
            f"[{number}/{len(SCRIPTS)}] COMPLETED: {filename} "
            f"({elapsed:.2f} seconds)",
            flush=True,
        )

    total_elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 60, flush=True)
    print("ALL SEVEN SCRIPTS COMPLETED SUCCESSFULLY", flush=True)
    print(f"Total runtime: {total_elapsed:.2f} seconds", flush=True)
    print("=" * 60, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
