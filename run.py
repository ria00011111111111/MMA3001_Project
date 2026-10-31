
"""
Main execution script for the MMA3001 occupancy forecasting project.

Runs the seven project scripts sequentially in the intended workflow order.
Each script is executed using the Python interpreter associated with the
current environment.

The runner records the execution time of each script and stops immediately
if a script returns a non-zero exit code.
"""

import subprocess
import sys
import time

from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------------------------

# Locate the project root from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parent

# All modelling scripts are stored in this directory.
SCRIPT_DIR = PROJECT_ROOT / "src" / "occupancy_modelling"

# Use the Python interpreter from the currently active environment.
PYTHON_EXECUTABLE = sys.executable

# Scripts are executed in the same order as the project's documented workflow.
SCRIPTS = [
    "01_occupancy_timing_analysis.py",
    "02_occupancy_forecast_visualisation.py",
    "03_occupancy_forecast_validation_audit.py",
    "04_actual_forecasting.py",
    "05_rolling_occupancy_modelling.py",
    "06_rolling_class_analysis.py",
    "07_occupancy_sensitivity.py",
]


# ---------------------------------------------------------------------------
# 2. VALIDATE PROJECT CONFIGURATION
# ---------------------------------------------------------------------------

def validate_configuration():
    """
    Check that the project directory and required scripts exist.

    Raises:
        FileNotFoundError: If the script directory or any configured script
        cannot be found.
    """

    # Confirm that the expected script directory exists.
    if not SCRIPT_DIR.is_dir():
        raise FileNotFoundError(
            f"Script directory not found: {SCRIPT_DIR}"
        )

    # Check every script before starting execution.
    missing_scripts = [
        filename
        for filename in SCRIPTS
        if not (SCRIPT_DIR / filename).is_file()
    ]

    if missing_scripts:
        missing_list = "\n".join(
            f"  - {filename}"
            for filename in missing_scripts
        )

        raise FileNotFoundError(
            "The following project scripts could not be found:\n"
            f"{missing_list}"
        )


# ---------------------------------------------------------------------------
# 3. EXECUTE INDIVIDUAL SCRIPTS
# ---------------------------------------------------------------------------

def run_script(filename):
    """
    Execute one project script and measure its execution time.

    Args:
        filename (str): Name of the Python script to execute.

    Returns:
        dict: Script name, execution time and exit status.

    Raises:
        subprocess.CalledProcessError: If the script exits unsuccessfully.
    """

    script_path = SCRIPT_DIR / filename

    print("\n" + "=" * 70)
    print(f"RUNNING: {filename}")
    print("=" * 70)

    # Record the start time to calculate the script's execution duration.
    start_time = time.perf_counter()

    try:
        # Run the script using the current Python environment.
        # The project root is used as the working directory so relative
        # project paths behave consistently across script executions.
        subprocess.run(
            [
                PYTHON_EXECUTABLE,
                str(script_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        # Calculate elapsed time after successful completion.
        elapsed_time = time.perf_counter() - start_time

        print("\n" + "-" * 70)
        print(f"COMPLETED: {filename}")
        print(f"Execution time: {elapsed_time:.2f} seconds")
        print("-" * 70)

        return {
            "script": filename,
            "execution_time": elapsed_time,
            "status": "SUCCESS",
        }

    except subprocess.CalledProcessError as error:

        # Calculate execution time even when the script fails.
        elapsed_time = time.perf_counter() - start_time

        print("\n" + "!" * 70)
        print(f"FAILED: {filename}")
        print(f"Exit code: {error.returncode}")
        print(f"Execution time: {elapsed_time:.2f} seconds")
        print("!" * 70)

        # Re-raise the error so the main runner stops the workflow.
        raise


# ---------------------------------------------------------------------------
# 4. DISPLAY EXECUTION SUMMARY
# ---------------------------------------------------------------------------

def display_summary(results, total_time):
    """
    Display a summary of successfully executed scripts.

    Args:
        results (list): Execution information for completed scripts.
        total_time (float): Total workflow execution time in seconds.
    """

    print("\n" + "=" * 70)
    print("OCCUPANCY MODELLING PIPELINE SUMMARY")
    print("=" * 70)

    print(
        f"{'Script':45s} {'Time (s)':>12s} {'Status':>12s}"
    )

    print("-" * 70)

    for result in results:
        print(
            f"{result['script']:45s} "
            f"{result['execution_time']:12.2f} "
            f"{result['status']:>12s}"
        )

    print("-" * 70)

    print(
        f"Scripts completed: {len(results)}/{len(SCRIPTS)}"
    )

    print(
        f"Total execution time: {total_time:.2f} seconds"
    )

    print("=" * 70)


# ---------------------------------------------------------------------------
# 5. MAIN WORKFLOW
# ---------------------------------------------------------------------------

def main():
    """
    Execute the complete occupancy modelling workflow.

    The scripts are run sequentially in the configured order. If any script
    fails, subsequent scripts are not executed.
    
    Returns:
        int: Zero if the complete workflow succeeds, otherwise one.
    """

    print("=" * 70)
    print("MMA3001 OCCUPANCY MODELLING PIPELINE")
    print("=" * 70)

    print(
        f"Execution started: "
        f"{datetime.now().astimezone()}"
    )

    print(
        f"Python interpreter: {PYTHON_EXECUTABLE}"
    )

    print(
        f"Project directory: {PROJECT_ROOT}"
    )

    # Validate all paths before beginning the workflow.
    try:
        validate_configuration()

    except FileNotFoundError as error:
        print(f"\nCONFIGURATION ERROR:\n{error}")
        return 1

    results = []

    # Start timing the complete workflow.
    workflow_start = time.perf_counter()

    # Execute each script in the documented project order.
    for filename in SCRIPTS:

        try:
            result = run_script(filename)
            results.append(result)

        except subprocess.CalledProcessError:
            # Stop immediately so later scripts do not run using incomplete
            # or missing outputs from the failed script.
            total_time = (
                time.perf_counter() - workflow_start
            )

            print("\nPIPELINE TERMINATED DUE TO SCRIPT FAILURE.")

            display_summary(
                results,
                total_time,
            )

            return 1

    # Calculate total workflow execution time.
    total_time = (
        time.perf_counter() - workflow_start
    )

    # Display the final execution summary.
    display_summary(
        results,
        total_time,
    )

    print(
        "\nAll occupancy modelling scripts completed successfully."
    )

    return 0


# ---------------------------------------------------------------------------
# 6. SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
