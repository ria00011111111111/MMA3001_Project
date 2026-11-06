
"""Run the complete occupancy modelling test suite and save a report."""

import sys
import unittest
import shutil

from pathlib import Path
from datetime import datetime


def main():
    """Clean Python caches, execute tests and generate a test report."""

    # Identify project directories and report location.
    test_dir = Path(__file__).resolve().parent
    project_root = test_dir.parent
    report_path = test_dir / "test_report.txt"

    # Prevent Python from generating new bytecode cache files.
    sys.dont_write_bytecode = True

    # Remove existing __pycache__ directories throughout the project.
    for cache_dir in project_root.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)

    # Allow tests to import modules from the project root.
    sys.path.insert(0, str(project_root))

    # Discover all test files.
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(test_dir),
        pattern="test_*.py",
    )

    # Execute tests and save the results.
    with report_path.open("w", encoding="utf-8") as report:

        report.write("OCCUPANCY MODELLING TEST REPORT\n")
        report.write("=" * 40 + "\n")
        report.write(
            f"Execution date: {datetime.now().astimezone()}\n\n"
        )

        runner = unittest.TextTestRunner(
            stream=report,
            verbosity=2,
        )

        result = runner.run(suite)

        # Calculate the number of successful tests.
        passed = (
            result.testsRun
            - len(result.failures)
            - len(result.errors)
            - len(result.skipped)
        )

        # Write the final test summary.
        report.write("\n" + "=" * 40 + "\n")
        report.write(f"Tests run: {result.testsRun}\n")
        report.write(f"Passed: {passed}\n")
        report.write(f"Failures: {len(result.failures)}\n")
        report.write(f"Errors: {len(result.errors)}\n")
        report.write(f"Skipped: {len(result.skipped)}\n")

        report.write(
            "Overall result: "
            + ("PASSED" if result.wasSuccessful() else "FAILED")
            + "\n"
        )

    # Display the test results in the terminal.
    print(f"Test report saved to: {report_path}")
    print(f"Tests executed: {result.testsRun}")
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {len(result.failures)}")
    print(f"Tests errored: {len(result.errors)}")
    print(f"Tests skipped: {len(result.skipped)}")

    print(
        "Overall result:",
        "PASSED" if result.wasSuccessful() else "FAILED",
    )

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
