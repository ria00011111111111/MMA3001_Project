
"""Run the complete occupancy modelling test suite and save a report."""

import sys
import unittest
from pathlib import Path
from datetime import datetime


def main():
    test_dir = Path(__file__).resolve().parent
    project_root = test_dir.parent
    report_path = test_dir / "test_report.txt"

    # Allow tests to import modules from the project root.
    sys.path.insert(0, str(project_root))

    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(test_dir),
        pattern="test_*.py",
    )

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

        passed = (
            result.testsRun
            - len(result.failures)
            - len(result.errors)
            - len(result.skipped)
        )

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

    print(f"Test report saved to: {report_path}")
    print(f"Tests executed: {result.testsRun}")
    print("Overall result:", "PASSED" if result.wasSuccessful() else "FAILED")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
