
"""Tests for sensitivity and robustness calculations."""

import unittest


class TestSensitivity(unittest.TestCase):

    def test_absolute_change(self):
        baseline = 0.80
        modified = 0.84

        self.assertAlmostEqual(modified - baseline, 0.04)

    def test_relative_percentage_change(self):
        baseline = 0.80
        modified = 0.84

        percentage_change = (
            (modified - baseline) / baseline
        ) * 100

        self.assertAlmostEqual(percentage_change, 5.0)

    def test_no_change_gives_zero_sensitivity(self):
        baseline = 0.80
        modified = 0.80

        self.assertEqual(modified - baseline, 0.0)

    def test_sensitivity_detects_degradation(self):
        baseline = 0.90
        modified = 0.81

        change = modified - baseline

        self.assertLess(change, 0.0)


if __name__ == "__main__":
    unittest.main()
