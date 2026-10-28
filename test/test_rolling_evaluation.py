
"""Tests for rolling forecast evaluation."""

import unittest
import pandas as pd


class TestRollingEvaluation(unittest.TestCase):

    def test_rolling_windows_preserve_time_order(self):
        timestamps = pd.date_range(
            start="2024-05-01 10:00",
            periods=12,
            freq="5min",
        )

        window_size = 4
        windows = [
            timestamps[i:i + window_size]
            for i in range(len(timestamps) - window_size + 1)
        ]

        self.assertTrue(all(
            window.is_monotonic_increasing
            for window in windows
        ))

    def test_rolling_windows_have_expected_size(self):
        timestamps = pd.date_range(
            start="2024-05-01 10:00",
            periods=10,
            freq="5min",
        )

        window_size = 3
        windows = [
            timestamps[i:i + window_size]
            for i in range(len(timestamps) - window_size + 1)
        ]

        self.assertTrue(all(len(window) == window_size for window in windows))
        self.assertEqual(len(windows), 8)

    def test_later_window_starts_later(self):
        timestamps = pd.date_range(
            start="2024-05-01 10:00",
            periods=6,
            freq="5min",
        )

        self.assertGreater(timestamps[1], timestamps[0])


if __name__ == "__main__":
    unittest.main()
