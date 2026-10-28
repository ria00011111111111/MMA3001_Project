
"""Tests for occupancy observation timing calculations."""

import unittest
import pandas as pd


class TestTimingAnalysis(unittest.TestCase):

    def test_regular_observation_gaps(self):
        timestamps = pd.to_datetime([
            "2024-05-01 10:00:00",
            "2024-05-01 10:05:00",
            "2024-05-01 10:10:00",
        ])

        gaps = timestamps.to_series().diff().dt.total_seconds().div(60).dropna()

        self.assertEqual(gaps.tolist(), [5.0, 5.0])

    def test_irregular_observation_gap(self):
        timestamps = pd.to_datetime([
            "2024-05-01 10:00:00",
            "2024-05-01 10:05:00",
            "2024-05-01 10:20:00",
        ])

        gaps = timestamps.to_series().diff().dt.total_seconds().div(60).dropna()

        self.assertEqual(gaps.tolist(), [5.0, 15.0])

    def test_gaps_are_calculated_in_minutes(self):
        start = pd.Timestamp("2024-05-01 10:00:00")
        end = pd.Timestamp("2024-05-01 10:30:00")

        gap_minutes = (end - start).total_seconds() / 60

        self.assertEqual(gap_minutes, 30.0)


if __name__ == "__main__":
    unittest.main()
