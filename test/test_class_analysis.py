
"""Tests for occupancy classification."""

import unittest
from sklearn.metrics import confusion_matrix, recall_score


class TestClassAnalysis(unittest.TestCase):

    def test_confusion_matrix_counts_correct_predictions(self):
        actual = ["occupied", "empty", "occupied", "empty"]
        predicted = ["occupied", "empty", "empty", "empty"]

        matrix = confusion_matrix(
            actual,
            predicted,
            labels=["empty", "occupied"],
        )

        self.assertEqual(matrix.sum(), 4)
        self.assertEqual(matrix[0, 0], 2)
        self.assertEqual(matrix[1, 1], 1)

    def test_recall_for_each_class(self):
        actual = ["occupied", "occupied", "empty", "empty"]
        predicted = ["occupied", "empty", "empty", "empty"]

        scores = recall_score(
            actual,
            predicted,
            labels=["empty", "occupied"],
            average=None,
            zero_division=0,
        )

        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores[1], 0.5)

    def test_unexpected_class_can_be_detected(self):
        expected_classes = {"occupied", "empty"}
        observed_classes = {"occupied", "empty", "unknown"}

        self.assertEqual(
            observed_classes - expected_classes,
            {"unknown"},
        )


if __name__ == "__main__":
    unittest.main()
