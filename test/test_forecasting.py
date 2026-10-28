
"""Tests for forecast validation metrics."""

import unittest
from sklearn.metrics import accuracy_score, f1_score


class TestValidation(unittest.TestCase):

    def test_perfect_prediction_accuracy(self):
        actual = ["occupied", "empty", "occupied"]
        predicted = ["occupied", "empty", "occupied"]

        self.assertEqual(accuracy_score(actual, predicted), 1.0)

    def test_incorrect_prediction_reduces_accuracy(self):
        actual = ["occupied", "empty", "occupied", "empty"]
        predicted = ["occupied", "occupied", "occupied", "empty"]

        self.assertEqual(accuracy_score(actual, predicted), 0.75)

    def test_macro_f1_is_between_zero_and_one(self):
        actual = ["occupied", "empty", "occupied", "empty"]
        predicted = ["occupied", "occupied", "occupied", "empty"]

        score = f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0,
        )

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_perfect_predictions_have_perfect_f1(self):
        actual = ["occupied", "empty", "occupied"]
        score = f1_score(actual, actual, average="macro")

        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
