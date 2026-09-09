"""Tests for classification evaluation metrics."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError, ShapeMismatchError
from pyml.metrics import accuracy_score, f1_score, precision_score, recall_score


class TestAccuracyScore:
    def test_perfect_predictions_give_one(self):
        y = np.array([0, 1, 2, 1, 0])
        assert accuracy_score(y, y) == pytest.approx(1.0)

    def test_known_value(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        # 3 out of 4 correct
        assert accuracy_score(y_true, y_pred) == pytest.approx(0.75)

    def test_rejects_mismatched_samples(self):
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])
        with pytest.raises(ShapeMismatchError):
            accuracy_score(y_true, y_pred)


class TestPrecisionScore:
    def test_perfect_predictions_give_one(self):
        y = np.array([0, 1, 2, 1, 0])
        assert precision_score(y, y) == pytest.approx(1.0)

    def test_micro_equals_accuracy(self):
        y_true = np.array([0, 1, 1, 0, 2])
        y_pred = np.array([0, 1, 0, 0, 2])
        assert precision_score(y_true, y_pred, "micro") == pytest.approx(
            accuracy_score(y_true, y_pred)
        )

    def test_known_binary_value(self):
        # y_true = [1, 1, 0, 0], y_pred = [1, 1, 1, 0]
        # class 0: TP=1 (idx 3), FP=0 -> precision 1.0
        # class 1: TP=2 (idx 0,1), FP=1 (idx 2 predicted 1 but true 0) -> precision 2/3
        # macro average: (1.0 + 2/3) / 2 = 5/6
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0])
        precisions_macro = precision_score(y_true, y_pred, "macro")
        assert precisions_macro == pytest.approx(5 / 6)

    def test_rejects_invalid_average(self):
        y = np.array([0, 1, 0])
        with pytest.raises(InvalidParameterError):
            precision_score(y, y, "invalid")  # type: ignore[arg-type]


class TestRecallScore:
    def test_perfect_predictions_give_one(self):
        y = np.array([0, 1, 2, 1, 0])
        assert recall_score(y, y) == pytest.approx(1.0)

    def test_micro_equals_accuracy(self):
        y_true = np.array([0, 1, 1, 0, 2])
        y_pred = np.array([0, 1, 0, 0, 2])
        assert recall_score(y_true, y_pred, "micro") == pytest.approx(
            accuracy_score(y_true, y_pred)
        )

    def test_known_value_with_missed_positive(self):
        # class 1 has 2 true instances, only 1 correctly recalled
        y_true = np.array([1, 1, 0])
        y_pred = np.array([1, 0, 0])
        recalls_macro = recall_score(y_true, y_pred, "macro")
        # class 0: recall 1.0, class 1: recall 0.5 -> macro avg 0.75
        assert recalls_macro == pytest.approx(0.75)


class TestF1Score:
    def test_perfect_predictions_give_one(self):
        y = np.array([0, 1, 2, 1, 0])
        assert f1_score(y, y) == pytest.approx(1.0)

    def test_micro_equals_accuracy(self):
        y_true = np.array([0, 1, 1, 0, 2])
        y_pred = np.array([0, 1, 0, 0, 2])
        assert f1_score(y_true, y_pred, "micro") == pytest.approx(accuracy_score(y_true, y_pred))

    def test_is_harmonic_mean_of_precision_and_recall(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        p = precision_score(y_true, y_pred, "macro")
        r = recall_score(y_true, y_pred, "macro")
        f1 = f1_score(y_true, y_pred, "macro")
        expected = 2 * (p * r) / (p + r)
        assert f1 == pytest.approx(expected)

    def test_zero_precision_and_recall_gives_zero(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert f1_score(y_true, y_pred, "macro") == pytest.approx(0.0, abs=1e-9)
