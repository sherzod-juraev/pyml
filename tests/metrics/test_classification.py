"""Tests for classification evaluation metrics."""

import numpy as np
import pytest

from pyml.core.exceptions import InvalidParameterError, ShapeMismatchError
from pyml.metrics import accuracy_score, f1_score, precision_score, recall_score

_CLASSIFICATION_METRICS = [accuracy_score, precision_score, recall_score, f1_score]


class TestInputValidation:
    @pytest.mark.parametrize("metric", _CLASSIFICATION_METRICS)
    def test_rejects_2d_y_true(self, metric):
        with pytest.raises(ValueError, match="y_true"):
            metric(np.array([[0, 1], [1, 0]]), np.array([0, 1]))

    @pytest.mark.parametrize("metric", _CLASSIFICATION_METRICS)
    def test_rejects_2d_y_pred(self, metric):
        with pytest.raises(ValueError, match="y_pred"):
            metric(np.array([0, 1]), np.array([[0, 1], [1, 0]]))

    @pytest.mark.parametrize("metric", _CLASSIFICATION_METRICS)
    def test_rejects_float_y_true(self, metric):
        with pytest.raises(TypeError, match="y_true"):
            metric(np.array([0.0, 1.0]), np.array([0, 1]))

    @pytest.mark.parametrize("metric", _CLASSIFICATION_METRICS)
    def test_rejects_float_y_pred(self, metric):
        with pytest.raises(TypeError, match="y_pred"):
            metric(np.array([0, 1]), np.array([0.0, 1.0]))

    @pytest.mark.parametrize("metric", _CLASSIFICATION_METRICS)
    def test_rejects_mismatched_samples(self, metric):
        with pytest.raises(ShapeMismatchError):
            metric(np.array([0, 1, 0]), np.array([0, 1]))


class TestPerClassCounts:
    def test_known_three_class_values(self):
        # class 0: TP=1, FP=1, FN=1 | class 1: TP=1, FP=1, FN=1 | class 2: TP=1, FP=0, FN=0
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2])

        assert precision_score(y_true, y_pred, "macro") == pytest.approx((0.5 + 0.5 + 1.0) / 3)
        assert recall_score(y_true, y_pred, "macro") == pytest.approx((0.5 + 0.5 + 1.0) / 3)
        assert f1_score(y_true, y_pred, "macro") == pytest.approx((0.5 + 0.5 + 1.0) / 3)

    def test_class_never_predicted_has_zero_precision_not_nan(self):
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 0, 0])

        result = precision_score(y_true, y_pred, "macro")

        assert np.isfinite(result)
        assert result == pytest.approx((1 / 3 + 0.0) / 2)

    def test_weighted_scores_match_manual_counts(self, rng):
        y_true = rng.integers(0, 4, size=200)
        y_pred = rng.integers(0, 4, size=200)
        support = np.array([np.sum(y_true == c) for c in range(4)]) / 200
        tp = np.array([np.sum((y_pred == c) & (y_true == c)) for c in range(4)])
        predicted = np.array([np.sum(y_pred == c) for c in range(4)])
        actual = np.array([np.sum(y_true == c) for c in range(4)])

        precision = tp / predicted
        recall = tp / actual
        f1 = 2 * precision * recall / (precision + recall)

        assert precision_score(y_true, y_pred) == pytest.approx(np.sum(support * precision))
        assert recall_score(y_true, y_pred) == pytest.approx(np.sum(support * recall))
        assert f1_score(y_true, y_pred) == pytest.approx(np.sum(support * f1))


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
