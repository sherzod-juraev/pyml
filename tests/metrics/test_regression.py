"""Tests for regression evaluation metrics."""

import numpy as np
import pytest

from pyml.core.exceptions import ShapeMismatchError
from pyml.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

_REGRESSION_METRICS = [
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
]


class TestInputValidation:
    @pytest.mark.parametrize("metric", _REGRESSION_METRICS)
    def test_rejects_2d_y_true(self, metric):
        with pytest.raises(ValueError, match="y_true"):
            metric(np.ones((2, 2)), np.ones(2))

    @pytest.mark.parametrize("metric", _REGRESSION_METRICS)
    def test_rejects_2d_y_pred(self, metric):
        with pytest.raises(ValueError, match="y_pred"):
            metric(np.ones(2), np.ones((2, 2)))

    @pytest.mark.parametrize("metric", _REGRESSION_METRICS)
    def test_rejects_integer_y_true(self, metric):
        with pytest.raises(TypeError, match="y_true"):
            metric(np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]))

    @pytest.mark.parametrize("metric", _REGRESSION_METRICS)
    def test_rejects_integer_y_pred(self, metric):
        with pytest.raises(TypeError, match="y_pred"):
            metric(np.array([1.0, 2.0, 3.0]), np.array([1, 2, 3]))

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf])
    @pytest.mark.parametrize("metric", _REGRESSION_METRICS)
    def test_rejects_non_finite_y_true(self, metric, bad_value):
        with pytest.raises(ValueError, match="y_true"):
            metric(np.array([1.0, bad_value, 3.0]), np.array([1.0, 2.0, 3.0]))

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf])
    @pytest.mark.parametrize("metric", _REGRESSION_METRICS)
    def test_rejects_non_finite_y_pred(self, metric, bad_value):
        with pytest.raises(ValueError, match="y_pred"):
            metric(np.array([1.0, 2.0, 3.0]), np.array([1.0, bad_value, 3.0]))

    @pytest.mark.parametrize("metric", _REGRESSION_METRICS)
    def test_rejects_mismatched_samples(self, metric):
        with pytest.raises(ShapeMismatchError):
            metric(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


class TestMeanSquaredError:
    def test_perfect_predictions_give_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 5.0])
        # errors: 1, 0, 2 -> squared: 1, 0, 4 -> mean: 5/3
        assert mean_squared_error(y_true, y_pred) == pytest.approx(5 / 3)

    def test_rejects_mismatched_samples(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])
        with pytest.raises(ShapeMismatchError):
            mean_squared_error(y_true, y_pred)


class TestRootMeanSquaredError:
    def test_perfect_predictions_give_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert root_mean_squared_error(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 5.0])
        # mse = 5/3 -> rmse = sqrt(5/3)
        assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(5 / 3))


class TestMeanAbsoluteError:
    def test_perfect_predictions_give_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 5.0])
        # errors: 1, 0, 2 -> mean: 1.0
        assert mean_absolute_error(y_true, y_pred) == pytest.approx(1.0)

    def test_less_sensitive_to_outliers_than_mse(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 100.0])
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        assert mae < mse


class TestR2Score:
    def test_perfect_predictions_give_one(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_predicting_the_mean_gives_zero(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.full(3, np.mean(y_true))
        assert r2_score(y_true, y_pred) == pytest.approx(0.0)

    def test_worse_than_mean_gives_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])
        assert r2_score(y_true, y_pred) < 0.0

    def test_zero_variance_perfect_fit_gives_one(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        assert r2_score(y_true, y_pred) == pytest.approx(1.0)

    def test_zero_variance_imperfect_fit_gives_zero(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 4.0, 5.0])
        assert r2_score(y_true, y_pred) == pytest.approx(0.0)
