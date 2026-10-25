"""Tests for the shared hyper-parameter checks in pyml.core.param_checks."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from pyml.core.exceptions import InvalidParameterError, ValidationError
from pyml.core.param_checks import (
    DISTANCE_METRICS,
    check_at_least,
    check_choice,
    check_in_range,
    check_non_negative,
    check_positive,
)

NAN = float("nan")
INF = float("inf")


class TestCheckPositive:
    @pytest.mark.parametrize("value", [1e-12, 0.5, 1, 7, INF])
    def test_accepts_positive_values(self, value):
        check_positive("p", value)

    @pytest.mark.parametrize("value", [0, 0.0, -1e-12, -3, -INF, NAN])
    def test_rejects_zero_negative_and_nan(self, value):
        with pytest.raises(InvalidParameterError):
            check_positive("p", value)

    def test_infinity_is_rejected_only_when_finite_is_required(self):
        check_positive("p", INF)

        with pytest.raises(InvalidParameterError):
            check_positive("p", INF, finite=True)

    def test_finite_flag_still_accepts_finite_positive_values(self):
        check_positive("p", 2.5, finite=True)

    def test_message_names_the_parameter_and_the_value(self):
        with pytest.raises(InvalidParameterError, match=r"radius.*-2\.5"):
            check_positive("radius", -2.5)


class TestCheckNonNegative:
    @pytest.mark.parametrize("value", [0, 0.0, 1e-12, 3, INF])
    def test_accepts_zero_and_positive_values(self, value):
        check_non_negative("p", value)

    @pytest.mark.parametrize("value", [-1e-12, -3, -INF, NAN])
    def test_rejects_negative_and_nan(self, value):
        with pytest.raises(InvalidParameterError):
            check_non_negative("p", value)

    def test_infinity_is_rejected_only_when_finite_is_required(self):
        check_non_negative("p", INF)

        with pytest.raises(InvalidParameterError):
            check_non_negative("p", INF, finite=True)

    def test_message_names_the_parameter_and_the_value(self):
        with pytest.raises(InvalidParameterError, match=r"alpha.*-1"):
            check_non_negative("alpha", -1)


class TestCheckAtLeast:
    @pytest.mark.parametrize(("value", "minimum"), [(1, 1), (5, 1), (0, 0), (2.5, 2), (INF, 1)])
    def test_accepts_values_at_or_above_the_minimum(self, value, minimum):
        check_at_least("p", value, minimum)

    @pytest.mark.parametrize(("value", "minimum"), [(0, 1), (-5, 0), (0.99, 1), (NAN, 1)])
    def test_rejects_values_below_the_minimum_and_nan(self, value, minimum):
        with pytest.raises(InvalidParameterError):
            check_at_least("p", value, minimum)

    def test_message_names_the_parameter_the_minimum_and_the_value(self):
        with pytest.raises(InvalidParameterError, match=r"n_clusters.*at least 1.*0"):
            check_at_least("n_clusters", 0, 1)


class TestCheckInRange:
    @pytest.mark.parametrize("value", [0, 0.0, 0.5, 1, 1.0])
    def test_accepts_values_inside_and_on_both_bounds(self, value):
        check_in_range("p", value, 0, 1)

    @pytest.mark.parametrize("value", [-1e-12, 1.0000001, 2, -INF, INF, NAN])
    def test_rejects_values_outside_the_range_and_nan(self, value):
        with pytest.raises(InvalidParameterError):
            check_in_range("p", value, 0, 1)

    def test_message_names_the_parameter_the_bounds_and_the_value(self):
        with pytest.raises(InvalidParameterError, match=r"l1_ratio.*\[0, 1\].*1\.5"):
            check_in_range("l1_ratio", 1.5, 0, 1)


class TestCheckChoice:
    @pytest.mark.parametrize("choices", [("a", "b"), ["a", "b"], {"a", "b"}, frozenset("ab")])
    def test_accepts_a_member_of_any_collection(self, choices):
        check_choice("p", "a", choices)

    @pytest.mark.parametrize("value", ["c", "A", "", None, 1, ["a"]])
    def test_rejects_non_members(self, value):
        with pytest.raises(InvalidParameterError):
            check_choice("p", value, ("a", "b"))

    def test_message_lists_the_options_and_the_value(self):
        with pytest.raises(InvalidParameterError, match=r"weights.*'uniform', 'distance'.*'x'"):
            check_choice("weights", "x", ("uniform", "distance"))


class TestExceptionType:
    @pytest.mark.parametrize(
        "call",
        [
            lambda: check_positive("p", 0),
            lambda: check_non_negative("p", -1),
            lambda: check_at_least("p", 0, 1),
            lambda: check_in_range("p", 2, 0, 1),
            lambda: check_choice("p", "x", ("a",)),
        ],
        ids=["positive", "non_negative", "at_least", "in_range", "choice"],
    )
    def test_every_check_raises_invalid_parameter_error(self, call):
        with pytest.raises(InvalidParameterError) as error:
            call()

        assert isinstance(error.value, ValidationError)


class TestDistanceMetrics:
    def test_lists_exactly_the_supported_metrics(self):
        assert set(DISTANCE_METRICS) == {"euclidean", "cityblock", "chebyshev"}

    def test_is_an_immutable_tuple_without_duplicates(self):
        assert isinstance(DISTANCE_METRICS, tuple)
        assert len(set(DISTANCE_METRICS)) == len(DISTANCE_METRICS)

    @pytest.mark.parametrize("metric", DISTANCE_METRICS)
    def test_every_metric_is_understood_by_scipy(self, metric):
        point = np.zeros((1, 2))

        assert cdist(point, point, metric=metric).shape == (1, 1)
