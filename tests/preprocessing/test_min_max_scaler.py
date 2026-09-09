"""Tests for MinMaxScaler."""

import numpy as np

from pyml.preprocessing import MinMaxScaler


class TestFit:
    def test_computes_min_and_range(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        scaler = MinMaxScaler().fit(X)

        assert np.allclose(scaler.min_, [1.0, 10.0])
        assert np.allclose(scaler.range_, [2.0, 20.0])


class TestTransform:
    def test_result_is_bounded_zero_to_one(self, rng):
        X = rng.uniform(0, 100, size=(100, 3))

        scaler = MinMaxScaler().fit(X)
        X_scaled = scaler.transform(X)

        assert np.all(X_scaled >= 0.0)
        assert np.all(X_scaled <= 1.0)

    def test_min_maps_to_zero_and_max_maps_to_one(self):
        X = np.array([[1.0], [5.0], [10.0]])

        scaler = MinMaxScaler().fit(X)
        X_scaled = scaler.transform(X)

        assert np.allclose(X_scaled[0, 0], 0.0)
        assert np.allclose(X_scaled[-1, 0], 1.0)

    def test_constant_feature_does_not_raise(self):
        X = np.array([[5.0], [5.0], [5.0]])

        scaler = MinMaxScaler().fit(X)
        result = scaler.transform(X)

        assert np.all(np.isfinite(result))


class TestInverseTransform:
    def test_reverses_transform(self, rng):
        X = rng.uniform(0, 100, size=(50, 3))

        scaler = MinMaxScaler().fit(X)
        X_scaled = scaler.transform(X)
        X_restored = scaler.inverse_transform(X_scaled)

        assert np.allclose(X_restored, X)


class TestFitTransform:
    def test_matches_fit_then_transform(self, rng):
        X = rng.uniform(0, 100, size=(30, 2))

        combined = MinMaxScaler().fit_transform(X)
        separate = MinMaxScaler().fit(X).transform(X)

        assert np.allclose(combined, separate)
