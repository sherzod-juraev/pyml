"""Tests for StandardScaler."""

import numpy as np

from pyml.preprocessing import StandardScaler


class TestFit:
    def test_computes_mean_and_std(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        scaler = StandardScaler().fit(X)

        assert np.allclose(scaler.mean_, [2.0, 20.0])
        assert np.allclose(scaler.std_, np.std(X, axis=0))


class TestTransform:
    def test_result_has_zero_mean_and_unit_std(self, rng):
        X = rng.normal(loc=50, scale=10, size=(100, 3))

        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        assert np.allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-8)
        assert np.allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-8)

    def test_constant_feature_does_not_raise(self):
        X = np.array([[5.0], [5.0], [5.0]])

        scaler = StandardScaler().fit(X)
        result = scaler.transform(X)

        assert np.all(np.isfinite(result))


class TestInverseTransform:
    def test_reverses_transform(self, rng):
        X = rng.normal(loc=50, scale=10, size=(50, 3))

        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        X_restored = scaler.inverse_transform(X_scaled)

        assert np.allclose(X_restored, X)


class TestFitTransform:
    def test_matches_fit_then_transform(self, rng):
        X = rng.normal(loc=50, scale=10, size=(30, 2))

        combined = StandardScaler().fit_transform(X)
        separate = StandardScaler().fit(X).transform(X)

        assert np.allclose(combined, separate)
