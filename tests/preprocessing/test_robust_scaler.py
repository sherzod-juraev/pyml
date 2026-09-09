"""Tests for RobustScaler."""

import numpy as np

from pyml.preprocessing import RobustScaler


class TestFit:
    def test_computes_median_and_iqr(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        scaler = RobustScaler().fit(X)

        assert np.allclose(scaler.q2_, [3.0])
        assert np.allclose(scaler.iqr_, [np.percentile(X, 75) - np.percentile(X, 25)])


class TestTransform:
    def test_result_has_zero_median(self, rng):
        X = rng.normal(loc=50, scale=10, size=(100, 3))

        scaler = RobustScaler().fit(X)
        X_scaled = scaler.transform(X)

        assert np.allclose(np.median(X_scaled, axis=0), 0.0, atol=1e-8)

    def test_less_affected_by_outliers_than_standard_scaling(self, rng):
        X = rng.normal(loc=50, scale=10, size=(100, 1))
        X_with_outlier = X.copy()
        X_with_outlier[0] = 10000.0

        scaler = RobustScaler().fit(X)
        scaler_with_outlier = RobustScaler().fit(X_with_outlier)

        # median and IQR should barely shift despite one extreme outlier
        assert np.allclose(scaler.q2_, scaler_with_outlier.q2_, atol=1.0)

    def test_constant_feature_does_not_raise(self):
        X = np.array([[5.0], [5.0], [5.0], [5.0], [5.0]])

        scaler = RobustScaler().fit(X)
        result = scaler.transform(X)

        assert np.all(np.isfinite(result))


class TestInverseTransform:
    def test_reverses_transform(self, rng):
        X = rng.normal(loc=50, scale=10, size=(50, 3))

        scaler = RobustScaler().fit(X)
        X_scaled = scaler.transform(X)
        X_restored = scaler.inverse_transform(X_scaled)

        assert np.allclose(X_restored, X)


class TestFitTransform:
    def test_matches_fit_then_transform(self, rng):
        X = rng.normal(loc=50, scale=10, size=(30, 2))

        combined = RobustScaler().fit_transform(X)
        separate = RobustScaler().fit(X).transform(X)

        assert np.allclose(combined, separate)
