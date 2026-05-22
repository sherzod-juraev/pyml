import numpy as np
import pytest
from sklearn.preprocessing import (
    MinMaxScaler as SkMinMaxScaler,
)
from sklearn.preprocessing import (
    RobustScaler as SkRobustScaler,
)
from sklearn.preprocessing import (
    StandardScaler as SkStandardScaler,
)

from pyml.preprocess import MinMaxScaler, RobustScaler, StandardScaler


class TestMinMaxScaler:
    """Compare pyml MinMaxScaler against sklearn."""

    def test_against_sklearn(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-10, 10, size=(100, 5)).astype(np.float64)
        X_test = rng.uniform(-10, 10, size=(50, 5)).astype(np.float64)

        sk = SkMinMaxScaler()
        my = MinMaxScaler()

        sk.fit(X)
        my.fit(X)

        assert np.allclose(sk.data_min_, my.min_), "min_ mismatch"
        assert np.allclose(sk.data_max_, my.max_), "max_ mismatch"

        sk_transformed = sk.transform(X_test)
        my_transformed = my.transform(X_test)

        assert np.allclose(sk_transformed, my_transformed, rtol=1e-10)

    def test_inverse_transform(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(-5, 5, size=(50, 3)).astype(np.float64)
        X_test = rng.uniform(-5, 5, size=(20, 3)).astype(np.float64)

        my = MinMaxScaler().fit(X)
        transformed = my.transform(X_test)
        original = my.inverse_transform(transformed)

        assert np.allclose(original, X_test, rtol=1e-10)

    def test_fit_transform(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(-3, 3, size=(30, 4)).astype(np.float64)

        my = MinMaxScaler()
        X_scaled = my.fit_transform(X)

        # fit_transform should produce same result as fit + transform
        my2 = MinMaxScaler()
        my2.fit(X)
        assert np.allclose(X_scaled, my2.transform(X), rtol=1e-10)

    def test_constant_feature(self):
        X = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]], dtype=np.float64)

        sk = SkMinMaxScaler().fit(X)
        my = MinMaxScaler().fit(X)

        assert np.allclose(sk.transform(X), my.transform(X), rtol=1e-10)

    def test_not_fitted_raises(self):
        scaler = MinMaxScaler()
        X = np.random.randn(10, 3)
        with pytest.raises(Exception):
            scaler.transform(X)


class TestStandardScaler:
    """Compare pyml StandardScaler against sklearn."""

    def test_against_sklearn(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-10, 10, size=(100, 5)).astype(np.float64)
        X_test = rng.uniform(-10, 10, size=(50, 5)).astype(np.float64)

        sk = SkStandardScaler()
        my = StandardScaler()

        sk.fit(X)
        my.fit(X)

        assert np.allclose(sk.mean_, my.mean_), "mean_ mismatch"
        assert np.allclose(sk.scale_, my.std_), "std_ mismatch"

        sk_transformed = sk.transform(X_test)
        my_transformed = my.transform(X_test)

        assert np.allclose(sk_transformed, my_transformed, rtol=1e-10)

    def test_inverse_transform(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(-5, 5, size=(50, 3)).astype(np.float64)
        X_test = rng.uniform(-5, 5, size=(20, 3)).astype(np.float64)

        my = StandardScaler().fit(X)
        transformed = my.transform(X_test)
        original = my.inverse_transform(transformed)

        assert np.allclose(original, X_test, rtol=1e-10)

    def test_fit_transform(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(-3, 3, size=(30, 4)).astype(np.float64)

        my = StandardScaler()
        X_scaled = my.fit_transform(X)

        my2 = StandardScaler().fit(X)
        assert np.allclose(X_scaled, my2.transform(X), rtol=1e-10)

    def test_constant_feature(self):
        X = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]], dtype=np.float64)

        sk = SkStandardScaler().fit(X)
        my = StandardScaler().fit(X)

        assert np.allclose(sk.transform(X), my.transform(X), rtol=1e-10)

    def test_not_fitted_raises(self):
        scaler = StandardScaler()
        X = np.random.randn(10, 3)
        with pytest.raises(Exception):
            scaler.transform(X)


class TestRobustScaler:
    """Compare pyml RobustScaler against sklearn."""

    def test_against_sklearn(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-10, 10, size=(100, 5)).astype(np.float64)
        X_test = rng.uniform(-10, 10, size=(50, 5)).astype(np.float64)

        sk = SkRobustScaler()
        my = RobustScaler()

        sk.fit(X)
        my.fit(X)

        assert np.allclose(sk.center_, my.median_), "median_ mismatch"
        assert np.allclose(sk.scale_, my.iqr_), "iqr_ mismatch"

        sk_transformed = sk.transform(X_test)
        my_transformed = my.transform(X_test)

        assert np.allclose(sk_transformed, my_transformed, rtol=1e-10)

    def test_inverse_transform(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(-5, 5, size=(50, 3)).astype(np.float64)
        X_test = rng.uniform(-5, 5, size=(20, 3)).astype(np.float64)

        my = RobustScaler().fit(X)
        transformed = my.transform(X_test)
        original = my.inverse_transform(transformed)

        assert np.allclose(original, X_test, rtol=1e-10)

    def test_fit_transform(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(-3, 3, size=(30, 4)).astype(np.float64)

        my = RobustScaler()
        X_scaled = my.fit_transform(X)

        my2 = RobustScaler().fit(X)
        assert np.allclose(X_scaled, my2.transform(X), rtol=1e-10)

    def test_constant_feature(self):
        X = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]], dtype=np.float64)

        sk = SkRobustScaler().fit(X)
        my = RobustScaler().fit(X)

        assert np.allclose(sk.transform(X), my.transform(X), rtol=1e-10)

    def test_not_fitted_raises(self):
        scaler = RobustScaler()
        X = np.random.randn(10, 3)
        with pytest.raises(Exception):
            scaler.transform(X)


class TestScalersEdgeCases:
    """Edge cases for all scalers."""

    def test_single_sample(self):
        X = np.array([[1.0, 2.0]], dtype=np.float64)

        for MyScaler, SkScaler in [
            (MinMaxScaler, SkMinMaxScaler),
            (StandardScaler, SkStandardScaler),
            (RobustScaler, SkRobustScaler),
        ]:
            sk = SkScaler().fit(X)
            my = MyScaler().fit(X)
            assert np.allclose(sk.transform(X), my.transform(X), rtol=1e-10)

    def test_single_feature(self):
        X = np.random.randn(50, 1).astype(np.float64)

        for MyScaler, SkScaler in [
            (MinMaxScaler, SkMinMaxScaler),
            (StandardScaler, SkStandardScaler),
            (RobustScaler, SkRobustScaler),
        ]:
            sk = SkScaler().fit(X)
            my = MyScaler().fit(X)
            assert np.allclose(sk.transform(X), my.transform(X), rtol=1e-10)

    def test_large_values(self):
        X = np.random.uniform(1e6, 1e7, size=(50, 3)).astype(np.float64)

        for MyScaler, SkScaler in [
            (MinMaxScaler, SkMinMaxScaler),
            (StandardScaler, SkStandardScaler),
            (RobustScaler, SkRobustScaler),
        ]:
            sk = SkScaler().fit(X)
            my = MyScaler().fit(X)
            assert np.allclose(sk.transform(X), my.transform(X), rtol=1e-5)
