import numpy as np
import pytest

from pyml.exc import NotFittedError
from pyml import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)


class TestMinMaxScaler:
    def test_fit_computes_min_and_max(self):
        X = np.array([
            [1.0, 5.0],
            [3.0, 7.0],
            [2.0, 6.0],
        ])

        scaler = MinMaxScaler().fit(X)

        assert np.allclose(scaler.min_, [1.0, 5.0])
        assert np.allclose(scaler.max_, [3.0, 7.0])

    def test_transform_matches_formula(self):
        X = np.array([
            [0.0, 10.0],
            [10.0, 20.0],
        ])

        scaler = MinMaxScaler().fit(X)

        X_test = np.array([[5.0, 15.0]])
        expected = np.array([[0.5, 0.5]])

        assert np.allclose(
            scaler.transform(X_test),
            expected,
        )

    def test_inverse_transform(self):
        rng = np.random.default_rng(0)

        X = rng.normal(size=(40, 4))

        scaler = MinMaxScaler().fit(X)

        recovered = scaler.inverse_transform(
            scaler.transform(X)
        )

        assert np.allclose(recovered, X)

    def test_fit_transform_equals_fit_then_transform(self):
        rng = np.random.default_rng(1)

        X = rng.normal(size=(50, 3))

        scaler1 = MinMaxScaler()
        out1 = scaler1.fit_transform(X)

        scaler2 = MinMaxScaler()
        scaler2.fit(X)
        out2 = scaler2.transform(X)

        assert np.allclose(out1, out2)

    def test_constant_feature_becomes_zero(self):
        X = np.array([
            [5.0, 1.0],
            [5.0, 2.0],
            [5.0, 3.0],
        ])

        scaler = MinMaxScaler().fit(X)

        Xt = scaler.transform(X)

        assert np.allclose(Xt[:, 0], 0.0)

    def test_transform_before_fit_raises(self):
        scaler = MinMaxScaler()

        with pytest.raises(NotFittedError):
            scaler.transform(np.ones((2, 2)))

    def test_single_sample(self):
        X = np.array([[5.0, 10.0]])

        scaler = MinMaxScaler().fit(X)

        Xt = scaler.transform(X)

        assert Xt.shape == X.shape
        assert np.allclose(Xt, np.zeros_like(X))

    def test_single_feature(self):
        X = np.array([
            [1.0],
            [2.0],
            [3.0],
        ])

        scaler = MinMaxScaler().fit(X)

        Xt = scaler.transform(X)

        expected = np.array([
            [0.0],
            [0.5],
            [1.0],
        ])

        assert np.allclose(Xt, expected)


class TestStandardScaler:
    def test_fit_computes_mean_and_std(self):
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])

        scaler = StandardScaler().fit(X)

        assert np.allclose(scaler.mean_, np.mean(X, axis=0))
        assert np.allclose(scaler.std_, np.std(X, axis=0))

    def test_transform_matches_formula(self):
        X = np.array([
            [1.0],
            [2.0],
            [3.0],
        ])

        scaler = StandardScaler().fit(X)

        expected = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        assert np.allclose(
            scaler.transform(X),
            expected,
        )

    def test_transformed_mean_is_zero(self):
        rng = np.random.default_rng(0)

        X = rng.normal(size=(200, 5))

        scaler = StandardScaler()

        Xt = scaler.fit_transform(X)

        assert np.allclose(
            Xt.mean(axis=0),
            np.zeros(5),
            atol=1e-10,
        )

    def test_transformed_std_is_one(self):
        rng = np.random.default_rng(1)

        X = rng.normal(size=(200, 5))

        scaler = StandardScaler()

        Xt = scaler.fit_transform(X)

        assert np.allclose(
            Xt.std(axis=0),
            np.ones(5),
            atol=1e-10,
        )

    def test_inverse_transform(self):
        rng = np.random.default_rng(2)

        X = rng.normal(size=(50, 4))

        scaler = StandardScaler().fit(X)

        recovered = scaler.inverse_transform(
            scaler.transform(X)
        )

        assert np.allclose(recovered, X)

    def test_fit_transform_equals_fit_then_transform(self):
        rng = np.random.default_rng(3)

        X = rng.normal(size=(40, 3))

        scaler1 = StandardScaler()
        out1 = scaler1.fit_transform(X)

        scaler2 = StandardScaler()
        scaler2.fit(X)
        out2 = scaler2.transform(X)

        assert np.allclose(out1, out2)

    def test_constant_feature_becomes_zero(self):
        X = np.array([
            [5.0, 1.0],
            [5.0, 2.0],
            [5.0, 3.0],
        ])

        scaler = StandardScaler().fit(X)

        Xt = scaler.transform(X)

        assert np.allclose(Xt[:, 0], 0.0)

    def test_transform_before_fit_raises(self):
        scaler = StandardScaler()

        with pytest.raises(NotFittedError):
            scaler.transform(np.ones((2, 2)))

    def test_single_sample(self):
        X = np.array([[10.0, 20.0]])

        scaler = StandardScaler().fit(X)

        Xt = scaler.transform(X)

        assert Xt.shape == X.shape
        assert np.allclose(Xt, np.zeros_like(X))

    def test_single_feature(self):
        X = np.array([
            [1.0],
            [2.0],
            [3.0],
        ])

        scaler = StandardScaler().fit(X)

        Xt = scaler.transform(X)

        assert Xt.shape == X.shape
        assert np.allclose(np.mean(Xt), 0.0, atol=1e-10)
        assert np.allclose(np.std(Xt), 1.0, atol=1e-10)


class TestRobustScaler:
    def test_fit_computes_median_and_iqr(self):
        X = np.array([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [100.0],
        ])

        scaler = RobustScaler().fit(X)

        expected_median = np.median(X, axis=0)
        expected_iqr = (
            np.percentile(X, 75, axis=0)
            - np.percentile(X, 25, axis=0)
        )

        assert np.allclose(scaler.median_, expected_median)
        assert np.allclose(scaler.iqr_, expected_iqr)

    def test_transform_matches_formula(self):
        X = np.array([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ])

        scaler = RobustScaler().fit(X)

        expected = (
            X - np.median(X, axis=0)
        ) / (
            np.percentile(X, 75, axis=0)
            - np.percentile(X, 25, axis=0)
        )

        assert np.allclose(
            scaler.transform(X),
            expected,
        )

    def test_transformed_median_is_zero(self):
        rng = np.random.default_rng(0)

        X = rng.normal(size=(200, 4))

        scaler = RobustScaler()

        Xt = scaler.fit_transform(X)

        assert np.allclose(
            np.median(Xt, axis=0),
            np.zeros(4),
            atol=1e-10,
        )

    def test_inverse_transform(self):
        rng = np.random.default_rng(1)

        X = rng.normal(size=(60, 5))

        scaler = RobustScaler().fit(X)

        recovered = scaler.inverse_transform(
            scaler.transform(X)
        )

        assert np.allclose(recovered, X)

    def test_fit_transform_equals_fit_then_transform(self):
        rng = np.random.default_rng(2)

        X = rng.normal(size=(40, 3))

        scaler1 = RobustScaler()
        out1 = scaler1.fit_transform(X)

        scaler2 = RobustScaler()
        scaler2.fit(X)
        out2 = scaler2.transform(X)

        assert np.allclose(out1, out2)

    def test_constant_feature_becomes_zero(self):
        X = np.array([
            [5.0, 1.0],
            [5.0, 2.0],
            [5.0, 3.0],
        ])

        scaler = RobustScaler().fit(X)

        Xt = scaler.transform(X)

        assert np.allclose(Xt[:, 0], 0.0)

    def test_transform_before_fit_raises(self):
        scaler = RobustScaler()

        with pytest.raises(NotFittedError):
            scaler.transform(np.ones((2, 2)))

    def test_single_sample(self):
        X = np.array([[5.0, 10.0]])

        scaler = RobustScaler().fit(X)

        Xt = scaler.transform(X)

        assert Xt.shape == X.shape
        assert np.allclose(Xt, np.zeros_like(X))

    def test_single_feature(self):
        X = np.array([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ])

        scaler = RobustScaler().fit(X)

        Xt = scaler.transform(X)

        assert Xt.shape == X.shape
        assert np.allclose(np.median(Xt), 0.0, atol=1e-10)


class TestScalerEdgeCases:
    @pytest.mark.parametrize(
        "Scaler",
        [MinMaxScaler, StandardScaler, RobustScaler],
    )
    def test_integer_input_returns_float_output(self, Scaler):
        X = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
        ], dtype=int)

        scaler = Scaler()

        Xt = scaler.fit_transform(X)

        assert np.issubdtype(Xt.dtype, np.floating)

    @pytest.mark.parametrize(
        "Scaler",
        [MinMaxScaler, StandardScaler, RobustScaler],
    )
    def test_output_shape_matches_input(self, Scaler):
        rng = np.random.default_rng(0)

        X = rng.normal(size=(50, 7))

        scaler = Scaler()

        Xt = scaler.fit_transform(X)

        assert Xt.shape == X.shape

    @pytest.mark.parametrize(
        "Scaler",
        [MinMaxScaler, StandardScaler, RobustScaler],
    )
    def test_inverse_transform_recovers_original(self, Scaler):
        rng = np.random.default_rng(1)

        X = rng.normal(size=(100, 4))

        scaler = Scaler().fit(X)

        recovered = scaler.inverse_transform(
            scaler.transform(X)
        )

        assert np.allclose(recovered, X)

    @pytest.mark.parametrize(
        "Scaler",
        [MinMaxScaler, StandardScaler, RobustScaler],
    )
    def test_fit_transform_equals_fit_then_transform(self, Scaler):
        rng = np.random.default_rng(2)

        X = rng.normal(size=(60, 3))

        scaler1 = Scaler()
        Xt1 = scaler1.fit_transform(X)

        scaler2 = Scaler()
        scaler2.fit(X)
        Xt2 = scaler2.transform(X)

        assert np.allclose(Xt1, Xt2)

    @pytest.mark.parametrize(
        "Scaler",
        [MinMaxScaler, StandardScaler, RobustScaler],
    )
    def test_large_values(self, Scaler):
        rng = np.random.default_rng(3)

        X = rng.uniform(
            1e8,
            1e9,
            size=(100, 5),
        )

        scaler = Scaler()

        Xt = scaler.fit_transform(X)

        assert np.isfinite(Xt).all()

    @pytest.mark.parametrize(
        "Scaler",
        [MinMaxScaler, StandardScaler, RobustScaler],
    )
    def test_negative_values(self, Scaler):
        rng = np.random.default_rng(4)

        X = rng.uniform(
            -100,
            100,
            size=(80, 4),
        )

        scaler = Scaler()

        Xt = scaler.fit_transform(X)

        assert np.isfinite(Xt).all()

    @pytest.mark.parametrize(
        "Scaler",
        [MinMaxScaler, StandardScaler, RobustScaler],
    )
    def test_transform_before_fit_raises(self, Scaler):
        scaler = Scaler()

        with pytest.raises(NotFittedError):
            scaler.transform(np.ones((5, 2)))
