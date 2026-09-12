"""Tests for GaussianNB."""

import numpy as np
from scipy import stats

from pyml.naive_bayes import GaussianNB


class TestFit:
    def test_computes_correct_per_class_mean_and_variance(self, rng):
        X0 = rng.normal(loc=0.0, scale=1.0, size=(50, 2))
        X1 = rng.normal(loc=5.0, scale=2.0, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = GaussianNB().fit(X, y)

        mean_0, var_0 = model._class_params[0]
        mean_1, var_1 = model._class_params[1]

        assert np.allclose(mean_0, np.mean(X0, axis=0))
        assert np.allclose(var_0, np.var(X0, axis=0))
        assert np.allclose(mean_1, np.mean(X1, axis=0))
        assert np.allclose(var_1, np.var(X1, axis=0))

    def test_computes_correct_log_prior(self, rng):
        X = rng.normal(size=(100, 2))
        y = np.array([0] * 30 + [1] * 70)

        model = GaussianNB().fit(X, y)

        assert np.allclose(model.class_log_prior_, np.log(np.array([0.3, 0.7])))


class TestLogLikelihood:
    def test_matches_scipy_norm_logpdf(self, rng):
        X0 = rng.normal(loc=0.0, scale=1.0, size=(30, 3))
        X1 = rng.normal(loc=4.0, scale=1.5, size=(30, 3))
        X = np.vstack([X0, X1])
        y = np.array([0] * 30 + [1] * 30)

        model = GaussianNB().fit(X, y)
        X_query = rng.normal(size=(5, 3))
        log_likelihood = model._log_likelihood(X_query)

        for class_idx in range(2):
            mean, var = model._class_params[class_idx]
            std = np.sqrt(var + 1e-10)
            expected = stats.norm.logpdf(X_query, loc=mean, scale=std).sum(axis=1)
            assert np.allclose(log_likelihood[:, class_idx], expected, rtol=1e-6)

    def test_var_smoothing_prevents_division_by_zero(self):
        X = np.column_stack(
            [
                np.random.default_rng(0).normal(size=60),
                np.array([1.0] * 30 + [2.0] * 30),
            ]
        )
        y = np.array([0] * 30 + [1] * 30)

        model = GaussianNB().fit(X, y)
        assert np.all(np.isfinite(model._log_likelihood(X)))


class TestPredict:
    def test_separates_well_separated_clusters(self, rng):
        X0 = rng.normal(loc=-5.0, scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=5.0, scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = GaussianNB().fit(X, y)
        assert np.array_equal(model.predict(X), y)

    def test_output_shape(self, rng):
        X = rng.normal(size=(40, 3))
        y = np.array([0, 1] * 20)

        model = GaussianNB().fit(X, y)
        assert model.predict(rng.normal(size=(7, 3))).shape == (7,)

    def test_supports_more_than_two_classes(self, rng):
        X0 = rng.normal(loc=-6.0, scale=0.5, size=(30, 2))
        X1 = rng.normal(loc=0.0, scale=0.5, size=(30, 2))
        X2 = rng.normal(loc=6.0, scale=0.5, size=(30, 2))
        X = np.vstack([X0, X1, X2])
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)

        model = GaussianNB().fit(X, y)
        predictions = model.predict(X)

        assert np.array_equal(predictions, y)
        assert set(model.classes_) == {0, 1, 2}


class TestScore:
    def test_returns_accuracy(self, rng):
        X0 = rng.normal(loc=-5.0, scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=5.0, scale=0.5, size=(50, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = GaussianNB().fit(X, y)
        assert model.score(X, y) == 1.0
