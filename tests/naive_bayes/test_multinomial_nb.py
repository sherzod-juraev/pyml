"""Tests for MultinomialNB."""

import numpy as np

from pyml.naive_bayes import MultinomialNB


class TestFit:
    def test_computes_correct_log_feature_prob(self, rng):
        X0 = rng.poisson(lam=[8, 1], size=(50, 2)).astype(np.float64)
        X1 = rng.poisson(lam=[1, 8], size=(50, 2)).astype(np.float64)
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = MultinomialNB(alpha=1.0).fit(X, y)

        counts_0 = X0.sum(axis=0)
        expected_0 = np.log((counts_0 + 1.0) / (counts_0.sum() + 1.0 * 2))
        assert np.allclose(model._class_params[0], expected_0)

    def test_computes_correct_log_prior(self, rng):
        X = rng.poisson(lam=3, size=(100, 2)).astype(np.float64)
        y = np.array([0] * 30 + [1] * 70)

        model = MultinomialNB().fit(X, y)

        assert np.allclose(model.class_log_prior_, np.log(np.array([0.3, 0.7])))

    def test_higher_alpha_shrinks_probability_gap_between_features(self, rng):
        X = np.array([[10, 0]] * 20).astype(np.float64)
        y = np.array([0] * 20)

        low_alpha = MultinomialNB(alpha=0.01).fit(X, y)
        high_alpha = MultinomialNB(alpha=10.0).fit(X, y)

        gap_low = low_alpha._class_params[0][0] - low_alpha._class_params[0][1]
        gap_high = high_alpha._class_params[0][0] - high_alpha._class_params[0][1]

        assert gap_high < gap_low


class TestLogLikelihood:
    def test_matches_manual_weighted_sum(self, rng):
        X0 = rng.poisson(lam=[8, 1], size=(30, 2)).astype(np.float64)
        X1 = rng.poisson(lam=[1, 8], size=(30, 2)).astype(np.float64)
        X = np.vstack([X0, X1])
        y = np.array([0] * 30 + [1] * 30)

        model = MultinomialNB().fit(X, y)
        X_query = rng.poisson(lam=4, size=(5, 2)).astype(np.float64)
        log_likelihood = model._log_likelihood(X_query)

        for class_idx in range(2):
            log_probs = model._class_params[class_idx]
            expected = X_query @ log_probs
            assert np.allclose(log_likelihood[:, class_idx], expected)

    def test_zero_alpha_gives_negative_infinity_for_unseen_feature(self, rng):
        X = np.array([[5, 0]] * 20).astype(np.float64)
        y = np.array([0] * 20)

        model = MultinomialNB(alpha=0.0).fit(X, y)

        assert np.isneginf(model._class_params[0][1])


class TestPredict:
    def test_separates_well_separated_classes(self, rng):
        X0 = rng.poisson(lam=[10, 1], size=(50, 2)).astype(np.float64)
        X1 = rng.poisson(lam=[1, 10], size=(50, 2)).astype(np.float64)
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = MultinomialNB().fit(X, y)
        assert np.array_equal(model.predict(X), y)

    def test_output_shape(self, rng):
        X = rng.poisson(lam=3, size=(40, 3)).astype(np.float64)
        y = np.array([0, 1] * 20)

        model = MultinomialNB().fit(X, y)
        assert model.predict(rng.poisson(lam=3, size=(7, 3)).astype(np.float64)).shape == (7,)

    def test_supports_more_than_two_classes(self, rng):
        X0 = rng.poisson(lam=[30, 1, 1], size=(30, 3)).astype(np.float64)
        X1 = rng.poisson(lam=[1, 30, 1], size=(30, 3)).astype(np.float64)
        X2 = rng.poisson(lam=[1, 1, 30], size=(30, 3)).astype(np.float64)
        X = np.vstack([X0, X1, X2])
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)

        model = MultinomialNB().fit(X, y)
        predictions = model.predict(X)

        assert np.array_equal(predictions, y)
        assert set(model.classes_) == {0, 1, 2}


class TestScore:
    def test_returns_accuracy(self, rng):
        X0 = rng.poisson(lam=[10, 1], size=(50, 2)).astype(np.float64)
        X1 = rng.poisson(lam=[1, 10], size=(50, 2)).astype(np.float64)
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)

        model = MultinomialNB().fit(X, y)
        assert model.score(X, y) == 1.0
