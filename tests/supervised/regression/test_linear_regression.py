import numpy as np
import pytest
from ml_collection.supervised.regression import LinearRegression
from ml_collection.exception import NotFitted
from sklearn.datasets import make_regression


def test_init():
    lr = LinearRegression()

    assert hasattr(lr, 'eta')
    assert hasattr(lr, 'max_iter')
    assert hasattr(lr, '_LinearRegression__fitted')
    assert getattr(lr, '_LinearRegression__fitted') == False


def test_fit():
    X, y = make_regression(
        n_samples=200,
        random_state=42
    )

    lr = LinearRegression()
    res = lr.fit(X, y)

    assert hasattr(lr, 'coef')
    assert hasattr(lr, 'intercept')
    assert isinstance(lr.coef, np.ndarray)
    assert lr.coef.shape[0] == X.shape[1]
    assert isinstance(lr.intercept, np.number)
    assert isinstance(res, LinearRegression)
    assert getattr(lr, '_LinearRegression__fitted') == True


def test_predict():
    X, y = make_regression(
        n_samples=200,
        random_state=42
    )

    lr = LinearRegression()
    X_train, X_test = X[:150, :], X[150:, :]
    y_train, y_test = y[:150], y[150:]
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    assert y_pred.shape == y_test.shape


def test_notfitted():
    X, y = make_regression(
        n_samples=200,
        random_state=42
    )

    lr = LinearRegression()

    with pytest.raises(NotFitted):
        lr.predict(X)