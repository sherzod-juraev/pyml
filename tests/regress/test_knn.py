import numpy as np
import pytest
from mlkit import KNNRegression
from mlkit.exc import NotFitted



def test_init():

    knn = KNNRegression()

    assert hasattr(knn, '_KNNRegression__fitted')
    assert getattr(knn, '_KNNRegression__fitted') == False
    assert hasattr(knn, 'metric')
    assert hasattr(knn, 'weighting')


def test_fit():

    knn = KNNRegression()
    rng = np.random.default_rng(42)
    X = rng.integers(1, 20, size=(5,2))
    y = rng.integers(1, 9, size=5)
    result = knn.fit(X, y)

    assert hasattr(knn, 'X_')
    assert isinstance(getattr(knn, 'X_'), np.ndarray)
    assert hasattr(knn, 'y_')
    assert isinstance(getattr(knn, 'y_'), np.ndarray)
    assert getattr(knn, '_KNNRegression__fitted') == True
    assert isinstance(result, KNNRegression)


def test_predict():

    knn = KNNRegression(weighting='distance')
    rng = np.random.default_rng(42)
    X_ = rng.integers(1, 20, size=(20,2))
    X_test = rng.integers(1, 20, size=(5,2))
    y_ = rng.integers(1, 9, size=20)
    knn.fit(X_, y_)
    y_pred = knn.predict(X_test)

    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred.ndim == 1

    knn = KNNRegression(weighting='uniform')
    knn.fit(X_, y_)
    y_pred = knn.predict(X_test)

    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred.ndim == 1

    knn = KNNRegression(metric='euclidean')
    knn.fit(X_, y_)
    y_pred = knn.predict(X_test)

    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred.ndim == 1

    knn = KNNRegression(metric='chebyshev')
    knn.fit(X_, y_)
    y_pred = knn.predict(X_test)

    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred.ndim == 1

    knn = KNNRegression(metric='cityblock')
    knn.fit(X_, y_)
    y_pred = knn.predict(X_test)

    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred.ndim == 1


def test_notfitted():

    knn = KNNRegression()
    rng = np.random.default_rng(42)
    X_test = rng.integers(1, 20, size=(5, 2))
    with pytest.raises(NotFitted):
        knn.predict(X_test)