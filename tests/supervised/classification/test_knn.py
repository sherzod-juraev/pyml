import numpy as np
import pytest
from ml_collection.supervised.classification.knn import KNNClassifier
from ml_collection.exception import NotFitted


def test_init():
    knn = KNNClassifier()

    assert hasattr(knn, '_KNNClassifier__fitted')
    assert getattr(knn, '_KNNClassifier__fitted') == False


def test_fit():
    X = np.array([[1, 2], [1, 2]])
    y = np.array([0, 1])

    knn = KNNClassifier()
    knn.fit(X, y)

    assert hasattr(knn, 'X_train')
    assert hasattr(knn, 'y_train')
    assert np.array_equal(X, knn.X_train)
    assert np.array_equal(y, knn.y_train)
    assert getattr(knn, '_KNNClassifier__fitted') == True


def test_predict():
    X_train = np.array([[1,2],[2,3],[3,4]])
    y_train = np.array([0,1,0])
    X_test = np.array([[1,2],[3,4]])

    knn = KNNClassifier(weighting='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred.ndim == 1

    knn = KNNClassifier(weighting='distance')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred.ndim == 1

    knn = KNNClassifier(metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred.ndim == 1

    knn = KNNClassifier(metric='chebyshev')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred.ndim == 1

    knn = KNNClassifier(metric='cosine')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred.ndim == 1

    knn = KNNClassifier(metric='cityblock')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    assert y_pred.shape[0] == X_test.shape[0]
    assert y_pred.ndim == 1


def test_notfitted():
    X_test = np.array([[1,2]])

    knn = KNNClassifier()
    with pytest.raises(NotFitted):
        knn.predict(X_test)