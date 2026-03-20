import numpy as np
import pytest
from ml_collection.supervised.classification.knn import KNNClassifier
from ml_collection.exception import NotFitted


def test_init():
    knn = KNNClassifier(k=3)

    assert hasattr(knn, '_KNNClassifier__fitted')
    assert getattr(knn, '_KNNClassifier__fitted') == False


def test_fit():
    X = np.array([[1, 2], [1, 2]])
    y = np.array([0, 1])

    knn = KNNClassifier(k=3)
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

    knn = KNNClassifier(k=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    assert y_pred.shape[0] == X_test.shape[0]


def test_notfitted():
    X_test = np.array([[1,2]])

    knn = KNNClassifier(k=3)
    with pytest.raises(NotFitted):
        knn.predict(X_test)