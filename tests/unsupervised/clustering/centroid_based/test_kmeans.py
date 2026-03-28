import numpy as np
import pytest
from ml_collection.unsupervised.clustering.centroid_based.kmeans import Kmeans
from ml_collection.exception import NotFitted
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def test_init():
    kmeans = Kmeans()

    assert hasattr(kmeans, 'k')
    assert hasattr(kmeans, 'max_iter')
    assert hasattr(kmeans, 'tol')
    assert hasattr(kmeans, 'init')
    assert hasattr(kmeans, 'centroids')
    assert hasattr(kmeans, 'metric')
    assert hasattr(kmeans, 'random_state')
    assert hasattr(kmeans, '_Kmeans__fitted')
    assert getattr(kmeans, '_Kmeans__fitted') == False


def test_functions():
    X, y = make_blobs(
        n_samples=1000,
        centers=3,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25
    )
    def test_fit():
        kmeans = Kmeans(
            k=3,
            init='uniform',
            metric='euclidean',
            random_state=42
        )
        kmeans.fit(X_train)

        assert isinstance(kmeans.centroids, np.ndarray)
        assert kmeans.centroids.shape[0] == 3
        assert kmeans.centroids.shape[1] == X_train.shape[1]
        assert getattr(kmeans, '_Kmeans__fitted') == True


        kmeans = Kmeans(
            k=3,
            init='kmeans++',
            metric='chebyshev'
        )
        kmeans.fit(X_train)

        assert isinstance(kmeans.centroids, np.ndarray)
        assert kmeans.centroids.shape[0] == 3
        assert kmeans.centroids.shape[1] == X_train.shape[1]
        assert getattr(kmeans, '_Kmeans__fitted') == True


        kmeans = Kmeans(
            k=3,
            init='kmeans++',
            metric='cityblock'
        )
        kmeans.fit(X_train)

        assert isinstance(kmeans.centroids, np.ndarray)
        assert kmeans.centroids.shape[0] == 3
        assert kmeans.centroids.shape[1] == X_train.shape[1]
        assert getattr(kmeans, '_Kmeans__fitted') == True


    def test_predict():
        kmeans = Kmeans(k=3)
        kmeans.fit(X_train)
        y_pred = kmeans.predict(X_test)

        assert y_pred.shape[0] == X_test.shape[0]


    def test_notfitted():
        kmeans = Kmeans(k=3)
        with pytest.raises(NotFitted):
            kmeans.predict(X_test)


    test_fit()
    test_predict()
    test_notfitted()