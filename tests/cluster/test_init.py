import pytest

from pyml import cluster


class TestLazyImports:
    def test_dbscan_is_importable(self):
        assert cluster.DBSCAN.__name__ == "DBSCAN"

    def test_kmeans_is_importable(self):
        assert cluster.KMeans.__name__ == "KMeans"

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="NonExistent"):
            _ = cluster.NonExistent  # type: ignore[attr-defined]


class TestDir:
    def test_dir_matches_all(self):
        assert set(dir(cluster)) == set(cluster.__all__)  # type: ignore[attr-defined]

    def test_dir_is_sorted(self):
        assert dir(cluster) == sorted(cluster.__all__)  # type: ignore[attr-defined]
