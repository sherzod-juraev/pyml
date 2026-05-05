import pytest
import numpy as np
from mlkit.preprocess import MinMaxScaler, StandardScaler, RobustScaler
from mlkit.exc import NotFitted


# ─── Fixtures ───────────────────────────────────────────────

@pytest.fixture
def data():
    np.random.seed(42)
    return np.random.randn(100, 4)


@pytest.fixture
def data_with_outliers():
    X = np.random.randn(100, 4)
    X[0] = 1000
    X[-1] = -1000
    return X


@pytest.fixture
def constant_feature():
    X = np.random.randn(50, 3)
    X[:, 1] = 5.0  # ikkinchi ustun constant
    return X


# ════════════════════════════════════════════════════════════
#  MinMaxScaler
# ════════════════════════════════════════════════════════════

class TestMinMaxFit:

    def test_fit_returns_self(self, data):
        scaler = MinMaxScaler()
        assert scaler.fit(data) is scaler

    def test_min_shape(self, data):
        scaler = MinMaxScaler()
        scaler.fit(data)
        assert scaler.min_.shape == (data.shape[1],)

    def test_max_shape(self, data):
        scaler = MinMaxScaler()
        scaler.fit(data)
        assert scaler.max_.shape == (data.shape[1],)

    def test_min_values_correct(self, data):
        scaler = MinMaxScaler()
        scaler.fit(data)
        np.testing.assert_array_almost_equal(scaler.min_, np.min(data, axis=0))

    def test_max_values_correct(self, data):
        scaler = MinMaxScaler()
        scaler.fit(data)
        np.testing.assert_array_almost_equal(scaler.max_, np.max(data, axis=0))

    def test_constant_feature_handled(self, constant_feature):
        scaler = MinMaxScaler()
        scaler.fit(constant_feature)
        assert scaler.max_[1] == scaler.min_[1] + 1

class TestMinMaxTransform:

    def test_transform_raises_not_fitted(self):
        scaler = MinMaxScaler()
        with pytest.raises(NotFitted):
            scaler.transform(np.random.randn(10, 3))

    def test_output_shape(self, data):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data)
        assert X_scaled.shape == data.shape

    def test_output_min_is_zero(self, data):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data)
        np.testing.assert_array_almost_equal(np.min(X_scaled, axis=0), 0)

    def test_output_max_is_one(self, data):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data)
        np.testing.assert_array_almost_equal(np.max(X_scaled, axis=0), 1)

    def test_values_in_range(self, data):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data)
        assert np.all(X_scaled >= 0)
        assert np.all(X_scaled <= 1)

    def test_constant_feature_is_zero(self, constant_feature):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(constant_feature)
        np.testing.assert_array_almost_equal(X_scaled[:, 1], 0)


class TestMinMaxInverseTransform:

    def test_inverse_raises_not_fitted(self):
        scaler = MinMaxScaler()
        with pytest.raises(NotFitted):
            scaler.inverse_transform(np.random.randn(10, 3))

    def test_inverse_restores_original(self, data):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data)
        X_restored = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X_restored, data)

    def test_fit_transform_equals_fit_then_transform(self, data):
        s1 = MinMaxScaler()
        s1.fit(data)
        X1 = s1.transform(data)

        s2 = MinMaxScaler()
        X2 = s2.fit_transform(data)

        np.testing.assert_array_almost_equal(X1, X2)


# ════════════════════════════════════════════════════════════
#  StandardScaler
# ════════════════════════════════════════════════════════════

class TestStandardFit:

    def test_fit_returns_self(self, data):
        scaler = StandardScaler()
        assert scaler.fit(data) is scaler

    def test_mean_shape(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        assert scaler.mean_.shape == (data.shape[1],)

    def test_std_shape(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        assert scaler.std_.shape == (data.shape[1],)

    def test_mean_values_correct(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        np.testing.assert_array_almost_equal(scaler.mean_, np.mean(data, axis=0))

    def test_std_values_correct(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        np.testing.assert_array_almost_equal(scaler.std_, np.std(data, axis=0))

    def test_constant_feature_handled(self, constant_feature):
        scaler = StandardScaler()
        scaler.fit(constant_feature)
        assert scaler.std_[1] == 1
        assert scaler.mean_[1] == 5.0


class TestStandardTransform:

    def test_transform_raises_not_fitted(self):
        scaler = StandardScaler()
        with pytest.raises(NotFitted):
            scaler.transform(np.random.randn(10, 3))

    def test_output_shape(self, data):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        assert X_scaled.shape == data.shape

    def test_mean_near_zero(self, data):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        np.testing.assert_array_almost_equal(
            np.mean(X_scaled, axis=0), 0, decimal=10
        )

    def test_std_near_one(self, data):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        np.testing.assert_array_almost_equal(
            np.std(X_scaled, axis=0), 1, decimal=10
        )

    def test_constant_feature_is_zero(self, constant_feature):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(constant_feature)
        np.testing.assert_array_almost_equal(X_scaled[:, 1], 0)


class TestStandardInverseTransform:

    def test_inverse_raises_not_fitted(self):
        scaler = StandardScaler()
        with pytest.raises(NotFitted):
            scaler.inverse_transform(np.random.randn(10, 3))

    def test_inverse_restores_original(self, data):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        X_restored = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X_restored, data)

    def test_fit_transform_equals_fit_then_transform(self, data):
        s1 = StandardScaler()
        s1.fit(data)
        X1 = s1.transform(data)

        s2 = StandardScaler()
        X2 = s2.fit_transform(data)

        np.testing.assert_array_almost_equal(X1, X2)


# ════════════════════════════════════════════════════════════
#  RobustScaler
# ════════════════════════════════════════════════════════════

class TestRobustFit:

    def test_fit_returns_self(self, data):
        scaler = RobustScaler()
        assert scaler.fit(data) is scaler

    def test_median_shape(self, data):
        scaler = RobustScaler()
        scaler.fit(data)
        assert scaler.median_.shape == (data.shape[1],)

    def test_iqr_shape(self, data):
        scaler = RobustScaler()
        scaler.fit(data)
        assert scaler.iqr_.shape == (data.shape[1],)

    def test_median_values_correct(self, data):
        scaler = RobustScaler()
        scaler.fit(data)
        np.testing.assert_array_almost_equal(
            scaler.median_, np.percentile(data, 50, axis=0)
        )

    def test_iqr_values_correct(self, data):
        scaler = RobustScaler()
        scaler.fit(data)
        expected = np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)
        np.testing.assert_array_almost_equal(scaler.iqr_, expected)

    def test_constant_feature_handled(self, constant_feature):
        scaler = RobustScaler()
        scaler.fit(constant_feature)
        assert scaler.iqr_[1] == 1
        assert scaler.median_[1] == 5.0


class TestRobustTransform:

    def test_transform_raises_not_fitted(self):
        scaler = RobustScaler()
        with pytest.raises(NotFitted):
            scaler.transform(np.random.randn(10, 3))

    def test_output_shape(self, data):
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(data)
        assert X_scaled.shape == data.shape

    def test_median_near_zero_after_scale(self, data):
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(data)
        np.testing.assert_array_almost_equal(
            np.median(X_scaled, axis=0), 0, decimal=10
        )

    def test_robust_to_outliers(self, data, data_with_outliers):
        s1 = RobustScaler()
        s1.fit(data)

        s2 = RobustScaler()
        s2.fit(data_with_outliers)

        # Outlier bo'lsa ham median va IQR katta farq qilmasligi kerak
        median_diff = np.abs(s1.median_ - s2.median_)
        assert np.all(median_diff < 5)

    def test_standard_scaler_sensitive_to_outliers(self, data, data_with_outliers):
        s1 = StandardScaler()
        s1.fit(data)

        s2 = StandardScaler()
        s2.fit(data_with_outliers)

        # StandardScaler outlierga sezgir — std katta farq qiladi
        std_diff = np.abs(s1.std_ - s2.std_)
        assert np.any(std_diff > 10)

    def test_constant_feature_is_zero(self, constant_feature):
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(constant_feature)
        np.testing.assert_array_almost_equal(X_scaled[:, 1], 0)


class TestRobustInverseTransform:

    def test_inverse_raises_not_fitted(self):
        scaler = RobustScaler()
        with pytest.raises(NotFitted):
            scaler.inverse_transform(np.random.randn(10, 3))

    def test_inverse_restores_original(self, data):
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(data)
        X_restored = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X_restored, data)

    def test_fit_transform_equals_fit_then_transform(self, data):
        s1 = RobustScaler()
        s1.fit(data)
        X1 = s1.transform(data)

        s2 = RobustScaler()
        X2 = s2.fit_transform(data)

        np.testing.assert_array_almost_equal(X1, X2)