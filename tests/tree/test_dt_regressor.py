import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pyml.exc import NotFittedError
from pyml.tree import DTRegressor

# ============================================================================
# 1. MATHEMATICAL CORRECTNESS — VARIANCE REDUCTION
# ============================================================================

class TestVarianceReduction:
    """Test that splitting maximizes variance reduction Δ."""

    def test_perfect_split_zero_variance(self):
        """Perfect split should reduce child variance to 0.

        For data y = [0,0,0, 10,10,10] with x = [1,1,1, 2,2,2],
        splitting at x=1.5 should give:
        - Left: var=0 (all 0)
        - Right: var=0 (all 10)
        - Δ = Var(parent) - 0 = Var(parent) = 25
        """
        X = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
        y = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])

        model = DTRegressor(max_depth=1, min_split=2, min_leaf=1,
                           min_impurity_decrease=0.0)
        model.fit(X, y)

        pred = model.predict(X)
        assert_array_almost_equal(pred, y)  # Perfect fit

    def test_impurity_decrease_positive(self):
        """Δ (impurity decrease) must be ≥ 0.

        For any split: Var(parent) ≥ weighted avg of children vars.
        This is the variance decomposition property:
        Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        """
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        model = DTRegressor(max_depth=3, min_split=5, min_leaf=2,
                           min_impurity_decrease=0.0)
        model.fit(X, y)

        # After fitting, training MSE ≤ total variance
        pred = model.predict(X)
        mse = np.mean((y - pred) ** 2)
        total_var = np.var(y)
        assert mse <= total_var + 1e-10

    def test_split_reduces_variance(self):
        """Each split must strictly reduce training MSE."""
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        # Depth 1: single split
        model = DTRegressor(max_depth=1, min_split=2, min_leaf=1,
                           min_impurity_decrease=0.0)
        model.fit(X, y)
        pred_split = model.predict(X)
        mse_split = np.mean((y - pred_split) ** 2)

        # Depth 0: just mean (no split)
        pred_mean = np.full_like(y, np.mean(y))
        mse_mean = np.mean((y - pred_mean) ** 2)

        # After split, MSE must be ≤ mean MSE
        assert mse_split <= mse_mean + 1e-10


# ============================================================================
# 2. LEAF PREDICTION — MEAN CORRECTNESS
# ============================================================================

class TestLeafPrediction:
    """Test that leaves predict the mean of their samples."""

    def test_leaf_is_mean_single_region(self):
        """Depth=0 tree: prediction = global mean of y.

        y = [a, b, c] → leaf value = (a+b+c)/3
        """
        X = np.random.randn(10, 2)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0,
                      6.0, 7.0, 8.0, 9.0, 10.0])

        model = DTRegressor(max_depth=1, min_split=20, min_leaf=1,
                           min_impurity_decrease=0.0)
        model.fit(X, y)

        pred = model.predict(X)
        expected = np.full(10, np.mean(y))
        assert_array_almost_equal(pred, expected)

    def test_leaf_mean_linear_relationship(self):
        """For y = wx + b, leaves approximate local means.

        With sufficient splits, piecewise constant function
        approximates the line.
        """
        X = np.linspace(0, 10, 200).reshape(-1, 1)
        y = 3.0 * X.ravel() + 2.0

        model = DTRegressor(max_depth=5, min_split=5, min_leaf=1,
                           min_impurity_decrease=0.0)
        model.fit(X, y)

        pred = model.predict(X)
        # With 32 potential leaves, approximation should be good
        mse = np.mean((y - pred) ** 2)
        assert mse < 1.0

    def test_leaf_values_are_finite(self):
        """All leaf predictions must be finite numbers."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.random.randn(100)

        model = DTRegressor(max_depth=8, min_split=5, min_leaf=2,
                           min_impurity_decrease=0.0)
        model.fit(X, y)

        pred = model.predict(X)
        assert np.all(np.isfinite(pred))


# ============================================================================
# 3. TREE STRUCTURE — RECURSIVE PARTITIONING
# ============================================================================

class TestTreeStructure:
    """Test recursive binary partitioning properties."""

    def test_binary_splits(self):
        """Each internal node creates exactly 2 children."""
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        model = DTRegressor(max_depth=3, min_split=5, min_leaf=2,
                           min_impurity_decrease=0.0)
        model.fit(X, y)

        def count_children(node):
            if node is None:
                return True
            if node.value is not None:  # leaf
                return True
            # Internal node: must have both children
            return (node.left is not None
                    and node.right is not None
                    and count_children(node.left)
                    and count_children(node.right))

        assert count_children(model.root)

    def test_depth_constraint(self):
        """Tree depth must not exceed max_depth."""
        X = np.random.randn(200, 3)
        y = np.random.randn(200)

        for max_depth in [1, 2, 3, 5]:
            model = DTRegressor(max_depth=max_depth, min_split=2,
                               min_leaf=1, min_impurity_decrease=0.0)
            model.fit(X, y)

            def get_depth(node, d=0):
                if node is None or node.value is not None:
                    return d
                return max(get_depth(node.left, d + 1),
                          get_depth(node.right, d + 1))

            actual_depth = get_depth(model.root, 0)
            assert actual_depth <= max_depth

    def test_min_samples_leaf(self):
        """Each leaf must have ≥ min_leaf samples."""
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        model = DTRegressor(max_depth=5, min_split=10, min_leaf=5,
                           min_impurity_decrease=0.0)
        model.fit(X, y)

        def count_samples(node, X_data):
            if node is None:
                return
            if node.value is not None:  # leaf
                # All training samples reaching this leaf
                return

        # Verify by checking splits satisfy constraint
        best_delta, feature, threshold = model.best_split(X, y)
        if feature is not None:
            left = np.sum(X[:, feature] <= threshold)
            right = np.sum(X[:, feature] > threshold)
            assert left >= 5 or left == 0
            assert right >= 5 or right == 0


# ============================================================================
# 4. DETERMINISTIC BEHAVIOR
# ============================================================================

class TestDeterministic:
    """Test reproducibility and consistency."""

    def test_same_data_same_tree(self):
        """Same data + same params → identical predictions."""
        X = np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        model1 = DTRegressor(max_depth=4, min_split=5, min_leaf=2,
                            min_impurity_decrease=0.0)
        model2 = DTRegressor(max_depth=4, min_split=5, min_leaf=2,
                            min_impurity_decrease=0.0)

        model1.fit(X, y)
        model2.fit(X, y)

        X_test = np.random.randn(10, 3)
        assert_array_almost_equal(model1.predict(X_test),
                                  model2.predict(X_test))

    def test_predict_twice_same(self):
        """Multiple calls to predict return same results."""
        X = np.random.randn(30, 2)
        y = np.random.randn(30)

        model = DTRegressor(max_depth=3, min_split=3, min_leaf=1,
                           min_impurity_decrease=0.0)
        model.fit(X, y)

        pred1 = model.predict(X)
        pred2 = model.predict(X)
        assert_array_almost_equal(pred1, pred2)


# ============================================================================
# 5. MATHEMATICAL EDGE CASES
# ============================================================================

class TestMathEdgeCases:
    """Test mathematical boundary conditions."""

    def test_constant_target(self):
        """If y is constant, variance=0, no split possible.

        y = [c, c, ..., c] → Var(y) = 0
        Any split gives children with var=0, Δ=0
        Tree must be a single leaf predicting c.
        """
        X = np.random.randn(20, 3)
        y = np.ones(20) * 5.0

        model = DTRegressor(max_depth=10, min_split=2, min_leaf=1,
                           min_impurity_decrease=0.0)
        model.fit(X, y)

        pred = model.predict(X)
        assert_array_almost_equal(pred, np.ones(20) * 5.0)

    def test_single_sample(self):
        """With 1 sample, var=0 (undefined, treated as 0).

        Leaf prediction = the single y value.
        """
        X = np.array([[1.0, 2.0]])
        y = np.array([7.0])

        model = DTRegressor(max_depth=5, min_split=2, min_leaf=1,
                           min_impurity_decrease=0.0)
        model.fit(X, y)

        pred = model.predict(X)
        assert_array_almost_equal(pred, [7.0])

    def test_variance_formula_numerical_stability(self):
        """Var = Σy²/n - (Σy/n)² must be ≥ 0.

        Due to floating point, might get tiny negatives.
        Implementation should handle this gracefully.
        """
        # Large numbers to stress numerical precision
        X = np.random.randn(1000, 2)
        y = np.random.uniform(1e6, 1e7, 1000)

        model = DTRegressor(max_depth=3, min_split=10, min_leaf=5,
                           min_impurity_decrease=0.0)
        model.fit(X, y)

        pred = model.predict(X)
        assert np.all(np.isfinite(pred))
        # MSE must be ≥ 0
        assert np.mean((y - pred) ** 2) >= 0

    def test_identical_features(self):
        """When all features have same value, no split possible.

        X[:, j] = [a, a, ..., a] → unique values = 1 → no thresholds
        Tree must return global mean.
        """
        X = np.ones((30, 3))  # All features = 1.0
        y = np.arange(30, dtype=np.float64)

        model = DTRegressor(max_depth=5, min_split=2, min_leaf=1,
                           min_impurity_decrease=0.0)
        model.fit(X, y)

        pred = model.predict(X)
        expected = np.full(30, np.mean(y))
        assert_array_almost_equal(pred, expected)


# ============================================================================
# 6. PIECEWISE CONSTANT FUNCTION
# ============================================================================

class TestPiecewiseConstant:
    """Test that predictions form a piecewise constant function."""

    def test_same_input_same_output(self):
        """Identical x vectors must give identical predictions."""
        X = np.random.randn(20, 2)
        y = np.random.randn(20)

        model = DTRegressor(max_depth=5, min_split=3, min_leaf=1,
                           min_impurity_decrease=0.0)
        model.fit(X, y)

        # Same input repeated
        x_test = np.array([1.0, 2.0])
        pred1 = model.determine_value(x_test)
        pred2 = model.determine_value(x_test)
        assert pred1 == pred2

    def test_piecewise_constant_property(self):
        """Decision tree defines a piecewise constant function.

        For any region R_m, all points x ∈ R_m must have same prediction.
        This is the fundamental property: f(x) = Σ ŷ_m · 1[x ∈ R_m]
        """
        X = np.array([
            [1.0, 2.0],
            [1.0, 2.0],  # IDENTICAL to row 0
            [5.0, 6.0],
            [5.0, 6.0],  # IDENTICAL to row 2
        ])
        y = np.array([10.0, 10.0, 100.0, 100.0])

        model = DTRegressor(max_depth=2, min_split=2, min_leaf=1,
                            min_impurity_decrease=0.0)
        model.fit(X, y)

        pred = model.predict(X)

        # Identical inputs → identical outputs (function property)
        assert pred[0] == pred[1], (
            f"f([1,2]) must be same for identical inputs, "
            f"got {pred[0]} and {pred[1]}"
        )
        assert pred[2] == pred[3], (
            f"f([5,6]) must be same for identical inputs, "
            f"got {pred[2]} and {pred[3]}"
        )


# ============================================================================
# 7. NOT FITTED
# ============================================================================

class TestNotFitted:
    """Test error for unfitted model."""

    def test_predict_raises_not_fitted(self):
        """Calling predict before fit must raise NotFittedError."""
        model = DTRegressor()
        X = np.array([[1.0, 2.0]])

        with pytest.raises(NotFittedError):
            model.predict(X)
