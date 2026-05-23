from typing import Optional, Self

import numpy as np
import numpy.typing as npt

from ..exc import NotFittedError
from ._node import Node


class DTRegressor:
    """
    Decision Tree Regressor.

    A non-parametric supervised learning algorithm that partitions the
    feature space into axis-aligned regions and predicts the mean target
    value of training samples in each region.

    The tree is built by recursively selecting the feature and threshold
    that minimize the weighted Mean Squared Error of the child nodes.

    **Splitting Criterion — Variance Reduction** (maximized):

    .. math::

        \\Delta(j, s) = \\text{Var}(y) - \\left[
            \\frac{n_L}{n}\\text{Var}(y_L) + \\frac{n_R}{n}\\text{Var}(y_R)
        \\right]


    where variance is computed efficiently as:

    .. math::

        \\text{Var}(y) = \\frac{\\sum y_i^2}{n}
            - \\left(\\frac{\\sum y_i}{n}\\right)^2

    **Leaf Prediction** (mean of assigned samples):

    .. math::

        \\hat{y}_m = \\frac{1}{|R_m|} \\sum_{x_i \\in R_m} y_i

    **Prediction** for a new sample :math:`x`:

    .. math::

        \\hat{f}(x) = \\sum_{m=1}^{M} \\hat{y}_m \\cdot \\mathbf{1}[x \\in R_m]

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree. Controls model complexity and
        prevents overfitting. Must be :math:`\\geq 1`.

    .. math::

        \\text{max\\_nodes} \\leq 2^{\\text{max\\_depth}} - 1

    min_split : int, default=5
        Minimum number of samples required to attempt a split.
        Nodes with fewer samples become leaves.

    min_leaf : int, default=5
        Minimum number of samples required in each child node.
        Splits creating leaves with fewer samples are rejected.

        .. math::

            n_L \\geq \\text{min\\_leaf} \\land n_R \\geq \\text{min\\_leaf}

    min_impurity_decrease : float, default=1e-2
        Minimum impurity decrease required for a split to be accepted.
        Acts as a regularization parameter.

        .. math::

            \\Delta(j, s) > \\text{min\\_impurity\\_decrease}

    Attributes
    ----------
    root : Node or None
        Root node of the fitted decision tree. ``None`` before fitting.

    max_depth : int
        Maximum depth constraint.

    min_split : int
        Minimum samples for splitting.

    min_leaf : int
        Minimum samples per leaf.

    min_impurity_decrease : float
        Minimum variance reduction threshold.

    Notes
    -----
    Candidate thresholds are midpoints between consecutive sorted feature
    values:

    .. math::

        t = \\frac{u_{(i)} + u_{(i+1)}}{2},
        \\quad u_{(i)} \\in \\text{sorted}(X_{\\cdot f})

    The algorithm uses cumulative sum precomputation for :math:`O(n \\log n)`
    per-feature split search. Recursion stops when:

    - ``max_depth`` is reached.
    - ``min_split`` constraint is violated.
    - No split satisfies ``min_leaf`` or ``min_impurity_decrease``.

    Leaf nodes store the **mean** of their assigned target values.

    Examples
    --------
    >>> import numpy as np
    >>> from pyml.tree import DTRegressor
    >>> X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> model = DTRegressor(max_depth=3, min_split=2, min_leaf=1)
    >>> model.fit(X, y)
    >>> model.predict(np.array([[2.5], [4.5]]))
    array([5., 9.])
    """

    def __init__(
            self,
            max_depth: int = 5,
            min_split: int = 5,
            min_leaf: int = 5,
            min_impurity_decrease: float = 1e-2
    ) -> None:
        self.root: Optional[Node] = None
        self.max_depth: int = max_depth
        self.min_split: int = min_split
        self.min_leaf: int = min_leaf
        self.min_impurity_decrease: float = min_impurity_decrease
        self.__fitted: bool = False

    def best_split(
            self,
            X: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64]
    ) -> tuple[float, Optional[int], Optional[np.float64]]:
        """
        Find the optimal feature and threshold to split the data.

        Searches all features and candidate thresholds — midpoints
        between consecutive sorted unique feature values:

        .. math::

            t = \\frac{u_{(i)} + u_{(i+1)}}{2},
            \\quad u_{(i)} \\in \\text{sorted}(X_{\\cdot f})

        Uses cumulative sums for :math:`O(n \\log n)` per-feature
        computation. For each candidate split, computes the variance
        reduction:

        .. math::

            \\Delta = \\text{Var}(y) - \\left[
                \\frac{n_L}{n}\\text{Var}(y_L)
                + \\frac{n_R}{n}\\text{Var}(y_R)
            \\right]

        where variances are computed via the running-sum formula:

        .. math::

            \\text{Var}(y_{1:k}) = \\frac{S^{(2)}_k}{k}
                - \\left(\\frac{S^{(1)}_k}{k}\\right)^2

            S^{(1)}_k = \\sum_{i=1}^{k} y_{(i)}, \\quad
            S^{(2)}_k = \\sum_{i=1}^{k} y_{(i)}^2

        Only considers splits where both children satisfy the
        ``min_leaf`` constraint and feature values are distinct.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix at the current node.

        y : np.ndarray of shape (n_samples,)
            Target values at the current node.

        Returns
        -------
        best_delta : float
            Maximum variance reduction achieved.
            Returns ``-inf`` if no valid split is found.

        best_feature : int or None
            Index of the best feature for splitting.
            ``None`` if no valid split is found.

        best_threshold : np.float64 or None
            Optimal threshold value. Midpoint between two
            consecutive sorted feature values.
            ``None`` if no valid split is found.
        """
        best_threshold: Optional[np.float64] = None
        best_feature: Optional[int] = None
        best_delta: float = - np.inf
        features = X.shape[1]
        n = X.shape[0]
        for feature_id in range(features):
            sorted_idx = np.argsort(X[:, feature_id])
            feature_col = X[:, feature_id]
            X_sorted = feature_col[sorted_idx]
            y_sorted = y[sorted_idx]
            cumsum_y = np.cumsum(y_sorted)
            cumsum_y_sq = np.cumsum(y_sorted ** 2)
            total_loss = cumsum_y_sq[n - 1] / n - (cumsum_y[n - 1] / n) ** 2
            for i in range(1, n):
                n_left = i
                n_right = n - i
                if X_sorted[i - 1] == X_sorted[i]:
                    continue
                if n_left < self.min_leaf or n_right < self.min_leaf:
                    continue
                sum_left = cumsum_y[i - 1]
                sum_right = cumsum_y[n - 1] - cumsum_y[i - 1]
                sum_left_sq = cumsum_y_sq[i - 1]
                sum_right_sq = cumsum_y_sq[n - 1] - cumsum_y_sq[i - 1]
                mse_left = sum_left_sq / n_left - (sum_left / n_left) ** 2
                mse_right = sum_right_sq / n_right - (sum_right / n_right) ** 2
                delta = total_loss - (n_left * mse_left + n_right * mse_right) / n
                if delta > best_delta:
                    best_delta = delta
                    best_feature = feature_id
                    best_threshold = (X_sorted[i - 1] + X_sorted[i]) / 2.0
        return (best_delta, best_feature, best_threshold)

    def build_tree(
            self,
            X: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            depth: int
    ) -> Node:
        """
        Recursively construct the regression tree.

        At each node, finds the best split and recurses on the left
        and right partitions. Stops recursion when:

        - ``max_depth`` is reached.
        - ``min_split`` constraint is violated (:math:`n <` ``min_split``).
        - No split satisfies ``min_leaf`` constraint.
        - No split achieves impurity decrease above
          ``min_impurity_decrease``.

        When stopping, creates a leaf node with the **mean** target value:

        .. math::

            \\hat{y}_{\\text{leaf}} = \\frac{1}{n} \\sum_{i=1}^{n} y_i

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix at the current node.

        y : np.ndarray of shape (n_samples,)
            Target values at the current node.

        depth : int
            Current depth. Root starts at :math:`0`.

        Returns
        -------
        node : Node
            Root of the constructed subtree. Leaf nodes contain only
            ``value`` (mean target); internal nodes also contain
            ``feature``, ``threshold``, ``left``, and ``right``.
        """
        if depth >= self.max_depth or y.shape[0] < self.min_split:
            return Node(value=np.float64(np.mean(y)))
        best_delta, best_feature, best_threshold = self.best_split(X, y)
        if best_feature is None or best_threshold is None or \
                best_delta <= self.min_impurity_decrease:
            return Node(value=np.float64(np.mean(y)))
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        left_node = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        node = Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node
        )
        return node

    def fit(
            self,
            X: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64]
    ) -> Self:
        """
        Build the decision tree from training data.

        Constructs a binary tree by recursively partitioning the
        feature space to minimize the weighted MSE. The resulting
        tree defines a piecewise-constant prediction function:

        .. math::

            \\hat{f}(x) = \\sum_{m=1}^{M} \\hat{y}_m
                \\cdot \\mathbf{1}[x \\in R_m]

        where :math:`\\hat{y}_m` is the mean of training samples
        in region :math:`R_m`.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.

        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : DTRegressor
            Fitted regressor with ``root`` attribute set.
        """
        self.root = self.build_tree(X, y, 0)
        self.__fitted = True
        return self

    def determine_value(self, x: npt.NDArray[np.float64]) -> np.float64:
        """
        Predict target value for a single sample.

        Traverses the tree from ``root`` to a leaf by comparing the
        sample's feature values against node thresholds:

        - If :math:`x_f \\leq t` → go left.
        - If :math:`x_f > t` → go right.

        The leaf returns its stored mean value:

        .. math::

            \\hat{y} = \\bar{y}_{\\text{leaf}}

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            A single sample feature vector.

        Returns
        -------
        value : np.float64
            Predicted target value from the reached leaf node.

        Raises
        ------
        RuntimeError
            If an invalid node state is encountered during traversal.
        """
        current = self.root
        while current is not None:
            if current.value is not None:
                return np.float64(current.value)
            elif current.feature is not None and \
                current.threshold is not None:
                if x[current.feature] <= current.threshold:
                    current = current.left
                else:
                    current = current.right
            else:
                raise RuntimeError(
                    f"Invalid node state: feature={current.feature}, "
                    f"threshold={current.threshold}, value={current.value}"
                )
        raise RuntimeError("Unexpectedly reached end of tree traversal")

    def predict(
            self,
            X: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Predict target values for all samples in X.

        Each sample independently traverses the tree via
        :meth:`determine_value`:

        .. math::

            \\hat{y}_i = \\hat{f}(x_i) = \\sum_{m=1}^{M}
                \\hat{y}_m \\cdot \\mathbf{1}[x_i \\in R_m]

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix of samples to predict.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.

        Raises
        ------
        NotFittedError
            If called before fitting the model.
        """
        if not self.__fitted:
            raise NotFittedError(self)
        y_pred = np.zeros(shape=X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            y_pred[i] = self.determine_value(X[i])
        return y_pred
