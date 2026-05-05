import numpy as np
from typing import Literal, Self
from .node import Node
from ...exc import NotFitted


class DTClassifier:
    """
    Decision Tree Classifier.

    A non-parametric supervised learning algorithm that partitions the
    feature space into axis-aligned regions and assigns a majority class
    label to each region.

    The tree is built by recursively selecting the feature and threshold
    that best separates the classes, measured by one of two impurity criteria:

    **Gini Impurity** (minimized):

    .. math::

        \\text{Gini}(S) = 1 - \\sum_{i=1}^{c} p_i^2

    where :math:`p_i` is the proportion of class :math:`i` in node :math:`S`.
    Ranges from :math:`0` (pure) to :math:`0.5` (binary, maximally mixed).

    **Information Gain** (maximized):

    .. math::

        IG = H(S) - \\frac{n_L}{n} H(S_L) - \\frac{n_R}{n} H(S_R)

    where Shannon entropy is:

    .. math::

        H(S) = -\\sum_{i=1}^{c} p_i \\log_2(p_i)

    and :math:`S_L`, :math:`S_R` are the left and right child nodes.

    Parameters
    ----------
    depth : int, default=5
        Maximum depth of the tree. Limits model complexity and
        controls overfitting.
    min_sample : int, default=5
        Minimum number of samples required in each child node for a
        split to be considered valid.
    algorithm : {'gini', 'entropy'}, default='gini'
        Splitting criterion.

        - ``'gini'`` — minimizes weighted Gini impurity.
        - ``'entropy'`` — maximizes Information Gain.

    Attributes
    ----------
    root : Node or None
        Root node of the fitted decision tree.

    Notes
    -----
    Candidate thresholds are computed as midpoints between consecutive
    unique feature values:

    .. math::

        t = \\frac{u_j + u_{j+1}}{2}, \\quad u_j \\in \\text{unique}(x_{\\cdot f})

    Leaf nodes store the majority class of their assigned samples.
    Recursion stops when the node is pure, ``max_depth`` is reached,
    or no valid split exists.

    Examples
    --------
    >>> clf = DTClassifier(depth=3, min_sample=2, algorithm='gini')
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(
            self,
            depth: int = 5,
            min_sample: int = 5,
            algorithm: Literal['gini', 'entropy'] = 'gini'
    ) -> None:

        self.root: Node | None = None
        self.max_depth = depth
        self.min_sample = min_sample
        self.algorithm = algorithm
        self.__fitted = False

    def gini(self, y: np.ndarray) -> float:
        """
        Compute Gini impurity for a node.

        Measures the probability of misclassifying a randomly chosen
        sample if it were labeled according to the class distribution:

        .. math::

            \\text{Gini}(y) = 1 - \\sum_{i=1}^{c} p_i^2,
            \\quad p_i = \\frac{|\\{y_j = i\\}|}{n}

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Class labels at this node.

        Returns
        -------
        impurity : float
            Gini impurity in :math:`[0, 0.5]` for binary classification.
            :math:`0` indicates a perfectly pure node.
        """

        n = y.shape[0]
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return 1 - np.sum(probs ** 2)

    def weighted_gini(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Compute weighted Gini impurity for a binary split.

        Averages child impurities weighted by their sample proportions:

        .. math::

            \\text{WG} = \\frac{n_L}{n} \\text{Gini}(S_L)
                       + \\frac{n_R}{n} \\text{Gini}(S_R)

        where :math:`n = n_L + n_R`. Lower values indicate better splits.

        Parameters
        ----------
        y_left : np.ndarray of shape (n_left,)
            Labels in the left child node.
        y_right : np.ndarray of shape (n_right,)
            Labels in the right child node.

        Returns
        -------
        weighted_impurity : float
            Weighted Gini impurity of the split.
        """

        n = y_left.shape[0] + y_right.shape[0]
        n_left, n_right = y_left.shape[0], y_right.shape[0]
        weighted_gini = (n_left / n) * self.gini(y_left) \
            + (n_right / n) * self.gini(y_right)
        return weighted_gini

    def entropy(self, y: np.ndarray) -> float:
        """
        Compute Shannon entropy for a node.

        Measures the uncertainty in the class distribution.
        Higher entropy indicates more mixed classes:

        .. math::

            H(y) = -\\sum_{i=1}^{c} p_i \\log_2(p_i + \\varepsilon),
            \\quad p_i = \\frac{|\\{y_j = i\\}|}{n}

        where :math:`\\varepsilon = 10^{-12}` prevents :math:`\\log_2(0)`.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Class labels at this node.

        Returns
        -------
        entropy : float
            Entropy in :math:`[0, \\log_2(c)]` where :math:`c` is the
            number of classes. :math:`0` indicates a perfectly pure node.
        """

        _, counts = np.unique(y, return_counts=True)
        probs = counts / y.shape[0]
        entropy = - np.sum(probs * np.log2(probs + 1e-12))
        return entropy

    def weighted_entropy(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Compute weighted Shannon entropy for a binary split.

        .. math::

            WH = \\frac{n_L}{n} H(S_L) + \\frac{n_R}{n} H(S_R)

        where :math:`n = n_L + n_R`. Lower values indicate purer splits.

        Parameters
        ----------
        y_left : np.ndarray of shape (n_left,)
            Labels in the left child node.
        y_right : np.ndarray of shape (n_right,)
            Labels in the right child node.

        Returns
        -------
        weighted_entropy : float
            Weighted entropy of the split.
        """

        n = y_left.shape[0] + y_right.shape[0]
        return (y_left.shape[0] / n) * self.entropy(y_left) \
            + (y_right.shape[0] / n) * self.entropy(y_right)

    def information_gain(self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Compute Information Gain for a binary split.

        Measures the reduction in entropy achieved by splitting
        the parent node. Higher values indicate purer children:

        .. math::

            IG = H(S) - \\frac{n_L}{n} H(S_L) - \\frac{n_R}{n} H(S_R)

        Ranges from :math:`0` (no improvement) to :math:`H(S)` (perfect split).

        Parameters
        ----------
        y_parent : np.ndarray of shape (n_samples,)
            Labels at the parent node before splitting.
        y_left : np.ndarray of shape (n_left,)
            Labels in the left child node.
        y_right : np.ndarray of shape (n_right,)
            Labels in the right child node.

        Returns
        -------
        gain : float
            Information gain. Higher is better.
        """

        return self.entropy(y_parent) - self.weighted_entropy(y_left, y_right)

    def best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find the optimal feature and threshold to split the data.

        Searches all features and candidate thresholds — midpoints
        between consecutive unique feature values:

        .. math::

            t = \\frac{u_j + u_{j+1}}{2},
            \\quad u_j \\in \\text{unique}(X_{\\cdot f})

        For ``'gini'`` — minimizes weighted Gini impurity.
        For ``'entropy'`` — maximizes Information Gain.

        Only considers splits where both children contain at least
        ``min_sample`` samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix at the current node.
        y : np.ndarray of shape (n_samples,)
            Class labels at the current node.

        Returns
        -------
        best_feature : int or None
            Index of the best splitting feature.
            ``None`` if no valid split is found.
        best_threshold : float or None
            Optimal threshold value for the best split.
            ``None`` if no valid split is found.
        """

        best_threshold = None
        best_feature = None
        if self.algorithm == 'gini':
            best_score = 1
        elif self.algorithm == 'entropy':
            best_score = - np.inf
        features = X.shape[1]
        for feature_id in range(features):
            feature_col = X[:, feature_id]
            feature_col_uniq = np.unique(feature_col)
            thresholds = (feature_col_uniq[:-1] + feature_col_uniq[1:]) / 2
            for threshold in thresholds:
                y_left: np.ndarray = y[X[:, feature_id] < threshold]
                y_right: np.ndarray = y[X[:, feature_id] >= threshold]

                if y_left.shape[0] < self.min_sample or \
                        y_right.shape[0] < self.min_sample:
                    continue

                if self.algorithm == 'gini':
                    score = self.weighted_gini(y_left, y_right)
                    if score < best_score:
                        best_score = score
                        best_threshold = threshold
                        best_feature = feature_id
                elif self.algorithm == 'entropy':
                    score = self.information_gain(y, y_left, y_right)
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                        best_feature = feature_id
        return best_feature, best_threshold

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively construct the decision tree.

        At each node, finds the best split and recurses on the left
        and right partitions. Stops recursion when:

        - The node is **pure** — all samples belong to one class.
        - ``max_depth`` is reached.
        - No valid split exists (returns a leaf with majority class).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix at the current node.
        y : np.ndarray of shape (n_samples,)
            Class labels at the current node.
        depth : int
            Current depth. Root starts at :math:`0`.

        Returns
        -------
        node : Node
            Root of the constructed subtree. Leaf nodes contain only
            ``value`` (majority class); internal nodes also contain
            ``feature``, ``threshold``, ``left``, and ``right``.
        """

        y_uniq, counts = np.unique(y, return_counts=True)
        if y_uniq.shape[0] == 1 or depth >= self.max_depth:
            return Node(value=y_uniq[np.argmax(counts)])
        best_feature, best_threshold = self.best_split(X, y)
        if best_feature is None or best_threshold is None:
            return Node(value=y_uniq[np.argmax(counts)])
        left_mask = X[:, best_feature] < best_threshold
        right_mask = X[:, best_feature] >= best_threshold
        left_node = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        node = Node(best_feature, best_threshold, left_node, right_node)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Build the decision tree from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : DTClassifier
            Fitted classifier with ``root`` node set.
        """

        self.root = self.build_tree(X, y, 0)
        self.__fitted = True
        return self

    def determine_label(self, x: np.ndarray) -> int:
        """
        Predict class label for a single sample.

        Traverses the tree from ``root`` to a leaf by comparing the
        sample's feature values against node thresholds:

        - If :math:`x[f] \\geq t` → go right.
        - If :math:`x[f] < t` → go left.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            A single sample to classify.

        Returns
        -------
        label : int
            Predicted class label at the reached leaf node.
        """

        cur_node = self.root
        while cur_node is not None:
            if cur_node.value is not None:
                return cur_node.value
            elif x[cur_node.feature] >= cur_node.threshold:
                cur_node = cur_node.right
            else:
                cur_node = cur_node.left

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for all samples in X.

        Each sample traverses the tree via :meth:`determine_label`
        until reaching a leaf node whose majority class is returned.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix of samples to classify.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted integer class labels.

        Raises
        ------
        NotFitted
            If called before fitting the model.
        """

        if not self.__fitted:
            raise NotFitted(self)
        y = np.zeros(shape=X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            y[i] = self.determine_label(X[i])
        return y