import numpy as np
from typing import Literal
from .node import Node
from mlkit.exc import NotFitted


class DTClassifier:
    """
    Decision Tree Classifier.

    A non-parametric supervised learning method used for classify.
    The goal is to create a model that predicts the class of a target variable
    by learning simple decision rules inferred from the data features.

    Supports two impurity measures for splitting:

    Supports two impurity measures for splitting:

    **Gini Impurity**:
        Gini(S) = 1 - Σ(pᵢ²)

        where pᵢ is the proportion of samples belonging to class i.
        Lower Gini indicates purer nodes. Ranges from 0 (pure) to 0.5 (binary).

    **Entropy and Information Gain**:
        Entropy(S) = -Σ(pᵢ × log₂(pᵢ))

        Information Gain = Entropy(parent) - Weighted_Entropy(children)

        where Weighted_Entropy = (n_left/n) × Entropy(left) + (n_right/n) × Entropy(right).
        Higher Information Gain indicates better split quality. IG ranges from 0 to 1.

    Parameters
    ----------
    depth : int, default=5
        Maximum depth of the tree. Controls tree complexity and prevents
        overfitting by limiting the number of levels from root to leaf.

    min_sample : int, default=5
        Minimum number of samples required to split an internal node.
        Splits with fewer samples in either child are rejected.

    algorithm : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split:
        - 'gini': Gini impurity (minimized during split search)
        - 'entropy': Information gain (maximized during split search)

    Raises
    ------
    NotFitted
        If prediction is attempted before fitting the model.

    Examples
    --------
    >>> from mlkit.classify import DTClassifier
    >>> import numpy as np
    >>>
    >>> X = np.array([[1, 2], [2, 3], [3, 1], [4, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>>
    >>> clf = DTClassifier(depth=3, min_sample=2, algorithm='gini')
    >>> clf.fit(X, y)
    >>> predictions = clf.predict(X)
    >>> print(predictions)
    [0 0 1 1]
    """

    def __init__(
            self,
            depth: int = 5,
            min_sample: int = 5,
            algorithm: Literal['gini', 'entropy'] = 'gini'
    ):
        """
        Initialize the Decision Tree Classifier.

        Parameters
        ----------
        depth : int, default=5
            Maximum depth of the decision tree. Larger values increase model
            complexity and risk of overfitting.

        min_sample : int, default=5
            Minimum number of samples required in each child node for a split
            to be considered valid.

        algorithm : {'gini', 'entropy'}, default='gini'
            Impurity measure for split evaluation:
            - 'gini': Minimizes Gini impurity (1 - Σpᵢ²)
            - 'entropy': Maximizes Information Gain (H_parent - H_weighted_children)
        """

        self.root: Node | None = None
        self.max_depth = depth
        self.min_sample = min_sample
        self.algorithm = algorithm
        self.__fitted = False

    def gini(self, y: np.ndarray) -> float:
        """
        Compute Gini impurity for a set of labels.

        Gini impurity measures the probability of misclassifying a randomly
        chosen element if it were labeled according to the class distribution.

        Formula:
            Gini(y) = 1 - Σ(pᵢ²)

        where pᵢ = count(class_i) / n_samples.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            Array of class labels.

        Returns
        -------
        float
            Gini impurity value in range [0, 0.5] for binary classify,
            where 0 indicates a perfectly pure node.
        """

        n = y.shape[0]
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return 1 - np.sum(probs ** 2)

    def weighted_gini(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Compute weighted Gini impurity for a binary split.

        Averages child node Gini impurities weighted by their sample sizes
        to evaluate overall split quality. Lower values indicate better splits.

        Formula:
            Weighted_Gini = (n_left/n) × Gini(left) + (n_right/n) × Gini(right)

        Parameters
        ----------
        y_left : np.ndarray, shape (n_left,)
            Labels for samples in the left child node.

        y_right : np.ndarray, shape (n_right,)
            Labels for samples in the right child node.

        Returns
        -------
        float
            Weighted Gini impurity of the split.
        """

        n = y_left.shape[0] + y_right.shape[0]
        n_left, n_right = y_left.shape[0], y_right.shape[0]
        weighted_gini = (n_left / n) * self.gini(y_left) \
            + (n_right / n) * self.gini(y_right)
        return weighted_gini

    def entropy(self, y: np.ndarray) -> float:
        """
        Compute Shannon entropy for a set of labels.

        Entropy measures the uncertainty or impurity in the label distribution.
        Higher entropy indicates more mixed classes.

        Formula:
            Entropy(y) = -Σ(pᵢ × log₂(pᵢ))

        where pᵢ = count(class_i) / n_samples.
        A small epsilon (1e-12) is added to avoid log(0).

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            Array of class labels.

        Returns
        -------
        float
            Entropy value in range [0, log₂(k)], where k is the number
            of classes. Value 0 indicates perfect purity.
        """

        _, counts = np.unique(y, return_counts=True)
        probs = counts / y.shape[0]
        entropy = - np.sum(probs * np.log2(probs + 1e-12))
        return entropy

    def weighted_entropy(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Compute weighted entropy for a binary split.

        Calculates the average entropy of child nodes weighted by their
        sample proportions. Lower values indicate better splits.

        Formula:
            Weighted_Entropy = (n_left/n) × Entropy(left) + (n_right/n) × Entropy(right)

        Parameters
        ----------
        y_left : np.ndarray, shape (n_left,)
            Labels for samples in the left child node.

        y_right : np.ndarray, shape (n_right,)
            Labels for samples in the right child node.

        Returns
        -------
        float
            Weighted entropy of the split.
        """

        n = y_left.shape[0] + y_right.shape[0]
        return (y_left.shape[0] / n) * self.entropy(y_left) \
            + (y_right.shape[0] / n) * self.entropy(y_right)

    def information_gain(self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Compute Information Gain for a binary split.

        Information Gain measures the reduction in entropy achieved by
        splitting the parent node into two child nodes. Higher values
        indicate that the split creates purer children.

        Formula:
            IG = Entropy(parent) - Weighted_Entropy(left, right)

            where:
                Weighted_Entropy = (n_left/n) × Entropy(left) + (n_right/n) × Entropy(right)

        Parameters
        ----------
        y_parent : np.ndarray, shape (n_samples,)
            Labels of the parent node before splitting.

        y_left : np.ndarray, shape (n_left,)
            Labels of samples in the left child node.

        y_right : np.ndarray, shape (n_right,)
            Labels of samples in the right child node.

        Returns
        -------
        float
            Information gain value. Higher is better. Ranges from 0 (no gain)
            to Entropy(parent) (perfect split).
        """

        return self.entropy(y_parent) - self.weighted_entropy(y_left, y_right)

    def best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find the optimal feature and threshold to split the data.

        Iterates through all features and their unique values as candidate
        split points. Evaluates each split using the selected impurity
        measure (Gini or Entropy/Information Gain).

        Search strategy:
            - For 'gini': Minimize weighted Gini impurity
            - For 'entropy': Maximize Information Gain

        Only considers splits where both children have at least
        ``min_sample`` samples.

                Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix of training samples.

        y : np.ndarray, shape (n_samples,)
            Target labels corresponding to each sample.

        Returns
        -------
        tuple : (best_feature, best_threshold)
            best_feature : int or None
                Index of the best feature for splitting.
                None if no valid split is found.

            best_threshold : float or None
                Threshold value for the best split.
                None if no valid split is found.


        Notes
        -----
        Candidate thresholds are computed as midpoints between consecutive
        unique feature values: thresholds = (unique[:-1] + unique[1:]) / 2.
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
        Recursively build the decision tree from training data.

        Creates tree nodes by finding the best split at each step and
        recursing on the resulting partitions. Stops when:

        - All samples belong to the same class (pure node)
        - Maximum depth is reached
        - No valid split can be found
        - Best feature or threshold is None

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix of the current node's samples.

        y : np.ndarray, shape (n_samples,)
            Target labels for the current node's samples.

        depth : int
            Current depth in the tree. Root starts at depth 0.

        Returns
        -------
        Node
            Root node of the constructed subtree. Leaf nodes contain
            only ``value`` (predicted class), while internal nodes
            also contain ``feature``, ``threshold``, ``left``, and ``right``.

        Notes
        -----
        This method is called recursively. The recursion depth is bounded
        by ``max_depth`` parameter to prevent infinite growth.
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DTClassifier':
        """
        Build the decision tree classifier from training data.

        Constructs the tree structure by recursively partitioning the
        feature space. The tree is then used for making predictions.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training feature matrix. Each row represents a sample
            and each column represents a feature.

        y : np.ndarray, shape (n_samples,)
            Target class labels. Values should be integers representing
            different classes (e.g., 0, 1, 2 for multi-class).

        Returns
        -------
        self : DTClassifier
            Fitted classifier instance.

        Raises
        ------
        ValueError
            If X and y have incompatible shapes or if input arrays are empty.

        Examples
        --------
        >>> clf = DTClassifier(depth=3, algorithm='gini')
        >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> y = np.array([0, 1])
        >>> clf.fit(X, y)
        <DTClassifier object>
        """

        self.root = self.build_tree(X, y, 0)
        self.__fitted = True
        return self

    def determine_label(self, x: np.ndarray) -> int | None:
        """
        Predict class label for a single sample.

        Traverses the tree from root to leaf, following splits based on
        feature thresholds until reaching a leaf node.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)
            A single sample to classify.

        Returns
        -------
        int or None
            Predicted class label from the leaf node's majority class.
            Returns None if the tree is empty (no root node).

        Notes
        -----
        Internal helper method used by ``predict``. At each internal node,
        the sample's feature value is compared against the node's threshold:
        - If ``x[feature] >= threshold``, goes to right child
        - Otherwise, goes to left child
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
        Predict class labels for samples in X.

        Each sample is passed through the tree structure until reaching
        a leaf node. The leaf's majority class is returned as the prediction.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix of samples to predict.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class labels for each input sample.

        Raises
        ------
        NotFitted
            If the model has not been fitted yet (``fit`` must be called first).

        Examples
        --------
        >>> clf = DTClassifier()
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
        >>> print(predictions.shape)
        (n_samples,)
        """

        if not self.__fitted:
            raise NotFitted('DecisionTree not fitted yet')
        y = np.zeros(shape=X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            y[i] = self.determine_label(X[i])
        return y