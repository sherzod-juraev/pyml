import numpy as np
from typing import Literal
from .node import Node
from mlkit.exc import NotFitted


class DTRegressor:

    def __init__(
            self,
            depth: int = 5,
            min_sample: int = 5,
            algorithm: Literal['mse', 'mae'] = 'mse'
    ):

        self.root: Node | None = None
        self.max_depth = depth
        self.min_sample = min_sample
        self.algorithm = algorithm
        self.__fitted = False

    def mse(self, y: np.ndarray, y_pred: np.ndarray) -> float:

        return np.mean((y - y_pred) ** 2)

    def mae(self, y: np.ndarray, y_pred: np.ndarray) -> float:

        return np.mean(np.abs(y - y_pred))

    def best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:

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

        y_uniq, counts = np.unique(y, return_counts=True)
        if y_uniq.shape[0] == 1 or depth >= self.max_depth:
            return Node(value=np.mean(y))
        best_feature, best_threshold = self.best_split(X, y)
        if best_feature is None or best_threshold is None:
            return Node(value=np.mean(y))
        left_mask = X[:, best_feature] < best_threshold
        right_mask = X[:, best_feature] >= best_threshold
        left_node = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        node = Node(best_feature, best_threshold, left_node, right_node)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DTRegressor':

        self.root = self.build_tree(X, y, 0)
        self.__fitted = True
        return self

    def determine_label(self, x: np.ndarray) -> int:

        cur_node = self.root
        while cur_node is not None:
            if cur_node.value is not None:
                return cur_node.value
            elif x[cur_node.feature] >= cur_node.threshold:
                cur_node = cur_node.right
            else:
                cur_node = cur_node.left

    def predict(self, X: np.ndarray) -> np.ndarray:

        if not self.__fitted:
            raise NotFitted('DecisionTree not fitted yet')
        y = np.zeros(shape=X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            y[i] = self.determine_label(X[i])
        return y