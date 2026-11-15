"""Random forest classifier averaging the class probabilities of decision trees."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..core.base import Classifier
from ..core.dtypes import ClassificationTarget, FeatureMatrix, RegressionTarget
from ..tree import DecisionTreeClassifier
from ._base import _ForestBase


class RandomForestClassifier(_ForestBase[DecisionTreeClassifier], Classifier):
    r"""Random forest classifier averaging the class probabilities of decision trees.

    .. versionadded:: 0.5.0

    The forest grows ``n_estimators`` decision trees. Each tree is fitted on
    its own bootstrap sample of the training rows, and at every node it
    considers only a random subset of ``max_features`` features as split
    candidates. These two sources of randomness make the trees differ from one
    another, so averaging them lowers the variance of a single deep tree while
    its bias stays nearly the same.

    Prediction is a soft vote. For a sample :math:`x`, tree :math:`b` gives
    the share :math:`p_b(c \mid x)` of each class :math:`c` among the training
    samples in the leaf that :math:`x` reaches, and the forest averages these
    shares over the :math:`B` trees:

    .. math::
        P(c \mid x) = \frac{1}{B} \sum_{b=1}^{B} p_b(c \mid x)

    ``predict`` returns the class with the largest average, and on a tie the
    smallest label. The trees are fully reproducible from ``random_state``.

    Parameters
    ----------
    max_depth : int
        Maximum depth of every tree; must be non-negative. Passed to each tree
        unchanged.
    n_estimators : int, optional
        Number of trees in the forest; must be at least 1. A forest of a single
        tree is allowed. Defaults to 100.
    min_samples : int, optional
        Minimum number of samples required to consider splitting a node of a
        tree; must be at least 1. Defaults to 2.
    min_impurity_decrease : float, optional
        Minimum impurity gain required to accept a split in a tree; must be
        non-negative. Defaults to 1e-7.
    max_features : int, {"sqrt"} or None, optional
        Number of features drawn at random, without replacement, as split
        candidates at each node of every tree. An int must be at least 1 and at
        most the number of features seen in fit. "sqrt" uses the integer part
        of the square root of the number of features. None uses every feature,
        so only the bootstrap decorrelates the trees. Defaults to "sqrt".
    bootstrap : bool, optional
        If True, each tree is grown on a bootstrap sample: as many rows as the
        training set, drawn with replacement. If False, every tree sees the
        whole training set. Defaults to True.
    random_state : int or None, optional
        Seed that makes the forest reproducible; must be non-negative. It
        controls the bootstrap samples and the feature sampling of the trees.
        None draws fresh randomness on every fit. Defaults to None.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The fitted trees, in the order they were grown, of length
        ``n_estimators``.
    classes_ : ClassificationTarget
        Sorted class labels seen in fit, of shape (n_classes,). The columns of
        ``predict_proba`` follow this order.


    .. note::
        The trees are fitted one after another, so the fit time grows linearly
        with ``n_estimators`` on top of the cost of a single tree, whose
        splitting is exhaustive. Prediction visits every tree as well. A tree
        whose bootstrap sample misses a class contributes zero to that class.
        With ``n_estimators=1``, ``bootstrap=False`` and ``max_features=None``
        the forest is a single ``DecisionTreeClassifier``.

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pyml.ensemble import RandomForestClassifier

        rng = np.random.default_rng(42)
        X0 = rng.normal(loc=(-1, -1), scale=1.2, size=(60, 2))
        X1 = rng.normal(loc=(1, 1), scale=1.2, size=(60, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 60 + [1] * 60)

        model = RandomForestClassifier(max_depth=4, n_estimators=25, random_state=42).fit(X, y)

        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 80),
            np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 80),
        )
        grid_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, grid_proba.reshape(xx.shape), levels=10, alpha=0.4, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("RandomForestClassifier probability of class 1 (25 trees)")
    """

    def _make_tree(
        self, random_state: int | None, max_features: int | None, /
    ) -> DecisionTreeClassifier:
        """Build one unfitted decision tree for the forest.

        Parameters
        ----------
        random_state : int or None
            Seed of the tree, used to draw the candidate features at each node.
        max_features : int or None
            Number of candidate features per node, already resolved from the
            max_features of the forest; None uses every feature.

        Returns
        -------
        DecisionTreeClassifier
            An unfitted tree created with the max_depth, min_samples and
            min_impurity_decrease of the forest and the given max_features and
            random_state.
        """
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples=self.min_samples,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=max_features,
            random_state=random_state,
        )

    def _before_build(self, y: ClassificationTarget | RegressionTarget, /) -> None:
        """Set the classes of the forest from the whole target.

        The classes are taken from all of y, before the bootstrap samples are
        drawn, because a bootstrap sample can miss a class and its tree would
        then know fewer classes than the forest. The columns of
        ``predict_proba`` follow this order.

        Parameters
        ----------
        y : ClassificationTarget or RegressionTarget
            Class labels of shape (n_samples,).
        """
        self.classes_: ClassificationTarget = np.unique(y)

    def _predict_proba(self, X: FeatureMatrix, /) -> npt.NDArray[np.floating[Any]]:
        """Average the class probabilities of the trees.

        Does no input validation: it is called by ``predict_proba`` and
        ``predict`` after the checks. Each tree is asked for its leaf shares
        and its columns are added to the columns of the forest that hold the
        same classes, because a tree whose bootstrap sample missed a class has
        fewer columns than the forest.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            Class probabilities of shape (n_samples, n_classes). The columns
            follow ``classes_`` and every row sums to 1.
        """
        n_samples = X.shape[0]
        total = np.zeros((n_samples, self.classes_.shape[0]))
        for tree in self.estimators_:
            proba = tree._predict_proba(X)
            cols = np.searchsorted(self.classes_, tree.classes_)
            total[:, cols] += proba
        return total / self.n_estimators

    def predict_proba(self, X: FeatureMatrix, /) -> npt.NDArray[np.floating[Any]]:
        """Predict the class probabilities for each sample.

        The probabilities of a sample are the average, over the trees, of the
        shares of the classes among the training samples in the leaf the
        sample reaches. The class with the largest average is the one
        ``predict`` returns, and on a tie both give the smallest label.
        Validates the input, then delegates to ``_predict_proba``.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            Class probabilities of shape (n_samples, n_classes). The columns
            follow ``classes_`` and every row sums to 1.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        ValueError
            If X is not 2D, or contains NaN or infinite values.
        TypeError
            If X's dtype is not floating-point.
        ShapeMismatchError
            If X has a different number of features than seen in fit.
        """
        self._check_is_fitted()
        self._validate_X(X)
        self._check_n_features(X)
        return self._predict_proba(X)

    def _predict(self, X: FeatureMatrix, /) -> ClassificationTarget:
        """Predict the class with the largest average probability.

        Parameters
        ----------
        X : FeatureMatrix
            Input data of shape (n_samples, n_features).

        Returns
        -------
        ClassificationTarget
            Predicted class label for each sample, of shape (n_samples,). On a
            tie the smallest label is returned.
        """
        proba = self._predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
