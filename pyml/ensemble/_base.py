"""Shared base class for random forests.

Holds the construction loop that every forest has in common: validating the
hyper-parameters, drawing a seed and a bootstrap sample for each tree, growing
the trees and storing them in ``estimators_``. Subclasses decide which kind of
tree the forest is made of and how the predictions of the trees are combined,
a vote of class probabilities for classification and an average for
regression.
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Literal, cast, get_args

import numpy as np

from ..core.dtypes import ClassificationTarget, FeatureMatrix, RegressionTarget
from ..core.exceptions import InvalidParameterError
from ..core.param_checks import check_bool, check_choice, check_integer
from ..core.protocols import _ForestTree

_MaxFeatures = Literal["sqrt"]
"""Names accepted by max_features besides an int or None."""


class _ForestBase[TreeT: _ForestTree](ABC):
    """Abstract base class for random forests.

    A random forest combines many decision trees that are each grown on a
    different view of the data. Two kinds of randomness make the trees differ:
    every tree is fitted on its own bootstrap sample of the rows when
    ``bootstrap`` is True, and, when ``max_features`` is smaller than the
    number of features, every node of a tree considers only a random subset of
    the features as split candidates. Combining the predictions of such
    decorrelated trees lowers the variance of a single deep tree and leaves its
    bias nearly unchanged.

    One ``random_state`` makes the whole forest reproducible. A generator
    seeded with it first draws a seed for every tree, which the tree uses for
    its feature sampling, and then the bootstrap indices of each tree in turn.

    Subclasses implement ``_make_tree`` to say which kind of tree the forest
    is made of, override ``_before_build`` when the trees need something from
    the whole target, such as the classes of a classifier, and combine the
    predictions of the trees themselves.

    Attributes
    ----------
    estimators_ : list
        The fitted trees, in the order they were grown, of length
        ``n_estimators``.

    .. note::
        The trees are fitted one after another, so the fit time grows
        linearly with ``n_estimators`` on top of the cost of a single tree.
    """

    def __init__(
        self,
        max_depth: int,
        n_estimators: int = 100,
        min_samples: int = 2,
        min_impurity_decrease: float = 1e-7,
        max_features: int | _MaxFeatures | None = "sqrt",
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the forest and the tree-growth hyper-parameters.

        Parameters
        ----------
        max_depth : int
            Maximum depth of every tree; must be non-negative. Passed to each
            tree unchanged.
        n_estimators : int, optional
            Number of trees in the forest; must be at least 1. A forest of a
            single tree is allowed. Defaults to 100.
        min_samples : int, optional
            Minimum number of samples required to consider splitting a node
            of a tree; must be at least 1. Defaults to 2.
        min_impurity_decrease : float, optional
            Minimum impurity gain required to accept a split in a tree; must
            be non-negative. Defaults to 1e-7.
        max_features : int, {"sqrt"} or None, optional
            Number of features drawn at random, without replacement, as split
            candidates at each node of every tree. An int must be at least 1
            and at most the number of features seen in fit. "sqrt" uses the
            integer part of the square root of the number of features. None
            uses every feature, so only the bootstrap decorrelates the trees.
            The default is the usual choice for classification; regression
            forests often use about a third of the features instead, which is
            set with an int. Defaults to "sqrt".
        bootstrap : bool, optional
            If True, each tree is grown on a bootstrap sample: as many rows as
            the training set, drawn with replacement. If False, every tree
            sees the whole training set. Defaults to True.
        random_state : int or None, optional
            Seed that makes the forest reproducible; must be non-negative. It
            controls the bootstrap samples and the feature sampling of the
            trees. None draws fresh randomness on every fit. Defaults to None.
        """
        super().__init__()
        self.max_depth: int = max_depth
        self.n_estimators: int = n_estimators
        self.min_samples: int = min_samples
        self.min_impurity_decrease: float = min_impurity_decrease
        self.max_features: int | _MaxFeatures | None = max_features
        self.bootstrap: bool = bootstrap
        self.random_state: int | None = random_state

    def _validate_params(self) -> None:
        """Validate the forest and tree hyper-parameters.

        The forest checks n_estimators and bootstrap itself. The five
        parameters every tree receives are checked by a prototype tree built
        with ``_make_tree``, so the tree stays the single source of its own
        bounds. A string max_features is checked against the allowed names
        here and the prototype gets None in its place. Checks that need the
        data, such as max_features against the number of features, belong in
        ``_fit``.

        Raises
        ------
        InvalidParameterError
            If n_estimators is not an integer of at least 1, bootstrap is not
            a bool, max_features is a string other than "sqrt", or the
            prototype tree rejects max_depth, min_samples,
            min_impurity_decrease, max_features or random_state.
        """
        # This class sits in front of the family base in the MRO, so super() reaches
        # DataValidatorMixin; mypy cannot see that from a plain ABC base.
        cast(Any, super())._validate_params()
        check_integer("n_estimators", self.n_estimators, minimum=1)
        check_bool("bootstrap", self.bootstrap)
        tree_max_features: int | None
        if isinstance(self.max_features, str):
            check_choice("max_features", self.max_features, get_args(_MaxFeatures))
            tree_max_features = None
        else:
            tree_max_features = self.max_features
        self._make_tree(self.random_state, tree_max_features)._validate_params()

    def _resolve_max_features(self, n_features: int, /) -> int | None:
        """Turn max_features into the value handed to every tree.

        Parameters
        ----------
        n_features : int
            Number of features seen in fit.

        Returns
        -------
        int or None
            ``isqrt(n_features)`` for "sqrt", otherwise max_features as given
            (an int, or None for every feature).
        """
        if isinstance(self.max_features, str):
            return math.isqrt(n_features)
        return self.max_features

    @abstractmethod
    def _make_tree(self, random_state: int | None, max_features: int | None, /) -> TreeT:
        """Build one unfitted decision tree with the forest's tree parameters.

        Parameters
        ----------
        random_state : int or None
            Seed handed to the tree for its per-node feature sampling.
        max_features : int or None
            Number of candidate features per node, already resolved from the
            forest's max_features; None uses every feature.

        Returns
        -------
        _DecisionTreeBase
            An unfitted tree of the type the forest is made of, created with
            max_depth, min_samples, min_impurity_decrease and the given
            max_features and random_state.
        """

    def _before_build(self, y: ClassificationTarget | RegressionTarget, /) -> None:  # noqa: B027
        """Prepare the forest for growing its trees.

        Called once per fit, after the checks that need the data and before
        the first tree is grown, so what a subclass sets here is not changed
        by a fit that those checks reject. A subclass overrides it to set what
        it needs from the whole target before the bootstrap samples are drawn,
        such as the classes of a classifier. The default does nothing on
        purpose, so it is not abstract.

        Parameters
        ----------
        y : ClassificationTarget or RegressionTarget
            Target values of shape (n_samples,).
        """

    def _fit(self, X: FeatureMatrix, y: ClassificationTarget | RegressionTarget, /) -> None:
        """Grow the trees of the forest.

        One generator seeded with random_state first draws a seed for every
        tree. Then, for each tree in turn, the bootstrap indices are drawn when
        bootstrap is True, the tree is created with its seed, fitted on the
        sample and stored in ``estimators_``.

        Parameters
        ----------
        X : FeatureMatrix
            Training data of shape (n_samples, n_features).
        y : ClassificationTarget or RegressionTarget
            Target values of shape (n_samples,).

        Raises
        ------
        InvalidParameterError
            If max_features, once resolved, exceeds the number of features
            in X.
        """
        n_samples, n_features = X.shape
        resolved = self._resolve_max_features(n_features)
        if resolved is not None and resolved > n_features:
            raise InvalidParameterError(
                f"max_features={resolved} cannot exceed the number of features ({n_features})."
            )
        self._before_build(y)
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(np.iinfo(np.int32).max, size=self.n_estimators)
        self.estimators_: list[TreeT] = []
        for i in range(self.n_estimators):
            X_i, y_i = X, y
            if self.bootstrap:
                idx = rng.integers(n_samples, size=n_samples)
                X_i, y_i = X[idx], y[idx]
            tree = self._make_tree(int(seeds[i]), resolved)
            tree.fit(X_i, y_i)
            self.estimators_.append(tree)
