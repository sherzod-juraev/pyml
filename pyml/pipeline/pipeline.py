"""Chain of transformers ending in a model, used as one estimator."""

from typing import Any, Self

import numpy.typing as npt

from ..core.base.estimator import BaseEstimator
from ..core.exceptions import InvalidParameterError, ValidationError
from ..core.protocols import (
    ClustererLike,
    PredictableClustererLike,
    SupervisedLike,
    SupportsParams,
    SupportsPredictProba,
    TransformerLike,
)

type _Step = tuple[str, TransformerLike | SupervisedLike | ClustererLike | PredictableClustererLike]


class Pipeline(BaseEstimator):
    """Chain transformers and a final model into one estimator.

    .. versionadded:: 0.6.0

    ``fit`` fits every transformer in order, each on the output of the one
    before, then fits the last step on the final output. ``predict``,
    ``predict_proba`` and ``score`` pass the data through the already fitted
    transformers with ``transform`` and hand the result to the last step. The
    transformers are fitted in place, so the objects in ``steps`` are the fitted
    ones.

    The last step is a supervised model (regressor or classifier) or a
    clusterer. Which of the methods work depends on it: ``predict`` needs a
    supervised model or a clusterer that can predict, ``predict_proba`` needs a
    model that has it, and ``score`` needs a supervised model.

    Parameters
    ----------
    steps : list of tuple
        The steps in the order they run. Each tuple holds a unique name and the
        estimator. The name must be non-empty and must not contain ``"__"``.
        Every estimator except the last must be a transformer, the last must be
        a supervised model or a clusterer.

    Attributes
    ----------
    steps : list of tuple
        The steps as given. Fitting changes the estimators inside, not the list.
    is_fitted_ : bool
        True after ``fit``.

    Notes
    -----
    The pipeline checks only the structure of ``steps``. The data, including the
    number of features, is validated by each step itself, so a wrong input is
    reported by the step that meets it first. The structure is checked again in
    every method, because ``steps`` can be replaced with ``set_params``.
    """

    def __init__(self, steps: list[_Step]) -> None:
        """Create a pipeline from its steps.

        Parameters
        ----------
        steps : list of tuple
            The steps in the order they run. The steps are validated when the
            pipeline is fitted or used, not here.
        """
        super().__init__()
        self.steps: list[_Step] = steps

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any] | None = None, /) -> Self:
        """Fit the transformers and the last step.

        Parameters
        ----------
        X : ndarray
            Training data of shape (n_samples, n_features).
        y : ndarray or None, optional
            Targets of shape (n_samples,). Required when the last step is a
            supervised model, and must be None when it is a clusterer.

        Returns
        -------
        Pipeline
            The fitted pipeline.

        Raises
        ------
        InvalidParameterError
            If ``steps`` is not a valid list of steps.
        ValidationError
            If y is missing for a supervised last step, or given for a
            clusterer. Errors of the steps themselves for invalid data or
            parameters are raised as they are.
        """
        transformers, final = self._validate_steps()
        if isinstance(final, SupervisedLike):
            if y is None:
                raise ValidationError(
                    f"The last step {type(final).__name__} is supervised, so y is required."
                )
            Xt = self._apply_transformers(transformers, X, True)
            final.fit(Xt, y)
        else:
            if y is not None:
                raise ValidationError(
                    f"The last step {type(final).__name__} is a clusterer, so y must be None."
                )
            Xt = self._apply_transformers(transformers, X, True)
            final.fit(Xt)
        self.is_fitted_ = True
        return self

    def predict(self, X: npt.NDArray[Any], /) -> npt.NDArray[Any]:
        """Transform the data and predict with the last step.

        Parameters
        ----------
        X : ndarray
            Data of shape (n_samples, n_features).

        Returns
        -------
        ndarray
            The prediction of the last step, a target estimate or a cluster
            label for each sample.

        Raises
        ------
        NotFittedError
            If the pipeline has not been fitted yet.
        InvalidParameterError
            If ``steps`` is not valid, or the last step has no ``predict``
            (a clusterer such as DBSCAN).
        """
        self._check_is_fitted()
        transformers, final = self._validate_steps()
        if isinstance(final, SupervisedLike | PredictableClustererLike):
            return final.predict(self._apply_transformers(transformers, X, False))
        raise InvalidParameterError(f"The last step {type(final).__name__} has no predict method.")

    def predict_proba(self, X: npt.NDArray[Any], /) -> npt.NDArray[Any]:
        """Transform the data and predict probabilities with the last step.

        Parameters
        ----------
        X : ndarray
            Data of shape (n_samples, n_features).

        Returns
        -------
        ndarray
            The probabilities of the last step. Their shape is the one the last
            step returns, which differs between the models.

        Raises
        ------
        NotFittedError
            If the pipeline has not been fitted yet.
        InvalidParameterError
            If ``steps`` is not valid, or the last step has no ``predict_proba``.
        """
        self._check_is_fitted()
        transformers, final = self._validate_steps()
        if isinstance(final, SupportsPredictProba):
            return final.predict_proba(self._apply_transformers(transformers, X, False))
        raise InvalidParameterError(
            f"The last step {type(final).__name__} has no predict_proba method."
        )

    def score(self, X: npt.NDArray[Any], y: npt.NDArray[Any], /) -> float:
        """Transform the data and score the last step.

        Parameters
        ----------
        X : ndarray
            Data of shape (n_samples, n_features).
        y : ndarray
            True targets of shape (n_samples,).

        Returns
        -------
        float
            The score of the last step: R^2 for a regressor, accuracy for a
            classifier.

        Raises
        ------
        NotFittedError
            If the pipeline has not been fitted yet.
        InvalidParameterError
            If ``steps`` is not valid, or the last step is not a supervised
            model.
        """
        self._check_is_fitted()
        transformers, final = self._validate_steps()
        if isinstance(final, SupervisedLike):
            return final.score(self._apply_transformers(transformers, X, False), y)
        raise InvalidParameterError(f"The last step {type(final).__name__} has no score method.")

    def _apply_transformers(
        self, transformers: list[TransformerLike], X: npt.NDArray[Any], fit: bool, /
    ) -> npt.NDArray[Any]:
        """Run the data through the transformers, fitting them if ``fit`` is True."""
        Xt = X
        for transformer in transformers:
            Xt = transformer.fit_transform(Xt) if fit else transformer.transform(Xt)
        return Xt

    def _validate_steps(
        self,
    ) -> tuple[list[TransformerLike], SupervisedLike | ClustererLike]:
        """Check the structure of ``steps`` and split it into transformers and last step.

        Returns
        -------
        tuple
            The transformers (every step but the last) and the last step.

        Raises
        ------
        InvalidParameterError
            If ``steps`` is not a non-empty list of unique-named
            ``(name, estimator)`` tuples, a step before the last is not a
            transformer, or the last step is neither a supervised model nor a
            clusterer.
        """
        steps = self.steps
        if not isinstance(steps, list) or not steps:
            raise InvalidParameterError(
                "steps must be a non-empty list of (name, estimator) tuples."
            )

        names: set[str] = set()
        for index, step in enumerate(steps):
            if not isinstance(step, tuple) or len(step) != 2 or not isinstance(step[0], str):
                raise InvalidParameterError(
                    f"steps[{index}] must be a (name, estimator) tuple with a str name, "
                    f"got {step!r}."
                )
            name = step[0]
            if not name or "__" in name:
                raise InvalidParameterError(
                    f"Step name {name!r} must be non-empty and must not contain '__'."
                )
            if name in names:
                raise InvalidParameterError(f"Step names must be unique, {name!r} is repeated.")
            names.add(name)

        transformers: list[TransformerLike] = []
        for name, transformer in steps[:-1]:
            if not isinstance(transformer, TransformerLike):
                raise InvalidParameterError(
                    f"Every step except the last must be a transformer, but step {name!r} "
                    f"({type(transformer).__name__}) has no transform or fit_transform."
                )
            transformers.append(transformer)

        name, final = steps[-1]
        if not isinstance(final, SupervisedLike | ClustererLike):
            raise InvalidParameterError(
                f"The last step must be a supervised model or a clusterer, but step {name!r} "
                f"({type(final).__name__}) is neither."
            )
        return transformers, final

    def __getitem__(self, item: str | int) -> SupportsParams:
        """Return the estimator of one step, found by its name or its position.

        Parameters
        ----------
        item : str or int
            The name of the step, or its position. A negative position counts
            from the end, so ``-1`` is the last step. Slices are not supported.

        Returns
        -------
        SupportsParams
            The estimator of the step, not the ``(name, estimator)`` tuple. It is
            the object stored in ``steps``, not a copy.

        Raises
        ------
        InvalidParameterError
            If ``steps`` is not valid.
        TypeError
            If ``item`` is not a ``str`` or an ``int``. A ``bool`` is rejected too.
        KeyError
            If no step has the given name.
        IndexError
            If the position is outside the steps.
        """
        self._validate_steps()
        if isinstance(item, bool) or not isinstance(item, str | int):
            raise TypeError(f"Pipeline indices must be str or int, not {type(item).__name__}.")

        if isinstance(item, str):
            for name, estimator in self.steps:
                if item == name:
                    return estimator
            available = ", ".join(repr(name) for name, _ in self.steps)
            raise KeyError(f"No step named {item!r}. Available steps: {available}.")

        index = item + len(self) if item < 0 else item
        if not 0 <= index < len(self):
            raise IndexError(
                f"Step index {item} is out of range for a pipeline with {len(self)} steps."
            )
        return self.steps[index][1]

    def __len__(self) -> int:
        """Return the number of steps.

        The steps are not validated, so this works on any ``steps`` that has a
        length.

        Returns
        -------
        int
            The length of ``steps``.
        """
        return len(self.steps)
