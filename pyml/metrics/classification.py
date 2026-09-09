"""Classification evaluation metrics.

Provides accuracy and per-class precision/recall/F1 measures (with
macro, micro, and weighted averaging) for comparing predicted class
labels against ground truth, used by Classifier subclasses' score
method and by test suites for verifying model correctness.
"""

from typing import Literal, get_args

import numpy as np

from ..core.dtypes import ClassificationTarget
from ..core.exceptions import InvalidParameterError, ShapeMismatchError

AverageMethod = Literal["macro", "micro", "weighted"]
_VALID_AVERAGE = get_args(AverageMethod)


def _validate_classification_inputs(
    y_true: ClassificationTarget, y_pred: ClassificationTarget, average: str | None = None, /
) -> None:
    """Validate classification targets and, optionally, an averaging method.

    Parameters
    ----------
    y_true : ClassificationTarget
        Ground truth class labels.
    y_pred : ClassificationTarget
        Predicted class labels.
    average : str or None, optional
        Averaging method to validate. Skipped entirely when None, since
        metrics like accuracy do not use an averaging method.

    Raises
    ------
    ValueError
        If y_true or y_pred is not 1D.
    TypeError
        If y_true or y_pred does not have an integer dtype.
    ShapeMismatchError
        If y_true and y_pred have a different number of samples.
    InvalidParameterError
        If average is given and is not one of the valid averaging methods.
    """
    if y_true.ndim != 1:
        raise ValueError(f"y_true labels expected 1D array, got {y_true.ndim}D array.")
    if not np.issubdtype(y_true.dtype, np.integer):
        raise TypeError(f"y_true labels must be integers, got {y_true.dtype.name}")
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred labels expected 1D array, got {y_pred.ndim}D array.")
    if not np.issubdtype(y_pred.dtype, np.integer):
        raise TypeError(f"y_pred labels must be integers, got {y_pred.dtype.name}")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ShapeMismatchError(
            "Found input variables with inconsistent numbers of samples: "
            f"[{y_true.shape[0]}, {y_pred.shape[0]}]"
        )
    if average is not None and average not in _VALID_AVERAGE:
        raise InvalidParameterError(
            f"Invalid parameter {average} for average. "
            f"Valid parameters are: {', '.join(_VALID_AVERAGE)}"
        )


def accuracy_score(y_true: ClassificationTarget, y_pred: ClassificationTarget, /) -> float:
    r"""Compute the fraction of correctly predicted labels.

    .. math::
        Accuracy = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}(y_i = \hat{y}_i)

    Parameters
    ----------
    y_true : ClassificationTarget
        Ground truth class labels.
    y_pred : ClassificationTarget
        Predicted class labels.

    Returns
    -------
    float
        Fraction of samples where the prediction matches the ground
        truth, between 0.0 and 1.0.
    """
    _validate_classification_inputs(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def precision_score(
    y_true: ClassificationTarget,
    y_pred: ClassificationTarget,
    average: AverageMethod = "weighted",
    /,
) -> float:
    r"""Compute precision, the fraction of positive predictions that are correct.

    For each class c, treating c as the positive class and all other
    classes as negative:

    .. math::
        Precision_c = \frac{TP_c}{TP_c + FP_c}

    Per-class scores are combined according to average:

    .. math::
        Precision_{macro} = \frac{1}{C} \sum_{c=1}^{C} Precision_c

        Precision_{weighted} = \sum_{c=1}^{C} \frac{n_c}{n} \times Precision_c

        Precision_{micro} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}(y_i = \hat{y}_i) = Accuracy

    where C is the number of classes, n is the total number of samples,
    and n_c is the number of samples truly belonging to class c (support).
    Micro-averaged precision is always numerically identical to accuracy,
    since global TP/FP counts across all classes reduce to the same
    fraction of correct predictions.

    Parameters
    ----------
    y_true : ClassificationTarget
        Ground truth class labels.
    y_pred : ClassificationTarget
        Predicted class labels.
    average : {"macro", "micro", "weighted"}, optional
        Averaging method across classes. Defaults to "weighted".

    Returns
    -------
    float
        Precision score, between 0.0 and 1.0.
    """
    _validate_classification_inputs(y_true, y_pred, average)
    # micro
    if average == "micro":
        return accuracy_score(y_true, y_pred)
    y_uniq, class_counts = np.unique(y_true, return_counts=True)
    classes = y_uniq[:, np.newaxis]
    tp = np.sum((classes == y_true) & (classes == y_pred), axis=1)
    fp = np.sum((classes != y_true) & (classes == y_pred), axis=1)
    precisions = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    # macro
    if average == "macro":
        return float(np.mean(precisions))
    # weighted
    return float(np.sum(precisions * (class_counts / y_true.shape[0])))


def recall_score(
    y_true: ClassificationTarget,
    y_pred: ClassificationTarget,
    average: AverageMethod = "weighted",
    /,
) -> float:
    r"""Compute recall, the fraction of actual positives that are correctly identified.

    For each class c, treating c as the positive class and all other
    classes as negative:

    .. math::
        Recall_c = \frac{TP_c}{TP_c + FN_c}

    Per-class scores are combined according to average:

    .. math::
        Recall_{macro} = \frac{1}{C} \sum_{c=1}^{C} Recall_c

        Recall_{weighted} = \sum_{c=1}^{C} \frac{n_c}{n} \times Recall_c

        Recall_{micro} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}(y_i = \hat{y}_i) = Accuracy

    where C is the number of classes, n is the total number of samples,
    and n_c is the number of samples truly belonging to class c (support).
    Micro-averaged recall is always numerically identical to accuracy,
    since global TP/FN counts across all classes reduce to the same
    fraction of correct predictions.

    Parameters
    ----------
    y_true : ClassificationTarget
        Ground truth class labels.
    y_pred : ClassificationTarget
        Predicted class labels.
    average : {"macro", "micro", "weighted"}, optional
        Averaging method across classes. Defaults to "weighted".

    Returns
    -------
    float
        Recall score, between 0.0 and 1.0.
    """
    _validate_classification_inputs(y_true, y_pred, average)
    # micro
    if average == "micro":
        return accuracy_score(y_true, y_pred)
    y_uniq, class_counts = np.unique(y_true, return_counts=True)
    classes = y_uniq[:, np.newaxis]
    tp = np.sum((classes == y_true) & (classes == y_pred), axis=1)
    fn = np.sum((classes == y_true) & (classes != y_pred), axis=1)
    recalls = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    # macro
    if average == "macro":
        return float(np.mean(recalls))
    # weighted
    return float(np.sum(recalls * (class_counts / y_true.shape[0])))


def f1_score(
    y_true: ClassificationTarget,
    y_pred: ClassificationTarget,
    average: AverageMethod = "weighted",
    /,
) -> float:
    r"""Compute the F1 score, the harmonic mean of precision and recall.

    For each class c:

    .. math::
        F1_c = 2 \times \frac{Precision_c \times Recall_c}{Precision_c + Recall_c}

    The harmonic mean penalizes a large imbalance between precision and
    recall more heavily than an arithmetic mean would. Per-class scores
    are combined according to average:

    .. math::
        F1_{macro} = \frac{1}{C} \sum_{c=1}^{C} F1_c

        F1_{weighted} = \sum_{c=1}^{C} \frac{n_c}{n} \times F1_c

        F1_{micro} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}(y_i = \hat{y}_i) = Accuracy

    where C is the number of classes, n is the total number of samples,
    and n_c is the number of samples truly belonging to class c (support).
    Micro-averaged F1 is always numerically identical to accuracy, since
    global TP/FP/FN counts across all classes reduce to the same fraction
    of correct predictions.

    Parameters
    ----------
    y_true : ClassificationTarget
        Ground truth class labels.
    y_pred : ClassificationTarget
        Predicted class labels.
    average : {"macro", "micro", "weighted"}, optional
        Averaging method across classes. Defaults to "weighted".

    Returns
    -------
    float
        F1 score, between 0.0 and 1.0.
    """
    _validate_classification_inputs(y_true, y_pred, average)
    # micro
    if average == "micro":
        return accuracy_score(y_true, y_pred)
    y_uniq, class_counts = np.unique(y_true, return_counts=True)
    classes = y_uniq[:, np.newaxis]
    tp = np.sum((classes == y_true) & (classes == y_pred), axis=1)
    fp = np.sum((classes != y_true) & (classes == y_pred), axis=1)
    fn = np.sum((classes == y_true) & (classes != y_pred), axis=1)
    precisions = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    recalls = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    f1_per_class = np.where(
        (precisions + recalls) > 0, 2 * ((precisions * recalls) / (precisions + recalls)), 0.0
    )
    # macro
    if average == "macro":
        return float(np.mean(f1_per_class))
    # weighted
    return float(np.sum(f1_per_class * (class_counts / y_true.shape[0])))
