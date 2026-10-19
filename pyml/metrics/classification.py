"""Classification evaluation metrics.

Provides accuracy and per-class precision/recall/F1 measures (with
macro, micro, and weighted averaging) for comparing predicted class
labels against ground truth, used by Classifier subclasses' score
method and by test suites for verifying model correctness.
"""

from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

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


def _count_per_class(
    y_true: ClassificationTarget, y_pred: ClassificationTarget, classes: ClassificationTarget, /
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Count the true positives, false positives and false negatives of every class.

    Classes are processed one at a time, so the temporary boolean masks have
    shape (n_samples,) instead of (n_classes, n_samples).

    Parameters
    ----------
    y_true : ClassificationTarget
        Ground truth class labels.
    y_pred : ClassificationTarget
        Predicted class labels.
    classes : ClassificationTarget
        Class labels to count for, of shape (n_classes,).

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        True positives, false positives and false negatives, each of shape
        (n_classes,) and ordered like `classes`.
    """
    tp = np.empty(classes.shape[0], dtype=np.intp)
    fp = np.empty(classes.shape[0], dtype=np.intp)
    fn = np.empty(classes.shape[0], dtype=np.intp)
    for index, label in enumerate(classes):
        is_true = y_true == label
        is_pred = y_pred == label
        tp[index] = np.sum(is_true & is_pred)
        fp[index] = np.sum(~is_true & is_pred)
        fn[index] = np.sum(is_true & ~is_pred)
    return tp, fp, fn


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
    tp, fp, _ = _count_per_class(y_true, y_pred, y_uniq)
    denom = tp + fp
    precisions = np.zeros_like(denom, dtype=float)
    np.divide(tp, denom, out=precisions, where=denom > 0)
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
    tp, _, fn = _count_per_class(y_true, y_pred, y_uniq)
    denom = tp + fn
    recalls = np.zeros_like(denom, dtype=float)
    np.divide(tp, denom, out=recalls, where=denom > 0)
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
    tp, fp, fn = _count_per_class(y_true, y_pred, y_uniq)

    denom_p = tp + fp
    precisions = np.divide(
        tp,
        denom_p,
        out=np.zeros_like(denom_p, dtype=float),
        where=denom_p > 0,
    )

    denom_r = tp + fn
    recalls = np.divide(
        tp,
        denom_r,
        out=np.zeros_like(denom_r, dtype=float),
        where=denom_r > 0,
    )

    denom_f = precisions + recalls
    f1_per_class = np.divide(
        2 * precisions * recalls,
        denom_f,
        out=np.zeros_like(denom_f, dtype=float),
        where=denom_f > 0,
    )
    # macro
    if average == "macro":
        return float(np.mean(f1_per_class))
    # weighted
    return float(np.sum(f1_per_class * (class_counts / y_true.shape[0])))
