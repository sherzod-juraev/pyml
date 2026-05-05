import numpy as np
from typing import Self, Literal
from ...exc import NotFitted

class LogReg:
    """
    Logistic Regression classifier with optional L1/L2 regularization
    trained via Batch Gradient Descent.

    Models the probability that a sample belongs to class 1 using
    the sigmoid function applied to a linear combination of features:

    .. math::

        z = Xw + b

    .. math::

        \\hat{y} = \\sigma(z) = \\frac{1}{1 + e^{-z}}

    Parameters are learned by minimizing Binary Cross-Entropy (BCE) loss.
    BCE is derived from Maximum Likelihood Estimation under a Bernoulli
    distribution:

    .. math::

        J(w, b) = -\\frac{1}{n} \\sum_{i=1}^{n}
        \\left[ y_i \\log(\\hat{y}_i) + (1 - y_i)
        \\log(1 - \\hat{y}_i) \\right]

    When regularization is enabled, a penalty term is added to the
    cost function to prevent overfitting by constraining weight magnitudes.

    **L2 (Ridge) regularization:**

    .. math::

        J_{L2}(w, b) = J(w, b) +
        \\frac{1}{2n} \\sum_{j=1}^{p} w_j^2

    **L1 (Lasso) regularization:**

    .. math::

        J_{L1}(w, b) = J(w, b) +
        \\frac{\\alpha}{n} \\sum_{j=1}^{p} |w_j|

    The bias term :math:`b` is excluded from regularization in both cases,
    as it controls the decision boundary shift and does not contribute
    to overfitting.

    Numerical stability is ensured by:

    - Clipping the linear output :math:`z` to :math:`[-300,\\ 300]`
      before applying the sigmoid to prevent overflow in :math:`e^{-z}`
    - Clipping sigmoid output to :math:`[10^{-15},\\ 1 - 10^{-15}]`
      before computing logarithms to prevent :math:`\\log(0)`

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size :math:`\\eta` used at each gradient descent iteration.
        Too large a value causes divergence; too small causes slow convergence.
    tol : float, default=0.1
        Relative tolerance for early stopping. Training halts when the
        relative change in loss between iterations falls below this value:

        .. math::

            \\frac{|J_{old} - J_{new}|}{J_{old}} < tol

    max_iter : int, default=100
        Maximum number of gradient descent iterations regardless of
        convergence.
    alpha : float, default=0.1
        Regularization strength :math:`\\alpha`. Larger values apply stronger
        penalty to weights, reducing model complexity. Has no effect when
        ``penalty=None``.
    penalty : {'l1', 'l2', None}, default='l2'
        Type of regularization to apply:

        - ``'l2'``: Ridge — penalizes :math:`\\sum w_j^2`, shrinks weights
          toward zero but never exactly to zero.
        - ``'l1'``: Lasso — penalizes :math:`\\sum |w_j|`, produces sparse
          solutions by driving some weights to exactly zero (feature selection).
        - ``None``: No regularization.

    Attributes
    ----------
    w_ : np.ndarray of shape (n_features,)
        Learned weight vector after fitting. Each element represents
        the contribution of the corresponding feature to the log-odds.
    b_ : float
        Learned bias term after fitting. Shifts the decision boundary
        independently of the feature values.

    Notes
    -----
    BCE loss is convex with respect to the model parameters :math:`w`
    and :math:`b`, guaranteeing a single global minimum. Gradient
    descent is therefore guaranteed to converge given a sufficiently
    small ``learning_rate``.

    The gradient of BCE with respect to :math:`w` simplifies elegantly
    due to the cancellation between the sigmoid derivative and the BCE
    derivative:

    .. math::

        \\frac{\\partial J}{\\partial w} =
        \\frac{1}{n} X^{T} (\\hat{y} - y)

    .. math::

        \\frac{\\partial J}{\\partial b} =
        \\frac{1}{n} \\sum_{i=1}^{n} (\\hat{y}_i - y_i)

    With regularization, the weight gradient gains an extra penalty term:

    **L2:**

    .. math::

        \\frac{\\partial J_{L2}}{\\partial w} =
        \\frac{1}{n} X^{T}(\\hat{y} - y) + \\frac{\\alpha}{n} w

    **L1:**

    .. math::

        \\frac{\\partial J_{L1}}{\\partial w} =
        \\frac{1}{n} X^{T}(\\hat{y} - y) +
        \\frac{\\alpha}{n} \\text{sign}(w)

    where :math:`\\text{sign}(w_j)` is:

    .. math::

        \\text{sign}(w_j) = \\begin{cases}
            +1 & w_j > 0 \\\\
            -1 & w_j < 0 \\\\
            0  & w_j = 0
        \\end{cases}

    The parameter update rule at each iteration is:

    .. math::

        w := w - \\frac{\\eta}{n}
        \\left[ X^T(\\hat{y} - y) + \\alpha \\cdot \\text{penalty\_grad} \\right]

    .. math::

        b := b - \\frac{\\eta}{n} \\sum_{i=1}^{n}(\\hat{y}_i - y_i)

    Examples
    --------
    >>> model = LogReg(learning_rate=0.1, max_iter=500, penalty='l2', alpha=0.01)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """

    def __init__(
            self,
            learning_rate: float = 1e-1,
            tol: float = 1e-1,
            max_iter: int = 100,
            alpha: float = 0.1,
            penalty: Literal['l1', 'l2', None] = 'l2'
    ) -> None:

        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.penalty = penalty
        self.w_ = None
        self.b_ = None
        self.__fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fit the model to training data using Batch Gradient Descent.

        Initializes :math:`w` to zeros and :math:`b` to zero, then
        iteratively updates them by computing the full-batch gradient of
        the cost function. Training stops early when the relative change
        in loss drops below ``tol``, or when ``max_iter`` is reached.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix. Each row is one sample, each
            column is one feature.
        y : np.ndarray of shape (n_samples,)
            Binary target vector. Must contain only 0 and 1.

        Returns
        -------
        self : LogReg
            The fitted estimator. Enables method chaining:
            ``model.fit(X, y).predict(X_test)``.
        """

        self.w_ = np.zeros(X.shape[1], dtype=float)
        self.b_ = 0
        n = X.shape[0]
        J_old = np.inf
        for _ in range(self.max_iter):
            sigmoid = self.sigmoid(X)
            error = sigmoid - y
            self.param_update(X, error)
            sigmoid = self.sigmoid(X)
            J_new = self.loss(y, sigmoid, n)
            if np.isinf(J_old):
                J_old = J_new
                continue
            if J_old == 0 or \
                    np.abs(J_old - J_new) / J_old < self.tol:
                break
            J_old = J_new
        self.__fitted = True
        return self

    def loss(self, y: np.ndarray, y_pred: np.ndarray, n: int) -> float:
        """
        Compute the total cost function including regularization penalty.

        Combines Binary Cross-Entropy with the selected regularization term:

        .. math::

            J = \\text{BCE}(y,\\ \\hat{y}) + \\text{penalty}(w)

        **No regularization** (``penalty=None``):

        .. math::

            J = -\\frac{1}{n} \\sum_{i=1}^{n}
            \\left[ y_i \\log(\\hat{y}_i) +
            (1 - y_i) \\log(1 - \\hat{y}_i) \\right]

        **L2 regularization:**

        .. math::

            J = \\text{BCE} + \\frac{1}{2n} \\sum_{j=1}^{p} w_j^2

        **L1 regularization:**

        .. math::

            J = \\text{BCE} + \\frac{\\alpha}{n} \\sum_{j=1}^{p} |w_j|

        Note: the bias :math:`b` is excluded from all penalty terms.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            True binary labels (0 or 1).
        y_pred : np.ndarray of shape (n_samples,)
            Predicted probabilities, already clipped to
            :math:`[10^{-15},\\ 1 - 10^{-15}]` by ``sigmoid``.
        n : int
            Number of training samples. Used for normalization.

        Returns
        -------
        loss : float
            Scalar total loss value. Lower is better.
        """
        bce = - (1 / n) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        if self.penalty == 'l1':
            lasso = (1 / n) * np.sum(np.abs(self.w_))
            return bce + lasso
        elif self.penalty == 'l2':
            ridge = (1 / (2 * n)) * np.sum(self.w_ ** 2)
            return bce + ridge
        return bce

    def param_update(self, X: np.ndarray, error: np.ndarray) -> None:
        """
        Perform one Batch Gradient Descent parameter update step.

        Computes gradients of the cost function with respect to :math:`w`
        and :math:`b`, then updates both parameters in-place.

        The sign of :math:`w` for L1 is captured **before** the weight
        update to ensure the regularization penalty reflects the original
        weight direction, not the post-gradient direction.

        The bias :math:`b` is updated without any regularization term
        in all cases.

        **No regularization:**

        .. math::

            w := w - \\frac{\\eta}{n} X^T (\\hat{y} - y)

        **L2 (Ridge):**

        .. math::

            w := w - \\frac{\\eta}{n}
            \\left[ X^T (\\hat{y} - y) + \\alpha w \\right]

        **L1 (Lasso):**

        .. math::

            w := w - \\frac{\\eta}{n}
            \\left[ X^T (\\hat{y} - y) +
            \\alpha \\cdot \\text{sign}(w) \\right]

        **Bias update (all cases):**

        .. math::

            b := b - \\frac{\\eta}{n}
            \\sum_{i=1}^{n} (\\hat{y}_i - y_i)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        error : np.ndarray of shape (n_samples,)
            Residuals :math:`\\hat{y} - y` from the current iteration.
        """

        n = X.shape[0]
        if self.penalty == 'l1':
            sign = np.where(
                self.w_ > 0, 1, np.where(self.w_ < 0, -1, 0)
            )
            self.w_ -= (self.learning_rate / n) * ((X.T @ error) + \
                self.alpha * sign)
        elif self.penalty == 'l2':
            self.w_ -= (self.learning_rate / n) * ((X.T @ error) + \
                self.alpha * self.w_)
        else:
            self.w_ -= (self.learning_rate / n) * (X.T @ error)
        self.b_ -= (self.learning_rate / n) * np.sum(error)


    def sigmoid(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid activation over the linear output :math:`z`.

        Applies the sigmoid function to map the linear combination
        :math:`z = Xw + b` to a probability in :math:`(0, 1)`:

        .. math::

            \\sigma(z) = \\frac{1}{1 + e^{-z}}

        Two clipping steps ensure numerical stability:

        1. :math:`z` is clipped to :math:`[-300,\\ 300]` before
           computing :math:`e^{-z}` to prevent float overflow.
        2. The output is clipped to :math:`[10^{-15},\\ 1 - 10^{-15}]`
           to prevent :math:`\\log(0)` in BCE computation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted probabilities in :math:`[10^{-15},\\ 1 - 10^{-15}]`.
        """

        z = X @ self.w_ + self.b_
        z = np.clip(z, -3e+2, 3e+2)
        y_pred = 1 / (1 + np.exp(- z))
        return np.clip(y_pred, 1e-15, 1 - 1e-15)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for input samples.

        Computes predicted probabilities via ``sigmoid`` and applies
        a decision threshold of 0.5:

        .. math::

            \\hat{y}_i = \\begin{cases}
                1 & \\sigma(z_i) \\geq 0.5 \\\\
                0 & \\sigma(z_i) < 0.5
            \\end{cases}

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted binary class labels (0 or 1).

        Raises
        ------
        NotFitted
            If ``predict`` is called before ``fit``.
        """

        if not self.__fitted:
            raise NotFitted(self)
        sigmoid = self.sigmoid(X)
        y_pred = np.where(
            sigmoid >= 0.5, 1, 0
        )
        return y_pred