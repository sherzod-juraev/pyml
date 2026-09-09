Robust Scaler
=============

Robust scaling rescales each feature using its median and
interquartile range instead of the mean and standard deviation used
by :doc:`standard_scaler`. Because the median and IQR depend only on
the middle portion of the data, extreme values barely affect them.

Use it when your data contains outliers that would otherwise distort
:doc:`standard_scaler` or :doc:`min_max_scaler`.

.. code-block:: python

    from pyml.preprocessing import RobustScaler

    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.preprocessing import RobustScaler, StandardScaler

    rng = np.random.default_rng(42)
    X = rng.normal(loc=50, scale=10, size=(200, 1))
    X[0] = 500.0  # a single extreme outlier

    robust = RobustScaler().fit(X)
    standard = StandardScaler().fit(X)

    X_standard = standard.transform(X)
    X_robust = robust.transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    ax1.scatter(range(len(X)), X_standard, alpha=0.5, s=15)
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_ylim(-3, 3)
    ax1.set_title(
        f"StandardScaler\n(199 normal points squeezed into "
        f"[{X_standard[1:].min():.2f}, {X_standard[1:].max():.2f}])"
    )
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("Scaled value")

    ax2.scatter(range(len(X)), X_robust, alpha=0.5, s=15)
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_ylim(-3, 3)
    ax2.set_title(
        f"RobustScaler\n(199 normal points spread over "
        f"[{X_robust[1:].min():.2f}, {X_robust[1:].max():.2f}])"
    )
    ax2.set_xlabel("Sample index")

    fig.suptitle("A single outlier compresses StandardScaler's normal points")
    fig.tight_layout()

.. note::
    StandardScaler divides by the standard deviation, and a single
    extreme value inflates that standard deviation for the whole
    feature — so even the normal points end up compressed toward
    zero. RobustScaler divides by the interquartile range instead,
    which only depends on the middle 50% of the data and barely
    moves when one point is extreme.

.. note::
    Always fit the scaler on your training data only, then use the
    same fitted scaler to transform both training and test data.
    Fitting separately on each would let information from the test
    set leak into the scaling.
