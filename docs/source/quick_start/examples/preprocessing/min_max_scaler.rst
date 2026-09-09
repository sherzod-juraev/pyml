Min Max Scaler
==============

Min-max scaling rescales each feature to a fixed [0, 1] range, based
on the minimum and maximum values seen during training.

Use it when you want features bounded to a known range — for
example, for algorithms that expect non-negative input, or when
comparing features that are naturally on very different scales. If
your data has significant outliers, prefer :doc:`robust_scaler`
instead, since a single extreme value here stretches the whole range.

.. code-block:: python

    from pyml.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.preprocessing import MinMaxScaler

    rng = np.random.default_rng(42)
    X = rng.normal(loc=50, scale=10, size=(200, 1))

    scaler = MinMaxScaler().fit(X)
    X_scaled = scaler.transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.hist(X, bins=20)
    ax1.set_title("Original")
    ax2.hist(X_scaled, bins=20)
    ax2.set_title("Min-max scaled")

.. note::
    Always fit the scaler on your training data only, then use the
    same fitted scaler to transform both training and test data.
    Fitting separately on each would let information from the test
    set leak into the scaling.
