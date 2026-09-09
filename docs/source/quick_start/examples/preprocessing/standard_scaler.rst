Standard Scaler
================

Standard scaling rescales each feature so it has a mean of 0 and a
standard deviation of 1. Many models — especially the
gradient-descent-based ones in :doc:`/api/pyml/linear_model/index` —
converge faster and more reliably when features are on a similar
scale.

Use it as your default choice for scaling, unless your data has
significant outliers (see :doc:`robust_scaler` instead).

.. code-block:: python

    from pyml.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pyml.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    X = rng.normal(loc=50, scale=10, size=(200, 1))

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.hist(X, bins=20)
    ax1.set_title("Original")
    ax2.hist(X_scaled, bins=20)
    ax2.set_title("Standardized")

.. note::
    Always fit the scaler on your training data only, then use the
    same fitted scaler to transform both training and test data.
    Fitting separately on each would let information from the test
    set leak into the scaling.
