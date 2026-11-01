Preprocessing
=============

Transformers that change the features before a model sees them. A transformer learns a few
numbers from the training data in ``fit``, such as a mean or a range, and ``transform`` applies
them to any data. The three scalers here put the features on a common scale, which the distance
based and the gradient descent models need.

Where to start
--------------

* :class:`~pyml.preprocessing.StandardScaler` is the default choice. It centers each feature at 0
  and gives it a standard deviation of 1.
* :class:`~pyml.preprocessing.MinMaxScaler` gives every feature the range from 0 to 1. Use it when
  the features have natural limits and no outliers.
* :class:`~pyml.preprocessing.RobustScaler` uses the median and the interquartile range, so
  outliers do not set the scale. Do not use it on a feature that is the same for most samples.

Fit a scaler on the training data only, then use it to transform the training and the test data.
