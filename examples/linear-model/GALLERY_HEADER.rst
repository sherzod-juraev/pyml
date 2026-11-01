Linear model
============

A linear model predicts from a weighted sum of the features. It is the simplest family in pyml,
and the one to try first: it is fast, its weights can be read, and it gives a baseline that
other models have to beat. Each page below takes one model, fits it on a real dataset from
`pyml-datasets <https://pyml-datasets.readthedocs.io/en/latest/>`_, and shows what the result
means and where the model goes wrong.

Where to start
--------------

* New to the family? Open **Linear Regression** for a number, or **Logistic Regression** for a
  class. The other pages build on these two.
* Features that repeat each other, and weights that change a lot between samples? Open
  **Ridge** or its classifier.
* Too many features, and you want a short list? Open **Lasso** or its classifier.
* Both of these at once? Open **Elastic Net** or its classifier.

All the models are fit by gradient descent, so the features must be standardized first. Every
page shows what happens when they are not.
