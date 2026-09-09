"""Tests for pyml's type aliases.

Type aliases have no runtime behavior of their own (they only affect
static type checking), so this only verifies they remain importable.
Correct usage of these aliases is checked implicitly by mypy wherever
they appear in other test files.
"""

from pyml.core.dtypes import (
    ClassificationTarget,
    ClusterLabels,
    FeatureMatrix,
    RegressionTarget,
)


def test_dtypes_are_importable():
    assert ClassificationTarget is not None
    assert ClusterLabels is not None
    assert FeatureMatrix is not None
    assert RegressionTarget is not None
