"""Tests for clone.

Every estimator in the registry is cloned, so a new estimator is covered
without a change here. The nesting rules are checked on dummy estimators that
hold other estimators in their parameters, and on a real ``Pipeline``. The
parameter checks of the estimators are tested in the contract tests.
"""

from collections import OrderedDict, defaultdict, namedtuple
from typing import Any, Self

import numpy as np
import pytest

from pyml.core.base.estimator import BaseEstimator
from pyml.core.clone import clone
from pyml.linear_model import Lasso, Ridge
from pyml.pipeline import Pipeline
from pyml.preprocessing import StandardScaler
from tests.registry import ENTRIES, Entry, ids

_Pair = namedtuple("_Pair", ["first", "second"])


class _ListSubclass(list[Any]):
    pass


class _Holder(BaseEstimator):
    """Keeps one parameter of any kind and learns one attribute in ``fit``."""

    def __init__(self, value: Any = None) -> None:
        super().__init__()
        self.value = value

    def fit(self) -> Self:
        self.coef_ = np.arange(3.0)
        self.is_fitted_ = True
        return self


def _fit(entry: Entry, model, rng):
    X = rng.uniform(1, 10, size=(60, 3))
    if entry.family == "classifier":
        return model.fit(X, np.array([0, 1] * 30))
    if entry.family == "regressor":
        return model.fit(X, rng.uniform(0, 10, size=60))
    return model.fit(X)


class TestClone:
    @pytest.mark.parametrize("entry", ENTRIES, ids=ids(list(ENTRIES)))
    def test_returns_a_new_estimator_of_the_same_class_and_parameters(self, entry):
        original = entry.build()

        copy = clone(original)

        assert copy is not original
        assert type(copy) is type(original)
        assert copy.get_params() == original.get_params()

    @pytest.mark.parametrize("entry", ENTRIES, ids=ids(list(ENTRIES)))
    def test_copy_of_a_fitted_estimator_is_not_fitted(self, entry, rng):
        original = _fit(entry, entry.build(), rng)

        copy = clone(original)

        assert copy.is_fitted_ is False
        assert [name for name in vars(copy) if name.endswith("_")] == ["is_fitted_"]

    @pytest.mark.parametrize("entry", ENTRIES, ids=ids(list(ENTRIES)))
    def test_original_stays_fitted(self, entry, rng):
        original = _fit(entry, entry.build(), rng)
        fitted_attributes = {name for name in vars(original) if name.endswith("_")}

        clone(original)

        assert original.is_fitted_ is True
        assert {name for name in vars(original) if name.endswith("_")} == fitted_attributes

    def test_fitted_attribute_is_not_carried_over(self):
        original = _Holder().fit()

        copy = clone(original)

        assert not hasattr(copy, "coef_")

    def test_array_parameter_is_copied_not_shared(self):
        original = _Holder(value=np.array([1.0, 2.0, 3.0]))

        copy = clone(original)
        copy.value[0] = 99.0

        assert copy.value is not original.value
        np.testing.assert_array_equal(original.value, [1.0, 2.0, 3.0])

    def test_mutable_leaf_inside_a_list_is_copied_not_shared(self):
        original = _Holder(value=[{1, 2}])

        copy = clone(original)
        copy.value[0].add(3)

        assert original.value == [{1, 2}]

    def test_set_parameter_is_copied(self):
        original = _Holder(value={1, 2})

        copy = clone(original)

        assert copy.value == {1, 2}
        assert copy.value is not original.value

    def test_class_as_parameter_value_is_kept_as_it_is(self):
        original = _Holder(value=Ridge)

        copy = clone(original)

        assert copy.value is Ridge


class TestNested:
    def test_estimator_parameter_is_cloned(self):
        inner = Ridge(alpha=2.0)
        original = _Holder(value=inner)

        copy = clone(original)

        assert copy.value is not inner
        assert type(copy.value) is Ridge
        assert copy.value.get_params() == inner.get_params()

    def test_fitted_inner_estimator_is_cloned_unfitted(self):
        inner = _Holder().fit()
        original = _Holder(value=inner)

        copy = clone(original)

        assert copy.value.is_fitted_ is False
        assert inner.is_fitted_ is True

    def test_list_keeps_its_type_and_clones_its_items(self):
        items = [Ridge(), Lasso()]
        original = _Holder(value=items)

        copy = clone(original)

        assert type(copy.value) is list
        assert copy.value is not items
        assert [type(item) for item in copy.value] == [Ridge, Lasso]
        assert all(new is not old for new, old in zip(copy.value, items, strict=True))

    def test_tuple_keeps_its_type_and_clones_its_items(self):
        items = (("a", Ridge()), ("b", Lasso()))
        original = _Holder(value=items)

        copy = clone(original)

        assert type(copy.value) is tuple
        assert type(copy.value[0]) is tuple
        assert [name for name, _ in copy.value] == ["a", "b"]
        assert all(new[1] is not old[1] for new, old in zip(copy.value, items, strict=True))

    def test_dict_keeps_its_keys_and_clones_its_values(self):
        items = {"a": Ridge(), "b": Lasso()}
        original = _Holder(value=items)

        copy = clone(original)

        assert type(copy.value) is dict
        assert list(copy.value) == ["a", "b"]
        assert all(copy.value[key] is not items[key] for key in items)

    def test_estimator_deep_inside_containers_is_cloned(self):
        inner = Ridge()
        original = _Holder(value={"group": [("name", inner)]})

        copy = clone(original)

        assert copy.value["group"][0][1] is not inner
        assert type(copy.value["group"][0][1]) is Ridge

    def test_estimator_holding_an_estimator_is_cloned_to_every_level(self):
        innermost = Ridge()
        original = _Holder(value=_Holder(value=innermost))

        copy = clone(original)

        assert copy.value.value is not innermost
        assert type(copy.value.value) is Ridge

    def test_fitted_pipeline_is_cloned_to_a_new_unfitted_pipeline(self, linear_data):
        X, y = linear_data(n_samples=60)
        original = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=0.5))])
        original.fit(X, y)
        predictions = original.predict(X)

        copy = clone(original)

        assert type(copy) is Pipeline
        assert copy.is_fitted_ is False
        assert [name for name, _ in copy.steps] == ["scale", "model"]
        new_steps: list[Any] = copy.steps
        old_steps: list[Any] = original.steps
        assert all(new is not old for (_, new), (_, old) in zip(new_steps, old_steps, strict=True))
        assert all(step.is_fitted_ is False for _, step in new_steps)
        assert new_steps[1][1].get_params() == old_steps[1][1].get_params()
        np.testing.assert_array_equal(original.predict(X), predictions)

    def test_fitting_the_clone_leaves_the_original_steps_untouched(self, linear_data):
        X, y = linear_data(n_samples=60)
        original = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
        original.fit(X, y)
        scaler: Any = original.steps[0][1]
        mean_before = scaler.mean_.copy()

        clone(original).fit(X + 100.0, y)

        np.testing.assert_array_equal(scaler.mean_, mean_before)


class TestValidation:
    def test_rejects_a_class(self):
        with pytest.raises(TypeError, match="not the class Ridge"):
            clone(Ridge)  # type: ignore[type-var]

    @pytest.mark.parametrize("value", [[1, 2], 5, None, "ridge", {"alpha": 1.0}])
    def test_rejects_an_object_that_is_not_an_estimator(self, value):
        with pytest.raises(TypeError, match=f"not {type(value).__name__}"):
            clone(value)

    @pytest.mark.parametrize(
        "value",
        [
            _Pair(1, 2),
            OrderedDict(a=1),
            defaultdict(int),
            _ListSubclass([1]),
        ],
        ids=["namedtuple", "ordered_dict", "default_dict", "list_subclass"],
    )
    def test_rejects_a_subclass_of_a_container_as_a_parameter(self, value):
        original = _Holder(value=value)

        with pytest.raises(TypeError, match=f"not {type(value).__name__}"):
            clone(original)

    def test_rejects_a_subclass_of_a_container_at_any_depth(self):
        original = _Holder(value={"a": [(1, _Pair(1, 2))]})

        with pytest.raises(TypeError, match="not _Pair"):
            clone(original)

    def test_rejects_a_subclass_of_a_container_inside_a_nested_estimator(self):
        original = _Holder(value=_Holder(value=OrderedDict(a=1)))

        with pytest.raises(TypeError, match="not OrderedDict"):
            clone(original)
