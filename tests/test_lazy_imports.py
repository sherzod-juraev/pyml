"""Tests for the lazy-loading mechanism of every pyml package.

The packages are discovered from ``__all__``, starting at ``pyml`` and
following every name that resolves to another lazy package, so a new
package or a new public name is covered without editing this file.
"""

import ast
import inspect
from pathlib import Path
from types import ModuleType

import pytest

import pyml


def _lazy_packages() -> list[ModuleType]:
    found: list[ModuleType] = []
    pending: list[ModuleType] = [pyml]
    while pending:
        module = pending.pop()
        if "__getattr__" not in vars(module):
            continue
        found.append(module)
        for name in module.__all__:
            member = getattr(module, name)
            if inspect.ismodule(member):
                pending.append(member)
    return sorted(found, key=lambda module: module.__name__)


PACKAGES = _lazy_packages()
PACKAGE_IDS = [module.__name__ for module in PACKAGES]
PUBLIC_NAMES = [(module, name) for module in PACKAGES for name in module.__all__]
PUBLIC_NAME_IDS = [f"{module.__name__}.{name}" for module, name in PUBLIC_NAMES]


def _stub_names(module: ModuleType) -> set[str]:
    stub = Path(inspect.getfile(module)).with_suffix(".pyi")
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname == alias.name
    }


class TestDiscovery:
    def test_finds_the_root_and_nested_packages(self):
        names = {module.__name__ for module in PACKAGES}

        assert {"pyml", "pyml.core", "pyml.core.base", "pyml.core.exceptions"} <= names
        assert {"pyml.linear_model", "pyml.tree"} <= names


class TestLazyImports:
    @pytest.mark.parametrize(("module", "name"), PUBLIC_NAMES, ids=PUBLIC_NAME_IDS)
    def test_public_name_resolves_to_the_object_of_that_name(self, module, name):
        member = getattr(module, name)

        assert member.__name__ in {name, f"{module.__name__}.{name}"}

    @pytest.mark.parametrize("module", PACKAGES, ids=PACKAGE_IDS)
    def test_unknown_attribute_raises_attribute_error(self, module):
        with pytest.raises(AttributeError, match="NonExistent"):
            _ = module.NonExistent


class TestDir:
    @pytest.mark.parametrize("module", PACKAGES, ids=PACKAGE_IDS)
    def test_dir_matches_all(self, module):
        assert set(dir(module)) == set(module.__all__)

    @pytest.mark.parametrize("module", PACKAGES, ids=PACKAGE_IDS)
    def test_dir_is_sorted(self, module):
        assert dir(module) == sorted(module.__all__)


class TestStubs:
    @pytest.mark.parametrize("module", PACKAGES, ids=PACKAGE_IDS)
    def test_stub_re_exports_exactly_the_names_in_all(self, module):
        assert _stub_names(module) == set(module.__all__)
