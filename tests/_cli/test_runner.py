"""Tests for _runner.py's shared CLI helpers."""

import sys

import click
import pytest

from pyml._cli._runner import remove_tree, require_source_checkout, run_checks


class TestRequireSourceCheckout:
    def test_raises_click_exception_when_path_missing(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(click.ClickException):
            require_source_checkout(missing, "docs/")

    def test_does_not_raise_when_path_exists(self, tmp_path):
        require_source_checkout(tmp_path, "docs/")


class TestRunChecks:
    def test_returns_zero_when_all_steps_pass(self):
        steps = [("ok", [sys.executable, "-c", "pass"])]
        assert run_checks(steps) == 0

    def test_returns_one_when_a_step_fails(self):
        steps = [("fail", [sys.executable, "-c", "exit(1)"])]
        assert run_checks(steps) == 1

    def test_reports_missing_command_as_failure(self):
        steps = [("missing", ["nonexistent-command-xyz"])]
        assert run_checks(steps) == 1


class TestRemoveTree:
    def test_removes_directory_inside_allowed_root(self, tmp_path):
        target = tmp_path / "subdir"
        target.mkdir()

        result = remove_tree(target, allowed_root=tmp_path)

        assert result is True
        assert not target.exists()

    def test_refuses_to_remove_outside_allowed_root(self, tmp_path):
        outside = tmp_path.parent / "some_other_dir"
        with pytest.raises(click.ClickException):
            remove_tree(outside, allowed_root=tmp_path)

    def test_dry_run_does_not_delete(self, tmp_path):
        target = tmp_path / "subdir"
        target.mkdir()

        result = remove_tree(target, allowed_root=tmp_path, dry_run=True)

        assert result is True
        assert target.exists()

    def test_returns_false_when_path_does_not_exist(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        result = remove_tree(missing, allowed_root=tmp_path)
        assert result is False
