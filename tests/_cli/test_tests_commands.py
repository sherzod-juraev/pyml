"""Tests for the tests CLI command group."""

from click.testing import CliRunner

import pyml._cli.commands.tests as tests_module
from pyml._cli.commands.tests import tests_group


class TestTestsGroupHelp:
    def test_help_shows_check_subcommand(self):
        runner = CliRunner()
        result = runner.invoke(tests_group, ["--help"])

        assert result.exit_code == 0
        assert "check" in result.output


class TestTestsCheck:
    def test_fails_when_tests_root_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tests_module, "TESTS_ROOT", tmp_path / "does_not_exist")

        runner = CliRunner()
        result = runner.invoke(tests_group, ["check"])

        assert result.exit_code != 0

    def test_runs_when_tests_root_exists(self, tmp_path, monkeypatch):
        tests_root = tmp_path / "tests"
        tests_root.mkdir()
        monkeypatch.setattr(tests_module, "TESTS_ROOT", tests_root)
        monkeypatch.setattr(tests_module, "CHECK_STEPS", [])

        runner = CliRunner()
        result = runner.invoke(tests_group, ["check"])

        assert result.exit_code == 0
