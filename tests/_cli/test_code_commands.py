"""Tests for the code CLI command group."""

from click.testing import CliRunner

from pyml._cli.commands.code import code_group


class TestCodeGroupHelp:
    def test_help_shows_check_subcommand(self):
        runner = CliRunner()
        result = runner.invoke(code_group, ["--help"])

        assert result.exit_code == 0
        assert "check" in result.output


class TestCodeCheck:
    def test_check_runs_and_returns_exit_code(self, monkeypatch):
        import pyml._cli.commands.code as code_module

        monkeypatch.setattr(code_module, "CHECK_STEPS", [])

        runner = CliRunner()
        result = runner.invoke(code_group, ["check"])

        assert result.exit_code == 0
