"""Tests for the main pyml CLI entry point."""

from click.testing import CliRunner

from pyml import __version__
from pyml._cli.main import cli


class TestVersion:
    def test_dash_capital_v_shows_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["-V"])

        assert result.exit_code == 0
        assert __version__ in result.output

    def test_dash_dash_version_shows_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_output_includes_prog_name(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert "pyml" in result.output


class TestHelp:
    def test_dash_h_shows_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["-h"])

        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_dash_dash_help_shows_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_help_lists_all_subcommands(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert "check" in result.output
        assert "code" in result.output
        assert "docs" in result.output
        assert "tests" in result.output


class TestNoArgs:
    def test_no_args_shows_usage_and_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(cli, [])

        assert result.exit_code != 0
        assert "Usage" in result.output
