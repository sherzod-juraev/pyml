"""Tests for the top-level `pyml check` command."""

from click.testing import CliRunner

import pyml._cli.main as main_module
from pyml._cli.main import cli


class TestCheckAll:
    def test_fails_when_docs_root_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main_module, "DOCS_ROOT", tmp_path / "does_not_exist")

        runner = CliRunner()
        result = runner.invoke(cli, ["check"])

        assert result.exit_code != 0

    def test_fails_when_tests_root_missing(self, tmp_path, monkeypatch, tmp_path_factory):
        docs_root = tmp_path_factory.mktemp("docs")
        monkeypatch.setattr(main_module, "DOCS_ROOT", docs_root)
        monkeypatch.setattr(main_module, "TESTS_ROOT", tmp_path / "does_not_exist")

        runner = CliRunner()
        result = runner.invoke(cli, ["check"])

        assert result.exit_code != 0

    def test_prints_section_headers_in_order(self, tmp_path, monkeypatch):
        docs_root = tmp_path / "docs"
        docs_root.mkdir()
        tests_root = tmp_path / "tests"
        tests_root.mkdir()
        monkeypatch.setattr(main_module, "DOCS_ROOT", docs_root)
        monkeypatch.setattr(main_module, "TESTS_ROOT", tests_root)
        monkeypatch.setattr(main_module, "CODE_CHECK_STEPS", [])
        monkeypatch.setattr(main_module, "DOCS_CHECK_STEPS", [])
        monkeypatch.setattr(main_module, "TEST_CHECK_STEPS", [])

        runner = CliRunner()
        result = runner.invoke(cli, ["check"])

        code_idx = result.output.index("Checking code")
        docs_idx = result.output.index("Checking docs")
        tests_idx = result.output.index("Checking tests")
        assert code_idx < docs_idx < tests_idx
