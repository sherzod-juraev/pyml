"""Tests for the docs CLI command group."""

import webbrowser

from click.testing import CliRunner

import pyml._cli.commands.docs as docs_module
from pyml._cli.commands.docs import docs_group


class TestDocsClean:
    def test_reports_nothing_to_clean_when_build_dir_missing(self, tmp_path, monkeypatch):
        docs_root = tmp_path / "docs"
        docs_root.mkdir()
        build_root = docs_root / "build"

        monkeypatch.setattr(docs_module, "DOCS_ROOT", docs_root)
        monkeypatch.setattr(docs_module, "DOCS_BUILD_ROOT", build_root)

        runner = CliRunner()
        result = runner.invoke(docs_group, ["clean"])

        assert result.exit_code == 0
        assert "Nothing to clean" in result.output

    def test_removes_build_dir_when_present(self, tmp_path, monkeypatch):
        docs_root = tmp_path / "docs"
        build_root = docs_root / "build"
        build_root.mkdir(parents=True)
        (build_root / "index.html").write_text("stub")

        monkeypatch.setattr(docs_module, "DOCS_ROOT", docs_root)
        monkeypatch.setattr(docs_module, "DOCS_BUILD_ROOT", build_root)

        runner = CliRunner()
        result = runner.invoke(docs_group, ["clean"])

        assert result.exit_code == 0
        assert not build_root.exists()

    def test_dry_run_does_not_delete(self, tmp_path, monkeypatch):
        docs_root = tmp_path / "docs"
        build_root = docs_root / "build"
        build_root.mkdir(parents=True)

        monkeypatch.setattr(docs_module, "DOCS_ROOT", docs_root)
        monkeypatch.setattr(docs_module, "DOCS_BUILD_ROOT", build_root)

        runner = CliRunner()
        result = runner.invoke(docs_group, ["clean", "--dry-run"])

        assert result.exit_code == 0
        assert build_root.exists()

    def test_fails_when_docs_root_missing(self, tmp_path, monkeypatch):
        missing_docs_root = tmp_path / "does_not_exist"

        monkeypatch.setattr(docs_module, "DOCS_ROOT", missing_docs_root)

        runner = CliRunner()
        result = runner.invoke(docs_group, ["clean"])

        assert result.exit_code != 0


class TestBuild:
    def test_warns_when_browser_fails_to_open(self, tmp_path, monkeypatch):
        docs_root = tmp_path / "docs"
        docs_root.mkdir()
        docs_build = docs_root / "build" / "html"
        docs_build.mkdir(parents=True)
        (docs_build / "index.html").write_text("stub")

        monkeypatch.setattr(docs_module, "DOCS_ROOT", docs_root)
        monkeypatch.setattr(docs_module, "DOCS_BUILD", docs_build)
        monkeypatch.setattr(docs_module, "run_live", lambda command: 0)
        monkeypatch.setattr(webbrowser, "open", lambda uri: False)

        runner = CliRunner()
        result = runner.invoke(docs_group, ["build", "--open"])

        assert "could not open a browser" in result.output
