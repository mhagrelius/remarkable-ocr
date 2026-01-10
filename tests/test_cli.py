"""Tests for CLI commands."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_version():
    """CLI should show version with --version."""
    from remarkable_ocr.cli import app

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_help():
    """CLI should show help."""
    from remarkable_ocr.cli import app

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "process" in result.stdout
    assert "health" in result.stdout


@patch("remarkable_ocr.cli.check_ollama_health")
def test_health_command_success(mock_health):
    """health command should report success when Ollama is available."""
    from remarkable_ocr.cli import app

    mock_health.return_value = True

    result = runner.invoke(app, ["health"])

    assert result.exit_code == 0
    assert "Ollama" in result.stdout or "✓" in result.stdout


@patch("remarkable_ocr.cli.check_ollama_health")
def test_health_command_failure(mock_health):
    """health command should report failure when Ollama unavailable."""
    from remarkable_ocr.cli import app

    mock_health.return_value = False

    result = runner.invoke(app, ["health"])

    assert result.exit_code != 0


@patch("remarkable_ocr.cli.get_available_models")
def test_models_command(mock_models):
    """models command should list available models."""
    from remarkable_ocr.cli import app

    mock_models.return_value = ["qwen2.5-vl:7b", "llava:13b"]

    result = runner.invoke(app, ["models"])

    assert result.exit_code == 0
    assert "qwen2.5-vl:7b" in result.stdout


def test_process_file_not_found():
    """process command should error on missing file."""
    from remarkable_ocr.cli import app

    result = runner.invoke(app, ["process", "/nonexistent/file.pdf"])

    assert result.exit_code != 0
    assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()
