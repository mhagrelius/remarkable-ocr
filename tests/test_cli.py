"""Tests for CLI commands."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()

# Get path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_cli_version():
    """CLI should show version with --version."""
    from remarkable_ocr import __version__
    from remarkable_ocr.cli import app

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


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


def test_process_not_pdf():
    """process command should error on non-PDF file."""
    from remarkable_ocr.cli import app

    # Create a temp file that's not a PDF
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not a pdf")
        temp_path = f.name

    try:
        result = runner.invoke(app, ["process", temp_path])
        assert result.exit_code != 0
        assert "not a pdf" in result.stdout.lower()
    finally:
        Path(temp_path).unlink()


@patch("remarkable_ocr.cli.check_ollama_health")
@patch("remarkable_ocr.cli.get_page_count")
@patch("remarkable_ocr.cli.extract_pages")
@patch("remarkable_ocr.cli.process_image")
@patch("remarkable_ocr.cli.write_output")
def test_process_success(mock_write, mock_ocr, mock_extract, mock_count, mock_health, tmp_path):
    """process command should successfully process a PDF."""
    from remarkable_ocr.cli import app
    from remarkable_ocr.ocr import OCRResult
    from PIL import Image

    # Setup mocks
    mock_health.return_value = True
    mock_count.return_value = 1

    # Mock page extraction
    test_image = Image.new("RGB", (100, 100), color="white")
    mock_extract.return_value = [(1, test_image)]

    # Mock OCR result
    mock_ocr.return_value = OCRResult(text="Test content", confidence=0.95, page_num=1)

    # Mock output path
    output_path = tmp_path / "output.md"
    mock_write.return_value = output_path

    # Run with real fixture PDF
    pdf_path = FIXTURES_DIR / "sample_1page.pdf"
    result = runner.invoke(app, ["process", str(pdf_path), "-o", str(tmp_path)])

    assert result.exit_code == 0
    assert "Completed" in result.stdout or "✓" in result.stdout


@patch("remarkable_ocr.cli.check_ollama_health")
def test_process_ollama_unavailable(mock_health):
    """process command should error when Ollama unavailable."""
    from remarkable_ocr.cli import app

    mock_health.return_value = False

    pdf_path = FIXTURES_DIR / "sample_1page.pdf"
    result = runner.invoke(app, ["process", str(pdf_path)])

    assert result.exit_code != 0
    assert "ollama" in result.stdout.lower()


@patch("remarkable_ocr.cli.check_ollama_health")
@patch("remarkable_ocr.cli.get_page_count")
@patch("remarkable_ocr.cli.extract_pages")
@patch("remarkable_ocr.cli.process_image")
@patch("remarkable_ocr.cli.write_output")
def test_process_output_formats(mock_write, mock_ocr, mock_extract, mock_count, mock_health, tmp_path):
    """process command should support different output formats."""
    from remarkable_ocr.cli import app
    from remarkable_ocr.ocr import OCRResult
    from PIL import Image

    # Setup mocks
    mock_health.return_value = True
    mock_count.return_value = 1
    test_image = Image.new("RGB", (100, 100), color="white")
    mock_extract.return_value = [(1, test_image)]
    mock_ocr.return_value = OCRResult(text="Test", confidence=0.9, page_num=1)
    mock_write.return_value = tmp_path / "output.json"

    pdf_path = FIXTURES_DIR / "sample_1page.pdf"

    for fmt in ["markdown", "json", "txt"]:
        result = runner.invoke(app, ["process", str(pdf_path), "-o", str(tmp_path), "-f", fmt])
        assert result.exit_code == 0, f"Failed for format: {fmt}"
