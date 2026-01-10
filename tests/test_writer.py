"""Tests for output writing."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_results():
    """Sample OCR results for testing."""
    from remarkable_ocr.ocr import OCRResult

    return [
        OCRResult(text="Page one content", confidence=0.85, page_num=1),
        OCRResult(text="Page two content", confidence=0.72, page_num=2),
    ]


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


def test_write_markdown(sample_results, output_dir: Path):
    """write_output should create markdown file."""
    from remarkable_ocr.writer import write_output

    source = Path("test_notes.pdf")
    output_path = write_output(
        results=sample_results,
        source=source,
        output_dir=output_dir,
        output_format="markdown",
        model="qwen2.5-vl:7b",
    )

    assert output_path.exists()
    assert output_path.suffix == ".md"

    content = output_path.read_text()
    assert "source: test_notes.pdf" in content
    assert "Page one content" in content
    assert "Page two content" in content
    assert "# Page 1" in content
    assert "# Page 2" in content


def test_write_json(sample_results, output_dir: Path):
    """write_output should create JSON file."""
    from remarkable_ocr.writer import write_output

    source = Path("test_notes.pdf")
    output_path = write_output(
        results=sample_results,
        source=source,
        output_dir=output_dir,
        output_format="json",
        model="qwen2.5-vl:7b",
    )

    assert output_path.exists()
    assert output_path.suffix == ".json"

    data = json.loads(output_path.read_text())
    assert data["metadata"]["source"] == "test_notes.pdf"
    assert data["metadata"]["pages"] == 2
    assert len(data["pages"]) == 2
    assert data["pages"][0]["text"] == "Page one content"


def test_write_txt(sample_results, output_dir: Path):
    """write_output should create plain text file."""
    from remarkable_ocr.writer import write_output

    source = Path("test_notes.pdf")
    output_path = write_output(
        results=sample_results,
        source=source,
        output_dir=output_dir,
        output_format="txt",
        model="qwen2.5-vl:7b",
    )

    assert output_path.exists()
    assert output_path.suffix == ".txt"

    content = output_path.read_text()
    assert "test_notes.pdf" in content
    assert "Page one content" in content
    assert "=== Page 1 ===" in content


def test_write_creates_output_dir(sample_results, tmp_path: Path):
    """write_output should create output directory if missing."""
    from remarkable_ocr.writer import write_output

    output_dir = tmp_path / "nonexistent" / "nested"
    source = Path("test.pdf")

    output_path = write_output(
        results=sample_results,
        source=source,
        output_dir=output_dir,
        output_format="markdown",
        model="test",
    )

    assert output_dir.exists()
    assert output_path.exists()


def test_calculate_average_confidence(sample_results):
    """_calculate_average_confidence should compute average."""
    from remarkable_ocr.writer import _calculate_average_confidence

    avg = _calculate_average_confidence(sample_results)
    assert avg == pytest.approx(0.785, rel=0.01)
