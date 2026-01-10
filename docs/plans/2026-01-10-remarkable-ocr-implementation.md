# Remarkable OCR Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI tool that extracts handwritten text from Remarkable PDF exports using local Ollama OCR.

**Architecture:** Flat Python package with modules for config, PDF extraction, OCR, text processing, and output writing. CLI built with Typer. All processing local via Ollama.

**Tech Stack:** Python 3.10+, pymupdf, ollama-python, Typer, Pydantic, Rich

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `remarkable_ocr/__init__.py`
- Create: `remarkable_ocr/__main__.py`
- Create: `.env.example`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "remarkable-ocr"
version = "0.1.0"
description = "Extract handwritten notes from Remarkable PDF exports using local OCR"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "typer>=0.9.0",
    "rich>=13.0.0",
    "pymupdf>=1.24.0",
    "pillow>=10.0.0",
    "ollama>=0.3.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
]

[project.scripts]
remarkable-ocr = "remarkable_ocr.cli:app"
```

**Step 2: Create package files**

`remarkable_ocr/__init__.py`:
```python
"""Remarkable OCR - Extract handwritten notes from Remarkable PDF exports."""

__version__ = "0.1.0"
```

`remarkable_ocr/__main__.py`:
```python
"""Enable running as `python -m remarkable_ocr`."""

from remarkable_ocr.cli import app

if __name__ == "__main__":
    app()
```

**Step 3: Create .env.example**

```bash
# Remarkable OCR Configuration
REMARKABLE_OCR_MODEL=qwen2.5-vl:7b
REMARKABLE_OCR_OLLAMA_HOST=http://localhost:11434
REMARKABLE_OCR_OUTPUT_DIR=./output
REMARKABLE_OCR_CONFIDENCE_THRESHOLD=0.5
REMARKABLE_OCR_TIMEOUT=60
REMARKABLE_OCR_LOG_LEVEL=INFO
```

**Step 4: Create test scaffolding**

`tests/__init__.py`:
```python
"""Remarkable OCR tests."""
```

`tests/conftest.py`:
```python
"""Shared pytest fixtures."""

import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return FIXTURES_DIR
```

**Step 5: Create fixtures directory**

```bash
mkdir -p tests/fixtures
```

**Step 6: Install package in dev mode and verify**

Run: `pip install -e ".[dev]"`
Expected: Installs successfully

Run: `python -c "import remarkable_ocr; print(remarkable_ocr.__version__)"`
Expected: `0.1.0`

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with pyproject.toml and package structure"
```

---

## Task 2: Config Module

**Files:**
- Create: `remarkable_ocr/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing test for Settings defaults**

`tests/test_config.py`:
```python
"""Tests for configuration management."""

import os
from pathlib import Path

import pytest


def test_settings_defaults():
    """Settings should have sensible defaults."""
    # Clear any env vars that might interfere
    env_vars = [k for k in os.environ if k.startswith("REMARKABLE_OCR_")]
    for var in env_vars:
        os.environ.pop(var, None)

    from remarkable_ocr.config import Settings

    settings = Settings()

    assert settings.model == "qwen2.5-vl:7b"
    assert settings.ollama_host == "http://localhost:11434"
    assert settings.output_dir == Path("./output")
    assert settings.confidence_threshold == 0.5
    assert settings.timeout == 60
    assert settings.log_level == "INFO"


def test_settings_from_env(monkeypatch):
    """Settings should load from environment variables."""
    monkeypatch.setenv("REMARKABLE_OCR_MODEL", "qwen2.5-vl:3b")
    monkeypatch.setenv("REMARKABLE_OCR_TIMEOUT", "120")
    monkeypatch.setenv("REMARKABLE_OCR_OUTPUT_DIR", "/tmp/ocr-output")

    # Force reimport to pick up new env vars
    import importlib
    import remarkable_ocr.config
    importlib.reload(remarkable_ocr.config)
    from remarkable_ocr.config import Settings

    settings = Settings()

    assert settings.model == "qwen2.5-vl:3b"
    assert settings.timeout == 120
    assert settings.output_dir == Path("/tmp/ocr-output")


def test_settings_validation():
    """Settings should validate values."""
    from remarkable_ocr.config import Settings

    with pytest.raises(ValueError):
        Settings(confidence_threshold=1.5)  # Must be 0-1

    with pytest.raises(ValueError):
        Settings(timeout=-1)  # Must be positive
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

`remarkable_ocr/config.py`:
```python
"""Configuration management using Pydantic settings."""

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model: str = "qwen2.5-vl:7b"
    ollama_host: str = "http://localhost:11434"
    output_dir: Path = Path("./output")
    confidence_threshold: float = 0.5
    timeout: int = 60
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    keep_temp_images: bool = False
    output_format: Literal["markdown", "json", "txt"] = "markdown"

    model_config = SettingsConfigDict(
        env_prefix="REMARKABLE_OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Confidence threshold must be between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Timeout must be positive."""
        if v <= 0:
            raise ValueError("timeout must be positive")
        return v
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add remarkable_ocr/config.py tests/test_config.py
git commit -m "feat(config): add Settings class with env var support and validation"
```

---

## Task 3: Logging Module

**Files:**
- Create: `remarkable_ocr/logging.py`
- Create: `tests/test_logging.py`

**Step 1: Write failing test for logging setup**

`tests/test_logging.py`:
```python
"""Tests for logging configuration."""

import logging
from pathlib import Path


def test_setup_logging_creates_logger():
    """setup_logging should return a configured logger."""
    from remarkable_ocr.logging import setup_logging

    logger = setup_logging(level="INFO")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "remarkable_ocr"
    assert logger.level == logging.INFO


def test_setup_logging_debug_level():
    """setup_logging should handle DEBUG level."""
    from remarkable_ocr.logging import setup_logging

    logger = setup_logging(level="DEBUG")

    assert logger.level == logging.DEBUG


def test_get_logger_returns_child():
    """get_logger should return a child logger."""
    from remarkable_ocr.logging import get_logger, setup_logging

    setup_logging(level="INFO")
    logger = get_logger("pdf")

    assert logger.name == "remarkable_ocr.pdf"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_logging.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

`remarkable_ocr/logging.py`:
```python
"""Logging configuration with rich console output."""

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# Module-level logger instance
_logger: logging.Logger | None = None

# Log file location
LOG_DIR = Path.home() / ".remarkable_ocr" / "logs"
LOG_FILE = LOG_DIR / "remarkable_ocr.log"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging with rich console and file output.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    global _logger

    # Create log directory if needed
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("remarkable_ocr")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with rich formatting
    console_handler = RichHandler(
        console=Console(stderr=True),
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )
    console_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)

    # File handler for persistent logs
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: Module name (e.g., 'pdf', 'ocr')

    Returns:
        Child logger instance
    """
    return logging.getLogger(f"remarkable_ocr.{name}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_logging.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add remarkable_ocr/logging.py tests/test_logging.py
git commit -m "feat(logging): add rich console and file logging setup"
```

---

## Task 4: PDF Module

**Files:**
- Create: `remarkable_ocr/pdf.py`
- Create: `tests/test_pdf.py`
- Create: `tests/fixtures/sample_1page.pdf` (manually or via script)

**Step 1: Create a simple test PDF fixture**

We need a test PDF. Create a minimal one with pymupdf:

`tests/fixtures/create_test_pdfs.py` (helper script, not part of package):
```python
"""Create test PDF fixtures."""

import fitz  # pymupdf
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent


def create_sample_1page():
    """Create a simple 1-page PDF with text."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Letter size

    # Add some text
    text = "Hello, this is a test page.\n\nSecond paragraph here."
    page.insert_text((72, 72), text, fontsize=12)

    output_path = FIXTURES_DIR / "sample_1page.pdf"
    doc.save(output_path)
    doc.close()
    print(f"Created: {output_path}")


def create_sample_blank():
    """Create a blank PDF page."""
    doc = fitz.open()
    doc.new_page(width=612, height=792)

    output_path = FIXTURES_DIR / "sample_blank.pdf"
    doc.save(output_path)
    doc.close()
    print(f"Created: {output_path}")


def create_sample_multipage():
    """Create a 3-page PDF."""
    doc = fitz.open()

    for i in range(3):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"This is page {i + 1}", fontsize=12)

    output_path = FIXTURES_DIR / "sample_multipage.pdf"
    doc.save(output_path)
    doc.close()
    print(f"Created: {output_path}")


if __name__ == "__main__":
    create_sample_1page()
    create_sample_blank()
    create_sample_multipage()
```

Run: `python tests/fixtures/create_test_pdfs.py`

**Step 2: Write failing tests for PDF extraction**

`tests/test_pdf.py`:
```python
"""Tests for PDF processing."""

from pathlib import Path

import pytest
from PIL import Image


def test_extract_pages_yields_images(fixtures_dir: Path):
    """extract_pages should yield page images."""
    from remarkable_ocr.pdf import extract_pages

    pdf_path = fixtures_dir / "sample_1page.pdf"
    pages = list(extract_pages(pdf_path))

    assert len(pages) == 1
    page_num, image = pages[0]
    assert page_num == 1
    assert isinstance(image, Image.Image)


def test_extract_pages_multipage(fixtures_dir: Path):
    """extract_pages should handle multi-page PDFs."""
    from remarkable_ocr.pdf import extract_pages

    pdf_path = fixtures_dir / "sample_multipage.pdf"
    pages = list(extract_pages(pdf_path))

    assert len(pages) == 3
    for i, (page_num, image) in enumerate(pages):
        assert page_num == i + 1
        assert isinstance(image, Image.Image)


def test_extract_pages_width(fixtures_dir: Path):
    """extract_pages should scale to specified width."""
    from remarkable_ocr.pdf import extract_pages

    pdf_path = fixtures_dir / "sample_1page.pdf"
    pages = list(extract_pages(pdf_path, width=800))

    _, image = pages[0]
    assert image.width == 800


def test_extract_pages_file_not_found():
    """extract_pages should raise FileNotFoundError for missing files."""
    from remarkable_ocr.pdf import extract_pages

    with pytest.raises(FileNotFoundError):
        list(extract_pages(Path("/nonexistent/file.pdf")))


def test_get_page_count(fixtures_dir: Path):
    """get_page_count should return correct count."""
    from remarkable_ocr.pdf import get_page_count

    assert get_page_count(fixtures_dir / "sample_1page.pdf") == 1
    assert get_page_count(fixtures_dir / "sample_multipage.pdf") == 3
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_pdf.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 4: Write minimal implementation**

`remarkable_ocr/pdf.py`:
```python
"""PDF processing using pymupdf."""

from collections.abc import Iterator
from pathlib import Path

import fitz  # pymupdf
from PIL import Image

from remarkable_ocr.logging import get_logger

logger = get_logger("pdf")


def extract_pages(
    pdf_path: Path, width: int = 1288
) -> Iterator[tuple[int, Image.Image]]:
    """Extract pages from PDF as PIL Images.

    Args:
        pdf_path: Path to PDF file
        width: Target width for rendered images (default 1288 for Qwen2.5-VL)

    Yields:
        Tuples of (page_number, PIL.Image) where page_number is 1-indexed

    Raises:
        FileNotFoundError: If PDF file doesn't exist
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.debug(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)

    try:
        for page_num, page in enumerate(doc, start=1):
            # Calculate zoom factor to achieve target width
            zoom = width / page.rect.width
            matrix = fitz.Matrix(zoom, zoom)

            # Render page to pixmap
            pixmap = page.get_pixmap(matrix=matrix)

            # Convert to PIL Image
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

            logger.debug(f"Page {page_num}: {image.width}x{image.height}px")
            yield page_num, image

    finally:
        doc.close()


def get_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Number of pages

    Raises:
        FileNotFoundError: If PDF file doesn't exist
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def save_temp_images(pdf_path: Path, cache_dir: Path, width: int = 1288) -> list[Path]:
    """Save PDF pages as temporary images.

    Args:
        pdf_path: Path to PDF file
        cache_dir: Directory to save images
        width: Target width for rendered images

    Returns:
        List of paths to saved images
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for page_num, image in extract_pages(pdf_path, width):
        image_path = cache_dir / f"{pdf_path.stem}_page_{page_num:03d}.png"
        image.save(image_path, "PNG")
        saved_paths.append(image_path)
        logger.debug(f"Saved: {image_path}")

    return saved_paths
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_pdf.py -v`
Expected: PASS (5 tests)

**Step 6: Commit**

```bash
git add remarkable_ocr/pdf.py tests/test_pdf.py tests/fixtures/
git commit -m "feat(pdf): add PDF page extraction with pymupdf"
```

---

## Task 5: Text Processor Module

**Files:**
- Create: `remarkable_ocr/processor.py`
- Create: `tests/test_processor.py`

**Step 1: Write failing tests for text processing**

`tests/test_processor.py`:
```python
"""Tests for text processing."""

import pytest


def test_clean_text_strips_whitespace():
    """clean_text should strip leading/trailing whitespace."""
    from remarkable_ocr.processor import clean_text

    assert clean_text("  hello  ") == "hello"
    assert clean_text("\n\nhello\n\n") == "hello"


def test_clean_text_normalizes_newlines():
    """clean_text should normalize multiple newlines to max 2."""
    from remarkable_ocr.processor import clean_text

    text = "paragraph one\n\n\n\n\nparagraph two"
    result = clean_text(text)

    assert result == "paragraph one\n\nparagraph two"


def test_clean_text_preserves_structure():
    """clean_text should preserve paragraph breaks and lists."""
    from remarkable_ocr.processor import clean_text

    text = "# Header\n\n- Item 1\n- Item 2\n\nParagraph"
    result = clean_text(text)

    assert "# Header" in result
    assert "- Item 1" in result
    assert "- Item 2" in result


def test_clean_text_fixes_common_artifacts():
    """clean_text should fix common OCR artifacts."""
    from remarkable_ocr.processor import clean_text

    # Common OCR mistakes
    assert "morning" in clean_text("rnorning")  # rn -> m
    assert clean_text("hello   world") == "hello world"  # multiple spaces


def test_is_blank_page_true():
    """is_blank_page should return True for blank content."""
    from remarkable_ocr.processor import is_blank_page

    assert is_blank_page("") is True
    assert is_blank_page("   ") is True
    assert is_blank_page("\n\n") is True
    assert is_blank_page("[blank page]") is True
    assert is_blank_page("[Blank Page]") is True


def test_is_blank_page_false():
    """is_blank_page should return False for content."""
    from remarkable_ocr.processor import is_blank_page

    assert is_blank_page("Hello world") is False
    assert is_blank_page("- Item") is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_processor.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

`remarkable_ocr/processor.py`:
```python
"""Text processing and cleaning for OCR output."""

import re

from remarkable_ocr.logging import get_logger

logger = get_logger("processor")

# Common OCR artifacts and their corrections
OCR_CORRECTIONS = [
    (r"rn(?=[aeiou])", "m"),  # rn before vowel -> m (e.g., rnorning -> morning)
    (r"(?<=[a-z])l(?=[a-z])", "l"),  # Keep l, but could add l/1 confusion fixes
    (r"0(?=[a-zA-Z])", "O"),  # 0 before letter -> O
    (r"(?<=[a-zA-Z])0", "o"),  # 0 after letter -> o
]


def clean_text(raw: str) -> str:
    """Clean OCR output for better readability.

    Args:
        raw: Raw OCR output text

    Returns:
        Cleaned text with normalized whitespace and fixed artifacts
    """
    text = raw

    # Apply OCR corrections
    for pattern, replacement in OCR_CORRECTIONS:
        text = re.sub(pattern, replacement, text)

    # Normalize multiple spaces to single space
    text = re.sub(r" {2,}", " ", text)

    # Normalize multiple newlines to max 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def is_blank_page(text: str) -> bool:
    """Check if page content indicates a blank page.

    Args:
        text: Extracted text from page

    Returns:
        True if page appears to be blank
    """
    cleaned = text.strip().lower()

    if not cleaned:
        return True

    # Check for explicit blank page markers
    blank_markers = ["[blank page]", "[blank]", "[empty]", "[no text]", "[no text detected]"]
    for marker in blank_markers:
        if cleaned == marker:
            return True

    return False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_processor.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add remarkable_ocr/processor.py tests/test_processor.py
git commit -m "feat(processor): add text cleaning and blank page detection"
```

---

## Task 6: Writer Module

**Files:**
- Create: `remarkable_ocr/writer.py`
- Create: `tests/test_writer.py`

**Step 1: Write failing tests for output writing**

`tests/test_writer.py`:
```python
"""Tests for output writing."""

import json
from datetime import datetime, timezone
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_writer.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation (first create OCRResult dataclass stub)**

First, create a minimal `ocr.py` with just the dataclass:

`remarkable_ocr/ocr.py`:
```python
"""OCR processing using Ollama."""

from dataclasses import dataclass


@dataclass
class OCRResult:
    """Result from OCR processing of a single page."""

    text: str
    confidence: float | None
    page_num: int
```

Now write the writer module:

`remarkable_ocr/writer.py`:
```python
"""Output writing in various formats."""

import json
from datetime import datetime, timezone
from pathlib import Path

from remarkable_ocr.logging import get_logger
from remarkable_ocr.ocr import OCRResult

logger = get_logger("writer")


def _calculate_average_confidence(results: list[OCRResult]) -> float | None:
    """Calculate average confidence across all results.

    Args:
        results: List of OCR results

    Returns:
        Average confidence or None if no confidence scores available
    """
    confidences = [r.confidence for r in results if r.confidence is not None]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def write_output(
    results: list[OCRResult],
    source: Path,
    output_dir: Path,
    output_format: str,
    model: str,
) -> Path:
    """Write OCR results to file.

    Args:
        results: List of OCR results
        source: Source PDF path
        output_dir: Output directory
        output_format: Format (markdown, json, txt)
        model: Model name used for OCR

    Returns:
        Path to written file
    """
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output path
    suffix_map = {"markdown": ".md", "json": ".json", "txt": ".txt"}
    suffix = suffix_map.get(output_format, ".md")
    output_path = output_dir / f"{source.stem}{suffix}"

    # Generate content
    if output_format == "json":
        content = _format_json(results, source, model)
    elif output_format == "txt":
        content = _format_txt(results, source)
    else:
        content = _format_markdown(results, source, model)

    # Write file
    output_path.write_text(content)
    logger.info(f"Output written to: {output_path}")

    return output_path


def _format_markdown(results: list[OCRResult], source: Path, model: str) -> str:
    """Format results as markdown with YAML frontmatter."""
    now = datetime.now(timezone.utc).isoformat()
    avg_conf = _calculate_average_confidence(results)

    lines = [
        "---",
        f"source: {source.name}",
        f"date_processed: {now}",
        f"pages: {len(results)}",
        f"confidence_avg: {avg_conf:.2f}" if avg_conf else "confidence_avg: null",
        f"model: {model}",
        "---",
        "",
    ]

    for result in results:
        lines.extend([
            f"# Page {result.page_num}",
            "",
            result.text,
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


def _format_json(results: list[OCRResult], source: Path, model: str) -> str:
    """Format results as JSON."""
    now = datetime.now(timezone.utc).isoformat()
    avg_conf = _calculate_average_confidence(results)

    data = {
        "metadata": {
            "source": source.name,
            "date_processed": now,
            "pages": len(results),
            "confidence_avg": round(avg_conf, 2) if avg_conf else None,
            "model": model,
        },
        "pages": [
            {
                "page_number": r.page_num,
                "text": r.text,
                "confidence": r.confidence,
            }
            for r in results
        ],
    }

    return json.dumps(data, indent=2)


def _format_txt(results: list[OCRResult], source: Path) -> str:
    """Format results as plain text."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = [
        f"[{source.name} - Processed {now}]",
        "",
    ]

    for result in results:
        lines.extend([
            f"=== Page {result.page_num} ===",
            "",
            result.text,
            "",
        ])

    return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_writer.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add remarkable_ocr/ocr.py remarkable_ocr/writer.py tests/test_writer.py
git commit -m "feat(writer): add markdown, JSON, and plain text output formats"
```

---

## Task 7: OCR Module (Ollama Integration)

**Files:**
- Modify: `remarkable_ocr/ocr.py`
- Create: `tests/test_ocr.py`

**Step 1: Write failing tests for OCR**

`tests/test_ocr.py`:
```python
"""Tests for OCR processing."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    return Image.new("RGB", (100, 100), color="white")


def test_ocr_result_dataclass():
    """OCRResult should store text, confidence, and page number."""
    from remarkable_ocr.ocr import OCRResult

    result = OCRResult(text="Hello", confidence=0.95, page_num=1)

    assert result.text == "Hello"
    assert result.confidence == 0.95
    assert result.page_num == 1


def test_ocr_result_none_confidence():
    """OCRResult should accept None confidence."""
    from remarkable_ocr.ocr import OCRResult

    result = OCRResult(text="Hello", confidence=None, page_num=1)
    assert result.confidence is None


def test_build_prompt():
    """_build_prompt should return the OCR prompt."""
    from remarkable_ocr.ocr import _build_prompt

    prompt = _build_prompt()

    assert "handwriting recognition" in prompt.lower()
    assert "transcribe" in prompt.lower()


@patch("remarkable_ocr.ocr.ollama")
def test_check_ollama_health_success(mock_ollama):
    """check_ollama_health should return True when Ollama is available."""
    from remarkable_ocr.ocr import check_ollama_health

    mock_ollama.list.return_value = {"models": [{"name": "qwen2.5-vl:7b"}]}

    result = check_ollama_health("http://localhost:11434", "qwen2.5-vl:7b")

    assert result is True


@patch("remarkable_ocr.ocr.ollama")
def test_check_ollama_health_model_missing(mock_ollama):
    """check_ollama_health should return False when model not found."""
    from remarkable_ocr.ocr import check_ollama_health

    mock_ollama.list.return_value = {"models": [{"name": "other-model"}]}

    result = check_ollama_health("http://localhost:11434", "qwen2.5-vl:7b")

    assert result is False


@patch("remarkable_ocr.ocr.ollama")
def test_check_ollama_health_connection_error(mock_ollama):
    """check_ollama_health should return False on connection error."""
    from remarkable_ocr.ocr import check_ollama_health

    mock_ollama.list.side_effect = Exception("Connection refused")

    result = check_ollama_health("http://localhost:11434", "qwen2.5-vl:7b")

    assert result is False


@patch("remarkable_ocr.ocr.ollama")
def test_process_image_success(mock_ollama, sample_image):
    """process_image should return OCRResult with extracted text."""
    from remarkable_ocr.ocr import process_image

    mock_ollama.chat.return_value = {
        "message": {"content": "Extracted text from image"}
    }

    result = process_image(
        image=sample_image,
        page_num=1,
        model="qwen2.5-vl:7b",
        host="http://localhost:11434",
        timeout=60,
    )

    assert result.text == "Extracted text from image"
    assert result.page_num == 1


@patch("remarkable_ocr.ocr.ollama")
def test_process_image_blank_page(mock_ollama, sample_image):
    """process_image should handle blank page response."""
    from remarkable_ocr.ocr import process_image

    mock_ollama.chat.return_value = {
        "message": {"content": "[blank page]"}
    }

    result = process_image(
        image=sample_image,
        page_num=1,
        model="qwen2.5-vl:7b",
        host="http://localhost:11434",
        timeout=60,
    )

    assert result.text == "[blank page]"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ocr.py -v`
Expected: FAIL (functions not defined)

**Step 3: Write full implementation**

`remarkable_ocr/ocr.py`:
```python
"""OCR processing using Ollama."""

import base64
import io
from dataclasses import dataclass

import ollama
from PIL import Image

from remarkable_ocr.logging import get_logger

logger = get_logger("ocr")


@dataclass
class OCRResult:
    """Result from OCR processing of a single page."""

    text: str
    confidence: float | None
    page_num: int


def _build_prompt() -> str:
    """Build the OCR prompt for handwriting recognition."""
    return """You are a handwriting recognition system. Extract all handwritten text from this image.

Rules:
- Transcribe exactly what is written, preserving the author's words
- Maintain paragraph breaks and list structure
- Use markdown formatting: **bold** for emphasized/underlined text, - for bullet points
- If text is unclear, make your best attempt (do not skip words)
- For diagrams or sketches, briefly describe them in [brackets]
- If the page is blank or contains no text, respond with: [blank page]

Output only the extracted text, no commentary."""


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def check_ollama_health(host: str, model: str) -> bool:
    """Check if Ollama is reachable and model is available.

    Args:
        host: Ollama API host URL
        model: Model name to check

    Returns:
        True if Ollama is healthy and model available
    """
    try:
        # Set the host
        client = ollama.Client(host=host)
        response = client.list()

        # Check if model is available
        available_models = [m.get("name", "") for m in response.get("models", [])]

        # Handle model name variations (with/without tag)
        model_base = model.split(":")[0]
        for available in available_models:
            if available.startswith(model_base):
                logger.debug(f"Found model: {available}")
                return True

        logger.warning(f"Model {model} not found. Available: {available_models}")
        return False

    except Exception as e:
        logger.error(f"Failed to connect to Ollama at {host}: {e}")
        return False


def get_available_models(host: str) -> list[str]:
    """Get list of available vision models from Ollama.

    Args:
        host: Ollama API host URL

    Returns:
        List of model names
    """
    try:
        client = ollama.Client(host=host)
        response = client.list()
        models = [m.get("name", "") for m in response.get("models", [])]
        # Filter to likely vision models
        vision_keywords = ["vl", "vision", "llava", "bakllava"]
        vision_models = [
            m for m in models
            if any(kw in m.lower() for kw in vision_keywords)
        ]
        return vision_models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return []


def process_image(
    image: Image.Image,
    page_num: int,
    model: str,
    host: str,
    timeout: int,
) -> OCRResult:
    """Process a single image through OCR.

    Args:
        image: PIL Image to process
        page_num: Page number (1-indexed)
        model: Ollama model name
        host: Ollama API host URL
        timeout: Timeout in seconds

    Returns:
        OCRResult with extracted text
    """
    logger.debug(f"Processing page {page_num} ({image.width}x{image.height}px)")

    client = ollama.Client(host=host, timeout=timeout)
    prompt = _build_prompt()

    # Convert image to base64
    image_b64 = _image_to_base64(image)

    try:
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            options={
                "temperature": 0.1,
                "num_predict": 4096,
            },
        )

        text = response["message"]["content"]
        logger.debug(f"Page {page_num}: extracted {len(text)} characters")

        return OCRResult(
            text=text,
            confidence=None,  # Qwen2.5-VL doesn't provide confidence
            page_num=page_num,
        )

    except Exception as e:
        logger.error(f"OCR failed for page {page_num}: {e}")
        return OCRResult(
            text="[OCR Failed]",
            confidence=None,
            page_num=page_num,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ocr.py -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add remarkable_ocr/ocr.py tests/test_ocr.py
git commit -m "feat(ocr): add Ollama integration with health check and image processing"
```

---

## Task 8: CLI Module

**Files:**
- Create: `remarkable_ocr/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write failing tests for CLI**

`tests/test_cli.py`:
```python
"""Tests for CLI commands."""

from pathlib import Path
from unittest.mock import patch, MagicMock

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write implementation**

`remarkable_ocr/cli.py`:
```python
"""Command-line interface using Typer."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from remarkable_ocr import __version__
from remarkable_ocr.config import Settings
from remarkable_ocr.logging import setup_logging, get_logger
from remarkable_ocr.ocr import check_ollama_health, get_available_models, process_image
from remarkable_ocr.pdf import extract_pages, get_page_count
from remarkable_ocr.processor import clean_text, is_blank_page
from remarkable_ocr.writer import write_output

app = typer.Typer(
    name="remarkable-ocr",
    help="Extract handwritten notes from Remarkable PDF exports using local OCR.",
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"remarkable-ocr {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-V", callback=version_callback, is_eager=True),
    ] = None,
):
    """Remarkable OCR - Extract handwritten notes from Remarkable PDFs."""
    pass


@app.command()
def health(
    model: Annotated[str, typer.Option("--model", "-m")] = "qwen2.5-vl:7b",
    host: Annotated[str, typer.Option("--host")] = "http://localhost:11434",
):
    """Check if Ollama is running and model is available."""
    console.print(f"Checking Ollama at {host}...")

    if check_ollama_health(host, model):
        console.print(f"[green]✓[/green] Ollama is reachable")
        console.print(f"[green]✓[/green] Model {model} is available")
        raise typer.Exit(0)
    else:
        err_console.print(f"[red]✗[/red] Cannot connect to Ollama or model not found")
        err_console.print(f"\nTry: ollama pull {model}")
        raise typer.Exit(2)


@app.command()
def models(
    host: Annotated[str, typer.Option("--host")] = "http://localhost:11434",
):
    """List available vision models from Ollama."""
    available = get_available_models(host)

    if not available:
        console.print("[yellow]No vision models found.[/yellow]")
        console.print("\nTry: ollama pull qwen2.5-vl:7b")
        raise typer.Exit(1)

    console.print("[bold]Available vision models:[/bold]")
    for model in available:
        if "qwen2.5-vl" in model:
            console.print(f"  {model} [dim](recommended)[/dim]")
        else:
            console.print(f"  {model}")


@app.command()
def process(
    pdf_path: Annotated[Path, typer.Argument(help="Path to PDF file")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory")
    ] = Path("./output"),
    model: Annotated[
        str, typer.Option("--model", "-m", help="Ollama model to use")
    ] = "qwen2.5-vl:7b",
    host: Annotated[
        str, typer.Option("--host", help="Ollama API host")
    ] = "http://localhost:11434",
    confidence_threshold: Annotated[
        float, typer.Option("--confidence-threshold", help="Minimum confidence")
    ] = 0.5,
    output_format: Annotated[
        str, typer.Option("--output-format", "-f", help="Output format")
    ] = "markdown",
    timeout: Annotated[
        int, typer.Option("--timeout", help="Timeout per page in seconds")
    ] = 60,
    keep_temp_images: Annotated[
        bool, typer.Option("--keep-temp-images", help="Keep temporary images")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug logging")
    ] = False,
):
    """Process a Remarkable PDF and extract handwritten text."""
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    logger = get_logger("cli")

    # Validate input
    if not pdf_path.exists():
        err_console.print(f"[red]Error:[/red] File not found: {pdf_path}")
        raise typer.Exit(2)

    if not pdf_path.suffix.lower() == ".pdf":
        err_console.print(f"[red]Error:[/red] Not a PDF file: {pdf_path}")
        raise typer.Exit(2)

    # Check Ollama
    console.print(f"Checking Ollama...")
    if not check_ollama_health(host, model):
        err_console.print(f"[red]Error:[/red] Cannot connect to Ollama or model {model} not found")
        err_console.print(f"\nMake sure Ollama is running: ollama serve")
        err_console.print(f"And pull the model: ollama pull {model}")
        raise typer.Exit(2)

    # Get page count
    page_count = get_page_count(pdf_path)
    console.print(f"Processing [bold]{pdf_path.name}[/bold] ({page_count} pages)")

    # Process pages
    results = []
    failed_pages = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=page_count)

        for page_num, image in extract_pages(pdf_path):
            progress.update(task, description=f"Page {page_num}/{page_count}")

            result = process_image(
                image=image,
                page_num=page_num,
                model=model,
                host=host,
                timeout=timeout,
            )

            # Clean the text
            result.text = clean_text(result.text)

            if result.text == "[OCR Failed]":
                failed_pages += 1
                logger.warning(f"Page {page_num} failed")

            results.append(result)
            progress.advance(task)

    # Write output
    output_path = write_output(
        results=results,
        source=pdf_path,
        output_dir=output,
        output_format=output_format,
        model=model,
    )

    # Summary
    success_pages = page_count - failed_pages
    console.print()
    console.print(f"[green]✓[/green] Completed: {output_path}")
    console.print(f"  Pages: {success_pages}/{page_count} successful")

    if failed_pages > 0:
        console.print(f"  [yellow]Warning: {failed_pages} pages failed[/yellow]")
        raise typer.Exit(1)

    raise typer.Exit(0)


if __name__ == "__main__":
    app()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add remarkable_ocr/cli.py tests/test_cli.py
git commit -m "feat(cli): add process, health, and models commands"
```

---

## Task 9: Integration Testing & Polish

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/test_integration.py`
- Create: `README.md`

**Step 1: Add integration test marker to conftest**

`tests/conftest.py`:
```python
"""Shared pytest fixtures."""

import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require Ollama)"
    )


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return FIXTURES_DIR
```

**Step 2: Create integration test (skipped by default)**

`tests/test_integration.py`:
```python
"""Integration tests requiring Ollama."""

from pathlib import Path

import pytest


@pytest.mark.integration
def test_full_pipeline(fixtures_dir: Path, tmp_path: Path):
    """Test full PDF to text pipeline with real Ollama."""
    from remarkable_ocr.ocr import check_ollama_health, process_image
    from remarkable_ocr.pdf import extract_pages
    from remarkable_ocr.processor import clean_text
    from remarkable_ocr.writer import write_output

    host = "http://localhost:11434"
    model = "qwen2.5-vl:7b"

    # Skip if Ollama not available
    if not check_ollama_health(host, model):
        pytest.skip("Ollama not available")

    pdf_path = fixtures_dir / "sample_1page.pdf"
    results = []

    for page_num, image in extract_pages(pdf_path):
        result = process_image(
            image=image,
            page_num=page_num,
            model=model,
            host=host,
            timeout=60,
        )
        result.text = clean_text(result.text)
        results.append(result)

    output_path = write_output(
        results=results,
        source=pdf_path,
        output_dir=tmp_path,
        output_format="markdown",
        model=model,
    )

    assert output_path.exists()
    content = output_path.read_text()
    assert "Page 1" in content
```

**Step 3: Create README**

`README.md`:
```markdown
# Remarkable OCR

Extract handwritten notes from Remarkable tablet PDF exports using local OCR.

## Features

- **Local Processing**: All OCR runs locally via Ollama - no cloud APIs
- **High Accuracy**: Uses Qwen2.5-VL vision model optimized for handwriting
- **Multiple Formats**: Output as Markdown, JSON, or plain text
- **Batch Processing**: Process multiple PDFs in one command

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/remarkable-ocr.git
cd remarkable-ocr

# Install in development mode
pip install -e ".[dev]"

# Pull the vision model
ollama pull qwen2.5-vl:7b
```

## Usage

### Basic Usage

```bash
# Process a single PDF
remarkable-ocr process notes.pdf

# Output goes to ./output/notes.md by default
```

### Options

```bash
remarkable-ocr process notes.pdf \
  --output ./my-notes/ \
  --model qwen2.5-vl:7b \
  --output-format markdown \
  --verbose
```

### Available Commands

```bash
# Check Ollama status
remarkable-ocr health

# List available vision models
remarkable-ocr models

# Process PDF
remarkable-ocr process <pdf_path> [options]
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | `./output` | Output directory |
| `--model, -m` | `qwen2.5-vl:7b` | Ollama model |
| `--output-format, -f` | `markdown` | Format: markdown, json, txt |
| `--confidence-threshold` | `0.5` | Minimum confidence |
| `--timeout` | `60` | Seconds per page |
| `--verbose, -v` | False | Debug logging |

## Environment Variables

```bash
REMARKABLE_OCR_MODEL=qwen2.5-vl:7b
REMARKABLE_OCR_OLLAMA_HOST=http://localhost:11434
REMARKABLE_OCR_OUTPUT_DIR=./output
REMARKABLE_OCR_TIMEOUT=60
```

## Output Formats

### Markdown (default)

```markdown
---
source: notes.pdf
date_processed: 2026-01-10T14:30:00Z
pages: 3
confidence_avg: 0.78
model: qwen2.5-vl:7b
---

# Page 1

[Extracted text...]
```

### JSON

```json
{
  "metadata": { "source": "notes.pdf", "pages": 3 },
  "pages": [{ "page_number": 1, "text": "..." }]
}
```

## Development

```bash
# Run tests (unit tests only)
pytest -m "not integration"

# Run all tests (requires Ollama)
pytest

# Run with coverage
pytest --cov=remarkable_ocr
```

## Troubleshooting

**Ollama not found**
```bash
# Make sure Ollama is running
ollama serve

# Pull the required model
ollama pull qwen2.5-vl:7b
```

**Slow processing**
- Use the smaller model: `--model qwen2.5-vl:3b`
- Ensure GPU is being used by Ollama

## License

MIT
```

**Step 4: Run all tests**

Run: `pytest -m "not integration" -v`
Expected: All unit tests pass

**Step 5: Verify CLI works**

Run: `remarkable-ocr --help`
Expected: Shows help with all commands

Run: `remarkable-ocr health`
Expected: Shows Ollama status

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add integration tests and README documentation"
```

---

## Task 10: Final Verification

**Step 1: Run full test suite**

```bash
pytest -m "not integration" --cov=remarkable_ocr --cov-report=term-missing
```

Expected: All tests pass with good coverage

**Step 2: Verify package installation**

```bash
pip install -e . --force-reinstall
remarkable-ocr --version
remarkable-ocr health
```

**Step 3: Test with a real PDF (if available)**

```bash
remarkable-ocr process /path/to/your/notes.pdf --verbose
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final verification and cleanup"
```

---

## Summary

| Task | Module | Tests | Estimated LOC |
|------|--------|-------|---------------|
| 1 | Project scaffolding | - | ~50 |
| 2 | Config | 3 | ~50 |
| 3 | Logging | 3 | ~60 |
| 4 | PDF | 5 | ~70 |
| 5 | Processor | 6 | ~50 |
| 6 | Writer | 5 | ~100 |
| 7 | OCR | 9 | ~120 |
| 8 | CLI | 6 | ~180 |
| 9 | Integration | 1 | ~50 |
| 10 | Verification | - | - |

**Total: ~730 lines of code, 38 tests**
