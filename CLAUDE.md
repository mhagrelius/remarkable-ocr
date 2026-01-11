# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all unit tests (excludes integration tests)
pytest -m "not integration"

# Run all tests including integration (requires Ollama with qwen2.5-vl:7b)
pytest

# Run a single test file
pytest tests/test_ocr.py

# Run a single test function
pytest tests/test_ocr.py::test_check_ollama_health_success

# Run with coverage
pytest --cov=remarkable_ocr

# Run the CLI
remarkable-ocr --help
remarkable-ocr process <pdf_path>
remarkable-ocr health
remarkable-ocr models
```

## Architecture

This is a CLI tool that extracts handwritten text from Remarkable tablet PDF exports using local OCR via Ollama.

### Processing Pipeline

1. **PDF → Images** (`pdf.py`): Uses pymupdf (fitz) to render PDF pages as PIL Images at 1288px width
2. **Images → Text** (`ocr.py`): Sends images to Ollama vision model (qwen2.5-vl) for handwriting recognition
3. **Text Cleaning** (`processor.py`): Fixes common OCR artifacts (rn→m, 0→O/o patterns)
4. **Output Writing** (`writer.py`): Formats results as markdown (with YAML frontmatter), JSON, or plain text

### Key Components

- **`cli.py`**: Typer-based CLI with three commands: `process`, `health`, `models`
- **`config.py`**: Pydantic Settings for env vars (prefix: `REMARKABLE_OCR_`)
- **`ocr.py`**: `OCRResult` dataclass, Ollama client wrapper, prompt engineering for handwriting
- **`logging.py`**: Rich console output + file logging to `~/.remarkable_ocr/logs/`

### Test Markers

- Integration tests are marked with `@pytest.mark.integration` and require a running Ollama instance
- Unit tests mock the Ollama client using `unittest.mock`
