# Remarkable OCR Design Document

**Date:** 2026-01-10
**Status:** Approved
**Scope:** MVP (Phase 1)

---

## Overview

A CLI application that extracts handwritten notes from Remarkable tablet PDF exports using local OCR via Qwen2.5-VL through Ollama.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| PDF processing | pymupdf | No system deps, faster than pdf2image |
| Project structure | Flat single package | Simple for MVP, easy to navigate |
| Installation | pip + script | `pip install -e .` and `python -m` both work |
| Ollama client | ollama-python | Official library, handles retries, typed |
| Configuration | CLI flags + env vars | CLI overrides env, no config file for MVP |

---

## Project Structure

```
remarkable-ocr/
├── pyproject.toml           # Package config, dependencies, entry points
├── README.md                # Setup + usage docs
├── .env.example             # Template for env vars
├── remarkable_ocr/
│   ├── __init__.py          # Version, package metadata
│   ├── __main__.py          # Enables `python -m remarkable_ocr`
│   ├── cli.py               # Typer app, all commands
│   ├── config.py            # Pydantic settings (env + CLI merge)
│   ├── pdf.py               # PDF loading, image extraction (pymupdf)
│   ├── ocr.py               # Ollama wrapper, prompt, inference
│   ├── processor.py         # Text cleaning, formatting
│   ├── writer.py            # Output to markdown/json/txt
│   └── logging.py           # Logging setup with rich
└── tests/
    ├── conftest.py          # Shared fixtures
    ├── fixtures/            # Sample PDFs for testing
    ├── test_pdf.py
    ├── test_ocr.py
    ├── test_processor.py
    └── test_cli.py
```

---

## Data Flow

```
CLI (cli.py)
    │
    ▼
Config (config.py) ─── Load env vars, merge CLI flags, validate
    │
    ▼
PDF Processor (pdf.py) ─── Open PDF, render pages to PIL Images (in memory)
    │
    ▼
OCR Engine (ocr.py) ─── Send to Ollama, get text + confidence
    │
    ▼
Text Processor (processor.py) ─── Clean artifacts, normalize whitespace
    │
    ▼
Writer (writer.py) ─── Output markdown/json/txt with metadata
```

---

## Module Interfaces

### config.py

```python
class Settings(BaseSettings):
    model: str = "qwen2.5-vl:7b"
    ollama_host: str = "http://localhost:11434"
    output_dir: Path = Path("./output")
    confidence_threshold: float = 0.5
    timeout: int = 60
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="REMARKABLE_OCR_")
```

### pdf.py

```python
def extract_pages(pdf_path: Path, width: int = 1288) -> Iterator[tuple[int, Image.Image]]:
    """Yield (page_number, image) for each page. 1-indexed."""

def save_temp_images(pdf_path: Path, cache_dir: Path) -> list[Path]:
    """Save page images to disk. Returns list of image paths."""
```

### ocr.py

```python
@dataclass
class OCRResult:
    text: str
    confidence: float | None
    page_num: int

def check_ollama_health(host: str) -> bool:
    """Return True if Ollama is reachable and model available."""

def process_image(image: Image.Image, page_num: int, settings: Settings) -> OCRResult:
    """Send image to Ollama, return extracted text."""
```

### processor.py

```python
def clean_text(raw: str) -> str:
    """Fix OCR artifacts, normalize whitespace, preserve structure."""

def is_blank_page(text: str) -> bool:
    """Return True if page has no meaningful content."""
```

### writer.py

```python
def write_output(results: list[OCRResult], source: Path, settings: Settings) -> Path:
    """Write results to file in configured format. Returns output path."""
```

---

## CLI Commands

### process

```bash
remarkable-ocr process <pdf_path> [options]

Options:
  --output, -o PATH              Output directory (default: ./output)
  --model, -m TEXT               Ollama model (default: qwen2.5-vl:7b)
  --confidence-threshold FLOAT   Min confidence (default: 0.5)
  --output-format [markdown|json|txt]  Output format (default: markdown)
  --keep-temp-images             Preserve intermediate images
  --verbose, -v                  Debug logging
```

### health

```bash
remarkable-ocr health
# Checks: Ollama reachable, model available, output dir writable
```

### models

```bash
remarkable-ocr models
# Lists available vision models from Ollama
```

### Exit codes

- 0: Success
- 1: Partial failure (some pages failed)
- 2: Fatal error (Ollama unavailable, invalid input)

---

## OCR Prompt

```
You are a handwriting recognition system. Extract all handwritten text from this image.

Rules:
- Transcribe exactly what is written, preserving the author's words
- Maintain paragraph breaks and list structure
- Use markdown formatting: **bold** for emphasized/underlined text, - for bullet points
- If text is unclear, make your best attempt (do not skip words)
- For diagrams or sketches, briefly describe them in [brackets]
- If the page is blank or contains no text, respond with: [blank page]

Output only the extracted text, no commentary.
```

**Model settings:**
- temperature: 0.1 (deterministic)
- num_predict: 4096 (allow long responses)

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Ollama not running | Exit with error, code 2 |
| Model not found | Suggest fallback, exit code 2 |
| PDF file not found | Log, skip file, continue |
| Corrupted PDF page | Log warning, skip page, continue |
| OCR timeout (60s) | Retry once, then skip with warning |
| Output dir missing | Create automatically |

**Partial success:** Write output with `[OCR Failed]` placeholder for failed pages.

---

## Logging

- **File:** `~/.remarkable_ocr/logs/remarkable_ocr.log`
- **Console:** Rich colored output
- **Levels:** DEBUG, INFO, WARN, ERROR
- **Default:** INFO (progress, warnings, errors)
- **Verbose:** DEBUG (per-page timing, dimensions)

---

## Dependencies

```toml
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
```

**Prerequisites:**
- Python 3.10+
- Ollama with qwen2.5-vl:7b

---

## Testing Strategy

**Unit tests (no Ollama):**
- PDF loading, image extraction
- Text cleaning, artifact fixes
- Output formatting
- Config loading

**Integration tests (require Ollama):**
- Marked with `@pytest.mark.integration`
- End-to-end with sample PDFs

**Fixtures:**
- `sample_1page.pdf` - Simple note
- `sample_blank.pdf` - Blank page
- `sample_mixed.pdf` - Text + diagram

---

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

[Extracted text]

---

# Page 2

[Extracted text]
```

### JSON

```json
{
  "metadata": {
    "source": "notes.pdf",
    "date_processed": "2026-01-10T14:30:00Z",
    "pages": 3,
    "confidence_avg": 0.78,
    "model": "qwen2.5-vl:7b"
  },
  "pages": [
    {"page_number": 1, "text": "...", "confidence": 0.82}
  ]
}
```

### Plain text

```
[notes.pdf - Processed 2026-01-10]

=== Page 1 ===

[Extracted text]
```

---

## Out of Scope (MVP)

- Vector embeddings / RAG
- Config file (`~/.remarkable_ocr/config.yaml`)
- Parallel page processing
- Fine-tuning on user handwriting
- Multi-language support
- Web UI
