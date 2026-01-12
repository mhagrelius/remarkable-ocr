# remarkable-ocr

Extract handwritten text from reMarkable tablet PDF exports using local OCR via Ollama.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally

```bash
ollama serve
ollama pull ministral-3:14b-instruct-2512-q8_0
```

## Installation

```bash
pip install remarkable-ocr
```

From source:

```bash
git clone https://github.com/yourusername/remarkable-ocr.git
cd remarkable-ocr
pip install .
```

## Quick Start

```bash
remarkable-ocr health              # Check Ollama connection
remarkable-ocr process notes.pdf   # Extract text to ./output/notes.md
remarkable-ocr gui                 # Launch graphical interface
```

## Graphical Interface

The GUI provides a drag-and-drop interface for processing PDFs and images.

### Installation

Install the GUI dependencies:

```bash
pip install remarkable-ocr[gui]
```

You also need GTK4 and libadwaita system libraries:

**Linux (Fedora/RHEL)**
```bash
sudo dnf install gtk4-devel libadwaita-devel gobject-introspection-devel
```

**Linux (Ubuntu/Debian)**
```bash
sudo apt install libgtk-4-dev libadwaita-1-dev gobject-introspection
```

**Linux (Arch)**
```bash
sudo pacman -S gtk4 libadwaita gobject-introspection
```

**macOS**
```bash
brew install gtk4 libadwaita pygobject3
```

**Windows**

Install [MSYS2](https://www.msys2.org/), then in the MSYS2 terminal:
```bash
pacman -S mingw-w64-x86_64-gtk4 mingw-w64-x86_64-libadwaita mingw-w64-x86_64-python-gobject
```

Add the MSYS2 bin directory to your PATH (typically `C:\msys64\mingw64\bin`).

### Usage

Launch the GUI:

```bash
remarkable-ocr gui
```

Features:
- Drag and drop PDF or image files onto the window
- Select OCR model from dropdown (defaults to your configured model)
- Choose output format (Markdown, JSON, or plain text)
- View and edit extracted text before saving
- Copy results to clipboard or save to file
- Progress updates for each page and chunk

## CLI Commands

### `process`

```bash
remarkable-ocr process <pdf> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `./output` | Output directory |
| `-f, --output-format` | `markdown` | `markdown`, `json`, or `txt` |
| `-m, --model` | `ministral-3:14b-instruct-2512-q8_0` | Ollama vision model |
| `--host` | `http://localhost:11434` | Ollama API URL |
| `--timeout` | `60` | Seconds per page |
| `-v, --verbose` | - | Debug logging |

### `health`

Check Ollama connection and model availability.

```bash
remarkable-ocr health [-m MODEL] [--host HOST]
```

### `models`

List available vision models.

```bash
remarkable-ocr models [--host HOST]
```

## Configuration

Set defaults via environment variables or `.env` file:

```bash
REMARKABLE_OCR_MODEL=ministral-3:14b-instruct-2512-q8_0
REMARKABLE_OCR_OLLAMA_HOST=http://localhost:11434
REMARKABLE_OCR_OUTPUT_DIR=./output
REMARKABLE_OCR_OUTPUT_FORMAT=markdown
REMARKABLE_OCR_TIMEOUT=60
REMARKABLE_OCR_LOG_LEVEL=INFO
```

## Library Usage

```python
from pathlib import Path
from remarkable_ocr.pdf import extract_pages
from remarkable_ocr.ocr import process_image, check_ollama_health
from remarkable_ocr.processor import clean_text

# Verify Ollama is ready
if not check_ollama_health("http://localhost:11434", "ministral-3:14b-instruct-2512-q8_0"):
    raise RuntimeError("Ollama not available")

# Process PDF
for page_num, image in extract_pages(Path("notes.pdf")):
    result = process_image(
        image=image,
        page_num=page_num,
        model="ministral-3:14b-instruct-2512-q8_0",
        host="http://localhost:11434",
        timeout=60,
    )
    print(f"Page {page_num}: {clean_text(result.text)}")
```

### API Reference

| Function | Returns | Description |
|----------|---------|-------------|
| `pdf.extract_pages(path)` | `Iterator[(int, Image)]` | Yields page images |
| `pdf.get_page_count(path)` | `int` | Page count |
| `ocr.process_image(...)` | `OCRResult` | Extract text from image |
| `ocr.check_ollama_health(host, model)` | `bool` | Check Ollama status |
| `ocr.get_available_models(host)` | `list[str]` | List vision models |
| `processor.clean_text(raw)` | `str` | Fix OCR artifacts |
| `writer.write_output(...)` | `Path` | Write results to file |

## Output Formats

**Markdown** (default): YAML frontmatter + page sections

```markdown
---
source: notes.pdf
date_processed: 2026-01-10T14:30:00Z
pages: 3
model: ministral-3:14b-instruct-2512-q8_0
---

# Page 1

[extracted text]
```

**JSON**: Structured with metadata and pages array

**Plain text**: Simple page markers

## Troubleshooting

**Cannot connect to Ollama**
```bash
ollama serve           # Start Ollama
ollama pull ministral-3:14b-instruct-2512-q8_0  # Pull model
```

**Slow processing**: Use a smaller model (`ministral-3:latest`) or ensure GPU acceleration is enabled in Ollama.

## Development

```bash
pip install -e ".[dev]"
pytest -m "not integration"  # Unit tests only
pytest                       # All tests (requires Ollama)
pytest --cov=remarkable_ocr  # With coverage
```

## License

MIT
