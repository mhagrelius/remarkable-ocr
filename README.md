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
