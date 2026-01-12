# GTK4 GUI Frontend Design

## Overview

A simple GTK4 desktop application for remarkable-ocr that provides drag-and-drop file processing with in-window result preview and editing before save.

## Decisions

| Decision | Choice |
|----------|--------|
| Complexity | Polished single-window with progress, error handling, model dropdown |
| Output handling | Preview then save - user reviews/edits before explicit save |
| Input types | PDFs + images (.pdf, .png, .jpg, .jpeg) |
| Settings exposure | Minimal - model, output format, output path only |

## Window Layout

```
┌─────────────────────────────────────────────────┐
│  Remarkable OCR                            [—][×]│
├─────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────┐  │
│  │                                           │  │
│  │     Drop PDF or image here                │  │
│  │         (or click to browse)              │  │
│  │                                           │  │
│  └───────────────────────────────────────────┘  │
├─────────────────────────────────────────────────┤
│  Model: [ministral-3:14b... ▼]  Format: [md ▼]  │
├─────────────────────────────────────────────────┤
│  [Process]                        ░░░░░░░░ 0%   │
├─────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────┐  │
│  │                                           │  │
│  │  (extracted text appears here)            │  │
│  │  (editable before saving)                 │  │
│  │                                           │  │
│  └───────────────────────────────────────────┘  │
├─────────────────────────────────────────────────┤
│  Output: [~/Documents/output.md    ] [Browse]   │
│                                                 │
│                           [Copy]  [Save]        │
└─────────────────────────────────────────────────┘
```

- Drop zone accepts drag-drop or click-to-browse
- Text area is editable `GtkTextView`
- Progress bar shows page-by-page progress

## Application States

### 1. Idle (startup)
- Drop zone visible and active
- Model dropdown populated from `get_available_models()` on startup
- Process button disabled (no file selected)
- Text area empty or showing placeholder
- Save/Copy buttons disabled

### 2. File Selected
- Drop zone shows filename and page count (for PDFs)
- Process button enabled
- Status shows "Ready to process X pages" or "Ready to process image"

### 3. Processing
- Drop zone disabled (prevents new drops mid-process)
- Process button changes to "Cancel"
- Progress bar animates with page progress (e.g., "Page 2/5")
- Text area shows live output as each page completes
- Background thread keeps UI responsive

### 4. Complete
- Progress bar shows 100%
- Text area contains full editable result
- Save/Copy buttons enabled
- Drop zone re-enabled for next file
- Status shows "Processing complete - review and save"

## Library Integration

### Startup
```python
from remarkable_ocr.ocr import check_ollama_health, get_available_models
from remarkable_ocr.config import Settings

settings = Settings()  # Load defaults from env
models = get_available_models(settings.ollama_host)  # Populate dropdown
```

### Processing (background thread)
```python
from remarkable_ocr.pdf import extract_pages, get_page_count
from remarkable_ocr.ocr import process_image
from remarkable_ocr.processor import clean_text
from PIL import Image

# For PDFs:
for page_num, image in extract_pages(pdf_path):
    result = process_image(image, page_num, model, host, timeout)
    text = clean_text(result.text)
    # Emit signal to update UI with text

# For images:
image = Image.open(image_path)
result = process_image(image, page_num=1, model, host, timeout)
text = clean_text(result.text)
```

### Saving (user-triggered)
```python
from remarkable_ocr.writer import write_output
from remarkable_ocr.ocr import OCRResult

# Build OCRResult from edited text
results = [OCRResult(text=edited_text, confidence=None, page_num=1)]
write_output(results, source_path, output_dir, format, model)
```

Threading uses Python's `threading` module with `GLib.idle_add()` for safe UI updates.

## File Structure

```
remarkable_ocr/
├── __init__.py
├── cli.py
├── config.py
├── ocr.py
├── pdf.py
├── processor.py
├── writer.py
├── chunking.py
├── logging.py
├── prompts/
│   └── ocr_prompt.txt
└── gui/                    # NEW
    ├── __init__.py
    ├── app.py              # Main GtkApplication class
    ├── window.py           # Main window with all widgets
    └── worker.py           # Background processing thread
```

### Entry Point
New CLI command: `remarkable-ocr gui`

### Dependencies
```toml
[project.optional-dependencies]
gui = ["PyGObject>=3.42"]
```

Install with `pip install -e ".[gui]"` or `pip install remarkable-ocr[gui]`.

### System Requirements
GTK4 libraries required. On Fedora:
```bash
dnf install gtk4-devel gobject-introspection-devel
```

## Error Handling

### Ollama connectivity
- Startup: attempt `check_ollama_health()` silently
- If fails: show banner "Ollama not available at localhost:11434" with Retry button
- Model dropdown shows "No models available", Process button disabled

### File validation
- Unsupported types: toast "Unsupported file type. Use PDF, PNG, or JPG"
- Corrupted PDFs: "Could not read PDF file"
- Empty PDFs: "PDF has no pages"

### Processing failures
- Timeout on page: warn, continue to next page, note in output
- Complete Ollama failure: stop, keep partial results, show error
- Cancel: stop after current page, show partial results

### Save errors
- Permission denied: error dialog with path
- Invalid path: error, keep save dialog open
