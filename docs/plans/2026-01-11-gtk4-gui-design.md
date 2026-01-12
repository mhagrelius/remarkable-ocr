# GTK4 + Libadwaita GUI Frontend Design

## Overview

A modern GNOME desktop application for remarkable-ocr using GTK4 and libadwaita. Provides drag-and-drop file processing with in-window result preview and editing before save.

## Decisions

| Decision | Choice |
|----------|--------|
| Toolkit | GTK4 + libadwaita (modern GNOME HIG styling) |
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
- Text area is editable `Gtk.TextView`
- Progress bar shows page-by-page progress
- Toast notifications via `Adw.ToastOverlay` for errors/success

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
- Toast notification: "Processing complete"

## GTK4/Libadwaita Patterns

### Application Setup
```python
import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
from gi.repository import Gtk, Adw, Gdk, Gio, GLib, GObject

class RemarkableOCRApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id='com.github.remarkable-ocr')

    def do_activate(self):
        win = MainWindow(application=self)
        win.present()
```

### File Drop Target
```python
drop_target = Gtk.DropTarget.new(GObject.TYPE_NONE, Gdk.DragAction.COPY)
drop_target.set_gtypes([Gdk.FileList])
drop_target.connect('drop', self.on_drop)
self.drop_zone.add_controller(drop_target)

def on_drop(self, _ctrl, value, _x, _y):
    if isinstance(value, Gdk.FileList):
        files = value.get_files()
        if files:
            self.load_file(files[0].get_path())
    return True
```

### File Dialog (click to browse)
```python
def open_file_dialog(self):
    dialog = Gtk.FileDialog()

    # Set up filters
    filters = Gio.ListStore.new(Gtk.FileFilter)

    f = Gtk.FileFilter()
    f.set_name("Supported files")
    f.add_mime_type("application/pdf")
    f.add_mime_type("image/png")
    f.add_mime_type("image/jpeg")
    filters.append(f)

    dialog.set_filters(filters)
    dialog.set_default_filter(f)
    dialog.open(self, None, self.on_file_dialog_response)

def on_file_dialog_response(self, dialog, result):
    try:
        file = dialog.open_finish(result)
        self.load_file(file.get_path())
    except GLib.Error:
        pass  # User cancelled
```

### Threading with UI Updates
```python
import threading

def start_processing(self):
    self.cancel_requested = False
    thread = threading.Thread(target=self.process_worker, daemon=True)
    thread.start()

def process_worker(self):
    try:
        for page_num, image in extract_pages(self.file_path):
            if self.cancel_requested:
                GLib.idle_add(self.on_cancelled)
                return

            result = process_image(image, page_num, self.model, ...)
            text = clean_text(result.text)

            # Safe UI update from thread
            GLib.idle_add(self.append_result, page_num, text)

        GLib.idle_add(self.on_complete)
    except Exception as e:
        GLib.idle_add(self.on_error, str(e))
```

### Toast Notifications
```python
# Window setup with toast overlay
self.toast_overlay = Adw.ToastOverlay()
self.set_content(self.toast_overlay)
self.toast_overlay.set_child(main_content)

# Show toast
def show_toast(self, message):
    toast = Adw.Toast(title=message)
    self.toast_overlay.add_toast(toast)
```

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
    GLib.idle_add(update_ui, page_num, text)

# For images:
image = Image.open(image_path)
result = process_image(image, page_num=1, ...)
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
    ├── app.py              # Adw.Application subclass
    ├── window.py           # Adw.ApplicationWindow with all widgets
    └── worker.py           # Background processing thread
```

### Entry Point
New CLI command: `remarkable-ocr gui`

### Python Dependencies
```toml
[project.optional-dependencies]
gui = ["PyGObject>=3.46"]
```

Install with `pip install -e ".[gui]"` or `pip install remarkable-ocr[gui]`.

### System Requirements (Fedora)
```bash
sudo dnf install gtk4-devel libadwaita-devel gobject-introspection-devel
```

### System Requirements (Ubuntu/Debian)
```bash
sudo apt install libgtk-4-dev libadwaita-1-dev gobject-introspection
```

## Error Handling

### Ollama connectivity
- Startup: attempt `check_ollama_health()` silently
- If fails: show banner "Ollama not available" with Retry button
- Model dropdown shows "No models available", Process button disabled

### File validation
- Unsupported types: toast "Unsupported file type. Use PDF, PNG, or JPG"
- Corrupted PDFs: toast "Could not read PDF file"
- Empty PDFs: toast "PDF has no pages"

### Processing failures
- Timeout on page: warn via toast, continue to next page
- Complete Ollama failure: stop, keep partial results, show error toast
- Cancel: stop after current page, show partial results

### Save errors
- Permission denied: error toast with path
- Invalid path: error toast, file dialog stays open
