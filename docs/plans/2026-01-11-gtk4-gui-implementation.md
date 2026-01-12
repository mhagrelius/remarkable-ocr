# GTK4 GUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a GTK4 + libadwaita GUI frontend with drag-drop file input, OCR processing, and editable result preview.

**Architecture:** Single-window app using Adw.ApplicationWindow. Drop zone accepts PDFs/images, processing runs in background thread with GLib.idle_add() for UI updates, results shown in editable TextView before explicit save.

**Tech Stack:** GTK4, libadwaita, PyGObject, threading

---

## Task 1: Add GUI Dependencies

**Files:**
- Modify: `pyproject.toml:39-43`

**Step 1: Add gui optional dependency**

Add after line 43 (after the `dev` section):

```toml
gui = ["PyGObject>=3.46"]
```

**Step 2: Verify syntax**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`
Expected: No output (success)

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add gui optional dependency for PyGObject"
```

---

## Task 2: Create GUI Module Structure

**Files:**
- Create: `remarkable_ocr/gui/__init__.py`

**Step 1: Create gui directory and __init__.py**

```python
"""GTK4 + libadwaita GUI for Remarkable OCR."""

from remarkable_ocr.gui.app import RemarkableOCRApp

__all__ = ["RemarkableOCRApp"]
```

**Step 2: Verify import structure**

Run: `python -c "from remarkable_ocr.gui import RemarkableOCRApp" 2>&1 || echo "Expected to fail - app.py not created yet"`
Expected: Import error (app.py doesn't exist yet)

**Step 3: Commit**

```bash
git add remarkable_ocr/gui/__init__.py
git commit -m "feat(gui): add gui module structure"
```

---

## Task 3: Create Application Class

**Files:**
- Create: `remarkable_ocr/gui/app.py`

**Step 1: Write the application class**

```python
"""Main GTK4 application."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gio

from remarkable_ocr.gui.window import MainWindow


class RemarkableOCRApp(Adw.Application):
    """Main application class."""

    def __init__(self) -> None:
        super().__init__(
            application_id="com.github.remarkable-ocr",
            flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
        )

    def do_activate(self) -> None:
        """Handle application activation."""
        win = self.props.active_window
        if not win:
            win = MainWindow(application=self)
        win.present()


def main() -> None:
    """Entry point for the GUI application."""
    app = RemarkableOCRApp()
    app.run(None)
```

**Step 2: Verify syntax**

Run: `python -m py_compile remarkable_ocr/gui/app.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add remarkable_ocr/gui/app.py
git commit -m "feat(gui): add Adw.Application class"
```

---

## Task 4: Create Worker Thread Class

**Files:**
- Create: `remarkable_ocr/gui/worker.py`

**Step 1: Write the worker class**

```python
"""Background processing worker."""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from gi.repository import GLib
from PIL import Image

from remarkable_ocr.config import Settings
from remarkable_ocr.ocr import process_image
from remarkable_ocr.pdf import extract_pages, get_page_count
from remarkable_ocr.processor import clean_text


@dataclass
class ProcessingResult:
    """Result from processing a single page."""

    page_num: int
    total_pages: int
    text: str


@dataclass
class ProcessingError:
    """Error during processing."""

    message: str
    page_num: int | None = None


class OCRWorker:
    """Background worker for OCR processing."""

    def __init__(
        self,
        file_path: Path,
        model: str,
        on_progress: Callable[[ProcessingResult], None],
        on_complete: Callable[[list[ProcessingResult]], None],
        on_error: Callable[[ProcessingError], None],
    ) -> None:
        self.file_path = file_path
        self.model = model
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error

        self._cancel_requested = False
        self._thread: threading.Thread | None = None
        self._settings = Settings()

    def start(self) -> None:
        """Start processing in background thread."""
        self._cancel_requested = False
        self._thread = threading.Thread(target=self._process, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Request cancellation of current processing."""
        self._cancel_requested = True

    def _process(self) -> None:
        """Process the file (runs in background thread)."""
        results: list[ProcessingResult] = []

        try:
            # Determine file type
            suffix = self.file_path.suffix.lower()

            if suffix == ".pdf":
                total_pages = get_page_count(self.file_path)
                pages = extract_pages(self.file_path)
            else:
                # Single image
                total_pages = 1
                image = Image.open(self.file_path)
                pages = [(1, image)]

            for page_num, image in pages:
                if self._cancel_requested:
                    GLib.idle_add(self._emit_complete, results)
                    return

                try:
                    result = process_image(
                        image=image,
                        page_num=page_num,
                        model=self.model,
                        host=self._settings.ollama_host,
                        timeout=self._settings.timeout,
                        num_ctx=self._settings.num_ctx,
                        temperature=self._settings.temperature,
                    )
                    text = clean_text(result.text)
                except Exception as e:
                    error = ProcessingError(
                        message=str(e),
                        page_num=page_num,
                    )
                    GLib.idle_add(self._emit_error, error)
                    text = f"[Error on page {page_num}: {e}]"

                proc_result = ProcessingResult(
                    page_num=page_num,
                    total_pages=total_pages,
                    text=text,
                )
                results.append(proc_result)
                GLib.idle_add(self._emit_progress, proc_result)

            GLib.idle_add(self._emit_complete, results)

        except Exception as e:
            error = ProcessingError(message=str(e))
            GLib.idle_add(self._emit_error, error)

    def _emit_progress(self, result: ProcessingResult) -> bool:
        """Emit progress update (called via GLib.idle_add)."""
        self.on_progress(result)
        return False  # Don't repeat

    def _emit_complete(self, results: list[ProcessingResult]) -> bool:
        """Emit completion (called via GLib.idle_add)."""
        self.on_complete(results)
        return False

    def _emit_error(self, error: ProcessingError) -> bool:
        """Emit error (called via GLib.idle_add)."""
        self.on_error(error)
        return False
```

**Step 2: Verify syntax**

Run: `python -m py_compile remarkable_ocr/gui/worker.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add remarkable_ocr/gui/worker.py
git commit -m "feat(gui): add background OCR worker thread"
```

---

## Task 5: Create Main Window - Part 1 (Structure)

**Files:**
- Create: `remarkable_ocr/gui/window.py`

**Step 1: Write window class with basic structure**

```python
"""Main application window."""

from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk

from remarkable_ocr.config import Settings
from remarkable_ocr.gui.worker import OCRWorker, ProcessingError, ProcessingResult
from remarkable_ocr.ocr import OCRResult, check_ollama_health, get_available_models
from remarkable_ocr.writer import write_output


class MainWindow(Adw.ApplicationWindow):
    """Main application window."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._settings = Settings()
        self._file_path: Path | None = None
        self._worker: OCRWorker | None = None
        self._results: list[ProcessingResult] = []

        self.set_title("Remarkable OCR")
        self.set_default_size(600, 700)

        self._build_ui()
        self._setup_drop_target()
        self._load_models()

    def _build_ui(self) -> None:
        """Build the user interface."""
        # Toast overlay for notifications
        self.toast_overlay = Adw.ToastOverlay()
        self.set_content(self.toast_overlay)

        # Main content box
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.toast_overlay.set_child(main_box)

        # Header bar
        header = Adw.HeaderBar()
        main_box.append(header)

        # Content with margins
        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)
        main_box.append(content)

        # Drop zone
        self._build_drop_zone(content)

        # Settings row (model, format)
        self._build_settings_row(content)

        # Process button and progress
        self._build_process_row(content)

        # Text view (scrollable)
        self._build_text_view(content)

        # Output row (path, save button)
        self._build_output_row(content)

    def _build_drop_zone(self, parent: Gtk.Box) -> None:
        """Build the drop zone widget."""
        frame = Gtk.Frame()
        frame.add_css_class("view")
        parent.append(frame)

        self.drop_zone = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            valign=Gtk.Align.CENTER,
            halign=Gtk.Align.CENTER,
        )
        self.drop_zone.set_size_request(-1, 120)
        frame.set_child(self.drop_zone)

        self.drop_icon = Gtk.Image(icon_name="document-open-symbolic")
        self.drop_icon.set_pixel_size(48)
        self.drop_zone.append(self.drop_icon)

        self.drop_label = Gtk.Label(label="Drop PDF or image here")
        self.drop_label.add_css_class("dim-label")
        self.drop_zone.append(self.drop_label)

        self.drop_sublabel = Gtk.Label(label="or click to browse")
        self.drop_sublabel.add_css_class("dim-label")
        self.drop_sublabel.add_css_class("caption")
        self.drop_zone.append(self.drop_sublabel)

        # Make clickable
        click = Gtk.GestureClick()
        click.connect("pressed", self._on_drop_zone_clicked)
        frame.add_controller(click)

    def _build_settings_row(self, parent: Gtk.Box) -> None:
        """Build the settings row (model, format dropdowns)."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        parent.append(row)

        # Model selector
        model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        row.append(model_box)

        model_label = Gtk.Label(label="Model:")
        model_box.append(model_label)

        self.model_dropdown = Gtk.DropDown()
        self.model_dropdown.set_hexpand(True)
        model_box.append(self.model_dropdown)

        # Format selector
        format_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        row.append(format_box)

        format_label = Gtk.Label(label="Format:")
        format_box.append(format_label)

        formats = Gtk.StringList.new(["markdown", "json", "txt"])
        self.format_dropdown = Gtk.DropDown(model=formats)
        format_box.append(self.format_dropdown)

    def _build_process_row(self, parent: Gtk.Box) -> None:
        """Build the process button and progress bar row."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        parent.append(row)

        self.process_button = Gtk.Button(label="Process")
        self.process_button.add_css_class("suggested-action")
        self.process_button.set_sensitive(False)
        self.process_button.connect("clicked", self._on_process_clicked)
        row.append(self.process_button)

        self.progress_bar = Gtk.ProgressBar()
        self.progress_bar.set_hexpand(True)
        self.progress_bar.set_valign(Gtk.Align.CENTER)
        self.progress_bar.set_show_text(True)
        row.append(self.progress_bar)

    def _build_text_view(self, parent: Gtk.Box) -> None:
        """Build the scrollable text view."""
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_min_content_height(200)
        parent.append(scrolled)

        self.text_view = Gtk.TextView()
        self.text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.text_view.set_monospace(True)
        self.text_view.set_left_margin(8)
        self.text_view.set_right_margin(8)
        self.text_view.set_top_margin(8)
        self.text_view.set_bottom_margin(8)
        scrolled.set_child(self.text_view)

        self.text_buffer = self.text_view.get_buffer()
        self.text_buffer.set_text("Results will appear here...")

    def _build_output_row(self, parent: Gtk.Box) -> None:
        """Build the output path and save buttons row."""
        row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        parent.append(row)

        # Path row
        path_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        row.append(path_row)

        path_label = Gtk.Label(label="Output:")
        path_row.append(path_label)

        self.output_entry = Gtk.Entry()
        self.output_entry.set_hexpand(True)
        self.output_entry.set_text(str(self._settings.output_dir))
        path_row.append(self.output_entry)

        browse_button = Gtk.Button(icon_name="folder-open-symbolic")
        browse_button.connect("clicked", self._on_browse_output_clicked)
        path_row.append(browse_button)

        # Button row
        button_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        button_row.set_halign(Gtk.Align.END)
        row.append(button_row)

        self.copy_button = Gtk.Button(label="Copy")
        self.copy_button.set_sensitive(False)
        self.copy_button.connect("clicked", self._on_copy_clicked)
        button_row.append(self.copy_button)

        self.save_button = Gtk.Button(label="Save")
        self.save_button.add_css_class("suggested-action")
        self.save_button.set_sensitive(False)
        self.save_button.connect("clicked", self._on_save_clicked)
        button_row.append(self.save_button)

    def _setup_drop_target(self) -> None:
        """Set up drag-and-drop for files."""
        drop_target = Gtk.DropTarget.new(GObject.TYPE_NONE, Gdk.DragAction.COPY)
        drop_target.set_gtypes([Gdk.FileList])
        drop_target.connect("drop", self._on_drop)
        drop_target.connect("enter", self._on_drag_enter)
        drop_target.connect("leave", self._on_drag_leave)
        self.drop_zone.get_parent().add_controller(drop_target)

    def _load_models(self) -> None:
        """Load available models from Ollama."""
        try:
            models = get_available_models(self._settings.ollama_host)
            if models:
                model_list = Gtk.StringList.new(models)
                self.model_dropdown.set_model(model_list)

                # Select default model if available
                for i, m in enumerate(models):
                    if m == self._settings.model:
                        self.model_dropdown.set_selected(i)
                        break
            else:
                self._show_toast("No vision models found in Ollama")
                model_list = Gtk.StringList.new(["No models available"])
                self.model_dropdown.set_model(model_list)
                self.model_dropdown.set_sensitive(False)
        except Exception as e:
            self._show_toast(f"Cannot connect to Ollama: {e}")
            model_list = Gtk.StringList.new(["Ollama not available"])
            self.model_dropdown.set_model(model_list)
            self.model_dropdown.set_sensitive(False)

    def _show_toast(self, message: str) -> None:
        """Show a toast notification."""
        toast = Adw.Toast(title=message)
        self.toast_overlay.add_toast(toast)

    def _get_selected_model(self) -> str:
        """Get the currently selected model."""
        selected = self.model_dropdown.get_selected_item()
        if selected:
            return selected.get_string()
        return self._settings.model

    def _get_selected_format(self) -> str:
        """Get the currently selected output format."""
        selected = self.format_dropdown.get_selected_item()
        if selected:
            return selected.get_string()
        return "markdown"

    def _load_file(self, path: Path) -> None:
        """Load a file for processing."""
        suffix = path.suffix.lower()
        supported = {".pdf", ".png", ".jpg", ".jpeg"}

        if suffix not in supported:
            self._show_toast(f"Unsupported file type: {suffix}")
            return

        self._file_path = path
        self.drop_label.set_text(path.name)

        if suffix == ".pdf":
            try:
                from remarkable_ocr.pdf import get_page_count

                pages = get_page_count(path)
                self.drop_sublabel.set_text(f"{pages} pages")
            except Exception as e:
                self._show_toast(f"Cannot read PDF: {e}")
                return
        else:
            self.drop_sublabel.set_text("1 image")

        self.drop_icon.set_from_icon_name("document-open-symbolic")
        self.process_button.set_sensitive(True)

        # Update output filename
        output_dir = Path(self.output_entry.get_text())
        fmt = self._get_selected_format()
        ext = "md" if fmt == "markdown" else fmt
        output_file = output_dir / f"{path.stem}.{ext}"
        self.output_entry.set_text(str(output_file))

    # Event handlers

    def _on_drop(self, _target: Gtk.DropTarget, value: GObject.Value, _x: float, _y: float) -> bool:
        """Handle file drop."""
        if isinstance(value, Gdk.FileList):
            files = value.get_files()
            if files:
                path = Path(files[0].get_path())
                self._load_file(path)
                return True
        return False

    def _on_drag_enter(self, _target: Gtk.DropTarget, _x: float, _y: float) -> Gdk.DragAction:
        """Handle drag enter."""
        self.drop_zone.get_parent().add_css_class("drop-active")
        return Gdk.DragAction.COPY

    def _on_drag_leave(self, _target: Gtk.DropTarget) -> None:
        """Handle drag leave."""
        self.drop_zone.get_parent().remove_css_class("drop-active")

    def _on_drop_zone_clicked(self, _gesture: Gtk.GestureClick, _n: int, _x: float, _y: float) -> None:
        """Handle click on drop zone to open file dialog."""
        dialog = Gtk.FileDialog()

        # Set up filters
        filters = Gio.ListStore.new(Gtk.FileFilter)

        f = Gtk.FileFilter()
        f.set_name("Supported files (PDF, PNG, JPG)")
        f.add_mime_type("application/pdf")
        f.add_mime_type("image/png")
        f.add_mime_type("image/jpeg")
        filters.append(f)

        dialog.set_filters(filters)
        dialog.set_default_filter(f)
        dialog.open(self, None, self._on_file_dialog_response)

    def _on_file_dialog_response(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        """Handle file dialog response."""
        try:
            file = dialog.open_finish(result)
            if file:
                path = Path(file.get_path())
                self._load_file(path)
        except GLib.Error:
            pass  # User cancelled

    def _on_process_clicked(self, _button: Gtk.Button) -> None:
        """Handle process button click."""
        if self._worker is not None:
            # Cancel current processing
            self._worker.cancel()
            self.process_button.set_label("Process")
            self.process_button.remove_css_class("destructive-action")
            self.process_button.add_css_class("suggested-action")
            self._worker = None
            return

        if not self._file_path:
            return

        # Start processing
        self.process_button.set_label("Cancel")
        self.process_button.remove_css_class("suggested-action")
        self.process_button.add_css_class("destructive-action")

        self.progress_bar.set_fraction(0)
        self.progress_bar.set_text("Starting...")
        self.text_buffer.set_text("")
        self._results = []

        self._worker = OCRWorker(
            file_path=self._file_path,
            model=self._get_selected_model(),
            on_progress=self._on_worker_progress,
            on_complete=self._on_worker_complete,
            on_error=self._on_worker_error,
        )
        self._worker.start()

    def _on_worker_progress(self, result: ProcessingResult) -> None:
        """Handle progress from worker."""
        self._results.append(result)

        # Update progress bar
        fraction = result.page_num / result.total_pages
        self.progress_bar.set_fraction(fraction)
        self.progress_bar.set_text(f"Page {result.page_num}/{result.total_pages}")

        # Append text
        end_iter = self.text_buffer.get_end_iter()
        if result.page_num > 1:
            self.text_buffer.insert(end_iter, "\n\n---\n\n")
            end_iter = self.text_buffer.get_end_iter()
        self.text_buffer.insert(end_iter, f"# Page {result.page_num}\n\n{result.text}")

    def _on_worker_complete(self, results: list[ProcessingResult]) -> None:
        """Handle worker completion."""
        self._worker = None
        self.process_button.set_label("Process")
        self.process_button.remove_css_class("destructive-action")
        self.process_button.add_css_class("suggested-action")

        self.progress_bar.set_fraction(1.0)
        self.progress_bar.set_text("Complete")

        self.copy_button.set_sensitive(True)
        self.save_button.set_sensitive(True)

        self._show_toast("Processing complete")

    def _on_worker_error(self, error: ProcessingError) -> None:
        """Handle worker error."""
        if error.page_num:
            self._show_toast(f"Error on page {error.page_num}: {error.message}")
        else:
            self._show_toast(f"Error: {error.message}")

    def _on_browse_output_clicked(self, _button: Gtk.Button) -> None:
        """Handle browse output button click."""
        dialog = Gtk.FileDialog()
        dialog.set_initial_folder(Gio.File.new_for_path(str(Path.home())))
        dialog.save(self, None, self._on_save_dialog_response)

    def _on_save_dialog_response(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        """Handle save dialog response."""
        try:
            file = dialog.save_finish(result)
            if file:
                self.output_entry.set_text(file.get_path())
        except GLib.Error:
            pass  # User cancelled

    def _on_copy_clicked(self, _button: Gtk.Button) -> None:
        """Handle copy button click."""
        start, end = self.text_buffer.get_bounds()
        text = self.text_buffer.get_text(start, end, False)

        clipboard = Gdk.Display.get_default().get_clipboard()
        clipboard.set(text)
        self._show_toast("Copied to clipboard")

    def _on_save_clicked(self, _button: Gtk.Button) -> None:
        """Handle save button click."""
        if not self._file_path or not self._results:
            return

        output_path = Path(self.output_entry.get_text())
        output_format = self._get_selected_format()

        # Build OCRResults from current text
        start, end = self.text_buffer.get_bounds()
        text = self.text_buffer.get_text(start, end, False)

        # Create single result with all text (user may have edited)
        results = [OCRResult(text=text, confidence=None, page_num=1)]

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            write_output(
                results=results,
                source=self._file_path,
                output_dir=output_path.parent,
                output_format=output_format,
                model=self._get_selected_model(),
            )
            self._show_toast(f"Saved to {output_path}")
        except Exception as e:
            self._show_toast(f"Save failed: {e}")
```

**Step 2: Verify syntax**

Run: `python -m py_compile remarkable_ocr/gui/window.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add remarkable_ocr/gui/window.py
git commit -m "feat(gui): add main window with all widgets"
```

---

## Task 6: Add GUI Command to CLI

**Files:**
- Modify: `remarkable_ocr/cli.py`

**Step 1: Add gui command after the models command (around line 95)**

Add this new command:

```python
@app.command()
def gui() -> None:
    """Launch the graphical user interface."""
    try:
        from remarkable_ocr.gui import RemarkableOCRApp
    except ImportError:
        err_console.print("[red]Error:[/red] GUI dependencies not installed")
        err_console.print("\nInstall with: pip install remarkable-ocr[gui]")
        err_console.print("Also ensure GTK4 and libadwaita are installed:")
        err_console.print("  Fedora: dnf install gtk4-devel libadwaita-devel")
        err_console.print("  Ubuntu: apt install libgtk-4-dev libadwaita-1-dev")
        raise typer.Exit(1)

    app = RemarkableOCRApp()
    app.run(None)
```

**Step 2: Verify CLI still works**

Run: `python -m remarkable_ocr.cli --help`
Expected: Should show `gui` command in the list

**Step 3: Commit**

```bash
git add remarkable_ocr/cli.py
git commit -m "feat(cli): add gui command to launch GTK interface"
```

---

## Task 7: Install and Test

**Step 1: Install system dependencies (Fedora)**

Run: `sudo dnf install -y gtk4-devel libadwaita-devel gobject-introspection-devel`
Expected: Packages installed

**Step 2: Install package with GUI extras**

Run: `pip install -e ".[gui]"`
Expected: Successfully installed with PyGObject

**Step 3: Test GUI launch**

Run: `remarkable-ocr gui`
Expected: GTK window opens with drop zone, model dropdown, etc.

**Step 4: Test drag-drop**

Action: Drag a PDF or image file onto the drop zone
Expected: File name appears, page count shown, Process button enabled

**Step 5: Test file dialog**

Action: Click the drop zone
Expected: File chooser dialog opens with PDF/image filter

**Step 6: Test processing**

Action: Click Process button
Expected: Progress bar updates, text appears in text view

**Step 7: Test save**

Action: Click Save button after processing
Expected: Toast shows "Saved to [path]"

---

## Task 8: Final Commit and Tag

**Step 1: Run existing tests**

Run: `pytest -m "not integration" -v`
Expected: All existing tests pass

**Step 2: Update version (optional)**

If this is a significant feature, bump version in `remarkable_ocr/__init__.py`:

```python
__version__ = "0.4.0"
```

And in `pyproject.toml`:

```toml
version = "0.4.0"
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: add GTK4 GUI with drag-drop, live OCR, and editable preview

- Add Adw.Application with libadwaita styling
- Drag-drop and click-to-browse file input
- Model/format selection dropdowns
- Background OCR processing with progress
- Editable text preview before save
- Toast notifications for errors/success
- New 'gui' CLI command

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

| Task | Description | Est. Complexity |
|------|-------------|-----------------|
| 1 | Add GUI dependencies to pyproject.toml | Simple |
| 2 | Create gui module structure | Simple |
| 3 | Create Application class | Simple |
| 4 | Create Worker thread class | Medium |
| 5 | Create Main Window | Complex |
| 6 | Add gui CLI command | Simple |
| 7 | Install and manual test | Testing |
| 8 | Final commit | Simple |
