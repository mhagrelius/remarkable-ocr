"""Main application window."""

from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk

from remarkable_ocr.config import Settings
from remarkable_ocr.gui.worker import OCRWorker, ProcessingError, ProcessingResult
from remarkable_ocr.ocr import OCRResult, get_available_models
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
