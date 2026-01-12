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
