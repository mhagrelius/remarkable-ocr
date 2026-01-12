"""Background processing worker."""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from gi.repository import GLib
from PIL import Image

from remarkable_ocr.chunking import (
    ChunkOCRResult,
    merge_chunk_texts,
    split_image_into_chunks,
)
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
class ProcessingStatus:
    """Status update during processing."""

    page_num: int
    total_pages: int
    message: str
    chunk_num: int | None = None
    total_chunks: int | None = None


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
        on_status: Callable[[ProcessingStatus], None] | None = None,
        on_chunk_text: Callable[[str], None] | None = None,
    ) -> None:
        self.file_path = file_path
        self.model = model
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error
        self.on_status = on_status
        self.on_chunk_text = on_chunk_text

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

                # Check if this page should be chunked (auto-chunk for tall images)
                should_chunk = self._settings.chunk_pages
                if not should_chunk and self._settings.auto_chunk:
                    if image.height > self._settings.max_chunk_height:
                        should_chunk = True

                if should_chunk:
                    text = self._process_chunked(image, page_num, total_pages)
                else:
                    text = self._process_single(image, page_num, total_pages)

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

    def _process_single(self, image: Image.Image, page_num: int, total_pages: int) -> str:
        """Process a single image without chunking."""
        # Emit status before processing
        status = ProcessingStatus(
            page_num=page_num,
            total_pages=total_pages,
            message=f"Processing page {page_num}/{total_pages}...",
        )
        GLib.idle_add(self._emit_status, status)

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
            return clean_text(result.text)
        except Exception as e:
            error = ProcessingError(
                message=str(e),
                page_num=page_num,
            )
            GLib.idle_add(self._emit_error, error)
            return f"[Error on page {page_num}: {e}]"

    def _process_chunked(self, image: Image.Image, page_num: int, total_pages: int) -> str:
        """Process an image in chunks with per-chunk progress updates."""
        # Split image into chunks
        chunks = split_image_into_chunks(
            image,
            chunk_count=self._settings.chunk_count,
            overlap_percent=self._settings.overlap_percent,
            smart_chunking=self._settings.smart_chunking,
            max_chunk_height=self._settings.max_chunk_height,
        )

        chunk_results = []
        for chunk in chunks:
            if self._cancel_requested:
                break

            # Emit status for this chunk (with chunk info for progress bar)
            status = ProcessingStatus(
                page_num=page_num,
                total_pages=total_pages,
                message=f"Page {page_num}/{total_pages}, chunk {chunk.chunk_index + 1}/{chunk.total_chunks}...",
                chunk_num=chunk.chunk_index + 1,
                total_chunks=chunk.total_chunks,
            )
            GLib.idle_add(self._emit_status, status)

            try:
                result = process_image(
                    image=chunk.image,
                    page_num=page_num,
                    model=self.model,
                    host=self._settings.ollama_host,
                    timeout=self._settings.timeout,
                    chunk_info=(chunk.chunk_index, chunk.total_chunks),
                    previous_chunk_text=None,
                    num_ctx=self._settings.num_ctx,
                    temperature=self._settings.temperature,
                )

                chunk_text = clean_text(result.text)
                chunk_results.append(ChunkOCRResult(
                    text=chunk_text,
                    chunk_index=chunk.chunk_index,
                    page_num=page_num,
                ))

                # Emit chunk text immediately so user sees progress
                GLib.idle_add(self._emit_chunk_text, chunk_text)

            except Exception as e:
                error = ProcessingError(
                    message=str(e),
                    page_num=page_num,
                )
                GLib.idle_add(self._emit_error, error)
                error_text = f"[Error on chunk {chunk.chunk_index + 1}: {e}]"
                chunk_results.append(ChunkOCRResult(
                    text=error_text,
                    chunk_index=chunk.chunk_index,
                    page_num=page_num,
                ))
                GLib.idle_add(self._emit_chunk_text, error_text)

        # Merge chunk texts with deduplication
        merged_text = merge_chunk_texts(chunk_results)
        return clean_text(merged_text)

    def _emit_status(self, status: ProcessingStatus) -> bool:
        """Emit status update (called via GLib.idle_add)."""
        if self.on_status:
            self.on_status(status)
        return False  # Don't repeat

    def _emit_chunk_text(self, text: str) -> bool:
        """Emit chunk text update (called via GLib.idle_add)."""
        if self.on_chunk_text:
            self.on_chunk_text(text)
        return False  # Don't repeat

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
