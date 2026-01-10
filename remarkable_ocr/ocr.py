"""OCR processing using Ollama."""

from dataclasses import dataclass


@dataclass
class OCRResult:
    """Result from OCR processing of a single page."""

    text: str
    confidence: float | None
    page_num: int
