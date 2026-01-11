"""PDF processing using pymupdf."""

from collections.abc import Iterator
from pathlib import Path

import fitz  # pymupdf
from PIL import Image

from remarkable_ocr.logging import get_logger

logger = get_logger("pdf")


def extract_pages(
    pdf_path: Path, width: int = 1288
) -> Iterator[tuple[int, Image.Image]]:
    """Extract pages from PDF as PIL Images.

    Args:
        pdf_path: Path to PDF file
        width: Target width for rendered images (default 1288 for Qwen2.5-VL)

    Yields:
        Tuples of (page_number, PIL.Image) where page_number is 1-indexed

    Raises:
        FileNotFoundError: If PDF file doesn't exist
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.debug(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)

    try:
        for page_num, page in enumerate(doc, start=1):  # type: ignore[arg-type]
            # Calculate zoom factor to achieve target width
            zoom = width / page.rect.width
            matrix = fitz.Matrix(zoom, zoom)

            # Render page to pixmap
            pixmap = page.get_pixmap(matrix=matrix)

            # Convert to PIL Image
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

            logger.debug(f"Page {page_num}: {image.width}x{image.height}px")
            yield page_num, image

    finally:
        doc.close()


def get_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Number of pages

    Raises:
        FileNotFoundError: If PDF file doesn't exist
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        return len(doc)
