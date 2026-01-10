"""Create test PDF fixtures."""

import fitz  # pymupdf
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent


def create_sample_1page():
    """Create a simple 1-page PDF with text."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Letter size
    text = "Hello, this is a test page.\n\nSecond paragraph here."
    page.insert_text((72, 72), text, fontsize=12)
    output_path = FIXTURES_DIR / "sample_1page.pdf"
    doc.save(output_path)
    doc.close()
    print(f"Created: {output_path}")


def create_sample_blank():
    """Create a blank PDF page."""
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    output_path = FIXTURES_DIR / "sample_blank.pdf"
    doc.save(output_path)
    doc.close()
    print(f"Created: {output_path}")


def create_sample_multipage():
    """Create a 3-page PDF."""
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"This is page {i + 1}", fontsize=12)
    output_path = FIXTURES_DIR / "sample_multipage.pdf"
    doc.save(output_path)
    doc.close()
    print(f"Created: {output_path}")


if __name__ == "__main__":
    create_sample_1page()
    create_sample_blank()
    create_sample_multipage()
