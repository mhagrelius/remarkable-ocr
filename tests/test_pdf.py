"""Tests for PDF processing."""

from pathlib import Path

import pytest
from PIL import Image


def test_extract_pages_yields_images(fixtures_dir: Path):
    """extract_pages should yield page images."""
    from remarkable_ocr.pdf import extract_pages

    pdf_path = fixtures_dir / "sample_1page.pdf"
    pages = list(extract_pages(pdf_path))

    assert len(pages) == 1
    page_num, image = pages[0]
    assert page_num == 1
    assert isinstance(image, Image.Image)


def test_extract_pages_multipage(fixtures_dir: Path):
    """extract_pages should handle multi-page PDFs."""
    from remarkable_ocr.pdf import extract_pages

    pdf_path = fixtures_dir / "sample_multipage.pdf"
    pages = list(extract_pages(pdf_path))

    assert len(pages) == 3
    for i, (page_num, image) in enumerate(pages):
        assert page_num == i + 1
        assert isinstance(image, Image.Image)


def test_extract_pages_width(fixtures_dir: Path):
    """extract_pages should scale to specified width."""
    from remarkable_ocr.pdf import extract_pages

    pdf_path = fixtures_dir / "sample_1page.pdf"
    pages = list(extract_pages(pdf_path, width=800))

    _, image = pages[0]
    assert image.width == 800


def test_extract_pages_file_not_found():
    """extract_pages should raise FileNotFoundError for missing files."""
    from remarkable_ocr.pdf import extract_pages

    with pytest.raises(FileNotFoundError):
        list(extract_pages(Path("/nonexistent/file.pdf")))


def test_get_page_count(fixtures_dir: Path):
    """get_page_count should return correct count."""
    from remarkable_ocr.pdf import get_page_count

    assert get_page_count(fixtures_dir / "sample_1page.pdf") == 1
    assert get_page_count(fixtures_dir / "sample_multipage.pdf") == 3
