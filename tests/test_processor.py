"""Tests for text processing."""

import pytest


def test_clean_text_strips_whitespace():
    """clean_text should strip leading/trailing whitespace."""
    from remarkable_ocr.processor import clean_text

    assert clean_text("  hello  ") == "hello"
    assert clean_text("\n\nhello\n\n") == "hello"


def test_clean_text_normalizes_newlines():
    """clean_text should normalize multiple newlines to max 2."""
    from remarkable_ocr.processor import clean_text

    text = "paragraph one\n\n\n\n\nparagraph two"
    result = clean_text(text)

    assert result == "paragraph one\n\nparagraph two"


def test_clean_text_preserves_structure():
    """clean_text should preserve paragraph breaks and lists."""
    from remarkable_ocr.processor import clean_text

    text = "# Header\n\n- Item 1\n- Item 2\n\nParagraph"
    result = clean_text(text)

    assert "# Header" in result
    assert "- Item 1" in result
    assert "- Item 2" in result


def test_clean_text_fixes_common_artifacts():
    """clean_text should fix common OCR artifacts."""
    from remarkable_ocr.processor import clean_text

    # Common OCR mistakes
    assert "morning" in clean_text("rnorning")  # rn -> m
    assert clean_text("hello   world") == "hello world"  # multiple spaces


def test_is_blank_page_true():
    """is_blank_page should return True for blank content."""
    from remarkable_ocr.processor import is_blank_page

    assert is_blank_page("") is True
    assert is_blank_page("   ") is True
    assert is_blank_page("\n\n") is True
    assert is_blank_page("[blank page]") is True
    assert is_blank_page("[Blank Page]") is True


def test_is_blank_page_false():
    """is_blank_page should return False for content."""
    from remarkable_ocr.processor import is_blank_page

    assert is_blank_page("Hello world") is False
    assert is_blank_page("- Item") is False
