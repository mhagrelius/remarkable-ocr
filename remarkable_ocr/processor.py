"""Text processing and cleaning for OCR output."""

import re

from remarkable_ocr.logging import get_logger

logger = get_logger("processor")

# Common OCR artifacts and their corrections
OCR_CORRECTIONS = [
    (r"\brn(?=[aeiou])", "m"),  # rn at word start before vowel -> m (e.g., rnorning -> morning)
    (r"0(?=[a-zA-Z])", "O"),  # 0 before letter -> O
    (r"(?<=[a-zA-Z])0", "o"),  # 0 after letter -> o
]


def clean_text(raw: str) -> str:
    """Clean OCR output for better readability.

    Args:
        raw: Raw OCR output text

    Returns:
        Cleaned text with normalized whitespace and fixed artifacts
    """
    text = raw

    # Apply OCR corrections
    for pattern, replacement in OCR_CORRECTIONS:
        text = re.sub(pattern, replacement, text)

    # Normalize multiple spaces to single space
    text = re.sub(r" {2,}", " ", text)

    # Normalize multiple newlines to max 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def is_blank_page(text: str) -> bool:
    """Check if page content indicates a blank page.

    Args:
        text: Extracted text from page

    Returns:
        True if page appears to be blank
    """
    cleaned = text.strip().lower()

    if not cleaned:
        return True

    # Check for explicit blank page markers
    blank_markers = ["[blank page]", "[blank]", "[empty]", "[no text]", "[no text detected]"]
    for marker in blank_markers:
        if cleaned == marker:
            return True

    return False
