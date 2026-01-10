"""Tests for logging configuration."""

import logging
from pathlib import Path


def test_setup_logging_creates_logger():
    """setup_logging should return a configured logger."""
    from remarkable_ocr.logging import setup_logging

    logger = setup_logging(level="INFO")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "remarkable_ocr"
    assert logger.level == logging.INFO


def test_setup_logging_debug_level():
    """setup_logging should handle DEBUG level."""
    from remarkable_ocr.logging import setup_logging

    logger = setup_logging(level="DEBUG")

    assert logger.level == logging.DEBUG


def test_get_logger_returns_child():
    """get_logger should return a child logger."""
    from remarkable_ocr.logging import get_logger, setup_logging

    setup_logging(level="INFO")
    logger = get_logger("pdf")

    assert logger.name == "remarkable_ocr.pdf"
