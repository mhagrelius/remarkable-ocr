"""Logging configuration with rich console output."""

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# Module-level logger instance
_logger: logging.Logger | None = None

# Log file location
LOG_DIR = Path.home() / ".remarkable_ocr" / "logs"
LOG_FILE = LOG_DIR / "remarkable_ocr.log"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging with rich console and file output.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    global _logger

    # Create log directory if needed
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("remarkable_ocr")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with rich formatting
    console_handler = RichHandler(
        console=Console(stderr=True),
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )
    console_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)

    # File handler for persistent logs
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: Module name (e.g., 'pdf', 'ocr')

    Returns:
        Child logger instance
    """
    return logging.getLogger(f"remarkable_ocr.{name}")
