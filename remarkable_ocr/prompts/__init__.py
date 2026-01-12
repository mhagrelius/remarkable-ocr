"""Prompt templates for OCR processing."""

from importlib import resources


def load_prompt(name: str) -> str:
    """Load a prompt template by name.

    Args:
        name: Prompt name without extension (e.g., 'ocr_base')

    Returns:
        Prompt content as string
    """
    return resources.files(__package__).joinpath(f"{name}.txt").read_text()
