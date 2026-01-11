"""Tests for configuration management."""

import os
from pathlib import Path

import pytest


def test_settings_defaults():
    """Settings should have sensible defaults."""
    # Clear any env vars that might interfere
    env_vars = [k for k in os.environ if k.startswith("REMARKABLE_OCR_")]
    for var in env_vars:
        os.environ.pop(var, None)

    from remarkable_ocr.config import Settings

    settings = Settings()

    assert settings.model == "ministral-3:14b-instruct-2512-q8_0"
    assert settings.ollama_host == "http://localhost:11434"
    assert settings.output_dir == Path("./output")
    assert settings.timeout == 60
    assert settings.log_level == "INFO"
    assert settings.output_format == "markdown"
    assert settings.num_ctx == 8192
    assert settings.temperature == 0.1


def test_settings_from_env(monkeypatch):
    """Settings should load from environment variables."""
    monkeypatch.setenv("REMARKABLE_OCR_MODEL", "qwen2.5-vl:3b")
    monkeypatch.setenv("REMARKABLE_OCR_TIMEOUT", "120")
    monkeypatch.setenv("REMARKABLE_OCR_OUTPUT_DIR", "/tmp/ocr-output")

    # Force reimport to pick up new env vars
    import importlib
    import remarkable_ocr.config
    importlib.reload(remarkable_ocr.config)
    from remarkable_ocr.config import Settings

    settings = Settings()

    assert settings.model == "qwen2.5-vl:3b"
    assert settings.timeout == 120
    assert settings.output_dir == Path("/tmp/ocr-output")


def test_settings_validation():
    """Settings should validate values."""
    from remarkable_ocr.config import Settings

    with pytest.raises(ValueError):
        Settings(timeout=-1)  # Must be positive


def test_settings_validation_num_ctx():
    """Context window must be at least 2048."""
    from remarkable_ocr.config import Settings

    with pytest.raises(ValueError):
        Settings(num_ctx=1024)  # Must be >= 2048


def test_settings_validation_temperature():
    """Temperature must be between 0 and 1."""
    from remarkable_ocr.config import Settings

    with pytest.raises(ValueError):
        Settings(temperature=1.5)  # Must be <= 1.0
    with pytest.raises(ValueError):
        Settings(temperature=-0.1)  # Must be >= 0.0
