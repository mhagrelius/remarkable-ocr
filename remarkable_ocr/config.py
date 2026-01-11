"""Configuration management using Pydantic settings."""

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model: str = "qwen2.5-vl:7b"
    ollama_host: str = "http://localhost:11434"
    output_dir: Path = Path("./output")
    timeout: int = 60
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    output_format: Literal["markdown", "json", "txt"] = "markdown"

    model_config = SettingsConfigDict(
        env_prefix="REMARKABLE_OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Timeout must be positive."""
        if v <= 0:
            raise ValueError("timeout must be positive")
        return v
