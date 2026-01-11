"""Configuration management using Pydantic settings."""

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model: str = "ministral-3:14b-instruct-2512-q8_0"
    ollama_host: str = "http://localhost:11434"
    output_dir: Path = Path("./output")
    timeout: int = 60
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    output_format: Literal["markdown", "json", "txt"] = "markdown"
    num_ctx: int = 8192  # Ollama context window size (4096, 8192, 16384, etc.)
    temperature: float = 0.1  # LLM temperature (0.0-1.0, higher = more creative)

    # Chunking settings
    chunk_pages: bool = False
    chunk_count: int = 3
    overlap_percent: float = 20.0
    max_chunk_height: int = 2000  # Pixels - chunks taller than this may have OCR issues
    smart_chunking: bool = True  # Use whitespace detection to find natural chunk boundaries
    auto_chunk: bool = True  # Automatically chunk tall images (height > max_chunk_height)

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

    @field_validator("chunk_count")
    @classmethod
    def validate_chunk_count(cls, v: int) -> int:
        """Chunk count must be between 2 and 4."""
        if v < 2 or v > 4:
            raise ValueError("chunk_count must be between 2 and 4")
        return v

    @field_validator("overlap_percent")
    @classmethod
    def validate_overlap_percent(cls, v: float) -> float:
        """Overlap percent must be between 0 and 50."""
        if v < 0 or v > 50:
            raise ValueError("overlap_percent must be between 0 and 50")
        return v

    @field_validator("num_ctx")
    @classmethod
    def validate_num_ctx(cls, v: int) -> int:
        """Context window must be at least 2048."""
        if v < 2048:
            raise ValueError("num_ctx must be at least 2048")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Temperature must be between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("temperature must be between 0 and 1")
        return v
