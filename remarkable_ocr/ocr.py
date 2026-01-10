"""OCR processing using Ollama."""

import base64
import io
from dataclasses import dataclass

import ollama
from PIL import Image

from remarkable_ocr.logging import get_logger

logger = get_logger("ocr")


@dataclass
class OCRResult:
    """Result from OCR processing of a single page."""

    text: str
    confidence: float | None
    page_num: int


def _build_prompt() -> str:
    """Build the OCR prompt for handwriting recognition."""
    return """You are a handwriting recognition system. Extract all handwritten text from this image.

Rules:
- Transcribe exactly what is written, preserving the author's words
- Maintain paragraph breaks and list structure
- Use markdown formatting: **bold** for emphasized/underlined text, - for bullet points
- If text is unclear, make your best attempt (do not skip words)
- For diagrams or sketches, briefly describe them in [brackets]
- If the page is blank or contains no text, respond with: [blank page]

Output only the extracted text, no commentary."""


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def check_ollama_health(host: str, model: str) -> bool:
    """Check if Ollama is reachable and model is available.

    Args:
        host: Ollama API host URL
        model: Model name to check

    Returns:
        True if Ollama is healthy and model available
    """
    try:
        client = ollama.Client(host=host)
        response = client.list()

        # Check if model is available
        available_models = [m.get("name", "") for m in response.get("models", [])]

        # Handle model name variations (with/without tag)
        model_base = model.split(":")[0]
        for available in available_models:
            if available.startswith(model_base):
                logger.debug(f"Found model: {available}")
                return True

        logger.warning(f"Model {model} not found. Available: {available_models}")
        return False

    except Exception as e:
        logger.error(f"Failed to connect to Ollama at {host}: {e}")
        return False


def get_available_models(host: str) -> list[str]:
    """Get list of available vision models from Ollama.

    Args:
        host: Ollama API host URL

    Returns:
        List of model names
    """
    try:
        client = ollama.Client(host=host)
        response = client.list()
        models = [m.get("name", "") for m in response.get("models", [])]
        # Filter to likely vision models
        vision_keywords = ["vl", "vision", "llava", "bakllava"]
        vision_models = [
            m for m in models
            if any(kw in m.lower() for kw in vision_keywords)
        ]
        return vision_models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return []


def process_image(
    image: Image.Image,
    page_num: int,
    model: str,
    host: str,
    timeout: int,
) -> OCRResult:
    """Process a single image through OCR.

    Args:
        image: PIL Image to process
        page_num: Page number (1-indexed)
        model: Ollama model name
        host: Ollama API host URL
        timeout: Timeout in seconds

    Returns:
        OCRResult with extracted text
    """
    logger.debug(f"Processing page {page_num} ({image.width}x{image.height}px)")

    client = ollama.Client(host=host, timeout=timeout)
    prompt = _build_prompt()

    # Convert image to base64
    image_b64 = _image_to_base64(image)

    try:
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            options={
                "temperature": 0.1,
                "num_predict": 4096,
            },
        )

        text = response["message"]["content"]
        logger.debug(f"Page {page_num}: extracted {len(text)} characters")

        return OCRResult(
            text=text,
            confidence=None,  # Qwen2.5-VL doesn't provide confidence
            page_num=page_num,
        )

    except Exception as e:
        logger.error(f"OCR failed for page {page_num}: {e}")
        return OCRResult(
            text="[OCR Failed]",
            confidence=None,
            page_num=page_num,
        )
