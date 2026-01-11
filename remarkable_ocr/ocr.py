"""OCR processing using Ollama."""

import base64
import io
from dataclasses import dataclass

import httpx
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
    return """You are an expert handwriting recognition and document digitization system. Your goal is to accurately transcribe all text from the provided image into a structured markdown format.

### Core Role & Objective
Extract all textual content—both handwritten and printed—preserving the original layout, structure, and intent of the author.

### Transcription Rules
1. **Content Scope**
   - Transcribe **everything** visible on the page, including printed text (e.g., headers, form labels) and handwriting.
   - **Do not autocorrect** spelling or grammar errors. Transcribe exactly what is written, strictly preserving the author's diction.

2. **Handling Edits & Corrections**
   - Interpret the final intended text.
   - If text is crossed out, omit it.
   - If text is inserted (e.g., via caret `^` or arrow), place it in the position indicated by the author.

3. **Formatting & Structure**
   - Maintain paragraph breaks as they appear visually.
   - Use Markdown for emphasis: **bold** for text that is underlined, circled, or otherwise visually emphasized.
   - Reproduce lists and indentation hierarchies visually:
     - Level 1: `- Item`
     - Level 2: `  - Sub-item` (2 spaces indentation)
     - Level 3: `    - Sub-sub-item` (4 spaces indentation)
   - Only use numbered lists (`1.`, `2.`, etc.) if the original document has explicit numbering. Do not invent numbers for bullet points.

4. **Visual Elements & Uncertainty**
   - **Diagrams:** Briefly describe any non-textual visuals in brackets, e.g., `[Sketch of a house plan]`.
   - **Unclear Text:** If a word is difficult to read, provide your best guess with uncertainty marker: `[unclear: best guess]`.
   - **Empty Content:** If the page is blank or contains no legible content, output only: `[blank page]`.

### Output Format
- Provide **only** the extracted text in markdown.
- Do not include conversational filler (e.g., "Here is the text...")."""


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
        available_models = [m.model for m in response.models]

        # Handle model name variations (with/without tag)
        model_base = model.split(":")[0]
        for available in available_models:
            if available.startswith(model_base):
                logger.debug(f"Found model: {available}")
                return True

        logger.warning(f"Model {model} not found. Available: {available_models}")
        return False

    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to Ollama at {host}: {e}")
        return False
    except httpx.TimeoutException as e:
        logger.error(f"Timeout connecting to Ollama at {host}: {e}")
        return False
    except ollama.ResponseError as e:
        logger.error(f"Ollama API error: {e}")
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
        models = [m.model for m in response.models]
        # Filter to likely vision models
        vision_keywords = ["vl", "vision", "llava", "bakllava"]
        vision_models = [
            m for m in models
            if any(kw in m.lower() for kw in vision_keywords)
        ]
        return vision_models
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to Ollama at {host}: {e}")
        return []
    except httpx.TimeoutException as e:
        logger.error(f"Timeout connecting to Ollama at {host}: {e}")
        return []
    except ollama.ResponseError as e:
        logger.error(f"Ollama API error: {e}")
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

    except httpx.ConnectError as e:
        logger.error(f"OCR failed for page {page_num}: Cannot connect to Ollama: {e}")
        return OCRResult(text="[OCR Failed]", confidence=None, page_num=page_num)
    except httpx.TimeoutException as e:
        logger.error(f"OCR failed for page {page_num}: Timeout: {e}")
        return OCRResult(text="[OCR Failed]", confidence=None, page_num=page_num)
    except ollama.ResponseError as e:
        logger.error(f"OCR failed for page {page_num}: API error: {e}")
        return OCRResult(text="[OCR Failed]", confidence=None, page_num=page_num)
