"""OCR processing using Ollama."""

import base64
import io
import time
from dataclasses import dataclass

import httpx
import ollama
from PIL import Image

from remarkable_ocr.logging import get_logger
from remarkable_ocr.prompts import load_prompt

logger = get_logger("ocr")

# Load base prompt once at module import
_BASE_PROMPT = load_prompt("ocr_base")

# Default retry settings for model loading
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_DELAY = 2.0
DEFAULT_BACKOFF_MULTIPLIER = 1.5


def _is_model_loading_error(error: Exception) -> bool:
    """Check if an error is related to model loading/unloading.

    These errors occur when Ollama is switching models and the previous
    model is being unloaded or the new model is being loaded.
    """
    error_str = str(error).lower()
    loading_indicators = [
        "load",  # "do load request" errors
        "eof",   # Connection closed during model switch
        "connection reset",
        "broken pipe",
    ]
    return any(indicator in error_str for indicator in loading_indicators)


def _retry_on_model_loading(
    func,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
):
    """Execute a function with retry logic for model loading errors.

    Args:
        func: Callable to execute
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries are exhausted
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except (ollama.ResponseError, httpx.ReadError, httpx.RemoteProtocolError) as e:
            last_exception = e
            if _is_model_loading_error(e) and attempt < max_retries:
                logger.info(
                    f"Model loading in progress, retry {attempt + 1}/{max_retries} "
                    f"in {delay:.1f}s... ({type(e).__name__}: {e})"
                )
                time.sleep(delay)
                delay *= backoff_multiplier
            else:
                raise

    # Should not reach here, but just in case
    raise last_exception  # type: ignore


@dataclass
class OCRResult:
    """Result from OCR processing of a single page."""

    text: str
    confidence: float | None
    page_num: int


def _build_prompt(
    chunk_info: tuple[int, int] | None = None,
    previous_chunk_text: str | None = None,
) -> str:
    """Build the OCR prompt for handwriting recognition.

    Args:
        chunk_info: Optional tuple of (chunk_index, total_chunks) for chunked processing.
        previous_chunk_text: Text from the previous chunk (last ~10 lines) for context.

    Returns:
        Complete prompt string with base instructions and optional chunk context.
    """
    # Add chunk context if processing in chunks
    if chunk_info:
        chunk_index, total_chunks = chunk_info

        # Base context about position
        if chunk_index == 0:
            position_context = "This is the TOP portion of a page. Text may continue below."
            edge_warning = """
**Edge Handling**: Content at the BOTTOM edge of this image may be cut off mid-word or mid-line.
- Transcribe only what is clearly visible
- Do NOT guess or complete partial words at the bottom edge
- The next section will capture any cut-off content"""
        elif chunk_index == total_chunks - 1:
            position_context = "This is the BOTTOM portion of a page. Text started above."
            edge_warning = """
**Edge Handling**: Content at the TOP edge of this image overlaps with the previous section.
- The first few lines may repeat content already captured
- Focus on continuing from where the previous section ended
- Do NOT transcribe content that appears cut off at the top"""
        else:
            position_context = f"This is the MIDDLE portion ({chunk_index + 1}/{total_chunks}) of a page."
            edge_warning = """
**Edge Handling**: Both the TOP and BOTTOM edges of this image may have partial content.
- Top edge overlaps with previous section
- Bottom edge overlaps with next section
- Focus on the clearly visible central content"""

        context = f"\n\n### Context\n{position_context}\n{edge_warning}"

        # Add previous chunk text for continuity (for non-first chunks)
        if previous_chunk_text and chunk_index > 0:
            # Get last ~10 lines for context
            prev_lines = previous_chunk_text.strip().split('\n')
            context_lines = prev_lines[-10:] if len(prev_lines) > 10 else prev_lines
            prev_text_snippet = '\n'.join(context_lines)

            context += f"""

**Previous Section Context**: The section above this one ended with the following text.
Use this ONLY as a reference to understand where you are in the document.
Note: This text may contain OCR artifacts or errors—do not propagate them.
```
{prev_text_snippet}
```
Continue transcribing from where this content ends, avoiding duplication."""

        return _BASE_PROMPT + context

    return _BASE_PROMPT


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def cleanup_merged_chunks(
    text: str,
    model: str,
    host: str,
    timeout: int,
) -> str:
    """Use LLM to clean up merged chunk text, removing artifacts and duplicates.

    Args:
        text: Merged text from multiple chunks that may have artifacts
        model: Ollama model name
        host: Ollama API host URL
        timeout: Timeout in seconds

    Returns:
        Cleaned text with duplicates and artifacts removed
    """
    logger.debug(f"Cleaning up merged chunks ({len(text)} characters)")

    prompt = """You are a text cleanup assistant. The following text was extracted from a document using OCR in multiple overlapping sections. It may contain:

1. **Duplicated sections** - The same content appearing twice due to overlapping OCR regions
2. **Garbled text** - Random characters or nonsense text at section boundaries (e.g., "She quar to t", "fine ext tear r")
3. **Formatting artifacts** - Errant markdown code blocks, extra dashes, or broken formatting

Your task:
- Remove any duplicated content (keep the cleaner version)
- Remove obviously garbled/nonsense text that doesn't fit the context
- Fix any broken formatting
- Preserve all legitimate content, even if it seems incomplete (e.g., truncated bullet points are OK)
- Maintain the original structure (headings, lists, indentation)
- Do NOT add any new content or "complete" partial sentences

Output ONLY the cleaned text, no explanations or commentary.

---
TEXT TO CLEAN:
---
""" + text

    try:
        client = ollama.Client(host=host, timeout=timeout)

        def _do_cleanup():
            """Inner function for cleanup call, wrapped with retry logic."""
            return client.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                options={
                    "temperature": 0.1,
                    "num_predict": 8192,
                },
            )

        # Use retry logic to handle model loading delays
        response = _retry_on_model_loading(_do_cleanup)

        cleaned = response["message"]["content"]
        logger.debug(f"Cleanup complete: {len(text)} -> {len(cleaned)} characters")
        return cleaned

    except (
        httpx.ConnectError,
        httpx.TimeoutException,
        httpx.ReadError,
        httpx.RemoteProtocolError,
        ollama.ResponseError,
    ) as e:
        logger.warning(f"Cleanup failed, returning original text: {e}")
        return text


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
        available_models = [m.model for m in response.models if m.model]

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
        models = [m.model for m in response.models if m.model]
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
    chunk_info: tuple[int, int] | None = None,
    previous_chunk_text: str | None = None,
    num_ctx: int = 8192,
    temperature: float = 0.1,
) -> OCRResult:
    """Process a single image through OCR.

    Args:
        image: PIL Image to process
        page_num: Page number (1-indexed)
        model: Ollama model name
        host: Ollama API host URL
        timeout: Timeout in seconds
        chunk_info: Optional tuple of (chunk_index, total_chunks) for chunked processing
        previous_chunk_text: Text from the previous chunk for context continuity
        num_ctx: Ollama context window size
        temperature: LLM temperature (0.0-1.0)

    Returns:
        OCRResult with extracted text
    """
    if chunk_info:
        logger.debug(
            f"Processing page {page_num} chunk {chunk_info[0] + 1}/{chunk_info[1]} "
            f"({image.width}x{image.height}px)"
        )
    else:
        logger.debug(f"Processing page {page_num} ({image.width}x{image.height}px)")

    client = ollama.Client(host=host, timeout=timeout)
    prompt = _build_prompt(chunk_info, previous_chunk_text)

    # Convert image to base64
    image_b64 = _image_to_base64(image)

    def _do_ocr():
        """Inner function for OCR call, wrapped with retry logic."""
        return client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            options={
                "temperature": temperature,
                "num_predict": 4096,
                "num_ctx": num_ctx,
            },
        )

    try:
        # Use retry logic to handle model loading delays
        response = _retry_on_model_loading(_do_ocr)

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
    except (httpx.ReadError, httpx.RemoteProtocolError) as e:
        logger.error(f"OCR failed for page {page_num}: Connection error: {e}")
        return OCRResult(text="[OCR Failed]", confidence=None, page_num=page_num)
