"""Tests for OCR processing."""

from unittest.mock import Mock, patch

import httpx
import pytest
from PIL import Image


def _mock_model(name: str) -> Mock:
    """Create a mock model object with .model attribute."""
    m = Mock()
    m.model = name
    return m


def _mock_list_response(model_names: list[str]) -> Mock:
    """Create a mock list response with .models attribute."""
    response = Mock()
    response.models = [_mock_model(name) for name in model_names]
    return response


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    return Image.new("RGB", (100, 100), color="white")


def test_ocr_result_dataclass():
    """OCRResult should store text, confidence, and page number."""
    from remarkable_ocr.ocr import OCRResult

    result = OCRResult(text="Hello", confidence=0.95, page_num=1)

    assert result.text == "Hello"
    assert result.confidence == 0.95
    assert result.page_num == 1


def test_ocr_result_none_confidence():
    """OCRResult should accept None confidence."""
    from remarkable_ocr.ocr import OCRResult

    result = OCRResult(text="Hello", confidence=None, page_num=1)
    assert result.confidence is None


def test_build_prompt():
    """_build_prompt should return the OCR prompt."""
    from remarkable_ocr.ocr import _build_prompt

    prompt = _build_prompt()

    assert "handwriting recognition" in prompt.lower()
    assert "transcribe" in prompt.lower()


@patch("remarkable_ocr.ocr.ollama")
def test_check_ollama_health_success(mock_ollama):
    """check_ollama_health should return True when Ollama is available."""
    from remarkable_ocr.ocr import check_ollama_health

    mock_ollama.Client.return_value.list.return_value = _mock_list_response(["qwen2.5-vl:7b"])

    result = check_ollama_health("http://localhost:11434", "qwen2.5-vl:7b")

    assert result is True


@patch("remarkable_ocr.ocr.ollama")
def test_check_ollama_health_model_missing(mock_ollama):
    """check_ollama_health should return False when model not found."""
    from remarkable_ocr.ocr import check_ollama_health

    mock_ollama.Client.return_value.list.return_value = _mock_list_response(["other-model"])

    result = check_ollama_health("http://localhost:11434", "qwen2.5-vl:7b")

    assert result is False


@patch("remarkable_ocr.ocr.ollama")
def test_check_ollama_health_connection_error(mock_ollama):
    """check_ollama_health should return False on connection error."""
    from remarkable_ocr.ocr import check_ollama_health

    mock_ollama.Client.return_value.list.side_effect = httpx.ConnectError("Connection refused")

    result = check_ollama_health("http://localhost:11434", "qwen2.5-vl:7b")

    assert result is False


@patch("remarkable_ocr.ocr.ollama")
def test_process_image_success(mock_ollama, sample_image):
    """process_image should return OCRResult with extracted text."""
    from remarkable_ocr.ocr import process_image

    mock_ollama.Client.return_value.chat.return_value = {
        "message": {"content": "Extracted text from image"}
    }

    result = process_image(
        image=sample_image,
        page_num=1,
        model="qwen2.5-vl:7b",
        host="http://localhost:11434",
        timeout=60,
    )

    assert result.text == "Extracted text from image"
    assert result.page_num == 1


@patch("remarkable_ocr.ocr.ollama")
def test_process_image_blank_page(mock_ollama, sample_image):
    """process_image should handle blank page response."""
    from remarkable_ocr.ocr import process_image

    mock_ollama.Client.return_value.chat.return_value = {
        "message": {"content": "[blank page]"}
    }

    result = process_image(
        image=sample_image,
        page_num=1,
        model="qwen2.5-vl:7b",
        host="http://localhost:11434",
        timeout=60,
    )

    assert result.text == "[blank page]"


@patch("remarkable_ocr.ocr.ollama")
def test_get_available_models_success(mock_ollama):
    """get_available_models should return vision models."""
    from remarkable_ocr.ocr import get_available_models

    mock_ollama.Client.return_value.list.return_value = _mock_list_response([
        "qwen2.5-vl:7b",
        "llava:13b",
        "llama3:8b",  # Not a vision model
    ])

    models = get_available_models("http://localhost:11434")

    assert "qwen2.5-vl:7b" in models
    assert "llava:13b" in models
    assert "llama3:8b" not in models  # Filtered out


@patch("remarkable_ocr.ocr.ollama")
def test_get_available_models_error(mock_ollama):
    """get_available_models should return empty list on error."""
    from remarkable_ocr.ocr import get_available_models

    mock_ollama.Client.return_value.list.side_effect = httpx.ConnectError("Connection refused")

    models = get_available_models("http://localhost:11434")

    assert models == []


@patch("remarkable_ocr.ocr.ollama")
def test_process_image_error(mock_ollama, sample_image):
    """process_image should return [OCR Failed] on error."""
    from remarkable_ocr.ocr import process_image

    mock_ollama.Client.return_value.chat.side_effect = httpx.TimeoutException("Timeout")

    result = process_image(
        image=sample_image,
        page_num=1,
        model="qwen2.5-vl:7b",
        host="http://localhost:11434",
        timeout=60,
    )

    assert result.text == "[OCR Failed]"
    assert result.page_num == 1
