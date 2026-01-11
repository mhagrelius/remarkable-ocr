"""Tests for OCR processing."""

from unittest.mock import Mock, patch, call

import httpx
import ollama
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
    # Must preserve the real ResponseError class for exception handling
    mock_ollama.ResponseError = ollama.ResponseError

    result = process_image(
        image=sample_image,
        page_num=1,
        model="qwen2.5-vl:7b",
        host="http://localhost:11434",
        timeout=60,
    )

    assert result.text == "[OCR Failed]"
    assert result.page_num == 1


# Tests for model loading retry logic


class TestIsModelLoadingError:
    """Tests for _is_model_loading_error function."""

    def test_load_request_error(self):
        """Should detect 'do load request' errors."""
        from remarkable_ocr.ocr import _is_model_loading_error

        error = ollama.ResponseError("do load request: Post http://127.0.0.1:38035/load: EOF")
        assert _is_model_loading_error(error) is True

    def test_eof_error(self):
        """Should detect EOF errors."""
        from remarkable_ocr.ocr import _is_model_loading_error

        error = ollama.ResponseError("EOF")
        assert _is_model_loading_error(error) is True

    def test_connection_reset_error(self):
        """Should detect connection reset errors."""
        from remarkable_ocr.ocr import _is_model_loading_error

        error = httpx.ReadError("Connection reset by peer")
        assert _is_model_loading_error(error) is True

    def test_broken_pipe_error(self):
        """Should detect broken pipe errors."""
        from remarkable_ocr.ocr import _is_model_loading_error

        error = httpx.ReadError("Broken pipe")
        assert _is_model_loading_error(error) is True

    def test_unrelated_error(self):
        """Should not detect unrelated errors as model loading."""
        from remarkable_ocr.ocr import _is_model_loading_error

        error = ollama.ResponseError("Invalid model format")
        assert _is_model_loading_error(error) is False

    def test_timeout_error(self):
        """Should not detect timeout as model loading error."""
        from remarkable_ocr.ocr import _is_model_loading_error

        error = Exception("Request timed out")
        assert _is_model_loading_error(error) is False


class TestRetryOnModelLoading:
    """Tests for _retry_on_model_loading function."""

    def test_success_no_retry(self):
        """Should return result immediately on success."""
        from remarkable_ocr.ocr import _retry_on_model_loading

        mock_func = Mock(return_value="success")

        result = _retry_on_model_loading(mock_func, max_retries=3, initial_delay=0.01)

        assert result == "success"
        assert mock_func.call_count == 1

    @patch("remarkable_ocr.ocr.time.sleep")
    def test_retry_on_loading_error(self, mock_sleep):
        """Should retry on model loading errors."""
        from remarkable_ocr.ocr import _retry_on_model_loading

        mock_func = Mock(
            side_effect=[
                ollama.ResponseError("do load request: EOF"),
                ollama.ResponseError("do load request: EOF"),
                "success",
            ]
        )

        result = _retry_on_model_loading(mock_func, max_retries=3, initial_delay=0.01)

        assert result == "success"
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("remarkable_ocr.ocr.time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        """Should use exponential backoff for delays."""
        from remarkable_ocr.ocr import _retry_on_model_loading

        mock_func = Mock(
            side_effect=[
                ollama.ResponseError("do load request: EOF"),
                ollama.ResponseError("do load request: EOF"),
                "success",
            ]
        )

        result = _retry_on_model_loading(
            mock_func, max_retries=3, initial_delay=1.0, backoff_multiplier=2.0
        )

        assert result == "success"
        # First retry: 1.0s, second retry: 2.0s
        assert mock_sleep.call_args_list == [call(1.0), call(2.0)]

    @patch("remarkable_ocr.ocr.time.sleep")
    def test_max_retries_exhausted(self, mock_sleep):
        """Should raise after max retries exhausted."""
        from remarkable_ocr.ocr import _retry_on_model_loading

        mock_func = Mock(side_effect=ollama.ResponseError("do load request: EOF"))

        with pytest.raises(ollama.ResponseError, match="load"):
            _retry_on_model_loading(mock_func, max_retries=2, initial_delay=0.01)

        # Initial attempt + 2 retries = 3 calls
        assert mock_func.call_count == 3

    def test_no_retry_on_unrelated_error(self):
        """Should not retry on non-loading errors."""
        from remarkable_ocr.ocr import _retry_on_model_loading

        mock_func = Mock(side_effect=ollama.ResponseError("Invalid model"))

        with pytest.raises(ollama.ResponseError, match="Invalid"):
            _retry_on_model_loading(mock_func, max_retries=3, initial_delay=0.01)

        # Should fail immediately without retry
        assert mock_func.call_count == 1

    @patch("remarkable_ocr.ocr.time.sleep")
    def test_retry_on_httpx_read_error(self, mock_sleep):
        """Should retry on httpx.ReadError with loading indicators."""
        from remarkable_ocr.ocr import _retry_on_model_loading

        mock_func = Mock(
            side_effect=[
                httpx.ReadError("Connection reset by peer"),
                "success",
            ]
        )

        result = _retry_on_model_loading(mock_func, max_retries=3, initial_delay=0.01)

        assert result == "success"
        assert mock_func.call_count == 2


class TestProcessImageRetry:
    """Tests for process_image retry behavior."""

    @patch("remarkable_ocr.ocr.time.sleep")
    @patch("remarkable_ocr.ocr.ollama")
    def test_process_image_retries_on_loading_error(self, mock_ollama, mock_sleep, sample_image):
        """process_image should retry on model loading errors."""
        from remarkable_ocr.ocr import process_image

        # First two calls fail with loading error, third succeeds
        mock_ollama.Client.return_value.chat.side_effect = [
            ollama.ResponseError("do load request: EOF"),
            ollama.ResponseError("do load request: EOF"),
            {"message": {"content": "Extracted text"}},
        ]
        mock_ollama.ResponseError = ollama.ResponseError

        result = process_image(
            image=sample_image,
            page_num=1,
            model="qwen2.5-vl:7b",
            host="http://localhost:11434",
            timeout=60,
        )

        assert result.text == "Extracted text"
        assert mock_ollama.Client.return_value.chat.call_count == 3

    @patch("remarkable_ocr.ocr.time.sleep")
    @patch("remarkable_ocr.ocr.ollama")
    def test_process_image_fails_after_max_retries(self, mock_ollama, mock_sleep, sample_image):
        """process_image should fail gracefully after max retries."""
        from remarkable_ocr.ocr import process_image

        # All calls fail with loading error
        mock_ollama.Client.return_value.chat.side_effect = ollama.ResponseError(
            "do load request: EOF"
        )
        mock_ollama.ResponseError = ollama.ResponseError

        result = process_image(
            image=sample_image,
            page_num=1,
            model="qwen2.5-vl:7b",
            host="http://localhost:11434",
            timeout=60,
        )

        assert result.text == "[OCR Failed]"
        # Initial attempt + 5 retries = 6 calls
        assert mock_ollama.Client.return_value.chat.call_count == 6
