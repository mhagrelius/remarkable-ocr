"""Integration tests requiring Ollama."""

from pathlib import Path

import pytest


@pytest.mark.integration
def test_full_pipeline(fixtures_dir: Path, tmp_path: Path):
    """Test full PDF to text pipeline with real Ollama."""
    from remarkable_ocr.ocr import check_ollama_health, process_image
    from remarkable_ocr.pdf import extract_pages
    from remarkable_ocr.processor import clean_text
    from remarkable_ocr.writer import write_output

    host = "http://localhost:11434"
    model = "qwen2.5-vl:7b"

    # Skip if Ollama not available
    if not check_ollama_health(host, model):
        pytest.skip("Ollama not available")

    pdf_path = fixtures_dir / "sample_1page.pdf"
    results = []

    for page_num, image in extract_pages(pdf_path):
        result = process_image(
            image=image,
            page_num=page_num,
            model=model,
            host=host,
            timeout=60,
        )
        result.text = clean_text(result.text)
        results.append(result)

    output_path = write_output(
        results=results,
        source=pdf_path,
        output_dir=tmp_path,
        output_format="markdown",
        model=model,
    )

    assert output_path.exists()
    content = output_path.read_text()
    assert "Page 1" in content
