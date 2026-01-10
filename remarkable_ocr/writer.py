"""Output writing in various formats."""

import json
from datetime import datetime, timezone
from pathlib import Path

from remarkable_ocr.logging import get_logger
from remarkable_ocr.ocr import OCRResult

logger = get_logger("writer")


def _calculate_average_confidence(results: list[OCRResult]) -> float | None:
    """Calculate average confidence across all results.

    Args:
        results: List of OCR results

    Returns:
        Average confidence or None if no confidence scores available
    """
    confidences = [r.confidence for r in results if r.confidence is not None]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def write_output(
    results: list[OCRResult],
    source: Path,
    output_dir: Path,
    output_format: str,
    model: str,
) -> Path:
    """Write OCR results to file.

    Args:
        results: List of OCR results
        source: Source PDF path
        output_dir: Output directory
        output_format: Format (markdown, json, txt)
        model: Model name used for OCR

    Returns:
        Path to written file
    """
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output path
    suffix_map = {"markdown": ".md", "json": ".json", "txt": ".txt"}
    suffix = suffix_map.get(output_format, ".md")
    output_path = output_dir / f"{source.stem}{suffix}"

    # Generate content
    if output_format == "json":
        content = _format_json(results, source, model)
    elif output_format == "txt":
        content = _format_txt(results, source)
    else:
        content = _format_markdown(results, source, model)

    # Write file
    output_path.write_text(content)
    logger.info(f"Output written to: {output_path}")

    return output_path


def _format_markdown(results: list[OCRResult], source: Path, model: str) -> str:
    """Format results as markdown with YAML frontmatter."""
    now = datetime.now(timezone.utc).isoformat()
    avg_conf = _calculate_average_confidence(results)

    lines = [
        "---",
        f"source: {source.name}",
        f"date_processed: {now}",
        f"pages: {len(results)}",
        f"confidence_avg: {avg_conf:.2f}" if avg_conf else "confidence_avg: null",
        f"model: {model}",
        "---",
        "",
    ]

    for result in results:
        lines.extend([
            f"# Page {result.page_num}",
            "",
            result.text,
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


def _format_json(results: list[OCRResult], source: Path, model: str) -> str:
    """Format results as JSON."""
    now = datetime.now(timezone.utc).isoformat()
    avg_conf = _calculate_average_confidence(results)

    data = {
        "metadata": {
            "source": source.name,
            "date_processed": now,
            "pages": len(results),
            "confidence_avg": round(avg_conf, 2) if avg_conf else None,
            "model": model,
        },
        "pages": [
            {
                "page_number": r.page_num,
                "text": r.text,
                "confidence": r.confidence,
            }
            for r in results
        ],
    }

    return json.dumps(data, indent=2)


def _format_txt(results: list[OCRResult], source: Path) -> str:
    """Format results as plain text."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = [
        f"[{source.name} - Processed {now}]",
        "",
    ]

    for result in results:
        lines.extend([
            f"=== Page {result.page_num} ===",
            "",
            result.text,
            "",
        ])

    return "\n".join(lines)
