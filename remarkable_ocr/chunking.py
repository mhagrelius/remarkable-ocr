"""Image chunking for improved OCR accuracy on long pages."""

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np
from PIL import Image

from remarkable_ocr.logging import get_logger

logger = get_logger("chunking")


def is_likely_garbled(text: str, min_length: int = 5) -> bool:
    """Detect text that's likely a boundary artifact or garbled OCR.

    Garbled text often has:
    - High ratio of non-alphabetic characters
    - Unusual character sequences (multiple consecutive symbols)
    - Very short "words" that aren't common English
    - Random character patterns

    Args:
        text: Text to check
        min_length: Minimum length to check (very short text is not evaluated)

    Returns:
        True if text appears to be garbled/artifact
    """
    text = text.strip()

    # Very short text is not evaluated
    if len(text) < min_length:
        return False

    # Check alphabetic character ratio
    alpha_count = sum(c.isalpha() for c in text)
    total_chars = len(text.replace(" ", ""))
    if total_chars > 0:
        alpha_ratio = alpha_count / total_chars
        if alpha_ratio < 0.5:
            logger.debug(f"Garbled (low alpha ratio {alpha_ratio:.2f}): {text[:50]}")
            return True

    # Check for unusual character sequences (3+ consecutive non-alpha, non-space)
    if re.search(r"[^a-zA-Z\s]{3,}", text):
        # But allow common patterns like "->", "...", "--"
        if not re.search(r"^[\-\.\>\<\=]+$", re.findall(r"[^a-zA-Z\s]{3,}", text)[0]):
            logger.debug(f"Garbled (unusual sequence): {text[:50]}")
            return True

    # Check for high ratio of very short words (1-2 chars) that aren't articles/pronouns
    words = text.split()
    if len(words) >= 3:
        common_short = {"a", "i", "to", "of", "in", "on", "at", "by", "is", "it", "or", "an", "as", "be", "do", "go", "he", "if", "me", "my", "no", "so", "up", "us", "we"}
        short_words = [w.lower() for w in words if len(w) <= 2 and w.lower() not in common_short]
        short_ratio = len(short_words) / len(words)
        if short_ratio > 0.5:
            logger.debug(f"Garbled (short words ratio {short_ratio:.2f}): {text[:50]}")
            return True

    # Check for repeated character patterns (like "aaaa" or "rrrr")
    if re.search(r"(.)\1{3,}", text.lower()):
        logger.debug(f"Garbled (repeated chars): {text[:50]}")
        return True

    return False


def find_whitespace_rows(
    image: Image.Image,
    threshold: int = 250,
    min_gap: int = 20,
) -> list[tuple[int, int]]:
    """Find rows that are predominantly whitespace (blank lines between text).

    Args:
        image: PIL Image to analyze
        threshold: Minimum mean pixel intensity to consider whitespace (0-255)
        min_gap: Minimum consecutive whitespace rows to be considered a gap

    Returns:
        List of (start_y, end_y) tuples for whitespace regions
    """
    # Convert to grayscale and get pixel data as numpy array
    gray = image.convert("L")
    pixels = np.array(gray)

    # Calculate mean intensity per row
    row_means = pixels.mean(axis=1)

    # Find rows above threshold (whitespace)
    is_whitespace = row_means >= threshold

    # Find runs of consecutive whitespace rows
    whitespace_regions = []
    start = None

    for y, is_white in enumerate(is_whitespace):
        if is_white and start is None:
            start = y
        elif not is_white and start is not None:
            if y - start >= min_gap:
                whitespace_regions.append((start, y))
            start = None

    # Handle case where whitespace extends to bottom
    if start is not None and len(is_whitespace) - start >= min_gap:
        whitespace_regions.append((start, len(is_whitespace)))

    logger.debug(f"Found {len(whitespace_regions)} whitespace regions")
    return whitespace_regions


def find_nearest_whitespace(
    whitespace_regions: list[tuple[int, int]],
    target_y: int,
    search_radius: int = 500,
) -> int | None:
    """Find the whitespace region closest to a target y-coordinate.

    Args:
        whitespace_regions: List of (start_y, end_y) whitespace regions
        target_y: Target y-coordinate to find whitespace near
        search_radius: Maximum distance from target to search

    Returns:
        Midpoint of nearest whitespace region, or None if none found
    """
    best_midpoint = None
    best_distance = float("inf")

    for start, end in whitespace_regions:
        midpoint = (start + end) // 2
        distance = abs(midpoint - target_y)

        if distance < best_distance and distance <= search_radius:
            best_distance = distance
            best_midpoint = midpoint

    return best_midpoint


def find_best_cut_points(
    whitespace_regions: list[tuple[int, int]],
    image_height: int,
    max_chunk_height: int = 2000,
    overlap_percent: float = 20.0,
) -> list[int]:
    """Find cut points that create overlapping chunks while respecting max_chunk_height.

    Calculates the number of chunks needed to fit within max_chunk_height when
    accounting for overlap, then finds whitespace closest to ideal cut positions.

    Args:
        whitespace_regions: List of (start_y, end_y) whitespace regions
        image_height: Total image height
        max_chunk_height: Maximum height per chunk (hard limit)
        overlap_percent: Target overlap percentage between adjacent chunks

    Returns:
        List of y-coordinates to cut at (midpoint of selected whitespace regions)
    """
    if image_height <= max_chunk_height:
        return []

    # Calculate minimum number of chunks needed accounting for overlap
    # Middle chunks have overlap on BOTH sides, so they're larger than edge chunks.
    # For N chunks with P% total overlap:
    #   - Base height per chunk = H/N
    #   - Overlap per boundary = H * P/100 / (N-1)
    #   - Middle chunk height = base + 2*overlap_per_boundary
    # We need middle chunk height <= max_chunk_height
    overlap_fraction = overlap_percent / 100

    # Iteratively find minimum N that keeps all chunks under max_chunk_height
    min_chunks = 2
    for n in range(2, 10):
        base_height = image_height / n
        overlap_per_boundary = image_height * overlap_fraction / (n - 1) if n > 1 else 0
        # Middle chunks get overlap on both sides (worst case)
        middle_chunk_height = base_height + 2 * overlap_per_boundary
        if middle_chunk_height <= max_chunk_height:
            min_chunks = n
            break
    else:
        min_chunks = 10  # Fallback for very tall images

    logger.debug(
        f"Image height {image_height}px, max_chunk {max_chunk_height}px, "
        f"overlap {overlap_percent}% -> need {min_chunks} chunks"
    )

    # Calculate ideal cut positions (evenly spaced)
    # With N chunks, we need N-1 cuts at positions H/N, 2H/N, ..., (N-1)H/N
    target_positions = [
        int(image_height * (i + 1) / min_chunks)
        for i in range(min_chunks - 1)
    ]

    cut_points = []
    search_radius = int(image_height * 0.15)  # Search within 15% of image height

    for target in target_positions:
        if whitespace_regions:
            # Find whitespace closest to target
            best_ws = find_nearest_whitespace(whitespace_regions, target, search_radius)
            if best_ws is not None:
                cut_points.append(best_ws)
                logger.debug(f"Cut point at y={best_ws} (target was {target})")
            else:
                # No whitespace found near target, use target directly
                cut_points.append(target)
                logger.debug(f"Forced cut at y={target} (no whitespace near target)")
        else:
            # No whitespace regions, use target directly
            cut_points.append(target)
            logger.debug(f"Cut at y={target} (no whitespace regions)")

    return cut_points


def smart_chunk_image(
    image: Image.Image,
    max_chunk_height: int = 2000,
    overlap_percent: float = 20.0,
    whitespace_threshold: int = 250,
    min_gap: int = 20,
) -> list["ImageChunk"]:
    """Split image at natural whitespace boundaries with overlap.

    Finds whitespace cut points near ideal overlap positions, then extends
    each chunk past its boundary to create overlapping regions. This ensures
    content at boundaries appears in both adjacent chunks for better OCR.

    Args:
        image: PIL Image to split
        max_chunk_height: Maximum height per chunk in pixels (hard limit)
        overlap_percent: Target overlap percentage between adjacent chunks
        whitespace_threshold: Minimum pixel intensity for whitespace detection
        min_gap: Minimum consecutive whitespace rows to be a valid cut point

    Returns:
        List of ImageChunk objects with overlap
    """
    height = image.height
    width = image.width

    # If image is small enough, no chunking needed
    if height <= max_chunk_height:
        return [ImageChunk(
            image=image,
            chunk_index=0,
            total_chunks=1,
            overlap_top=False,
            overlap_bottom=False,
        )]

    # Find whitespace regions
    whitespace_regions = find_whitespace_rows(
        image, threshold=whitespace_threshold, min_gap=min_gap
    )

    # Find optimal cut points with overlap consideration
    cut_points = find_best_cut_points(
        whitespace_regions,
        height,
        max_chunk_height=max_chunk_height,
        overlap_percent=overlap_percent,
    )

    # If no cut points (image needs chunking but no cuts found), fall back to fixed
    if not cut_points:
        logger.info("No cut points found, using fixed chunking")
        chunk_count = max(2, math.ceil(height * (1 + overlap_percent / 100) / max_chunk_height))
        return split_image_into_chunks(
            image,
            chunk_count=min(chunk_count, 4),
            overlap_percent=overlap_percent,
            smart_chunking=False,
        )

    # Calculate overlap extension per boundary
    num_chunks = len(cut_points) + 1
    # Total overlap pixels = height * overlap_percent / 100
    # Distributed across (num_chunks - 1) boundaries
    overlap_pixels = int(height * (overlap_percent / 100) / (num_chunks - 1))

    logger.debug(
        f"Creating {num_chunks} chunks with {overlap_pixels}px overlap per boundary"
    )

    # Create overlapping chunks
    chunks = []

    for i in range(num_chunks):
        # Determine base boundaries from cut points
        if i == 0:
            base_top = 0
            base_bottom = cut_points[0]
        elif i == num_chunks - 1:
            base_top = cut_points[-1]
            base_bottom = height
        else:
            base_top = cut_points[i - 1]
            base_bottom = cut_points[i]

        # Extend into overlap region
        if i == 0:
            # First chunk: extend bottom into overlap
            top = 0
            bottom = min(base_bottom + overlap_pixels, height)
        elif i == num_chunks - 1:
            # Last chunk: extend top into overlap
            top = max(base_top - overlap_pixels, 0)
            bottom = height
        else:
            # Middle chunks: extend both directions
            top = max(base_top - overlap_pixels, 0)
            bottom = min(base_bottom + overlap_pixels, height)

        # Ensure chunk doesn't exceed max_chunk_height
        chunk_height = bottom - top
        if chunk_height > max_chunk_height:
            # Reduce overlap to fit within limit
            excess = chunk_height - max_chunk_height
            if i == 0:
                # Trim from bottom
                bottom -= excess
            elif i == num_chunks - 1:
                # Trim from top
                top += excess
            else:
                # Trim from both sides
                top += excess // 2
                bottom -= (excess - excess // 2)

            logger.debug(
                f"Chunk {i + 1}: reduced overlap to fit max_chunk_height "
                f"(excess {excess}px)"
            )

        chunk_image = image.crop((0, top, width, bottom))

        logger.debug(
            f"Smart chunk {i + 1}/{num_chunks}: rows {top}-{bottom} "
            f"({chunk_image.height}px, overlap_top={i > 0}, overlap_bottom={i < num_chunks - 1})"
        )

        chunks.append(ImageChunk(
            image=chunk_image,
            chunk_index=i,
            total_chunks=num_chunks,
            overlap_top=i > 0,
            overlap_bottom=i < num_chunks - 1,
        ))

    return chunks


@dataclass
class ImageChunk:
    """A chunk of a page image."""

    image: Image.Image
    chunk_index: int
    total_chunks: int
    overlap_top: bool
    overlap_bottom: bool


@dataclass
class ChunkOCRResult:
    """OCR result from a single chunk."""

    text: str
    chunk_index: int
    page_num: int


def split_image_into_chunks(
    image: Image.Image,
    chunk_count: int = 2,
    overlap_percent: float = 20.0,
    smart_chunking: bool = True,
    max_chunk_height: int = 2000,
) -> list[ImageChunk]:
    """Split an image into overlapping vertical chunks.

    When smart_chunking is enabled (default), uses whitespace detection to find
    natural boundaries for splitting. Falls back to fixed chunking if no good
    whitespace is found.

    For fixed chunking with chunk_count=2 and overlap_percent=20:
    - Chunk 0: top 60% of image
    - Chunk 1: bottom 60% of image
    - Middle 20% is covered by both

    Args:
        image: PIL Image to split
        chunk_count: Number of chunks (2-4) - used for fixed chunking fallback
        overlap_percent: Percentage of image height for total overlap
        smart_chunking: Use whitespace detection for natural chunk boundaries
        max_chunk_height: Maximum height per chunk (cuts at whitespace below this)

    Returns:
        List of ImageChunk objects
    """
    # Use smart chunking if enabled
    if smart_chunking:
        return smart_chunk_image(
            image,
            max_chunk_height=max_chunk_height,
            overlap_percent=overlap_percent,
        )

    if chunk_count < 2:
        return [ImageChunk(
            image=image,
            chunk_index=0,
            total_chunks=1,
            overlap_top=False,
            overlap_bottom=False,
        )]

    height = image.height
    width = image.width

    # Calculate overlap in pixels
    # For N chunks with P% total overlap, each adjacent pair shares overlap/(N-1)
    overlap_pixels = int(height * (overlap_percent / 100))
    overlap_per_boundary = overlap_pixels // (chunk_count - 1)

    # Base height per chunk without overlap
    base_height = height // chunk_count

    # Extended height with overlap
    extended_height = base_height + overlap_per_boundary

    chunks = []
    for i in range(chunk_count):
        # Calculate top and bottom positions
        if i == 0:
            # First chunk: starts at top
            top = 0
            bottom = min(base_height + overlap_per_boundary, height)
        elif i == chunk_count - 1:
            # Last chunk: ends at bottom
            top = max(height - base_height - overlap_per_boundary, 0)
            bottom = height
        else:
            # Middle chunks: centered on their region
            center = (i + 0.5) * base_height
            top = max(int(center - extended_height / 2), 0)
            bottom = min(int(center + extended_height / 2), height)

        # Crop the image
        chunk_image = image.crop((0, top, width, bottom))

        logger.debug(
            f"Chunk {i + 1}/{chunk_count}: rows {top}-{bottom} "
            f"({chunk_image.height}px)"
        )

        chunks.append(ImageChunk(
            image=chunk_image,
            chunk_index=i,
            total_chunks=chunk_count,
            overlap_top=i > 0,
            overlap_bottom=i < chunk_count - 1,
        ))

    return chunks


def line_similarity(line1: str, line2: str) -> float:
    """Calculate similarity between two lines (0.0 to 1.0)."""
    # Strip whitespace for comparison
    l1 = line1.strip()
    l2 = line2.strip()

    # Empty lines
    if not l1 and not l2:
        return 1.0
    if not l1 or not l2:
        return 0.0

    return SequenceMatcher(None, l1, l2).ratio()


def find_overlap_boundary(
    chunk1_lines: list[str],
    chunk2_lines: list[str],
    min_similarity: float = 0.8,
) -> tuple[int, int]:
    """Find where chunk1 ends and chunk2 begins in the overlap region.

    Looks for matching lines between the end of chunk1 and the start of chunk2.

    Args:
        chunk1_lines: Lines from first chunk
        chunk2_lines: Lines from second chunk
        min_similarity: Minimum similarity score to consider a match

    Returns:
        (chunk1_end_line, chunk2_start_line) - indices for cutting.
        chunk1_end_line: Include lines 0 to chunk1_end_line-1 from chunk1
        chunk2_start_line: Include lines from chunk2_start_line onwards from chunk2
    """
    if not chunk1_lines or not chunk2_lines:
        return len(chunk1_lines), 0

    # Look at the last portion of chunk1 (potential overlap region)
    search_depth = min(len(chunk1_lines), max(10, len(chunk1_lines) // 3))

    best_match = None
    best_score = 0.0

    # Try to find where chunk2 begins that matches end of chunk1
    for i in range(len(chunk1_lines) - search_depth, len(chunk1_lines)):
        if i < 0:
            continue

        line1 = chunk1_lines[i]
        if not line1.strip():
            continue

        # Look for this line in the first portion of chunk2
        for j in range(min(search_depth, len(chunk2_lines))):
            line2 = chunk2_lines[j]
            similarity = line_similarity(line1, line2)

            if similarity >= min_similarity:
                # Found a match - calculate a score that prefers later chunk1 matches
                # and earlier chunk2 matches
                score = similarity + (i / len(chunk1_lines)) - (j / len(chunk2_lines)) * 0.5

                if score > best_score:
                    best_score = score
                    best_match = (i, j)

    if best_match:
        logger.debug(
            f"Found overlap boundary: chunk1 line {best_match[0]} matches "
            f"chunk2 line {best_match[1]} (score: {best_score:.2f})"
        )
        # Include up to and including the matching line from chunk1,
        # skip past the matching line in chunk2
        return best_match[0] + 1, best_match[1] + 1

    # No match found - fall back to simple concatenation
    logger.debug("No overlap boundary found, using simple concatenation")
    return len(chunk1_lines), 0


def merge_chunk_texts(
    chunks: list[ChunkOCRResult],
    deduplicate: bool = True,
) -> str:
    """Merge chunk texts with optional deduplication of overlapping content.

    When deduplicate=True, uses fuzzy line matching to detect and remove
    duplicate content that appears in overlapping regions between chunks.

    Args:
        chunks: List of OCR results from each chunk, in order
        deduplicate: Whether to remove duplicate content from overlaps

    Returns:
        Merged text with overlapping content deduplicated
    """
    if not chunks:
        return ""

    if len(chunks) == 1:
        return chunks[0].text

    # Sort by chunk index
    sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index)

    if not deduplicate:
        return "\n".join(chunk.text for chunk in sorted_chunks)

    # Merge with deduplication
    merged_lines: list[str] = []

    for i, chunk in enumerate(sorted_chunks):
        chunk_lines = chunk.text.split("\n")

        if i == 0:
            # First chunk: include all lines, but find where overlap starts
            if len(sorted_chunks) > 1:
                next_chunk_lines = sorted_chunks[1].text.split("\n")
                chunk1_end, _ = find_overlap_boundary(chunk_lines, next_chunk_lines)
                merged_lines.extend(chunk_lines[:chunk1_end])
                logger.debug(
                    f"Chunk {i}: keeping lines 0-{chunk1_end - 1} of {len(chunk_lines)}"
                )
            else:
                merged_lines.extend(chunk_lines)
        else:
            # Subsequent chunks: find where overlap ends
            prev_chunk_lines = sorted_chunks[i - 1].text.split("\n")
            _, chunk2_start = find_overlap_boundary(prev_chunk_lines, chunk_lines)

            # For non-last chunks, also find where overlap with next chunk starts
            if i < len(sorted_chunks) - 1:
                next_chunk_lines = sorted_chunks[i + 1].text.split("\n")
                chunk1_end, _ = find_overlap_boundary(chunk_lines, next_chunk_lines)
                merged_lines.extend(chunk_lines[chunk2_start:chunk1_end])
                logger.debug(
                    f"Chunk {i}: keeping lines {chunk2_start}-{chunk1_end - 1} of {len(chunk_lines)}"
                )
            else:
                # Last chunk: include from overlap end to end
                merged_lines.extend(chunk_lines[chunk2_start:])
                logger.debug(
                    f"Chunk {i}: keeping lines {chunk2_start}-{len(chunk_lines) - 1} of {len(chunk_lines)}"
                )

    return "\n".join(merged_lines)
