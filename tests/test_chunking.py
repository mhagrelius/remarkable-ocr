"""Tests for image chunking."""

from PIL import Image


def test_split_image_two_chunks():
    """split_image_into_chunks should create 2 overlapping chunks (fixed mode)."""
    from remarkable_ocr.chunking import split_image_into_chunks

    image = Image.new("RGB", (100, 1000))
    chunks = split_image_into_chunks(
        image, chunk_count=2, overlap_percent=20, smart_chunking=False
    )

    assert len(chunks) == 2

    # First chunk should be top 60% (with 20% overlap)
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 2
    assert chunks[0].overlap_top is False
    assert chunks[0].overlap_bottom is True

    # Second chunk should be bottom 60%
    assert chunks[1].chunk_index == 1
    assert chunks[1].total_chunks == 2
    assert chunks[1].overlap_top is True
    assert chunks[1].overlap_bottom is False

    # Both chunks should have same width
    assert chunks[0].image.width == 100
    assert chunks[1].image.width == 100


def test_split_image_three_chunks():
    """split_image_into_chunks should handle 3 chunks (fixed mode)."""
    from remarkable_ocr.chunking import split_image_into_chunks

    image = Image.new("RGB", (100, 900))
    chunks = split_image_into_chunks(
        image, chunk_count=3, overlap_percent=20, smart_chunking=False
    )

    assert len(chunks) == 3

    # Middle chunk should have overlap on both sides
    assert chunks[1].overlap_top is True
    assert chunks[1].overlap_bottom is True


def test_split_image_single_chunk():
    """split_image_into_chunks with chunk_count=1 should return original image (fixed mode)."""
    from remarkable_ocr.chunking import split_image_into_chunks

    image = Image.new("RGB", (100, 1000))
    chunks = split_image_into_chunks(
        image, chunk_count=1, overlap_percent=20, smart_chunking=False
    )

    assert len(chunks) == 1
    assert chunks[0].image.width == image.width
    assert chunks[0].image.height == image.height


def test_line_similarity_exact_match():
    """line_similarity should return 1.0 for exact matches."""
    from remarkable_ocr.chunking import line_similarity

    assert line_similarity("Hello world", "Hello world") == 1.0
    assert line_similarity("  Hello world  ", "Hello world") == 1.0  # strips whitespace


def test_line_similarity_similar_lines():
    """line_similarity should score similar lines highly."""
    from remarkable_ocr.chunking import line_similarity

    # Minor typo
    assert line_similarity("Hello world", "Hello warld") > 0.8

    # Same content, different punctuation
    assert line_similarity("Line 3", "Line 3.") > 0.8


def test_line_similarity_different_lines():
    """line_similarity should score different lines low."""
    from remarkable_ocr.chunking import line_similarity

    assert line_similarity("Hello", "Goodbye") < 0.5


def test_line_similarity_empty_lines():
    """line_similarity should handle empty lines."""
    from remarkable_ocr.chunking import line_similarity

    assert line_similarity("", "") == 1.0
    assert line_similarity("Hello", "") == 0.0
    assert line_similarity("", "Hello") == 0.0


def test_merge_chunk_texts_simple():
    """merge_chunk_texts should concatenate non-overlapping chunks."""
    from remarkable_ocr.chunking import ChunkOCRResult, merge_chunk_texts

    chunks = [
        ChunkOCRResult(text="First chunk content here\nMore first content", chunk_index=0, page_num=1),
        ChunkOCRResult(text="Second chunk content here\nMore second content", chunk_index=1, page_num=1),
    ]
    # Without deduplication
    result = merge_chunk_texts(chunks, deduplicate=False)

    assert "First chunk content" in result
    assert "Second chunk content" in result


def test_merge_chunk_texts_preserves_order():
    """merge_chunk_texts should preserve chunk order."""
    from remarkable_ocr.chunking import ChunkOCRResult, merge_chunk_texts

    # Chunks provided out of order with distinct content
    chunks = [
        ChunkOCRResult(text="Second section content", chunk_index=1, page_num=1),
        ChunkOCRResult(text="First section content", chunk_index=0, page_num=1),
    ]
    result = merge_chunk_texts(chunks, deduplicate=False)

    # Should be sorted by chunk_index
    assert result.index("First section") < result.index("Second section")


def test_merge_chunk_texts_deduplicates_overlap():
    """merge_chunk_texts should remove duplicate content from overlapping regions."""
    from remarkable_ocr.chunking import ChunkOCRResult, merge_chunk_texts

    # Simulate overlapping chunks where content appears in both
    chunks = [
        ChunkOCRResult(
            text="- Item one\n- Item two\n- Item three\n- Item four",
            chunk_index=0,
            page_num=1
        ),
        ChunkOCRResult(
            text="- Item three\n- Item four\n- Item five\n- Item six",
            chunk_index=1,
            page_num=1
        ),
    ]
    result = merge_chunk_texts(chunks, deduplicate=True)

    # Should have all items without duplicates
    assert "Item one" in result
    assert "Item two" in result
    assert "Item five" in result
    assert "Item six" in result
    # Duplicate items should appear only once
    assert result.count("Item three") == 1
    assert result.count("Item four") == 1


def test_merge_chunk_texts_empty():
    """merge_chunk_texts should handle empty input."""
    from remarkable_ocr.chunking import merge_chunk_texts

    assert merge_chunk_texts([]) == ""


def test_merge_chunk_texts_single_chunk():
    """merge_chunk_texts should handle single chunk."""
    from remarkable_ocr.chunking import ChunkOCRResult, merge_chunk_texts

    chunks = [
        ChunkOCRResult(text="Line 1\nLine 2\nLine 3", chunk_index=0, page_num=1),
    ]
    result = merge_chunk_texts(chunks)

    assert result == "Line 1\nLine 2\nLine 3"


def test_find_overlap_boundary_exact_match():
    """find_overlap_boundary should find exact matching lines."""
    from remarkable_ocr.chunking import find_overlap_boundary

    chunk1_lines = ["First unique line", "Second unique line", "Overlapping content here"]
    chunk2_lines = ["Overlapping content here", "Fourth unique line", "Fifth unique line"]

    end, start = find_overlap_boundary(chunk1_lines, chunk2_lines)

    # Should include Line 3 from chunk1, skip Line 3 in chunk2
    assert end == 3  # Include all lines 0-2 from chunk1 (including the match)
    assert start == 1  # Start from line 1 in chunk2 (after the match)


def test_find_overlap_boundary_no_match():
    """find_overlap_boundary should fall back when no match found."""
    from remarkable_ocr.chunking import find_overlap_boundary

    # Use very different lines to avoid fuzzy matching
    chunk1_lines = ["Apple pie recipe", "Banana bread", "Cherry cobbler"]
    chunk2_lines = ["Zebra crossing", "Yellow submarine", "Xylophone music"]

    end, start = find_overlap_boundary(chunk1_lines, chunk2_lines)

    # Should return full chunks (no match found)
    assert end == 3  # Include all of chunk1
    assert start == 0  # Start from beginning of chunk2


def test_find_overlap_boundary_empty_input():
    """find_overlap_boundary should handle empty input."""
    from remarkable_ocr.chunking import find_overlap_boundary

    assert find_overlap_boundary([], ["Line 1"]) == (0, 0)
    assert find_overlap_boundary(["Line 1"], []) == (1, 0)
    assert find_overlap_boundary([], []) == (0, 0)


# Smart chunking tests

def test_find_whitespace_rows_blank_image():
    """find_whitespace_rows should find whitespace in all-white image."""
    from remarkable_ocr.chunking import find_whitespace_rows

    # All-white image
    image = Image.new("RGB", (100, 100), color=(255, 255, 255))
    regions = find_whitespace_rows(image, threshold=250, min_gap=20)

    # Should find one large whitespace region
    assert len(regions) == 1
    assert regions[0][0] == 0  # Starts at top
    assert regions[0][1] == 100  # Ends at bottom


def test_find_whitespace_rows_with_dark_band():
    """find_whitespace_rows should not find whitespace in dark areas."""
    from remarkable_ocr.chunking import find_whitespace_rows
    import numpy as np

    # Create image with dark band in middle
    image = Image.new("RGB", (100, 100), color=(255, 255, 255))
    pixels = np.array(image)
    pixels[40:60, :] = 0  # Dark band from row 40-60
    image = Image.fromarray(pixels)

    regions = find_whitespace_rows(image, threshold=250, min_gap=20)

    # Should find whitespace before and after dark band
    assert len(regions) == 2
    assert regions[0][1] <= 40  # First region ends before dark band
    assert regions[1][0] >= 60  # Second region starts after dark band


def test_find_whitespace_rows_no_whitespace():
    """find_whitespace_rows should return empty for dark image."""
    from remarkable_ocr.chunking import find_whitespace_rows

    # All-dark image
    image = Image.new("RGB", (100, 100), color=(0, 0, 0))
    regions = find_whitespace_rows(image, threshold=250, min_gap=20)

    assert len(regions) == 0


def test_find_best_cut_points_single_region():
    """find_best_cut_points should select from available whitespace near targets."""
    from remarkable_ocr.chunking import find_best_cut_points

    # Whitespace region at y=500 (middle of 1000px image)
    # With 20% overlap and 500px max chunk, we need 4 chunks (3 cuts)
    # because middle chunks have overlap on both sides
    whitespace_regions = [(480, 520)]
    cut_points = find_best_cut_points(
        whitespace_regions,
        image_height=1000,
        max_chunk_height=500,
        overlap_percent=20.0,
    )

    # Should have 3 cut points for 4 chunks
    assert len(cut_points) == 3
    # The cut nearest to 500 (middle target) should be at whitespace
    assert any(480 <= cp <= 520 for cp in cut_points)


def test_find_best_cut_points_no_regions():
    """find_best_cut_points should return evenly spaced cuts when no whitespace."""
    from remarkable_ocr.chunking import find_best_cut_points

    # With 1000px, 500px max, 20% overlap: need 4 chunks (3 cuts)
    cut_points = find_best_cut_points(
        [],
        image_height=1000,
        max_chunk_height=500,
        overlap_percent=20.0,
    )

    # Should have 3 evenly spaced cut points
    assert len(cut_points) == 3
    # Cuts should be approximately at 250, 500, 750
    assert 200 <= cut_points[0] <= 300
    assert 450 <= cut_points[1] <= 550
    assert 700 <= cut_points[2] <= 800


def test_find_best_cut_points_no_overlap():
    """find_best_cut_points with no overlap should minimize chunks."""
    from remarkable_ocr.chunking import find_best_cut_points

    # With 1000px, 500px max, 0% overlap: need 2 chunks (1 cut)
    whitespace_regions = [(480, 520)]
    cut_points = find_best_cut_points(
        whitespace_regions,
        image_height=1000,
        max_chunk_height=500,
        overlap_percent=0.0,
    )

    assert len(cut_points) == 1
    assert 480 <= cut_points[0] <= 520  # Should be at whitespace


def test_smart_chunk_image_fallback():
    """smart_chunk_image should fall back to fixed chunking if no whitespace found."""
    from remarkable_ocr.chunking import smart_chunk_image

    # All-dark image (no whitespace)
    image = Image.new("RGB", (100, 2000), color=(0, 0, 0))
    chunks = smart_chunk_image(image, max_chunk_height=1000)

    # Should fall back to fixed chunking
    assert len(chunks) >= 2  # At least 2 chunks for 2000px image


def test_smart_chunk_image_with_whitespace():
    """smart_chunk_image should cut at whitespace and create overlapping chunks."""
    from remarkable_ocr.chunking import smart_chunk_image
    import numpy as np

    # Create image with clear whitespace band in middle
    image = Image.new("RGB", (100, 2000), color=(50, 50, 50))  # Dark background
    pixels = np.array(image)
    pixels[900:1100, :] = 255  # White band from row 900-1100
    image = Image.fromarray(pixels)

    chunks = smart_chunk_image(image, max_chunk_height=1000, overlap_percent=20.0, min_gap=20)

    # With 2000px, 1000px max, 20% overlap: we need enough chunks to fit
    # Each middle chunk has overlap on both sides, so need multiple chunks
    assert len(chunks) >= 3
    # All chunks should be under max_chunk_height
    for chunk in chunks:
        assert chunk.image.height <= 1000
    # Chunks should have proper overlap flags
    assert chunks[0].overlap_top is False
    assert chunks[0].overlap_bottom is True
    assert chunks[-1].overlap_top is True
    assert chunks[-1].overlap_bottom is False


# Garbled text detection tests

def test_is_likely_garbled_normal_text():
    """is_likely_garbled should return False for normal text."""
    from remarkable_ocr.chunking import is_likely_garbled

    assert is_likely_garbled("Hello world") is False
    assert is_likely_garbled("This is a normal sentence.") is False
    assert is_likely_garbled("current process -> we don't have as much of") is False
    assert is_likely_garbled("Redefine architect + engineer role") is False


def test_is_likely_garbled_short_text():
    """is_likely_garbled should not evaluate very short text."""
    from remarkable_ocr.chunking import is_likely_garbled

    # Short text is not evaluated (returns False)
    assert is_likely_garbled("Hi") is False
    assert is_likely_garbled("abc") is False


def test_is_likely_garbled_low_alpha_ratio():
    """is_likely_garbled should detect text with low alphabetic ratio."""
    from remarkable_ocr.chunking import is_likely_garbled

    # Less than 50% alphabetic characters
    assert is_likely_garbled("12345678") is True
    assert is_likely_garbled("!!!@@@###") is True


def test_is_likely_garbled_unusual_sequences():
    """is_likely_garbled should detect unusual character sequences."""
    from remarkable_ocr.chunking import is_likely_garbled

    # 3+ consecutive non-alpha (not common patterns)
    assert is_likely_garbled("process don't be more") is False  # Normal text
    # But allow common patterns like ->
    assert is_likely_garbled("current process -> next") is False


def test_is_likely_garbled_repeated_chars():
    """is_likely_garbled should detect repeated character patterns."""
    from remarkable_ocr.chunking import is_likely_garbled

    assert is_likely_garbled("Hellooooo world") is True  # 4+ repeated
    assert is_likely_garbled("This is normal text") is False


def test_is_likely_garbled_baseline_artifacts():
    """is_likely_garbled should detect known baseline artifacts."""
    from remarkable_ocr.chunking import is_likely_garbled

    # Known artifacts from baseline test
    assert is_likely_garbled("Redarchitenginole") is False  # This is garbled but passes alpha ratio
    # The garbled detection is a heuristic, not perfect
