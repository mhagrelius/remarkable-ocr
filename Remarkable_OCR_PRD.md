# Product Requirements Document: Remarkable Notes OCR Pipeline

## Executive Summary

A self-hosted CLI application that extracts handwritten notes from Remarkable tablet PDF exports, performs high-accuracy OCR using local vision-language models, and outputs clean, structured text optimized for LLM consumption. 

**Version:** 1.0 (MVP - OCR + Text Export)  
**Status:** In Development  
**Date:** January 2026

---

## 1. Overview

### Problem Statement
The Remarkable tablet's built-in OCR is cloud-dependent and produces mediocre results for handwritten notes. Users need:
- Local, privacy-preserving handwriting recognition
- Output format optimized for LLM consumption (summaries, synthesis, RAG)
- Reliable batch processing without manual intervention
- Extensible architecture for future ML enhancement

### Scope: MVP (Phase 1)
**In Scope:**
- Extract handwritten notes from Remarkable PDF exports
- Local OCR using Qwen2.5-VL via Ollama
- Clean text output with confidence metadata
- CLI interface for triggering conversions
- Batch processing with error recovery
- Markdown-formatted output

**Out of Scope (Future Phases):**
- Vector embeddings and storage
- Semantic chunking
- Retrieval-augmented generation (RAG) setup
- Fine-tuning on user handwriting samples
- Web UI dashboard
- Cloud sync integration

---

## 2. Goals & Success Metrics

### Primary Goals
1. **Accuracy:** Achieve 75%+ character-level accuracy on typical handwritten notes
2. **Usability:** Single CLI command to process Remarkable exports
3. **Privacy:** 100% local processing—no external APIs or cloud calls
4. **Speed:** Process 10-page note set in <5 minutes on consumer GPU
5. **Output Quality:** Produce text directly consumable by LLMs without manual cleanup

### Success Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Character Accuracy | 75%+ | Manual spot-check on sample notes |
| Processing Speed | <30s per page | Timing on test PDF set |
| Confidence Metadata | Present on all outputs | JSON includes `confidence_score` per segment |
| Error Recovery | 100% | Failed pages logged; pipeline continues |
| LLM Compatibility | Pass-through ready | Output format directly usable by Claude/GPT |

---

## 3. User Personas & Use Cases

### Primary User
**The Technical Note-Taker**
- Senior cloud architect / ML engineer
- Takes handwritten notes on Remarkable during research/meetings
- Wants to: extract notes → feed to LLM for synthesis → integrate into knowledge base
- Pain point: Manual transcription is slow; cloud OCR is privacy risk

### Use Cases

**UC1: Single Note Transcription**
```
User exports PDF from Remarkable
→ Runs: remarkable-ocr process notes.pdf
→ Output: notes.txt (clean markdown)
→ Feeds output to Claude for summary
```

**UC2: Batch Processing Meeting Notes**
```
User has 3 PDFs from daily standups
→ Runs: remarkable-ocr process *.pdf --output summary/
→ Output: summary/standup_2026_01_10.txt
→ Auto-trigger: Cat all .txt → pipe to LLM for synthesis
```

**UC3: Continuous Note Ingestion**
```
Setup file watcher on Remarkable export folder
→ New PDF detected → Auto-process via OCR
→ Output to structured folder: notes/{date}/{title}.txt
→ LLM agent periodically reviews new notes
```

---

## 4. Feature Specifications

### 4.1 Core Features

#### Feature 1: PDF to Image Conversion
**What:** Extract individual pages from Remarkable PDF exports as images
**How:**
- Input: `.pdf` file (single or batch)
- Process: Convert each page to 1288px width image (optimal for Qwen2.5-VL)
- Output: Temporary image files in `./.remarkable_cache/`
- Metadata: Track original page number, DPI, dimensions

**Requirements:**
- Support PDFs up to 200 pages
- Preserve image quality (at least 150 DPI equivalent)
- Handle corrupted PDFs gracefully (skip, log, continue)
- Clean up temp images after processing

#### Feature 2: Local OCR Engine
**What:** Convert handwritten images to text using Qwen2.5-VL
**How:**
- Model: Qwen2.5-VL-7B via Ollama
- Inference: Call Ollama local API endpoint
- Prompt: Specialized handwriting extraction prompt
- Output: Structured text with per-segment metadata

**Requirements:**
- Require Ollama running (check on startup)
- Fallback to Qwen2.5-VL-3B if 7B unavailable
- Timeout: 60 seconds per page (with retry logic)
- Confidence scoring: Extract from LLM response
- Handle edge cases:
  - Blank/empty pages → mark as blank
  - Images with no text → return `[No text detected]`
  - Mixed printed + handwritten → extract both
  - Diagrams/sketches → convert to ASCII art (if readable)

#### Feature 3: Text Cleaning & Formatting
**What:** Post-process OCR output for LLM consumption
**How:**
- Remove redundant whitespace
- Fix common OCR artifacts (e.g., `rn` → `m`)
- Normalize line breaks (preserve paragraph structure)
- Retain original emphasis/underlining markers → convert to Markdown
- Add metadata headers (date, source file, page count)

**Output Format:**
```markdown
---
source: my_notes.pdf
date: 2026-01-10
pages: 3
confidence_avg: 0.78
---

# Page 1

[Extracted text from page 1]

---

# Page 2

[Extracted text from page 2]

---
```

#### Feature 4: CLI Interface
**What:** Command-line tool for triggering OCR pipeline
**Commands:**

```bash
# Single file processing
remarkable-ocr process <pdf_path>

# Batch processing
remarkable-ocr process <pdf_path_pattern> --output <output_dir>

# With options
remarkable-ocr process notes.pdf \
  --model qwen2.5-vl:7b \
  --confidence-threshold 0.6 \
  --output-format markdown \
  --keep-temp-images \
  --verbose

# Status/health check
remarkable-ocr health

# List available models
remarkable-ocr models
```

**Options:**
| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--output` | Path | `./output/` | Directory for output text files |
| `--model` | String | `qwen2.5-vl:7b` | Ollama model to use |
| `--confidence-threshold` | Float | `0.5` | Minimum confidence to include text |
| `--output-format` | Enum | `markdown` | `markdown`, `json`, `txt` |
| `--keep-temp-images` | Flag | False | Preserve intermediate image files |
| `--verbose` | Flag | False | Debug logging output |
| `--batch-size` | Int | `1` | Parallel page processing (future) |

#### Feature 5: Error Handling & Logging
**What:** Robust error recovery and observability
**Behaviors:**
- If Ollama unavailable: Exit with clear error message
- If PDF corrupted: Skip corrupted pages, continue processing
- If OCR times out: Log timeout, retry once, skip if persistent
- If output dir missing: Create automatically
- All errors logged to: `~/.remarkable_ocr/logs/remarkable_ocr.log`

**Log Levels:**
```
DEBUG: Page 1 processing started (1024x2048 px)
INFO: Processing notes.pdf [3 pages]
WARN: Page 2 timeout (60s), retrying...
ERROR: PDF file not found: /path/to/notes.pdf
```

---

## 5. Technical Architecture

### 5.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Entry Point                      │
│                   (remarkable-ocr process)                   │
└────────────────────────────┬────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────┐      ┌──────────────┐    ┌─────────────┐
    │   PDF   │      │ PDF to Image │    │   Config    │
    │ Loader  │──────│ Converter    │    │ Management  │
    └─────────┘      └──────────────┘    └─────────────┘
         │                   │
         └───────────────────┼───────────────────┐
                             │                   │
                             ▼                   │
                    ┌──────────────────┐         │
                    │  Image Cache     │         │
                    │ (.remarkable_    │         │
                    │  cache/)         │         │
                    └──────────────────┘         │
                             │                   │
                             ▼                   │
                    ┌──────────────────┐         │
                    │  OCR Engine      │         │
                    │  (Ollama API)    │         │
                    │  Qwen2.5-VL      │         │
                    └──────────────────┘         │
                             │                   │
                             ▼                   │
                    ┌──────────────────┐         │
                    │ Text Processor   │         │
                    │ (clean, format)  │         │
                    └──────────────────┘         │
                             │                   │
         ┌───────────────────┴───────────────────┘
         │
         ▼
    ┌────────────────┐
    │  Output Files  │
    │ (txt/md/json)  │
    └────────────────┘
```

### 5.2 Data Flow

```
Input: remarkable_notes.pdf (3 pages, 5MB)
  ↓
[PDF Validator] → Valid? Yes
  ↓
[Image Converter] → Page 1: 1288x1920px PNG → cache/page_001.png
                    Page 2: 1288x1920px PNG → cache/page_002.png
                    Page 3: 1288x1920px PNG → cache/page_003.png
  ↓
[OCR Engine Loop]
  ├─ Page 1 → Ollama API (Qwen2.5-VL) → "Meeting notes from standup..."
  ├─ Page 2 → Ollama API (Qwen2.5-VL) → "TODO items: Review PR #234..."
  └─ Page 3 → Ollama API (Qwen2.5-VL) → "Decision: Move to Azure..."
  ↓
[Text Processor]
  ├─ Clean whitespace
  ├─ Fix OCR artifacts
  ├─ Extract confidence scores
  ├─ Add metadata headers
  ↓
[Output Writer]
  └─ output/remarkable_notes.md
     (with frontmatter, page breaks, confidence data)
  ↓
Output: Clean markdown ready for LLM consumption
```

### 5.3 Component Specifications

#### PDF Loader
- **Input:** File path (single or glob pattern)
- **Output:** PyPDF2.PdfReader object or list of page objects
- **Error Handling:** Validate PDF format, handle encrypted PDFs
- **Caching:** Don't re-load same file twice in session

#### Image Converter
- **Input:** PDF page object
- **Process:** Convert to PNG at 1288px width (Qwen2.5-VL optimal size)
- **Output:** PIL.Image object + temporary file
- **Metadata:** Store original DPI, dimensions, page number
- **Cleanup:** Delete temp images after processing (unless `--keep-temp-images`)

#### OCR Engine (Ollama Wrapper)
- **Model:** Qwen2.5-VL (7B preferred, 3B fallback)
- **Endpoint:** `http://localhost:11434/api/generate` (Ollama default)
- **Prompt:** Specialized handwriting extraction prompt (from system)
- **Inference Settings:**
  ```
  temperature: 0.1 (deterministic)
  top_p: 0.9
  timeout: 60 seconds
  retry: 1 attempt
  ```
- **Output Parsing:** Extract text from LLM response + confidence if available

#### Text Processor
- **Input:** Raw OCR output
- **Operations:**
  1. Strip leading/trailing whitespace
  2. Normalize newlines (preserve empty lines for paragraph breaks)
  3. Fix common OCR errors (regex patterns)
  4. Detect and preserve structure (headings, lists, tables)
  5. Convert emphasis markers to Markdown
  6. Generate metadata (source, date, confidence)
- **Output:** Formatted text + metadata dict

#### Output Writer
- **Formats:** Markdown (default), JSON, plain text
- **Files:**
  - Markdown: `output/{input_filename}.md` (with frontmatter)
  - JSON: `output/{input_filename}.json` (structured data)
  - Text: `output/{input_filename}.txt` (plain)
- **Metadata:** Include source file, processing date, page count, confidence

---

## 6. Implementation Plan

### Phase 1: MVP (Weeks 1-2)
**Deliverables:**
- [x] PDF loading + image conversion
- [x] Ollama integration (basic API wrapper)
- [x] OCR processing (single page → full PDF)
- [x] Text cleaning + Markdown formatting
- [x] CLI interface (basic commands)
- [x] Error handling + logging
- [x] Testing on sample Remarkable PDFs

**Acceptance Criteria:**
- Process 5-page Remarkable PDF in <3 minutes
- Output markdown directly consumable by Claude/GPT
- Handle missing Ollama gracefully
- Log all errors to file

### Phase 2: Optimization (Future)
- Parallel page processing
- Confidence thresholding + selective output
- Fine-tuning on user handwriting samples
- Advanced text cleaning (table detection, diagram handling)

### Phase 3: Vector Storage (Future)
- Integrate vector embeddings
- Chroma or Weaviate storage
- Semantic chunking for RAG

---

## 7. Non-Functional Requirements

### Performance
- Single page: <30 seconds (including image conversion + OCR)
- 10-page batch: <5 minutes
- Memory: <4GB during processing
- GPU VRAM: 8GB minimum (3B model) / 16GB+ (7B model)

### Reliability
- Zero data loss (always preserve input PDFs)
- Error recovery: Continue on page failure
- Retry logic: 1 automatic retry on timeout
- Uptime: Offline-first (no external dependencies)

### Usability
- Single command: `remarkable-ocr process <file>`
- Clear error messages
- Auto-create output directories
- Sensible defaults (output → `./output/`)

### Maintainability
- Clean Python codebase (type hints, docstrings)
- Modular architecture (loader, converter, ocr, processor, writer)
- Configuration via env vars + CLI flags
- Test coverage >80%

---

## 8. Configuration & Defaults

### Environment Variables
```bash
REMARKABLE_OCR_MODEL=qwen2.5-vl:7b          # Ollama model
REMARKABLE_OCR_OUTPUT=./output/              # Default output dir
REMARKABLE_OCR_CONFIDENCE_THRESHOLD=0.5      # Min confidence
REMARKABLE_OCR_TIMEOUT=60                    # Seconds per page
REMARKABLE_OCR_LOG_LEVEL=INFO                # DEBUG, INFO, WARN, ERROR
OLLAMA_API_BASE=http://localhost:11434       # Ollama endpoint
```

### Config File (Optional)
`~/.remarkable_ocr/config.yaml`
```yaml
model: qwen2.5-vl:7b
output_dir: ./output/
confidence_threshold: 0.5
timeout: 60
log_level: INFO
ollama_api: http://localhost:11434
keep_temp_images: false
```

---

## 9. Output Specifications

### Format 1: Markdown (Default)
```markdown
---
source: meeting_notes.pdf
date_processed: 2026-01-10T14:30:00Z
pages: 3
confidence_avg: 0.78
model: qwen2.5-vl:7b
---

# Page 1

## Meeting Notes - January 10

Attendees: Alice, Bob, Carol

Key Discussion Points:
- Reviewed Q1 roadmap
- Decided on Azure migration timeline
- Assigned action items

---

# Page 2

## Action Items

1. Alice: Complete architectural review (due Jan 17)
2. Bob: Set up CI/CD pipeline (due Jan 24)
3. Carol: Update documentation (due Jan 15)

---

# Page 3

## Decision Log

**Decision:** Prioritize Azure Copilot integration over on-prem AI agents

**Rationale:** Faster time-to-market, built-in security compliance

**Owner:** Alice

---
```

### Format 2: JSON
```json
{
  "metadata": {
    "source": "meeting_notes.pdf",
    "date_processed": "2026-01-10T14:30:00Z",
    "pages": 3,
    "confidence_avg": 0.78,
    "model": "qwen2.5-vl:7b"
  },
  "pages": [
    {
      "page_number": 1,
      "text": "Meeting Notes - January 10\n\nAttendees: Alice, Bob, Carol\n...",
      "confidence": 0.82,
      "has_diagrams": false
    },
    {
      "page_number": 2,
      "text": "Action Items\n\n1. Alice: Complete architectural review...",
      "confidence": 0.76,
      "has_diagrams": false
    }
  ]
}
```

### Format 3: Plain Text
```
[remarkable_notes.pdf - Processed 2026-01-10]

=== Page 1 ===

Meeting Notes - January 10

Attendees: Alice, Bob, Carol

Key Discussion Points:
- Reviewed Q1 roadmap
- Decided on Azure migration timeline
- Assigned action items

=== Page 2 ===

Action Items

1. Alice: Complete architectural review (due Jan 17)
2. Bob: Set up CI/CD pipeline (due Jan 24)
3. Carol: Update documentation (due Jan 15)

=== Page 3 ===

Decision Log

Decision: Prioritize Azure Copilot integration over on-prem AI agents

Rationale: Faster time-to-market, built-in security compliance

Owner: Alice
```

---

## 10. Out of Scope (MVP)

- ❌ Vector embeddings / chunking
- ❌ RAG pipelines
- ❌ Web UI dashboard
- ❌ Cloud sync (Remarkable cloud, Dropbox, etc.)
- ❌ Fine-tuning on user handwriting
- ❌ Multi-language support (English only)
- ❌ Diagram/sketch interpretation (ASCII art only)
- ❌ Parallel processing (single-threaded MVP)
- ❌ Integration with Copilot Studio
- ❌ Mobile app support

---

## 11. Success Criteria (MVP Release)

**Functional:**
- ✅ Extract text from Remarkable PDFs with 75%+ accuracy
- ✅ Single CLI command to trigger processing
- ✅ Output markdown ready for LLM consumption
- ✅ Handle 5-200 page PDFs
- ✅ Error recovery (continue on failures)
- ✅ Local-only processing (no external APIs)

**Non-Functional:**
- ✅ Process time: <30s per page
- ✅ Memory usage: <4GB
- ✅ Error logging to file
- ✅ Clear CLI help + examples

**Documentation:**
- ✅ README with setup instructions
- ✅ Usage examples (single file, batch, CLI options)
- ✅ Troubleshooting guide
- ✅ Config reference

---

## 12. Appendix: Technical Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Runtime | Python 3.10+ | Your ecosystem, asyncio support future-ready |
| PDF Processing | PyPDF2 + pdf2image | Lightweight, no external binaries |
| Image Processing | Pillow | Efficient image manipulation |
| LLM Runtime | Ollama | Local, simple API, cross-platform |
| Vision Model | Qwen2.5-VL | SOTA for handwriting, low hallucination |
| CLI Framework | Typer | Type-safe, auto-generates help |
| Config | python-dotenv + Pydantic | Type-safe, validation |
| Logging | Python logging + rich | Structured logs with colors |
| Testing | pytest + pytest-asyncio | Async-ready testing |

---

## Glossary

- **Remarkable:** E-ink tablet for note-taking and sketching
- **OCR:** Optical Character Recognition (printed text)
- **HWR:** Handwriting Recognition (our focus)
- **Qwen2.5-VL:** Vision-Language model from Alibaba for multimodal understanding
- **Ollama:** Local LLM runtime service
- **LLM:** Large Language Model (Claude, GPT, etc.)
- **RAG:** Retrieval-Augmented Generation (future phase)

---

**Document Version:** 1.0  
**Last Updated:** January 10, 2026  
**Owner:** [Your Name]  
**Status:** APPROVED FOR DEVELOPMENT
