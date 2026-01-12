"""Command-line interface using Typer."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from remarkable_ocr import __version__
from remarkable_ocr.chunking import (
    ChunkOCRResult,
    merge_chunk_texts,
    split_image_into_chunks,
)
from remarkable_ocr.config import Settings
from remarkable_ocr.logging import setup_logging, get_logger
from remarkable_ocr.ocr import (
    OCRResult,
    check_ollama_health,
    get_available_models,
    process_image,
)
from remarkable_ocr.pdf import extract_pages, get_page_count
from remarkable_ocr.processor import clean_text
from remarkable_ocr.writer import write_output

# Load settings for default values
settings = Settings()

app = typer.Typer(
    name="remarkable-ocr",
    help="Extract handwritten notes from Remarkable PDF exports using local OCR.",
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"remarkable-ocr {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-V", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """Remarkable OCR - Extract handwritten notes from Remarkable PDFs."""
    pass


@app.command()
def health(
    model: Annotated[str, typer.Option("--model", "-m")] = settings.model,
    host: Annotated[str, typer.Option("--host")] = settings.ollama_host,
) -> None:
    """Check if Ollama is running and model is available."""
    console.print(f"Checking Ollama at {host}...")

    if check_ollama_health(host, model):
        console.print(f"[green]✓[/green] Ollama is reachable")
        console.print(f"[green]✓[/green] Model {model} is available")
        raise typer.Exit(0)
    else:
        err_console.print(f"[red]✗[/red] Cannot connect to Ollama or model not found")
        err_console.print(f"\nTry: ollama pull {model}")
        raise typer.Exit(2)


@app.command()
def models(
    host: Annotated[str, typer.Option("--host")] = settings.ollama_host,
) -> None:
    """List available vision models from Ollama."""
    available = get_available_models(host)

    if not available:
        console.print("[yellow]No vision models found.[/yellow]")
        console.print("\nTry: ollama pull qwen2.5-vl:7b")
        raise typer.Exit(1)

    console.print("[bold]Available vision models:[/bold]")
    for model in available:
        if "qwen2.5-vl" in model:
            console.print(f"  {model} [dim](recommended)[/dim]")
        else:
            console.print(f"  {model}")


@app.command()
def gui() -> None:
    """Launch the graphical user interface."""
    try:
        from remarkable_ocr.gui import RemarkableOCRApp
    except ImportError:
        err_console.print("[red]Error:[/red] GUI dependencies not installed")
        err_console.print("\nInstall with: pip install remarkable-ocr[gui]")
        err_console.print("Also ensure GTK4 and libadwaita are installed:")
        err_console.print("  Fedora: dnf install gtk4-devel libadwaita-devel")
        err_console.print("  Ubuntu: apt install libgtk-4-dev libadwaita-1-dev")
        raise typer.Exit(1)

    app = RemarkableOCRApp()
    app.run(None)


@app.command()
def process(
    pdf_path: Annotated[Path, typer.Argument(help="Path to PDF file")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory")
    ] = settings.output_dir,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Ollama model to use")
    ] = settings.model,
    host: Annotated[
        str, typer.Option("--host", help="Ollama API host")
    ] = settings.ollama_host,
    output_format: Annotated[
        str, typer.Option("--output-format", "-f", help="Output format")
    ] = settings.output_format,
    timeout: Annotated[
        int, typer.Option("--timeout", help="Timeout per page in seconds")
    ] = settings.timeout,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug logging")
    ] = False,
    chunk_pages: Annotated[
        bool, typer.Option("--chunk-pages/--no-chunk-pages",
                          help="Split pages into overlapping chunks for better OCR")
    ] = settings.chunk_pages,
    chunk_count: Annotated[
        int, typer.Option("--chunks", help="Number of chunks per page (2-4) for fixed chunking")
    ] = settings.chunk_count,
    overlap_percent: Annotated[
        float, typer.Option("--overlap", help="Overlap percentage between chunks (0-50)")
    ] = settings.overlap_percent,
    smart_chunking: Annotated[
        bool, typer.Option("--smart-chunk/--no-smart-chunk",
                          help="Use whitespace detection for natural chunk boundaries (default: enabled)")
    ] = settings.smart_chunking,
    auto_chunk: Annotated[
        bool, typer.Option("--auto-chunk/--no-auto-chunk",
                          help="Automatically chunk tall images (height > max_chunk_height)")
    ] = settings.auto_chunk,
    context_size: Annotated[
        Optional[int], typer.Option("--context-size",
                                    help="Ollama context window size (e.g., 8192, 16384)")
    ] = None,
    temperature: Annotated[
        Optional[float], typer.Option("--temperature",
                                      help="LLM temperature (0.0-1.0, higher = more creative/uncertain)")
    ] = None,
) -> None:
    """Process a Remarkable PDF and extract handwritten text."""
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    logger = get_logger("cli")

    # Resolve num_ctx: CLI option takes precedence over settings
    num_ctx = context_size if context_size is not None else settings.num_ctx

    # Resolve temperature: CLI option takes precedence over settings
    temp = temperature if temperature is not None else settings.temperature

    # Validate input
    if not pdf_path.exists():
        err_console.print(f"[red]Error:[/red] File not found: {pdf_path}")
        raise typer.Exit(2)

    if not pdf_path.suffix.lower() == ".pdf":
        err_console.print(f"[red]Error:[/red] Not a PDF file: {pdf_path}")
        raise typer.Exit(2)

    # Check Ollama
    console.print(f"Checking Ollama...")
    if not check_ollama_health(host, model):
        err_console.print(f"[red]Error:[/red] Cannot connect to Ollama or model {model} not found")
        err_console.print(f"\nMake sure Ollama is running: ollama serve")
        err_console.print(f"And pull the model: ollama pull {model}")
        raise typer.Exit(2)

    # Get page count
    page_count = get_page_count(pdf_path)
    if chunk_pages:
        chunk_mode = "smart" if smart_chunking else f"{chunk_count} fixed"
        chunk_info = f", {chunk_mode} chunks"
    elif auto_chunk:
        chunk_info = ", auto-chunk enabled"
    else:
        chunk_info = ""
    console.print(f"Processing [bold]{pdf_path.name}[/bold] ({page_count} pages{chunk_info})")

    # Process pages
    results = []
    failed_pages = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=page_count)

        for page_num, image in extract_pages(pdf_path):
            # Determine if this page should be chunked
            should_chunk = chunk_pages
            if not should_chunk and auto_chunk and image.height > settings.max_chunk_height:
                should_chunk = True
                logger.info(
                    f"Page {page_num}: Auto-chunking enabled "
                    f"(height {image.height}px > {settings.max_chunk_height}px)"
                )

            if should_chunk:
                # Split image into overlapping chunks
                chunks = split_image_into_chunks(
                    image,
                    chunk_count=chunk_count,
                    overlap_percent=overlap_percent,
                    smart_chunking=smart_chunking,
                )
                chunk_results = []

                for chunk in chunks:
                    progress.update(
                        task,
                        description=f"Page {page_num}/{page_count} chunk {chunk.chunk_index + 1}/{chunk.total_chunks}"
                    )

                    # Note: previous_chunk_text disabled - it caused garbled output
                    # on later chunks. Deduplication handles overlaps instead.
                    chunk_result = process_image(
                        image=chunk.image,
                        page_num=page_num,
                        model=model,
                        host=host,
                        timeout=timeout,
                        chunk_info=(chunk.chunk_index, chunk.total_chunks),
                        previous_chunk_text=None,
                        num_ctx=num_ctx,
                        temperature=temp,
                    )

                    chunk_results.append(ChunkOCRResult(
                        text=chunk_result.text,
                        chunk_index=chunk.chunk_index,
                        page_num=page_num,
                    ))

                # Merge chunk texts with deduplication
                merged_text = merge_chunk_texts(chunk_results)

                result = OCRResult(
                    text=merged_text,
                    confidence=None,
                    page_num=page_num,
                )
            else:
                progress.update(task, description=f"Page {page_num}/{page_count}")

                result = process_image(
                    image=image,
                    page_num=page_num,
                    model=model,
                    host=host,
                    timeout=timeout,
                    num_ctx=num_ctx,
                    temperature=temp,
                )

            # Clean the text
            result.text = clean_text(result.text)

            if result.text == "[OCR Failed]":
                failed_pages += 1
                logger.warning(f"Page {page_num} failed")

            results.append(result)
            progress.advance(task)

    # Write output
    output_path = write_output(
        results=results,
        source=pdf_path,
        output_dir=output,
        output_format=output_format,
        model=model,
    )

    # Summary
    success_pages = page_count - failed_pages
    console.print()
    console.print(f"[green]✓[/green] Completed: {output_path}")
    console.print(f"  Pages: {success_pages}/{page_count} successful")

    if failed_pages > 0:
        console.print(f"  [yellow]Warning: {failed_pages} pages failed[/yellow]")
        raise typer.Exit(1)

    raise typer.Exit(0)


if __name__ == "__main__":
    app()
