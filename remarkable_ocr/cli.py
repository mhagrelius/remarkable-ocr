"""Command-line interface using Typer."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from remarkable_ocr import __version__
from remarkable_ocr.config import Settings
from remarkable_ocr.logging import setup_logging, get_logger
from remarkable_ocr.ocr import check_ollama_health, get_available_models, process_image
from remarkable_ocr.pdf import extract_pages, get_page_count
from remarkable_ocr.processor import clean_text, is_blank_page
from remarkable_ocr.writer import write_output

app = typer.Typer(
    name="remarkable-ocr",
    help="Extract handwritten notes from Remarkable PDF exports using local OCR.",
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)


def version_callback(value: bool):
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
):
    """Remarkable OCR - Extract handwritten notes from Remarkable PDFs."""
    pass


@app.command()
def health(
    model: Annotated[str, typer.Option("--model", "-m")] = "qwen2.5-vl:7b",
    host: Annotated[str, typer.Option("--host")] = "http://localhost:11434",
):
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
    host: Annotated[str, typer.Option("--host")] = "http://localhost:11434",
):
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
def process(
    pdf_path: Annotated[Path, typer.Argument(help="Path to PDF file")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory")
    ] = Path("./output"),
    model: Annotated[
        str, typer.Option("--model", "-m", help="Ollama model to use")
    ] = "qwen2.5-vl:7b",
    host: Annotated[
        str, typer.Option("--host", help="Ollama API host")
    ] = "http://localhost:11434",
    confidence_threshold: Annotated[
        float, typer.Option("--confidence-threshold", help="Minimum confidence")
    ] = 0.5,
    output_format: Annotated[
        str, typer.Option("--output-format", "-f", help="Output format")
    ] = "markdown",
    timeout: Annotated[
        int, typer.Option("--timeout", help="Timeout per page in seconds")
    ] = 60,
    keep_temp_images: Annotated[
        bool, typer.Option("--keep-temp-images", help="Keep temporary images")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable debug logging")
    ] = False,
):
    """Process a Remarkable PDF and extract handwritten text."""
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    logger = get_logger("cli")

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
    console.print(f"Processing [bold]{pdf_path.name}[/bold] ({page_count} pages)")

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
            progress.update(task, description=f"Page {page_num}/{page_count}")

            result = process_image(
                image=image,
                page_num=page_num,
                model=model,
                host=host,
                timeout=timeout,
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
