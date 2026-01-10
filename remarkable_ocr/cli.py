"""Command-line interface - stub for scaffolding."""

import typer

app = typer.Typer(
    name="remarkable-ocr",
    help="Extract handwritten notes from Remarkable PDF exports using local OCR.",
    add_completion=False,
)


@app.command()
def placeholder():
    """Placeholder command - full CLI coming soon."""
    typer.echo("remarkable-ocr is not yet fully implemented.")
    raise typer.Exit(1)
