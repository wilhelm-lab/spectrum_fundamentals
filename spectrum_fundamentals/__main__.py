#!/usr/bin/env python
"""Command-line interface."""
import click
from rich import traceback


@click.command()
@click.version_option(version="0.4.3", message=click.style("spectrum_fundamentals Version: 0.4.3"))
def main() -> None:
    """spectrum_fundamentals."""


if __name__ == "__main__":
    traceback.install()
    main(prog_name="spectrum_fundamentals")  # pragma: no cover
