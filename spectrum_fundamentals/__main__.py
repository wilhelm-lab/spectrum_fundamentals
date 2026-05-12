#!/usr/bin/env python
"""Command-line interface."""

import click

from spectrum_fundamentals import __version__


@click.command()
@click.version_option(version=__version__)
def main() -> None:
    """spectrum_fundamentals."""


if __name__ == "__main__":
    main(prog_name="spectrum_fundamentals")  # pragma: no cover
