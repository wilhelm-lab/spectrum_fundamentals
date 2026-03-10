"""Spectrum Fundamentals."""

import logging
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

__author__ = """The Oktoberfest development team (Wilhelmlab at Technical University of Munich)"""
__copyright__ = f"Copyright {datetime.now():%Y}, Wilhelmlab at Technical University of Munich"
__license__ = "MIT"

try:
    __version__ = version("spectrum_fundamentals")
except PackageNotFoundError:  # package not installed
    __version__ = "unknown"

# Library-level logger — handlers and levels are configured by the application.
logger = logging.getLogger(__name__)
