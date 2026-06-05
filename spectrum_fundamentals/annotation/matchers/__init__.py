"""Peak match resolvers.

A resolver collapses the candidate list returned by ``match_peaks`` (zero or
more observed peaks per theoretical fragment, all already inside the ppm
tolerance window) into one matched peak per fragment slot — i.e. one row per
``(ion_type, no, charge)``.

Resolver signature::

    resolver(
        candidates: list[dict],
        peaks_masses: np.ndarray,
        peaks_intensity: np.ndarray,
        unmod_sequence: str,
        **kwargs,
    ) -> tuple[pd.DataFrame, int]

Returns ``(matched_peaks_df, n_dropped)``. The DataFrame must contain the
columns consumed downstream by ``generate_annotation_matrix``:
``ion_type, no, charge, exp_mass, theoretical_mass, intensity`` (plus
``full_name`` when ``multifrag=True``).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from .dp_calibrated import dp_calibrated_resolver
from .dp_ladder import dp_ladder_resolver
from .global_ransac import global_ransac_resolver
from .nearest import nearest_resolver

Resolver = Callable[..., tuple[pd.DataFrame, int]]

MATCHERS: dict[str, Resolver] = {
    "nearest": nearest_resolver,
    "global_ransac": global_ransac_resolver,
    "dp_ladder": dp_ladder_resolver,
    "dp_calibrated": dp_calibrated_resolver,
}


def resolve_matches(
    method: str,
    candidates: list[dict],
    peaks_masses: np.ndarray,
    peaks_intensity: np.ndarray,
    unmod_sequence: str,
    **kwargs,
) -> tuple[pd.DataFrame, int]:
    """Dispatch candidate-list resolution to the named matcher."""
    try:
        resolver = MATCHERS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown matching_method '{method}'. Available: {sorted(MATCHERS)}") from exc
    return resolver(
        candidates=candidates,
        peaks_masses=peaks_masses,
        peaks_intensity=peaks_intensity,
        unmod_sequence=unmod_sequence,
        **kwargs,
    )
