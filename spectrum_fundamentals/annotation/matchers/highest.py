"""Highest-intensity resolver — pick the tallest observed peak in the ppm window."""

import numpy as np
import pandas as pd


def highest_resolver(
    candidates: list[dict],
    peaks_masses: np.ndarray | None = None,
    peaks_intensity: np.ndarray | None = None,
    unmod_sequence: str | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, int]:
    """Pick, per fragment slot, the most intense candidate inside the ppm window.

    Byte-identical to ``handle_multiple_matches(..., sort_by="intensity")``: among the
    observed peaks that fall inside the ppm tolerance window for a given
    ``(ion_type, no, charge)`` slot, keep the tallest one instead of the closest in
    m/z (``nearest_resolver``). ``ppm_error`` is still computed and kept so the
    downstream Percolator ppm-error features are identical to the nearest matcher.

    The ``peaks_*`` and ``unmod_sequence`` arguments are part of the resolver
    contract for forward compatibility with global matchers; this resolver does
    not use them.
    """
    df = pd.DataFrame(candidates)
    if len(df) == 0:
        return df, 0
    df["mass_diff"] = (df["exp_mass"] - df["theoretical_mass"]).abs()
    df["ppm_error"] = df["mass_diff"] / df["theoretical_mass"] * 1e6
    # ^ Scale-invariant mass deviation in ppm — used as rescoring feature for Percolator.
    df = df.sort_values(by="intensity", ascending=False)
    original_length = len(df.index)
    df = df.drop_duplicates(subset=["ion_type", "no", "charge"], keep="first")
    return df, original_length - len(df.index)
