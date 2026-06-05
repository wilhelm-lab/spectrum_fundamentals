"""Per-spectrum mass-calibration matcher.

Fits a RANSAC line ``ppm_residual ~ a + b * theoretical_mass`` over the
candidate cloud returned by ``match_peaks``, then for each fragment slot picks
the inlier closest to the fitted line. Optionally enforces one-peak-per-slot
uniqueness via greedy assignment on ascending deviation from the fit.

Designed to correct systematic mass-calibration drift within a single MS2
spectrum and to reject noise spikes that happen to fall inside the ppm window
but don't lie on the global drift line.

The matcher degrades gracefully to ``nearest`` whenever the fit can't be
trusted: too few candidates, a non-converging RANSAC, no inliers, or a
converged fit that still retains fewer than ``min_inlier_fraction`` of the
fragment slots (a "bad" fit that would otherwise silently decimate matches).
"""

import logging
import numbers
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor

from .nearest import nearest_resolver

logger = logging.getLogger(__name__)

# Columns every candidate row must carry. ``match_peaks`` always emits these
# for the linear b/y path; multifrag additionally carries ``full_name``,
# which we propagate transparently when present.
_REQUIRED_COLUMNS = ("ion_type", "no", "charge", "exp_mass", "theoretical_mass", "intensity")
_DIAGNOSTIC_COLUMNS = ("ppm_residual", "abs_dev_from_fit")
_SLOT_COLUMNS = ["ion_type", "no", "charge"]

# Used when the residual threshold cannot be derived from the matching
# tolerance (tolerance unset or given in Da). Reasonable for FTMS data.
_DEFAULT_RESIDUAL_THRESHOLD_PPM = 5.0
# Fraction of the (ppm) matching tolerance to use as the inlier band when the
# threshold is derived rather than given explicitly.
_TOLERANCE_THRESHOLD_FRACTION = 0.5


def global_ransac_resolver(
    candidates: list[dict[str, Any]] | None,
    peaks_masses: np.ndarray | None = None,
    peaks_intensity: np.ndarray | None = None,
    unmod_sequence: str | None = None,
    *,
    residual_threshold_ppm: float | None = None,
    min_samples: int = 2,
    max_trials: int = 100,
    random_state: int | None = 42,
    unique_peak: bool = True,
    min_inlier_fraction: float = 0.5,
    mass_tolerance: float | None = None,
    unit_mass_tolerance: str | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, int]:
    """Resolve candidates via RANSAC mass-calibration fit, then per-slot pick.

    :param candidates: rows from ``match_peaks``. May be ``None`` or empty.
    :param peaks_masses: unused; part of the resolver contract.
    :param peaks_intensity: unused; part of the resolver contract.
    :param unmod_sequence: unused; part of the resolver contract.
    :param residual_threshold_ppm: positive ppm tolerance for inlier classification.
        If ``None`` (default), it is derived from the matching tolerance:
        ``_TOLERANCE_THRESHOLD_FRACTION * mass_tolerance`` when the tolerance is in
        ppm, otherwise ``_DEFAULT_RESIDUAL_THRESHOLD_PPM``. Pass an explicit value to
        override.
    :param min_samples: minimum samples per RANSAC trial; must be ``>= 2``.
    :param max_trials: maximum RANSAC iterations; must be ``>= 1``.
    :param random_state: RANSAC seed; pass ``None`` for non-deterministic behaviour.
    :param unique_peak: if True (default), enforce one observed peak per fragment
        slot AND one fragment slot per observed peak via greedy assignment in
        ascending deviation from the fit. If False, peaks may be reused across
        slots (preserves the legacy linear-path behaviour).
    :param min_inlier_fraction: in ``[0, 1]``. If the converged fit keeps inliers
        spanning fewer than this fraction of the candidate fragment slots, defer to
        ``nearest`` instead of returning a decimated match set. ``0`` disables the
        check. Default ``0.5``.
    :param mass_tolerance: matching tolerance used to derive ``residual_threshold_ppm``
        when it is not given explicitly. Forwarded by ``_annotate_linear_spectrum``.
    :param unit_mass_tolerance: unit of ``mass_tolerance`` (``"ppm"`` or ``"da"``).
    :raises ValueError: if any hyperparameter is out of range, or candidate rows
        miss any required column.
    :return: ``(matched_peaks_df, n_dropped)``. ``n_dropped`` is the count of
        input rows that did not survive into the output (rejected as outliers,
        lost to greedy uniqueness, or dropped as non-finite). The DataFrame
        carries the contract columns plus ``ppm_residual`` and
        ``abs_dev_from_fit`` for downstream diagnostics; the multifrag
        ``full_name`` column is preserved when present.
    """
    _validate_hyperparameters(min_samples, max_trials, min_inlier_fraction)
    residual_threshold = _resolve_residual_threshold(
        residual_threshold_ppm, mass_tolerance, unit_mass_tolerance
    )

    if not candidates:
        return pd.DataFrame(), 0
    n_input = len(candidates)

    df = pd.DataFrame(candidates)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"global_ransac: candidate rows missing required columns {missing}"
        )

    # Drop rows that can't yield a finite ppm residual. sklearn's RANSAC raises
    # on NaN/inf in y, and a non-positive theoretical_mass would divide-by-zero
    # or flip the sign of the ppm scale. Real fragment m/z is always positive
    # and finite; this guard catches corrupt or partially-populated inputs
    # without taking the whole annotate_spectra loop down.
    exp = df["exp_mass"].to_numpy()
    theo = df["theoretical_mass"].to_numpy()
    valid = np.isfinite(exp) & np.isfinite(theo) & (theo > 0)
    n_invalid = int((~valid).sum())
    if n_invalid:
        logger.warning(
            "global_ransac: dropping %d/%d candidate rows with non-finite or non-positive mass",
            n_invalid, n_input,
        )
        df = df.loc[valid].reset_index(drop=True)

    if len(df) == 0:
        return pd.DataFrame(columns=df.columns), n_input

    theo_masses = df["theoretical_mass"].to_numpy()
    df["ppm_residual"] = 1e6 * (df["exp_mass"].to_numpy() - theo_masses) / theo_masses

    if len(df) < min_samples:
        logger.info(
            "global_ransac: %d valid candidates < min_samples=%d, falling back to nearest",
            len(df), min_samples,
        )
        return _fallback_to_nearest(df, n_input)

    n_slots = df[_SLOT_COLUMNS].drop_duplicates().shape[0]

    X = df[["theoretical_mass"]].to_numpy()
    y = df["ppm_residual"].to_numpy()

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        random_state=random_state,
    )
    try:
        ransac.fit(X, y)
    except ValueError as exc:
        # Covers InvalidParameterError (subclass of ValueError) and "no consensus
        # set found" — both mean the fit can't be trusted; defer to nearest so
        # the spectrum still gets matched rather than producing a zero vector.
        logger.info(
            "global_ransac: RANSAC fit did not converge (%s), falling back to nearest",
            exc,
        )
        return _fallback_to_nearest(df, n_input)

    predicted = ransac.predict(X)
    df["abs_dev_from_fit"] = np.abs(y - predicted)
    inlier_mask = np.asarray(ransac.inlier_mask_, dtype=bool)
    if not inlier_mask.any():
        logger.info("global_ransac: no inliers after fit, falling back to nearest")
        return _fallback_to_nearest(df, n_input)

    inliers = df.loc[inlier_mask]

    # Guard against a converged-but-bad fit. RANSAC always reports at least
    # ``min_samples`` inliers, so "it converged" does not mean "it explained the
    # spectrum". If the inliers cover too few of the fragment slots, the fitted
    # line is more likely noise than calibration drift; deferring to nearest is
    # safer than silently dropping most matches.
    inlier_slots = inliers[_SLOT_COLUMNS].drop_duplicates().shape[0]
    if inlier_slots < min_inlier_fraction * n_slots:
        logger.info(
            "global_ransac: fit kept inliers for only %d/%d slots (< %.2f), falling back to nearest",
            inlier_slots, n_slots, min_inlier_fraction,
        )
        return _fallback_to_nearest(df, n_input)

    inliers = inliers.sort_values("abs_dev_from_fit", ascending=True, kind="stable")

    if unique_peak:
        chosen = _greedy_unique_assignment(inliers)
    else:
        chosen = inliers.drop_duplicates(subset=_SLOT_COLUMNS, keep="first")

    chosen = chosen.reset_index(drop=True)
    return chosen, n_input - len(chosen)


def _greedy_unique_assignment(inliers: pd.DataFrame) -> pd.DataFrame:
    """Walk inliers (already sorted by ascending abs_dev_from_fit) and keep a row
    iff its ``(ion_type, no, charge)`` slot AND its ``exp_mass`` peak are both
    still unclaimed. Deterministic given a stable input order.
    """
    ion_types = inliers["ion_type"].to_numpy()
    nos = inliers["no"].to_numpy()
    charges = inliers["charge"].to_numpy()
    masses = inliers["exp_mass"].to_numpy()

    used_slots: set[tuple] = set()
    used_peaks: set = set()
    keep = np.zeros(len(inliers), dtype=bool)
    for i in range(len(inliers)):
        slot = (ion_types[i], nos[i], charges[i])
        peak = masses[i]
        if slot in used_slots or peak in used_peaks:
            continue
        keep[i] = True
        used_slots.add(slot)
        used_peaks.add(peak)
    return inliers[keep]


def _fallback_to_nearest(df: pd.DataFrame, n_input: int) -> tuple[pd.DataFrame, int]:
    """Delegate to ``nearest_resolver`` with diagnostic columns stripped.

    Uses the *already-filtered* candidate set (i.e. non-finite rows have been
    removed) so the fallback path inherits the same data hygiene as the RANSAC
    path. ``n_input`` is the count of rows the user originally passed in; we
    use it so ``n_dropped`` correctly reflects "input rows not in output",
    including those filtered as non-finite.
    """
    cleaned = df.drop(columns=list(_DIAGNOSTIC_COLUMNS), errors="ignore")
    chosen, _ = nearest_resolver(candidates=cleaned.to_dict("records"))
    chosen = chosen.reset_index(drop=True)
    return chosen, n_input - len(chosen)


def _resolve_residual_threshold(
    residual_threshold_ppm: float | None,
    mass_tolerance: float | None,
    unit_mass_tolerance: str | None,
) -> float:
    """Resolve the RANSAC inlier band (ppm).

    An explicit ``residual_threshold_ppm`` always wins (and is validated). When
    omitted, derive it from the matching tolerance so the band scales with the
    instrument: a fraction of the window for ppm tolerances, else a fixed
    fallback.

    :raises ValueError: if an explicit ``residual_threshold_ppm`` is not a positive
        finite number.
    """
    if residual_threshold_ppm is not None:
        if not (
            isinstance(residual_threshold_ppm, numbers.Real)
            and np.isfinite(float(residual_threshold_ppm))
            and residual_threshold_ppm > 0
        ):
            raise ValueError(
                f"residual_threshold_ppm must be a positive finite number, "
                f"got {residual_threshold_ppm!r}"
            )
        return float(residual_threshold_ppm)

    if (
        unit_mass_tolerance is not None
        and str(unit_mass_tolerance).lower() == "ppm"
        and isinstance(mass_tolerance, numbers.Real)
        and np.isfinite(float(mass_tolerance))
        and mass_tolerance > 0
    ):
        return _TOLERANCE_THRESHOLD_FRACTION * float(mass_tolerance)

    return _DEFAULT_RESIDUAL_THRESHOLD_PPM


def _validate_hyperparameters(
    min_samples: int, max_trials: int, min_inlier_fraction: float
) -> None:
    if not (isinstance(min_samples, numbers.Integral) and min_samples >= 2):
        raise ValueError(
            f"min_samples must be an integer >= 2, got {min_samples!r}"
        )
    if not (isinstance(max_trials, numbers.Integral) and max_trials >= 1):
        raise ValueError(
            f"max_trials must be an integer >= 1, got {max_trials!r}"
        )
    if not (
        isinstance(min_inlier_fraction, numbers.Real)
        and np.isfinite(float(min_inlier_fraction))
        and 0.0 <= min_inlier_fraction <= 1.0
    ):
        raise ValueError(
            f"min_inlier_fraction must be a number in [0, 1], got {min_inlier_fraction!r}"
        )
