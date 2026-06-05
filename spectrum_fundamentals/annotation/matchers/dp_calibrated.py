"""Calibrated DP matcher -- RANSAC drift fit + b/y ladder DP combined.

Fuses ``global_ransac`` and ``dp_ladder``: it first robustly fits a global
mass-calibration line ``ppm_residual ~ a + b * theoretical_mass`` over the
candidate cloud (RANSAC), then runs the ladder DP but measures each fragment's
emission penalty as the deviation *from that fitted line* instead of from the
raw theoretical position. De-drifting the emission removes the tension in plain
``dp_ladder`` between absolute m/z closeness and ladder consistency, so a
surviving peak must agree with both the global calibration and the local ladder
spacing/monotonicity.

The two parent matchers are exactly the two block-coordinate steps of one joint
objective over (line parameters, peak assignment): fixing the assignment and
solving the line is a regression (``global_ransac``'s fit); fixing the line and
solving the assignment is the ladder DP (``dp_ladder``). With ``iterations > 1``
the resolver alternates them EM/ICP-style -- re-fit the line on the
DP-selected matches (now de-noised), then re-run the DP.

Robust degradation: if the calibration line can't be trusted (too few
candidates, a non-converging RANSAC, a non-finite fit, or one that explains too
few fragment slots) the emission falls back to the raw residual -- i.e. plain
``dp_ladder``. The ladder DP in turn falls back to ``nearest`` under the
``min_match_fraction`` floor. Every numeric input is range-checked and
non-finite rows/values are dropped or neutralised before the maths runs.
"""

import logging
import numbers
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor

from .dp_ladder import (
    _REQUIRED_COLUMNS,
    _SLOT_COLUMNS,
    _greedy_unique_peaks,
    _resolve_ppm_scale,
    _run_dp_assignment,
)
from .dp_ladder import _validate_hyperparameters as _validate_dp_hyperparameters
from .global_ransac import _resolve_residual_threshold
from .nearest import nearest_resolver

logger = logging.getLogger(__name__)

# Diagnostic columns this resolver adds; stripped before any nearest fallback.
_DIAGNOSTIC_COLUMNS = ("ppm_residual", "dev_from_line")


def dp_calibrated_resolver(
    candidates: list[dict[str, Any]] | None,
    peaks_masses: np.ndarray | None = None,
    peaks_intensity: np.ndarray | None = None,
    unmod_sequence: str | None = None,
    *,
    ppm_scale: float | None = None,
    skip_penalty: float = 1.0,
    ladder_weight: float = 2.0,
    intensity_weight: float = 0.0,
    unique_peak: bool = True,
    min_match_fraction: float = 0.0,
    residual_threshold_ppm: float | None = None,
    min_samples: int = 2,
    max_trials: int = 100,
    random_state: int | None = 42,
    min_inlier_fraction: float = 0.5,
    iterations: int = 1,
    mass_tolerance: float | None = None,
    unit_mass_tolerance: str | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, int]:
    """Resolve candidates by a RANSAC-calibrated ladder DP.

    :param candidates: rows from ``match_peaks``. May be ``None`` or empty.
    :param peaks_masses: unused; part of the resolver contract.
    :param peaks_intensity: unused; part of the resolver contract.
    :param unmod_sequence: unused; part of the resolver contract.
    :param ppm_scale: ppm scale ``s`` for the DP emission/transition penalties;
        see ``dp_ladder``. Derived from the matching tolerance when ``None``.
    :param skip_penalty: positive flat cost of leaving a fragment unmatched.
    :param ladder_weight: non-negative weight on the gap-consistency term. Once
        the emission is drift-corrected it no longer needs to fight global
        drift, so it can be lowered relative to ``dp_ladder`` if desired.
    :param intensity_weight: non-negative reward weight on observed (normalised)
        intensity. ``0`` keeps the cost purely geometric. Never uses Prosit.
    :param unique_peak: if True (default), enforce one fragment per observed peak
        across ladders, resolving collisions by ascending deviation-from-line.
    :param min_match_fraction: in ``[0, 1]``; if the DP matches fewer than this
        fraction of fragment slots, defer the spectrum to ``nearest``. ``0``
        (default) disables the floor.
    :param residual_threshold_ppm: RANSAC inlier band (ppm) for the calibration
        fit. Derived from the matching tolerance when ``None``.
    :param min_samples: minimum samples per RANSAC trial; must be ``>= 2``. Also
        the minimum candidate count below which no line is fitted.
    :param max_trials: maximum RANSAC iterations; must be ``>= 1``.
    :param random_state: RANSAC seed; ``None`` for non-deterministic behaviour.
    :param min_inlier_fraction: in ``[0, 1]``. The calibration line is trusted
        only if its inliers span at least this fraction of fragment slots;
        otherwise the emission falls back to the raw residual (plain
        ``dp_ladder``). ``0`` always trusts a finite fit.
    :param iterations: number of EM/ICP-style passes (``>= 1``). After the
        initial RANSAC fit, each extra pass re-fits the line (ordinary least
        squares) on the current DP-selected matches and re-runs the DP, stopping
        early once the assignment stabilises.
    :param mass_tolerance: matching tolerance used to derive ``ppm_scale`` and
        ``residual_threshold_ppm`` when those are not given. Forwarded by
        ``_annotate_linear_spectrum``.
    :param unit_mass_tolerance: unit of ``mass_tolerance`` (``"ppm"`` or ``"da"``).
    :raises ValueError: if any hyperparameter is out of range, or candidate rows
        miss a required column.
    :return: ``(matched_peaks_df, n_dropped)``. The DataFrame carries the
        contract columns plus ``ppm_residual`` (raw) and ``dev_from_line``
        (drift-corrected) diagnostics; ``full_name`` is preserved when present.
    """
    _validate_dp_hyperparameters(skip_penalty, ladder_weight, intensity_weight, min_match_fraction)
    _validate_fit_hyperparameters(min_samples, max_trials, min_inlier_fraction, iterations)
    scale = _resolve_ppm_scale(ppm_scale, mass_tolerance, unit_mass_tolerance)
    residual_threshold = _resolve_residual_threshold(residual_threshold_ppm, mass_tolerance, unit_mass_tolerance)

    if not candidates:
        return pd.DataFrame(), 0
    n_input = len(candidates)

    df = pd.DataFrame(candidates)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dp_calibrated: candidate rows missing required columns {missing}")

    # Same data hygiene as the sibling resolvers: drop rows that can't yield a
    # finite ppm residual (non-finite or non-positive mass).
    exp = df["exp_mass"].to_numpy(dtype=float)
    theo = df["theoretical_mass"].to_numpy(dtype=float)
    valid = np.isfinite(exp) & np.isfinite(theo) & (theo > 0)
    n_invalid = int((~valid).sum())
    if n_invalid:
        logger.warning(
            "dp_calibrated: dropping %d/%d candidate rows with non-finite or non-positive mass",
            n_invalid, n_input,
        )
        df = df.loc[valid].reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame(columns=df.columns), n_input

    theo = df["theoretical_mass"].to_numpy(dtype=float)
    ppm = 1e6 * (df["exp_mass"].to_numpy(dtype=float) - theo) / theo
    df["ppm_residual"] = ppm
    n_slots = df[_SLOT_COLUMNS].drop_duplicates().shape[0]

    line = _trustworthy_line(
        df, theo, ppm, residual_threshold, min_samples, max_trials, random_state, min_inlier_fraction, n_slots
    )

    if line is None:
        # No trustworthy drift estimate -> plain ladder DP on the raw residual.
        df["dev_from_line"] = ppm
        chosen_idx = _run_dp_assignment(
            df, scale, skip_penalty, ladder_weight, intensity_weight, emit_col="ppm_residual"
        )
    else:
        chosen_idx = _iterate_dp(
            df, theo, ppm, line, scale, skip_penalty, ladder_weight, intensity_weight, iterations
        )

    if unique_peak:
        chosen_idx = _greedy_unique_peaks(df, chosen_idx, sort_col="dev_from_line")

    matched_slots = df.loc[chosen_idx, _SLOT_COLUMNS].drop_duplicates().shape[0]
    if min_match_fraction > 0 and matched_slots < min_match_fraction * n_slots:
        logger.info(
            "dp_calibrated: matched only %d/%d slots (< %.2f), falling back to nearest",
            matched_slots, n_slots, min_match_fraction,
        )
        return _fallback_to_nearest(df, n_input)

    chosen = df.loc[chosen_idx].reset_index(drop=True)
    return chosen, n_input - len(chosen)


def _trustworthy_line(
    df: pd.DataFrame,
    theo: np.ndarray,
    ppm: np.ndarray,
    residual_threshold: float,
    min_samples: int,
    max_trials: int,
    random_state: int | None,
    min_inlier_fraction: float,
    n_slots: int,
) -> tuple[float, float] | None:
    """Fit ``ppm ~ a + b*theo`` by RANSAC; return ``(a, b)`` only if trustworthy.

    Returns ``None`` (so the caller uses the raw residual / plain ``dp_ladder``)
    on any of: too few candidates, a non-converging RANSAC, a non-finite fit, no
    inliers, or inliers spanning fewer than ``min_inlier_fraction`` of the slots.
    """
    if len(df) < min_samples:
        return None
    try:
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            random_state=random_state,
        )
        ransac.fit(theo.reshape(-1, 1), ppm)
    except ValueError as exc:
        # InvalidParameterError and "no consensus set found" both subclass
        # ValueError; either way the fit is untrustworthy.
        logger.info("dp_calibrated: RANSAC fit did not converge (%s), using raw ladder DP", exc)
        return None

    a = float(ransac.estimator_.intercept_)
    b = float(ransac.estimator_.coef_[0])
    if not (np.isfinite(a) and np.isfinite(b)):
        return None

    inlier_mask = np.asarray(ransac.inlier_mask_, dtype=bool)
    if not inlier_mask.any():
        return None
    inlier_slots = df.loc[inlier_mask, _SLOT_COLUMNS].drop_duplicates().shape[0]
    if inlier_slots < min_inlier_fraction * n_slots:
        logger.info(
            "dp_calibrated: calibration fit explains only %d/%d slots (< %.2f), using raw ladder DP",
            inlier_slots, n_slots, min_inlier_fraction,
        )
        return None
    return a, b


def _iterate_dp(
    df: pd.DataFrame,
    theo: np.ndarray,
    ppm: np.ndarray,
    line: tuple[float, float],
    scale: float,
    skip_penalty: float,
    ladder_weight: float,
    intensity_weight: float,
    iterations: int,
) -> list[int]:
    """Run the de-drifted DP, optionally re-fitting the line on its matches.

    Block-coordinate descent on the joint (line, assignment) objective: each
    pass writes ``dev_from_line`` for the current line, runs the DP, and -- for
    all but the last pass -- re-fits the line (OLS) on the DP-selected matches.
    Stops early when the assignment stops changing. ``df['dev_from_line']`` is
    left consistent with the returned assignment.
    """
    a, b = line
    chosen_idx: list[int] | None = None
    for it in range(iterations):
        df["dev_from_line"] = ppm - (a + b * theo)
        new_chosen = _run_dp_assignment(
            df, scale, skip_penalty, ladder_weight, intensity_weight, emit_col="dev_from_line"
        )
        if chosen_idx is not None and set(new_chosen) == set(chosen_idx):
            return new_chosen
        chosen_idx = new_chosen
        if it < iterations - 1:
            refit = _fit_line_ols(
                df.loc[chosen_idx, "theoretical_mass"].to_numpy(dtype=float),
                df.loc[chosen_idx, "ppm_residual"].to_numpy(dtype=float),
            )
            if refit is None:
                break
            a, b = refit
    return chosen_idx if chosen_idx is not None else []


def _fit_line_ols(theo: np.ndarray, ppm: np.ndarray) -> tuple[float, float] | None:
    """Ordinary least-squares fit of ``ppm ~ a + b*theo``; ``None`` if degenerate.

    Requires at least two distinct theoretical masses (otherwise the slope is
    undefined) and a finite result.
    """
    if theo.size < 2 or np.unique(theo).size < 2:
        return None
    try:
        b, a = np.polyfit(theo, ppm, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    return float(a), float(b)


def _fallback_to_nearest(df: pd.DataFrame, n_input: int) -> tuple[pd.DataFrame, int]:
    """Delegate to ``nearest_resolver`` with diagnostic columns stripped.

    Uses the already-filtered candidate set so the fallback inherits the same
    data hygiene; ``n_input`` is the original row count so ``n_dropped``
    reflects "input rows not in output", including non-finite drops.
    """
    cleaned = df.drop(columns=list(_DIAGNOSTIC_COLUMNS), errors="ignore")
    chosen, _ = nearest_resolver(candidates=cleaned.to_dict("records"))
    chosen = chosen.reset_index(drop=True)
    return chosen, n_input - len(chosen)


def _validate_fit_hyperparameters(
    min_samples: int, max_trials: int, min_inlier_fraction: float, iterations: int
) -> None:
    if not (isinstance(min_samples, numbers.Integral) and min_samples >= 2):
        raise ValueError(f"min_samples must be an integer >= 2, got {min_samples!r}")
    if not (isinstance(max_trials, numbers.Integral) and max_trials >= 1):
        raise ValueError(f"max_trials must be an integer >= 1, got {max_trials!r}")
    if not (
        isinstance(min_inlier_fraction, numbers.Real)
        and np.isfinite(float(min_inlier_fraction))
        and 0.0 <= min_inlier_fraction <= 1.0
    ):
        raise ValueError(f"min_inlier_fraction must be a number in [0, 1], got {min_inlier_fraction!r}")
    if not (isinstance(iterations, numbers.Integral) and iterations >= 1):
        raise ValueError(f"iterations must be an integer >= 1, got {iterations!r}")
